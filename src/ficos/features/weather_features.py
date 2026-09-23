"""
SIH26006 — Weather Feature Engineering

Transforms port_weather (158k rows, 5 ports) into daily features
for the forecasting pipeline.

Weather data columns:
- port_name, date, weather_code
- temperature_2m_max_c, temperature_2m_mean_c
- precipitation_sum_mm
- wind_speed_10m_max_kmh, wind_gusts_10m_max_kmh

Features are pivoted to wide format (one column per port-metric),
plus cross-port aggregates.

Weather indicator thresholds are CONFIGURABLE ENGINEERING INDICATORS,
NOT official severe-weather classifications.

All rolling features are backward-looking only.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_port_weather, load_config


def build_weather_features(save=True):
    """
    Full weather feature pipeline:
    1. Load port_weather
    2. Filter to Dataset A date range (2016+)
    3. Create port-specific wide features
    4. Create cross-port aggregates
    5. Add engineering indicator flags (configurable, documented)
    6. Add rolling features
    """
    cfg = load_config()
    thresholds = cfg.get("weather_indicators", {})
    high_wind = thresholds.get("high_wind_kmh", 50)
    heavy_precip = thresholds.get("heavy_precip_mm", 50)

    print("Loading port weather data...")
    pw = load_port_weather()
    print(f"  Loaded: {pw.shape}")
    print(f"  Full date range: {pw['date'].min()} -> {pw['date'].max()}")

    # Filter to 2016+ to align with Dataset A
    pw = pw[pw["date"] >= "2016-01-01"].copy()
    print(f"  After 2016 filter: {pw.shape}")

    # Select relevant numeric columns for pivoting
    metrics = [
        "temperature_2m_max_c", "temperature_2m_mean_c",
        "precipitation_sum_mm",
        "wind_speed_10m_max_kmh", "wind_gusts_10m_max_kmh",
        "weather_code",
    ]

    # Short metric names for column readability
    metric_short = {
        "temperature_2m_max_c": "temp_max",
        "temperature_2m_mean_c": "temp_mean",
        "precipitation_sum_mm": "precip_mm",
        "wind_speed_10m_max_kmh": "wind_max",
        "wind_gusts_10m_max_kmh": "gust_max",
        "weather_code": "wx_code",
    }

    # Short port names
    port_short = {
        "Haldia": "hal",
        "Paradip": "par",
        "Dhamra": "dha",
        "Visakhapatnam": "viz",
        "Gangavaram": "gan",
    }

    # Pivot to wide format: one column per (port_short, metric_short)
    dfs = []
    for port_name in pw["port_name"].unique():
        port_df = pw[pw["port_name"] == port_name][["date"] + metrics].copy()
        ps = port_short.get(port_name, port_name[:3].lower())

        rename_map = {}
        for m in metrics:
            ms = metric_short.get(m, m)
            rename_map[m] = f"wx_{ps}_{ms}"

        port_df = port_df.rename(columns=rename_map)
        port_df = port_df.set_index("date")
        dfs.append(port_df)

    # Merge all ports on date
    daily = dfs[0]
    for i in range(1, len(dfs)):
        daily = daily.join(dfs[i], how="outer")
    daily = daily.reset_index()

    # Cross-port aggregates
    wind_cols = [c for c in daily.columns if "wind_max" in c]
    gust_cols = [c for c in daily.columns if "gust_max" in c]
    precip_cols = [c for c in daily.columns if "precip_mm" in c]
    temp_cols = [c for c in daily.columns if "temp_max" in c]

    if wind_cols:
        daily["wx_all_wind_max"] = daily[wind_cols].max(axis=1)
        daily["wx_all_wind_mean"] = daily[wind_cols].mean(axis=1)
    if gust_cols:
        daily["wx_all_gust_max"] = daily[gust_cols].max(axis=1)
    if precip_cols:
        daily["wx_all_precip_max"] = daily[precip_cols].max(axis=1)
        daily["wx_all_precip_mean"] = daily[precip_cols].mean(axis=1)
        daily["wx_all_precip_sum"] = daily[precip_cols].sum(axis=1)
    if temp_cols:
        daily["wx_all_temp_max"] = daily[temp_cols].max(axis=1)
        daily["wx_all_temp_mean"] = daily[temp_cols].mean(axis=1)

    # Engineering indicator flags (configurable, NOT official classifications)
    # These flag potentially operationally relevant conditions
    daily["wx_high_wind_indicator"] = (daily["wx_all_wind_max"] > high_wind).astype(int)
    daily["wx_heavy_precip_indicator"] = (daily["wx_all_precip_max"] > heavy_precip).astype(int)
    daily["wx_adverse_indicator"] = (
        (daily["wx_high_wind_indicator"] == 1) | (daily["wx_heavy_precip_indicator"] == 1)
    ).astype(int)

    # Count how many ports have high wind / heavy precip
    for ps_code in port_short.values():
        wind_col = f"wx_{ps_code}_wind_max"
        precip_col = f"wx_{ps_code}_precip_mm"
        if wind_col in daily.columns:
            daily[f"wx_{ps_code}_high_wind"] = (daily[wind_col] > high_wind).astype(int)
        if precip_col in daily.columns:
            daily[f"wx_{ps_code}_heavy_precip"] = (daily[precip_col] > heavy_precip).astype(int)

    high_wind_port_cols = [c for c in daily.columns if c.endswith("_high_wind")]
    heavy_precip_port_cols = [c for c in daily.columns if c.endswith("_heavy_precip")]
    if high_wind_port_cols:
        daily["wx_ports_high_wind_count"] = daily[high_wind_port_cols].sum(axis=1)
    if heavy_precip_port_cols:
        daily["wx_ports_heavy_precip_count"] = daily[heavy_precip_port_cols].sum(axis=1)

    # Rolling features — backward looking
    for w in [3, 7]:
        daily[f"wx_wind_max_{w}d"] = daily["wx_all_wind_max"].rolling(w, min_periods=1).max()
        daily[f"wx_precip_max_{w}d"] = daily["wx_all_precip_max"].rolling(w, min_periods=1).max()
        daily[f"wx_precip_mean_{w}d"] = daily["wx_all_precip_mean"].rolling(w, min_periods=1).mean()
        daily[f"wx_adverse_sum_{w}d"] = daily["wx_adverse_indicator"].rolling(w, min_periods=1).sum()

    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"  Final weather features: {daily.shape}")

    if save:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "outputs", "weather_features.csv")
        daily.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    return daily


if __name__ == "__main__":
    build_weather_features()
