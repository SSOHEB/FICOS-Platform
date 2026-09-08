"""
SIH26006 — Cyclone Feature Engineering

Transforms cyclone_events (471 storm summaries, NI basin) into daily
time-series features for the forecasting pipeline.

The cyclone data contains storm-level summaries with:
- start_time, end_time, duration_hours
- max_wind, min_pressure
- bounding box (max_lat, min_lat, max_lon, min_lon)
- landfall_observed (0/1)

Features are created by expanding each storm to daily presence,
then aggregating. Port-region exposure uses lat/lon bounding box
overlap with port coordinates.

All rolling features are backward-looking only.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_cyclone_events, load_config


# East Coast India port coordinates (approximate)
PORT_COORDS = {
    "Haldia":         {"lat": 22.04, "lon": 88.06},
    "Paradip":        {"lat": 20.26, "lon": 86.61},
    "Dhamra":         {"lat": 20.79, "lon": 86.95},
    "Visakhapatnam":  {"lat": 17.68, "lon": 83.22},
    "Gangavaram":     {"lat": 17.62, "lon": 83.24},
}

# Proximity buffer for cyclone exposure (degrees latitude/longitude)
# ~2 degrees is roughly 200 km — a reasonable "close approach" buffer
EXPOSURE_BUFFER_DEG = 3.0


def check_port_exposure(storm_row, buffer=EXPOSURE_BUFFER_DEG):
    """
    Check if a storm's bounding box (expanded by buffer) overlaps
    with any East Coast India port.

    Returns list of exposed port names.
    This is called 'exposure', not 'landfall', because bounding-box
    overlap indicates potential proximity, not confirmed landfall.
    """
    exposed = []
    s_lat_min = storm_row["min_lat"] - buffer
    s_lat_max = storm_row["max_lat"] + buffer
    s_lon_min = storm_row["min_lon"] - buffer
    s_lon_max = storm_row["max_lon"] + buffer

    for port_name, coords in PORT_COORDS.items():
        if (s_lat_min <= coords["lat"] <= s_lat_max and
                s_lon_min <= coords["lon"] <= s_lon_max):
            exposed.append(port_name)
    return exposed


def expand_storms_to_daily(df):
    """
    Expand storm-level summaries to daily records.
    Each storm gets one row per day it was active.
    """
    rows = []
    for _, storm in df.iterrows():
        start = storm["start_time"]
        end = storm["end_time"]

        if pd.isna(start) or pd.isna(end):
            continue

        # Generate date range for the storm's active period
        dates = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")

        exposed_ports = check_port_exposure(storm)

        for d in dates:
            rows.append({
                "date": d,
                "storm_id": storm["storm_id"],
                "storm_name": storm["storm_name"],
                "max_wind": storm["max_wind"],
                "min_pressure": storm["min_pressure"],
                "landfall_observed": storm["landfall_observed"],
                "duration_hours": storm["duration_hours"],
                "port_exposure": 1 if len(exposed_ports) > 0 else 0,
                "exposed_ports": ",".join(exposed_ports) if exposed_ports else "",
                "num_exposed_ports": len(exposed_ports),
            })

    return pd.DataFrame(rows)


def aggregate_daily_cyclone(expanded):
    """Aggregate expanded storm-day records to daily features."""
    if expanded.empty:
        return pd.DataFrame(columns=["date"])

    daily = expanded.groupby("date").agg(
        cyclone_active_count=("storm_id", "nunique"),
        cyclone_max_wind=("max_wind", "max"),
        cyclone_min_pressure=("min_pressure", "min"),
        cyclone_port_exposure=("port_exposure", "max"),
        cyclone_num_exposed_ports=("num_exposed_ports", "max"),
        cyclone_landfall_any=("landfall_observed", "max"),
    ).reset_index()

    # Binary active flag
    daily["cyclone_active"] = (daily["cyclone_active_count"] > 0).astype(int)

    return daily


def build_cyclone_features(save=True):
    """
    Full cyclone feature pipeline:
    1. Load cyclone events
    2. Expand to daily
    3. Aggregate
    4. Add rolling features
    5. Create complete daily calendar (fill non-cyclone days with 0)
    """
    cfg = load_config()

    print("Loading cyclone events...")
    cy = load_cyclone_events()
    print(f"  Loaded {len(cy)} storm summaries")
    print(f"  Date range: {cy['start_time'].min()} -> {cy['end_time'].max()}")
    print(f"  Basin: {cy['basin'].unique()}")

    print("\nExpanding storms to daily records...")
    expanded = expand_storms_to_daily(cy)
    print(f"  Expanded rows: {len(expanded)}")

    print("Aggregating to daily features...")
    daily = aggregate_daily_cyclone(expanded)
    print(f"  Days with cyclone activity: {len(daily)}")

    # Create full date range aligned with Dataset A (2016-01-04 to 2026-09-04)
    full_dates = pd.DataFrame({
        "date": pd.date_range(start="2016-01-01", end="2026-09-07", freq="D")
    })

    daily = full_dates.merge(daily, on="date", how="left")

    # Fill non-cyclone days with 0
    fill_cols = [c for c in daily.columns if c != "date"]
    daily[fill_cols] = daily[fill_cols].fillna(0)

    # Rolling features — backward looking
    windows = cfg.get("cyclone_rolling_windows", [7, 14, 30])
    for w in windows:
        daily[f"cyclone_active_sum_{w}d"] = (
            daily["cyclone_active"].rolling(window=w, min_periods=1).sum()
        )
        daily[f"cyclone_max_wind_{w}d"] = (
            daily["cyclone_max_wind"].rolling(window=w, min_periods=1).max()
        )
        daily[f"cyclone_exposure_sum_{w}d"] = (
            daily["cyclone_port_exposure"].rolling(window=w, min_periods=1).sum()
        )

    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"  Final cyclone features: {daily.shape}")

    if save:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "outputs", "cyclone_features.csv")
        daily.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    return daily


if __name__ == "__main__":
    build_cyclone_features()
