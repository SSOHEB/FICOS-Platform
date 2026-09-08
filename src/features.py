"""
SIH26006 — Feature Engineering

Creates lag, rolling, momentum, and target features from Dataset A.
Merges with Dataset C-derived features (GDELT, cyclone, weather).

LEAKAGE PREVENTION RULES:
- All lag features use only past information: lag_k(t) = value(t-k)
- All rolling features are backward-looking: rolling_mean_w(t) uses [t-w+1, t]
- Target columns target_<class>_<h>d(t) = value(t+h) — the future value to predict
- For forecasting at time t, NO feature may use information from t+1 or beyond
- Percentage changes use safe division (denominator != 0)
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_dataset_a, load_config


def create_lag_features(df, columns, lags, prefix=""):
    """
    Create lag features. lag_k(t) = value(t-k).
    Only uses past information — no leakage.
    """
    for col in columns:
        for lag in lags:
            col_name = f"{prefix}{col}_lag_{lag}"
            df[col_name] = df[col].shift(lag)
    return df


def create_rolling_features(df, columns, windows, prefix=""):
    """
    Create backward-looking rolling features.
    rolling_mean_w(t) uses values at [t-w+1, ..., t].
    This is the default pandas behavior with shift(0).
    """
    for col in columns:
        for w in windows:
            df[f"{prefix}{col}_rmean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{prefix}{col}_rstd_{w}"] = df[col].rolling(window=w, min_periods=2).std()
    return df


def create_change_features(df, columns, periods, prefix=""):
    """
    Create momentum/change features.
    change_k(t) = value(t) - value(t-k)  — uses only past data.
    pct_change_k(t) = (value(t) - value(t-k)) / |value(t-k)| — safe division.
    """
    for col in columns:
        for p in periods:
            df[f"{prefix}{col}_chg_{p}"] = df[col] - df[col].shift(p)
            # Safe percentage change
            prev = df[col].shift(p)
            df[f"{prefix}{col}_pchg_{p}"] = np.where(
                prev.abs() > 1e-8,
                (df[col] - prev) / prev.abs(),
                np.nan
            )
    return df


def create_targets(df, freight_cols, horizons):
    """
    Create forecast target columns.
    target_<class>_<h>d(t) = value(t+h)
    These are future values — used ONLY as labels, never as features.
    """
    for col in freight_cols:
        for h in horizons:
            df[f"target_{col}_{h}d"] = df[col].shift(-h)
    return df


def create_directional_targets(df, freight_cols, horizons, threshold=0.0):
    """
    Create directional targets: UP (1), DOWN (-1), FLAT (0).
    direction(t, h) = sign(value(t+h) - value(t))

    threshold: minimum absolute change to count as UP/DOWN.
    If threshold=0, any non-zero change is directional.
    """
    for col in freight_cols:
        for h in horizons:
            future_val = df[col].shift(-h)
            change = future_val - df[col]
            direction = np.where(
                change > threshold, 1,
                np.where(change < -threshold, -1, 0)
            )
            # Keep NaN where future is unknown
            direction = np.where(pd.isna(change), np.nan, direction)
            df[f"dir_{col}_{h}d"] = direction
    return df


def build_feature_dictionary(df, freight_cols, cfg):
    """Build a feature dictionary documenting every feature."""
    rows = []
    for col in df.columns:
        if col == "date":
            category = "index"
        elif col in freight_cols:
            category = "freight_target_raw"
        elif col.startswith("target_"):
            category = "forecast_target"
        elif col.startswith("dir_"):
            category = "directional_target"
        elif "_lag_" in col:
            category = "lag_feature"
        elif "_rmean_" in col or "_rstd_" in col:
            category = "rolling_feature"
        elif "_chg_" in col or "_pchg_" in col:
            category = "momentum_feature"
        elif col.startswith("gdelt_"):
            category = "gdelt_feature"
        elif col.startswith("cyclone_"):
            category = "cyclone_feature"
        elif col.startswith("wx_"):
            category = "weather_feature"
        elif col in ["n10d", "gprd", "gprd_act", "gprd_threat", "gprd_ma7", "gprd_ma30"]:
            category = "market_feature"
        elif col in ["usd_inr", "brent_usd_per_barrel", "wti_usd_per_barrel"]:
            category = "market_feature"
        elif col in ["gscpi", "coal_australian", "coal_south_african", "iron_ore_cfr_spot",
                      "copper", "aluminum", "dap", "tsp", "urea", "potassium_chloride"]:
            category = "commodity_feature"
        elif col.startswith("mc") or col.startswith("mp") or col.startswith("ms") or col.startswith("mh"):
            category = "route_freight"
        else:
            category = "other"

        rows.append({
            "column": col,
            "category": category,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(100 * df[col].isnull().sum() / len(df), 2),
        })

    return pd.DataFrame(rows)


def build_modeling_dataset(save=True):
    """
    Full feature engineering pipeline:
    1. Load Dataset A
    2. Create lag/rolling/momentum features for freight targets
    3. Create lag/rolling/momentum features for market/commodity vars
    4. Create forecast targets
    5. Create directional targets
    6. Merge with GDELT, cyclone, weather features
    7. Save modeling dataset and feature dictionary
    """
    cfg = load_config()
    freight_cols = cfg["freight_targets"]
    horizons = cfg["horizons"]
    lags = cfg["lags"]
    rolling_windows = cfg["rolling_windows"]
    change_periods = cfg["change_periods"]
    threshold = cfg.get("directional_threshold", 0.0)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Loading Dataset A...")
    df = load_dataset_a(cfg)
    print(f"  Shape: {df.shape}")

    # ── Freight target features ──
    print("Creating freight lag features...")
    df = create_lag_features(df, freight_cols, lags)

    print("Creating freight rolling features...")
    df = create_rolling_features(df, freight_cols, rolling_windows)

    print("Creating freight momentum features...")
    df = create_change_features(df, freight_cols, change_periods)

    # ── Market/commodity features ──
    market_cols = ["usd_inr", "brent_usd_per_barrel", "wti_usd_per_barrel",
                   "n10d", "gprd"]
    commodity_cols = ["iron_ore_cfr_spot", "coal_australian", "coal_south_african",
                      "copper", "aluminum"]

    # Use shorter lag/rolling for market/commodity (avoid too many features)
    market_lags = [1, 7, 14, 30]
    market_rolling = [7, 14, 30]
    market_change = [1, 7, 14]

    print("Creating market/commodity features...")
    df = create_lag_features(df, market_cols + commodity_cols, market_lags, prefix="mkt_")
    df = create_rolling_features(df, market_cols + commodity_cols, market_rolling, prefix="mkt_")
    df = create_change_features(df, market_cols + commodity_cols, market_change, prefix="mkt_")

    # ── Targets ──
    print("Creating forecast targets...")
    df = create_targets(df, freight_cols, horizons)

    print("Creating directional targets...")
    df = create_directional_targets(df, freight_cols, horizons, threshold=threshold)

    # ── Merge Dataset C features ──
    print("Merging GDELT daily features...")
    gdelt_path = os.path.join(root, "outputs", "gdelt_daily_features.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, parse_dates=["date"])
        df = df.merge(gdelt, on="date", how="left")
        print(f"  GDELT features merged: {len(gdelt.columns) - 1} columns")
    else:
        print("  [WARN] GDELT features not found — run gdelt_features.py first")

    print("Merging cyclone features...")
    cyclone_path = os.path.join(root, "outputs", "cyclone_features.csv")
    if os.path.exists(cyclone_path):
        cyc = pd.read_csv(cyclone_path, parse_dates=["date"])
        df = df.merge(cyc, on="date", how="left")
        print(f"  Cyclone features merged: {len(cyc.columns) - 1} columns")
    else:
        print("  [WARN] Cyclone features not found — run cyclone_features.py first")

    print("Merging weather features...")
    weather_path = os.path.join(root, "outputs", "weather_features.csv")
    if os.path.exists(weather_path):
        wx = pd.read_csv(weather_path, parse_dates=["date"])
        df = df.merge(wx, on="date", how="left")
        print(f"  Weather features merged: {len(wx.columns) - 1} columns")
    else:
        print("  [WARN] Weather features not found — run weather_features.py first")

    # ── Final cleanup ──
    df = df.sort_values("date").reset_index(drop=True)

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"\nFinal modeling dataset: {df.shape}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")

    if save:
        # Save modeling dataset
        out_path = os.path.join(root, "outputs", "modeling_dataset.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        # Save feature dictionary
        feat_dict = build_feature_dictionary(df, freight_cols, cfg)
        feat_path = os.path.join(root, "outputs", "feature_dictionary.csv")
        feat_dict.to_csv(feat_path, index=False)
        print(f"Saved: {feat_path}")

    return df


if __name__ == "__main__":
    build_modeling_dataset()
