"""
SIH26006 — Data Loader
Robust reusable loaders for Dataset A, B, C.
Source files are NEVER modified.
"""

import pandas as pd
import numpy as np
import yaml
import os
import json
from pathlib import Path


def _project_root():
    """Return the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config():
    """Load the central YAML config."""
    cfg_path = _project_root() / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Dataset A
# ──────────────────────────────────────────────
def load_dataset_a(cfg=None):
    """
    Load Dataset A (daily freight / market / macro table).
    Returns a DataFrame with 'date' as datetime, sorted chronologically.
    """
    if cfg is None:
        cfg = load_config()
    path = _project_root() / cfg["paths"]["dataset_a"]

    df = pd.read_csv(path)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        n_bad = df["date"].isna().sum()
        print(f"[WARN] {n_bad} unparseable dates in Dataset A — dropping those rows.")
        df = df.dropna(subset=["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Drop record_id (just a row counter) and event (almost entirely null)
    cols_to_drop = [c for c in ["record_id", "event"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c != "date"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ──────────────────────────────────────────────
# Dataset B
# ──────────────────────────────────────────────
DATASET_B_SHEETS = [
    "ports", "vessel_types", "berths", "berth_constraints",
    "berth_cargo_rules", "berth_tide_rules", "port_operating_rules",
    "port_sources", "cargo_traffic", "average_output", "average_trt",
    "pbdt", "commodity_capacity", "merchant_fleet",
]


def load_dataset_b(sheet_name=None, cfg=None):
    """
    Load Dataset B sheets (read-only, never modified).
    sheet_name: str or list or None (all sheets).
    Returns dict of {sheet_name: DataFrame} when multiple sheets,
    or a single DataFrame when one sheet is specified.
    """
    if cfg is None:
        cfg = load_config()
    path = _project_root() / cfg["paths"]["dataset_b"]

    if sheet_name is None:
        sheets = DATASET_B_SHEETS
    elif isinstance(sheet_name, str):
        sheets = [sheet_name]
    else:
        sheets = list(sheet_name)

    result = {}
    for s in sheets:
        result[s] = pd.read_excel(path, sheet_name=s)

    if isinstance(sheet_name, str):
        return result[sheet_name]
    return result


# ──────────────────────────────────────────────
# Dataset C
# ──────────────────────────────────────────────
DATASET_C_SHEETS = ["port_weather", "cyclone_events", "geopolitical_events"]


def load_dataset_c(sheet_name=None, cfg=None):
    """
    Load Dataset C sheets (read-only, never modified).
    Returns dict or single DataFrame.
    """
    if cfg is None:
        cfg = load_config()
    path = _project_root() / cfg["paths"]["dataset_c"]

    if sheet_name is None:
        sheets = DATASET_C_SHEETS
    elif isinstance(sheet_name, str):
        sheets = [sheet_name]
    else:
        sheets = list(sheet_name)

    result = {}
    for s in sheets:
        df = pd.read_excel(path, sheet_name=s)
        result[s] = df

    if isinstance(sheet_name, str):
        return result[sheet_name]
    return result


def load_port_weather(cfg=None):
    """Load and parse port_weather with date column."""
    df = load_dataset_c("port_weather", cfg=cfg)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["port_name", "date"]).reset_index(drop=True)
    return df


def load_cyclone_events(cfg=None):
    """Load and parse cyclone_events with start/end times."""
    df = load_dataset_c("cyclone_events", cfg=cfg)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    return df


def load_geopolitical_events(cfg=None):
    """Load and parse geopolitical_events with event_date."""
    df = load_dataset_c("geopolitical_events", cfg=cfg)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Quick validation helpers
# ──────────────────────────────────────────────
def check_duplicate_dates(df, date_col="date"):
    """Return count of duplicate dates."""
    return df[date_col].duplicated().sum()


def check_missingness(df):
    """Return Series of null counts per column."""
    return df.isnull().sum()


def check_infinities(df):
    """Return Series of infinity counts per numeric column."""
    numeric = df.select_dtypes(include=[np.number])
    return np.isinf(numeric).sum()


if __name__ == "__main__":
    # Quick smoke test
    cfg = load_config()
    print("Loading Dataset A...")
    a = load_dataset_a(cfg)
    print(f"  Shape: {a.shape}, Date: {a['date'].min()} → {a['date'].max()}")

    print("Loading Dataset B (ports)...")
    ports = load_dataset_b("ports", cfg)
    print(f"  Ports: {list(ports['port_name'])}")

    print("Loading Dataset C (geopolitical_events)...")
    ge = load_geopolitical_events(cfg)
    print(f"  Shape: {ge.shape}, Date: {ge['event_date'].min()} → {ge['event_date'].max()}")
    print("Done.")
