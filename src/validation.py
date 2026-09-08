"""
SIH26006 — Data Validation
Comprehensive validation of all datasets. Produces reports under reports/.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import (
    load_dataset_a, load_dataset_b, load_port_weather,
    load_cyclone_events, load_geopolitical_events,
    load_config, check_duplicate_dates, check_missingness, check_infinities,
    DATASET_B_SHEETS,
)


def validate_dataset_a(cfg=None):
    """Validate Dataset A and return summary dict."""
    df = load_dataset_a(cfg)
    report = {
        "dataset": "A",
        "shape": list(df.shape),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "duplicate_dates": int(check_duplicate_dates(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missingness": {c: int(v) for c, v in check_missingness(df).items()},
        "infinities": {c: int(v) for c, v in check_infinities(df).items() if v > 0},
        "total_nulls": int(df.isnull().sum().sum()),
    }

    # Date gap analysis
    dates = df["date"].sort_values()
    gaps = dates.diff().dt.days
    report["date_gaps"] = {
        "max_gap_days": int(gaps.max()) if not gaps.isna().all() else 0,
        "mean_gap_days": float(round(gaps.mean(), 2)) if not gaps.isna().all() else 0,
        "gaps_over_3_days": int((gaps > 3).sum()),
    }

    # Freight target stats
    targets = ["kdci", "cape", "panamax", "supramax", "handy"]
    target_stats = {}
    for t in targets:
        if t in df.columns:
            s = df[t].dropna()
            target_stats[t] = {
                "count": int(len(s)),
                "mean": float(round(s.mean(), 2)),
                "std": float(round(s.std(), 2)),
                "min": float(s.min()),
                "max": float(s.max()),
                "nulls": int(df[t].isnull().sum()),
            }
    report["freight_target_stats"] = target_stats

    return df, report


def validate_dataset_b(cfg=None):
    """Validate Dataset B sheets and return summary dict."""
    all_sheets = load_dataset_b(sheet_name=None, cfg=cfg)
    report = {"dataset": "B", "sheets": {}}

    for name, df in all_sheets.items():
        report["sheets"][name] = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "nulls": int(df.isnull().sum().sum()),
        }

    return all_sheets, report


def validate_dataset_c(cfg=None):
    """Validate Dataset C sheets and return summary dict."""
    report = {"dataset": "C", "sheets": {}}

    # Port weather
    pw = load_port_weather(cfg)
    report["sheets"]["port_weather"] = {
        "shape": list(pw.shape),
        "date_min": str(pw["date"].min()),
        "date_max": str(pw["date"].max()),
        "ports": list(pw["port_name"].unique()),
        "rows_per_port": {k: int(v) for k, v in pw.groupby("port_name").size().items()},
        "missingness": {c: int(v) for c, v in check_missingness(pw).items() if v > 0},
    }

    # Cyclone events
    cy = load_cyclone_events(cfg)
    report["sheets"]["cyclone_events"] = {
        "shape": list(cy.shape),
        "date_min": str(cy["start_time"].min()),
        "date_max": str(cy["end_time"].max()),
        "basins": list(cy["basin"].unique()),
        "unique_storms": int(cy["storm_id"].nunique()),
        "seasons": sorted([int(x) for x in cy["season"].unique()]),
        "wind_null": int(cy["max_wind"].isnull().sum()),
        "pressure_null": int(cy["min_pressure"].isnull().sum()),
    }

    # Geopolitical events — report ACTUAL date range
    ge = load_geopolitical_events(cfg)
    report["sheets"]["geopolitical_events"] = {
        "shape": list(ge.shape),
        "date_min": str(ge["event_date"].min()),
        "date_max": str(ge["event_date"].max()),
        "quad_classes": sorted([int(x) for x in ge["QuadClass"].unique()]),
        "goldstein_min": float(ge["GoldsteinScale"].min()),
        "goldstein_max": float(ge["GoldsteinScale"].max()),
        "event_root_codes": sorted([int(x) for x in ge["EventRootCode"].unique()]),
        "unique_events": int(ge["GLOBALEVENTID"].nunique()),
        "top_countries": {
            k: int(v) for k, v in
            ge["ActionGeo_CountryCode"].value_counts().head(15).items()
        },
        "missingness": {c: int(v) for c, v in check_missingness(ge).items() if v > 0},
    }

    return {"port_weather": pw, "cyclone_events": cy, "geopolitical_events": ge}, report


def run_full_validation():
    """Run all validations and save reports."""
    cfg = load_config()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    full_report = {}

    print("=" * 60)
    print("VALIDATING DATASET A")
    print("=" * 60)
    df_a, rep_a = validate_dataset_a(cfg)
    full_report["dataset_a"] = rep_a
    print(f"  Shape: {rep_a['shape']}")
    print(f"  Date range: {rep_a['date_min']} → {rep_a['date_max']}")
    print(f"  Duplicate dates: {rep_a['duplicate_dates']}")
    print(f"  Total nulls: {rep_a['total_nulls']}")
    print(f"  Max date gap: {rep_a['date_gaps']['max_gap_days']} days")
    print(f"  Gaps > 3 days: {rep_a['date_gaps']['gaps_over_3_days']}")
    for t, s in rep_a["freight_target_stats"].items():
        print(f"  {t}: count={s['count']}, mean={s['mean']}, nulls={s['nulls']}")

    print()
    print("=" * 60)
    print("VALIDATING DATASET B")
    print("=" * 60)
    all_b, rep_b = validate_dataset_b(cfg)
    full_report["dataset_b"] = rep_b
    for name, info in rep_b["sheets"].items():
        print(f"  {name}: shape={info['shape']}, nulls={info['nulls']}")

    print()
    print("=" * 60)
    print("VALIDATING DATASET C")
    print("=" * 60)
    all_c, rep_c = validate_dataset_c(cfg)
    full_report["dataset_c"] = rep_c

    pw_info = rep_c["sheets"]["port_weather"]
    print(f"  port_weather: shape={pw_info['shape']}")
    print(f"    Date range: {pw_info['date_min']} → {pw_info['date_max']}")
    print(f"    Ports: {pw_info['ports']}")

    cy_info = rep_c["sheets"]["cyclone_events"]
    print(f"  cyclone_events: shape={cy_info['shape']}")
    print(f"    Date range: {cy_info['date_min']} → {cy_info['date_max']}")
    print(f"    Basins: {cy_info['basins']}, Unique storms: {cy_info['unique_storms']}")

    ge_info = rep_c["sheets"]["geopolitical_events"]
    print(f"  geopolitical_events: shape={ge_info['shape']}")
    print(f"    ACTUAL date range: {ge_info['date_min']} → {ge_info['date_max']}")
    print(f"    QuadClasses: {ge_info['quad_classes']}")
    print(f"    GoldsteinScale: {ge_info['goldstein_min']} → {ge_info['goldstein_max']}")
    print(f"    EventRootCodes: {ge_info['event_root_codes']}")
    print(f"    Top countries: {ge_info['top_countries']}")

    # Save JSON report
    json_path = os.path.join(reports_dir, "data_validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n✓ JSON report saved: {json_path}")

    # Save CSV summary
    rows = []
    # Dataset A missingness
    for col, n in rep_a["missingness"].items():
        rows.append({"dataset": "A", "sheet": "-", "column": col, "null_count": n,
                      "total_rows": rep_a["shape"][0],
                      "pct_missing": round(100 * n / rep_a["shape"][0], 2)})
    # Dataset C missingness
    for sheet_name, sheet_info in rep_c["sheets"].items():
        miss = sheet_info.get("missingness", {})
        nrows = sheet_info["shape"][0]
        for col, n in miss.items():
            rows.append({"dataset": "C", "sheet": sheet_name, "column": col,
                          "null_count": n, "total_rows": nrows,
                          "pct_missing": round(100 * n / nrows, 2)})

    csv_path = os.path.join(reports_dir, "data_validation_report.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✓ CSV report saved: {csv_path}")

    return full_report


if __name__ == "__main__":
    run_full_validation()
