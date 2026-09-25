"""Validate a historical procurement ledger before using it for economic claims."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "economic_data_readiness"
REQUIRED = [
    "voyage_id", "decision_timestamp", "vessel_class", "route", "origin", "destination",
    "actual_volume_mt", "vessel_capacity_mt", "duration_days", "procurement_strategy", "contract_id",
    "contract_start_date", "contract_end_date", "spot_rate_usd_per_mt", "contract_rate_usd_per_mt",
    "realized_procurement_cost_usd", "execution_slippage_usd", "budget_period", "budget_limit_usd",
    "contract_capacity_limit_mt", "source_record_id", "source_system",
]
STRATEGIES = {"SPOT", "SHORT_TERM", "MEDIUM_TERM", "MULTI_VOYAGE_CONTRACT"}


def validate(path: Path) -> dict:
    errors: list[str] = []
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    errors.extend(f"missing column: {c}" for c in missing)
    if missing:
        return {"status": "INVALID", "path": str(path), "rows": len(df), "errors": errors}
    if df.empty:
        return {"status": "TEMPLATE_ONLY", "path": str(path), "rows": 0, "errors": [], "message": "Header is valid; populate historical rows before economic evaluation."}

    for col in ["actual_volume_mt", "vessel_capacity_mt", "duration_days", "spot_rate_usd_per_mt", "realized_procurement_cost_usd", "execution_slippage_usd", "budget_limit_usd", "contract_capacity_limit_mt"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            errors.append(f"non-numeric values in {col}")
    decision = pd.to_datetime(df.decision_timestamp, errors="coerce")
    if decision.isna().any():
        errors.append("unparseable decision_timestamp")
    if df.voyage_id.isna().any() or df.voyage_id.astype(str).str.strip().eq("").any():
        errors.append("blank voyage_id")
    if df.voyage_id.duplicated().any():
        errors.append("voyage_id is not unique")
    invalid_strategy = ~df.procurement_strategy.isin(STRATEGIES)
    if invalid_strategy.any():
        errors.append("unsupported procurement_strategy")
    volume = pd.to_numeric(df.actual_volume_mt, errors="coerce")
    capacity = pd.to_numeric(df.vessel_capacity_mt, errors="coerce")
    if (volume <= 0).any() or (capacity <= 0).any() or (volume > capacity).any():
        errors.append("invalid volume/capacity relationship")
    duration = pd.to_numeric(df.duration_days, errors="coerce")
    if (duration <= 0).any():
        errors.append("duration_days must be positive")
    for col in ["contract_start_date", "contract_end_date"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().any() and df.procurement_strategy.ne("SPOT").any():
            errors.append(f"missing/unparseable {col} for contract rows")
    start = pd.to_datetime(df.contract_start_date, errors="coerce")
    end = pd.to_datetime(df.contract_end_date, errors="coerce")
    if (end < start).fillna(False).any():
        errors.append("contract_end_date precedes contract_start_date")
    contract_rows = df.procurement_strategy.ne("SPOT")
    if df.loc[contract_rows, "contract_id"].isna().any() or df.loc[contract_rows, "contract_id"].astype(str).str.strip().eq("").any():
        errors.append("contract rows require contract_id")
    contract_rate = pd.to_numeric(df.contract_rate_usd_per_mt, errors="coerce")
    if contract_rate.loc[contract_rows].isna().any() or (contract_rate.loc[contract_rows] <= 0).any():
        errors.append("contract rows require positive contract_rate_usd_per_mt")
    for col in ["source_record_id", "source_system"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            errors.append(f"blank provenance field: {col}")
    return {"status": "VALID" if not errors else "INVALID", "path": str(path), "rows": len(df), "errors": errors, "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    result = validate(args.ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "procurement_ledger_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] in {"VALID", "TEMPLATE_ONLY"} else 1)


if __name__ == "__main__":
    main()
