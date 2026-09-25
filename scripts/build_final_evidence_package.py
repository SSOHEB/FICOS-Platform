"""Assemble the final evidence manifest without changing canonical results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "authoritative"
CF = ROOT / "outputs" / "experiments" / "historical_counterfactual"
ABL = ROOT / "outputs" / "experiments" / "architectural_ablation"
STRESS = ROOT / "outputs" / "experiments" / "architectural_stress_grid"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    benchmark = pd.read_csv(CF / "policy_benchmark.csv").where(pd.notna, None).to_dict("records")
    master = pd.read_csv(ABL / "master_ablation_table.csv").where(pd.notna, None).to_dict("records")
    controlled = pd.read_csv(ABL / "controlled_incremental_value_table.csv").where(pd.notna, None).to_dict("records")
    stress = pd.read_csv(STRESS / "stress_grid_results.csv")
    evidence = {
        "experiment_id": "FICOS_FINAL_HISTORICAL_COUNTERFACTUAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_identity": {"path": "data/modeling_dataset.csv", "sha256": sha256(ROOT / "data/modeling_dataset.csv"), "rows": 2581, "columns": 482, "date_range": ["2016-01-04", "2026-09-04"]},
        "source_identities": {"interim_spot_prices": sha256(ROOT / "data/interim/cleaned_spot_prices.csv"), "raw_maritime_workbook": sha256(ROOT / "data/raw/maritime_macro_data.xlsx"), "raw_port_workbook": sha256(ROOT / "data/raw/baltic_freight_indices.xlsx")},
        "forecast_population": {"fresh_oos_rows": 4804, "canonical_gate_retained": 641, "gate_coverage": 0.1334304746, "gated_directional_accuracy": 0.7909516381, "model": "RF_STANDARD_FRESH_ABLATION", "seed": 42},
        "opportunity_population": {"historical_market_opportunities": 4804, "status": "HISTORICAL_MARKET_OPPORTUNITY_NOT_SAIL_TRANSACTION"},
        "policy_benchmark": benchmark,
        "architectural_ablation": {"master": master, "controlled_incremental": controlled},
        "coupling_stress": {"grid_cells": int(len(stress)), "optimal_cells": int((stress.solver_status == "OPTIMAL").sum()), "infeasible_cells": int((stress.solver_status == "FAILED").sum()), "evidence": "constraint feasibility and allocation changes; not causal savings"},
        "statistical_results": {"historical_policy_bootstrap": "descriptive paired bootstrap intervals under counterfactual assumptions", "causal_significance": "NOT_ESTABLISHED", "private_cost_reconciliation": "UNAVAILABLE"},
        "assumption_counts": {"scenario_assumptions": 6, "private_fields_unavailable": 7},
        "private_sail_fields_unavailable": ["contract prices", "negotiated discounts", "voyage-specific procurement volumes", "realized procurement costs", "historical procurement decisions", "budget allocations", "contract-capacity commitments"],
        "reproducibility": {"code": ["scripts/run_historical_counterfactual.py", "scripts/run_architectural_ablation.py", "scripts/run_architectural_stress_grid.py"], "canonical_model_config": "configs/canonical_config.py", "seed": 42},
        "claim_boundary": "Public historical market-counterfactual evidence; no actual SAIL savings claim.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FICOS_FINAL_EVIDENCE.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(OUT / 'FICOS_FINAL_EVIDENCE.json'), "policies": len(benchmark), "ablation_systems": len(master)}, indent=2))


if __name__ == "__main__":
    main()
