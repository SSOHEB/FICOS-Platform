"""Stress-test whether portfolio coupling matters when constraints bind.

The forecasting model and canonical folds are unchanged. This script creates one
fresh OOS prediction table, then varies only explicit economic scenario inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.canonical_config import CANONICAL_DATASET_SHA256
from ml.planning import DeterministicMILPPlanner, PlanningConstraints, Voyage, VoyageOpportunity
from scripts.run_architectural_ablation import FEATURE_COUNT, SEED, VESSELS, fresh_forecasts

OUT = ROOT / "outputs" / "experiments" / "architectural_stress_grid"
N_VALUES = (4, 8, 12)
DISCOUNTS = (0.01, 0.045, 0.10)
CAPACITY_MULTIPLIERS = (0.25, 0.50, 1.00)
BUDGET_MULTIPLIERS = (0.90, 1.00, 1.10)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def opportunities(selected: pd.DataFrame, discount: float) -> list[VoyageOpportunity]:
    result = []
    for i, row in selected.reset_index(drop=True).iterrows():
        voyage = Voyage(
            voyage_id=f"STRESS-{i + 1:02d}", vessel_class=row.vessel.upper(), route="SCENARIO_ROUTE",
            origin="SCENARIO_ORIGIN", destination="SCENARIO_DESTINATION", laycan_start=row.date,
            laycan_end=row.date, expected_duration_days=20, volume_mt=75_000.0, capacity_mt=82_000.0,
            timestamp=row.date, metadata={"when_decision": row.when_decision},
        )
        spot = max(1.0, row.current_rate + row.predicted_delta) * voyage.expected_duration_days
        costs = {"SPOT": spot, "SHORT_TERM": spot * (1 - discount * 0.33), "MEDIUM_TERM": spot * (1 - discount * 0.66), "MULTI_VOYAGE_CONTRACT": spot * (1 - discount)}
        result.append(VoyageOpportunity(voyage, row.current_rate, row.current_rate + row.predicted_delta, row.lower_bound, row.upper_bound, costs, "RISING" if row.predicted_delta > 0 else "FALLING", "validation residual P10/P90", ("SCENARIO_ASSUMPTION: volume", "SCENARIO_ASSUMPTION: discount", "SCENARIO_ASSUMPTION: capacity")))
    return result


def main() -> None:
    data_path = ROOT / "data" / "modeling_dataset.csv"
    assert sha256(data_path) == CANONICAL_DATASET_SHA256
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    features = [c for c in df.columns if c != "date" and not c.startswith("target_") and not c.startswith("dir_")]
    assert len(features) == FEATURE_COUNT
    predictions = fresh_forecasts(df, features)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "fresh_oos_predictions.csv", index=False)

    rows = []
    for n in N_VALUES:
        selected = predictions.sort_values(["date", "vessel"]).tail(n).reset_index(drop=True)
        wait_count = int(selected.when_decision.eq("WAIT").sum())
        action_indices = [i for i, row in selected.iterrows() if row.when_decision != "WAIT"]
        for discount in DISCOUNTS:
            ops = opportunities(selected, discount)
            action_ops = [ops[i] for i in action_indices]
            unconstrained_independent = [min(("SPOT", "MULTI_VOYAGE_CONTRACT"), key=lambda s: o.strategy_costs[s]) for o in action_ops]
            spot_total = sum(o.strategy_costs["SPOT"] for o in action_ops)
            for cap_multiplier in CAPACITY_MULTIPLIERS:
                capacity = max(75_000.0, cap_multiplier * max(1, len(action_ops)) * 75_000.0)
                for budget_multiplier in BUDGET_MULTIPLIERS:
                    budget = spot_total * budget_multiplier
                    constraints = PlanningConstraints(budget_usd=budget, contract_capacity_mt=capacity, max_contracts=max(1, int(capacity // 75_000)))
                    result = DeterministicMILPPlanner().solve(action_ops, constraints) if action_ops else None
                    independent_contracts = int(sum(s == "MULTI_VOYAGE_CONTRACT" for s in unconstrained_independent))
                    if result is None or result.solver_status != "OPTIMAL":
                        rows.append({"opportunities": n, "wait_decisions": wait_count, "actionable_decisions": len(action_ops), "discount_pct": discount * 100, "capacity_mt": capacity, "budget_multiplier": budget_multiplier, "budget_usd": budget, "independent_contracts": independent_contracts, "joint_contracts": None, "decision_changes": None, "independent_capacity_violation_mt": max(0.0, independent_contracts * 75_000 - capacity), "joint_forecast_cost_usd": None, "independent_forecast_cost_usd": sum(o.strategy_costs[s] for o, s in zip(action_ops, unconstrained_independent)), "coupling_cost_difference_usd": None, "solver_status": "INFEASIBLE" if result is None else result.solver_status})
                        continue
                    joint = [s for _, s in result.selected]
                    independent_cost = sum(o.strategy_costs[s] for o, s in zip(action_ops, unconstrained_independent))
                    joint_cost = sum(o.strategy_costs[s] for o, s in zip(action_ops, joint))
                    rows.append({"opportunities": n, "wait_decisions": wait_count, "actionable_decisions": len(action_ops), "discount_pct": discount * 100, "capacity_mt": capacity, "budget_multiplier": budget_multiplier, "budget_usd": budget, "independent_contracts": independent_contracts, "joint_contracts": int(sum(s == "MULTI_VOYAGE_CONTRACT" for s in joint)), "decision_changes": int(sum(a != b for a, b in zip(unconstrained_independent, joint))), "independent_capacity_violation_mt": max(0.0, independent_contracts * 75_000 - capacity), "joint_forecast_cost_usd": joint_cost, "independent_forecast_cost_usd": independent_cost, "coupling_cost_difference_usd": independent_cost - joint_cost, "solver_status": result.solver_status})

    grid = pd.DataFrame(rows)
    grid["independent_capacity_utilization"] = (grid["independent_contracts"] * 75_000.0) / grid["capacity_mt"]
    grid["independent_budget_utilization"] = grid["independent_forecast_cost_usd"] / grid["budget_usd"]
    grid["coupling_activation"] = grid[["independent_capacity_utilization", "independent_budget_utilization"]].ge(1.0).any(axis=1)
    grid["coupling_activation_reason"] = np.where(
        grid["independent_capacity_utilization"] >= 1.0,
        "contract_capacity_at_or_above_threshold",
        np.where(grid["independent_budget_utilization"] >= 1.0, "budget_at_or_above_threshold", "none"),
    )
    grid.to_csv(OUT / "stress_grid_results.csv", index=False)
    summary = {"experiment_id": "FICOS_ARCHITECTURAL_STRESS_GRID", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": sha256(data_path), "seed": SEED, "fresh_oos_predictions": len(predictions), "grid_rows": len(grid), "scenario_dimensions": {"opportunities": N_VALUES, "discount_pct": [d * 100 for d in DISCOUNTS], "capacity_multiplier": CAPACITY_MULTIPLIERS, "budget_multiplier": BUDGET_MULTIPLIERS}, "interpretation": "The grid varies scenario assumptions only; it does not modify the canonical forecasting model or claim commercial causality."}
    (OUT / "stress_grid_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(grid.groupby("solver_status").size().to_string())
    print(grid[grid.solver_status.eq("OPTIMAL")].sort_values("decision_changes", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
