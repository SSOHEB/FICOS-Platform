"""Replay frozen forecasts over observable history as a market counterfactual.

This script never reads a SAIL ledger and never labels a counterfactual cost as
an actual company cost. Contract economics are explicit scenario assumptions.
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
from ml.counterfactual import HistoricalMarketState, build_opportunity
from scripts.run_architectural_ablation import FEATURE_COUNT, fresh_forecasts

OUT = ROOT / "outputs" / "experiments" / "historical_counterfactual"
SEED = 42
VOLUME_MT = 75_000.0
CONTRACT_DISCOUNT = 0.045


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_costs(row: pd.Series) -> dict[str, float | None]:
    current = float(row.current_rate) * VOLUME_MT
    future = float(row.actual_rate) * VOLUME_MT
    contract = current * (1.0 - CONTRACT_DISCOUNT)
    wait = row.when_decision == "WAIT"
    fixed_horizon_wait = float(row.predicted_delta) < 0.0
    return {
        "ALWAYS_SPOT": current,
        "ALWAYS_CONTRACT": contract,
        "FIXED_HORIZON_POLICY": future if fixed_horizon_wait else current,
        "TIMING_ONLY": future if wait else current,
        "TIMING_PLUS_HOW": future if wait else contract,
        "FICOS_PORTFOLIO": None,
    }


def bootstrap_ci(values: np.ndarray, seed: int = SEED, draws: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    data_path = ROOT / "data" / "modeling_dataset.csv"
    assert sha256(data_path) == CANONICAL_DATASET_SHA256
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    features = [c for c in df.columns if c != "date" and not c.startswith("target_") and not c.startswith("dir_")]
    assert len(features) == FEATURE_COUNT
    predictions = fresh_forecasts(df, features)
    OUT.mkdir(parents=True, exist_ok=True)

    replay_rows = []
    opportunity_rows = []
    policy_rows = []
    for index, row in predictions.iterrows():
        state = HistoricalMarketState(
            date=str(row.date), vessel=str(row.vessel), current_rate=float(row.current_rate),
            actual_future_rate=float(row.actual_rate), forecast_delta=float(row.predicted_delta),
            lower_bound=float(row.lower_bound), upper_bound=float(row.upper_bound),
            when_decision=str(row.when_decision),
            observed_features={"date": str(row.date), "vessel": str(row.vessel), "current_rate": float(row.current_rate)},
            provenance={"date": "OBSERVED", "vessel": "OBSERVED", "current_rate": "OBSERVED", "actual_future_rate": "OUTCOME_ONLY_NOT_INPUT", "forecast_delta": "RECONSTRUCTED_TRAINING_ONLY", "uncertainty_bounds": "RECONSTRUCTED_VALIDATION_RESIDUALS", "when_decision": "RECONSTRUCTED_LOCKED_GATE"},
        )
        opportunity = build_opportunity(index + 1, state)
        opportunity_rows.append(opportunity.to_dict())
        replay_rows.append({"opportunity_id": opportunity.opportunity_id, "date": state.date, "vessel": state.vessel, "current_rate": state.current_rate, "actual_future_rate_outcome_only": state.actual_future_rate, "forecast_delta": state.forecast_delta, "lower_bound": state.lower_bound, "upper_bound": state.upper_bound, "when_decision": state.when_decision, "status": opportunity.status, "evidence_type": opportunity.evidence_type, "rate_provenance": "OBSERVED", "forecast_provenance": "RECONSTRUCTED", "private_procurement_fields": "PRIVATE_DATA_UNAVAILABLE"})
        costs = policy_costs(row)
        policy_rows.append({"opportunity_id": opportunity.opportunity_id, "date": state.date, **costs})

    replay = pd.DataFrame(replay_rows)
    opportunities = pd.DataFrame(opportunity_rows)
    costs = pd.DataFrame(policy_rows)
    replay.to_csv(OUT / "forecast_replay.csv", index=False)
    opportunities.to_csv(OUT / "historical_market_opportunities.csv", index=False)

    benchmark_rows = []
    baseline = costs.ALWAYS_SPOT
    for policy in ["ALWAYS_SPOT", "ALWAYS_CONTRACT", "FIXED_HORIZON_POLICY", "TIMING_ONLY", "TIMING_PLUS_HOW"]:
        series = costs[policy].dropna().to_numpy(float)
        difference = baseline.to_numpy(float) - series
        lo, hi = bootstrap_ci(difference)
        benchmark_rows.append({"policy": policy, "historical_opportunities": len(series), "cost_usd": float(series.sum()), "difference_vs_always_spot_usd": float(difference.sum()), "mean_difference_per_opportunity_usd": float(difference.mean()), "paired_bootstrap_95pct_difference_usd": f"[{lo:.2f}, {hi:.2f}]", "feasibility": "ROW_LEVEL_NO_SHARED_CONSTRAINTS", "evidence_type": "COUNTERFACTUAL_WITH_SCENARIO_CONTRACT_ECONOMICS" if policy != "ALWAYS_SPOT" else "OBSERVED_MARKET_RATE_COUNTERFACTUAL"})
    benchmark_rows.append({"policy": "FICOS_PORTFOLIO", "historical_opportunities": 12, "cost_usd": None, "difference_vs_always_spot_usd": None, "mean_difference_per_opportunity_usd": None, "paired_bootstrap_95pct_difference_usd": "UNAVAILABLE", "feasibility": "EVALUATED_IN_ARCHITECTURAL_ABLATION_AND_STRESS_GRID", "evidence_type": "COUNTERFACTUAL_SCENARIO_NOT_FULL_HISTORY"})
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(OUT / "policy_benchmark.csv", index=False)

    inventory = {"existing_forecasting_layer": "fresh canonical RF replay, 100 trees, depth 5, seed 42, train-only preprocessing", "existing_oos_predictions": "4804 fresh rows; 641 canonical gate-retained", "existing_when_logic": "validation residual P10/P90 plus 1% relative threshold", "existing_how_logic": "SPOT/contract strategy selection", "existing_milp": "scipy.optimize.milp / HiGHS", "existing_robust_cvar": "implemented and empirically ablated", "existing_constraint_stress": "81-cell grid; 59 optimal, 22 infeasible", "existing_provenance": "dataset and source hashes in authoritative reports", "existing_public_sources": ["freight rates", "macro", "weather", "cyclones", "GDELT-derived events", "ports/berths/traffic/capacity", "merchant fleet"], "assumptions": ["volume", "duration", "contract discount", "budget", "contract capacity", "execution slippage"], "limitations": ["private SAIL procurement fields unavailable", "historical market opportunities are not SAIL transactions"]}
    (OUT / "current_architecture_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    replay_summary = {"rows": len(replay), "date_min": replay.date.min(), "date_max": replay.date.max(), "canonical_gate_retained": int(predictions.gate_retained.sum()), "wait_decisions": int(predictions.when_decision.eq("WAIT").sum()), "actual_sail_cost": "PRIVATE_DATA_UNAVAILABLE", "contract_economics": "SCENARIO_ASSUMPTION", "seed": SEED, "dataset_sha256": sha256(data_path)}
    (OUT / "forecast_replay.json").write_text(json.dumps(replay_summary, indent=2), encoding="utf-8")
    print(benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
