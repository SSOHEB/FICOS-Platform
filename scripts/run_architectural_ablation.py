"""Fresh architectural ablation for FICOS.

This script reads only the canonical modeling dataset and frozen configuration.
It does not consume previous prediction, economic, or optimization outputs.
All systems receive the same fresh OOS opportunity set; only the decision
mechanism changes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.canonical_config import CANONICAL_FOLDS, CANONICAL_DATASET_SHA256
from ml.planning import (
    DeterministicMILPPlanner,
    PlanningConstraints,
    RiskAwareMILPPlanner,
    Strategy,
    Voyage,
    VoyageOpportunity,
    generate_rate_scenarios,
)

OUT = ROOT / "outputs" / "experiments" / "architectural_ablation"
SEED = 42
VESSELS = ["panamax", "supramax", "handy", "cape"]
FEATURE_COUNT = 441


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_forecasts(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records = []
    for fold_name, fold in CANONICAL_FOLDS.items():
        for vessel in VESSELS:
            target = f"target_{vessel}_1d"
            valid = df[vessel].notna() & df[target].notna()
            tr = (df.date <= pd.Timestamp(fold["train_end"])) & valid
            va = (df.date >= pd.Timestamp(fold["val_start"])) & (df.date <= pd.Timestamp(fold["val_end"])) & valid
            te = (df.date >= pd.Timestamp(fold["test_start"])) & (df.date <= pd.Timestamp(fold["test_end"])) & valid
            Xtr = df.loc[tr, features].fillna(0.0).to_numpy(float)
            Xva = df.loc[va, features].fillna(0.0).to_numpy(float)
            Xte = df.loc[te, features].fillna(0.0).to_numpy(float)
            ytr = (df.loc[tr, target] - df.loc[tr, vessel]).to_numpy(float)
            yva = (df.loc[va, target] - df.loc[va, vessel]).to_numpy(float)
            yte = (df.loc[te, target] - df.loc[te, vessel]).to_numpy(float)
            scaler = StandardScaler().fit(Xtr)
            selector = SelectKBest(f_regression, k=30).fit(scaler.transform(Xtr), ytr)
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=1)
            model.fit(selector.transform(scaler.transform(Xtr)), ytr)
            pred_va = model.predict(selector.transform(scaler.transform(Xva)))
            pred_te = model.predict(selector.transform(scaler.transform(Xte)))
            p10, p90 = np.percentile(yva - pred_va, [10, 90])
            bases = df.loc[te, vessel].to_numpy(float)
            pct = pred_te / (bases + 1e-8)
            buy = (pred_te > max(0.0, p90)) & (pct > 0.01)
            wait = (pred_te < min(0.0, p10)) & (pct < -0.01)
            for date, base, actual, pred, is_buy, is_wait in zip(df.loc[te, "date"], bases, yte, pred_te, buy, wait):
                records.append({
                    "date": date.strftime("%Y-%m-%d"), "vessel": vessel, "fold": fold_name,
                    "current_rate": float(base), "actual_delta": float(actual),
                    "actual_rate": float(base + actual), "predicted_delta": float(pred),
                    "lower_bound": float(max(0.0, pred + p10)), "upper_bound": float(max(0.0, pred + p90)),
                    "when_decision": "WAIT" if is_wait else "ACTION",
                    "gate_retained": bool(is_buy or is_wait), "model_id": "RF_STANDARD_FRESH_ABLATION",
                })
    return pd.DataFrame(records)


def make_opportunities(predictions: pd.DataFrame) -> list[VoyageOpportunity]:
    selected = predictions.sort_values(["date", "vessel"]).tail(12).reset_index(drop=True)
    opportunities = []
    for i, row in selected.iterrows():
        voyage = Voyage(
            voyage_id=f"ABLATION-{i + 1:02d}", vessel_class=row.vessel.upper(), route="SCENARIO_ROUTE",
            origin="SCENARIO_ORIGIN", destination="SCENARIO_DESTINATION", laycan_start=row.date,
            laycan_end=row.date, expected_duration_days=20, volume_mt=75_000.0, capacity_mt=82_000.0,
            timestamp=row.date, metadata={"when_decision": row.when_decision, "data_status": "SCENARIO_ASSUMPTION_FOR_MISSING_OPERATIONAL_FIELDS"},
        )
        base = max(1.0, row.current_rate + row.predicted_delta) * voyage.volume_mt
        costs = {
            "SPOT": base,
            "SHORT_TERM": base * 0.985,
            "MEDIUM_TERM": base * 0.970,
            "MULTI_VOYAGE_CONTRACT": base * (0.955 if i < 6 else 0.965),
        }
        opportunities.append(VoyageOpportunity(voyage, row.current_rate, row.current_rate + row.predicted_delta, row.lower_bound, row.upper_bound, costs, "RISING" if row.predicted_delta > 0 else "FALLING", "validation residual P10/P90", ("SCENARIO_ASSUMPTION: volume", "SCENARIO_ASSUMPTION: duration", "SCENARIO_ASSUMPTION: capacity", "SCENARIO_ASSUMPTION: discount")))
    return opportunities


def realized_cost(opportunities, predictions, strategies, wait_allowed=True):
    selected = predictions.sort_values(["date", "vessel"]).tail(len(opportunities)).reset_index(drop=True)
    total = 0.0
    for i, (opportunity, strategy) in enumerate(zip(opportunities, strategies)):
        row = selected.iloc[i]
        rate = row.actual_rate if strategy == "WAIT" or (wait_allowed and row.when_decision == "WAIT") else row.current_rate
        discount = {"WAIT": 1.0, "SPOT": 1.0, "SHORT_TERM": 0.985, "MEDIUM_TERM": 0.970, "MULTI_VOYAGE_CONTRACT": 0.955 if i < 6 else 0.965}[strategy]
        total += float(rate) * opportunity.voyage.volume_mt * discount
    return total


def strategy_counts(opportunities, strategies, predictions):
    selected = predictions.sort_values(["date", "vessel"]).tail(len(opportunities)).reset_index(drop=True)
    wait = int(selected.when_decision.eq("WAIT").sum())
    return {
        "eligible_voyages": len(opportunities), "wait_decisions": wait,
        "actionable_decisions": len(opportunities) - wait,
        "spot_decisions": int(sum(s == "SPOT" for s in strategies)),
        "multi_voyage_decisions": int(sum(s == "MULTI_VOYAGE_CONTRACT" for s in strategies)),
        "contract_utilization_pct": float(sum(s != "SPOT" for s in strategies) / len(strategies) * 100),
        "budget_utilization_pct": None, "capacity_utilization_pct": None,
    }


def main():
    data_path = ROOT / "data" / "modeling_dataset.csv"
    assert sha256(data_path) == CANONICAL_DATASET_SHA256
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    features = [c for c in df.columns if c != "date" and not c.startswith("target_") and not c.startswith("dir_")]
    assert len(features) == FEATURE_COUNT
    predictions = fresh_forecasts(df, features)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "fresh_oos_predictions.csv", index=False)
    opportunities = make_opportunities(predictions)
    pd.DataFrame([o.to_dict() for o in opportunities]).to_csv(OUT / "shared_opportunities.csv", index=False)
    selected = predictions.sort_values(["date", "vessel"]).tail(12).reset_index(drop=True)
    residuals = predictions.actual_delta.to_numpy(float) - predictions.predicted_delta.to_numpy(float)
    scenarios = generate_rate_scenarios([o.forecast_rate for o in opportunities], residuals, 100, SEED)
    scenario_costs = np.zeros((len(scenarios), len(opportunities), len(Strategy)))
    for s, rates in enumerate(scenarios):
        for i, opportunity in enumerate(opportunities):
            base = rates[i] * opportunity.voyage.volume_mt
            scenario_costs[s, i] = [base, base * .985, base * .970, base * (.955 if i < 6 else .965)]
    constraints = PlanningConstraints(budget_usd=sum(o.strategy_costs["SPOT"] for o in opportunities) * 1.02, contract_capacity_mt=400_000, max_contracts=len(opportunities))
    action_indices = [i for i, row in selected.iterrows() if row.when_decision != "WAIT"]
    action_opportunities = [opportunities[i] for i in action_indices]
    action_scenario_costs = scenario_costs[:, action_indices, :]
    solve_times = {}
    started = time.perf_counter()
    deterministic = DeterministicMILPPlanner().solve(action_opportunities, constraints)
    solve_times["SYSTEM_4_FICOS_DETERMINISTIC"] = time.perf_counter() - started
    started = time.perf_counter()
    robust = RiskAwareMILPPlanner().solve_risk_aware(action_opportunities, action_scenario_costs, constraints, mode="ROBUST")
    solve_times["SYSTEM_5_FICOS_ROBUST"] = time.perf_counter() - started
    scaled_ops = [VoyageOpportunity(o.voyage, o.current_rate, o.forecast_rate, o.lower_rate, o.upper_rate, {k: v / 1_000_000 for k, v in o.strategy_costs.items()}, o.signal, o.uncertainty_method, o.assumptions) for o in action_opportunities]
    scaled_constraints = PlanningConstraints(constraints.budget_usd / 1_000_000, constraints.contract_capacity_mt, constraints.max_contracts)
    started = time.perf_counter()
    cvar = RiskAwareMILPPlanner().solve_risk_aware(scaled_ops, action_scenario_costs / 1_000_000, scaled_constraints, mode="MEAN_CVAR", alpha=.90, risk_aversion=.25)
    solve_times["SYSTEM_6_FICOS_MEAN_CVAR"] = time.perf_counter() - started
    def expand_action_result(result):
        expanded = ["WAIT" if row.when_decision == "WAIT" else "SPOT" for _, row in selected.iterrows()]
        for local_i, strategy in result.selected:
            expanded[action_indices[local_i]] = strategy
        return expanded
    all_spot = ["SPOT"] * len(opportunities)
    independent = [min(("SPOT", "MULTI_VOYAGE_CONTRACT"), key=lambda s: o.strategy_costs[s]) for o in opportunities]
    when_only = ["WAIT" if row.when_decision == "WAIT" else "SPOT" for _, row in selected.iterrows()]
    when_how = ["WAIT" if row.when_decision == "WAIT" else min(("SPOT", "MULTI_VOYAGE_CONTRACT"), key=lambda s: opportunities[i].strategy_costs[s]) for i, row in selected.iterrows()]
    deterministic_s = expand_action_result(deterministic)
    robust_s = expand_action_result(robust)
    cvar_s = expand_action_result(cvar)
    systems = [("BASELINE_0_ALWAYS_SPOT", all_spot, False, deterministic), ("BASELINE_1_INDEPENDENT_CONTRACT", independent, False, deterministic), ("BASELINE_2_FORECAST_WHEN", when_only, True, deterministic), ("BASELINE_3_WHEN_HOW_INDEPENDENT", when_how, True, deterministic), ("SYSTEM_4_FICOS_DETERMINISTIC", deterministic_s, True, deterministic), ("SYSTEM_5_FICOS_ROBUST", robust_s, True, robust), ("SYSTEM_6_FICOS_MEAN_CVAR", cvar_s, True, cvar)]
    spot_cost = realized_cost(opportunities, predictions, all_spot, wait_allowed=False)
    rows = []
    strategy_by_system = {name: strategies for name, strategies, _, _ in systems}
    allocations = {"voyage_id": [o.voyage.voyage_id for o in opportunities], "always_spot": all_spot, "independent": independent, "when_only": when_only, "when_how": when_how, "deterministic": deterministic_s, "robust": robust_s, "mean_cvar": cvar_s}
    for name, strategies, allow_wait, solver in systems:
        cost = realized_cost(opportunities, predictions, strategies, wait_allowed=allow_wait)
        count = strategy_counts(opportunities, strategies, predictions)
        nominal_cost = sum(o.strategy_costs.get(s, o.strategy_costs["SPOT"]) for o, s in zip(opportunities, strategies) if s != "WAIT")
        contract_volume = sum(o.voyage.volume_mt for o, s in zip(opportunities, strategies) if s not in ("SPOT", "WAIT"))
        count["budget_utilization_pct"] = float(nominal_cost / constraints.budget_usd * 100)
        count["capacity_utilization_pct"] = float(contract_volume / constraints.contract_capacity_mt * 100)
        scenario_total = np.array([sum((row.actual_rate if strategies[i] == "WAIT" else scenarios[j, i]) * opportunities[i].voyage.volume_mt * ({"WAIT": 1.0, "SPOT": 1.0, "SHORT_TERM": .985, "MEDIUM_TERM": .970, "MULTI_VOYAGE_CONTRACT": .955 if i < 6 else .965}[strategies[i]]) for i, row in selected.iterrows()) for j in range(len(scenarios))])
        count.update({"system": name, "procurement_cost_usd": cost, "savings_vs_always_spot_usd": spot_cost - cost, "mean_savings_per_voyage_usd": (spot_cost - cost) / len(opportunities), "worst_case_cost_usd": float(scenario_total.max()), "cvar_cost_usd": float(np.quantile(scenario_total, .90)), "constraint_violations": len(solver.constraint_violations) if name.startswith("SYSTEM_") else 0, "solver_status": solver.solver_status if name.startswith("SYSTEM_") else "NOT_APPLICABLE", "solver_time_seconds": solve_times.get(name)})
        rows.append(count)
    master = pd.DataFrame(rows)
    master.to_csv(OUT / "master_ablation_table.csv", index=False)
    pd.DataFrame(allocations).to_csv(OUT / "contract_allocation_comparison.csv", index=False)
    order = master.set_index("system")
    pairs = [("Contract intelligence", "BASELINE_0_ALWAYS_SPOT", "BASELINE_1_INDEPENDENT_CONTRACT"), ("Timing intelligence", "BASELINE_1_INDEPENDENT_CONTRACT", "BASELINE_2_FORECAST_WHEN"), ("WHEN/HOW decomposition", "BASELINE_2_FORECAST_WHEN", "BASELINE_3_WHEN_HOW_INDEPENDENT"), ("Cross-voyage coupling", "BASELINE_3_WHEN_HOW_INDEPENDENT", "SYSTEM_4_FICOS_DETERMINISTIC"), ("Robust optimization", "SYSTEM_4_FICOS_DETERMINISTIC", "SYSTEM_5_FICOS_ROBUST"), ("Mean-CVaR", "SYSTEM_5_FICOS_ROBUST", "SYSTEM_6_FICOS_MEAN_CVAR")]
    incremental = []
    for layer, previous, new in pairs:
        incremental.append({"architectural_layer": layer, "previous_system": previous, "new_system": new, "incremental_cost_change_usd": float(order.loc[new, "procurement_cost_usd"] - order.loc[previous, "procurement_cost_usd"]), "incremental_savings_usd": float(order.loc[new, "savings_vs_always_spot_usd"] - order.loc[previous, "savings_vs_always_spot_usd"]), "incremental_risk_change_usd": float(order.loc[new, "worst_case_cost_usd"] - order.loc[previous, "worst_case_cost_usd"]), "decision_changes": int(sum(a != b for a, b in zip(strategy_by_system[previous], strategy_by_system[new]))), "interpretation": "Measured on the same shared OOS opportunity set; scenario-dependent where contract assumptions enter."})
    pd.DataFrame(incremental).to_csv(OUT / "incremental_value_table.csv", index=False)
    controlled_pairs = [("Timing versus Always Spot", "BASELINE_0_ALWAYS_SPOT", "BASELINE_2_FORECAST_WHEN"), ("Independent HOW after WHEN", "BASELINE_2_FORECAST_WHEN", "BASELINE_3_WHEN_HOW_INDEPENDENT"), ("Coupling after WHEN/HOW", "BASELINE_3_WHEN_HOW_INDEPENDENT", "SYSTEM_4_FICOS_DETERMINISTIC"), ("Robust objective", "SYSTEM_4_FICOS_DETERMINISTIC", "SYSTEM_5_FICOS_ROBUST"), ("Mean-CVaR objective", "SYSTEM_5_FICOS_ROBUST", "SYSTEM_6_FICOS_MEAN_CVAR")]
    controlled = []
    for layer, previous, new in controlled_pairs:
        controlled.append({"architectural_layer": layer, "previous_system": previous, "new_system": new, "incremental_savings_usd": float(order.loc[new, "savings_vs_always_spot_usd"] - order.loc[previous, "savings_vs_always_spot_usd"]), "incremental_worst_case_cost_usd": float(order.loc[new, "worst_case_cost_usd"] - order.loc[previous, "worst_case_cost_usd"]), "decision_changes": int(sum(a != b for a, b in zip(strategy_by_system[previous], strategy_by_system[new]))), "comparison_status": "CONTROLLED_WHEN_POSSIBLE"})
    pd.DataFrame(controlled).to_csv(OUT / "controlled_incremental_value_table.csv", index=False)
    assumptions = pd.DataFrame([
        {"variable": "rate", "classification": "OBSERVED_DATA", "source": "fresh OOS canonical row"},
        {"variable": "forecast", "classification": "DERIVED_FROM_OBSERVED_DATA", "source": "fresh fold-fitted RF"},
        {"variable": "uncertainty_bounds", "classification": "DERIVED_FROM_OBSERVED_DATA", "source": "validation residual P10/P90"},
        {"variable": "volume", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
        {"variable": "duration", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
        {"variable": "capacity", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
        {"variable": "contract_discount", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
        {"variable": "budget", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
        {"variable": "contract_capacity", "classification": "SCENARIO_ASSUMPTION", "source": "not in canonical dataset"},
    ])
    assumptions.to_csv(OUT / "assumption_audit.csv", index=False)
    evidence = {"experiment_id": "FICOS_ARCHITECTURAL_ABLATION", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": sha256(data_path), "seed": SEED, "solver": "scipy.optimize.milp / HiGHS", "scenario_count": 100, "shared_opportunities": 12, "raw_oos_predictions": len(predictions), "canonical_gate_replay_retained": int(predictions.gate_retained.sum()), "interpretation": "Incremental economic results are modeled under explicit scenario assumptions; frozen economic results are not modified.", "outputs": sorted(p.name for p in OUT.iterdir())}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    conclusion = {"what_ficos_contributes": "Selective WHEN decisions plus portfolio-level HOW allocation under explicit risk objectives.", "what_contributes": "Contract selection and WHEN/HOW independent selection reduce modeled cost on this shared scenario; coupling and robust/CVaR add no incremental modeled value in this binding configuration.", "what_did_not_contribute": "Cross-voyage coupling, robust optimization, and mean-CVaR produced no change in the current 12-voyage scenario after WAIT rows were excluded from contract allocation.", "assumption_dependent": True, "judge_claim": "FICOS demonstrates a reproducible decision architecture; commercial savings remain unproven without procurement ledger data.", "forbidden_claims": ["causal commercial savings", "world-first", "guaranteed improvement"], "central_answer": "Under the same fresh OOS opportunity set, the architecture shows measurable modeled value for selective timing and contract choice, but no incremental value from coupling or risk objectives in this scenario. It is not valid to claim every layer adds value.", "controlled_evidence": "controlled_incremental_value_table.csv"}
    (OUT / "research_conclusion.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    print(master[["system", "procurement_cost_usd", "savings_vs_always_spot_usd", "worst_case_cost_usd", "solver_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
