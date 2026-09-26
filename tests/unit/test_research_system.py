import numpy as np
import pandas as pd

from evaluation.locked_oos import assert_locked_oos_isolation
from ml.forecasting.models import PersistenceModel
from ml.planning import PlanningConstraints, Strategy, Voyage, VoyageOpportunity, assess_coupling_activation, assess_risk_activation
from ml.planning.planner import DeterministicMILPPlanner, RiskAwareMILPPlanner

def opportunities(n=3):
    result = []
    for i in range(n):
        voyage = Voyage(f"V{i}", "PANAMAX", "R", "O", "D", "2026-01-01", "2026-01-02", 20, 10, 12, "2025-12-01")
        result.append(VoyageOpportunity(voyage, 100, 105, 90, 120, {s.value: 1000 * (1 - 0.02 * j) for j, s in enumerate(Strategy)}, "RISING", "synthetic", ("SCENARIO_ASSUMPTION",)))
    return result

def test_real_milp_and_risk_milp_solve_with_one_strategy_per_voyage():
    ops = opportunities(); constraints = PlanningConstraints(budget_usd=4000, contract_capacity_mt=100)
    assert DeterministicMILPPlanner().solve(ops, constraints).solver_status == "OPTIMAL"
    scenarios = np.repeat(np.array([[[1000, 980, 960, 940]] * 3], dtype=float), 5, axis=0)
    result = RiskAwareMILPPlanner().solve_risk_aware(ops, scenarios, constraints, mode="MEAN_CVAR")
    assert result.solver_status == "OPTIMAL" and len(result.selected) == 3

def test_persistence_model_is_explicit_baseline():
    model = PersistenceModel().fit(pd.DataFrame({"x": [1, 2]}), [0, 1])
    assert np.all(model.predict([[1], [2]]) == 0)
    assert model.metadata()["research_status"] == "BASELINE"

def test_locked_oos_guard_accepts_disjoint_dates():
    assert_locked_oos_isolation(["2024-01-01"], ["2025-01-01"])

def test_coupling_activation_is_triggered_by_binding_capacity():
    decision = assess_coupling_activation(
        independent_contracts=4,
        independent_contract_volume_mt=300_000,
        independent_cost_usd=100,
        constraints={"contract_capacity_mt": 75_000, "budget_usd": 200, "max_contracts": 10},
    )
    assert decision.active
    assert "contract_capacity_at_or_above_threshold" in decision.reasons

def test_coupling_activation_stays_off_for_slack_portfolio():
    decision = assess_coupling_activation(
        independent_contracts=4,
        independent_contract_volume_mt=300_000,
        independent_cost_usd=100,
        constraints={"contract_capacity_mt": 600_000, "budget_usd": 200, "max_contracts": 8},
    )
    assert not decision.active
    assert decision.reasons == ()

def test_risk_activation_requires_preregistered_dispersion_threshold():
    dormant = assess_risk_activation([100, 101, 102, 103, 104])
    active = assess_risk_activation([100, 100, 100, 100, 110])
    assert not dormant.active
    assert active.active
