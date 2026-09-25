"""Real MILP planners using scipy.optimize.milp."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from .constraints import PlanningConstraints
from .voyage import Strategy, VoyageOpportunity

@dataclass
class PlanResult:
    selected: list[tuple[int, str]]
    objective_value: float
    solver_status: str
    message: str
    solve_time_seconds: float | None
    constraint_violations: list[str]
    def to_dict(self):
        return self.__dict__.copy()

class DeterministicMILPPlanner:
    strategies = tuple(strategy.value for strategy in Strategy)

    def _base_constraints(self, opportunities, constraints, variable_count):
        n, m = len(opportunities), len(self.strategies)
        rows, lower, upper = [], [], []
        for i in range(n):
            row = np.zeros(variable_count); row[i*m:(i+1)*m] = 1
            rows.append(row); lower.append(1); upper.append(1)
        costs = np.array([o.strategy_costs[s] for o in opportunities for s in self.strategies])
        contract = np.array([1 if s != "SPOT" else 0 for _ in opportunities for s in self.strategies])
        volume = np.array([o.voyage.volume_mt for o in opportunities for _ in self.strategies])
        for vec, bound in ((costs, constraints.budget_usd), (contract, constraints.max_contracts), (contract * volume, constraints.contract_capacity_mt)):
            row = np.zeros(variable_count); row[:n*m] = vec
            rows.append(row); lower.append(-np.inf); upper.append(bound)
        return rows, lower, upper

    def solve(self, opportunities, constraints):
        constraints.validate(opportunities)
        n, m = len(opportunities), len(self.strategies)
        costs = np.array([o.strategy_costs[s] for o in opportunities for s in self.strategies], dtype=float)
        rows, lower, upper = self._base_constraints(opportunities, constraints, n*m)
        result = milp(c=costs, integrality=np.ones(n*m), bounds=Bounds(0, 1), constraints=LinearConstraint(np.vstack(rows), lower, upper), options={"time_limit": 30.0})
        if not result.success:
            return PlanResult([], float("nan"), "FAILED", result.message, None, [result.message])
        selected = [(i, self.strategies[int(np.argmax(result.x[i*m:(i+1)*m]))]) for i in range(n)]
        return PlanResult(selected, float(result.fun), "OPTIMAL", result.message, None, [])

class RiskAwareMILPPlanner(DeterministicMILPPlanner):
    def solve_risk_aware(self, opportunities, scenario_costs, constraints, mode="MEAN_CVAR", alpha=0.9, risk_aversion=0.25):
        constraints.validate(opportunities)
        scenarios = np.asarray(scenario_costs, dtype=float)
        n, m, count = len(opportunities), len(self.strategies), len(scenarios)
        if scenarios.shape != (count, n, m):
            raise ValueError("scenario_costs must have shape (scenario, voyage, strategy)")
        base = n*m; tail = mode in {"CVaR", "MEAN_CVAR"}; eta = base if tail else None; zstart = base+1 if tail else base
        total = base + (1+count if tail else 0)
        c = np.zeros(total); c[:base] = scenarios.max(0).reshape(-1) if mode == "ROBUST" else scenarios.mean(0).reshape(-1)
        if tail:
            c[eta] = risk_aversion; c[zstart:] = risk_aversion / ((1-alpha)*count)
        rows, lower, upper = self._base_constraints(opportunities, constraints, total)
        if tail:
            for s, scenario in enumerate(scenarios):
                row = np.zeros(total); row[:base] = scenario.reshape(-1); row[eta] = -1; row[zstart+s] = -1
                rows.append(row); lower.append(-np.inf); upper.append(0)
        integrality = np.zeros(total); integrality[:base] = 1
        lb = np.zeros(total); ub = np.full(total, np.inf); ub[:base] = 1
        result = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=LinearConstraint(np.vstack(rows), lower, upper), options={"time_limit": 30.0})
        if not result.success:
            return PlanResult([], float("nan"), "FAILED", result.message, None, [result.message])
        selected = [(i, self.strategies[int(np.argmax(result.x[i*m:(i+1)*m]))]) for i in range(n)]
        return PlanResult(selected, float(result.fun), "OPTIMAL", result.message, None, [])
