"""Portfolio and decision explanation representations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ContractPortfolio:
    strategy_by_voyage: dict[str, str]
    objective_value_usd: float
    expected_savings_usd: float
    worst_case_savings_usd: float
    cvar_loss_usd: float | None
    solver_status: str
    rationale_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"strategy_by_voyage": self.strategy_by_voyage, "objective_value_usd": self.objective_value_usd, "expected_savings_usd": self.expected_savings_usd, "worst_case_savings_usd": self.worst_case_savings_usd, "cvar_loss_usd": self.cvar_loss_usd, "solver_status": self.solver_status, "rationale_json": self.rationale_json}

def build_portfolio(opportunities, selected, objective_value, status, cvar=None):
    strategy_by_voyage = {opportunities[i].voyage.voyage_id: strategy for i, strategy in selected}
    savings = [o.strategy_costs["SPOT"] - o.strategy_costs[strategy] for o, (_, strategy) in zip(opportunities, selected)]
    return ContractPortfolio(strategy_by_voyage, float(objective_value), float(sum(savings)), float(sum(savings)), cvar, status, {"why": "Lowest modeled cost subject to portfolio constraints.", "why_not_alternatives": "Alternatives were dominated under stated assumptions or constrained by portfolio limits.", "assumptions": ["Voyage, capacity, budget, and contract discount inputs are scenario assumptions."]})
