"""Pre-registered activation rules for dormant portfolio layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ActivationDecision:
    layer: str
    active: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "active": self.active,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
        }


def assess_coupling_activation(
    independent_contracts: int,
    independent_contract_volume_mt: float,
    independent_cost_usd: float,
    constraints: Mapping[str, float],
    *,
    capacity_threshold: float = 1.0,
    budget_threshold: float = 1.0,
    contract_count_threshold: float = 1.0,
) -> ActivationDecision:
    """Activate coupling when an independent plan reaches a shared limit."""
    capacity_utilization = independent_contract_volume_mt / float(constraints["contract_capacity_mt"])
    budget_utilization = independent_cost_usd / float(constraints["budget_usd"])
    contract_utilization = independent_contracts / float(constraints["max_contracts"])
    metrics = {
        "capacity_utilization": float(capacity_utilization),
        "budget_utilization": float(budget_utilization),
        "contract_count_utilization": float(contract_utilization),
    }
    reasons = []
    if capacity_utilization >= capacity_threshold:
        reasons.append("contract_capacity_at_or_above_threshold")
    if budget_utilization >= budget_threshold:
        reasons.append("budget_at_or_above_threshold")
    if contract_utilization >= contract_count_threshold:
        reasons.append("contract_count_at_or_above_threshold")
    return ActivationDecision(
        layer="coupling",
        active=bool(reasons),
        reasons=tuple(reasons),
        metrics=metrics,
        thresholds={
            "capacity_utilization_threshold": capacity_threshold,
            "budget_utilization_threshold": budget_threshold,
            "contract_count_utilization_threshold": contract_count_threshold,
        },
    )


def assess_risk_activation(
    scenario_costs: Iterable[float],
    *,
    p90_over_median_threshold: float = 0.05,
) -> ActivationDecision:
    """Activate risk-aware optimization when scenario downside dispersion is material."""
    values = np.asarray(list(scenario_costs), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scenario_costs must be a non-empty finite sequence")
    median = float(np.median(values))
    p90 = float(np.percentile(values, 90))
    relative_dispersion = 0.0 if median == 0 else (p90 - median) / abs(median)
    active = relative_dispersion >= p90_over_median_threshold
    return ActivationDecision(
        layer="risk",
        active=active,
        reasons=("scenario_p90_dispersion_at_or_above_threshold",) if active else (),
        metrics={"scenario_median_cost_usd": median, "scenario_p90_cost_usd": p90, "p90_over_median": relative_dispersion},
        thresholds={"scenario_p90_over_median_threshold": p90_over_median_threshold},
    )

