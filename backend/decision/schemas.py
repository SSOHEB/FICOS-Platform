"""
FICOS — Decision Schemas
Output schemas for the final decision recommendation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult, CostBreakdown, PolicyEvaluation
)


@dataclass
class RecommendationOutput:
    """
    Final structured recommendation output from DecisionEngine.
    """
    recommendation_id: str
    timestamp: str
    cargo: CargoRequirement
    selected_vessel: VesselClass
    origin_port: Port
    dest_port: Port
    recommended_strategy: str  # e.g. "SPOT", "TIME_CHARTER", "REJECT"
    recommended_cost_usd: float
    recommended_cost_per_mt: float
    forecast: ForecastResult
    feasibility: FeasibilityResult
    risk: RiskResult
    cost_breakdown: CostBreakdown
    candidate_evaluations: List[PolicyEvaluation]
    explanation: Dict[str, Any]
    warnings_and_alerts: List[str] = field(default_factory=list)

    @property
    def decision(self) -> Any:
        return self.recommended_strategy

    @property
    def expected_cost_now(self) -> float:
        return self.recommended_cost_usd

    @property
    def expected_cost_wait(self) -> float:
        return self.recommended_cost_usd

    @property
    def risk_level(self) -> Any:
        return self.risk.overall_level

    @property
    def risk_result(self) -> Any:
        return self.risk




    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation output to JSON-serializable dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp,
            "cargo_id": self.cargo.cargo_id,
            "asset_type": self.cargo.asset_type,
            "quantity_mt": self.cargo.quantity_mt,
            "selected_vessel": self.selected_vessel.name,
            "origin_port": self.origin_port.name,
            "dest_port": self.dest_port.name,
            "recommended_strategy": self.recommended_strategy,
            "recommended_cost_usd": self.recommended_cost_usd,
            "recommended_cost_per_mt": self.recommended_cost_per_mt,
            "point_forecast_usd_ton": self.forecast.point_forecast,
            "uncertainty_p10_p90": [self.forecast.uncertainty.p10, self.forecast.uncertainty.p90],
            "is_feasible": self.feasibility.is_feasible,
            "overall_risk_score": self.risk.overall_risk_score,
            "explanation": self.explanation,
            "warnings_and_alerts": self.warnings_and_alerts
        }
