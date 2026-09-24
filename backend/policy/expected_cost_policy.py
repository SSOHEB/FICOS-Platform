"""
FICOS — Expected-Cost Policy
Evaluates chartering strategies (SPOT, TIME_CHARTER, COA) under forecast uncertainty
and risk signals to determine expected total cost and policy recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult, CostBreakdown, PolicyEvaluation
)
from backend.cost.model import CostModel
from backend.cost.idle_assessment import IdleAssessmentEngine


class ExpectedCostPolicy:
    """
    Evaluates expected cost across strategies incorporating ML forecast uncertainty,
    risk parameters, and cost drivers defined in configs/decision_policy.yaml.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/decision_policy.yaml",
        cost_model: Optional[CostModel] = None,
        idle_engine: Optional[IdleAssessmentEngine] = None
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cost_model = cost_model or CostModel()
        self.idle_engine = idle_engine or IdleAssessmentEngine()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {
                "policy_weights": {"expected_cost": 0.5, "variance_penalty": 0.3, "risk_penalty": 0.2},
                "strategy_thresholds": {"max_acceptable_delay_days": 7.0, "risk_premium_multiplier": 1.2}
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def evaluate_strategy(
        self,
        strategy: str,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        forecast: ForecastResult,
        feasibility: FeasibilityResult,
        risk: RiskResult
    ) -> PolicyEvaluation:
        """
        Evaluates a single strategy (SPOT, TIME_CHARTER, or COA).
        """
        # Assess expected idle time
        idle_assessment = self.idle_engine.assess_idle_time(
            origin_port=origin_port,
            dest_port=dest_port,
            risk_result=risk
        )
        expected_wait_days = idle_assessment.expected_idle_days

        # Base rate according to strategy
        if strategy == "SPOT":
            base_rate = forecast.point_forecast
            uncertainty_spread = (forecast.uncertainty.p90 - forecast.uncertainty.p10)
        elif strategy == "TIME_CHARTER":
            # TC rate is locked in advance, lower variance but fixed charter commitments
            base_rate = forecast.point_forecast * 0.95  # baseline TC discount
            uncertainty_spread = 0.0
        else: # COA (Contract of Affreightment)
            base_rate = forecast.point_forecast * 0.98
            uncertainty_spread = (forecast.uncertainty.p90 - forecast.uncertainty.p10) * 0.5

        # Calculate cost breakdown
        cost_breakdown = self.cost_model.calculate_cost(
            cargo=cargo,
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            route=route,
            freight_rate_usd_ton=base_rate,
            strategy=strategy,
            expected_wait_days=expected_wait_days
        )

        # Variance / Risk Penalty
        weights = self.config.get("policy_weights", {})
        variance_penalty = uncertainty_spread * cargo.quantity_mt * weights.get("variance_penalty", 0.3)
        risk_penalty = (risk.overall_risk_score / 100.0) * cost_breakdown.total_cost_usd * weights.get("risk_penalty", 0.2)

        adjusted_expected_cost = cost_breakdown.total_cost_usd + variance_penalty + risk_penalty

        is_feasible = feasibility.is_feasible and (risk.overall_risk_score <= 85.0)

        reasoning = (
            f"{strategy}: Base cost ${cost_breakdown.total_cost_usd:,.0f} "
            f"(Rate ${base_rate:.2f}/MT, Wait {expected_wait_days:.1f}d). "
            f"Risk penalty ${risk_penalty:,.0f}, Variance penalty ${variance_penalty:,.0f}."
        )

        return PolicyEvaluation(
            strategy_name=strategy,
            expected_cost_usd=round(cost_breakdown.total_cost_usd, 2),
            adjusted_expected_cost_usd=round(adjusted_expected_cost, 2),
            cost_per_mt=round(cost_breakdown.cost_per_mt, 2),
            cost_breakdown=cost_breakdown,
            risk_score=risk.overall_risk_score,
            is_feasible=is_feasible,
            policy_reasoning=reasoning
        )
