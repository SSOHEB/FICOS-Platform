"""
FICOS — Decision Engine v2
Orchestrates Forecast Service, Feasibility Engine, Risk Engine, Cost Model,
and Expected Cost Policy to produce actionable, risk-gated recommendations.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult
)
from ml.forecasting.service import ForecastService
from backend.operational.feasibility_engine import FeasibilityEngine
from backend.operational.port_repository import PortRepository
from backend.operational.vessel_repository import VesselRepository
from backend.risk.engine import RiskEngine
from backend.policy.expected_cost_policy import ExpectedCostPolicy
from backend.decision.schemas import RecommendationOutput
from backend.decision.explanation import ExplanationGenerator


class DecisionEngine:
    """
    Main decision orchestrator. Integrates all platform services to recommend
    optimal freight chartering strategies under physical, operational, and risk constraints.
    """

    def __init__(
        self,
        forecast_service: Optional[ForecastService] = None,
        feasibility_engine: Optional[FeasibilityEngine] = None,
        risk_engine: Optional[RiskEngine] = None,
        cost_policy: Optional[ExpectedCostPolicy] = None,
        explanation_generator: Optional[ExplanationGenerator] = None
    ):
        self.forecast_service = forecast_service or ForecastService()
        self.feasibility_engine = feasibility_engine or FeasibilityEngine()
        self.risk_engine = risk_engine or RiskEngine()
        self.cost_policy = cost_policy or ExpectedCostPolicy()
        self.explanation_generator = explanation_generator or ExplanationGenerator()

    def decide(
        self,
        cargo: CargoRequirement,
        vessel: Optional[VesselClass] = None,
        origin_port: Optional[Port] = None,
        dest_port: Optional[Port] = None,
        route: Optional[Route] = None,
        current_rate: float = 25.0,
        **kwargs
    ) -> RecommendationOutput:
        """Alias for evaluate() for scenario testing compatibility."""
        vessel = vessel or VesselClass(code="PANA", display_name="Panamax")
        origin_port = origin_port or Port(code="TUBARAO", display_name="Tubarao", key="tubarao", country="Brazil", state="", constraints=None)
        dest_port = dest_port or Port(code="QINGDAO", display_name="Qingdao", key="qingdao", country="China", state="", constraints=None)
        route = route or Route(origin="TUBARAO", destination="QINGDAO")
        return self.evaluate(cargo=cargo, vessel=vessel, origin_port=origin_port, dest_port=dest_port, route=route)

    def evaluate(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        candidate_strategies: Optional[List[str]] = None
    ) -> RecommendationOutput:

        """
        Evaluates freight requirement and generates structured recommendation.

        Args:
            cargo: Cargo requirement details
            vessel: Vessel class details
            origin_port: Origin port details
            dest_port: Destination port details
            route: Transit route details
            candidate_strategies: List of charter strategies to consider (default: ['SPOT', 'TIME_CHARTER', 'COA'])
        """
        if candidate_strategies is None:
            candidate_strategies = ["SPOT", "TIME_CHARTER", "COA"]

        recommendation_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.utcnow().isoformat()

        # Step 1: Forecast Service
        forecast = self.forecast_service.get_forecast(
            asset_type=cargo.asset_type,
            horizon_days=cargo.laycan_days
        )

        # Step 2: Feasibility Engine (Physical Gate)
        feasibility = self.feasibility_engine.check_feasibility(
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            cargo=cargo
        )

        # Step 3: Risk Engine (Dataset C & Environmental Signals)
        risk = self.risk_engine.evaluate_risk(
            origin_port=origin_port.name,
            dest_port=dest_port.name,
            asset_type=cargo.asset_type
        )

        # Step 4: Policy Evaluation across Candidate Strategies
        evaluations = []
        for strat in candidate_strategies:
            eval_result = self.cost_policy.evaluate_strategy(
                strategy=strat,
                cargo=cargo,
                vessel=vessel,
                origin_port=origin_port,
                dest_port=dest_port,
                route=route,
                forecast=forecast,
                feasibility=feasibility,
                risk=risk
            )
            evaluations.append(eval_result)

        # Step 5: Selection & Action Generation
        warnings_and_alerts: List[str] = []
        if not feasibility.is_feasible:
            warnings_and_alerts.append("CRITICAL: Physical feasibility checks failed. Route/Vessel combination rejected.")
            recommended_strategy = "REJECT"
            recommended_cost_usd = 0.0
            recommended_cost_per_mt = 0.0
            best_breakdown = evaluations[0].cost_breakdown
        else:
            # Filter feasible candidate evaluations
            feasible_evals = [e for e in evaluations if e.is_feasible]
            if not feasible_evals:
                warnings_and_alerts.append("WARNING: All candidate strategies exceed risk or cost thresholds.")
                best_eval = min(evaluations, key=lambda x: x.adjusted_expected_cost_usd)
            else:
                best_eval = min(feasible_evals, key=lambda x: x.adjusted_expected_cost_usd)

            recommended_strategy = best_eval.strategy_name
            recommended_cost_usd = best_eval.expected_cost_usd
            recommended_cost_per_mt = best_eval.cost_per_mt
            best_breakdown = best_eval.cost_breakdown

        if risk.overall_risk_score >= 70.0:
            warnings_and_alerts.append(f"RISK ALERT: Elevated overall risk score ({risk.overall_risk_score:.1f}/100). Monitoring advised.")

        for alert in risk.risk_alerts:
            if alert.severity in ["HIGH", "CRITICAL"]:
                warnings_and_alerts.append(f"ALERT [{alert.category}]: {alert.message}")

        # Step 6: Generate Audit Explanation
        explanation = self.explanation_generator.generate_explanation(
            recommended_strategy=recommended_strategy,
            forecast=forecast,
            feasibility=feasibility,
            risk=risk,
            candidate_evaluations=evaluations
        )

        return RecommendationOutput(
            recommendation_id=recommendation_id,
            timestamp=timestamp,
            cargo=cargo,
            selected_vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            recommended_strategy=recommended_strategy,
            recommended_cost_usd=recommended_cost_usd,
            recommended_cost_per_mt=recommended_cost_per_mt,
            forecast=forecast,
            feasibility=feasibility,
            risk=risk,
            cost_breakdown=best_breakdown,
            candidate_evaluations=evaluations,
            explanation=explanation,
            warnings_and_alerts=warnings_and_alerts
        )
