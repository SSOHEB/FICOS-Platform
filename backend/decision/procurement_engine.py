"""
FICOS — Procurement Decision Engine (SIH 2026 SIH26006)
===========================================================
Orchestrates:
  - Forecast Provenance & Validation Metadata
  - Timing Decision (WHEN to commit: NOW / WAIT / FLEXIBLE)
  - Contract Strategy Selection (HOW to contract: SPOT / TIME_CHARTER / COA / FLEXIBLE_INDEX)
  - Contract Strategy Comparison (4-way evaluation)
  - Simple Multi-Voyage Planner (N-voyage term chartering)
  - Explainable Recommendation Rationale

Maintains 100% backward compatibility with existing APIs.
Does NOT run ML experiments, modify model promotions, or alter decision thresholds.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult
)
from ml.forecasting.service import ForecastService
from backend.operational.feasibility_engine import FeasibilityEngine
from backend.operational.port_repository import PortRepository
from backend.operational.vessel_repository import VesselRepository
from backend.risk.engine import RiskEngine
from backend.cost.model import CostModel
from backend.policy.expected_cost_policy import ExpectedCostPolicy
from backend.decision.contract_comparison import ContractStrategyComparator, ContractStrategyDetail
from backend.decision.multi_voyage_planner import MultiVoyagePlanner, MultiVoyagePlanOutput
from backend.decision.explanation import ExplanationGenerator


@dataclass
class ProcurementDecisionOutput:
    """Comprehensive procurement recommendation output."""
    decision_id: str
    timestamp: str
    
    # Cargo & Route Specification
    cargo: CargoRequirement
    vessel: VesselClass
    origin_port: Port
    dest_port: Port
    route: Route
    num_voyages: int
    
    # Timing & Strategy Recommendations (Separated WHEN vs HOW)
    recommended_timing: str                      # "NOW", "WAIT", "FLEXIBLE"
    recommended_contract_strategy: str          # "SPOT", "TIME_CHARTER", "COA", "FLEXIBLE_INDEX"
    timing_rationale: str                        # Why this entry timing was chosen
    strategy_rationale: str                      # Why this contract structure was chosen
    
    # Financial & Cost Metrics
    single_voyage_expected_cost_usd: float
    single_voyage_cost_per_mt: float
    multi_voyage_plan: Optional[MultiVoyagePlanOutput]
    
    # Provenance, Gating & Risk
    forecast_provenance: Dict[str, Any]
    feasibility: FeasibilityResult
    risk: RiskResult
    strategy_comparison: List[ContractStrategyDetail]
    
    explanation: Dict[str, Any]
    warnings_and_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "cargo_id": self.cargo.cargo_id,
            "asset_type": self.cargo.asset_type,
            "quantity_mt": self.cargo.quantity_mt,
            "num_voyages": self.num_voyages,
            "vessel_class": self.vessel.name,
            "origin_port": self.origin_port.name,
            "dest_port": self.dest_port.name,
            "recommended_timing": self.recommended_timing,
            "recommended_contract_strategy": self.recommended_contract_strategy,
            "timing_rationale": self.timing_rationale,
            "strategy_rationale": self.strategy_rationale,
            "single_voyage_expected_cost_usd": self.single_voyage_expected_cost_usd,
            "single_voyage_cost_per_mt": self.single_voyage_cost_per_mt,
            "multi_voyage_plan": self.multi_voyage_plan.__dict__ if self.multi_voyage_plan else None,
            "forecast_provenance": self.forecast_provenance,
            "is_feasible": self.feasibility.is_feasible,
            "overall_risk_score": self.risk.overall_risk_score,
            "strategy_comparison": [s.__dict__ for s in self.strategy_comparison],
            "explanation": self.explanation,
            "warnings_and_alerts": self.warnings_and_alerts
        }


class ProcurementDecisionEngine:
    """
    Main SIH 2026 Procurement Decision Orchestrator.
    Extends the decision layer to support optimal market entry timing,
    multi-voyage contract planning, and transparent explainability.
    """

    def __init__(
        self,
        forecast_service: Optional[ForecastService] = None,
        feasibility_engine: Optional[FeasibilityEngine] = None,
        risk_engine: Optional[RiskEngine] = None,
        cost_model: Optional[CostModel] = None,
        cost_policy: Optional[ExpectedCostPolicy] = None
    ):
        self.forecast_service = forecast_service or ForecastService()
        self.feasibility_engine = feasibility_engine or FeasibilityEngine()
        self.risk_engine = risk_engine or RiskEngine()
        self.cost_model = cost_model or CostModel()
        self.cost_policy = cost_policy or ExpectedCostPolicy(cost_model=self.cost_model)
        self.comparator = ContractStrategyComparator(cost_policy=self.cost_policy, cost_model=self.cost_model)
        self.multi_voyage_planner = MultiVoyagePlanner(cost_model=self.cost_model)
        self.explanation_generator = ExplanationGenerator()

    def evaluate_procurement(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        current_rate: float = 25.0,
        num_voyages: int = 1,
        candidate_strategies: Optional[List[str]] = None
    ) -> ProcurementDecisionOutput:
        """
        Evaluates freight requirement and produces a complete ProcurementDecisionOutput.
        """
        decision_id = f"PROC-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 1. Forecast Provenance
        forecast = self.forecast_service.get_forecast(
            asset_type=cargo.asset_type,
            horizon_days=cargo.laycan_days or 14,
            current_rate=current_rate
        )
        forecast_prov = forecast.to_provenance_dict()

        # 2. Feasibility Gate
        feasibility = self.feasibility_engine.check_feasibility(
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            cargo=cargo
        )

        # 3. Risk Engine
        risk = self.risk_engine.evaluate_risk(
            origin_port=origin_port.name,
            dest_port=dest_port.name,
            asset_type=cargo.asset_type
        )

        # 4. Strategy Comparison Across 4 Structures
        strategy_comparison = self.comparator.compare_all(
            cargo=cargo,
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            route=route,
            forecast=forecast,
            feasibility=feasibility,
            risk=risk
        )

        # 5. Timing Decision (WHEN) vs Contract Strategy (HOW)
        exp_delta = forecast.expected_delta
        tau = 0.01 * current_rate
        unc_spread = forecast.uncertainty_width

        if not feasibility.is_feasible:
            timing = "FLEXIBLE"
            contract_strat = "FLEXIBLE_INDEX"
            t_rationale = "Feasibility gate failed. Abstaining from firm commitment."
            s_rationale = "Physical port/vessel constraints violated. Capital-preserving flexible structure enforced."
        elif not forecast.is_promoted or forecast.confidence.value == "LOW":
            timing = "FLEXIBLE"
            contract_strat = "FLEXIBLE_INDEX"
            t_rationale = "Market uncertainty is elevated or model signal is unpromoted. Preserving market optionality."
            s_rationale = "Avoiding irreversible fixed-rate contract during high uncertainty. Floating index-linked contract selected with collar cap."
        elif exp_delta > tau:
            timing = "NOW"
            contract_strat = "TIME_CHARTER" if num_voyages > 1 else "SPOT"
            t_rationale = f"Promoted forecast signals rising freight momentum (+{forecast.expected_pct_change:.1f}% expected). Lock in forward coverage."
            s_rationale = f"Selected '{contract_strat}' to lock in favorable rates before projected rate increase."
        elif exp_delta < -tau:
            timing = "WAIT"
            contract_strat = "COA" if num_voyages > 1 else "SPOT"
            t_rationale = f"Promoted forecast signals softening freight momentum ({forecast.expected_pct_change:.1f}% expected). Delay fixture to capture lower spot rate."
            s_rationale = "Delaying fixture execution allows capturing expected lower spot freight rates."
        else:
            timing = "FLEXIBLE"
            contract_strat = "FLEXIBLE_INDEX"
            t_rationale = "Promoted coverage is available, but model has no strong directional commitment at this horizon."
            s_rationale = "Capital-preserving FLEXIBLE index-linked charter selected."

        # Single voyage costs
        best_strat_detail = next((s for s in strategy_comparison if s.strategy_name == contract_strat), strategy_comparison[0])
        single_cost_usd = best_strat_detail.expected_cost_usd
        single_cost_pmt = best_strat_detail.cost_per_mt

        # 6. Multi-Voyage Plan (if num_voyages >= 1)
        multi_plan: Optional[MultiVoyagePlanOutput] = None
        if num_voyages >= 1:
            multi_plan = self.multi_voyage_planner.plan_program(
                num_voyages=num_voyages,
                cargo=cargo,
                vessel=vessel,
                origin_port=origin_port,
                dest_port=dest_port,
                route=route,
                forecast=forecast,
                feasibility=feasibility,
                risk=risk
            )

        # 7. Warnings and Alerts
        warnings_list: List[str] = []
        if not feasibility.is_feasible:
            warnings_list.append("CRITICAL: Physical port/vessel feasibility checks failed.")
        if risk.overall_risk_score >= 70.0:
            warnings_list.append(f"RISK ALERT: Elevated risk score ({risk.overall_risk_score:.1f}/100). Monitoring advised.")

        # 8. Explainable Recommendation
        explanation_dict = {
            "procurement_objective": "SIH 2026 Objective: Transition from reactive single spot contracts to proactive multi-voyage planning",
            "recommended_timing": timing,
            "recommended_contract_strategy": contract_strat,
            "vessel_class": vessel.name,
            "origin_port": origin_port.name,
            "dest_port": dest_port.name,
            "timing_rationale": t_rationale,
            "strategy_rationale": s_rationale,
            "forecast_summary": f"Point forecast ${forecast.point_forecast:.2f}/MT (P10=${forecast.p10:.2f}, P90=${forecast.p90:.2f})",
            "forecast_provenance": forecast_prov,
            "gating_status": {
                "physical_feasibility": feasibility.is_feasible,
                "risk_within_threshold": risk.overall_risk_score <= 85.0
            }
        }

        return ProcurementDecisionOutput(
            decision_id=decision_id,
            timestamp=timestamp,
            cargo=cargo,
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            route=route,
            num_voyages=num_voyages,
            recommended_timing=timing,
            recommended_contract_strategy=contract_strat,
            timing_rationale=t_rationale,
            strategy_rationale=s_rationale,
            single_voyage_expected_cost_usd=single_cost_usd,
            single_voyage_cost_per_mt=single_cost_pmt,
            multi_voyage_plan=multi_plan,
            forecast_provenance=forecast_prov,
            feasibility=feasibility,
            risk=risk,
            strategy_comparison=strategy_comparison,
            explanation=explanation_dict,
            warnings_and_alerts=warnings_list
        )
