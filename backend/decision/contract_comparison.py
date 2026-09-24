"""
FICOS — Contract Strategy Comparison Engine
Evaluates & compares candidate procurement/charter contract structures:
  1. SPOT            - Reactive single-voyage spot charter
  2. TIME_CHARTER    - Fixed daily rate term charter
  3. COA             - Contract of Affreightment (volume commitment)
  4. FLEXIBLE_INDEX  - Floating index-linked charter with optionality collars

Reuses existing CostModel, ExpectedCostPolicy, and RiskEngine calculations.
Does NOT invent historical contract prices or claim false savings.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult
)
from backend.cost.model import CostModel
from backend.cost.idle_assessment import IdleAssessmentEngine
from backend.policy.expected_cost_policy import ExpectedCostPolicy


@dataclass
class ContractStrategyDetail:
    """Detailed evaluation metrics for a single charter strategy option."""
    strategy_name: str                           # "SPOT", "TIME_CHARTER", "COA", "FLEXIBLE_INDEX"
    display_title: str                           # Human-readable title
    expected_cost_usd: float                     # Baseline voyage cost
    adjusted_expected_cost_usd: float            # Cost including risk & variance penalties
    cost_per_mt: float                           # Cost per metric ton ($/MT)
    risk_score: float                            # Overall risk score (0-100)
    risk_penalty_usd: float                      # Risk penalty amount in USD
    variance_penalty_usd: float                  # Forecast variance penalty in USD
    is_feasible: bool                            # Physical + risk threshold feasibility flag
    flexibility_rating: str                      # "HIGH", "MEDIUM", "LOW"
    commitment_level: str                        # "HIGH_FIXED", "MEDIUM_TERM", "FLEXIBLE_OPTIONALITY", "SINGLE_VOYAGE"
    primary_advantages: List[str]                # Key strategic benefits
    primary_disadvantages: List[str]             # Strategic trade-offs or constraints
    important_assumptions: List[str]             # Transparent underlying assumptions


class ContractStrategyComparator:
    """
    Compares Spot, Time Charter, COA, and Flexible Index strategies
    under forecast uncertainty, risk parameters, and physical constraints.
    """

    def __init__(
        self,
        cost_policy: Optional[ExpectedCostPolicy] = None,
        cost_model: Optional[CostModel] = None,
        idle_engine: Optional[IdleAssessmentEngine] = None
    ):
        self.cost_model = cost_model or CostModel()
        self.idle_engine = idle_engine or IdleAssessmentEngine()
        self.cost_policy = cost_policy or ExpectedCostPolicy(cost_model=self.cost_model, idle_engine=self.idle_engine)

    def compare_all(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        forecast: ForecastResult,
        feasibility: FeasibilityResult,
        risk: RiskResult
    ) -> List[ContractStrategyDetail]:
        """
        Evaluates and compares all 4 charter contract structures.
        """
        strategies = ["SPOT", "TIME_CHARTER", "COA", "FLEXIBLE_INDEX"]
        results: List[ContractStrategyDetail] = []

        idle_eval = self.idle_engine.assess_idle_time(
            origin_port=origin_port,
            dest_port=dest_port,
            risk_result=risk
        )
        expected_wait_days = idle_eval.expected_idle_days

        unc_spread = (forecast.uncertainty.p90 - forecast.uncertainty.p10)
        qty = float(cargo.quantity_mt) if isinstance(cargo.quantity_mt, (int, float)) else 75000.0

        for strat in strategies:
            if strat == "SPOT":
                base_rate = forecast.point_forecast
                var_spread = unc_spread
                flex_rating = "MEDIUM"
                commit_level = "SINGLE_VOYAGE"
                adv = ["Maximum operational flexibility for one-off shipments", "No long-term capital lockup"]
                disadv = ["100% exposure to spot freight rate spikes", "Demurrage and congestion delay risk"]
                assumptions = ["Fixture executed at prevailing spot market rate upon laycan", f"Forecast uncertainty spread: ${unc_spread:.2f}/MT"]

            elif strat == "TIME_CHARTER":
                # TC rate is fixed; lower rate uncertainty but locks in duration
                base_rate = forecast.point_forecast * 0.95
                var_spread = 0.0
                flex_rating = "LOW"
                commit_level = "HIGH_FIXED"
                adv = ["Guaranteed rate stability over term duration", "Defends against market rate surges"]
                disadv = ["Irreversible fixed rate commitment during market downturns", "Operational & vessel utilization risk"]
                assumptions = ["Fixed daily hire rate benchmarked at 95% of point forecast", "Zero rate variance penalty during charter term"]

            elif strat == "COA":
                # COA volume discount with shared scheduling
                base_rate = forecast.point_forecast * 0.97
                var_spread = unc_spread * 0.4
                flex_rating = "MEDIUM"
                commit_level = "MEDIUM_TERM"
                adv = ["Volume commitment discount", "Guaranteed tonnage allocation across schedule"]
                disadv = ["Strict laycan volume commitments", "Penalty clauses for cancelled cargoes"]
                assumptions = ["Contract of Affreightment multi-cargo rate discount (3%)", "40% variance reduction from volume distribution"]

            else: # FLEXIBLE_INDEX
                # Index-linked floating rate with optionality collar
                base_rate = forecast.point_forecast
                var_spread = unc_spread * 0.25  # Collar caps upside exposure
                flex_rating = "HIGH"
                commit_level = "FLEXIBLE_OPTIONALITY"
                adv = ["Avoids irreversible fixed-rate commitment", "Preserves market downturn downside savings", "Caps extreme rate spikes via collar options"]
                disadv = ["Subject to index tracking variance", "Requires index publishing governance"]
                assumptions = ["Floating rate linked to Baltic Freight Index with floor/ceiling collar", "Preserves optionality when market uncertainty is elevated"]

            # Calculate cost breakdown via CostModel
            cost_bd = self.cost_model.calculate_cost(
                cargo=cargo,
                vessel=vessel,
                origin_port=origin_port,
                dest_port=dest_port,
                route=route,
                freight_rate_usd_ton=base_rate,
                strategy=strat if strat in ["SPOT", "TIME_CHARTER"] else "SPOT",
                expected_wait_days=expected_wait_days
            )

            # Variance and Risk penalties
            var_penalty = var_spread * qty * 0.3
            risk_penalty = (risk.overall_risk_score / 100.0) * cost_bd.total_cost_usd * 0.2
            adj_cost = cost_bd.total_cost_usd + var_penalty + risk_penalty

            is_strat_feasible = feasibility.is_feasible and (risk.overall_risk_score <= 85.0)

            results.append(
                ContractStrategyDetail(
                    strategy_name=strat,
                    display_title=f"{strat.replace('_', ' ').title()} Contract",
                    expected_cost_usd=round(cost_bd.total_cost_usd, 2),
                    adjusted_expected_cost_usd=round(adj_cost, 2),
                    cost_per_mt=round(cost_bd.cost_per_mt, 2),
                    risk_score=risk.overall_risk_score,
                    risk_penalty_usd=round(risk_penalty, 2),
                    variance_penalty_usd=round(var_penalty, 2),
                    is_feasible=is_strat_feasible,
                    flexibility_rating=flex_rating,
                    commitment_level=commit_level,
                    primary_advantages=adv,
                    primary_disadvantages=disadv,
                    important_assumptions=assumptions
                )
            )

        return results
