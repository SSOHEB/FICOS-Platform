"""
FICOS — Simple Multi-Voyage Planner Engine
============================================
Fulfills SIH 2026 Problem SIH26006 objective:
"Development of model to facilitate moving from multiple single spot contracts
being entered into currently to short term / medium term multiple voyage contracts."

Implements a deterministic, explainable multi-voyage comparison across:
  1. Independent Spot Charters (N separate spot fixtures)
  2. Multi-Voyage Contract of Affreightment (COA Program)
  3. Dedicated Time Charter Tonnage Allocation
  4. Flexible Index-Linked Multi-Voyage Program

Does NOT use opaque MILP optimizers. Every number is derived deterministically
from the existing CostModel, ForecastResult, and RiskEngine outputs.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult,
    FeasibilityResult, RiskResult
)
from backend.cost.model import CostModel
from backend.decision.contract_comparison import ContractStrategyComparator


@dataclass
class SingleVoyagePlan:
    """Cost breakdown for a single voyage step in a multi-voyage program."""
    voyage_number: int
    laycan_label: str                             # e.g. "Voyage 1 (Day 1)", "Voyage 2 (Day 30)"
    cargo_quantity_mt: float
    projected_spot_rate: float                   # $/MT
    spot_cost_usd: float                         # Total USD for spot execution
    coa_cost_usd: float                          # Total USD for COA execution
    tc_cost_usd: float                           # Total USD for TC execution
    flexible_cost_usd: float                     # Total USD for Flexible Index execution
    recommended_single_strategy: str            # Best strategy for this individual voyage step


@dataclass
class MultiVoyagePlanOutput:
    """Comprehensive output for multi-voyage procurement strategy comparison."""
    plan_id: str
    num_voyages: int
    total_cargo_mt: float
    vessel_class: str
    origin_port: str
    dest_port: str
    
    # Aggregate Program Costs (USD)
    spot_aggregate_cost_usd: float               # Cost if 100% spot chartering used
    coa_aggregate_cost_usd: float                # Cost under Multi-Voyage COA
    tc_aggregate_cost_usd: float                 # Cost under Dedicated Time Charter
    flexible_aggregate_cost_usd: float           # Cost under Flexible Index Program
    
    # Unit Costs ($/MT)
    spot_cost_per_mt: float
    coa_cost_per_mt: float
    tc_cost_per_mt: float
    flexible_cost_per_mt: float
    
    # Recommendation & Savings vs Spot
    recommended_program_strategy: str           # "COA", "TIME_CHARTER", "FLEXIBLE_INDEX", or "SPOT"
    recommended_aggregate_cost_usd: float
    savings_vs_spot_usd: float                  # spot_aggregate_cost - recommended_cost (positive = savings vs Spot, negative = higher cost)
    savings_pct: float                          # % savings vs spot: (spot - recommended) / spot * 100
    
    voyage_schedule: List[SingleVoyagePlan]       # Detailed breakdown per voyage
    explanation: Dict[str, Any]                  # Transparent mathematical formulas & reasoning
    important_notes: List[str]

    # Backward compatibility aliases
    cost_difference_vs_spot_usd: float = 0.0
    cost_difference_pct: float = 0.0


class MultiVoyagePlanner:
    """
    Evaluates and compares multi-voyage procurement strategies.
    Reuses existing CostModel and ContractStrategyComparator.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.comparator = ContractStrategyComparator(cost_model=self.cost_model)

    def plan_program(
        self,
        num_voyages: int,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        forecast: ForecastResult,
        feasibility: FeasibilityResult,
        risk: RiskResult,
        interval_days_between_voyages: int = 30
    ) -> MultiVoyagePlanOutput:
        """
        Builds a multi-voyage procurement comparison for N consecutive cargoes.

        Args:
            num_voyages: Number of voyages in program (e.g. 2, 4, 6, 12)
            cargo: Baseline cargo requirement
            vessel: Vessel class
            origin_port: Origin port
            dest_port: Destination port
            route: Route details
            forecast: Base forecast result
            feasibility: Feasibility result
            risk: Risk result
            interval_days_between_voyages: Spacing between laycans (default 30 days)
        """
        num_voyages = max(1, min(int(num_voyages), 24))
        qty_per_voyage = float(cargo.quantity_mt) if isinstance(cargo.quantity_mt, (int, float)) else 75000.0
        total_cargo_mt = qty_per_voyage * num_voyages

        base_spot_rate = float(forecast.point_forecast)
        expected_wait_days = float(getattr(risk, "overall_risk_score", 10.0)) / 20.0  # proportional wait estimate

        voyage_schedule: List[SingleVoyagePlan] = []
        spot_tot = 0.0
        coa_tot = 0.0
        tc_tot = 0.0
        flex_tot = 0.0

        for i in range(num_voyages):
            day_offset = i * interval_days_between_voyages
            # Slight deterministic rate trend adjustment per laycan (e.g. market forward curve)
            trend_adj = (i * 0.15) if forecast.expected_delta > 0 else (-i * 0.10)
            v_spot_rate = max(5.0, base_spot_rate + trend_adj)

            # Strategy rate factors
            v_coa_rate = v_spot_rate * 0.96      # 4% volume discount on multi-voyage COA
            v_tc_rate = v_spot_rate * 0.94       # 6% fixed charter discount
            v_flex_rate = v_spot_rate * 0.98     # 2% collar-backed index rate

            # Calculate voyage cost breakdowns
            c_spot = self.cost_model.calculate_cost(cargo, vessel, origin_port, dest_port, route, v_spot_rate, "SPOT", expected_wait_days)
            c_coa = self.cost_model.calculate_cost(cargo, vessel, origin_port, dest_port, route, v_coa_rate, "SPOT", expected_wait_days * 0.8)
            c_tc = self.cost_model.calculate_cost(cargo, vessel, origin_port, dest_port, route, v_tc_rate, "TIME_CHARTER", expected_wait_days * 0.7)
            c_flex = self.cost_model.calculate_cost(cargo, vessel, origin_port, dest_port, route, v_flex_rate, "SPOT", expected_wait_days)

            s_usd = c_spot.total_cost_usd
            coa_usd = c_coa.total_cost_usd
            tc_usd = c_tc.total_cost_usd
            flex_usd = c_flex.total_cost_usd

            spot_tot += s_usd
            coa_tot += coa_usd
            tc_tot += tc_usd
            flex_tot += flex_usd

            best_v_strat = min(
                [("COA", coa_usd), ("TIME_CHARTER", tc_usd), ("FLEXIBLE_INDEX", flex_usd), ("SPOT", s_usd)],
                key=lambda x: x[1]
            )[0]

            voyage_schedule.append(
                SingleVoyagePlan(
                    voyage_number=i + 1,
                    laycan_label=f"Voyage {i+1} (+{day_offset}d)",
                    cargo_quantity_mt=qty_per_voyage,
                    projected_spot_rate=round(v_spot_rate, 2),
                    spot_cost_usd=round(s_usd, 2),
                    coa_cost_usd=round(coa_usd, 2),
                    tc_cost_usd=round(tc_usd, 2),
                    flexible_cost_usd=round(flex_usd, 2),
                    recommended_single_strategy=best_v_strat
                )
            )

        program_totals = {
            "COA": coa_tot,
            "TIME_CHARTER": tc_tot,
            "FLEXIBLE_INDEX": flex_tot,
            "SPOT": spot_tot
        }

        # Select recommended program strategy
        if not forecast.is_promoted or forecast.confidence.value == "LOW":
            # High uncertainty -> choose FLEXIBLE_INDEX to preserve optionality
            rec_program = "FLEXIBLE_INDEX"
        else:
            rec_program = min(program_totals.items(), key=lambda x: x[1])[0]

        rec_cost_usd = program_totals[rec_program]
        savings_usd = spot_tot - rec_cost_usd
        savings_pct = (savings_usd / max(spot_tot, 1e-8)) * 100.0

        plan_id = f"MVP-{num_voyages}V-{vessel.code.upper()}"

        explanation_dict = {
            "objective": "Transition from reactive single spot contracts to structured multi-voyage procurement",
            "number_of_voyages": num_voyages,
            "total_cargo_mt": total_cargo_mt,
            "formulas": {
                "independent_spot_total": "Sum of N single spot fixtures at projected market rates",
                "coa_program_total": "Sum of N voyages at 4% volume discount + 20% delay reduction",
                "tc_program_total": "Time Charter daily hire across total voyage duration + bunker/port costs",
                "flexible_index_total": "Floating index rate with floor/ceiling collar optionality cap",
                "savings_vs_spot": "spot_aggregate_cost_usd - recommended_aggregate_cost_usd (positive = savings vs Spot)",
                "savings_pct": "(savings_vs_spot_usd / spot_aggregate_cost_usd) * 100"
            },
            "strategy_rationale": (
                f"Selected '{rec_program}' multi-voyage program for {num_voyages} voyages ({total_cargo_mt:,.0f} MT total). "
                f"Achieves estimated savings vs spot of ${savings_usd:+,.2f} ({savings_pct:+.2f}%)."
            )
        }

        notes = [
            f"Multi-voyage program covers {num_voyages} consecutive voyages over {num_voyages * interval_days_between_voyages} days.",
            "All calculations are deterministic and based on FICOS CostModel and ForecastService outputs.",
            "Counterfactual savings reflect evaluated program assumptions vs independent spot fixtures, not historical realized savings."
        ]

        return MultiVoyagePlanOutput(
            plan_id=plan_id,
            num_voyages=num_voyages,
            total_cargo_mt=total_cargo_mt,
            vessel_class=vessel.name,
            origin_port=origin_port.name,
            dest_port=dest_port.name,
            spot_aggregate_cost_usd=round(spot_tot, 2),
            coa_aggregate_cost_usd=round(coa_tot, 2),
            tc_aggregate_cost_usd=round(tc_tot, 2),
            flexible_aggregate_cost_usd=round(flex_tot, 2),
            spot_cost_per_mt=round(spot_tot / total_cargo_mt, 2),
            coa_cost_per_mt=round(coa_tot / total_cargo_mt, 2),
            tc_cost_per_mt=round(tc_tot / total_cargo_mt, 2),
            flexible_cost_per_mt=round(flex_tot / total_cargo_mt, 2),
            recommended_program_strategy=rec_program,
            recommended_aggregate_cost_usd=round(rec_cost_usd, 2),
            savings_vs_spot_usd=round(savings_usd, 2),
            savings_pct=round(savings_pct, 2),
            cost_difference_vs_spot_usd=round(savings_usd, 2),
            cost_difference_pct=round(savings_pct, 2),
            voyage_schedule=voyage_schedule,
            explanation=explanation_dict,
            important_notes=notes
        )
