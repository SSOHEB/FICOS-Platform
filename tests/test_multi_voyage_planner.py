"""
Tests for Simple Multi-Voyage Planner (SIH 2026 SIH26006)
==========================================================
Verifies:
  1. Multi-voyage program cost calculations (Spot, COA, TC, Flexible Index)
  2. Per-voyage schedule generation
  3. Aggregate cost savings / difference calculations vs Spot
  4. Transparent formula explanation output
"""

import pytest
from src.domain.schemas import CargoRequirement, VesselClass, Port, Route, ForecastResult, UncertaintyLevel
from src.forecast.service import ForecastService
from src.operational.feasibility_engine import FeasibilityEngine
from src.operational.port_repository import PortRepository
from src.operational.vessel_repository import VesselRepository
from src.risk.engine import RiskEngine
from src.cost.model import CostModel
from src.decision.multi_voyage_planner import MultiVoyagePlanner


@pytest.fixture
def setup_planner():
    port_repo = PortRepository()
    vessel_repo = VesselRepository()
    feasibility_engine = FeasibilityEngine(port_repo=port_repo, vessel_repo=vessel_repo)
    forecast_service = ForecastService()
    risk_engine = RiskEngine()
    cost_model = CostModel()
    planner = MultiVoyagePlanner(cost_model=cost_model)
    return {
        "planner": planner,
        "vessel_repo": vessel_repo,
        "port_repo": port_repo,
        "feasibility_engine": feasibility_engine,
        "forecast_service": forecast_service,
        "risk_engine": risk_engine,
        "cost_model": cost_model
    }


def test_multi_voyage_planner_4_voyages(setup_planner):
    planner = setup_planner["planner"]
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="Australia", destination="Dhamra")
    vessel = setup_planner["vessel_repo"].get_vessel_by_code("PANA")
    dest_p = setup_planner["port_repo"].get_port("DHAMRA")
    orig_p = Port(code="AUS", display_name="Australia", key="australia", country="Australia", state="", constraints=None)
    route = Route(origin="Australia", destination="Dhamra")
    
    fc_res = setup_planner["forecast_service"].get_forecast("panamax", 1, 14.5)
    feas_res = setup_planner["feasibility_engine"].check_feasibility(vessel, orig_p, dest_p, cargo)
    risk_res = setup_planner["risk_engine"].evaluate_risk("Australia", "Dhamra", "PANAMAX")
    
    plan = planner.plan_program(
        num_voyages=4,
        cargo=cargo,
        vessel=vessel,
        origin_port=orig_p,
        dest_port=dest_p,
        route=route,
        forecast=fc_res,
        feasibility=feas_res,
        risk=risk_res
    )
    
    assert plan.num_voyages == 4
    assert plan.total_cargo_mt == 300000.0  # 4 x 75,000 MT
    assert len(plan.voyage_schedule) == 4
    assert plan.spot_aggregate_cost_usd > 0
    assert plan.coa_aggregate_cost_usd > 0
    assert plan.tc_aggregate_cost_usd > 0
    assert plan.flexible_aggregate_cost_usd > 0
    assert plan.recommended_program_strategy in ["COA", "TIME_CHARTER", "FLEXIBLE_INDEX", "SPOT"]
    assert "formulas" in plan.explanation
    
    # Verify savings vs spot consistency
    expected_savings = round(plan.spot_aggregate_cost_usd - plan.recommended_aggregate_cost_usd, 2)
    expected_savings_pct = round((expected_savings / plan.spot_aggregate_cost_usd) * 100.0, 2)
    assert plan.savings_vs_spot_usd == expected_savings
    assert plan.savings_pct == expected_savings_pct
    assert plan.cost_difference_vs_spot_usd == expected_savings
    assert plan.cost_difference_pct == expected_savings_pct


def test_multi_voyage_planner_single_voyage_fallback(setup_planner):
    planner = setup_planner["planner"]
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="Australia", destination="Dhamra")
    vessel = setup_planner["vessel_repo"].get_vessel_by_code("PANA")
    dest_p = setup_planner["port_repo"].get_port("DHAMRA")
    orig_p = Port(code="AUS", display_name="Australia", key="australia", country="Australia", state="", constraints=None)
    route = Route(origin="Australia", destination="Dhamra")
    
    fc_res = setup_planner["forecast_service"].get_forecast("panamax", 1, 14.5)
    feas_res = setup_planner["feasibility_engine"].check_feasibility(vessel, orig_p, dest_p, cargo)
    risk_res = setup_planner["risk_engine"].evaluate_risk("Australia", "Dhamra", "PANAMAX")
    
    plan = planner.plan_program(
        num_voyages=1,
        cargo=cargo,
        vessel=vessel,
        origin_port=orig_p,
        dest_port=dest_p,
        route=route,
        forecast=fc_res,
        feasibility=feas_res,
        risk=risk_res
    )
    
    assert plan.num_voyages == 1
    assert plan.total_cargo_mt == 75000.0
    assert len(plan.voyage_schedule) == 1


def test_savings_semantics_cheaper_than_spot(setup_planner):
    """When the recommended strategy costs less than Spot, savings must be strictly positive."""
    planner = setup_planner["planner"]
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="Australia", destination="Dhamra")
    vessel = setup_planner["vessel_repo"].get_vessel_by_code("PANA")
    dest_p = setup_planner["port_repo"].get_port("DHAMRA")
    orig_p = Port(code="AUS", display_name="Australia", key="australia", country="Australia", state="", constraints=None)
    route = Route(origin="Australia", destination="Dhamra")
    
    fc_res = setup_planner["forecast_service"].get_forecast("panamax", 1, 14.85)
    feas_res = setup_planner["feasibility_engine"].check_feasibility(vessel, orig_p, dest_p, cargo)
    risk_res = setup_planner["risk_engine"].evaluate_risk("Australia", "Dhamra", "PANAMAX")
    
    plan = planner.plan_program(
        num_voyages=4,
        cargo=cargo,
        vessel=vessel,
        origin_port=orig_p,
        dest_port=dest_p,
        route=route,
        forecast=fc_res,
        feasibility=feas_res,
        risk=risk_res
    )
    
    # Time Charter is cheaper than Spot
    assert plan.recommended_aggregate_cost_usd < plan.spot_aggregate_cost_usd
    assert plan.savings_vs_spot_usd > 0
    assert plan.savings_pct > 0
    # spot_cost - strategy_cost
    assert plan.savings_vs_spot_usd == pytest.approx(plan.spot_aggregate_cost_usd - plan.recommended_aggregate_cost_usd, abs=0.01)
    assert plan.savings_pct == pytest.approx((plan.savings_vs_spot_usd / plan.spot_aggregate_cost_usd) * 100.0, abs=0.01)


def test_savings_semantics_equal_to_spot():
    """When a strategy equals Spot in cost, savings must be zero (0.00 / 0.0%)."""
    from src.decision.multi_voyage_planner import MultiVoyagePlanOutput
    
    spot_cost = 5000000.0
    rec_cost = 5000000.0
    savings_usd = spot_cost - rec_cost
    savings_pct = (savings_usd / spot_cost) * 100.0
    
    plan = MultiVoyagePlanOutput(
        plan_id="MVP-TEST",
        num_voyages=4,
        total_cargo_mt=300000.0,
        vessel_class="Panamax",
        origin_port="Australia",
        dest_port="Dhamra",
        spot_aggregate_cost_usd=spot_cost,
        coa_aggregate_cost_usd=5200000.0,
        tc_aggregate_cost_usd=5500000.0,
        flexible_aggregate_cost_usd=5100000.0,
        spot_cost_per_mt=16.67,
        coa_cost_per_mt=17.33,
        tc_cost_per_mt=18.33,
        flexible_cost_per_mt=17.00,
        recommended_program_strategy="SPOT",
        recommended_aggregate_cost_usd=rec_cost,
        savings_vs_spot_usd=savings_usd,
        savings_pct=savings_pct,
        voyage_schedule=[],
        explanation={},
        important_notes=[]
    )
    
    assert plan.savings_vs_spot_usd == 0.0
    assert plan.savings_pct == 0.0
    assert plan.cost_difference_vs_spot_usd == 0.0
    assert plan.cost_difference_pct == 0.0


def test_savings_semantics_more_expensive_than_spot():
    """When a strategy is more expensive than Spot, savings must be strictly negative."""
    from src.decision.multi_voyage_planner import MultiVoyagePlanOutput
    
    spot_cost = 5000000.0
    rec_cost = 6000000.0  # $1M more expensive (+20% cost increase)
    savings_usd = spot_cost - rec_cost  # -1,000,000
    savings_pct = (savings_usd / spot_cost) * 100.0  # -20.0%
    
    plan = MultiVoyagePlanOutput(
        plan_id="MVP-TEST-EXPENSIVE",
        num_voyages=4,
        total_cargo_mt=300000.0,
        vessel_class="Panamax",
        origin_port="Australia",
        dest_port="Dhamra",
        spot_aggregate_cost_usd=spot_cost,
        coa_aggregate_cost_usd=6200000.0,
        tc_aggregate_cost_usd=rec_cost,
        flexible_aggregate_cost_usd=6100000.0,
        spot_cost_per_mt=16.67,
        coa_cost_per_mt=20.67,
        tc_cost_per_mt=20.00,
        flexible_cost_per_mt=20.33,
        recommended_program_strategy="TIME_CHARTER",
        recommended_aggregate_cost_usd=rec_cost,
        savings_vs_spot_usd=savings_usd,
        savings_pct=savings_pct,
        voyage_schedule=[],
        explanation={},
        important_notes=[]
    )
    
    assert plan.savings_vs_spot_usd == -1000000.0
    assert plan.savings_pct == -20.0
    assert plan.savings_vs_spot_usd < 0
    assert plan.savings_pct < 0
    # Formatting verification
    formatted_display = f"Savings vs Spot: ${plan.savings_vs_spot_usd:+,.2f} ({plan.savings_pct:+.2f}%)"
    assert formatted_display == "Savings vs Spot: $-1,000,000.00 (-20.00%)"
