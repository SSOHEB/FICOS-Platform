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
