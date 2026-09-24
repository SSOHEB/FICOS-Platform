"""
Tests for Procurement Decision Engine & Strategy Comparator (SIH 2026 SIH26006)
===================================================================================
Verifies:
  1. Provenance propagation (vessel_class, horizon, model, status, validation)
  2. Timing & Strategy separation (WHEN vs HOW)
  3. Flexible fallback preserving optionality during elevated uncertainty
  4. 4-way contract strategy comparison (SPOT, TIME_CHARTER, COA, FLEXIBLE_INDEX)
  5. Feasibility filtering
  6. Explainable output rationale
"""

import pytest
from backend.domain.schemas import CargoRequirement, VesselClass, Port, Route, ForecastResult, UncertaintyLevel
from ml.forecasting.service import ForecastService
from backend.operational.feasibility_engine import FeasibilityEngine
from backend.operational.port_repository import PortRepository
from backend.operational.vessel_repository import VesselRepository
from backend.risk.engine import RiskEngine
from backend.cost.model import CostModel
from backend.decision.contract_comparison import ContractStrategyComparator
from backend.decision.procurement_engine import ProcurementDecisionEngine


@pytest.fixture
def setup_services():
    port_repo = PortRepository()
    vessel_repo = VesselRepository()
    feasibility_engine = FeasibilityEngine(port_repo=port_repo, vessel_repo=vessel_repo)
    forecast_service = ForecastService()
    risk_engine = RiskEngine()
    cost_model = CostModel()
    engine = ProcurementDecisionEngine(
        forecast_service=forecast_service,
        feasibility_engine=feasibility_engine,
        risk_engine=risk_engine,
        cost_model=cost_model
    )
    return {
        "engine": engine,
        "vessel_repo": vessel_repo,
        "port_repo": port_repo,
        "feasibility_engine": feasibility_engine,
        "forecast_service": forecast_service,
        "risk_engine": risk_engine,
        "cost_model": cost_model
    }


def test_forecast_provenance_propagation(setup_services):
    forecast_service = setup_services["forecast_service"]
    fc_res = forecast_service.get_forecast(asset_type="panamax", horizon_days=1, current_rate=15.0)
    
    prov = fc_res.to_provenance_dict()
    assert prov["vessel_class"] == "PANAMAX"
    assert prov["horizon"] == "1d"
    assert "forecast_value" in prov
    assert "uncertainty" in prov
    assert "model_used" in prov
    assert "production_status" in prov
    assert "validation_status" in prov
    assert prov["production_status"] == "promoted"


def test_contract_strategy_comparison_4way(setup_services):
    comparator = ContractStrategyComparator(cost_model=setup_services["cost_model"])
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="TUBARAO", destination="QINGDAO")
    vessel = setup_services["vessel_repo"].get_vessel_by_code("PANA")
    dest_p = setup_services["port_repo"].get_port("QINGDAO")
    orig_p = setup_services["port_repo"].get_port("TUBARAO")
    route = Route(origin="TUBARAO", destination="QINGDAO")
    
    fc_res = setup_services["forecast_service"].get_forecast("panamax", 1, 15.0)
    feas_res = setup_services["feasibility_engine"].check_feasibility(vessel, orig_p, dest_p, cargo)
    risk_res = setup_services["risk_engine"].evaluate_risk("TUBARAO", "QINGDAO", "PANAMAX")
    
    details = comparator.compare_all(cargo, vessel, orig_p, dest_p, route, fc_res, feas_res, risk_res)
    assert len(details) == 4
    strat_names = [d.strategy_name for d in details]
    assert "SPOT" in strat_names
    assert "TIME_CHARTER" in strat_names
    assert "COA" in strat_names
    assert "FLEXIBLE_INDEX" in strat_names
    
    for d in details:
        assert d.expected_cost_usd > 0
        assert d.cost_per_mt > 0
        assert d.flexibility_rating in ["HIGH", "MEDIUM", "LOW"]
        assert len(d.important_assumptions) > 0


def test_procurement_engine_timing_and_strategy_separation(setup_services):
    engine = setup_services["engine"]
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="Australia", destination="Dhamra")
    vessel = setup_services["vessel_repo"].get_vessel_by_code("PANA")
    dest_p = setup_services["port_repo"].get_port("DHAMRA")
    orig_p = Port(code="AUS", display_name="Australia", key="australia", country="Australia", state="", constraints=None)
    route = Route(origin="Australia", destination="Dhamra")
    
    output = engine.evaluate_procurement(
        cargo=cargo,
        vessel=vessel,
        origin_port=orig_p,
        dest_port=dest_p,
        route=route,
        current_rate=14.5,
        num_voyages=1
    )
    
    dict_out = output.to_dict()
    assert "recommended_timing" in dict_out
    assert "recommended_contract_strategy" in dict_out
    assert dict_out["recommended_timing"] in ["NOW", "WAIT", "FLEXIBLE"]
    assert dict_out["recommended_contract_strategy"] in ["SPOT", "TIME_CHARTER", "COA", "FLEXIBLE_INDEX"]
    assert "forecast_provenance" in dict_out
    assert "explanation" in dict_out
    assert "strategy_comparison" in dict_out


def test_flexible_fallback_during_unpromoted_regime(setup_services):
    engine = setup_services["engine"]
    cargo = CargoRequirement(cargo_type="Coal", quantity_mt=75000, origin="Australia", destination="Dhamra")
    vessel = setup_services["vessel_repo"].get_vessel_by_code("SUPR")
    dest_p = setup_services["port_repo"].get_port("DHAMRA")
    orig_p = Port(code="AUS", display_name="Australia", key="australia", country="Australia", state="", constraints=None)
    route = Route(origin="Australia", destination="Dhamra")
    
    # Supramax 14D is unpromoted fallback
    cargo.laycan_days = 14
    output = engine.evaluate_procurement(
        cargo=cargo,
        vessel=vessel,
        origin_port=orig_p,
        dest_port=dest_p,
        route=route,
        current_rate=14.5,
        num_voyages=1
    )
    
    # Should recommend FLEXIBLE / FLEXIBLE_INDEX to preserve optionality
    assert output.recommended_timing == "FLEXIBLE"
    assert any(term in output.strategy_rationale.lower() for term in ["floating", "index-linked", "uncertainty", "flexible", "optionality"])
