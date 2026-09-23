"""
Tests for ScenarioEngine.
"""
from src.domain.schemas import ScenarioType, CargoRequirement, Port, VesselClass, Route
from src.scenario.engine import ScenarioEngine
from src.decision.engine import DecisionEngine

def test_scenario_engine_execution():
    se = ScenarioEngine()
    de = DecisionEngine()
    cargo = CargoRequirement(cargo_type="Thermal Coal", quantity_mt=75000.0, origin="TUBARAO", destination="QINGDAO")
    vessel = VesselClass(code="PANA", display_name="Panamax")
    origin = Port(code="TUBARAO", display_name="Tubarao", key="tubarao", country="Brazil", state="", constraints=None)
    dest = Port(code="QINGDAO", display_name="Qingdao", key="qingdao", country="China", state="", constraints=None)
    route = Route(origin="TUBARAO", destination="QINGDAO")

    # Evaluate baseline
    base_rec = de.evaluate(cargo, vessel, origin, dest, route)
    assert base_rec.recommended_strategy in ["SPOT", "TIME_CHARTER", "COA", "REJECT"]

    # Run scenario
    result = se.run(
        scenario=ScenarioType.WEATHER_DISRUPTION,
        decision_engine=de,
        cargo=cargo,
        current_rate=25.0
    )
    assert result.scenario == ScenarioType.WEATHER_DISRUPTION
    assert result.explanation != ""
