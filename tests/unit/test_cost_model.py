"""
Tests for Cost Model Engine.
"""
from src.cost.model import CostModel
from src.domain.schemas import CargoRequirement, VesselClass, Port, Route, PortConstraints

def test_cost_model_calculation():
    cm = CostModel()
    cargo = CargoRequirement(cargo_type="PANAMAX_1D", quantity_mt=75000.0, origin="TUBARAO", destination="QINGDAO")
    vessel = VesselClass(code="PANA", display_name="Panamax", freight_index="panamax")
    origin = Port(code="TUBARAO", display_name="Tubarao", key="tubarao", country="Brazil", state="", constraints=PortConstraints(23.0, 300.0, 150000.0))
    dest = Port(code="QINGDAO", display_name="Qingdao", key="qingdao", country="China", state="", constraints=PortConstraints(18.0, 300.0, 150000.0))
    route = Route(origin="TUBARAO", destination="QINGDAO", distance_nautical_miles=11000.0)

    breakdown = cm.calculate_cost(cargo, vessel, origin, dest, route, freight_rate_usd_ton=25.0)
    assert breakdown.total_cost_usd > 0.0
    assert breakdown.cost_per_mt > 0.0
    assert len(breakdown.components) >= 5
