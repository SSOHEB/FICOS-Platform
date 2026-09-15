"""
Tests for physical feasibility engine.
"""
from src.operational.port_repository import PortRepository
from src.operational.vessel_repository import VesselRepository
from src.operational.feasibility_engine import FeasibilityEngine
from src.domain.schemas import CargoRequirement

def test_feasibility_check_pass():
    port_repo = PortRepository()
    vessel_repo = VesselRepository()
    engine = FeasibilityEngine(port_repo, vessel_repo)

    origin = port_repo.get_port("TUBARAO")
    dest = port_repo.get_port("QINGDAO")
    vessel = vessel_repo.get_vessel("PANAMAX")
    cargo = CargoRequirement(cargo_type="Thermal Coal", quantity_mt=75000.0, origin=origin.code, destination=dest.code)

    result = engine.check_feasibility(vessel, origin, dest, cargo)
    assert result.is_feasible is True

def test_feasibility_check_fail_draft():
    port_repo = PortRepository()
    vessel_repo = VesselRepository()
    engine = FeasibilityEngine(port_repo, vessel_repo)

    origin = port_repo.get_port("TUBARAO")
    dest = port_repo.get_port("QINGDAO")
    # Oversized vessel exceeding max draft
    vessel = vessel_repo.get_vessel("CAPESIZE")
    vessel.typical_draft_m = 25.0  # artificially high draft exceeding port max (18m)
    cargo = CargoRequirement(cargo_type="Thermal Coal", quantity_mt=170000.0, origin=origin.code, destination=dest.code)

    result = engine.check_feasibility(vessel, origin, dest, cargo)
    assert result.is_feasible is False
