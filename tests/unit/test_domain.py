"""
Tests for domain schemas and dataclasses.
"""
import pytest
from src.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, ForecastResult, FeasibilityResult, RiskResult
)

def test_cargo_requirement_creation():
    cargo = CargoRequirement(
        cargo_id="C001",
        asset_type="PANAMAX_1D",
        quantity_mt=75000.0,
        origin_port_code="TUBARAO",
        dest_port_code="QINGDAO",
        laycan_days=14
    )
    assert cargo.cargo_id == "C001"
    assert cargo.quantity_mt == 75000.0

def test_vessel_class_creation():
    vessel = VesselClass(
        name="Panamax",
        max_dwt=80000.0,
        max_draft_m=14.5,
        max_beam_m=32.3,
        max_loa_m=229.0,
        speed_knots=14.0,
        fuel_consumption_laden_tpd=30.0,
        fuel_consumption_port_tpd=3.0,
        loading_rate_tpd=25000.0,
        discharge_rate_tpd=20000.0,
        gross_tonnage=43000.0
    )
    assert vessel.name == "Panamax"
    assert vessel.max_dwt == 80000.0
