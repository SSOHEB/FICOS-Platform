"""
Tests for configuration loading and validation.
"""
from pathlib import Path
import yaml

def test_ports_yaml_exists_and_valid():
    p = Path("configs/ports.yaml")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "ports" in data
    assert len(data["ports"]) > 0

def test_vessels_yaml_exists_and_valid():
    p = Path("configs/vessels.yaml")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "vessel_classes" in data
    assert len(data["vessel_classes"]) > 0

def test_cost_model_yaml_exists_and_valid():
    p = Path("configs/cost_model.yaml")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "port_costs" in data
    assert "waiting_cost" in data
