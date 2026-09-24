"""
Tests for FICOS FastAPI Endpoints
==================================
Verifies:
  1. GET /health
  2. GET /forecast (Promoted directional vs unpromoted fallback routing)
  3. GET /vessel-feasibility
  4. GET /risk-breakdown
  5. GET /idle-risk
  6. GET /shock-response
  7. GET /fleet-status
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "service" in json_data
    assert "promoted_pairs" in json_data
    assert "model_version" in json_data


def test_forecast_promoted_pair():
    # Panamax 1d is promoted
    response = client.get("/forecast?vessel_class=panamax&horizon=1d&origin=Australia&destination=Dhamra&cargo_qty=75000")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["data"]["is_promoted"] is True
    assert json_data["data"]["recommended_action"] in ["BUY NOW", "WAIT", "FLEXIBLE"]
    assert "p10" in json_data["data"]
    assert "p50" in json_data["data"]
    assert "p90" in json_data["data"]


def test_forecast_unpromoted_pair_fallback():
    # Supramax 14d is unpromoted / excluded in 1d registry
    response = client.get("/forecast?vessel_class=supramax&horizon=14d&origin=Australia&destination=Dhamra&cargo_qty=75000")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "fallback"
    assert json_data["data"]["is_promoted"] is False
    assert json_data["data"]["recommended_action"] == "FLEXIBLE"
    assert "FLEXIBLE / INDEX-LINKED" in json_data["data"]["action_rationale"]


def test_vessel_feasibility():
    response = client.get("/vessel-feasibility?cargo_qty=75000&origin=Australia&destination=Dhamra")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "feasibility_percentage" in json_data["data"]
    assert "per_class_comparison" in json_data["data"]
    assert len(json_data["data"]["per_class_comparison"]) > 0


def test_risk_breakdown():
    response = client.get("/risk-breakdown?destination_port=Dhamra&route=Australia-India")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "overall_risk_score" in json_data["data"]
    assert "categories" in json_data["data"]
    assert len(json_data["data"]["categories"]) == 4


def test_idle_risk():
    response = client.get("/idle-risk")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "total_projected_idle_cost_usd" in json_data["data"]
    assert "vessels" in json_data["data"]


def test_shock_response():
    response = client.get("/shock-response?event_type=red_sea&route=Pacific-Indian")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["data"]["event_details"]["surge_peak"] == "+18.4%"

    # Unknown event returns unavailable
    bad_res = client.get("/shock-response?event_type=unknown_event&route=Pacific-Indian")
    assert bad_res.status_code == 200
    bad_json = bad_res.json()
    assert bad_json["status"] == "unavailable"


def test_fleet_status():
    response = client.get("/fleet-status")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "summary" in json_data["data"]
    assert "fleet" in json_data["data"]


def test_procurement_decision_endpoint():
    response = client.get("/procurement-decision?vessel_class=panamax&horizon=1d&origin=Australia&destination=Dhamra&cargo_qty=75000&num_voyages=4")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "recommended_timing" in json_data["data"]
    assert "recommended_contract_strategy" in json_data["data"]
    assert "multi_voyage_plan" in json_data["data"]
    assert "forecast_provenance" in json_data["data"]
    assert json_data["data"]["multi_voyage_plan"]["num_voyages"] == 4


def test_multi_voyage_plan_endpoint():
    response = client.get("/multi-voyage-plan?vessel_class=supramax&origin=Australia&destination=Dhamra&cargo_qty=75000&num_voyages=6")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["data"]["num_voyages"] == 6
    assert "spot_aggregate_cost_usd" in json_data["data"]
    assert "recommended_program_strategy" in json_data["data"]
