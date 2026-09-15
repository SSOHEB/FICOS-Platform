"""
Tests for Risk Engine and Decision Engine.
"""
from src.risk.engine import RiskEngine
from src.decision.engine import DecisionEngine
from src.application.recommendation_service import RecommendationService

def test_risk_engine_evaluation():
    re = RiskEngine()
    res = re.evaluate_risk("TUBARAO", "QINGDAO", "PANAMAX_1D")
    assert 0.0 <= res.overall_risk_score <= 100.0
    assert res.weather_risk_level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

def test_recommendation_service_end_to_end():
    service = RecommendationService()
    rec = service.get_recommendation(
        asset_type="PANAMAX_1D",
        quantity_mt=75000.0,
        origin_port_code="TUBARAO",
        dest_port_code="QINGDAO",
        laycan_days=14
    )
    assert rec.recommendation_id.startswith("REC-")
    assert rec.recommended_strategy in ["SPOT", "TIME_CHARTER", "COA", "REJECT"]
    assert rec.recommended_cost_usd >= 0.0
    assert rec.feasibility.is_feasible is True
    assert "summary_rationale" in rec.explanation
