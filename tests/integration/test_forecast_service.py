"""
Tests for Forecast Service and Uncertainty Engine.
"""
from ml.forecasting.service import ForecastService
from ml.forecasting.uncertainty import UncertaintyEngine

def test_forecast_service_promoted_model():
    service = ForecastService()
    fc = service.get_forecast("PANAMAX_1D", 14)
    assert fc.asset_type == "PANAMAX_1D"
    assert fc.point_forecast > 0.0
    assert fc.uncertainty.p10 <= fc.point_forecast <= fc.uncertainty.p90

def test_uncertainty_engine():
    engine = UncertaintyEngine()
    unc = engine.calculate_uncertainty(point_forecast=25.0, p10=22.0, p90=28.0)
    assert unc.p10 == 22.0
    assert unc.p90 == 28.0
    assert unc.confidence_interval_width == 6.0
