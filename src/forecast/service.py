"""
FICOS — Forecast Service
Provides structured freight rate forecasts using the validated Model Registry.

The service:
  1. Queries the registry for the promoted model for (asset, horizon_days).
  2. If no promoted model exists, falls back to FLEXIBLE_INDEX recommendation.
  3. Applies the same preprocessing pipeline as the walk-forward benchmark
     (median imputation, StandardScaler, SelectKBest) — BUT does NOT retrain here.
     At inference time, uses the current-rate + delta approach.
  4. Returns a structured ForecastResult.

IMPORTANT:
  This service PREDICTS. It does NOT decide.
  The DecisionEngine makes decisions based on ForecastResult.

IMPORTANT (Deployment Note):
  The walk-forward validation benchmark was used to SELECT the best model class
  and evaluate its robustness across 5 historical folds.
  A FINAL production model must be retrained via ModelTrainer on the full
  development set before this service can return ML predictions.
  Until final production training is done, the service returns a
  'current_rate_persistence' fallback (delta = 0) for safety.

  When model artifacts exist, load them via src/forecast/models.py.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.domain.schemas import ForecastResult, UncertaintyLevel
from src.registry.registry import ModelRegistry, get_registry
from src.forecast.uncertainty import UncertaintyEngine

warnings.filterwarnings("ignore")


class ForecastService:
    """
    Clean forecast abstraction. Accepts asset, date, horizon, and feature frame.
    Returns a structured ForecastResult.

    Separation of concerns:
      - ForecastService → predicts (P10/P50/P90, model metadata)
      - DecisionEngine  → decides (BUY_NOW / WAIT / FLEXIBLE)
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self._registry = registry or get_registry()
        self.registry = self._registry
        self._uncertainty = UncertaintyEngine()

    def get_forecast(

        self,
        asset_type: str,
        horizon_days: int = 14,
        current_rate: float = 25.0,
        forecast_date: str = "2026-09-14"
    ) -> ForecastResult:
        """Convenience method to query forecast by asset_type and horizon."""
        return self.forecast(
            asset=asset_type,
            forecast_date=forecast_date,
            horizon_days=horizon_days,
            current_rate=current_rate
        )

    def forecast(
        self,
        asset: str,
        forecast_date: str,
        horizon_days: int,
        current_rate: float,
        forecast_delta: Optional[float] = None,
    ) -> ForecastResult:

        """
        Generate a ForecastResult for the given asset/horizon.

        Parameters
        ----------
        asset : str
            Freight class, e.g. "panamax", "supramax"
        forecast_date : str
            ISO date string "YYYY-MM-DD"
        horizon_days : int
            Forecast horizon in days (1, 7, 14, 30)
        current_rate : float
            Current freight rate ($/day TCE equivalent)
        forecast_delta : float, optional
            Pre-computed delta forecast (from model output or simulation).
            If None, uses persistence (delta = 0) as safe fallback.

        Returns
        -------
        ForecastResult
        """
        asset_lower = asset.lower()
        is_promoted = self._registry.is_promoted(asset_lower, horizon_days)
        registry_entry = self._registry.get_promoted(asset_lower, horizon_days)

        # ── Model + uncertainty bounds ──
        if registry_entry is not None:
            p10_bound = registry_entry.get("p10_bound", -0.15 * current_rate)
            p90_bound = registry_entry.get("p90_bound", 0.15 * current_rate)
            model_name = registry_entry.get("model_type", "RandomForestRegressor")
            model_version = registry_entry.get("model_version", "1.0.0")
            fallback_used = False
        else:
            # Not promoted — use dynamic fallback bounds (±25% of current rate)
            p10_bound = -0.25 * current_rate
            p90_bound = 0.25 * current_rate
            model_name = "PersistenceFallback"
            model_version = "0.0.0"
            fallback_used = True
            is_promoted = False


        # ── Delta / point forecast ──
        if forecast_delta is not None and is_promoted and not fallback_used:
            delta = float(forecast_delta)
        else:
            # Safe fallback: predict no change (persistence)
            delta = 0.0
            if not fallback_used:
                fallback_used = True

        point_forecast = current_rate + delta
        pct_change = delta / max(abs(current_rate), 1e-8)

        # ── Uncertainty profile ──
        unc = self._uncertainty.compute(
            point_forecast=point_forecast,
            p10_bound=p10_bound,
            p90_bound=p90_bound,
            current_rate=current_rate,
        )

        return ForecastResult(
            asset=asset_lower,
            forecast_date=forecast_date,
            horizon_days=horizon_days,
            current_rate=round(current_rate, 2),
            point_forecast=round(point_forecast, 2),
            p10=unc["p10"],
            p50=unc["p50"],
            p90=unc["p90"],
            expected_delta=round(delta, 2),
            expected_pct_change=round(pct_change * 100, 4),
            model_name=model_name,
            model_version=model_version,
            confidence=unc["confidence"],
            uncertainty_width=unc["interval_width"],
            is_promoted=is_promoted,
            fallback_used=fallback_used,
            feature_metadata={
                "p10_bound_delta": p10_bound,
                "p90_bound_delta": p90_bound,
                "uncertainty_note": unc["note"],
            },
        )

    def is_supported(self, asset: str, horizon_days: int) -> bool:
        """Check if a promoted model exists for the asset/horizon combination."""
        return self._registry.is_promoted(asset.lower(), horizon_days)
