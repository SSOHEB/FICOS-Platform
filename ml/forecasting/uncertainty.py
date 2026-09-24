"""
FICOS — Uncertainty Engine
Converts empirical P10/P90 bounds from the registry into a full
UncertaintyResult, using the same residual-based calibration methodology
as the validated benchmark.

IMPORTANT:
  These are empirical validation-residual bounds, NOT mathematically
  guaranteed 80% prediction intervals. They are calibrated from
  out-of-sample walk-forward validation residuals.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from src.domain.schemas import UncertaintyLevel


def _load_policy() -> dict:
    root = Path(__file__).resolve().parent.parent.parent
    policy_path = root / "configs" / "decision_policy.yaml"
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UncertaintyEngine:
    """
    Converts P10/P90 registry bounds into classified uncertainty levels
    and computes the full uncertainty profile for a forecast.

    Design:
    - Bounds come from registry validation residuals (already computed).
    - Classification is purely configuration-driven.
    - Never claims guaranteed probability coverage.
    """

    def __init__(self):
        policy = _load_policy()
        unc_cfg = policy.get("uncertainty", {})
        self.low_threshold_pct = unc_cfg.get("low_threshold_pct", 5.0)
        self.medium_threshold_pct = unc_cfg.get("medium_threshold_pct", 15.0)

    def classify(
        self,
        p10: float,
        p90: float,
        current_rate: float,
    ) -> UncertaintyLevel:
        """
        Classify uncertainty level based on interval width relative to current rate.
        Configuration-driven thresholds from decision_policy.yaml.
        """
        if current_rate <= 0:
            return UncertaintyLevel.HIGH

        interval_width = abs(p90 - p10)
        width_pct = (interval_width / abs(current_rate)) * 100.0

        if width_pct < self.low_threshold_pct:
            return UncertaintyLevel.LOW
        elif width_pct < self.medium_threshold_pct:
            return UncertaintyLevel.MEDIUM
        else:
            return UncertaintyLevel.HIGH

    def calculate_uncertainty(
        self,
        point_forecast: float,
        p10: float,
        p90: float
    ) -> UncertaintyResult:
        """Calculate uncertainty result object."""
        from src.domain.schemas import UncertaintyResult
        width = abs(p90 - p10)
        return UncertaintyResult(p10=p10, p50=point_forecast, p90=p90, confidence_interval_width=width)

    def compute(
        self,
        point_forecast: float,
        p10_bound: float,
        p90_bound: float,
        current_rate: float,
    ) -> dict:

        """
        Compute the full uncertainty profile for a single forecast.

        Returns a dict with:
          p10, p50, p90, interval_width, confidence, uncertainty_width_pct
        """
        # p10 / p90 are delta bounds (from the registry).
        # Convert to absolute rate level terms for presentation, preventing non-physical negative rates.
        p10_level = max(0.0, current_rate + p10_bound)
        p90_level = max(0.0, current_rate + p90_bound)
        p50_level = max(0.0, point_forecast)  # point estimate is the median

        interval_width = p90_level - p10_level
        confidence = self.classify(p10_level, p90_level, current_rate)


        width_pct = (abs(interval_width) / max(abs(current_rate), 1.0)) * 100.0

        return {
            "p10": round(p10_level, 2),
            "p50": round(p50_level, 2),
            "p90": round(p90_level, 2),
            "interval_width": round(interval_width, 2),
            "uncertainty_width_pct": round(width_pct, 2),
            "confidence": confidence,
            "note": (
                "Empirical validation-residual bounds from walk-forward evaluation. "
                "Not a statistically guaranteed prediction interval."
            ),
        }
