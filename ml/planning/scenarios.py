"""Deterministic empirical/scenario rate generation."""
from __future__ import annotations
import numpy as np

def generate_rate_scenarios(forecast_rates, residuals, n_scenarios=100, seed=42, shock_scale=1.0):
    forecasts = np.asarray(forecast_rates, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    errors = errors[np.isfinite(errors)]
    if forecasts.ndim != 1 or len(forecasts) == 0:
        raise ValueError("forecast_rates must be a non-empty vector")
    if len(errors) == 0:
        errors = np.array([0.0])
    sampled = np.random.default_rng(seed).choice(errors, size=(n_scenarios, len(forecasts)), replace=True)
    return np.maximum(0.0, forecasts[None, :] + shock_scale * sampled)
