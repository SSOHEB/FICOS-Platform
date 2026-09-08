"""
SIH26006 — Model Evaluation Module

Metrics:
- MAE, RMSE, sMAPE, R-squared, directional accuracy
- All computed on the test set only
- No metric manipulation to achieve desired accuracy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return 100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 2


def directional_accuracy(y_true, y_pred, y_prev):
    """
    Directional accuracy: fraction of times the predicted direction
    (up/down relative to previous value) matches the actual direction.

    y_prev: the value at time t (before the forecast horizon)
    y_true: the actual value at time t+h
    y_pred: the predicted value at time t+h
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)

    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)

    # Only count where both are non-NaN and direction is not exactly zero
    valid = ~(np.isnan(actual_dir) | np.isnan(pred_dir))
    if valid.sum() == 0:
        return np.nan

    return float(np.mean(actual_dir[valid] == pred_dir[valid]))


def evaluate_model(y_true, y_pred, y_prev=None, model_name="", target="", horizon=""):
    """
    Compute all evaluation metrics for a single model/target/horizon.
    Returns a dict.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]

    n = len(y_t)
    if n == 0:
        return {
            "model": model_name, "freight_class": target, "horizon": horizon,
            "MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan, "R2": np.nan,
            "directional_accuracy": np.nan, "n_test": 0
        }

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    smape_val = smape(y_t, y_p)
    r2 = r2_score(y_t, y_p)

    dir_acc = np.nan
    if y_prev is not None:
        y_pr = y_prev[mask]
        dir_acc = directional_accuracy(y_t, y_p, y_pr)

    return {
        "model": model_name,
        "freight_class": target,
        "horizon": horizon,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "sMAPE": round(smape_val, 2),
        "R2": round(r2, 4),
        "directional_accuracy": round(dir_acc, 4) if not np.isnan(dir_acc) else np.nan,
        "n_test": n,
    }
