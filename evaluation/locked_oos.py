"""Utilities that make locked-OOS contamination explicit."""
from __future__ import annotations
import pandas as pd

def assert_locked_oos_isolation(train_dates, evaluation_dates):
    train = set(pd.to_datetime(train_dates).strftime("%Y-%m-%d"))
    evaluation = set(pd.to_datetime(evaluation_dates).strftime("%Y-%m-%d"))
    overlap = train.intersection(evaluation)
    if overlap:
        raise ValueError(f"Locked OOS dates used in training/tuning: {sorted(overlap)[:5]}")
    return {"status": "PASS", "train_rows": len(train), "evaluation_rows": len(evaluation), "overlap": 0}

def locked_metric_record(observed, predictions, dates, evaluation_start):
    dates = pd.to_datetime(dates)
    mask = dates >= pd.Timestamp(evaluation_start)
    if not mask.any():
        raise ValueError("No locked evaluation rows remain")
    actual = pd.Series(observed)[mask].to_numpy(float)
    pred = pd.Series(predictions)[mask].to_numpy(float)
    error = pred - actual
    return {"evaluation_start": str(evaluation_start), "n": int(mask.sum()), "mae": float(abs(error).mean()), "rmse": float((error ** 2).mean() ** 0.5)}
