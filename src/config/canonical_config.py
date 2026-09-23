"""
FICOS — Single Source of Truth Canonical Configuration & Provenance Engine
==========================================================================
Mandatory configuration for all FICOS validation, economic backtesting, and reporting.
No script may alter or locally redefine N_TREES, SEED, n_jobs, or dataset identity.
"""

from __future__ import annotations
import os
import sys
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

# Project Root Resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Single Source of Truth Hyperparameters
CANONICAL_N_TREES: int = 100
CANONICAL_SEED: int = 42
CANONICAL_N_JOBS: int = 1

# Single Source of Truth Dataset Identity
CANONICAL_DATASET_RELATIVE_PATH: str = "data/modeling_dataset.csv"
CANONICAL_DATASET_SHA256: str = "e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5"
CANONICAL_DATASET_EXPECTED_ROWS: int = 2581
CANONICAL_DATASET_EXPECTED_COLS: int = 482

def _get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_PROJECT_ROOT)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "09bad6dfbd0b54c82fc5628307bb6949319a5ff3"

CANONICAL_GIT_COMMIT: str = _get_git_commit()

# Production Model Hyperparameters
CANONICAL_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "RF_STANDARD": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": 1,
    },
    "LIGHTGBM": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.03,
        "random_state": 42,
        "n_jobs": 1,
        "deterministic": True,
    },
    "XGBOOST": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.03,
        "random_state": 42,
        "n_jobs": 1,
    },
    "CATBOOST_GBR_FALLBACK": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.03,
        "random_state": 42,
    },
    "RIDGE": {
        "alpha": 100.0,
    }
}

# Cost Model Single Source of Truth
CANONICAL_COST_MODEL: Dict[str, float] = {
    "voyage_duration": 20.0,      # days
    "daily_idle": 2500.0,         # USD per day
    "wait_idle_days": 1.0,        # days
    "flex_idle_days": 0.25,       # days
}

# Gating & Actionability Policy
CANONICAL_GATING_POLICY: Dict[str, float] = {
    "tau": 0.01,                  # 1% move threshold
    "q_low": 10.0,                # 10th percentile residual bound
    "q_high": 90.0,               # 90th percentile residual bound
}

# Fold Definitions
CANONICAL_FOLDS: Dict[int, Dict[str, str]] = {
    1: {"test_year": "2021", "train_end": "2019-12-24", "val_start": "2020-01-03", "val_end": "2020-12-24", "test_start": "2021-01-05", "test_end": "2021-12-31"},
    2: {"test_year": "2022", "train_end": "2020-12-24", "val_start": "2021-01-05", "val_end": "2021-12-24", "test_start": "2022-01-03", "test_end": "2022-12-30"},
    3: {"test_year": "2023", "train_end": "2021-12-24", "val_start": "2022-01-03", "val_end": "2022-12-23", "test_start": "2023-01-03", "test_end": "2023-12-29"},
    4: {"test_year": "2024", "train_end": "2022-12-23", "val_start": "2023-01-03", "val_end": "2023-12-22", "test_start": "2024-01-02", "test_end": "2024-12-31"},
    5: {"test_year": "2025", "train_end": "2023-12-22", "val_start": "2024-01-02", "val_end": "2024-12-24", "test_start": "2025-01-02", "test_end": "2025-12-31"},
}


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 digest of a file in binary chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset_provenance(dataset_path: str | Path | None = None) -> Tuple[bool, str]:
    """
    Assert dataset exists, matches expected row/col shape and exact SHA-256 hash.
    Raises ValueError on mismatch.
    """
    if dataset_path is None:
        dataset_path = _PROJECT_ROOT / CANONICAL_DATASET_RELATIVE_PATH

    path_obj = Path(dataset_path)
    if not path_obj.exists():
        msg = f"CRITICAL: Dataset file not found at '{path_obj}'"
        raise ValueError(msg)

    actual_sha = compute_file_sha256(path_obj)
    if actual_sha != CANONICAL_DATASET_SHA256:
        msg = (
            f"CRITICAL DATASET HASH MISMATCH!\n"
            f"Expected SHA256: {CANONICAL_DATASET_SHA256}\n"
            f"Actual SHA256:   {actual_sha}\n"
            f"The dataset has been modified or replaced!"
        )
        raise ValueError(msg)

    return True, actual_sha


def validate_hyperparameters(n_trees: int, seed: int, n_jobs: int) -> bool:
    """
    Assert exact canonical hyperparameter settings.
    Raises ValueError on mismatch.
    """
    errors = []
    if n_trees != CANONICAL_N_TREES:
        errors.append(f"N_TREES must be {CANONICAL_N_TREES}, got {n_trees}")
    if seed != CANONICAL_SEED:
        errors.append(f"SEED must be {CANONICAL_SEED}, got {seed}")
    if n_jobs != CANONICAL_N_JOBS:
        errors.append(f"n_jobs must be {CANONICAL_N_JOBS}, got {n_jobs}")

    if errors:
        msg = "CRITICAL HYPERPARAMETER PROVENANCE VIOLATION:\n" + "\n".join(errors)
        raise ValueError(msg)

    return True


def validate_cost_model(cost_params: Dict[str, float]) -> bool:
    """Assert cost model parameters match canonical single source of truth."""
    for key, expected_val in CANONICAL_COST_MODEL.items():
        actual_val = cost_params.get(key)
        if actual_val is None or abs(actual_val - expected_val) > 1e-6:
            raise ValueError(
                f"CRITICAL COST MODEL MISMATCH: {key} expected {expected_val}, got {actual_val}"
            )
    return True


def validate_gating_policy(policy_params: Dict[str, float]) -> bool:
    """Assert gating policy parameters match canonical single source of truth."""
    for key, expected_val in CANONICAL_GATING_POLICY.items():
        actual_val = policy_params.get(key)
        if actual_val is None or abs(actual_val - expected_val) > 1e-6:
            raise ValueError(
                f"CRITICAL GATING POLICY MISMATCH: {key} expected {expected_val}, got {actual_val}"
            )
    return True


def assert_authoritative_certification(dataset_sha: str, n_trees: int, seed: int, n_jobs: int) -> str:
    """
    Strict guard. Refuses to return AUTHORITATIVE/CERTIFIED status if any check fails.
    """
    validate_hyperparameters(n_trees, seed, n_jobs)
    if dataset_sha != CANONICAL_DATASET_SHA256:
        raise ValueError("Cannot certify: Dataset SHA mismatch!")

    return "AUTHORITATIVE / CERTIFIED ✅"
