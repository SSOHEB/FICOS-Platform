"""
FICOS — Configuration Module
"""
from src.config.settings import Settings, get_settings
from src.config.canonical_config import (
    CANONICAL_N_TREES,
    CANONICAL_SEED,
    CANONICAL_N_JOBS,
    CANONICAL_DATASET_RELATIVE_PATH,
    CANONICAL_DATASET_SHA256,
    CANONICAL_GIT_COMMIT,
    CANONICAL_MODEL_PARAMS,
    CANONICAL_COST_MODEL,
    CANONICAL_GATING_POLICY,
    CANONICAL_FOLDS,
    compute_file_sha256,
    validate_dataset_provenance,
    validate_hyperparameters,
    validate_cost_model,
    validate_gating_policy,
    assert_authoritative_certification,
)

__all__ = [
    "Settings",
    "get_settings",
    "CANONICAL_N_TREES",
    "CANONICAL_SEED",
    "CANONICAL_N_JOBS",
    "CANONICAL_DATASET_RELATIVE_PATH",
    "CANONICAL_DATASET_SHA256",
    "CANONICAL_GIT_COMMIT",
    "CANONICAL_MODEL_PARAMS",
    "CANONICAL_COST_MODEL",
    "CANONICAL_GATING_POLICY",
    "CANONICAL_FOLDS",
    "compute_file_sha256",
    "validate_dataset_provenance",
    "validate_hyperparameters",
    "validate_cost_model",
    "validate_gating_policy",
    "assert_authoritative_certification",
]
