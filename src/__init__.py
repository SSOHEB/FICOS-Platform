"""
FICOS Platform — Dry Bulk Freight Forecasting & Decision Support Architecture
"""

from src.ficos.data.data_loader import load_dataset_a, load_dataset_b, load_dataset_c, load_config
from src.ficos.features.features import build_modeling_dataset
from src.ficos.evaluation.metrics import evaluate_model, directional_accuracy, smape
from src.ficos.operational.feasibility_engine import FeasibilityEngine

__all__ = [
    "load_dataset_a",
    "load_dataset_b",
    "load_dataset_c",
    "load_config",
    "build_modeling_dataset",
    "evaluate_model",
    "directional_accuracy",
    "smape",
    "FeasibilityEngine",
]
