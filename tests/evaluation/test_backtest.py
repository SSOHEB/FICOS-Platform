"""
Tests for Historical Decision Backtest and Permutation Test.
These tests require outputs/modeling_dataset.csv which is generated locally
and not tracked in git (11 MB). They are skipped when the dataset is absent.
"""
import os
import pytest
from ml.evaluation.decision_backtest import DecisionBacktestEngine
from tests.evaluation.run_permutation_test import run_permutation_test

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "modeling_dataset.csv")
if not os.path.isfile(DATASET_PATH):
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeling_dataset.csv")
_has_dataset = os.path.isfile(DATASET_PATH)

@pytest.mark.skipif(not _has_dataset, reason="outputs/modeling_dataset.csv not available (local-only artifact)")
def test_decision_backtest_execution():
    dbe = DecisionBacktestEngine()
    res = dbe.run_backtest(asset_type="PANAMAX_1D", sample_stride=100)
    assert "samples_evaluated" in res
    assert "avg_ficos_cost_usd" in res
    assert "savings_percentage" in res

@pytest.mark.skipif(not _has_dataset, reason="outputs/modeling_dataset.csv not available (local-only artifact)")
def test_permutation_test_execution():
    res = run_permutation_test(B=10, method="circular_shift")
    assert res["B"] == 10
    assert "p_value" in res
    assert "baseline_accuracy" in res
