"""
Tests for Historical Decision Backtest and Permutation Test.
"""
from src.evaluation.decision_backtest import DecisionBacktestEngine
from tests.run_permutation_test import run_permutation_test

def test_decision_backtest_execution():
    dbe = DecisionBacktestEngine()
    res = dbe.run_backtest(asset_type="PANAMAX_1D", sample_stride=100)
    assert "samples_evaluated" in res
    assert "avg_ficos_cost_usd" in res
    assert "savings_percentage" in res

def test_permutation_test_execution():
    res = run_permutation_test(B=10, method="circular_shift")
    assert res["B"] == 10
    assert "p_value" in res
    assert "baseline_accuracy" in res
