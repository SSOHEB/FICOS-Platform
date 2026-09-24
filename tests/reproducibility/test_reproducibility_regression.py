"""
FICOS — Reproducibility & Provenance Regression Test
====================================================
Runs the canonical 100-tree evaluation pipeline twice from a clean state and asserts:
  1. Provenance and dataset SHA256 integrity
  2. Identical evaluation DataFrame shape
  3. Identical predictions (max absolute diff == 0.0)
  4. Identical gate decisions (0 decision mismatches)
  5. Identical retained observation count (641 for RF 1D)
  6. Identical portfolio economic totals (-$503,745.00 for RF 1D)
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.config.canonical_config import (
    CANONICAL_N_TREES,
    CANONICAL_SEED,
    CANONICAL_N_JOBS,
    CANONICAL_DATASET_SHA256,
    CANONICAL_COST_MODEL,
    CANONICAL_GATING_POLICY,
    validate_dataset_provenance,
    validate_hyperparameters,
    validate_cost_model,
    validate_gating_policy,
    assert_authoritative_certification,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    def CatBoostRegressor(iterations=100, depth=5, learning_rate=0.03, random_seed=42, **kw):
        return _GBR(n_estimators=iterations, max_depth=depth, learning_rate=learning_rate, random_state=random_seed)


def test_canonical_provenance_and_hyperparameters():
    """Assert provenance, SHA256, and hyperparameters comply with SSOT config."""
    dataset_ok, dataset_sha = validate_dataset_provenance(_PROJECT_ROOT / "data" / "modeling_dataset.csv")
    assert dataset_ok is True
    assert dataset_sha == CANONICAL_DATASET_SHA256

    hp_ok = validate_hyperparameters(CANONICAL_N_TREES, CANONICAL_SEED, CANONICAL_N_JOBS)
    assert hp_ok is True

    cost_ok = validate_cost_model(CANONICAL_COST_MODEL)
    assert cost_ok is True

    gating_ok = validate_gating_policy(CANONICAL_GATING_POLICY)
    assert gating_ok is True

    cert = assert_authoritative_certification(dataset_sha, CANONICAL_N_TREES, CANONICAL_SEED, CANONICAL_N_JOBS)
    assert "CERTIFIED" in cert


def _run_canonical_eval_pipeline():
    """Helper executing 1D fold evaluation clean-room run."""
    DATA_PATH = _PROJECT_ROOT / "data" / "modeling_dataset.csv"
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw.sort_values('date').reset_index(drop=True)

    vessels = ['panamax', 'supramax', 'handy', 'cape']
    horizons = [1]
    feature_cols = [c for c in df_raw.columns if c not in ["date"] and not c.startswith("target_") and not c.startswith("dir_")]

    FOLDS = [
        {"year": 2021, "train_end": "2019-12-24", "val_start": "2020-01-03", "val_end": "2020-12-24", "test_start": "2021-01-05", "test_end": "2021-12-31"},
        {"year": 2022, "train_end": "2020-12-24", "val_start": "2021-01-05", "val_end": "2021-12-24", "test_start": "2022-01-03", "test_end": "2022-12-30"},
        {"year": 2023, "train_end": "2021-12-24", "val_start": "2022-01-03", "val_end": "2022-12-23", "test_start": "2023-01-03", "test_end": "2023-12-29"},
        {"year": 2024, "train_end": "2022-12-23", "val_start": "2023-01-03", "val_end": "2023-12-22", "test_start": "2024-01-02", "test_end": "2024-12-31"},
        {"year": 2025, "train_end": "2023-12-22", "val_start": "2024-01-02", "val_end": "2024-12-24", "test_start": "2025-01-02", "test_end": "2025-12-31"},
    ]

    records = []
    np.random.seed(CANONICAL_SEED)

    for fold in FOLDS:
        for v in vessels:
            rate_col = v
            for h in horizons:
                tgt_col = f"target_{v}_{h}d"
                if tgt_col not in df_raw.columns or rate_col not in df_raw.columns:
                    continue

                valid_row = df_raw[rate_col].notnull() & df_raw[tgt_col].notnull()

                tr_mask  = (df_raw['date'] <= fold['train_end']) & valid_row
                val_mask = (df_raw['date'] >= fold['val_start']) & (df_raw['date'] <= fold['val_end']) & valid_row
                te_mask  = (df_raw['date'] >= fold['test_start']) & (df_raw['date'] <= fold['test_end']) & valid_row

                X_tr  = df_raw.loc[tr_mask, feature_cols].fillna(0.0).values
                y_tr  = df_raw.loc[tr_mask, tgt_col].values - df_raw.loc[tr_mask, rate_col].values

                X_val = df_raw.loc[val_mask, feature_cols].fillna(0.0).values
                y_val = df_raw.loc[val_mask, tgt_col].values - df_raw.loc[val_mask, rate_col].values

                X_te  = df_raw.loc[te_mask, feature_cols].fillna(0.0).values
                y_te  = df_raw.loc[te_mask, tgt_col].values - df_raw.loc[te_mask, rate_col].values

                scaler = StandardScaler()
                X_tr_sc  = scaler.fit_transform(X_tr)
                X_val_sc = scaler.transform(X_val)
                X_te_sc  = scaler.transform(X_te)

                selector = SelectKBest(f_regression, k=min(30, X_tr_sc.shape[1]))
                X_tr_fit  = selector.fit_transform(X_tr_sc, y_tr)
                X_val_fit = selector.transform(X_val_sc)
                X_te_fit  = selector.transform(X_te_sc)

                # RF_STANDARD canonical model
                model = RandomForestRegressor(
                    n_estimators=CANONICAL_N_TREES,
                    max_depth=5,
                    random_state=CANONICAL_SEED,
                    n_jobs=CANONICAL_N_JOBS
                )
                model.fit(X_tr_fit, y_tr)

                val_preds_delta = model.predict(X_val_fit)
                te_preds_delta  = model.predict(X_te_fit)

                val_residuals = y_val - val_preds_delta
                p10_bound = float(np.percentile(val_residuals, 10))
                p90_bound = float(np.percentile(val_residuals, 90))

                te_dates = df_raw.loc[te_mask, 'date'].dt.strftime('%Y-%m-%d').values
                te_base  = df_raw.loc[te_mask, rate_col].values
                te_actual_abs = df_raw.loc[te_mask, tgt_col].values
                te_true_delta = te_actual_abs - te_base

                for d, b, p_delta, t_delta, act_abs in zip(te_dates, te_base, te_preds_delta, te_true_delta, te_actual_abs):
                    pct_delta = p_delta / (b + 1e-8)
                    is_buy  = (p_delta > p90_bound) and (pct_delta > CANONICAL_GATING_POLICY["tau"])
                    is_wait = (p_delta < p10_bound) and (pct_delta < -CANONICAL_GATING_POLICY["tau"])
                    dec = 'NOW' if is_buy else ('WAIT' if is_wait else 'FLEXIBLE')

                    records.append({
                        'date': d, 'vessel': v, 'horizon': h, 'fold': fold['year'],
                        'base': b, 'pred_delta': p_delta, 'true_delta': t_delta,
                        'decision': dec, 'retained': dec in ('NOW', 'WAIT')
                    })

    return pd.DataFrame(records)


def test_reproducibility_dual_execution():
    """Run pipeline twice clean and assert 100% bitwise & numerical equality."""
    df1 = _run_canonical_eval_pipeline()
    df2 = _run_canonical_eval_pipeline()

    # 1. Shape equality
    assert df1.shape == df2.shape

    # 2. Prediction equality
    max_pred_diff = np.max(np.abs(df1['pred_delta'].values - df2['pred_delta'].values))
    assert max_pred_diff == 0.0

    # 3. Decision equality
    dec_mismatches = np.sum(df1['decision'].values != df2['decision'].values)
    assert dec_mismatches == 0

    # 4. Retained N equality
    retained_1 = int(df1['retained'].sum())
    retained_2 = int(df2['retained'].sum())
    assert retained_1 == retained_2 == 641

    # 5. Economic totals calculation
    def _calc_net(df):
        VOYAGE = CANONICAL_COST_MODEL["voyage_duration"]
        IDLE = CANONICAL_COST_MODEL["daily_idle"]
        spot_cost = df['base'] * VOYAGE
        wait_cost = (df['base'] + df['true_delta']) * VOYAGE + IDLE * CANONICAL_COST_MODEL["wait_idle_days"]
        flex_cost = 0.5 * (df['base'] + df['base'] + df['true_delta']) * VOYAGE + IDLE * CANONICAL_COST_MODEL["flex_idle_days"]

        ficos_cost = np.where(df['decision'] == 'NOW', spot_cost,
                              np.where(df['decision'] == 'WAIT', wait_cost, flex_cost))

        return float(spot_cost.sum() - ficos_cost.sum())

    net1 = _calc_net(df1)
    net2 = _calc_net(df2)

    assert abs(net1 - net2) < 1e-4
    assert abs(net1 - (-503745.0)) < 1e-4
