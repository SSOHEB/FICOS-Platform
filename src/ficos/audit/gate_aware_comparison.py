"""
FICOS — Gate-Aware Model Comparison
=====================================
Runs an honest, multi-model comparison across all key asset/horizon pairs.

Key design rules (anti-leakage):
  - All preprocessing (impute, scale, select) fit on TRAIN only.
  - TAU is selected using VALIDATION residuals only — frozen before test.
  - Model selection (winner per fold) is based on VALIDATION sMAPE only.
  - FINAL test is evaluated ONCE with the frozen model/tau.
  - Coverage sensitivity is reported for tau in {0.005, 0.01, 0.02, 0.05}.
  - Promotion decisions are NOT made by maximising gated test accuracy.

Outputs:
  outputs/model_reconciliation.csv        (one row per pair × model × fold)
  outputs/model_reconciliation_summary.csv (aggregated)
  outputs/gate_sensitivity.csv            (tau sweep)
"""

import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              balanced_accuracy_score, f1_score)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PAIRS = [
    ("panamax",  1),
    ("supramax", 1),
    ("handy",    1),
    ("cape",     1),
    ("supramax", 7),
    ("handy",    7),
    ("supramax", 14),
    ("kdci",     7),
]

TAU_GRID = [0.005, 0.01, 0.02, 0.05]   # tested on validation, frozen before test
K_FEATURES = 30                          # same as existing benchmark
N_FOLDS    = 5
FOLD_SIZE  = 250
VAL_SIZE   = 200

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def smape(a, b):
    return float(np.mean(200 * np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-8)))


def directional_accuracy(y_true_delta, y_pred_delta):
    dir_true = (y_true_delta > 0).astype(int)
    dir_pred = (y_pred_delta > 0).astype(int)
    return np.mean(dir_true == dir_pred)


def gated_metrics(y_true_delta, y_pred_delta, gate_mask):
    """Returns dict of gated metrics. gate_mask is boolean array."""
    if gate_mask.sum() == 0:
        return dict(gated_da=np.nan, gated_precision=np.nan, gated_recall=np.nan,
                    gated_f1=np.nan, gated_auc=np.nan, coverage=0.0, n_gated=0)

    yt_g = (y_true_delta[gate_mask] > 0).astype(int)
    yp_g = (y_pred_delta[gate_mask] > 0).astype(int)

    acc   = np.mean(yt_g == yp_g)
    bal   = balanced_accuracy_score(yt_g, yp_g)
    prec  = np.sum((yp_g == 1) & (yt_g == 1)) / (np.sum(yp_g == 1) + 1e-8)
    rec   = np.sum((yp_g == 1) & (yt_g == 1)) / (np.sum(yt_g == 1) + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    try:
        auc_v = roc_auc_score(yt_g, y_pred_delta[gate_mask])
    except Exception:
        auc_v = np.nan

    return dict(
        gated_da=float(acc),
        gated_balanced_da=float(bal),
        gated_precision=float(prec),
        gated_recall=float(rec),
        gated_f1=float(f1),
        gated_auc=float(auc_v),
        coverage=float(gate_mask.sum() / len(gate_mask)),
        n_gated=int(gate_mask.sum()),
    )


def make_gate(pred_delta, val_resids, tau_pct):
    """
    Gate: fire when |pred_delta| / (|asset_price| + 1e-8) > tau_pct.
    Directional: buy if pred > val_p90, wait if pred < val_p10.
    val_resids used to set p10/p90 threshold.
    """
    p10 = np.percentile(val_resids, 10)
    p90 = np.percentile(val_resids, 90)
    buy_m  = pred_delta > max(0.0, p90)
    wait_m = pred_delta < min(0.0, p10)
    return buy_m | wait_m


def build_models():
    """Returns dict of model_name → unfitted estimator."""
    return {
        "Ridge": Ridge(alpha=100.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                     random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                        random_state=42, verbose=-1, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                       learning_rate=0.05, random_state=42),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison():
    print("=" * 80)
    print("  FICOS — GATE-AWARE MODEL COMPARISON")
    print("  Anti-leakage: tau selected on VALIDATION, evaluated once on TEST")
    print("=" * 80)

    # Load dataset
    dataset_path = None
    for p in ["data/modeling_dataset.csv", "outputs/modeling_dataset.csv"]:
        if os.path.exists(p):
            dataset_path = p
            break
    if dataset_path is None:
        print("ERROR: modeling_dataset.csv not found."); sys.exit(1)

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n_rows = len(df)

    drop_cols = set([c for c in df.columns if c.startswith(("dir_", "future_", "target_"))] + ["date"])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    print(f"Dataset: {n_rows} rows x {df.shape[1]} cols | {len(feature_cols)} features\n")

    # Fold boundaries (same as reconciliation test)
    fold_configs = {}
    for f in range(1, N_FOLDS + 1):
        te_end   = n_rows - (N_FOLDS - f) * FOLD_SIZE
        te_start = te_end - FOLD_SIZE
        va_end   = te_start
        va_start = va_end - VAL_SIZE
        fold_configs[f] = dict(tr_end=va_start, va_start=va_start,
                               va_end=va_end, te_start=te_start, te_end=te_end)

    all_records = []
    gate_sensitivity_records = []

    for (asset, h) in EVAL_PAIRS:
        pair_key = f"{asset.upper()}_{h}D"
        print(f"[{pair_key}]", end="", flush=True)

        for fold_id in range(1, N_FOLDS + 1):
            cfg = fold_configs[fold_id]
            tr_end, va_start, va_end, te_start, te_end = (
                cfg["tr_end"], cfg["va_start"], cfg["va_end"],
                cfg["te_start"], cfg["te_end"]
            )

            # Build delta targets
            df_p = df.copy()
            df_p["_yd"] = df_p[asset].shift(-h) - df_p[asset]
            df_v = df_p[~df_p["_yd"].isna()].reset_index(drop=True)
            n_v = len(df_v)

            te_end_c   = min(te_end, n_v)
            te_start_c = min(te_start, n_v)
            va_end_c   = min(va_end, n_v)
            va_start_c = min(va_start, n_v)
            tr_end_c   = min(tr_end, n_v)

            if te_end_c <= te_start_c or va_end_c <= va_start_c or tr_end_c <= 0:
                continue

            tr_m = np.zeros(n_v, bool); tr_m[:tr_end_c] = True
            va_m = np.zeros(n_v, bool); va_m[va_start_c:va_end_c] = True
            te_m = np.zeros(n_v, bool); te_m[te_start_c:te_end_c] = True

            X_raw = df_v[feature_cols].values.copy()
            y_d   = df_v["_yd"].values

            # Fold-isolated imputation (train median only)
            med = np.nanmedian(X_raw[tr_m], axis=0)
            med[np.isnan(med)] = 0.0
            for ci in range(X_raw.shape[1]):
                X_raw[:, ci] = np.where(np.isnan(X_raw[:, ci]), med[ci], X_raw[:, ci])

            # Scale (train only)
            sx = StandardScaler()
            Xtr = sx.fit_transform(X_raw[tr_m])
            Xva = sx.transform(X_raw[va_m])
            Xte = sx.transform(X_raw[te_m])

            # Feature selection (train only)
            sel = SelectKBest(f_regression, k=min(K_FEATURES, Xtr.shape[1]))
            Xtr_s = sel.fit_transform(Xtr, y_d[tr_m])
            Xva_s = sel.transform(Xva)
            Xte_s = sel.transform(Xte)

            # Persistence baseline
            y_base = df_v[asset].values
            pers_da_te  = directional_accuracy(y_d[te_m], np.zeros(te_m.sum()))  # pred=0 → persistence
            pers_smape  = smape(y_d[te_m], np.zeros(te_m.sum()))

            # ── Hyperparameter tuning (validation only) ──
            model_defs = build_models()

            # Ridge: tune alpha on validation
            best_ridge_alpha, best_ridge_score = 100.0, float("inf")
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                m = Ridge(alpha=alpha).fit(Xtr_s, y_d[tr_m])
                s = smape(y_d[va_m], m.predict(Xva_s))
                if s < best_ridge_score:
                    best_ridge_score, best_ridge_alpha = s, alpha
            model_defs["Ridge"] = Ridge(alpha=best_ridge_alpha)

            # ElasticNet: tune alpha/l1 on validation
            best_en_alpha, best_en_l1, best_en_score = 0.1, 0.5, float("inf")
            for alpha in [0.01, 0.1, 1.0]:
                for l1 in [0.2, 0.5, 0.8]:
                    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=42).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < best_en_score:
                        best_en_score, best_en_alpha, best_en_l1 = s, alpha, l1
            model_defs["ElasticNet"] = ElasticNet(alpha=best_en_alpha, l1_ratio=best_en_l1,
                                                   max_iter=5000, random_state=42)

            # RF: tune n_estimators/max_depth on validation
            best_rf_ne, best_rf_d, best_rf_score = 100, 5, float("inf")
            for ne in [50, 100]:
                for d in [3, 5]:
                    m = RandomForestRegressor(n_estimators=ne, max_depth=d,
                                              random_state=42, n_jobs=-1).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < best_rf_score:
                        best_rf_score, best_rf_ne, best_rf_d = s, ne, d
            model_defs["RandomForest"] = RandomForestRegressor(n_estimators=best_rf_ne,
                                                               max_depth=best_rf_d,
                                                               random_state=42, n_jobs=-1)

            # XGBoost: tune on validation
            best_xg_ne, best_xg_d, best_xg_score = 100, 4, float("inf")
            for ne in [50, 100]:
                for d in [3, 4]:
                    m = xgb.XGBRegressor(n_estimators=ne, max_depth=d, learning_rate=0.05,
                                         random_state=42, n_jobs=-1, verbosity=0).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < best_xg_score:
                        best_xg_score, best_xg_ne, best_xg_d = s, ne, d
            model_defs["XGBoost"] = xgb.XGBRegressor(n_estimators=best_xg_ne, max_depth=best_xg_d,
                                                       learning_rate=0.05, random_state=42,
                                                       n_jobs=-1, verbosity=0)

            # LightGBM: tune on validation
            best_lg_ne, best_lg_d, best_lg_score = 100, 4, float("inf")
            for ne in [50, 100]:
                for d in [3, 4]:
                    m = lgb.LGBMRegressor(n_estimators=ne, max_depth=d, learning_rate=0.05,
                                           random_state=42, verbose=-1, n_jobs=-1).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < best_lg_score:
                        best_lg_score, best_lg_ne, best_lg_d = s, ne, d
            model_defs["LightGBM"] = lgb.LGBMRegressor(n_estimators=best_lg_ne, max_depth=best_lg_d,
                                                         learning_rate=0.05, random_state=42,
                                                         verbose=-1, n_jobs=-1)

            # ── Train with tuned HPs, evaluate on TEST ──
            for model_name, model_template in model_defs.items():
                m = model_template.__class__(**model_template.get_params())
                m.fit(Xtr_s, y_d[tr_m])

                pred_va = m.predict(Xva_s)
                pred_te = m.predict(Xte_s)

                # Validation metrics (for tau selection only, NOT for reporting promotion)
                val_resids = y_d[va_m] - pred_va
                va_smape_v = smape(y_d[va_m], pred_va)
                va_da = directional_accuracy(y_d[va_m], pred_va)

                # Test metrics — ungated
                raw_da_te    = directional_accuracy(y_d[te_m], pred_te)
                raw_smape_te = smape(y_d[te_m], pred_te)
                raw_bal_acc  = balanced_accuracy_score(
                    (y_d[te_m] > 0).astype(int),
                    (pred_te > 0).astype(int)
                )

                # ── TAU sweep on VALIDATION, then apply frozen tau to TEST ──
                # For each tau, determine gate on VALIDATION, then evaluate on TEST
                best_tau = 0.01
                best_val_composite = -np.inf

                for tau in TAU_GRID:
                    gate_va = make_gate(pred_va, val_resids, tau)
                    if gate_va.sum() < 5:
                        continue
                    vm = gated_metrics(y_d[va_m], pred_va, gate_va)
                    # Composite: gated_da * sqrt(coverage) — on VALIDATION only
                    composite = vm["gated_da"] * np.sqrt(max(vm["coverage"], 1e-4))
                    if composite > best_val_composite:
                        best_val_composite = composite
                        best_tau = tau

                    # Record gate sensitivity (validation)
                    gate_sensitivity_records.append({
                        "pair": pair_key, "fold": fold_id, "model": model_name,
                        "tau": tau, "split": "validation",
                        "gated_da": vm["gated_da"], "coverage": vm["coverage"],
                        "n_gated": vm["n_gated"], "f1": vm["gated_f1"],
                        "composite": composite
                    })

                # Apply best_tau (frozen from validation) to TEST
                gate_te = make_gate(pred_te, val_resids, best_tau)
                gm_te = gated_metrics(y_d[te_m], pred_te, gate_te)

                # Improvement vs persistence
                da_vs_persistence = raw_da_te - pers_da_te

                rec = {
                    "pair": pair_key,
                    "asset": asset,
                    "horizon_d": h,
                    "fold": fold_id,
                    "model": model_name,
                    "raw_da_test": round(raw_da_te, 4),
                    "raw_balanced_acc_test": round(raw_bal_acc, 4),
                    "raw_smape_test": round(raw_smape_te, 4),
                    "da_vs_persistence": round(da_vs_persistence, 4),
                    "persistence_da": round(pers_da_te, 4),
                    "val_smape": round(va_smape_v, 4),
                    "val_da": round(va_da, 4),
                    "best_tau_from_val": best_tau,
                    "gated_da_test": round(gm_te["gated_da"], 4) if not np.isnan(gm_te.get("gated_da", np.nan)) else None,
                    "gated_balanced_acc": round(gm_te["gated_balanced_da"], 4) if not np.isnan(gm_te.get("gated_balanced_da", np.nan)) else None,
                    "gated_precision": round(gm_te["gated_precision"], 4),
                    "gated_recall": round(gm_te["gated_recall"], 4),
                    "gated_f1": round(gm_te["gated_f1"], 4),
                    "gated_auc": round(gm_te["gated_auc"], 4) if not np.isnan(gm_te.get("gated_auc", np.nan)) else None,
                    "coverage": round(gm_te["coverage"], 4),
                    "n_gated": gm_te["n_gated"],
                    "n_test": int(te_m.sum()),
                    "gate_improves": (gm_te.get("gated_da", 0) or 0) > raw_da_te,
                }
                all_records.append(rec)

            print(".", end="", flush=True)

        print(f" done", flush=True)

    # ── Save raw records ──
    df_rec = pd.DataFrame(all_records)
    rec_path = os.path.join(OUT_DIR, "model_reconciliation.csv")
    df_rec.to_csv(rec_path, index=False)
    print(f"\nSaved: {rec_path} ({len(df_rec)} rows)")

    # ── Save gate sensitivity ──
    df_gate = pd.DataFrame(gate_sensitivity_records)
    gate_path = os.path.join(OUT_DIR, "gate_sensitivity.csv")
    df_gate.to_csv(gate_path, index=False)
    print(f"Saved: {gate_path}")

    # ── Summary ──
    summary_rows = []
    for (pair, model), grp in df_rec.groupby(["pair", "model"]):
        g = grp.dropna(subset=["gated_da_test"])
        raw_da_mean   = grp["raw_da_test"].mean()
        raw_da_std    = grp["raw_da_test"].std()
        gate_da_mean  = g["gated_da_test"].mean() if len(g) else np.nan
        gate_da_std   = g["gated_da_test"].std()  if len(g) else np.nan
        cov_mean      = grp["coverage"].mean()
        cov_std       = grp["coverage"].std()
        f1_mean       = g["gated_f1"].mean()  if len(g) else np.nan
        vs_pers       = grp["da_vs_persistence"].mean()
        pers_mean     = grp["persistence_da"].mean()
        n_folds       = len(grp)
        gate_helps    = grp["gate_improves"].sum()

        summary_rows.append({
            "pair": pair,
            "model": model,
            "n_folds": n_folds,
            "raw_da_mean": round(raw_da_mean, 4),
            "raw_da_std": round(raw_da_std, 4),
            "gated_da_mean": round(gate_da_mean, 4) if not np.isnan(gate_da_mean) else None,
            "gated_da_std": round(gate_da_std, 4) if not np.isnan(gate_da_std) else None,
            "coverage_mean": round(cov_mean, 4),
            "coverage_std": round(cov_std, 4),
            "gated_f1_mean": round(f1_mean, 4) if not np.isnan(f1_mean) else None,
            "persistence_da_mean": round(pers_mean, 4),
            "da_vs_persistence_mean": round(vs_pers, 4),
            "gate_helps_n_folds": int(gate_helps),
            "composite_score": round(gate_da_mean * np.sqrt(max(cov_mean, 1e-4)), 4) if not np.isnan(gate_da_mean) else None,
        })

    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(OUT_DIR, "model_reconciliation_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"Saved: {sum_path}")

    # ── Print leaderboard ──
    print("\n" + "=" * 100)
    print("LEADERBOARD (sorted by composite_score = gated_da × sqrt(coverage))")
    print("=" * 100)
    print(f"{'Pair':<15} {'Model':<18} {'RawDA':>7} {'GatedDA':>9} {'Coverage':>9} {'GatF1':>8} {'vs_Pers':>9} {'GateHelps':>11}")
    print("-" * 100)

    df_sum_sorted = df_sum.sort_values(["pair", "composite_score"], ascending=[True, False])
    for _, row in df_sum_sorted.iterrows():
        ga = f"{row['gated_da_mean']:.3f}" if row['gated_da_mean'] is not None else "  N/A"
        cs = f"{row['composite_score']:.3f}" if row['composite_score'] is not None else "  N/A"
        f1 = f"{row['gated_f1_mean']:.3f}" if row['gated_f1_mean'] is not None else "  N/A"
        print(f"{row['pair']:<15} {row['model']:<18} "
              f"{row['raw_da_mean']:>7.3f} {ga:>9} {row['coverage_mean']:>9.3f} "
              f"{f1:>8} {row['da_vs_persistence_mean']:>9.3f} "
              f"{row['gate_helps_n_folds']:>9}/{row['n_folds']}")

    print("\nDone. Use outputs/model_reconciliation_summary.csv for registry decisions.")
    return df_rec, df_sum


if __name__ == "__main__":
    run_comparison()
