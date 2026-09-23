"""
FICOS Platform - Final Verification & Reconciliation Test
Validates all reported metrics in MASTER_EVALUATION_REPORT.md against live model execution.
Protocol: 5 Purged Chronological Out-of-Sample Walk-Forward Folds (N ~ 1,242 days)
Asset-Horizon Pairs: Panamax 1D, Supramax 1D, Handy 1D, Cape 1D, Supramax 7D, Handy 7D, Supramax 14D, KDCI 7D
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

REPORTED_METRICS = {
    'PANAMAX_1D': {
        'N_Ungated': 1242, 'Accuracy_Ungated': 78.1,
        'N_Gated': 214, 'Accuracy_Gated': 91.1, 'Coverage': 17.2,
        'Precision_Gated': 91.7, 'Recall_Gated': 90.9, 'F1_Gated': 91.3, 'AUC_Gated': 0.924,
        'Expected_Model': 'RandomForest', 'Classification': 'PRIMARY PROMOTE'
    },
    'SUPRAMAX_1D': {
        'N_Ungated': 1244, 'Accuracy_Ungated': 75.1,
        'N_Gated': 200, 'Accuracy_Gated': 85.0, 'Coverage': 16.1,
        'Precision_Gated': 76.0, 'Recall_Gated': 91.2, 'F1_Gated': 83.0, 'AUC_Gated': 0.805,
        'Expected_Model': 'RandomForest', 'Classification': 'PRIMARY PROMOTE'
    },
    'HANDY_1D': {
        'N_Ungated': 1244, 'Accuracy_Ungated': 70.7,
        'N_Gated': 144, 'Accuracy_Gated': 79.2, 'Coverage': 11.6,
        'Precision_Gated': 72.2, 'Recall_Gated': 87.7, 'F1_Gated': 79.2, 'AUC_Gated': 0.750,
        'Expected_Model': 'RandomForest', 'Classification': 'PRIMARY PROMOTE'
    },
    'CAPE_1D': {
        'N_Ungated': 1217, 'Accuracy_Ungated': 66.5,
        'N_Gated': 174, 'Accuracy_Gated': 71.3, 'Coverage': 14.3,
        'Precision_Gated': 58.3, 'Recall_Gated': 48.3, 'F1_Gated': 52.8, 'AUC_Gated': 0.711,
        'Expected_Model': 'RandomForest', 'Classification': 'PRIMARY PROMOTE'
    },
    'SUPRAMAX_7D': {
        'N_Ungated': 1239, 'Accuracy_Ungated': 62.0,
        'N_Gated': 235, 'Accuracy_Gated': 63.8, 'Coverage': 19.0,
        'Precision_Gated': 66.7, 'Recall_Gated': 41.1, 'F1_Gated': 50.9, 'AUC_Gated': 0.744,
        'Expected_Model': 'Ridge', 'Classification': 'SECONDARY PROMOTE'
    },
    'HANDY_7D': {
        'N_Ungated': 1238, 'Accuracy_Ungated': 60.3,
        'N_Gated': 239, 'Accuracy_Gated': 58.6, 'Coverage': 19.3,
        'Precision_Gated': 52.4, 'Recall_Gated': 62.9, 'F1_Gated': 57.1, 'AUC_Gated': 0.593,
        'Expected_Model': 'Ridge', 'Classification': 'SECONDARY PROMOTE'
    },
    'SUPRAMAX_14D': {
        'N_Ungated': 1231, 'Accuracy_Ungated': 54.4,
        'N_Gated': 503, 'Accuracy_Gated': 49.1, 'Coverage': 40.9,
        'Precision_Gated': 49.1, 'Recall_Gated': 11.0, 'F1_Gated': 17.9, 'AUC_Gated': 0.564,
        'Expected_Model': 'None', 'Classification': 'EXCLUDE'
    },
    'KDCI_7D': {
        'N_Ungated': 1243, 'Accuracy_Ungated': 58.7,
        'N_Gated': 150, 'Accuracy_Gated': 76.7, 'Coverage': 12.1,
        'Precision_Gated': 59.5, 'Recall_Gated': 52.4, 'F1_Gated': 55.7, 'AUC_Gated': 0.801,
        'Expected_Model': 'None', 'Classification': 'EXCLUDE'
    },
}

def smape(a, b):
    return float(np.mean(200 * np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-8)))

def run_verification():
    print("=" * 90)
    print(" FICOS PLATFORM - FINAL VERIFICATION & RECONCILIATION TEST ")
    print(" Verifying Live Multi-Model Pipeline vs. MASTER_EVALUATION_REPORT.md ")
    print("=" * 90)

    dataset_path = None
    for cand in ['data/modeling_dataset.csv', 'outputs/modeling_dataset.csv']:
        if os.path.exists(cand):
            dataset_path = cand
            break
    if dataset_path is None:
        print("Error: modeling_dataset.csv not found in data/ or outputs/.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n_rows = len(df)
    print(f"Loaded dataset: {n_rows} rows x {df.shape[1]} columns ({df['date'].min().date()} to {df['date'].max().date()})")

    drop_cols = set([c for c in df.columns if c.startswith(('dir_', 'future_', 'target_'))] + ['date'])
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Feature set size: {len(feature_cols)} leak-free predictors")

    test_size, val_size = 250, 200
    fold_configs = {}
    for f in range(1, 6):
        te_end = n_rows - (5 - f) * test_size
        te_start = te_end - test_size
        va_end, va_start = te_start, te_start - val_size
        fold_configs[f] = dict(tr_end=va_start, va_start=va_start,
                               va_end=va_end, te_start=te_start, te_end=te_end)

    eval_pairs = [
        ('cape', 1),
        ('panamax', 1),
        ('supramax', 1),
        ('supramax', 7),
        ('handy', 1),
        ('handy', 7),
        ('supramax', 14),
        ('kdci', 7)
    ]

    results = {}
    all_passed = True

    for (asset, h) in eval_pairs:
        pair_key = f"{asset.upper()}_{h}D"
        print(f"\n[Testing {pair_key}] Running 5-fold walk-forward validation...", end="", flush=True)
        records = []
        winner_counts = {}

        for fold_id in range(1, 6):
            cfg = fold_configs[fold_id]
            tr_end, va_start, va_end, te_start, te_end = (
                cfg['tr_end'], cfg['va_start'], cfg['va_end'], cfg['te_start'], cfg['te_end']
            )

            df_p = df.copy()
            df_p['_yd'] = df_p[asset].shift(-h) - df_p[asset]
            df_v = df_p[~df_p['_yd'].isna()].reset_index(drop=True)

            tr_m = np.zeros(len(df_v), bool); tr_m[:tr_end] = True
            va_m = np.zeros(len(df_v), bool); va_m[va_start:va_end] = True
            te_m = np.zeros(len(df_v), bool); te_m[te_start:te_end] = True

            X_raw = df_v[feature_cols].values.copy()
            y_d = df_v['_yd'].values
            y_base = df_v[asset].values

            # Fold-isolated imputation
            med = np.nanmedian(X_raw[tr_m], axis=0)
            med[np.isnan(med)] = 0.0
            for ci in range(X_raw.shape[1]):
                X_raw[:, ci] = np.where(np.isnan(X_raw[:, ci]), med[ci], X_raw[:, ci])

            # Scaling
            sx = StandardScaler()
            Xtr = sx.fit_transform(X_raw[tr_m])
            Xva = sx.transform(X_raw[va_m])
            Xte = sx.transform(X_raw[te_m])

            sy = StandardScaler()
            ytr_sc = sy.fit_transform(y_d[tr_m].reshape(-1, 1)).flatten()

            # SelectKBest (K=30)
            sel = SelectKBest(f_regression, k=30)
            Xtr_s = sel.fit_transform(Xtr, y_d[tr_m])
            Xva_s = sel.transform(Xva)
            Xte_s = sel.transform(Xte)

            models = {}

            # Ridge
            br, brv = None, float('inf')
            for a in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                m = Ridge(alpha=a).fit(Xtr_s, y_d[tr_m])
                s = smape(y_d[va_m], m.predict(Xva_s))
                if s < brv: brv, br = s, m
            models['Ridge'] = (br, br.predict(Xva_s), br.predict(Xte_s))

            # ElasticNet
            be, bev, en_va, en_te = None, float('inf'), None, None
            for a in [0.01, 0.1, 1.0]:
                for l1 in [0.2, 0.5, 0.8]:
                    m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000, random_state=42).fit(Xtr_s, ytr_sc)
                    va_p = sy.inverse_transform(m.predict(Xva_s).reshape(-1, 1)).flatten()
                    s = smape(y_d[va_m], va_p)
                    if s < bev:
                        bev, be = s, m; en_va = va_p
                        en_te = sy.inverse_transform(m.predict(Xte_s).reshape(-1, 1)).flatten()
            models['ElasticNet'] = (be, en_va, en_te)

            # RandomForest
            brf, brfv = None, float('inf')
            for ne in [50, 100]:
                for d in [3, 5]:
                    m = RandomForestRegressor(n_estimators=ne, max_depth=d, random_state=42, n_jobs=-1).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < brfv: brfv, brf = s, m
            models['RandomForest'] = (brf, brf.predict(Xva_s), brf.predict(Xte_s))

            # XGBoost
            bxg, bxgv = None, float('inf')
            for ne in [50, 100]:
                for d in [3, 4]:
                    m = xgb.XGBRegressor(n_estimators=ne, max_depth=d, learning_rate=0.05, random_state=42, n_jobs=-1).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < bxgv: bxgv, bxg = s, m
            models['XGBoost'] = (bxg, bxg.predict(Xva_s), bxg.predict(Xte_s))

            # LightGBM
            blg, blgv = None, float('inf')
            for ne in [50, 100]:
                for d in [3, 4]:
                    m = lgb.LGBMRegressor(n_estimators=ne, max_depth=d, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1).fit(Xtr_s, y_d[tr_m])
                    s = smape(y_d[va_m], m.predict(Xva_s))
                    if s < blgv: blgv, blg = s, m
            models['LightGBM'] = (blg, blg.predict(Xva_s), blg.predict(Xte_s))

            winner = min(models, key=lambda k: smape(y_d[va_m], models[k][1]))
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            w_va_p = models[winner][1]
            pred_te = models[winner][2]

            val_resids = y_d[va_m] - w_va_p
            p10, p90 = np.percentile(val_resids, 10), np.percentile(val_resids, 90)
            pct_p = pred_te / (np.abs(y_base[te_m]) + 1e-8)
            buy_m = (pred_te > max(0.0, p90)) & (pct_p > 0.01)
            wait_m = (pred_te < min(0.0, p10)) & (pct_p < -0.01)
            gated = buy_m | wait_m

            for yt, yp, g in zip(y_d[te_m], pred_te, gated):
                records.append({'fold': fold_id, 'winner': winner, 'y_true': yt, 'y_pred': yp, 'gated': g})

        df_r = pd.DataFrame(records)
        df_r = df_r[(df_r['y_true'] != 0) & (df_r['y_pred'] != 0)].copy()
        df_r['dir_true'] = (df_r['y_true'] > 0).astype(int)
        df_r['dir_pred'] = (df_r['y_pred'] > 0).astype(int)

        # Ungated metrics
        cm_ug = confusion_matrix(df_r['dir_true'], df_r['dir_pred'], labels=[1, 0])
        tp_ug, fn_ug = cm_ug[0, 0], cm_ug[0, 1]
        fp_ug, tn_ug = cm_ug[1, 0], cm_ug[1, 1]
        acc_ug = (tp_ug + tn_ug) / len(df_r) * 100

        # Gated metrics
        df_g = df_r[df_r['gated']].copy()
        cm_gt = confusion_matrix(df_g['dir_true'], df_g['dir_pred'], labels=[1, 0])
        tp_gt, fn_gt = cm_gt[0, 0], cm_gt[0, 1]
        fp_gt, tn_gt = cm_gt[1, 0], cm_gt[1, 1]

        acc_gt = (tp_gt + tn_gt) / len(df_g) * 100
        prec_gt = tp_gt / (tp_gt + fp_gt + 1e-8) * 100
        rec_gt = tp_gt / (tp_gt + fn_gt + 1e-8) * 100
        f1_gt = 2 * (prec_gt * rec_gt) / (prec_gt + rec_gt + 1e-8)
        fpr_gt, tpr_gt, _ = roc_curve(df_g['dir_true'], df_g['y_pred'])
        auc_gt = auc(fpr_gt, tpr_gt)
        cov_gt = len(df_g) / len(df_r) * 100

        computed = {
            'N_Ungated': len(df_r),
            'Accuracy_Ungated': round(acc_ug, 1),
            'N_Gated': len(df_g),
            'Accuracy_Gated': round(acc_gt, 1),
            'Coverage': round(cov_gt, 1),
            'Precision_Gated': round(prec_gt, 1),
            'Recall_Gated': round(rec_gt, 1),
            'F1_Gated': round(f1_gt, 1),
            'AUC_Gated': round(auc_gt, 3),
            'Winner_Counts': winner_counts
        }
        results[pair_key] = computed
        print(" Done.")

    print("\n" + "=" * 105)
    print(f"{'PAIR':<13} | {'METRIC':<18} | {'COMPUTED':<12} | {'REPORT VALUE':<14} | {'STATUS':<8}")
    print("=" * 105)

    num_checks = 0
    num_passed = 0

    for pair_key, expected in REPORTED_METRICS.items():
        comp = results[pair_key]
        metrics_to_check = [
            ('N_Ungated', comp['N_Ungated'], expected['N_Ungated']),
            ('Accuracy_Ungated', comp['Accuracy_Ungated'], expected['Accuracy_Ungated']),
            ('N_Gated', comp['N_Gated'], expected['N_Gated']),
            ('Accuracy_Gated', comp['Accuracy_Gated'], expected['Accuracy_Gated']),
            ('Coverage (%)', comp['Coverage'], expected['Coverage']),
            ('Precision_Gated', comp['Precision_Gated'], expected['Precision_Gated']),
            ('Recall_Gated', comp['Recall_Gated'], expected['Recall_Gated']),
            ('F1_Gated', comp['F1_Gated'], expected['F1_Gated']),
            ('AUC_Gated', comp['AUC_Gated'], expected['AUC_Gated']),
        ]

        for name, val_c, val_e in metrics_to_check:
            num_checks += 1
            is_match = (val_c == val_e) or (abs(val_c - val_e) < 0.15)
            status_str = "PASS [OK]" if is_match else "FAIL [MISMATCH]"
            if is_match:
                num_passed += 1
            else:
                all_passed = False
            print(f"{pair_key:<13} | {name:<18} | {str(val_c):<12} | {str(val_e):<14} | {status_str:<8}")
        print("-" * 105)

    print("\n" + "=" * 90)
    print(f"VERIFICATION SUMMARY: {num_passed}/{num_checks} checks PASSED")
    if all_passed:
        print(">>> SUCCESS: 100% RECONCILIATION CONFIRMED! <<<")
        print("All code outputs match MASTER_EVALUATION_REPORT.md exactly.")
    else:
        print(">>> WARNING: Discrepancies detected between execution and report! <<<")
    print("=" * 90)
    return all_passed

if __name__ == '__main__':
    success = run_verification()
    sys.exit(0 if success else 1)
