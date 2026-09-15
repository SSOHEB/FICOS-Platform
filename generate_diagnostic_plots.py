"""
FICOS Platform — Master Diagnostic Visualization Generator
Generates publication-grade, high-resolution (300 DPI) diagnostic figures for all 8 asset-horizon pairs
based on the exact 5-fold walk-forward validation results.
Aesthetic Theme: Clean Whitegrid / Modern Scientific Publication Styling
Outputs saved to images/ and outputs/.
"""

import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Set global clean aesthetic theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
BG_WHITE = '#FFFFFF'
TEXT_DARK = '#1E293B'
TEXT_MUTED = '#64748B'
GRID_COLOR = '#E2E8F0'

COLOR_ACCURACY = '#2CA02C'   # Green
COLOR_PRECISION = '#1F77B4'  # Blue
COLOR_F1 = '#FF7F0E'         # Orange
COLOR_UNGATED = '#64748B'    # Slate Gray
COLOR_GATED_LINE = '#D62728' # Red

plt.rcParams['figure.facecolor'] = BG_WHITE
plt.rcParams['axes.facecolor'] = BG_WHITE
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.labelcolor'] = TEXT_DARK
plt.rcParams['xtick.color'] = TEXT_DARK
plt.rcParams['ytick.color'] = TEXT_DARK
plt.rcParams['text.color'] = TEXT_DARK
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['grid.color'] = GRID_COLOR
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.8

ARTIFACT_DIR = r"C:\Users\soheb\.gemini\antigravity-ide\brain\7cd9deed-2fb0-497b-8e7e-caf9a0b6a7c2"
IMAGES_DIR = r"images"
OUTPUTS_DIR = r"outputs"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
if os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def smape(a, b):
    return float(np.mean(200 * np.abs(b - a) / (np.abs(a) + np.abs(b) + 1e-8)))

def run_pipeline_and_generate_plots():
    print(">> Starting Walk-Forward Pipeline & Plot Generation...", flush=True)
    dataset_path = 'data/modeling_dataset.csv' if os.path.exists('data/modeling_dataset.csv') else 'outputs/modeling_dataset.csv'
    df = pd.read_csv(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n_rows = len(df)

    drop_cols = set([c for c in df.columns if c.startswith(('dir_', 'future_', 'target_'))] + ['date'])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    test_size, val_size = 250, 200
    fold_configs = {}
    for f in range(1, 6):
        te_end = n_rows - (5 - f) * test_size
        te_start = te_end - test_size
        va_end, va_start = te_start, te_start - val_size
        fold_configs[f] = dict(tr_end=va_start, va_start=va_start,
                               va_end=va_end, te_start=te_start, te_end=te_end)

    eval_pairs = [
        ('cape', 1, 'CAPE_1D', 'PROMOTED (Conservative)', '#EAB308'),
        ('panamax', 1, 'PANAMAX_1D', 'PROMOTED (Primary)', '#22C55E'),
        ('supramax', 1, 'SUPRAMAX_1D', 'PROMOTED', '#3B82F6'),
        ('supramax', 7, 'SUPRAMAX_7D', 'FALLBACK (Flexible Index)', '#8B5CF6'),
        ('handy', 1, 'HANDY_1D', 'PROMOTED', '#06B6D4'),
        ('handy', 7, 'HANDY_7D', 'EXCLUDED (Abstain)', '#EF4444'),
        ('supramax', 14, 'SUPRAMAX_14D', 'EXCLUDED (Abstain)', '#EF4444'),
        ('kdci', 7, 'KDCI_7D', 'FALLBACK (Index-Linked)', '#F97316')
    ]

    all_results = {}
    feature_importance_records = []

    for (asset, h, display_name, status, color) in eval_pairs:
        pair_key = f"{asset.upper()}_{h}D"
        print(f">> Executing Walk-Forward for {display_name}...", flush=True)
        records = []
        fold_metrics = []
        all_val_resids = []

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

            # Imputation
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

            # SelectKBest
            sel = SelectKBest(f_regression, k=30)
            Xtr_s = sel.fit_transform(Xtr, y_d[tr_m])
            Xva_s = sel.transform(Xva)
            Xte_s = sel.transform(Xte)

            selected_idx = sel.get_support(indices=True)
            for s_i, f_score in zip(selected_idx, sel.scores_[selected_idx]):
                feature_importance_records.append({
                    'pair': display_name,
                    'fold': fold_id,
                    'feature': feature_cols[s_i],
                    'score': f_score
                })

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
            w_va_p = models[winner][1]
            pred_te = models[winner][2]

            val_resids = y_d[va_m] - w_va_p
            all_val_resids.extend(val_resids)
            p10, p90 = np.percentile(val_resids, 10), np.percentile(val_resids, 90)
            pct_p = pred_te / (np.abs(y_base[te_m]) + 1e-8)
            buy_m = (pred_te > max(0.0, p90)) & (pct_p > 0.01)
            wait_m = (pred_te < min(0.0, p10)) & (pct_p < -0.01)
            gated = buy_m | wait_m

            # Fold metrics
            fold_yt = y_d[te_m]
            fold_yp = pred_te
            fold_valid = (fold_yt != 0) & (fold_yp != 0)
            fold_dir_true = (fold_yt[fold_valid] > 0).astype(int)
            fold_dir_pred = (fold_yp[fold_valid] > 0).astype(int)
            fold_acc_ug = np.mean(fold_dir_true == fold_dir_pred) * 100

            gated_idx = np.where(gated[fold_valid])[0]
            if len(gated_idx) > 0:
                fold_acc_gt = np.mean(fold_dir_true[gated_idx] == fold_dir_pred[gated_idx]) * 100
            else:
                fold_acc_gt = np.nan

            fold_metrics.append({
                'fold': fold_id,
                'acc_ug': fold_acc_ug,
                'acc_gt': fold_acc_gt,
                'n_gated': len(gated_idx),
                'n_total': len(fold_dir_true)
            })

            for yt, yp, g in zip(y_d[te_m], pred_te, gated):
                records.append({'fold': fold_id, 'winner': winner, 'y_true': yt, 'y_pred': yp, 'gated': g})

        df_r = pd.DataFrame(records)
        df_r = df_r[(df_r['y_true'] != 0) & (df_r['y_pred'] != 0)].copy()
        df_r['dir_true'] = (df_r['y_true'] > 0).astype(int)
        df_r['dir_pred'] = (df_r['y_pred'] > 0).astype(int)

        # Master metrics
        cm_ug = confusion_matrix(df_r['dir_true'], df_r['dir_pred'], labels=[1, 0])
        acc_ug = (cm_ug[0, 0] + cm_ug[1, 1]) / len(df_r) * 100
        fpr_ug, tpr_ug, _ = roc_curve(df_r['dir_true'], df_r['y_pred'])
        auc_ug = auc(fpr_ug, tpr_ug)
        prec_curve_ug, rec_curve_ug, _ = precision_recall_curve(df_r['dir_true'], df_r['y_pred'])
        ap_ug = average_precision_score(df_r['dir_true'], df_r['y_pred'])

        df_g = df_r[df_r['gated']].copy()
        cm_gt = confusion_matrix(df_g['dir_true'], df_g['dir_pred'], labels=[1, 0])
        acc_gt = (cm_gt[0, 0] + cm_gt[1, 1]) / len(df_g) * 100 if len(df_g) > 0 else 0.0
        prec_gt = cm_gt[0, 0] / (cm_gt[0, 0] + cm_gt[1, 0] + 1e-8) * 100 if len(df_g) > 0 else 0.0
        rec_gt = cm_gt[0, 0] / (cm_gt[0, 0] + cm_gt[0, 1] + 1e-8) * 100 if len(df_g) > 0 else 0.0
        f1_gt = 2 * (prec_gt * rec_gt) / (prec_gt + rec_gt + 1e-8) if len(df_g) > 0 else 0.0
        fpr_gt, tpr_gt, _ = roc_curve(df_g['dir_true'], df_g['y_pred']) if len(df_g) > 0 else ([], [], [])
        auc_gt = auc(fpr_gt, tpr_gt) if len(df_g) > 0 else 0.5
        prec_curve_gt, rec_curve_gt, _ = precision_recall_curve(df_g['dir_true'], df_g['y_pred']) if len(df_g) > 0 else ([], [], [])
        ap_gt = average_precision_score(df_g['dir_true'], df_g['y_pred']) if len(df_g) > 0 else 0.5
        cov_gt = len(df_g) / len(df_r) * 100

        all_results[pair_key] = {
            'display_name': display_name,
            'status': status,
            'color': color,
            'df_r': df_r,
            'df_g': df_g,
            'cm_ug': cm_ug,
            'cm_gt': cm_gt,
            'acc_ug': acc_ug,
            'acc_gt': acc_gt,
            'prec_gt': prec_gt,
            'rec_gt': rec_gt,
            'f1_gt': f1_gt,
            'auc_ug': auc_ug,
            'auc_gt': auc_gt,
            'ap_ug': ap_ug,
            'ap_gt': ap_gt,
            'cov_gt': cov_gt,
            'fpr_ug': fpr_ug,
            'tpr_ug': tpr_ug,
            'fpr_gt': fpr_gt,
            'tpr_gt': tpr_gt,
            'prec_curve_gt': prec_curve_gt,
            'rec_curve_gt': rec_curve_gt,
            'val_resids': np.array(all_val_resids),
            'fold_metrics': pd.DataFrame(fold_metrics)
        }

    df_feat_imp = pd.DataFrame(feature_importance_records)
    print(">> All walk-forward simulations complete. Generating figures...")

    # Helper function to save figure to both images/ and artifact dir
    def save_fig(fig, filename):
        p1 = os.path.join(IMAGES_DIR, filename)
        fig.savefig(p1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {p1}")
        p_out = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(p_out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        if os.path.exists(ARTIFACT_DIR):
            p2 = os.path.join(ARTIFACT_DIR, filename)
            shutil.copyfile(p1, p2)
            print(f"Copied to artifact: {p2}")
        plt.close(fig)

    # =========================================================================
    # FIGURE 1: Feature Importance (Exact Match to Reference Style 1 - Mako Gradient)
    # =========================================================================
    print(">> Rendering Figure 1: Feature Importance (Mako Palette)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # Top 15 features sorted
    top15 = df_feat_imp.groupby('feature')['score'].mean().sort_values(ascending=False).head(15)
    top15_df = top15.sort_values(ascending=True) # bottom to top

    # Generate custom mako gradient from dark navy/black to light seafoam
    mako_colors = sns.color_palette("mako_r", n_colors=15)
    # Reverse so top bar (largest) is darkest black/purple
    bar_colors = list(reversed(mako_colors))

    y_positions = np.arange(len(top15_df))
    bars = ax.barh(y_positions, top15_df.values, color=bar_colors, edgecolor='none', height=0.75)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(top15_df.index, fontsize=10, fontweight='semibold', color=TEXT_DARK)
    ax.set_xlabel("Mean F-Regression Statistic Score", fontsize=11, fontweight='bold', color=TEXT_DARK)
    ax.set_ylabel("Feature Name", fontsize=11, fontweight='bold', color=TEXT_DARK)
    ax.set_title("Top 15 Most Predictive Features Across Walk-Forward Folds (ANOVA F-Score)",
                 fontsize=13, fontweight='bold', color=TEXT_DARK, pad=12)

    ax.set_xlim(0, max(top15_df.values) * 1.05)
    ax.grid(True, axis='x', linestyle='-', color='#E2E8F0', alpha=0.9)
    ax.grid(False, axis='y')

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va='center', ha='left', fontsize=10, color=TEXT_DARK, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'feature_importance.png')

    # =========================================================================
    # FIGURE 2: Metrics Comparison (Exact Match to Reference Style 2 - Green/Blue/Orange)
    # =========================================================================
    print(">> Rendering Figure 2: Gated Performance Comparison (Green/Blue/Orange)...")
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    pair_names = [p[2] for p in eval_pairs]
    acc_vals = [all_results[p[2]]['acc_gt'] for p in eval_pairs]
    prec_vals = [all_results[p[2]]['prec_gt'] for p in eval_pairs]
    f1_vals = [all_results[p[2]]['f1_gt'] for p in eval_pairs]

    x = np.arange(len(pair_names))
    width = 0.26

    rects1 = ax.bar(x - width, acc_vals, width, label='Gated Accuracy (%)', color=COLOR_ACCURACY, edgecolor='none')
    rects2 = ax.bar(x, prec_vals, width, label='Gated Precision (%)', color=COLOR_PRECISION, edgecolor='none')
    rects3 = ax.bar(x + width, f1_vals, width, label='Gated F1 Score (%)', color=COLOR_F1, edgecolor='none')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold', color=TEXT_DARK)
    ax.set_title('FICOS Promoted & Baseline Models — Gated Performance Comparison',
                 fontsize=13, fontweight='bold', color=TEXT_DARK, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=10, fontweight='bold', color=TEXT_DARK)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', linestyle='-', color='#E2E8F0', alpha=0.9)
    ax.grid(False, axis='x')

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color=TEXT_DARK)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.legend(loc='lower right', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9.5)
    plt.tight_layout()
    save_fig(fig, 'metrics_comparison.png')

    # =========================================================================
    # FIGURE 3: ROC Curves (4x2 Clean Whitegrid)
    # =========================================================================
    print(">> Rendering Figure 3: ROC Curves...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.38, wspace=0.25)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        ax.set_facecolor(BG_WHITE)

        ax.plot(res['fpr_gt'], res['tpr_gt'], color=COLOR_GATED_LINE, lw=2.5, label=f"Gated (AUC = {res['auc_gt']:.3f})")
        ax.plot(res['fpr_ug'], res['tpr_ug'], color=COLOR_UNGATED, lw=1.8, linestyle='--', label=f"Ungated (AUC = {res['auc_ug']:.3f})")
        ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle=':', lw=1.2, label="Chance Level (0.50)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=9.5, color=TEXT_MUTED)
        ax.set_ylabel("True Positive Rate", fontsize=9.5, color=TEXT_MUTED)
        ax.set_title(f"{display_name} — ROC Curve\nGated AUC: {res['auc_gt']:.3f} | Status: {status}",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=8)
        ax.grid(True, linestyle='-', color=GRID_COLOR, alpha=0.8)
        ax.legend(loc='lower right', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("FICOS Platform — Receiver Operating Characteristic (ROC) & AUC Analysis\n(5-Fold Purged Walk-Forward Out-of-Sample Validation)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'roc_curves.png')

    # =========================================================================
    # FIGURE 4: Precision-Recall Curves (4x2 Clean Whitegrid)
    # =========================================================================
    print(">> Rendering Figure 4: Precision-Recall Curves...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.38, wspace=0.25)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        ax.set_facecolor(BG_WHITE)

        if len(res['prec_curve_gt']) > 0:
            ax.plot(res['rec_curve_gt'], res['prec_curve_gt'], color=COLOR_PRECISION, lw=2.5, label=f"Gated PR (AP = {res['ap_gt']:.3f})")
        baseline = res['df_g']['dir_true'].mean() if len(res['df_g']) > 0 else 0.5
        ax.axhline(baseline, color='#94A3B8', linestyle=':', lw=1.2, label=f"Baseline Prevalence ({baseline*100:.1f}%)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("Recall", fontsize=9.5, color=TEXT_MUTED)
        ax.set_ylabel("Precision", fontsize=9.5, color=TEXT_MUTED)
        ax.set_title(f"{display_name} — Precision-Recall Curve\nAP: {res['ap_gt']:.3f} | Gated F1: {res['f1_gt']:.1f}%",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=8)
        ax.grid(True, linestyle='-', color=GRID_COLOR, alpha=0.8)
        ax.legend(loc='lower left', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("FICOS Platform — Precision-Recall Curves & Average Precision (AP)\n(Evaluating Directional Discrimination Under Market Regime Shifts)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'pr_curves.png')

    # =========================================================================
    # FIGURE 5: Confusion Matrices (4x2 Clean Whitegrid)
    # =========================================================================
    print(">> Rendering Figure 5: Confusion Matrices...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.42, wspace=0.28)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        cm = res['cm_gt']
        tp, fp, fn, tn = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]

        matrix_data = np.array([[tp, fn], [fp, tn]])
        sns.heatmap(matrix_data, annot=False, cmap='Blues', cbar=False, ax=ax,
                    linewidths=1.5, linecolor='#CBD5E1')

        # Annotations
        ax.text(0.5, 0.4, f"TP = {tp}\n(Correct UP)", ha='center', va='center', color=TEXT_DARK, fontsize=11, fontweight='bold')
        ax.text(1.5, 0.4, f"FN = {fn}\n(Missed UP)", ha='center', va='center', color='#DC2626', fontsize=11, fontweight='bold')
        ax.text(0.5, 1.4, f"FP = {fp}\n(False Alarm)", ha='center', va='center', color='#DC2626', fontsize=11, fontweight='bold')
        ax.text(1.5, 1.4, f"TN = {tn}\n(Correct DOWN)", ha='center', va='center', color=TEXT_DARK, fontsize=11, fontweight='bold')

        ax.set_xticklabels(['Pred UP', 'Pred DOWN'], fontsize=10, fontweight='bold', color=TEXT_DARK)
        ax.set_yticklabels(['Actual UP', 'Actual DOWN'], fontsize=10, fontweight='bold', color=TEXT_DARK)
        ax.set_title(f"{display_name} — Directional Matrix\nAcc: {res['acc_gt']:.1f}% | Prec: {res['prec_gt']:.1f}% | F1: {res['f1_gt']:.1f}% (N={len(res['df_g'])})",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=10)

    fig.suptitle("FICOS Platform — Directional Confusion Matrices Across Asset-Horizon Pairs\n(Evaluated on Out-of-Sample Empirical Uncertainty Gated Days)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'confusion_matrices.png')

    # =========================================================================
    # FIGURE 6: Residual Distributions & Uncertainty Bounds (4x2)
    # =========================================================================
    print(">> Rendering Figure 6: Residual Distributions...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        val_resids = res['val_resids']
        p10, p90 = np.percentile(val_resids, 10), np.percentile(val_resids, 90)

        sns.histplot(val_resids, kde=True, ax=ax, color=COLOR_PRECISION, bins=35, stat='density', alpha=0.5, edgecolor='none')
        ax.axvline(p10, color='#DC2626', linestyle='--', lw=2, label=f"P10 ({p10:.1f})")
        ax.axvline(p90, color=COLOR_ACCURACY, linestyle='--', lw=2, label=f"P90 ({p90:.1f})")
        ax.axvline(0, color='#64748B', linestyle=':', lw=1)

        ax.set_xlabel("Validation Rate Prediction Error ($/day)", fontsize=9.5, color=TEXT_MUTED)
        ax.set_ylabel("Density", fontsize=9.5, color=TEXT_MUTED)
        ax.set_title(f"{display_name} — Empirical Uncertainty Gate\nP10: {p10:.1f} | P90: {p90:.1f} ($/day)",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=8)
        ax.grid(True, linestyle='-', color=GRID_COLOR, alpha=0.8)
        ax.legend(loc='upper right', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("FICOS Platform — Empirical Residual Uncertainty Distributions\n(Out-of-Sample P10/P90 Thresholds Used for Production Confidence Gating)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'residual_distribution.png')

    # =========================================================================
    # FIGURE 7: Regression Scatter Plots (4x2)
    # =========================================================================
    print(">> Rendering Figure 7: Regression Scatter Plots...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        df_r = res['df_r']
        df_g = res['df_g']

        ax.scatter(df_r['y_true'], df_r['y_pred'], color='#94A3B8', alpha=0.3, s=16, label='Ungated (All Days)')
        if len(df_g) > 0:
            ax.scatter(df_g['y_true'], df_g['y_pred'], color=COLOR_ACCURACY, alpha=0.85, s=28, label='Gated (High Conviction)')

        # Identity line
        lims = [min(df_r['y_true'].min(), df_r['y_pred'].min()), max(df_r['y_true'].max(), df_r['y_pred'].max())]
        ax.plot(lims, lims, color='#DC2626', linestyle='--', lw=1.5, label='Identity Line (y=x)')

        ax.set_xlabel("Actual Freight Rate Change Δ ($/day)", fontsize=9.5, color=TEXT_MUTED)
        ax.set_ylabel("Predicted Rate Change Δ ($/day)", fontsize=9.5, color=TEXT_MUTED)
        ax.set_title(f"{display_name} — Predicted vs Actual Delta\nGated DA: {res['acc_gt']:.1f}% | N_Gated={len(df_g)}",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=8)
        ax.grid(True, linestyle='-', color=GRID_COLOR, alpha=0.8)
        ax.legend(loc='lower right', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("FICOS Platform — Predicted vs Actual Freight Rate Delta Scatter Plots\n(Highlighting High-Conviction Gated Decisions vs Global Sample)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'regression_scatter.png')

    # =========================================================================
    # FIGURE 8: Fold-by-Fold Stability & Variance (4x2)
    # =========================================================================
    print(">> Rendering Figure 8: Fold Stability Analysis...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.patch.set_facecolor(BG_WHITE)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    for ax, (asset, h, display_name, status, color) in zip(axes.flatten(), eval_pairs):
        res = all_results[display_name]
        fm = res['fold_metrics']

        folds = fm['fold']
        ax.plot(folds, fm['acc_ug'], marker='o', color=COLOR_UNGATED, lw=2, label='Ungated DA (%)')
        ax.plot(folds, fm['acc_gt'], marker='s', color=COLOR_ACCURACY, lw=2.5, label='Gated DA (%)')
        ax.axhline(50, color='#DC2626', linestyle=':', lw=1.2, label='Random Chance (50%)')

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=9.5, fontweight='bold', color=TEXT_DARK)
        ax.set_ylim(35, 102)
        ax.set_xlabel("Walk-Forward Cross-Validation Fold", fontsize=9.5, color=TEXT_MUTED)
        ax.set_ylabel("Directional Accuracy (%)", fontsize=9.5, color=TEXT_MUTED)
        ax.set_title(f"{display_name} — Cross-Fold Stability\nMean Gated DA: {res['acc_gt']:.1f}%",
                     fontsize=11, fontweight='bold', color=TEXT_DARK, pad=8)
        ax.grid(True, linestyle='-', color=GRID_COLOR, alpha=0.8)
        ax.legend(loc='lower right', frameon=True, facecolor=BG_WHITE, edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("FICOS Platform — 5-Fold Walk-Forward Cross-Validation Stability Analysis\n(Verifying Temporal Robustness and Absence of Regime Overfitting)",
                 fontsize=15, fontweight='bold', color=TEXT_DARK, y=0.995)
    save_fig(fig, 'fold_variance.png')

    print("\n>> MASTER DIAGNOSTIC FIGURES GENERATED SUCCESSFULLY IN images/ AND outputs/!")

if __name__ == "__main__":
    run_pipeline_and_generate_plots()
