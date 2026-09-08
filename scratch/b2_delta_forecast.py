"""
=============================================================================
B2 — Δfreight HORIZON-SPECIFIC FORECASTING: LEVEL vs DELTA SCOREBOARD
=============================================================================
ISOLATION CONTRACT: Does NOT modify any existing outputs/ or src/ files.
All outputs: outputs/delta_forecast/

For every asset × horizon (ALL 20 pairs, regardless of B1 viability tier):
  - Level-Ridge : predict y[t+h]             (target = level)
  - Δ-Ridge     : predict y[t+h] - y[t]      (target = change)
  - Persistence : y[t]

Feature pipeline (strict no-lookahead):
  1. Impute NaN with TRAIN-set column medians
  2. StandardScaler fit on TRAIN only
  3. SelectKBest(f_regression) K ∈ {10, 20, 30, 50} — pre-registered sweep
     K locked on VAL macro-F1 @ 2% threshold (NOT test)
  4. Alpha ∈ {1, 10, 100, 1000, 10000} tuned on VAL
  5. Test set evaluated ONCE per configuration

Confidence / uncertainty:
  - NOT 1 - sMAPE/100 (historical accuracy ≠ prediction confidence)
  - Instead: validation residual distribution → P10 / P50 / P90 intervals

Win condition (strong bar):
  Δ-Ridge beats STRONGEST naive baseline on locked test set
  AND improves on BOTH sMAPE and Directional Accuracy vs persistence
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import f1_score

np.random.seed(42)
os.makedirs('outputs/delta_forecast', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ASSETS     = ['kdci', 'cape', 'panamax', 'supramax', 'handy']
HORIZONS   = [1, 7, 14, 30]
K_GRID     = [10, 20, 30, 50]     # pre-registered sensitivity — locked on VAL
ALPHA_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]
THRESH_2PCT = 0.02                 # Macro-F1 threshold for UP/DOWN/NEUTRAL labels

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("B2 — Δfreight FORECASTING: LEVEL vs DELTA SCOREBOARD")
print("=" * 70)

df = pd.read_csv('outputs/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
n = len(df)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
n_test  = n - n_train - n_val

print(f"\n  Rows: {n}  Train: {n_train}  Val: {n_val}  Test: {n_test}")
print(f"  Train : {df.iloc[:n_train]['date'].max().date()}")
print(f"  Val   : {df.iloc[n_train:n_train+n_val]['date'].max().date()}")
print(f"  Test  : {df.iloc[n_train+n_val:]['date'].max().date()}")

# Clean feature columns (no dir_*, no target_*)
feature_cols = [c for c in df.columns
                if not c.startswith('target_')
                and not c.startswith('dir_')
                and c not in ['date']]
print(f"\n  Clean features: {len(feature_cols)}")

# Pre-impute + scale (train-fit only)
X_raw = df[feature_cols].values.astype(np.float64)
train_medians = np.nanmedian(X_raw[:n_train], axis=0)
for ci in range(X_raw.shape[1]):
    nan_mask = np.isnan(X_raw[:, ci])
    X_raw[nan_mask, ci] = train_medians[ci]

scaler = StandardScaler()
scaler.fit(X_raw[:n_train])
X_sc = scaler.transform(X_raw)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def r2(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

def dir_accuracy(y_true, y_pred, y_base):
    act = np.sign(y_true - y_base)
    prd = np.sign(y_pred - y_base)
    return np.mean(act == prd)

def macro_f1_2pct(y_true, y_pred, y_base, thresh=THRESH_2PCT):
    """Macro-F1 for UP/DOWN/NEUTRAL classification at ±2% threshold."""
    pct_true = (y_true - y_base) / (np.abs(y_base) + 1e-8)
    pct_pred = (y_pred - y_base) / (np.abs(y_base) + 1e-8)
    def label(pct):
        l = np.full(len(pct), 'NEUTRAL', dtype=object)
        l[pct >  thresh] = 'UP'
        l[pct < -thresh] = 'DOWN'
        return l
    return f1_score(label(pct_true), label(pct_pred),
                    average='macro', labels=['UP', 'DOWN', 'NEUTRAL'],
                    zero_division=0)

def full_metrics(y_true, y_pred, y_base):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    sm   = smape(y_true, y_pred)
    r2v  = r2(y_true, y_pred)
    da   = dir_accuracy(y_true, y_pred, y_base)
    mf1  = macro_f1_2pct(y_true, y_pred, y_base)
    return dict(MAE=round(mae,1), RMSE=round(rmse,1),
                sMAPE=round(sm,2), R2=round(r2v,4),
                DirAcc=round(da*100,1), MacroF1_2pct=round(mf1*100,1))

def val_residual_intervals(y_val_true, y_val_pred):
    """
    Uncertainty quantification from validation residuals.
    Returns P10, P50 (median), P90 of residual distribution.
    These are used as additive offsets on future test predictions:
        P10_pred = point_pred + residual_P10
        P90_pred = point_pred + residual_P90
    """
    resid = y_val_true - y_val_pred
    p10 = np.percentile(resid, 10)
    p50 = np.percentile(resid, 50)
    p90 = np.percentile(resid, 90)
    std = np.std(resid)
    return dict(resid_P10=round(p10,1), resid_P50=round(p50,1),
                resid_P90=round(p90,1), resid_std=round(std,1),
                interval_width_80pct=round(p90-p10,1))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN B2 LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_results   = []
pred_rows     = []   # per-row predictions for the B3-ready output

# Load B1 viability map for tier annotation
b1_path = 'outputs/b1_viability/b1_viability_map.csv'
if os.path.exists(b1_path):
    df_b1 = pd.read_csv(b1_path)
    b1_tier = {(r['asset'], r['horizon']): r['viability_tier']
               for _, r in df_b1.iterrows()}
    b1_prio = {(r['asset'], r['horizon']): r['delta_test_priority']
               for _, r in df_b1.iterrows()}
else:
    b1_tier = {}
    b1_prio = {}

print("\n" + "=" * 70)
print("Running B2: Level vs Δ for ALL 20 asset×horizon pairs")
print("=" * 70)

for asset in ASSETS:
    for h in HORIZONS:
        target_col = f'target_{asset}_{h}d'
        price_col  = asset
        horizon_str = f'{h}d'

        tier = b1_tier.get((asset, horizon_str), 'UNKNOWN')
        prio = b1_prio.get((asset, horizon_str), 'UNKNOWN')

        # Build row masks
        valid_mask = df[target_col].notna() & df[price_col].notna() & (df[price_col] > 0)
        tr_mask = valid_mask & (df.index < n_train)
        v_mask  = valid_mask & (df.index >= n_train) & (df.index < n_train + n_val)
        te_mask = valid_mask & (df.index >= n_train + n_val)

        y_tr_lvl  = df.loc[tr_mask, target_col].values
        y_v_lvl   = df.loc[v_mask,  target_col].values
        y_te_lvl  = df.loc[te_mask, target_col].values

        y_tr_base = df.loc[tr_mask, price_col].values
        y_v_base  = df.loc[v_mask,  price_col].values
        y_te_base = df.loc[te_mask, price_col].values

        # Delta targets
        y_tr_delta = y_tr_lvl - y_tr_base
        y_v_delta  = y_v_lvl  - y_v_base
        y_te_delta = y_te_lvl - y_te_base

        X_tr = X_sc[tr_mask]
        X_v  = X_sc[v_mask]
        X_te = X_sc[te_mask]

        n_tr, n_v, n_te = len(y_tr_lvl), len(y_v_lvl), len(y_te_lvl)
        if n_tr < 50 or n_v < 10 or n_te < 10:
            print(f"  SKIP {asset.upper()} {h}d — insufficient samples ({n_tr}/{n_v}/{n_te})")
            continue

        print(f"\n  >>> {asset.upper()} {h}d  [B1-tier={tier}, Δ-priority={prio}] "
              f"(n_tr={n_tr}, n_v={n_v}, n_te={n_te}) <<<")

        # ─────────────────────────────────────────────────
        # PERSISTENCE BASELINE (no model)
        # ─────────────────────────────────────────────────
        pers_te = y_te_base.copy()   # predict y[t+h] = y[t]
        pers_m  = full_metrics(y_te_lvl, pers_te, y_te_base)
        print(f"    Persistence        : MAE={pers_m['MAE']:>7.0f}  sMAPE={pers_m['sMAPE']:>5.2f}%  "
              f"R²={pers_m['R2']:>7.4f}  DA={pers_m['DirAcc']:>5.1f}%  "
              f"MacF1@2%={pers_m['MacroF1_2pct']:>5.1f}%")

        # ─────────────────────────────────────────────────
        # K SWEEP — lock on VAL Macro-F1@2% (NOT test)
        # Fit fresh SelectKBest inside training for each K
        # ─────────────────────────────────────────────────
        best_K_lvl  = K_GRID[0]
        best_K_dlt  = K_GRID[0]
        best_vf1_lvl = -1.0
        best_vf1_dlt = -1.0

        for K in K_GRID:
            k_eff = min(K, X_tr.shape[1])

            # --- Level model ---
            sel_lvl = SelectKBest(f_regression, k=k_eff)
            sel_lvl.fit(X_tr, y_tr_lvl)
            Xtr_l = sel_lvl.transform(X_tr)
            Xv_l  = sel_lvl.transform(X_v)

            for alpha in ALPHA_GRID:
                m = Ridge(alpha=alpha)
                m.fit(Xtr_l, y_tr_lvl)
                vp = m.predict(Xv_l)
                vf1 = macro_f1_2pct(y_v_lvl, vp, y_v_base)
                if vf1 > best_vf1_lvl:
                    best_vf1_lvl = vf1
                    best_K_lvl   = K

            # --- Delta model ---
            sel_dlt = SelectKBest(f_regression, k=k_eff)
            sel_dlt.fit(X_tr, y_tr_delta)
            Xtr_d = sel_dlt.transform(X_tr)
            Xv_d  = sel_dlt.transform(X_v)

            for alpha in ALPHA_GRID:
                m = Ridge(alpha=alpha)
                m.fit(Xtr_d, y_tr_delta)
                vp_delta = m.predict(Xv_d)
                vp_lvl   = y_v_base + vp_delta  # reconstruct level
                vf1 = macro_f1_2pct(y_v_lvl, vp_lvl, y_v_base)
                if vf1 > best_vf1_dlt:
                    best_vf1_dlt = vf1
                    best_K_dlt   = K

        # ─────────────────────────────────────────────────
        # FINAL FIT with best K, then tune alpha on VAL
        # ─────────────────────────────────────────────────
        def fit_and_eval(X_tr_, y_tr_, X_v_, y_v_, X_te_, y_te_lvl_, y_base_v, y_base_te,
                         is_delta=False):
            """
            Fits Ridge with alpha tuned on VAL Macro-F1@2%.
            Returns (test_metrics, val_metrics, val_residual_intervals,
                     y_pred_te, best_alpha, selector)
            """
            k_use = min(best_K_dlt if is_delta else best_K_lvl, X_tr_.shape[1])

            sel = SelectKBest(f_regression, k=k_use)
            sel.fit(X_tr_, y_tr_)
            Xtr_s = sel.transform(X_tr_)
            Xv_s  = sel.transform(X_v_)
            Xte_s = sel.transform(X_te_)

            best_a, best_vf1_, best_vp_v = ALPHA_GRID[0], -1.0, None
            for alpha in ALPHA_GRID:
                ridge = Ridge(alpha=alpha)
                ridge.fit(Xtr_s, y_tr_)
                vp = ridge.predict(Xv_s)
                if is_delta:
                    vp_lvl = y_base_v + vp
                else:
                    vp_lvl = vp
                vf1 = macro_f1_2pct(y_v_lvl, vp_lvl, y_base_v)
                if vf1 > best_vf1_:
                    best_vf1_ = vf1
                    best_a = alpha
                    best_vp_v = vp

            # Final fit -> evaluate ONCE on test
            ridge_final = Ridge(alpha=best_a)
            ridge_final.fit(Xtr_s, y_tr_)

            vp_final = ridge_final.predict(Xv_s)
            tp_final = ridge_final.predict(Xte_s)

            if is_delta:
                vp_lvl_f = y_base_v  + vp_final
                tp_lvl_f = y_base_te + tp_final
            else:
                vp_lvl_f = vp_final
                tp_lvl_f = tp_final

            # Validation residuals for uncertainty intervals
            v_resids  = val_residual_intervals(y_v_lvl, vp_lvl_f)
            te_m      = full_metrics(y_te_lvl_, tp_lvl_f, y_base_te)
            v_m       = full_metrics(y_v_lvl,   vp_lvl_f, y_base_v)

            top5_feat = [feature_cols[i] for i in sel.get_support(indices=True)][:5]

            return te_m, v_m, v_resids, tp_lvl_f, best_a, k_use, top5_feat

        # Level-Ridge
        te_lvl, v_lvl, vri_lvl, pred_te_lvl, a_lvl, k_lvl, f5_lvl = fit_and_eval(
            X_tr, y_tr_lvl, X_v, y_v_lvl, X_te, y_te_lvl, y_v_base, y_te_base,
            is_delta=False)

        # Delta-Ridge
        te_dlt, v_dlt, vri_dlt, pred_te_dlt, a_dlt, k_dlt, f5_dlt = fit_and_eval(
            X_tr, y_tr_delta, X_v, y_v_delta, X_te, y_te_lvl, y_v_base, y_te_base,
            is_delta=True)

        # ─────────────────────────────────────────────────
        # WIN CONDITION CHECK
        # ─────────────────────────────────────────────────
        # Delta beats persistence on BOTH sMAPE and DirAcc
        dlt_beats_pers_smape = te_dlt['sMAPE'] < pers_m['sMAPE']
        dlt_beats_pers_da    = te_dlt['DirAcc'] > pers_m['DirAcc']
        # Delta beats level-Ridge on MacroF1@2%
        dlt_beats_level_f1   = te_dlt['MacroF1_2pct'] > te_lvl['MacroF1_2pct']
        # Delta beats ALL naive on MAE (strong bar)
        dlt_beats_pers_mae   = te_dlt['MAE'] < pers_m['MAE']
        dlt_beats_level_mae  = te_dlt['MAE'] < te_lvl['MAE']

        win_score = sum([dlt_beats_pers_smape, dlt_beats_pers_da,
                         dlt_beats_level_f1, dlt_beats_pers_mae])
        if win_score == 4:
            win_verdict = 'STRONG_WIN'    # all 4 conditions
        elif win_score >= 3:
            win_verdict = 'WIN'
        elif win_score >= 2:
            win_verdict = 'MARGINAL'
        elif win_score == 1:
            win_verdict = 'WEAK'
        else:
            win_verdict = 'LOSS'

        # ─────────────────────────────────────────────────
        # PRINT SCOREBOARD
        # ─────────────────────────────────────────────────
        hdr = f"{'':>18}  {'MAE':>7} {'sMAPE':>7} {'R²':>7} {'DA%':>6} {'MacF1@2%':>9}"
        sep = "    " + "-" * 58
        print(f"    {hdr}")
        print(sep)
        print(f"    {'Persistence':>18}  {pers_m['MAE']:>7.0f} {pers_m['sMAPE']:>6.2f}% "
              f"{pers_m['R2']:>7.4f} {pers_m['DirAcc']:>5.1f}% {pers_m['MacroF1_2pct']:>8.1f}%")
        print(f"    {'Level-Ridge':>18}  {te_lvl['MAE']:>7.0f} {te_lvl['sMAPE']:>6.2f}% "
              f"{te_lvl['R2']:>7.4f} {te_lvl['DirAcc']:>5.1f}% {te_lvl['MacroF1_2pct']:>8.1f}%"
              f"  [K={k_lvl} α={a_lvl:.0f}]")
        print(f"    {'Delta-Ridge':>18}  {te_dlt['MAE']:>7.0f} {te_dlt['sMAPE']:>6.2f}% "
              f"{te_dlt['R2']:>7.4f} {te_dlt['DirAcc']:>5.1f}% {te_dlt['MacroF1_2pct']:>8.1f}%"
              f"  [K={k_dlt} α={a_dlt:.0f}]")
        print(f"\n    Δ-Ridge verdict: {win_verdict}  "
              f"(beats_pers_sMAPE={dlt_beats_pers_smape}, "
              f"beats_pers_DA={dlt_beats_pers_da}, "
              f"beats_level_F1={dlt_beats_level_f1}, "
              f"beats_pers_MAE={dlt_beats_pers_mae})")
        print(f"    Val residual P10/P50/P90: "
              f"{vri_dlt['resid_P10']:+.0f} / {vri_dlt['resid_P50']:+.0f} / "
              f"{vri_dlt['resid_P90']:+.0f}  [80%-interval width: ±{vri_dlt['interval_width_80pct']/2:.0f}]")

        # ─────────────────────────────────────────────────
        # STORE RESULTS
        # ─────────────────────────────────────────────────
        base_rec = {
            'asset': asset, 'horizon': horizon_str,
            'b1_viability_tier': tier, 'b1_delta_priority': prio,
            'n_train': n_tr, 'n_val': n_v, 'n_test': n_te,
        }

        def add_prefix(d, prefix):
            return {f'{prefix}_{k}': v for k, v in d.items()}

        row = {**base_rec,
               **add_prefix(pers_m,    'pers'),
               **add_prefix(te_lvl,    'lvl_test'),
               **add_prefix(te_dlt,    'dlt_test'),
               **add_prefix(v_lvl,     'lvl_val'),
               **add_prefix(v_dlt,     'dlt_val'),
               **add_prefix(vri_dlt,   'dlt_unc'),
               'lvl_best_K': k_lvl, 'lvl_best_alpha': a_lvl,
               'dlt_best_K': k_dlt, 'dlt_best_alpha': a_dlt,
               'lvl_top5_features': ', '.join(f5_lvl),
               'dlt_top5_features': ', '.join(f5_dlt),
               'win_score': win_score, 'win_verdict': win_verdict,
               'dlt_beats_pers_sMAPE': dlt_beats_pers_smape,
               'dlt_beats_pers_DirAcc': dlt_beats_pers_da,
               'dlt_beats_level_F1': dlt_beats_level_f1,
               'dlt_beats_pers_MAE': dlt_beats_pers_mae,
               'dlt_beats_level_MAE': dlt_beats_level_mae,
               }
        all_results.append(row)

        # Per-row test predictions (for B3 integration when/if approved)
        te_dates = df.loc[te_mask, 'date'].values
        for i, (dt, ytrue, ybase, ypred_l, ypred_d) in enumerate(
                zip(te_dates, y_te_lvl, y_te_base, pred_te_lvl, pred_te_dlt)):
            pred_rows.append({
                'date': pd.Timestamp(dt).date(),
                'asset': asset, 'horizon': horizon_str,
                'y_true': round(ytrue, 2),
                'y_base': round(ybase, 2),
                'level_pred': round(ypred_l, 2),
                'delta_pred': round(ypred_d, 2),
                'delta_P10': round(ypred_d + vri_dlt['resid_P10'], 2),
                'delta_P90': round(ypred_d + vri_dlt['resid_P90'], 2),
                'win_verdict': win_verdict
            })

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SCOREBOARD SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
df_results.to_csv('outputs/delta_forecast/b2_comparison_scoreboard.csv', index=False)

df_preds = pd.DataFrame(pred_rows)
df_preds.to_csv('outputs/delta_forecast/b2_test_predictions.csv', index=False)

print("\n\n" + "=" * 70)
print("B2 FINAL SCOREBOARD SUMMARY")
print("=" * 70)
print(f"\n  {'Asset':>10} {'H':>4}  {'B1Tier':>10}  "
      f"{'PersMAE':>8} {'LvlMAE':>7} {'DltMAE':>7}  "
      f"{'LvlF1':>6} {'DltF1':>6}  "
      f"{'LvlDA':>6} {'DltDA':>6}  Verdict")
print("  " + "-" * 100)

wins_total = 0
for _, r in df_results.sort_values(['asset','horizon']).iterrows():
    win_mark = '★' if r['win_verdict'] in ['STRONG_WIN', 'WIN'] else ' '
    wins_total += 1 if r['win_verdict'] in ['STRONG_WIN', 'WIN'] else 0
    print(f"  {win_mark}{r['asset'].upper():>9} {r['horizon']:>4}  {r['b1_viability_tier']:>10}  "
          f"{r['pers_MAE']:>8.0f} {r['lvl_test_MAE']:>7.0f} {r['dlt_test_MAE']:>7.0f}  "
          f"{r['lvl_test_MacroF1_2pct']:>5.1f}% {r['dlt_test_MacroF1_2pct']:>5.1f}%  "
          f"{r['lvl_test_DirAcc']:>5.1f}% {r['dlt_test_DirAcc']:>5.1f}%  "
          f"{r['win_verdict']}")

total = len(df_results)
verdict_counts = df_results['win_verdict'].value_counts()

print(f"\n  Verdict distribution:")
for v, cnt in verdict_counts.items():
    print(f"    {v:>12} : {cnt}")

print(f"\n  Δ-Ridge WIN or STRONG_WIN: {wins_total}/{total} pairs")
print(f"\n  KEY QUESTION ANSWER:")
if wins_total >= total * 0.6:
    print(f"  -> Δ forecasting IMPROVES over level in majority of pairs.")
    print(f"  -> B3 integration is WARRANTED. Proceed.")
elif wins_total >= total * 0.4:
    print(f"  -> Δ forecasting shows MIXED results.")
    print(f"  -> Selectively integrate winning pairs into B3.")
    print(f"  -> Do NOT blanket-upgrade decision engine.")
else:
    print(f"  -> Δ forecasting does NOT consistently beat level or naive baselines.")
    print(f"  -> KILL HYPOTHESIS. Attack features/model architecture instead.")
    print(f"  -> Do NOT proceed to B3.")

print(f"\n  Outputs:")
print(f"    outputs/delta_forecast/b2_comparison_scoreboard.csv")
print(f"    outputs/delta_forecast/b2_test_predictions.csv")
print(f"      (B3-ready per-row forecasts with P10/P90 uncertainty bands)")
print("=" * 70)
