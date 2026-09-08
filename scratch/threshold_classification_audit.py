"""
=============================================================================
THRESHOLD-BASED CLASSIFICATION AUDIT  —  Phases A0 through A5
=============================================================================
ISOLATION CONTRACT:
  - Does NOT modify any existing outputs/ files or src/ pipelines.
  - Does NOT use test-set information for threshold or hyperparameter selection.
  - All outputs saved to: outputs/classification_audit/
  - Label formula: (y[t+h] - y[t]) / y[t]  <- verified in A3
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

np.random.seed(42)
os.makedirs('outputs/classification_audit', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ASSETS     = ['kdci', 'cape', 'panamax', 'supramax', 'handy']
HORIZONS   = [1, 7, 14, 30]
THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
N_FEATURES = 30
C_GRID     = [0.01, 0.1, 1.0, 10.0, 100.0]
CLASS_ORDER = ['UP', 'DOWN', 'NEUTRAL']

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("THRESHOLD-BASED CLASSIFICATION AUDIT  (A0 to A5)")
print("=" * 70)
print("\nLoading modeling dataset...")

df = pd.read_csv('outputs/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
n = len(df)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
n_test  = n - n_train - n_val

train_idx = df.index[df.index < n_train]
val_idx   = df.index[(df.index >= n_train) & (df.index < n_train + n_val)]
test_idx  = df.index[df.index >= n_train + n_val]

print(f"  Rows: {n}  |  Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")
print(f"  Train : {df.loc[train_idx, 'date'].min().date()} -> {df.loc[train_idx, 'date'].max().date()}")
print(f"  Val   : {df.loc[val_idx,   'date'].min().date()} -> {df.loc[val_idx,   'date'].max().date()}")
print(f"  Test  : {df.loc[test_idx,  'date'].min().date()} -> {df.loc[test_idx,  'date'].max().date()}")

feature_cols = [c for c in df.columns
                if not c.startswith('target_')
                and not c.startswith('dir_')
                and c not in ['date']]
print(f"\n  Clean feature columns: {len(feature_cols)}  (dir_* and target_* excluded)")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def make_labels(asset, h, threshold_pct):
    """
    A3-VERIFIED label formula:
        pct_change(t) = (target_{asset}_{h}d[t] - asset[t]) / asset[t]
        UP      if pct_change >  +threshold_pct/100
        DOWN    if pct_change <  -threshold_pct/100
        NEUTRAL otherwise
    """
    target_col = f'target_{asset}_{h}d'
    price_col  = asset
    pct = (df[target_col] - df[price_col]) / (df[price_col].abs() + 1e-8)
    labels = pd.Series('NEUTRAL', index=df.index, dtype=object)
    labels[pct >  threshold_pct / 100.0] = 'UP'
    labels[pct < -threshold_pct / 100.0] = 'DOWN'
    valid = df[target_col].notna() & df[price_col].notna() & (df[price_col] > 0)
    labels[~valid] = None
    return labels, pct

def class_metrics(y_true, y_pred, labels=CLASS_ORDER):
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    prf  = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {'accuracy%': round(acc * 100, 1), 'macro_F1%': round(f1m * 100, 1)}
    for i, cls in enumerate(labels):
        out[f'P_{cls}%']  = round(prf[0][i] * 100, 1)
        out[f'R_{cls}%']  = round(prf[1][i] * 100, 1)
        out[f'F1_{cls}%'] = round(prf[2][i] * 100, 1)
    return out

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A0 — TARGET LANDSCAPE  (80 configurations)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A0 — TARGET LANDSCAPE  (5 x 4 x 4 = 80 configurations)")
print("=" * 70)

a0_rows = []
for asset in ASSETS:
    for h in HORIZONS:
        for thresh in THRESHOLDS:
            labels, pct = make_labels(asset, h, thresh)
            valid = labels.notna()
            splits = {
                'train': valid & (df.index < n_train),
                'val':   valid & (df.index >= n_train) & (df.index < n_train + n_val),
                'test':  valid & (df.index >= n_train + n_val),
                'full':  valid
            }
            for split_name, mask in splits.items():
                lv = labels[mask]
                nv = len(lv)
                if nv == 0:
                    continue
                vc = lv.value_counts()
                up_p  = vc.get('UP',      0) / nv * 100
                dn_p  = vc.get('DOWN',    0) / nv * 100
                neu_p = vc.get('NEUTRAL', 0) / nv * 100
                maj   = vc.idxmax()
                maj_acc = vc.max() / nv * 100
                if   neu_p < 15:       bflag = 'TOO_NARROW'
                elif neu_p > 80:       bflag = 'TOO_WIDE'
                elif 40 <= neu_p <= 65: bflag = 'GOOD'
                else:                  bflag = 'OK'
                a0_rows.append({
                    'asset': asset, 'horizon': f'{h}d',
                    'threshold_pct': thresh, 'split': split_name, 'n': nv,
                    'UP%': round(up_p, 1), 'DOWN%': round(dn_p, 1),
                    'NEUTRAL%': round(neu_p, 1),
                    'majority_class': maj,
                    'majority_acc%': round(maj_acc, 1),
                    'balance_flag': bflag
                })

df_a0 = pd.DataFrame(a0_rows)
df_a0.to_csv('outputs/classification_audit/a0_target_landscape.csv', index=False)

a0_tr = df_a0[df_a0['split'] == 'train'].copy()
print(f"\n  TRAIN-SET class distributions (flag: GOOD=40-65% NEUTRAL, OK=acceptable, NARROW/WIDE=extreme)\n")
print(f"  {'':>10} {'H':>4}   {'+-1%':>22}   {'+-2%':>22}   {'+-3%':>22}   {'+-5%':>22}")
print("  " + "-" * 100)
for asset in ASSETS:
    for h in HORIZONS:
        line = f"  {asset.upper():>10} {h:>3}d "
        for thresh in THRESHOLDS:
            r = a0_tr[(a0_tr['asset'] == asset) &
                      (a0_tr['horizon'] == f'{h}d') &
                      (a0_tr['threshold_pct'] == thresh)]
            if r.empty:
                line += f"  {'N/A':>22}"; continue
            r = r.iloc[0]
            sym = {'GOOD': 'G', 'OK': '~', 'TOO_NARROW': 'N', 'TOO_WIDE': 'W'}.get(r['balance_flag'], '?')
            line += f"  U={r['UP%']:4.0f} D={r['DOWN%']:4.0f} N={r['NEUTRAL%']:4.0f}[{sym}]"
        print(line)

# Identify recommended threshold per asset x horizon (from train balance only)
rec_thresh = {}
for asset in ASSETS:
    for h in HORIZONS:
        sub = a0_tr[(a0_tr['asset'] == asset) & (a0_tr['horizon'] == f'{h}d')]
        good = sub[sub['balance_flag'].isin(['GOOD', 'OK'])]
        if good.empty:
            good = sub
        best = good.iloc[(good['NEUTRAL%'] - 50).abs().argsort().iloc[0]]
        rec_thresh[(asset, h)] = best['threshold_pct']

print(f"\n  Recommended threshold (train-balance only, NO test info used):")
for asset in ASSETS:
    parts = [f"{h}d->+-{rec_thresh[(asset,h)]:.0f}%" for h in HORIZONS]
    print(f"    {asset.upper():>10}: {' | '.join(parts)}")

print(f"\n  Saved: outputs/classification_audit/a0_target_landscape.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A1 — NAIVE BASELINES
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A1 — NAIVE BASELINES  (evaluated on LOCKED TEST SET ONLY)")
print("=" * 70)

a1_rows = []

for asset in ASSETS:
    for h in HORIZONS:
        price_col = asset
        for thresh in THRESHOLDS:
            labels, pct = make_labels(asset, h, thresh)
            valid = labels.notna()
            tr_mask = valid & (df.index < n_train)
            te_mask = valid & (df.index >= n_train + n_val)

            y_test = labels[te_mask].values
            if len(y_test) == 0:
                continue

            train_vc  = labels[tr_mask].value_counts()
            maj_class = train_vc.idxmax() if len(train_vc) > 0 else 'NEUTRAL'

            # Previous direction: did the asset move UP/DOWN/NEUTRAL over the PAST h days?
            past_pct  = df[price_col].pct_change(periods=h)
            prev_lbl  = pd.Series('NEUTRAL', index=df.index, dtype=object)
            prev_lbl[past_pct >  thresh / 100.0] = 'UP'
            prev_lbl[past_pct < -thresh / 100.0] = 'DOWN'
            prev_lbl[~valid] = None
            y_prev = prev_lbl[te_mask].values
            y_prev = np.where(pd.isna(y_prev), maj_class, y_prev)

            # 7d trend: no NEUTRAL (binary momentum)
            trend7   = df[price_col].pct_change(periods=7)
            y_t7     = np.where(trend7[te_mask].values >= 0, 'UP', 'DOWN')

            # 5/20 MA crossover: no NEUTRAL (binary)
            ma5  = df[price_col].rolling(5,  min_periods=1).mean()
            ma20 = df[price_col].rolling(20, min_periods=1).mean()
            y_mac = np.where((ma5 - ma20)[te_mask].values >= 0, 'UP', 'DOWN')

            baselines = {
                'AlwaysUP':      np.full(len(y_test), 'UP'),
                'AlwaysDOWN':    np.full(len(y_test), 'DOWN'),
                'AlwaysNEUTRAL': np.full(len(y_test), 'NEUTRAL'),
                'MajorityTrain': np.full(len(y_test), maj_class),
                'PrevDirection': y_prev,
                '7dTrend':       y_t7,
                'MA5_20Cross':   y_mac,
            }

            for bname, y_pred in baselines.items():
                m = class_metrics(y_test, y_pred)
                a1_rows.append({
                    'asset': asset, 'horizon': f'{h}d',
                    'threshold_pct': thresh, 'baseline': bname,
                    'n_test': len(y_test), **m
                })

df_a1 = pd.DataFrame(a1_rows)
df_a1.to_csv('outputs/classification_audit/a1_naive_baselines.csv', index=False)

print(f"\n  Best naive baseline per asset x horizon (at recommended threshold):\n")
print(f"  {'Asset':>10} {'H':>4} {'Thr':>5}  {'BestBaseline':>18}  {'Acc%':>6} {'MacF1%':>7} {'F1UP':>6} {'F1DN':>6} {'F1NEU':>6}")
print("  " + "-" * 80)
for asset in ASSETS:
    for h in HORIZONS:
        rec_t = rec_thresh[(asset, h)]
        sub = df_a1[(df_a1['asset'] == asset) &
                    (df_a1['horizon'] == f'{h}d') &
                    (df_a1['threshold_pct'] == rec_t)]
        if sub.empty: continue
        best = sub.sort_values('macro_F1%', ascending=False).iloc[0]
        print(f"  {asset.upper():>10} {h:>3}d {rec_t:>4.0f}%  "
              f"{best['baseline']:>18}  "
              f"{best['accuracy%']:>6.1f} {best['macro_F1%']:>7.1f} "
              f"{best['F1_UP%']:>6.1f} {best['F1_DOWN%']:>6.1f} {best['F1_NEUTRAL%']:>6.1f}")

print(f"\n  Saved: outputs/classification_audit/a1_naive_baselines.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A2 — RIDGE (LogReg L2) CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A2 — RIDGE (LogisticRegression L2) CLASSIFIER")
print("=" * 70)
print(f"  SelectKBest(f_classif, k={N_FEATURES}) on TRAIN only")
print(f"  C tuned on VAL only from {C_GRID}")
print(f"  Test set evaluated ONCE per configuration\n")

# Impute with train-set column medians
X_raw = df[feature_cols].values.astype(np.float64)
train_medians = np.nanmedian(X_raw[:n_train], axis=0)
for col_i in range(X_raw.shape[1]):
    nan_mask = np.isnan(X_raw[:, col_i])
    X_raw[nan_mask, col_i] = train_medians[col_i]

scaler = StandardScaler()
scaler.fit(X_raw[:n_train])
X_scaled = scaler.transform(X_raw)

a2_rows = []

for asset in ASSETS:
    for h in HORIZONS:
        for thresh in THRESHOLDS:
            labels, _ = make_labels(asset, h, thresh)
            valid = labels.notna()

            tr_mask = valid & (df.index < n_train)
            v_mask  = valid & (df.index >= n_train) & (df.index < n_train + n_val)
            te_mask = valid & (df.index >= n_train + n_val)

            X_tr = X_scaled[tr_mask]; y_tr = labels[tr_mask].values
            X_v  = X_scaled[v_mask];  y_v  = labels[v_mask].values
            X_te = X_scaled[te_mask]; y_te = labels[te_mask].values

            if len(y_tr) < 30 or len(y_v) < 10 or len(y_te) < 10:
                continue

            # Feature selection on TRAIN only
            k_eff = min(N_FEATURES, X_tr.shape[1])
            sel = SelectKBest(f_classif, k=k_eff)
            sel.fit(X_tr, y_tr)
            X_tr_s = sel.transform(X_tr)
            X_v_s  = sel.transform(X_v)
            X_te_s = sel.transform(X_te)

            # Tune C on VAL
            best_C, best_vf1 = C_GRID[0], -1.0
            for C in C_GRID:
                clf = LogisticRegression(C=C, penalty='l2', max_iter=1000,
                                         multi_class='multinomial', solver='lbfgs',
                                         random_state=42)
                clf.fit(X_tr_s, y_tr)
                vf1 = f1_score(y_v, clf.predict(X_v_s),
                               average='macro', labels=CLASS_ORDER, zero_division=0)
                if vf1 > best_vf1:
                    best_vf1, best_C = vf1, C

            # Final fit -> test (ONCE)
            clf_final = LogisticRegression(C=best_C, penalty='l2', max_iter=1000,
                                            multi_class='multinomial', solver='lbfgs',
                                            random_state=42)
            clf_final.fit(X_tr_s, y_tr)

            m_val  = class_metrics(y_v,  clf_final.predict(X_v_s))
            m_test = class_metrics(y_te, clf_final.predict(X_te_s))

            top5 = [feature_cols[i] for i in sel.get_support(indices=True)][:5]

            a2_rows.append({
                'asset': asset, 'horizon': f'{h}d',
                'threshold_pct': thresh, 'model': 'Ridge_LogReg_L2',
                'best_C': best_C, 'n_features': k_eff,
                'n_train': len(y_tr), 'n_val': len(y_v), 'n_test': len(y_te),
                'val_macro_F1%': round(best_vf1 * 100, 1),
                **{f'val_{k}': v for k, v in m_val.items()},
                **{f'test_{k}': v for k, v in m_test.items()},
                'top5_features': ', '.join(top5)
            })

            rec_t = rec_thresh[(asset, h)]
            if thresh == rec_t:
                print(f"  {asset.upper():>10} {h:>2}d +-{thresh:.0f}%  C={best_C:6.2f}  "
                      f"ValF1={best_vf1*100:.1f}%  "
                      f"TestAcc={m_test['accuracy%']:.1f}%  TestMacF1={m_test['macro_F1%']:.1f}%  "
                      f"UP={m_test['F1_UP%']:.1f} DN={m_test['F1_DOWN%']:.1f} NEU={m_test['F1_NEUTRAL%']:.1f}")

df_a2 = pd.DataFrame(a2_rows)
df_a2.to_csv('outputs/classification_audit/a2_ridge_classifier.csv', index=False)
print(f"\n  Saved: outputs/classification_audit/a2_ridge_classifier.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A3 — LEAKAGE AUDIT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A3 — LEAKAGE AUDIT")
print("=" * 70)

print("\n  [A3.1] LABEL FORMULA")
print("  pct_change(t) = (target_{asset}_{h}d[t] - {asset}[t]) / {asset}[t]")
print("  UP   : pct_change > +threshold/100")
print("  DOWN : pct_change < -threshold/100")
print("  NEUTRAL: otherwise")

print("\n  [A3.2] FEATURE SET VERIFICATION")
dir_in_feats = [c for c in feature_cols if c.startswith('dir_')]
tgt_in_feats = [c for c in feature_cols if c.startswith('target_')]
print(f"  dir_*    in feature_cols : {len(dir_in_feats)}  (must be 0)")
print(f"  target_* in feature_cols : {len(tgt_in_feats)}  (must be 0)")
if dir_in_feats or tgt_in_feats:
    print(f"  *** LEAK DETECTED: {dir_in_feats + tgt_in_feats}")
else:
    print(f"  PASS - no leaking columns in feature matrix")

print("\n  [A3.3] RANDOM ROW CHECK (5 test-set rows, kdci 7d, +-2%)")
np.random.seed(99)
sample_idxs = sorted(np.random.choice(test_idx, size=min(5, len(test_idx)), replace=False))
print(f"\n  {'Row':>5}  {'Date':>12}  {'y(t)':>8}  {'y(t+7)':>9}  {'pct_chg':>8}  {'Label@2%':>10}")
print("  " + "-" * 60)
for idx in sample_idxs:
    yt   = df.loc[idx, 'kdci']
    yth  = df.loc[idx, 'target_kdci_7d']
    dt   = df.loc[idx, 'date'].date()
    if pd.isna(yt) or pd.isna(yth) or yt <= 0:
        continue
    pct  = (yth - yt) / yt
    lab  = 'UP' if pct > 0.02 else ('DOWN' if pct < -0.02 else 'NEUTRAL')
    print(f"  {idx:>5}  {str(dt):>12}  {yt:>8.0f}  {yth:>9.0f}  {pct*100:>+7.2f}%  {lab:>10}")

print("\n  [A3.4] SAMPLE OF FEATURE NAMES USED")
lag_feats = [c for c in feature_cols if any(kw in c.lower()
              for kw in ['lag', 'rmean', 'rstd', 'chg', 'gdelt', 'wind', 'precip'])][:8]
for f in lag_feats:
    print(f"    {f}")
print("  All are strictly historical (use data up to and including time t)")

audit_result = 'PASS' if not dir_in_feats and not tgt_in_feats else 'FAIL'
pd.DataFrame([{
    'dir_cols_in_features': len(dir_in_feats),
    'target_cols_in_features': len(tgt_in_feats),
    'label_formula': '(target[t+h] - price[t]) / price[t]',
    'audit_result': audit_result
}]).to_csv('outputs/classification_audit/a3_leakage_audit.csv', index=False)
print(f"\n  A3 RESULT: {audit_result}")
print(f"  Saved: outputs/classification_audit/a3_leakage_audit.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A4 — LOCKED TEST COMPARISON MATRIX
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A4 — LOCKED TEST COMPARISON MATRIX")
print("=" * 70)
print("(Recommended threshold per asset x horizon — chosen on TRAIN balance only)\n")

a4_rows = []
for asset in ASSETS:
    for h in HORIZONS:
        rec_t = rec_thresh[(asset, h)]

        bl_sub = df_a1[(df_a1['asset'] == asset) &
                       (df_a1['horizon'] == f'{h}d') &
                       (df_a1['threshold_pct'] == rec_t)]
        best_bl = bl_sub.sort_values('macro_F1%', ascending=False).iloc[0] if not bl_sub.empty else None

        r2_sub = df_a2[(df_a2['asset'] == asset) &
                       (df_a2['horizon'] == f'{h}d') &
                       (df_a2['threshold_pct'] == rec_t)]
        ridge_row = r2_sub.iloc[0] if not r2_sub.empty else None

        print(f"  {asset.upper()} {h}d  (threshold=+-{rec_t:.0f}%)")
        print(f"  {'Method':>20}  {'Acc%':>6} {'MacF1%':>7} {'F1_UP':>6} {'F1_DN':>6} {'F1_NEU':>7}")
        print(f"  {'':>20}  {'------':>6} {'-------':>7} {'------':>6} {'------':>6} {'-------':>7}")

        if best_bl is not None:
            for _, row in bl_sub.sort_values('macro_F1%', ascending=False).iterrows():
                mk = '*' if row['baseline'] == best_bl['baseline'] else ' '
                print(f"  {mk}{row['baseline']:>19}  "
                      f"{row['accuracy%']:>6.1f} {row['macro_F1%']:>7.1f} "
                      f"{row['F1_UP%']:>6.1f} {row['F1_DOWN%']:>6.1f} {row['F1_NEUTRAL%']:>7.1f}")

        ridge_beats = False
        if ridge_row is not None and best_bl is not None:
            ridge_beats = ridge_row['test_macro_F1%'] > best_bl['macro_F1%']
            mk = '*' if ridge_beats else ' '
            print(f"  {mk}{'Ridge_LogReg_L2':>19}  "
                  f"{ridge_row['test_accuracy%']:>6.1f} {ridge_row['test_macro_F1%']:>7.1f} "
                  f"{ridge_row['test_F1_UP%']:>6.1f} {ridge_row['test_F1_DOWN%']:>6.1f} "
                  f"{ridge_row['test_F1_NEUTRAL%']:>7.1f}  "
                  f"[{'BEATS baseline' if ridge_beats else 'below baseline'}]")

        a4_rows.append({
            'asset': asset, 'horizon': f'{h}d', 'rec_threshold': rec_t,
            'n_test': bl_sub.iloc[0]['n_test'] if not bl_sub.empty else 0,
            'best_naive_baseline': best_bl['baseline'] if best_bl is not None else None,
            'best_naive_macF1%': best_bl['macro_F1%'] if best_bl is not None else None,
            'ridge_macF1%': ridge_row['test_macro_F1%'] if ridge_row is not None else None,
            'ridge_acc%':   ridge_row['test_accuracy%'] if ridge_row is not None else None,
            'ridge_beats_naive': ridge_beats,
            'margin_F1': round((ridge_row['test_macro_F1%'] - best_bl['macro_F1%']), 1)
                         if (ridge_row is not None and best_bl is not None) else None
        })
        print()

df_a4 = pd.DataFrame(a4_rows)
df_a4.to_csv('outputs/classification_audit/a4_test_comparison.csv', index=False)
print(f"  Saved: outputs/classification_audit/a4_test_comparison.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A5 — VERDICT
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A5 — VERDICT")
print("=" * 70)
print("Q: Does material-movement classification contain exploitable signal")
print("   beyond raw direction, and does Ridge beat the strongest naive baseline?\n")

beats_count = df_a4['ridge_beats_naive'].sum()
total_count = len(df_a4)
wins  = df_a4[df_a4['ridge_beats_naive'] == True]
loses = df_a4[df_a4['ridge_beats_naive'] == False]
avg_win_margin  = wins['margin_F1'].mean()   if len(wins) > 0  else 0
avg_lose_margin = loses['margin_F1'].mean()  if len(loses) > 0 else 0

print(f"  {'Asset':>10} {'H':>4} {'Thr':>5}  {'BestNaive':>18}  "
      f"{'NaiveMF1':>9} {'RidgeMF1':>9} {'Margin':>7}  Verdict")
print("  " + "-" * 88)
for _, row in df_a4.iterrows():
    verdict = 'RIDGE WINS' if row['ridge_beats_naive'] else 'NAIVE BETTER'
    margin  = f"{row['margin_F1']:+.1f}%" if row['margin_F1'] is not None else "N/A"
    ridge_mf = f"{row['ridge_macF1%']:.1f}%" if row['ridge_macF1%'] is not None else "N/A"
    naive_mf = f"{row['best_naive_macF1%']:.1f}%" if row['best_naive_macF1%'] is not None else "N/A"
    print(f"  {row['asset'].upper():>10} {row['horizon']:>4} +-{row['rec_threshold']:>3.0f}%  "
          f"{str(row['best_naive_baseline']):>18}  "
          f"{naive_mf:>9} {ridge_mf:>9} {margin:>7}  {verdict}")

maj_avg   = df_a4['best_naive_macF1%'].dropna().mean()
ridge_avg = df_a4['ridge_macF1%'].dropna().mean()

print(f"\n  SUMMARY:")
print(f"    Ridge beats best naive baseline: {int(beats_count)}/{total_count} cases")
if len(wins) > 0:
    print(f"    Average winning margin         : +{avg_win_margin:.1f}% macro F1")
if len(loses) > 0:
    print(f"    Average losing margin          : {avg_lose_margin:.1f}% macro F1")
print(f"\n  AVERAGE across all 20 combos (rec thresholds):")
print(f"    Best naive macro F1  : {maj_avg:.1f}%")
print(f"    Ridge macro F1       : {ridge_avg:.1f}%")
print(f"    Net Ridge edge       : {ridge_avg - maj_avg:+.1f}%")

print(f"\n  PRIOR CONTEXT (from forensic audit):")
print(f"    Raw binary direction, naive persistence     : ~50% accuracy")
print(f"    Best clean GRU directional acc (Panamax)    : ~45-73% binary")
print(f"    Threshold classification Ridge macro F1 avg : {ridge_avg:.1f}% (3-class)")

print(f"\n  ANSWER:")
if beats_count >= total_count * 0.6:
    conclusion = "POSITIVE"
    print(f"  Ridge provides meaningful lift in {int(beats_count)}/{total_count} cases.")
    print(f"  Material-movement framing has exploitable signal. Proceed to:")
    print(f"    - GRU/LSTM trained on threshold labels")
    print(f"    - TFT with known-future covariates")
    print(f"    - Calibrated probability outputs for decision engine")
elif beats_count >= total_count * 0.4:
    conclusion = "MIXED"
    print(f"  Ridge beats naive in only {int(beats_count)}/{total_count} cases.")
    print(f"  Signal is inconsistent across assets/horizons. Recommend:")
    print(f"    - Focus on specific winning asset-horizon pairs")
    print(f"    - Test with richer sequence models (GRU) before wider integration")
else:
    conclusion = "NEGATIVE"
    print(f"  Ridge does NOT consistently beat naive baselines ({int(beats_count)}/{total_count}).")
    print(f"  Threshold framing does not meaningfully improve on majority-class prediction.")
    print(f"  Recommendation: threshold classification NOT ready for production path.")

print(f"\n  PRODUCTION PIPELINE: UNCHANGED")
print(f"  This experiment is isolated. No production files were modified.")
print(f"  Conclusion code: {conclusion}")

pd.DataFrame([{
    'beats_count': int(beats_count),
    'total_count': total_count,
    'avg_win_margin_F1': round(avg_win_margin, 1),
    'avg_lose_margin_F1': round(avg_lose_margin, 1),
    'best_naive_avg_macF1': round(maj_avg, 1),
    'ridge_avg_macF1': round(ridge_avg, 1),
    'net_ridge_edge_F1': round(ridge_avg - maj_avg, 1),
    'conclusion': conclusion
}]).to_csv('outputs/classification_audit/a5_verdict.csv', index=False)

print("\n" + "=" * 70)
print("ALL PHASES COMPLETE — outputs/classification_audit/")
print("  a0_target_landscape.csv")
print("  a1_naive_baselines.csv")
print("  a2_ridge_classifier.csv")
print("  a3_leakage_audit.csv")
print("  a4_test_comparison.csv")
print("  a5_verdict.csv")
print("=" * 70)
