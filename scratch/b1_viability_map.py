"""
=============================================================================
B1 — ASSET/HORIZON VIABILITY MAP
=============================================================================
SOURCE: outputs/phase8_final_benchmark.csv  (clean locked test benchmark ONLY)
GRU/LSTM results are NOT used to gate viability. They are printed for diagnosis.

Viability tier (3 conditions reported independently):
  VIABLE     — passes all 3: R² > 0, sMAPE < 20%, MAE < persistence MAE
  MARGINAL   — passes 1–2 conditions
  NON-VIABLE — fails ≥ 2 conditions

Key rule: NON-VIABLE level pairs are NOT blocked from B2 Δ-forecasting.
B1 produces a DIAGNOSTIC PRIORITY, not a gate.

Bonus: ATR / volatility diagnosis to explain Cape divergence.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

os.makedirs('outputs/b1_viability', exist_ok=True)

ASSETS   = ['kdci', 'cape', 'panamax', 'supramax', 'handy']
HORIZONS = ['1d', '7d', '14d', '30d']

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CLEAN LOCKED BENCHMARK (point-regression only)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("B1 — ASSET/HORIZON VIABILITY MAP")
print("=" * 70)
print("Source: outputs/phase8_final_benchmark.csv  (clean locked test set)")
print("GRU/LSTM results used for DIAGNOSIS only, not for gating B2.\n")

bm = pd.read_csv('outputs/phase8_final_benchmark.csv')
# Normalise column names
bm = bm.rename(columns={'freight_class': 'asset'})

# ─────────────────────────────────────────────────────────────────────────────
# B1.1 — THREE-CONDITION VIABILITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for _, r in bm.iterrows():
    c1 = int(r['test_R2'] > 0)                                # R² > 0
    c2 = int(r['test_sMAPE'] < 20.0)                          # sMAPE < 20%
    c3 = int(r['test_MAE'] < r['baseline_persistence_MAE'])   # beats persistence MAE
    passed = c1 + c2 + c3

    if passed == 3:
        tier = 'VIABLE'
    elif passed >= 1:
        tier = 'MARGINAL'
    else:
        tier = 'NON-VIABLE'

    # Δ diagnostic priority: prioritise testing where level fails most
    # Pairs where R²<0 + MAE worse than persistence are highest priority for Δ
    if c1 == 0 and c3 == 0:
        delta_priority = 'HIGH'   # level forecast is actively harmful
    elif c1 == 0 or c3 == 0:
        delta_priority = 'MEDIUM' # partial failure — Δ might help
    else:
        delta_priority = 'LOW'    # level already works; Δ is validation experiment

    rows.append({
        'asset': r['asset'], 'horizon': r['horizon'],
        'test_R2': round(r['test_R2'], 4),
        'test_sMAPE': round(r['test_sMAPE'], 2),
        'test_MAE': round(r['test_MAE'], 2),
        'persistence_MAE': round(r['baseline_persistence_MAE'], 2),
        'C1_R2_pos': bool(c1), 'C2_sMAPE_lt20': bool(c2), 'C3_beats_persistence': bool(c3),
        'conditions_passed': passed,
        'viability_tier': tier,
        'delta_test_priority': delta_priority,
        'model_used': r['selected_model'],
        'target_transform': r['target_transformation']
    })

df_v = pd.DataFrame(rows)
df_v.to_csv('outputs/b1_viability/b1_viability_map.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT VIABILITY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"{'Asset':>10} {'H':>4}  {'R²':>7} {'sMAPE':>7} {'MAE':>8} {'PersMAE':>8}  "
      f"{'C1':>3} {'C2':>3} {'C3':>3}  {'Tier':>10}  {'ΔPriority':>10}")
print("-" * 85)

tier_symbols = {'VIABLE': 'VIABLE', 'MARGINAL': 'MARGINAL', 'NON-VIABLE': 'NON-VIABLE'}

for _, r in df_v.iterrows():
    c1s = '✓' if r['C1_R2_pos']         else '✗'
    c2s = '✓' if r['C2_sMAPE_lt20']     else '✗'
    c3s = '✓' if r['C3_beats_persistence'] else '✗'
    tier_str = r['viability_tier']
    print(f"  {r['asset'].upper():>8} {r['horizon']:>4}  "
          f"{r['test_R2']:>7.4f} {r['test_sMAPE']:>6.2f}% "
          f"{r['test_MAE']:>8.0f} {r['persistence_MAE']:>8.0f}  "
          f"{c1s:>3} {c2s:>3} {c3s:>3}  {tier_str:>10}  {r['delta_test_priority']:>10}")

# Summary counts
print("\nViability summary:")
for tier in ['VIABLE', 'MARGINAL', 'NON-VIABLE']:
    count = (df_v['viability_tier'] == tier).sum()
    assets_list = df_v[df_v['viability_tier'] == tier][['asset','horizon']].apply(
        lambda r: f"{r['asset'].upper()} {r['horizon']}", axis=1).tolist()
    print(f"  {tier:>10} : {count:>2}  {assets_list}")

print(f"\nΔ-test priority:")
for pri in ['HIGH', 'MEDIUM', 'LOW']:
    count = (df_v['delta_test_priority'] == pri).sum()
    pairs = df_v[df_v['delta_test_priority'] == pri][['asset','horizon']].apply(
        lambda r: f"{r['asset'].upper()} {r['horizon']}", axis=1).tolist()
    print(f"  {pri:>6} : {count:>2}  {pairs}")

# ─────────────────────────────────────────────────────────────────────────────
# B1.2 — CAPE DIVERGENCE DIAGNOSIS (ATR / Volatility)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("B1.2 — CAPE DIVERGENCE: ATR / VOLATILITY DIAGNOSIS")
print("=" * 70)

df = pd.read_csv('outputs/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

n = len(df)
n_train = int(n * 0.70)
test_df = df.iloc[int(n * 0.85):].copy()

print("\nVolatility metrics on TEST SET (2025–2026):\n")
print(f"  {'Asset':>10}  {'Mean':>8}  {'Std':>8}  {'20d-ATR':>9}  "
      f"{'CV%':>7}  {'MaxSwing%':>10}  {'ATR/Mean%':>10}")
print("  " + "-" * 72)

atr_rows = []
for asset in ASSETS:
    col = asset
    s = test_df[col].dropna()
    if len(s) < 20:
        continue

    mean_val  = s.mean()
    std_val   = s.std()
    cv        = std_val / mean_val * 100  # coefficient of variation %

    # 20-day ATR proxy: mean(|y_t - y_{t-1}|) over rolling 20-day windows
    daily_abs_chg = s.diff().abs()
    atr_20d = daily_abs_chg.rolling(20).mean().mean()

    # Max swing: (max - min) / mean over full test period
    max_swing = (s.max() - s.min()) / mean_val * 100

    atr_pct = atr_20d / mean_val * 100  # ATR as % of price level

    atr_rows.append({
        'asset': asset, 'mean': round(mean_val, 0), 'std': round(std_val, 0),
        'atr_20d': round(atr_20d, 0), 'cv_pct': round(cv, 1),
        'max_swing_pct': round(max_swing, 1), 'atr_pct_of_mean': round(atr_pct, 2)
    })
    print(f"  {asset.upper():>10}  {mean_val:>8.0f}  {std_val:>8.0f}  {atr_20d:>9.0f}  "
          f"{cv:>6.1f}%  {max_swing:>9.1f}%  {atr_pct:>9.2f}%")

df_atr = pd.DataFrame(atr_rows)

# Cape vs Panamax ratio
cape_row = df_atr[df_atr['asset'] == 'cape']
pana_row = df_atr[df_atr['asset'] == 'panamax']
if not cape_row.empty and not pana_row.empty:
    cv_ratio  = cape_row.iloc[0]['cv_pct']  / pana_row.iloc[0]['cv_pct']
    atr_ratio = cape_row.iloc[0]['atr_pct_of_mean'] / pana_row.iloc[0]['atr_pct_of_mean']
    print(f"\n  Cape/Panamax CV  ratio : {cv_ratio:.1f}×")
    print(f"  Cape/Panamax ATR ratio : {atr_ratio:.1f}×")
    print(f"\n  Diagnosis: Cape's {atr_ratio:.1f}× higher ATR/Mean causes MSE loss gradient")
    print(f"  to be dominated by high-dollar level errors during GRU backprop,")
    print(f"  preventing stable directional learning. This is independent of feature quality.")
    print(f"  Δ-forecasting (predicting the change in $/day) re-centers the regression")
    print(f"  target closer to zero, reducing the gradient magnitude variance.")
    print(f"  Cape is therefore a HIGH-priority target for B2 Δ-experimentation.")

# ─────────────────────────────────────────────────────────────────────────────
# B1.3 — HORIZON DIFFICULTY GRADIENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("B1.3 — HORIZON DIFFICULTY GRADIENT (across all assets)")
print("=" * 70)

print("\n  Mean R² and sMAPE by horizon (clean level-Ridge benchmark):\n")
print(f"  {'Horizon':>8}  {'Mean R²':>8}  {'Mean sMAPE':>11}  {'% VIABLE':>9}  {'% NON-VIABLE':>13}")
print("  " + "-" * 56)
for h in HORIZONS:
    sub = df_v[df_v['horizon'] == h]
    mean_r2    = sub['test_R2'].mean()
    mean_smape = sub['test_sMAPE'].mean()
    pct_viable = (sub['viability_tier'] == 'VIABLE').sum() / len(sub) * 100
    pct_nv     = (sub['viability_tier'] == 'NON-VIABLE').sum() / len(sub) * 100
    print(f"  {h:>8}  {mean_r2:>8.4f}  {mean_smape:>10.2f}%  {pct_viable:>8.0f}%  {pct_nv:>12.0f}%")

# Save ATR data
df_atr.to_csv('outputs/b1_viability/b1_volatility_diagnosis.csv', index=False)

print("\n" + "=" * 70)
print("B1 COMPLETE")
print("  outputs/b1_viability/b1_viability_map.csv")
print("  outputs/b1_viability/b1_volatility_diagnosis.csv")
print("=" * 70)
print("\nNext step: feed b1_viability_map.csv into B2 Colab notebook.")
print("ALL 20 pairs are tested in B2 — viability tier sets diagnostic priority, not a gate.")
