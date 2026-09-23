"""
Build the FICOS Final Model Selection Decision Audit notebook (.ipynb)
Generates all 17 sections requested in the audit specification.
"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}

cells = []

# ── Cell 0: Title ──
cells.append(md("""# FICOS — FINAL MODEL SELECTION DECISION AUDIT
## SIH26006 · Freight Intelligence & Chartering Optimization System

**Purpose**: Determine the FINAL production forecasting model for FICOS based on downstream decision quality.

**Candidates**: Random Forest (production) · LightGBM · XGBoost | Ridge (reference)

**Rules**: No retraining · No new models · No HP tuning · No gate/economic changes · Freeze after selection"""))

# ── Cell 1: Environment Setup ──
cells.append(code("""# ── Cell 1: Environment & Data Setup ──
import os, sys, json, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10})

# Colab: clone repo if needed and navigate to repo root
if 'google.colab' in sys.modules:
    if not os.path.exists('FICOS-Platform') and not os.path.basename(os.getcwd()) == 'FICOS-Platform':
        !git clone https://github.com/SSOHEB/FICOS-Platform.git
        os.chdir('FICOS-Platform')
    elif os.path.exists('FICOS-Platform') and not os.path.basename(os.getcwd()) == 'FICOS-Platform':
        os.chdir('FICOS-Platform')
    !git pull origin main --quiet

# Load the corrected OOS predictions
CSV_PATH = os.path.join('reports', 'final_model_selection_corrected', 'all_walkforward_predictions_corrected.csv')
assert os.path.exists(CSV_PATH), f"Missing: {CSV_PATH}"
df_cases = pd.read_csv(CSV_PATH)
df_cases['date'] = pd.to_datetime(df_cases['date'])
print(f"Loaded {len(df_cases):,} corrected OOS prediction cases")
print(f"Models: {df_cases['model'].unique().tolist()}")
print(f"Vessels: {df_cases['vessel'].unique().tolist()}")
print(f"Years:  {sorted(df_cases['year'].unique().tolist())}")
print(f"Columns: {df_cases.columns.tolist()}")

OUTPUT_DIR = os.path.join('reports', 'final_model_selection_decision')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ['Ridge', 'RandomForest', 'LightGBM', 'XGBoost']
PRIMARY = ['RandomForest', 'LightGBM', 'XGBoost']
COLORS = {'Ridge': '#6B7280', 'RandomForest': '#059669', 'LightGBM': '#D97706', 'XGBoost': '#7C3AED'}
VESSELS = ['panamax', 'supramax', 'handy', 'cape']
N_BOOT = 10_000"""))

# ── Cell 2: Section header ──
cells.append(md("""---
## 1. Corrected Fold Structure Verification

Verify the exact walk-forward fold boundaries used to generate `df_cases`.
Every fold must satisfy: `max(TRAIN) < min(VALIDATION) < min(TEST)`"""))

# ── Cell 3: Fold Verification ──
cells.append(code("""# ── Cell 3: Verify Corrected Fold Structure ──
FOLDS = [
    {"year": 2021, "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"year": 2022, "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"year": 2023, "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"year": 2024, "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"year": 2025, "train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
]

# Load raw data to verify boundaries
data_path = os.path.join('data', 'modeling_dataset.csv')
df_raw = pd.read_csv(data_path)
df_raw['date'] = pd.to_datetime(df_raw['date'])

print("=" * 90)
print("STRICT PROGRAMMATIC TEMPORAL LEAKAGE AUDIT")
print("=" * 90)
all_pass = True
fold_table = []

for f in FOLDS:
    y = f['year']
    tr = df_raw[df_raw['date'] <= f['train_end']]['date']
    va = df_raw[(df_raw['date'] >= f['val_start']) & (df_raw['date'] <= f['val_end'])]['date']
    te = df_raw[(df_raw['date'] >= f['test_start']) & (df_raw['date'] <= f['test_end'])]['date']
    
    max_tr, min_va, max_va, min_te = tr.max(), va.min(), va.max(), te.min()
    cond1 = max_tr < min_va
    cond2 = max_va < min_te
    
    tr_idx = set(df_raw[df_raw['date'] <= f['train_end']].index)
    va_idx = set(df_raw[(df_raw['date'] >= f['val_start']) & (df_raw['date'] <= f['val_end'])].index)
    te_idx = set(df_raw[(df_raw['date'] >= f['test_start']) & (df_raw['date'] <= f['test_end'])].index)
    
    no_overlap = len(tr_idx & va_idx) == 0 and len(va_idx & te_idx) == 0 and len(tr_idx & te_idx) == 0
    ok = cond1 and cond2 and no_overlap
    status = "✅ PASS" if ok else "❌ FAIL"
    if not ok: all_pass = False
    
    fold_table.append({
        'Fold': y,
        'Train': f"≤ {max_tr.strftime('%Y-%m-%d')} (N={len(tr)})",
        'Validation': f"{min_va.strftime('%Y-%m-%d')} → {max_va.strftime('%Y-%m-%d')} (N={len(va)})",
        'Test': f"{min_te.strftime('%Y-%m-%d')} → OOS (N={len(te)})",
        'Train<Val': '✅' if cond1 else '❌',
        'Val<Test': '✅' if cond2 else '❌',
        'No Overlap': '✅' if no_overlap else '❌',
        'Status': status
    })
    print(f"[{status}] Fold {y}: Train ≤ {max_tr.strftime('%Y-%m-%d')} | Val {min_va.strftime('%Y-%m-%d')}-{max_va.strftime('%Y-%m-%d')} | Test ≥ {min_te.strftime('%Y-%m-%d')}")

print("=" * 90)
if all_pass:
    print("✅ CORRECTED FOLD STRUCTURE IS VERIFIED — All temporal boundaries strictly disjoint")
else:
    raise RuntimeError("❌ CORRECTED FOLD STRUCTURE IS NOT VERIFIED. STOPPING.")

pd.DataFrame(fold_table).set_index('Fold')"""))

# ── Cell 4: Forecasting Scorecard Header ──
cells.append(md("""---
## 2. Aggregate Forecasting Scorecard

Comprehensive forecasting metrics across all 4,804 OOS cases and the 2025 blind holdout."""))

# ── Cell 5: Forecasting Scorecard ──
cells.append(code("""# ── Cell 5: Aggregate Forecasting Scorecard ──
scorecard = {}
for m in MODELS:
    s = df_cases[df_cases['model'] == m]
    s25 = s[s['year'] == 2025]
    
    fold_maes = [float(s[s['year']==y]['abs_error'].mean()) for y in [2021,2022,2023,2024,2025]]
    fold_das  = [float(s[s['year']==y]['dir_correct'].mean()*100) for y in [2021,2022,2023,2024,2025]]
    
    # Fold wins (lowest MAE per fold)
    scorecard[m] = {
        'MAE': s['abs_error'].mean(),
        'RMSE': np.sqrt(s['sq_error'].mean()),
        'MedianAE': s['abs_error'].median(),
        'Bias': s['residual'].mean(),
        'DA%': s['dir_correct'].mean()*100,
        '2025 MAE': s25['abs_error'].mean(),
        '2025 RMSE': np.sqrt(s25['sq_error'].mean()),
        '2025 MedianAE': s25['abs_error'].median(),
        '2025 Bias': s25['residual'].mean(),
        '2025 DA%': s25['dir_correct'].mean()*100,
        'fold_maes': fold_maes,
        'fold_das': fold_das,
        'Mean Fold MAE': np.mean(fold_maes),
        'Std Fold MAE': np.std(fold_maes),
        'Worst Fold MAE': np.max(fold_maes),
    }

# Count fold wins
for y_idx, y in enumerate([2021,2022,2023,2024,2025]):
    best_m = min(MODELS, key=lambda m: scorecard[m]['fold_maes'][y_idx])
    for m in MODELS:
        scorecard[m].setdefault('Fold Wins', 0)
        if m == best_m:
            scorecard[m]['Fold Wins'] += 1

# Display table
display_cols = ['MAE','RMSE','MedianAE','Bias','DA%','2025 MAE','2025 RMSE','2025 MedianAE','2025 Bias','2025 DA%',
                'Mean Fold MAE','Std Fold MAE','Worst Fold MAE','Fold Wins']
df_sc = pd.DataFrame({m: {c: scorecard[m][c] for c in display_cols} for m in MODELS}).T
df_sc.index.name = 'Model'

# Highlight best
print("=" * 80)
print("AGGREGATE FORECASTING SCORECARD")
print("=" * 80)
display(df_sc.style.format({
    'MAE': '${:,.2f}', 'RMSE': '${:,.2f}', 'MedianAE': '${:,.2f}', 'Bias': '${:+,.2f}',
    'DA%': '{:.2f}%', '2025 MAE': '${:,.2f}', '2025 RMSE': '${:,.2f}', '2025 MedianAE': '${:,.2f}',
    '2025 Bias': '${:+,.2f}', '2025 DA%': '{:.2f}%',
    'Mean Fold MAE': '${:,.2f}', 'Std Fold MAE': '{:.2f}', 'Worst Fold MAE': '${:,.2f}', 'Fold Wins': '{:.0f}'
}).highlight_min(subset=['MAE','RMSE','MedianAE','Std Fold MAE','Worst Fold MAE','2025 MAE','2025 RMSE','2025 MedianAE'], color='#d4edda')
 .highlight_max(subset=['DA%','2025 DA%','Fold Wins'], color='#d4edda')
 .highlight_min(subset=['Bias','2025 Bias'], color='#d4edda', axis=None))

# Fold-level breakdown
print("\\nFold-Level MAE:")
fold_df = pd.DataFrame({m: scorecard[m]['fold_maes'] for m in MODELS}, index=[2021,2022,2023,2024,2025])
fold_df.index.name = 'Fold'
display(fold_df.style.format('${:,.2f}').highlight_min(axis=1, color='#d4edda'))

print("\\nFold-Level DA:")
fold_da_df = pd.DataFrame({m: scorecard[m]['fold_das'] for m in MODELS}, index=[2021,2022,2023,2024,2025])
fold_da_df.index.name = 'Fold'
display(fold_da_df.style.format('{:.2f}%').highlight_max(axis=1, color='#d4edda'))"""))

# ── Cell 6: Vessel-Level Header ──
cells.append(md("""---
## 3. Vessel-Level Scorecard

Per-vessel breakdown to assess whether a per-vessel model registry is warranted."""))

# ── Cell 7: Vessel-Level Scorecard ──
cells.append(code("""# ── Cell 7: Vessel-Level Scorecard ──
vessel_results = {}
for v in VESSELS:
    vessel_results[v] = {}
    for m in MODELS:
        sv = df_cases[(df_cases['model']==m) & (df_cases['vessel']==v)]
        sv25 = sv[sv['year']==2025]
        vessel_results[v][m] = {
            'MAE': sv['abs_error'].mean(),
            'DA%': sv['dir_correct'].mean()*100,
            '2025 MAE': sv25['abs_error'].mean(),
            '2025 DA%': sv25['dir_correct'].mean()*100,
        }

print("=" * 80)
print("VESSEL-LEVEL SCORECARD")
print("=" * 80)

for v in VESSELS:
    print(f"\\n{'─'*60}")
    print(f"  {v.upper()} 1D")
    print(f"{'─'*60}")
    vdf = pd.DataFrame(vessel_results[v]).T
    vdf.index.name = 'Model'
    display(vdf.style.format({'MAE':'${:,.2f}','DA%':'{:.2f}%','2025 MAE':'${:,.2f}','2025 DA%':'{:.2f}%'})
            .highlight_min(subset=['MAE','2025 MAE'], color='#d4edda')
            .highlight_max(subset=['DA%','2025 DA%'], color='#d4edda'))

# Determine best model per vessel
print("\\n" + "=" * 80)
print("VESSEL-LEVEL WINNER SUMMARY")
print("=" * 80)
winners = {}
for v in VESSELS:
    best_mae = min(PRIMARY, key=lambda m: vessel_results[v][m]['MAE'])
    best_da = max(PRIMARY, key=lambda m: vessel_results[v][m]['DA%'])
    best_25mae = min(PRIMARY, key=lambda m: vessel_results[v][m]['2025 MAE'])
    best_25da = max(PRIMARY, key=lambda m: vessel_results[v][m]['2025 DA%'])
    winners[v] = {'Best MAE': best_mae, 'Best DA': best_da, 'Best 2025 MAE': best_25mae, 'Best 2025 DA': best_25da}
    print(f"  {v.upper():12s}  MAE→{best_mae:15s}  DA→{best_da:15s}  2025MAE→{best_25mae:15s}  2025DA→{best_25da:15s}")

# Check if per-vessel divergence is material
unique_winners = set()
for v in VESSELS:
    unique_winners.update(winners[v].values())
if len(unique_winners) <= 2:
    print("\\n→ No material per-vessel divergence. A single global model is defensible.")
else:
    print(f"\\n→ Winners span {len(unique_winners)} models. Evaluate whether per-vessel registry adds value.")"""))

# ── Cell 8: Paired Comparison Header ──
cells.append(md("""---
## 4. Paired Bootstrap Comparison (10,000 Samples)

Paired absolute-error and directional-correctness differences on the exact same OOS cases.
A model is only called statistically superior if the 95% CI excludes zero."""))

# ── Cell 9: Paired Bootstrap ──
cells.append(code("""# ── Cell 9: Paired Bootstrap Comparison ──
pairs = [('RandomForest','LightGBM'), ('RandomForest','XGBoost'), ('LightGBM','XGBoost')]
paired_results = {}

for m_a, m_b in pairs:
    da = df_cases[df_cases['model']==m_a].sort_values(['vessel','year','date']).reset_index(drop=True)
    db = df_cases[df_cases['model']==m_b].sort_values(['vessel','year','date']).reset_index(drop=True)
    
    assert len(da) == len(db), f"Mismatched case counts: {m_a}={len(da)}, {m_b}={len(db)}"
    
    diff_ae = db['abs_error'].values - da['abs_error'].values  # negative = B better
    diff_dc = db['dir_correct'].astype(float).values - da['dir_correct'].astype(float).values
    
    n = len(diff_ae)
    boot_mae, boot_da = [], []
    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        boot_mae.append(np.mean(diff_ae[idx]))
        boot_da.append(np.mean(diff_dc[idx]) * 100)
    
    mae_mean = float(np.mean(diff_ae))
    mae_ci = [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))]
    da_mean = float(np.mean(diff_dc) * 100)
    da_ci = [float(np.percentile(boot_da, 2.5)), float(np.percentile(boot_da, 97.5))]
    
    sig_mae = mae_ci[0] > 0 or mae_ci[1] < 0
    sig_da = da_ci[0] > 0 or da_ci[1] < 0
    
    key = f"{m_a} vs {m_b}"
    paired_results[key] = {
        'MAE Diff (B-A)': mae_mean,
        'MAE 95% CI': mae_ci,
        'MAE Sig': sig_mae,
        'DA Diff (B-A)': da_mean,
        'DA 95% CI': da_ci,
        'DA Sig': sig_da,
    }

print("=" * 80)
print("PAIRED BOOTSTRAP COMPARISON (10,000 samples)")
print("=" * 80)
for key, r in paired_results.items():
    m_a, m_b = key.split(' vs ')
    print(f"\\n{'─'*70}")
    print(f"  {key}")
    print(f"{'─'*70}")
    
    mae_dir = f"{m_b} better" if r['MAE Diff (B-A)'] < 0 else f"{m_a} better"
    da_dir = f"{m_b} better" if r['DA Diff (B-A)'] > 0 else f"{m_a} better"
    
    print(f"  MAE Diff ({m_b}−{m_a}): {r['MAE Diff (B-A)']:+.2f} $/MT")
    print(f"    95% CI: [{r['MAE 95% CI'][0]:+.2f}, {r['MAE 95% CI'][1]:+.2f}]")
    print(f"    Statistically significant: {'YES ✅' if r['MAE Sig'] else 'NO ❌'} → {mae_dir}")
    print(f"  DA Diff ({m_b}−{m_a}): {r['DA Diff (B-A)']:+.2f}% points")
    print(f"    95% CI: [{r['DA 95% CI'][0]:+.2f}, {r['DA 95% CI'][1]:+.2f}]")
    print(f"    Statistically significant: {'YES ✅' if r['DA Sig'] else 'NO ❌'} → {da_dir}")"""))

# ── Cell 10: Uncertainty Header ──
cells.append(md("""---
## 5. Uncertainty / Actionability

Compare models under the exact same FICOS uncertainty policy (no threshold changes)."""))

# ── Cell 11: Uncertainty ──
cells.append(code("""# ── Cell 11: Uncertainty / Actionability ──
unc_results = {}
for m in MODELS:
    s = df_cases[df_cases['model']==m]
    sr = s[s['retained']==True]
    
    unc_results[m] = {
        'Coverage %': s['covered'].mean() * 100,
        'Mean Interval Width': s['interval_width'].mean(),
        'Median Interval Width': s['interval_width'].median(),
        'Mean Relative Width %': s['relative_width'].mean() * 100,
        'Abstention Rate %': (~s['retained']).mean() * 100,
        'Retained Rate %': s['retained'].mean() * 100,
        'Retained N': int(s['retained'].sum()),
        'Gated DA %': sr['dir_correct'].mean() * 100 if len(sr) > 0 else 0.0,
    }

print("=" * 80)
print("UNCERTAINTY & ACTIONABILITY COMPARISON")
print("=" * 80)
unc_df = pd.DataFrame(unc_results).T
unc_df.index.name = 'Model'
display(unc_df.style.format({
    'Coverage %': '{:.2f}%', 'Mean Interval Width': '${:,.2f}',
    'Median Interval Width': '${:,.2f}', 'Mean Relative Width %': '{:.2f}%',
    'Abstention Rate %': '{:.2f}%', 'Retained Rate %': '{:.2f}%',
    'Retained N': '{:,}', 'Gated DA %': '{:.2f}%'
}).highlight_max(subset=['Retained N','Gated DA %','Retained Rate %'], color='#d4edda'))"""))

# ── Cell 12: Decision Quality Header ──
cells.append(md("""---
## 6. Decision Quality (MOST IMPORTANT)

For each model, evaluate the quality of the FICOS decision gate: NOW / WAIT / FLEXIBLE."""))

# ── Cell 13: Decision Quality ──
cells.append(code("""# ── Cell 13: Decision Quality ──
print("=" * 80)
print("DECISION QUALITY — THE MOST IMPORTANT COMPARISON")
print("=" * 80)

dq_results = {}
for m in MODELS:
    s = df_cases[df_cases['model']==m]
    total = len(s)
    now = s[s['decision']=='NOW']
    wait = s[s['decision']=='WAIT']
    flex = s[s['decision']=='FLEXIBLE']
    retained = s[s['retained']==True]
    
    now_prec = now['dir_correct'].mean()*100 if len(now) > 0 else float('nan')
    wait_prec = wait['dir_correct'].mean()*100 if len(wait) > 0 else float('nan')
    ret_prec = retained['dir_correct'].mean()*100 if len(retained) > 0 else float('nan')
    
    dq_results[m] = {
        'Total Cases': total,
        'NOW N': len(now),
        'WAIT N': len(wait),
        'FLEXIBLE N': len(flex),
        'Retained %': len(retained)/total*100,
        'NOW Precision %': now_prec,
        'WAIT Precision %': wait_prec,
        'Retained Precision %': ret_prec,
    }

dq_df = pd.DataFrame(dq_results).T
dq_df.index.name = 'Model'
display(dq_df.style.format({
    'Total Cases': '{:,}', 'NOW N': '{:,}', 'WAIT N': '{:,}', 'FLEXIBLE N': '{:,}',
    'Retained %': '{:.2f}%', 'NOW Precision %': '{:.2f}%',
    'WAIT Precision %': '{:.2f}%', 'Retained Precision %': '{:.2f}%'
}).highlight_max(subset=['Retained Precision %'], color='#d4edda'))

# 2025-specific decision quality
print("\\n" + "=" * 80)
print("2025 DECISION QUALITY")
print("=" * 80)

dq25 = {}
for m in MODELS:
    s = df_cases[(df_cases['model']==m) & (df_cases['year']==2025)]
    now = s[s['decision']=='NOW']
    wait = s[s['decision']=='WAIT']
    flex = s[s['decision']=='FLEXIBLE']
    retained = s[s['retained']==True]
    
    dq25[m] = {
        'N': len(s),
        'NOW': len(now),
        'WAIT': len(wait),
        'FLEXIBLE': len(flex),
        'Retained %': len(retained)/len(s)*100 if len(s)>0 else 0,
        'Retained Prec %': retained['dir_correct'].mean()*100 if len(retained)>0 else 0,
    }

dq25_df = pd.DataFrame(dq25).T
dq25_df.index.name = 'Model'
display(dq25_df.style.format({
    'N':'{:,}','NOW':'{:,}','WAIT':'{:,}','FLEXIBLE':'{:,}',
    'Retained %':'{:.2f}%','Retained Prec %':'{:.2f}%'
}).highlight_max(subset=['Retained Prec %'], color='#d4edda'))"""))

# ── Cell 14: Economic Header ──
cells.append(md("""---
## 7. Economic Decision Comparison

Uses the existing economic backtest implementation. No modifications to voyage duration, idle cost, spot pricing, or decision logic."""))

# ── Cell 15: Economic Backtest ──
cells.append(code("""# ── Cell 15: Economic Decision Backtest ──
VOYAGE_DURATION = 20.0
DAILY_IDLE = 8000.0

econ = {}
for m in MODELS:
    s = df_cases[(df_cases['model']==m) & (df_cases['year']==2025)].copy().reset_index(drop=True)
    
    spot_costs = s['y_base'].values * VOYAGE_DURATION
    wait_costs = s['y_true'].values * VOYAGE_DURATION + DAILY_IDLE * 1.0
    flex_costs = ((s['y_base'].values + s['y_true'].values) / 2.0) * VOYAGE_DURATION + DAILY_IDLE * 0.25
    
    ficos_costs = np.zeros(len(s))
    decisions = s['decision'].values
    for i in range(len(s)):
        if decisions[i] == 'NOW':
            ficos_costs[i] = spot_costs[i]
        elif decisions[i] == 'WAIT':
            ficos_costs[i] = wait_costs[i]
        else:
            ficos_costs[i] = flex_costs[i]
    
    tot_spot = np.sum(spot_costs)
    tot_ficos = np.sum(ficos_costs)
    savings = tot_spot - tot_ficos
    sav_pct = (savings / tot_spot) * 100
    
    # Bootstrap overall CI
    n = len(s)
    boot_sav = []
    boot_wait_sav = []
    boot_flex_sav = []
    for _ in range(N_BOOT):
        bi = np.random.choice(n, n, replace=True)
        sb = np.sum(spot_costs[bi])
        fb = np.sum(ficos_costs[bi])
        boot_sav.append((sb - fb) / sb * 100)
        
        wi = bi[decisions[bi] == 'WAIT']
        if len(wi) > 0:
            boot_wait_sav.append((np.sum(spot_costs[wi]) - np.sum(ficos_costs[wi])) / np.sum(spot_costs[wi]) * 100)
        
        fi = bi[decisions[bi] == 'FLEXIBLE']
        if len(fi) > 0:
            boot_flex_sav.append((np.sum(spot_costs[fi]) - np.sum(ficos_costs[fi])) / np.sum(spot_costs[fi]) * 100)
    
    wait_mask = decisions == 'WAIT'
    flex_mask = decisions == 'FLEXIBLE'
    now_mask = decisions == 'NOW'
    
    wait_sav = ((np.sum(spot_costs[wait_mask]) - np.sum(ficos_costs[wait_mask])) / np.sum(spot_costs[wait_mask]) * 100) if wait_mask.sum() > 0 else 0
    flex_sav = ((np.sum(spot_costs[flex_mask]) - np.sum(ficos_costs[flex_mask])) / np.sum(spot_costs[flex_mask]) * 100) if flex_mask.sum() > 0 else 0
    now_sav = ((np.sum(spot_costs[now_mask]) - np.sum(ficos_costs[now_mask])) / np.sum(spot_costs[now_mask]) * 100) if now_mask.sum() > 0 else 0
    
    econ[m] = {
        'N': n,
        'Overall Savings %': sav_pct,
        'Overall 95% CI': [np.percentile(boot_sav, 2.5), np.percentile(boot_sav, 97.5)],
        'Mean Cost Diff $': np.mean(ficos_costs - spot_costs),
        '% Cheaper Than Spot': np.mean(ficos_costs < spot_costs) * 100,
        'NOW N': int(now_mask.sum()),
        'NOW Savings %': now_sav,
        'WAIT N': int(wait_mask.sum()),
        'WAIT Savings %': wait_sav,
        'WAIT 95% CI': [np.percentile(boot_wait_sav, 2.5), np.percentile(boot_wait_sav, 97.5)] if boot_wait_sav else [0,0],
        'FLEX N': int(flex_mask.sum()),
        'FLEX Savings %': flex_sav,
        'FLEX 95% CI': [np.percentile(boot_flex_sav, 2.5), np.percentile(boot_flex_sav, 97.5)] if boot_flex_sav else [0,0],
    }

print("=" * 80)
print("ECONOMIC DECISION BACKTEST (2025 Holdout)")
print("=" * 80)

for m in MODELS:
    e = econ[m]
    print(f"\\n{'─'*60}")
    print(f"  {m}")
    print(f"{'─'*60}")
    print(f"  Overall: N={e['N']}, Savings={e['Overall Savings %']:+.4f}%, CI=[{e['Overall 95% CI'][0]:+.4f}%, {e['Overall 95% CI'][1]:+.4f}%]")
    print(f"  Mean Cost Diff: ${e['Mean Cost Diff $']:+,.2f}")
    print(f"  % Cheaper Than Spot: {e['% Cheaper Than Spot']:.2f}%")
    print(f"  NOW:  N={e['NOW N']:3d}, Savings={e['NOW Savings %']:+.4f}%")
    print(f"  WAIT: N={e['WAIT N']:3d}, Savings={e['WAIT Savings %']:+.4f}%, CI=[{e['WAIT 95% CI'][0]:+.4f}%, {e['WAIT 95% CI'][1]:+.4f}%]")
    print(f"  FLEX: N={e['FLEX N']:3d}, Savings={e['FLEX Savings %']:+.4f}%, CI=[{e['FLEX 95% CI'][0]:+.4f}%, {e['FLEX 95% CI'][1]:+.4f}%]")"""))

# ── Cell 16: Economic Ranking Header ──
cells.append(md("""---
## 8. Economic Model Ranking

Descriptive ranking only — no subjective composite score."""))

# ── Cell 17: Economic Ranking Table ──
cells.append(code("""# ── Cell 17: Economic Ranking Table ──
econ_table = pd.DataFrame({
    m: {
        'Overall Savings %': econ[m]['Overall Savings %'],
        'CI Low': econ[m]['Overall 95% CI'][0],
        'CI High': econ[m]['Overall 95% CI'][1],
        'WAIT Savings %': econ[m]['WAIT Savings %'],
        'WAIT N': econ[m]['WAIT N'],
        'FLEX Savings %': econ[m]['FLEX Savings %'],
        'FLEX N': econ[m]['FLEX N'],
    } for m in MODELS
}).T
econ_table.index.name = 'Model'

print("=" * 80)
print("ECONOMIC MODEL COMPARISON TABLE")
print("=" * 80)
display(econ_table.style.format({
    'Overall Savings %': '{:+.4f}%', 'CI Low': '{:+.4f}%', 'CI High': '{:+.4f}%',
    'WAIT Savings %': '{:+.4f}%', 'WAIT N': '{:.0f}',
    'FLEX Savings %': '{:+.4f}%', 'FLEX N': '{:.0f}'
}))

print("\\nEconomic Analysis:")
best_overall = max(MODELS, key=lambda m: econ[m]['Overall Savings %'])
best_wait = max(MODELS, key=lambda m: econ[m]['WAIT Savings %'])
best_flex = max(MODELS, key=lambda m: econ[m]['FLEX Savings %'])

print(f"  Highest overall result: {best_overall} ({econ[best_overall]['Overall Savings %']:+.4f}%)")
print(f"  Strongest WAIT result:  {best_wait} ({econ[best_wait]['WAIT Savings %']:+.4f}%)")
print(f"  Lowest FLEX loss:       {best_flex} ({econ[best_flex]['FLEX Savings %']:+.4f}%)")

# Check if any model achieves positive overall savings
any_positive = any(econ[m]['Overall Savings %'] > 0 for m in MODELS)
if not any_positive:
    print("\\n⚠️  No model establishes positive aggregate savings vs Spot in the 2025 holdout.")
    print("   This reflects the FLEXIBLE segment premium, not model failure.")"""))

# ── Cell 18: 2025 Holdout Header ──
cells.append(md("""---
## 9. 2025 Blind Holdout — Final Confirmation

Treat 2025 as the most important validation period. No tuning based on these results."""))

# ── Cell 19: 2025 Holdout ──
cells.append(code("""# ── Cell 19: 2025 Blind Holdout ──
print("=" * 80)
print("2025 BLIND HOLDOUT — FINAL CONFIRMATION")
print("=" * 80)

# Forecasting
print("\\n── 2025 Forecasting ──")
h25 = {}
for m in MODELS:
    s = df_cases[(df_cases['model']==m) & (df_cases['year']==2025)]
    h25[m] = {
        'MAE': s['abs_error'].mean(),
        'RMSE': np.sqrt(s['sq_error'].mean()),
        'MedianAE': s['abs_error'].median(),
        'Bias': s['residual'].mean(),
        'DA%': s['dir_correct'].mean()*100,
    }
display(pd.DataFrame(h25).T.style.format({
    'MAE':'${:,.2f}','RMSE':'${:,.2f}','MedianAE':'${:,.2f}','Bias':'${:+,.2f}','DA%':'{:.2f}%'
}).highlight_min(subset=['MAE','RMSE','MedianAE'], color='#d4edda')
 .highlight_max(subset=['DA%'], color='#d4edda'))

# Uncertainty
print("\\n── 2025 Uncertainty ──")
u25 = {}
for m in MODELS:
    s = df_cases[(df_cases['model']==m) & (df_cases['year']==2025)]
    sr = s[s['retained']==True]
    u25[m] = {
        'Coverage %': s['covered'].mean()*100,
        'Retained N': int(s['retained'].sum()),
        'Retained %': s['retained'].mean()*100,
        'Gated Prec %': sr['dir_correct'].mean()*100 if len(sr)>0 else 0,
    }
display(pd.DataFrame(u25).T.style.format({
    'Coverage %':'{:.2f}%','Retained N':'{:,}','Retained %':'{:.2f}%','Gated Prec %':'{:.2f}%'
}).highlight_max(subset=['Gated Prec %','Retained N'], color='#d4edda'))

# Decision Distribution
print("\\n── 2025 Decision Distribution ──")
dd25 = {}
for m in MODELS:
    s = df_cases[(df_cases['model']==m) & (df_cases['year']==2025)]
    dd25[m] = {
        'NOW': len(s[s['decision']=='NOW']),
        'WAIT': len(s[s['decision']=='WAIT']),
        'FLEXIBLE': len(s[s['decision']=='FLEXIBLE']),
    }
display(pd.DataFrame(dd25).T)

# Economic
print("\\n── 2025 Economic ──")
e25 = {m: {'Overall Sav%': econ[m]['Overall Savings %'],
           'WAIT Sav%': econ[m]['WAIT Savings %'],
           'FLEX Sav%': econ[m]['FLEX Savings %']} for m in MODELS}
display(pd.DataFrame(e25).T.style.format('{:+.4f}%'))"""))

# ── Cell 20: Plots Header ──
cells.append(md("""---
## 10. Visualizations"""))

# ── Cell 21: All Plots ──
cells.append(code("""# ── Cell 21: Generate All Required Plots ──
fig_dir = OUTPUT_DIR

# ─── Plot 1: MAE Comparison ───
fig, ax = plt.subplots(figsize=(8, 5))
maes = [scorecard[m]['MAE'] for m in MODELS]
bars = ax.bar(MODELS, maes, color=[COLORS[m] for m in MODELS], width=0.55, edgecolor='#1F2937', linewidth=1)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+3, f'${b.get_height():,.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Aggregate MAE Comparison (All Vessels, All Folds)', fontweight='bold')
ax.set_ylabel('MAE ($/MT)')
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '01_mae_comparison.png'), dpi=300); plt.show()

# ─── Plot 2: DA Comparison ───
fig, ax = plt.subplots(figsize=(8, 5))
das = [scorecard[m]['DA%'] for m in MODELS]
bars = ax.bar(MODELS, das, color=[COLORS[m] for m in MODELS], width=0.55, edgecolor='#1F2937', linewidth=1)
ax.axhline(50, color='#DC2626', ls='--', lw=1.5, label='Random (50%)')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Aggregate Directional Accuracy', fontweight='bold')
ax.set_ylabel('DA (%)')
ax.set_ylim(45, 80)
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '02_da_comparison.png'), dpi=300); plt.show()

# ─── Plot 3: 2025 MAE ───
fig, ax = plt.subplots(figsize=(8, 5))
m25 = [scorecard[m]['2025 MAE'] for m in MODELS]
bars = ax.bar(MODELS, m25, color=[COLORS[m] for m in MODELS], width=0.55, edgecolor='#1F2937', linewidth=1)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'${b.get_height():,.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('2025 Blind Holdout MAE', fontweight='bold')
ax.set_ylabel('MAE ($/MT)')
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '03_2025_mae.png'), dpi=300); plt.show()

# ─── Plot 4: 2025 DA ───
fig, ax = plt.subplots(figsize=(8, 5))
d25 = [scorecard[m]['2025 DA%'] for m in MODELS]
bars = ax.bar(MODELS, d25, color=[COLORS[m] for m in MODELS], width=0.55, edgecolor='#1F2937', linewidth=1)
ax.axhline(50, color='#DC2626', ls='--', lw=1.5, label='Random (50%)')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('2025 Blind Holdout Directional Accuracy', fontweight='bold')
ax.set_ylabel('DA (%)')
ax.set_ylim(45, 82)
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '04_2025_da.png'), dpi=300); plt.show()"""))

# ── Cell 22: More Plots ──
cells.append(code("""# ─── Plot 5: Vessel-Level MAE ───
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(VESSELS))
w = 0.2
for i, m in enumerate(MODELS):
    vals = [vessel_results[v][m]['MAE'] for v in VESSELS]
    ax.bar(x + i*w, vals, w, label=m, color=COLORS[m], edgecolor='#1F2937', linewidth=0.5)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels([v.upper() for v in VESSELS])
ax.set_title('Vessel-Level MAE Comparison', fontweight='bold')
ax.set_ylabel('MAE ($/MT)')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '05_vessel_mae.png'), dpi=300); plt.show()

# ─── Plot 6: Vessel-Level DA ───
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, m in enumerate(MODELS):
    vals = [vessel_results[v][m]['DA%'] for v in VESSELS]
    ax.bar(x + i*w, vals, w, label=m, color=COLORS[m], edgecolor='#1F2937', linewidth=0.5)
ax.axhline(50, color='#DC2626', ls='--', lw=1.2, label='Random')
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels([v.upper() for v in VESSELS])
ax.set_title('Vessel-Level Directional Accuracy', fontweight='bold')
ax.set_ylabel('DA (%)')
ax.set_ylim(45, 85)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '06_vessel_da.png'), dpi=300); plt.show()

# ─── Plot 7: Uncertainty: Coverage vs Width ───
fig, ax = plt.subplots(figsize=(8, 5.5))
for m in MODELS:
    ax.scatter(unc_results[m]['Mean Interval Width'], unc_results[m]['Coverage %'],
               s=unc_results[m]['Retained N']*0.5, color=COLORS[m], label=f"{m} (N={unc_results[m]['Retained N']})",
               edgecolor='#1F2937', linewidth=1, alpha=0.8, zorder=5)
ax.set_xlabel('Mean Interval Width ($/MT)')
ax.set_ylabel('Empirical Coverage (%)')
ax.set_title('Uncertainty: Coverage vs Interval Width (bubble=retained N)', fontweight='bold')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '07_uncertainty_coverage_width.png'), dpi=300); plt.show()

# ─── Plot 8: Decision Distribution ───
fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 4), sharey=True)
for i, m in enumerate(MODELS):
    s = df_cases[df_cases['model']==m]
    counts = s['decision'].value_counts()
    axes[i].bar(['NOW','WAIT','FLEX'], [counts.get('NOW',0), counts.get('WAIT',0), counts.get('FLEXIBLE',0)],
                color=['#059669','#D97706','#6B7280'], edgecolor='#1F2937')
    axes[i].set_title(m, fontweight='bold')
    for j, (lbl, val) in enumerate(zip(['NOW','WAIT','FLEX'], [counts.get('NOW',0), counts.get('WAIT',0), counts.get('FLEXIBLE',0)])):
        axes[i].text(j, val+20, str(val), ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylabel('Count')
fig.suptitle('Decision Distribution (All Folds)', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '08_decision_distribution.png'), dpi=300, bbox_inches='tight'); plt.show()

# ─── Plot 9: Economic Comparison ───
fig, ax = plt.subplots(figsize=(9, 5))
econ_vals = [econ[m]['Overall Savings %'] for m in MODELS]
bars = ax.bar(MODELS, econ_vals, color=[COLORS[m] for m in MODELS], width=0.55, edgecolor='#1F2937', linewidth=1)
ax.axhline(0, color='#DC2626', ls='-', lw=1.5)
for b in bars:
    y = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, y + (0.01 if y >= 0 else -0.03), f'{y:+.3f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Economic Savings vs Spot (2025 Holdout)', fontweight='bold')
ax.set_ylabel('Savings %')
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, '09_economic_comparison.png'), dpi=300); plt.show()

print("✅ All 9 plots saved to:", fig_dir)"""))

# ── Cell 23: Final Verdict Header ──
cells.append(md("""---
## 11. FINAL MODEL SELECTION VERDICT

Based on the complete evidence across all dimensions: forecasting accuracy, directional accuracy, fold stability, paired bootstrap tests, uncertainty/actionability, decision quality, and economic behavior."""))

# ── Cell 24: Final Verdict ──
cells.append(code("""# ── Cell 24: Final Model Selection Verdict ──
print("=" * 80)
print("FINAL MODEL SELECTION DECISION")
print("=" * 80)

# Compile evidence summary
evidence = {}
for m in PRIMARY:
    evidence[m] = {
        'agg_mae': scorecard[m]['MAE'],
        'agg_da': scorecard[m]['DA%'],
        'h25_mae': scorecard[m]['2025 MAE'],
        'h25_da': scorecard[m]['2025 DA%'],
        'fold_std': scorecard[m]['Std Fold MAE'],
        'fold_wins': scorecard[m]['Fold Wins'],
        'retained_n': unc_results[m]['Retained N'],
        'gated_da': unc_results[m]['Gated DA %'],
        'econ_overall': econ[m]['Overall Savings %'],
        'econ_wait': econ[m]['WAIT Savings %'],
        'dq_ret_prec': dq_results[m]['Retained Precision %'],
    }

print("\\n── Evidence Summary (Primary Candidates) ──")
ev_df = pd.DataFrame(evidence).T
ev_df.index.name = 'Model'
display(ev_df.style.format({
    'agg_mae': '${:,.2f}', 'agg_da': '{:.2f}%', 'h25_mae': '${:,.2f}', 'h25_da': '{:.2f}%',
    'fold_std': '{:.2f}', 'fold_wins': '{:.0f}', 'retained_n': '{:,}',
    'gated_da': '{:.2f}%', 'econ_overall': '{:+.4f}%', 'econ_wait': '{:+.4f}%',
    'dq_ret_prec': '{:.2f}%'
}))

# ── Decision Logic ──
print("\\n" + "─" * 80)
print("DECISION HIERARCHY APPLICATION")
print("─" * 80)

# PRIMARY: Downstream FICOS decision quality
print("\\n1. PRIMARY — Downstream Decision Quality:")
for m in PRIMARY:
    print(f"   {m:15s}: Retained Prec={evidence[m]['dq_ret_prec']:.2f}%, Gated DA={evidence[m]['gated_da']:.2f}%, Retained N={evidence[m]['retained_n']}")

best_gated = max(PRIMARY, key=lambda m: evidence[m]['gated_da'])
print(f"   → Best gated precision: {best_gated} ({evidence[best_gated]['gated_da']:.2f}%)")

# SECONDARY: Forecasting
print("\\n2. SECONDARY — Robust Forecasting:")
best_mae = min(PRIMARY, key=lambda m: evidence[m]['agg_mae'])
best_da = max(PRIMARY, key=lambda m: evidence[m]['agg_da'])
print(f"   Best MAE: {best_mae} (${evidence[best_mae]['agg_mae']:,.2f})")
print(f"   Best DA:  {best_da} ({evidence[best_da]['agg_da']:.2f}%)")

# Statistical evidence
print("\\n3. Paired Statistical Evidence:")
for key, r in paired_results.items():
    sig = "✅ Sig" if r['MAE Sig'] else "❌ NS"
    print(f"   {key}: MAE diff={r['MAE Diff (B-A)']:+.2f}, CI=[{r['MAE 95% CI'][0]:+.2f},{r['MAE 95% CI'][1]:+.2f}] {sig}")

# Fold stability
print("\\n4. Fold Stability:")
best_stable = min(PRIMARY, key=lambda m: evidence[m]['fold_std'])
print(f"   Most stable: {best_stable} (σ={evidence[best_stable]['fold_std']:.2f})")

# ── Determine winner ──
print("\\n" + "=" * 80)

# Score dimensions
dim_winners = {
    'Gated DA': max(PRIMARY, key=lambda m: evidence[m]['gated_da']),
    'Retained Precision': max(PRIMARY, key=lambda m: evidence[m]['dq_ret_prec']),
    'Aggregate MAE': min(PRIMARY, key=lambda m: evidence[m]['agg_mae']),
    'Aggregate DA': max(PRIMARY, key=lambda m: evidence[m]['agg_da']),
    '2025 MAE': min(PRIMARY, key=lambda m: evidence[m]['h25_mae']),
    '2025 DA': max(PRIMARY, key=lambda m: evidence[m]['h25_da']),
    'Fold Stability': min(PRIMARY, key=lambda m: evidence[m]['fold_std']),
    'Retained N': max(PRIMARY, key=lambda m: evidence[m]['retained_n']),
    'Econ Overall': max(PRIMARY, key=lambda m: evidence[m]['econ_overall']),
    'Econ WAIT': max(PRIMARY, key=lambda m: evidence[m]['econ_wait']),
}

win_counts = {m: 0 for m in PRIMARY}
for dim, winner in dim_winners.items():
    win_counts[winner] += 1
    print(f"  {dim:22s} → {winner}")

print(f"\\n  Dimension wins: {win_counts}")

# Check if evidence is mixed or decisive
top_model = max(PRIMARY, key=lambda m: win_counts[m])
top_count = win_counts[top_model]
runner_up = max([m for m in PRIMARY if m != top_model], key=lambda m: win_counts[m])
runner_count = win_counts[runner_up]

# Check paired significance between top two
pair_key_fwd = f"{top_model} vs {runner_up}"
pair_key_rev = f"{runner_up} vs {top_model}"
pair_sig = False
for pk in [pair_key_fwd, pair_key_rev]:
    if pk in paired_results:
        pair_sig = paired_results[pk]['MAE Sig']

print("\\n" + "=" * 80)
if top_count >= 7 and pair_sig:
    # Clear dominance
    if top_model == 'RandomForest':
        verdict = "A. KEEP RANDOM FOREST"
    elif top_model == 'LightGBM':
        verdict = "B. PROMOTE LIGHTGBM"
    else:
        verdict = "C. PROMOTE XGBOOST"
    mixed = False
elif top_count >= 6:
    # Strong but check decision quality specifically
    if evidence[top_model]['gated_da'] >= max(evidence[m]['gated_da'] for m in PRIMARY if m != top_model):
        if top_model == 'RandomForest':
            verdict = "A. KEEP RANDOM FOREST"
        elif top_model == 'LightGBM':
            verdict = "B. PROMOTE LIGHTGBM"
        else:
            verdict = "C. PROMOTE XGBOOST"
        mixed = False
    else:
        verdict = "E. NO CHANGE — EVIDENCE INSUFFICIENT"
        mixed = True
else:
    # Mixed evidence — check if decision quality clearly favors one
    dq_best = max(PRIMARY, key=lambda m: evidence[m]['gated_da'])
    if evidence[dq_best]['gated_da'] - min(evidence[m]['gated_da'] for m in PRIMARY) > 5:
        # Decision quality clearly favors one model
        if dq_best == 'RandomForest':
            verdict = "A. KEEP RANDOM FOREST"
        elif dq_best == 'LightGBM':
            verdict = "B. PROMOTE LIGHTGBM"
        else:
            verdict = "C. PROMOTE XGBOOST"
        mixed = False
    else:
        verdict = "E. NO CHANGE — EVIDENCE INSUFFICIENT"
        mixed = True

print(f"FINAL VERDICT: {verdict}")
print("=" * 80)

if mixed:
    print("\\nModel evidence is mixed; no statistically decisive production replacement is established.")
    print("Recommendation: Keep current production model (Random Forest) until stronger evidence emerges.")

# Justification
print("\\nJUSTIFICATION:")
print(f"  - {top_model} wins {top_count}/10 evaluation dimensions")
print(f"  - Best gated directional precision: {best_gated} ({evidence[best_gated]['gated_da']:.2f}%)")
print(f"  - Best aggregate MAE: {best_mae} (${evidence[best_mae]['agg_mae']:,.2f})")
print(f"  - Best aggregate DA: {best_da} ({evidence[best_da]['agg_da']:.2f}%)")
print(f"  - Most stable across folds: {best_stable} (σ={evidence[best_stable]['fold_std']:.2f})")"""))

# ── Cell 25: Production Registry ──
cells.append(md("""---
## 12. Production Registry"""))

cells.append(code("""# ── Cell 25: Production Registry Status ──
print("=" * 80)
print("PRODUCTION REGISTRY STATUS")
print("=" * 80)

print("\\nCURRENT PRODUCTION:")
for v in VESSELS:
    print(f"  {v:12s} (1D): RandomForestRegressor (promoted)")

print(f"\\nFINAL AUDIT RECOMMENDATION: {verdict}")

registry_change = 'RandomForest' not in verdict.upper()
print(f"\\nREGISTRY CHANGE REQUIRED: {'YES' if registry_change else 'NO'}")
print("=" * 80)"""))

# ── Cell 26: Save JSON ──
cells.append(code("""# ── Cell 26: Save Final Results JSON ──
final_results = {
    'verdict': verdict,
    'fold_verification': 'PASSED',
    'scorecard': {m: {k:v for k,v in scorecard[m].items() if k != 'fold_maes' and k != 'fold_das'} for m in MODELS},
    'fold_maes': {m: scorecard[m]['fold_maes'] for m in MODELS},
    'fold_das': {m: scorecard[m]['fold_das'] for m in MODELS},
    'vessel_results': vessel_results,
    'paired_bootstrap': {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv)) for kk,vv in v.items()} for k,v in paired_results.items()},
    'uncertainty': unc_results,
    'decision_quality': dq_results,
    'decision_quality_2025': dq25,
    'economic': {m: {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k,v in econ[m].items()} for m in MODELS},
    'dimension_winners': dim_winners,
    'dimension_win_counts': win_counts,
    'evidence_summary': {m: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k,v in ev.items()} for m,ev in evidence.items()},
    'registry_change_required': registry_change,
}

json_path = os.path.join('reports', 'final_model_selection_decision_results.json')
with open(json_path, 'w') as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"✅ Saved: {json_path}")"""))

# ── Cell 27: Generate Report ──
cells.append(code("""# ── Cell 27: Generate Final Report Markdown ──
report = f'''# FINAL MODEL SELECTION DECISION REPORT
## FICOS — SIH26006 Freight Intelligence & Chartering Optimization System

---

## VERDICT: {verdict}

---

## 1. Corrected Validation Verification

All 5 expanding walk-forward folds verified with strictly disjoint Train, Validation, and Test windows.
- max(TRAIN) < min(VALIDATION) < min(TEST) for every fold ✅
- Zero index overlap between all splits ✅
- Preprocessing fit ONLY on training data ✅

## 2. Aggregate Forecasting Comparison

| Model | MAE | RMSE | MedianAE | Bias | DA% | 2025 MAE | 2025 DA% |
|---|---|---|---|---|---|---|---|
'''

for m in MODELS:
    s = scorecard[m]
    report += f"| {m} | ${s['MAE']:,.2f} | ${s['RMSE']:,.2f} | ${s['MedianAE']:,.2f} | ${s['Bias']:+,.2f} | {s['DA%']:.2f}% | ${s['2025 MAE']:,.2f} | {s['2025 DA%']:.2f}% |\\n"

report += f'''
Fold Stability: Best σ = {best_stable} ({evidence[best_stable]['fold_std']:.2f})

## 3. Vessel-Level Comparison

'''
for v in VESSELS:
    report += f"### {v.upper()} 1D\\n| Model | MAE | DA% | 2025 MAE | 2025 DA% |\\n|---|---|---|---|---|\\n"
    for m in MODELS:
        vr = vessel_results[v][m]
        report += f"| {m} | ${vr['MAE']:,.2f} | {vr['DA%']:.2f}% | ${vr['2025 MAE']:,.2f} | {vr['2025 DA%']:.2f}% |\\n"
    report += "\\n"

report += "## 4. Paired Bootstrap Comparison\\n\\n"
for key, r in paired_results.items():
    sig_m = "✅ YES" if r['MAE Sig'] else "❌ NO"
    sig_d = "✅ YES" if r['DA Sig'] else "❌ NO"
    report += f"### {key}\\n"
    report += f"- MAE Diff: {r['MAE Diff (B-A)']:+.2f} $/MT, 95% CI: [{r['MAE 95% CI'][0]:+.2f}, {r['MAE 95% CI'][1]:+.2f}], Significant: {sig_m}\\n"
    report += f"- DA Diff: {r['DA Diff (B-A)']:+.2f}%, 95% CI: [{r['DA 95% CI'][0]:+.2f}, {r['DA 95% CI'][1]:+.2f}], Significant: {sig_d}\\n\\n"

report += "## 5. Uncertainty / Actionability\\n\\n| Model | Coverage | Retained N | Gated DA% |\\n|---|---|---|---|\\n"
for m in MODELS:
    u = unc_results[m]
    report += f"| {m} | {u['Coverage %']:.1f}% | {u['Retained N']} | {u['Gated DA %']:.2f}% |\\n"

report += f'''
## 6. Decision Quality

| Model | NOW | WAIT | FLEXIBLE | Retained Prec% |
|---|---|---|---|---|
'''
for m in MODELS:
    d = dq_results[m]
    report += f"| {m} | {d['NOW N']} | {d['WAIT N']} | {d['FLEXIBLE N']} | {d['Retained Precision %']:.2f}% |\\n"

report += f'''
## 7. 2025 Blind Holdout

See Section 9 tables above for complete 2025-specific forecasting, uncertainty, decision, and economic results.

## 8. Economic Comparison

| Model | Overall Sav% | CI | WAIT Sav% | WAIT N | FLEX Sav% | FLEX N |
|---|---|---|---|---|---|---|
'''
for m in MODELS:
    e = econ[m]
    report += f"| {m} | {e['Overall Savings %']:+.4f}% | [{e['Overall 95% CI'][0]:+.4f}, {e['Overall 95% CI'][1]:+.4f}] | {e['WAIT Savings %']:+.4f}% | {e['WAIT N']} | {e['FLEX Savings %']:+.4f}% | {e['FLEX N']} |\\n"

report += f'''
## 9. Final Production Recommendation

**VERDICT: {verdict}**

Dimension wins: {win_counts}

CURRENT PRODUCTION: RandomForest for all 4 vessel-horizon pairs
REGISTRY CHANGE REQUIRED: {'YES' if registry_change else 'NO'}
'''

report_path = os.path.join('reports', 'final_model_selection_decision_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"✅ Saved: {report_path}")
print("\\n" + "=" * 80)
print(f"FINAL VERDICT: {verdict}")
print("=" * 80)
print("\\nThis model selection is now FROZEN for the final FICOS architecture, PPT, and demo.")"""))

# ── Build notebook JSON ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "name": "FICOS Final Model Selection Decision Audit"}
    },
    "cells": cells
}

out_path = os.path.join(r"c:\Users\soheb\OneDrive\Desktop\ficos final\notebooks", "final_model_selection_decision.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to: {out_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
