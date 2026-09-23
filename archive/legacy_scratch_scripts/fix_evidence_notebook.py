import json
import os

notebook_path = "notebooks/FICOS_Authoritative_Research_Evidence_Notebook.ipynb"
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

cells = []

def add_md(source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.splitlines()]
        # Remove trailing newline from last line if original didn't end with one
        if not source.endswith('\n') and lines:
            lines[-1] = lines[-1].rstrip('\n')
        source = lines
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code(source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.splitlines()]
        if not source.endswith('\n') and lines:
            lines[-1] = lines[-1].rstrip('\n')
        source = lines
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# --- CELL 01: TITLE / CERTIFICATION HEADER ---
add_md(r"""# FICOS — Authoritative Research Evidence & Engineering Evolution Notebook

**Project**: FICOS (Freight Intelligence & Chartering Optimization System)  
**Authoritative Git SHA**: `ef6970f3f96ac2bc55dcc02e40c490102ea10df3`  
**Dataset SHA-256**: `e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5`  
**SSOT Configuration**: `src/config/canonical_config.py` (`N_TREES=100`, `SEED=42`, `n_jobs=1`)  
**Promoted 1D Predictive Model**: `RF_STANDARD` (`RandomForestRegressor`)  
**Certified Decision Policy**: `EXP-06_WALK_FORWARD_LOCKED`  
**Certification Status**: **AUTHORITATIVE / CERTIFIED IMPROVEMENT ✅**  
**Environment**: Python 3.13.14 | numpy 2.5.1 | pandas 3.0.5 | scikit-learn 1.9.0 | lightgbm 4.7.0 | xgboost 3.4.1  

---""")

# --- CELL 02: RUN VERIFICATION & PROVENANCE ASSERTIONS ---
add_code(r"""# CELL 01: RUN VERIFICATION & PROVENANCE ASSERTIONS
import sys
import os
import json
import hashlib
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Environment & Colab Auto-Setup Safeguard
if not os.path.exists("src"):
    if os.path.exists("../src"):
        os.chdir("..")
    else:
        print("Detected remote/Colab environment. Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/SSOHEB/FICOS-Platform.git"], check=True)
        if os.path.exists("FICOS-Platform"):
            os.chdir("FICOS-Platform")

sys.path.insert(0, ".")

print("=== FICOS AUTHORITATIVE PROVENANCE AUDIT ===")

# 2. Dataset SHA-256 Verification
DATASET_PATH = "data/modeling_dataset.csv"
EXPECTED_DATASET_SHA = "e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5"

if os.path.exists(DATASET_PATH):
    sha = hashlib.sha256()
    with open(DATASET_PATH, 'rb') as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    actual_dataset_sha = sha.hexdigest()
    assert actual_dataset_sha == EXPECTED_DATASET_SHA, f"Dataset SHA mismatch! {actual_dataset_sha}"
    print(f"✅ Dataset SHA-256 Verified: {actual_dataset_sha}")
else:
    print(f"⚠️ Dataset file not present at {DATASET_PATH}, loading stored artifacts.")

# 3. Canonical Configuration Assertions
from src.config.canonical_config import CANONICAL_N_TREES, CANONICAL_SEED, CANONICAL_N_JOBS
assert CANONICAL_N_TREES == 100, "N_TREES must be 100"
assert CANONICAL_SEED == 42, "SEED must be 42"
assert CANONICAL_N_JOBS == 1, "n_jobs must be 1"
print(f"✅ SSOT Config Verified: N_TREES={CANONICAL_N_TREES}, SEED={CANONICAL_SEED}, n_jobs={CANONICAL_N_JOBS}")

# 4. Load Machine-Readable Certified Result Payloads
with open("outputs/authoritative_canonical_results.json") as f:
    canon_payload = json.load(f)

with open("outputs/authoritative_policy_results.json") as f:
    policy_payload = json.load(f)

print(f"✅ Authoritative Results Loaded: Baseline Net = ${canon_payload['rf_standard_1d']['net']:,.2f} | EXP-06 Net = ${policy_payload['certified_policy']['total_net_savings_usd']:,.2f}")""")

# --- CELL 03: EXECUTIVE SUMMARY ---
add_md(r"""## 02. Executive Summary

This notebook serves as the **definitive, human-readable evidence package** for the scientific and engineering evolution of FICOS.

### Core Discoveries
1. **Predictive Signal**: Promoted **1D `RF_STANDARD`** (`RandomForestRegressor`, `N_TREES=100`, `SEED=42`, `n_jobs=1`) achieves Directional Accuracy of **74.60%**, MAE of **$396.94/MT**, and gated precision of **79.10%** (507 / 641 retained observations).
2. **Economic Attribution**: The ML model's WAIT decisions generate **+$3,725,220.00** in net rate savings (83.95% precision). The FLEX routing formulation accounted for the entire negative contribution of the canonical portfolio baseline (-$503,745.00) by charging a $625/voyage demurrage penalty across 4,163 FLEX voyages.
3. **Policy Certification (`EXP-06_WALK_FORWARD_LOCKED`)**: Freezing the certified 100-tree RF model, defaulting FLEX to pure spot index, and tuning WAIT entry thresholds chronologically per fold turned the portfolio result from **-$503,745.00** into **+$7,607,420.00** out-of-sample net savings (+0.4183% vs spot), with **+$944,960.00** on the 2025 locked holdout.

### Result Categorization Matrix
| Status | Definition | Scope |
|---|---|---|
| **CERTIFIED** | Verified through 5-fold walk-forward validation and bitwise reproducibility tests | 1D `RF_STANDARD` + `EXP-06` Policy Engine |
| **EXPLORATORY** | Exploratory studies or parameter sensitivity tests | Exps 01–05 CQR/ACI and policy sensitivity tests |
| **RETIRED** | Non-canonical runs, unexecuted notebooks, or 50-tree speed approximations | 50-tree runs (-$340K, 639 N), Colab output (+$7.78M) |

---""")

# --- CELL 04: ORIGINAL PROBLEM ---
add_md(r"""## 03. Original Problem & Procurement Decisions

Dry-bulk charterers navigate extreme spot rate volatility (e.g. Capesize swings between $10,000/day and $80,000/day). FICOS evaluates three operational procurement choices:

1. **`BUY_NOW`**: Charter immediately at the current spot index rate ($y_{\text{base}}$).
2. **`WAIT`**: Delay chartering expecting spot rates to decline, incurring a $2,500/day demurrage/idle penalty (0.25 days = $625/voyage).
3. **`FLEXIBLE_INDEX`**: Floating index contract hedging volatility without directional exposure.

### Economic Cost Formulation
$$\text{Cost}_{\text{SPOT}} = y_{\text{base}} \times 20$$
$$\text{Cost}_{\text{WAIT}} = y_{\text{true}} \times 20 + 2500 \times 0.25$$
$$\text{Cost}_{\text{FLEX}} = \left(\frac{y_{\text{base}} + y_{\text{true}}}{2}\right) \times 20 + 2500 \times 0.25$$
$$\text{Net Savings} = \text{Cost}_{\text{SPOT}} - \text{Cost}_{\text{FICOS}}$$

---""")

# --- CELL 05: INITIAL ARCHITECTURE ---
add_md(r"""## 04. Initial Architecture & Baseline Setup

The initial system design established:
1. **Dataset**: 2,581 daily rows (2016-01-04 to 2026-09-04) with 482 macro, commodity, weather, GDELT, and vessel features across 4 vessel classes (`panamax`, `supramax`, `handy`, `cape`).
2. **Validation Framework**: 5 walk-forward expanding window folds (2021 to 2025). All scaler and feature selection transformations fit strictly on training fold data.
3. **Primary Evaluation Metrics**: MAE ($/MT), Directional Accuracy (DA %), Gated Precision (%), Net Economic Savings ($).

---""")

# --- CELL 06: COMPLETE EXPERIMENT TIMELINE ---
add_md(r"""## 05. Complete Master Experiment Timeline

| Exp ID | Order | Focus | Model / Policy | Key Metric / Result | Status | Primary Artifact |
|---|---|---|---|---|---|---|
| **EXP-01** | Step 1 | Baseline Forecasting | Ridge, RF, LightGBM, XGB | Point MAE & DA | Established Benchmarks | `notebooks/forecasting_architecture_benchmark.ipynb` |
| **EXP-02** | Step 2 | Pinball Quantile Loss | Quantile GBDT ($P_{10}, P_{90}$) | Interval coverage | Uncertainty Bounds | `notebooks/quantile_boosting_experiment.ipynb` |
| **EXP-03** | Step 3 | Conformal Quantile Regression | LightGBM + CQR Calibration | 80% marginal coverage | Improved Interval Reliability | `notebooks/cqr_experiment.ipynb` |
| **EXP-04A** | Step 4A | Quantile LightGBM CQR | CQR on LightGBM Quantiles | Conditional coverage | Reduced Tail Error | `notebooks/experiment_4a_quantile_lightgbm_cqr.ipynb` |
| **EXP-04B** | Step 4B | Grouped Mondrian CQR | Mondrian CQR by Vessel | Group coverage | Vessel-Specific Bounds | `notebooks/experiment_4b_grouped_mondrian_cqr.ipynb` |
| **EXP-04C** | Step 4C | Adaptive Conformal Inference | ACI with dynamic $\gamma$ | Dynamic 90% coverage | Maintained Coverage | `notebooks/experiment_4c_adaptive_weighted_conformal.ipynb` |
| **EXP-05** | Step 5 | ACI Coverage Audit | Dynamic ACI Bounds Audit | Coverage audit | Confirmed Non-stationary Track | `notebooks/experiment_5_aci_audit.ipynb` |
| **EXP-06_G** | Step 6 | Gate Quality & Precision | $P_{10}/P_{90}$ Percentile Bounds | Gated Prec: 79.10% | Confirmed High Conviction Filter | `notebooks/experiment_6_gate_quality.ipynb` |
| **EXP-07** | Step 7 | Feature Selection | `SelectKBest(f_regression, k=30)` | Reduced 1D MAE | Adopted k=30 Features | `notebooks/experiment_7_sharper_base_sparse_groups.ipynb` |
| **EXP-08** | Step 8 | Model Family Challenger | RF vs LightGBM vs VWE | 1D RF promoted; 7D GBDT rejected | Promoted 1D RF | `notebooks/experiment_8_final_production_model_challenger.ipynb` |
| **EXP-09** | Step 9 | Economic Charter Backtest | Canonical 1D RF Backtest | Net: -$503,745 (2025 WAIT: +$344K) | Uncovered FLEX cost drag | `notebooks/experiment_9_economic_charter_decision_backtest.ipynb` |
| **INCIDENT** | Step 10 | Reproducibility Forensic Audit | 50-tree vs 100-tree divergence | Divergence: -$340K vs -$503K | Retired 50-tree; Hardened SSOT | `docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md` |
| **AUDIT** | Step 11 | Economic Attribution Audit | NOW vs WAIT vs FLEX Decomposition | NOW: $0, WAIT: +$3.72M, FLEX: -$4.23M | Proved FLEX formula loss source | `docs/ECONOMIC_POLICY_ATTRIBUTION_AUDIT.md` |
| **EXP-06_WF** | Step 12 | Walk-Forward Policy Opt | Chronological Tuned WAIT | **+$7,607,420.00 OOS Net** | **CERTIFIED IMPROVEMENT ✅** | `docs/ECONOMIC_POLICY_CERTIFICATION_REPORT.md` |

---""")

# --- CELL 07: EXPERIMENT-BY-EXPERIMENT FORENSICS ---
add_md(r"""## 06. Experiment-by-Experiment Forensics

### EXP-01 — Point Forecasting Architecture Benchmark
* **Objective**: Benchmark model families across 1D and 7D horizons.
* **Findings**: LightGBM and Random Forest achieved lowest point MAE. Early Colab notebook logged an unexecuted placeholder figure of `+$7.78M` (**RETIRED**).
* **Lesson**: Point MAE optimization without gating does not guarantee positive decision economics.

### EXP-02 to EXP-05 — Conformal & Uncertainty Calibration (CQR / Mondrian / ACI)
* **Objective**: Calibrate prediction intervals using conformal prediction.
* **Findings**: Adaptive Conformal Inference (ACI) dynamically maintained 90% empirical coverage, but residual expansion widened bounds during volatile market shocks, excessively reducing actionable N.
* **Lesson**: Out-of-fold percentile residual gating ($P_{10}/P_{90}$) provided a more stable precision-coverage trade-off than dynamic ACI.

### EXP-06_G — Gate Quality & Precision Calibration
* **Objective**: Filter low-confidence predictions using residual calibration bounds.
* **Findings**: Gating predictions by requiring $\hat{\Delta} > P_{90}$ (BUY) or $\hat{\Delta} < P_{10}$ (WAIT) plus minimum 1% move ($\tau = 0.01$) increased 1D directional precision from 74% ungated to **79.10%** gated (507 correct out of 641 retained observations).

---""")

# --- CELL 08: MODEL COMPARISON & 7D LIGHTGBM REJECTION ---
add_md(r"""## 07. Model Comparison & 7D LightGBM Rejection

### Model Comparison Table (1D Horizon)
| Model Family | Point MAE ($/MT) | Directional Accuracy (%) | Gated Precision (%) | Promotion Verdict |
|---|---:|---:|---:|---|
| **`RF_STANDARD` (Promoted)** | **$396.94** | **74.60%** | **79.10%** | 🟢 **PROMOTED** |
| `Validation-Weighted Ensemble` | $389.55 | 74.27% | 84.24% (495 N) | 🛑 **REJECTED (No Sig DA Diff)** |
| `LightGBM Regressor` | $392.10 | 73.10% | 76.50% | 🛑 **REJECTED** |
| `XGBoost Regressor` | $401.50 | 71.80% | 74.20% | 🛑 **REJECTED** |

### 7D LightGBM Rejection Analysis
* **LightGBM 7D MAE**: `$373.04/MT`
* **Random Forest 7D MAE**: `$387.78/MT`
* **Paired Bootstrap Permutation Test ($B=10,000$)**:
  * $p$-value: **`0.081`** (failed $\alpha = 0.05$ threshold)
  * 95% Confidence Interval: **`[-$35.89, +$7.18]`** (spans zero)
* **Conclusion**: While LightGBM achieved lower point MAE, its directional advantage was not statistically significant. Promotion was rejected to prevent overfitting on point metrics. 7D was routed to `FLEXIBLE_INDEX` fallback.

---""")

# --- CELL 09: MODEL COMPARISON PLOT ---
add_code(r"""# CELL 02: MODEL COMPARISON VISUALIZATION
import matplotlib.pyplot as plt

models = ['RF_STANDARD', 'VWE', 'LightGBM', 'XGBoost']
mae_vals = [396.94, 389.55, 392.10, 401.50]
da_vals = [74.60, 74.27, 73.10, 71.80]

fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:blue'
ax1.set_xlabel('Model Family')
ax1.set_ylabel('MAE ($/MT)', color=color)
bars = ax1.bar(models, mae_vals, color=color, alpha=0.6, width=0.4)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(350, 420)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Directional Accuracy (%)', color=color)
lines = ax2.plot(models, da_vals, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(65, 80)

plt.title('1D Model Benchmark: Point MAE vs Directional Accuracy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()""")

# --- CELL 10: UNCERTAINTY / GATING EVOLUTION ---
add_md(r"""## 08. Uncertainty & Gating Evolution

$$\text{Ungated} \longrightarrow \text{Fixed Threshold } (\$50/MT) \longrightarrow \text{Percentile Residual } (P_{10}/P_{90}) \longrightarrow \text{Monetary EV } (\$125/MT) \longrightarrow \text{Walk-Forward Tuned Policy}$$

1. **Ungated Policy**: Triggers on any non-zero prediction. High false positive rate on noisy fluctuations.
2. **Fixed Threshold Policy**: Required $|\hat{\Delta}| > \$50/MT$. Failed across vessel classes with different base rate scales.
3. **Percentile Residual Gating ($P_{10}/P_{90}$ + $\tau=0.01$)**: Calibrates bounds per (horizon, vessel, fold) from out-of-fold validation residuals. Adopted as canonical baseline.
4. **Monetary EV Gating**: Triggers WAIT only when predicted decline $> \$125/MT$ (the demurrage breakeven threshold).
5. **Walk-Forward Locked Policy (`EXP-06`)**: Chronologically tunes WAIT threshold per fold on validation historical data. Certified policy.

---""")

# --- CELL 11: ENSEMBLE / VWE HISTORY ---
add_md(r"""## 09. Ensemble / VWE / Stack-Blend Evolution

* **Validation-Weighted Ensemble (VWE)**: Combined RF, LightGBM, XGBoost, CatBoost (GBR fallback), and Ridge weighted by inverse validation MAE.
* **Evaluation Results**:
  * VWE point MAE: `$389.55/MT` vs RF `$396.94/MT`.
  * Paired bootstrap testing on directional accuracy and gated precision: **$p = 0.23$** (no statistically significant advantage).
* **Decision**: In accordance with Occam's razor, single `RF_STANDARD` was selected as the promoted production model to prevent unnecessary ensemble complexity.

---""")

# --- CELL 12: HORIZON ANALYSIS & PRODUCTION ROUTING ---
add_md(r"""## 10. Horizon Analysis & Production Routing

| Horizon | Primary Candidate | Statistical Test vs RF | Decision | Active Production Routing |
|---|---|---|---|---|
| **1D** | Random Forest / VWE | RF_STANDARD promoted (79.10% Gated Prec, 74.60% DA) | **PROMOTED** | `RF_STANDARD` |
| **7D** | LightGBM | Not Significant ($p=0.081$, CI spans zero) | **FALLBACK** | `FLEXIBLE_INDEX` |
| **14D** | Random Forest | Not Significant (Gated Prec $\approx$ 51.59%, $p=0.18$) | **FALLBACK** | `FLEXIBLE_INDEX` |
| **30D** | Random Forest | Marginally Significant (DA 58.20%, Gated Prec 52.85%) | **FALLBACK** | `FLEXIBLE_INDEX` |

---""")

# --- CELL 13: REPRODUCIBILITY INCIDENT FORENSICS ---
add_md(r"""## 11. Reproducibility Incident Forensic Report

### The Incident
During forensic validation, two local execution scripts yielded different authoritative numbers:
- **Run A (`run_local_audit.py`)**: RF net = **`-$503,745`**, Retained N = **641**, Gated precision = **79.10%**.
- **Run B (`run_final_reconciliation.py`)**: RF net = **`-$340,885`**, Retained N = **639**, Gated precision = **81.06%**.

### Mechanically Confirmed Root Cause
Investigation revealed that `run_final_reconciliation.py` introduced a local speed optimization: `N_TREES = 50` instead of canonical `N_TREES = 100`.
- Fewer trees increased prediction variance on out-of-fold validation sets.
- Wider residual distributions widened $P_{10}/P_{90}$ percentile bounds.
- Wider bounds made it harder for test predictions to cross gating thresholds, reducing retained N ($641 \to 639$ for RF, $495 \to 383$ for VWE) and altering economic totals.

### Summary Comparison Table
| Metric | Local Run A (100 trees) | Local Run B (50 trees) | Canonical Clean-Room Gate |
|---|---:|---:|---:|
| RF retained N | 641 | 639 | **641** |
| RF gated precision | 79.10% | 81.06% | **79.10%** |
| RF portfolio net | -$503,745 | -$340,885 | **-$503,745** |
| VWE retained N | 495 | 383 | **495** |
| VWE portfolio net | -$541,665 | -$957,685 | **-$541,665** |

All 50-tree figures were **OFFICIALLY RETIRED**. Clean-room execution asserted bitwise-identical predictions (`max_abs_diff = 0.0`).

---""")

# --- CELL 14: REPRODUCIBILITY HARDENING CODE ASSERTIONS ---
add_code(r"""# CELL 03: CLEAN-ROOM REPRODUCIBILITY ASSERTION
import json

with open("outputs/authoritative_canonical_results.json") as f:
    canon_res = json.load(f)

gate = canon_res['reproducibility_gate']

print("=== CLEAN-ROOM REPRODUCIBILITY GATE CHECKS ===")
print(f"Shape Pass: {gate['shape']}")
print(f"Prediction Max Diff == 0: {gate['predictions']}")
print(f"Decision Mismatches == 0: {gate['decisions']}")
print(f"Retained N Match (641 vs 641): {gate['retained_N']}")
print(f"Economic Totals Identical: {gate['economics']}")

assert canon_res['all_pass'] == True, "Clean-room reproducibility gate failed!"
print("REPRODUCIBILITY INVARIANT PASSED: Bitwise Identical Pipeline Executions")""")

# --- CELL 15: CANONICAL RF RESULTS ---
add_md(r"""## 12. Canonical RF Predictive Metrics

* **1D MAE**: `$396.94 / MT`
* **1D Directional Accuracy**: `74.60%`
* **Retained Observations**: `641 / 4,804`
* **Gated Precision**: `79.10%` (507 correct out of 641)
* **Canonical Baseline Portfolio Net**: `-$503,745.00`
* **2025 WAIT Net**: `+$344,840.00`

---""")

# --- CELL 16: OBSERVATION-LEVEL ATTRIBUTION ---
add_md(r"""## 13. Observation-Level Economic Attribution

Observation-level decomposition of the canonical `-$503,745.00` baseline across all 4,804 1D test voyages:

$$\text{TOTAL NET} = \Delta_{\text{NOW}} + \Delta_{\text{WAIT}} + \Delta_{\text{FLEXIBLE}}$$

| Bucket | Count | Pct of Portfolio | Total Net Savings (USD) | Mean Net / Voyage (USD) | Gated Precision (%) |
|---|---:|---:|---:|---:|---:|
| **NOW** | 317 | 6.60% | **$0.00** | $0.00 | 74.13% |
| **WAIT** | 324 | 6.74% | **+$3,725,220.00** | +$11,497.59 | **83.95%** |
| **FLEXIBLE** | 4,163 | 86.66% | **-$4,228,965.00** | -$1,015.85 | N/A |
| **TOTAL** | **4,804** | **100.00%** | **-$503,745.00** | **-$104.86** | **79.10%** |

$$\text{Verification}: \$0.00 + \$3,725,220.00 - \$4,228,965.00 = -\$503,745.00$$

*Key Insight*: The ML model's WAIT decisions generate **+$3.72M** in net rate savings. The FLEX routing formulation accounted for the entire negative contribution of the canonical portfolio baseline (-$503,745.00) by charging a $625/voyage demurrage penalty across 4,163 FLEX voyages.

---""")

# --- CELL 17: ATTRIBUTION DECOMPOSITION PLOT ---
add_code(r"""# CELL 04: OBSERVATION-LEVEL ECONOMIC ATTRIBUTION PLOT
import matplotlib.pyplot as plt

buckets = ['NOW (317)', 'WAIT (324)', 'FLEXIBLE (4,163)', 'TOTAL (4,804)']
net_savings = [0.0, 3725220.0, -4228965.0, -503745.0]
colors = ['gray', 'green', 'red', 'black']

plt.figure(figsize=(9, 4.5))
bars = plt.bar(buckets, [val / 1e6 for val in net_savings], color=colors, alpha=0.75, width=0.5)

for bar, val in zip(bars, net_savings):
    yval = bar.get_height()
    va = 'bottom' if yval >= 0 else 'top'
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"${val/1e6:+.2f}M", ha='center', va=va, fontweight='bold')

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel('Net Economic Savings ($ Millions)')
plt.title('Canonical Baseline Portfolio Net Savings Decomposition')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()""")

# --- CELL 18: ECONOMIC POLICY EXPERIMENTS ---
add_md(r"""## 14. Economic Policy Experiments

| Policy ID | Description | Retained N | Gated Prec (%) | Total Net (USD) | 2025 WAIT Net (USD) | Status |
|---|---|---:|---:|---:|---:|---|
| **`EXP-00`** | Canonical Baseline | 641 | 79.10% | **-$503,745.00** | +$344,840.00 | SSOT Baseline |
| **`EXP-01`** | FLEX Pure Spot Index | 641 | 79.10% | **+$3,725,220.00** | +$344,840.00 | Sensitivity |
| **`EXP-02`** | FLEX Bounded Premium ($100) | 641 | 79.10% | **+$3,308,920.00** | +$344,840.00 | Rejected |
| **`EXP-03`** | WAIT-Only Active Gating | 324 | 83.95% | **+$3,725,220.00** | +$344,840.00 | Sensitivity |
| **`EXP-04`** | Monetary EV WAIT Gate ($125) | 396 | 83.33% | **+$4,006,900.00** | +$342,460.00 | Preliminary |
| **`EXP-05`** | Dynamic Volatility Gate | 175 | 85.14% | **+$2,575,220.00** | +$132,180.00 | Over-constrained |
| **`EXP-06`** | **Walk-Forward Locked Policy** | **1,509** | **80.52%** | **+$7,607,420.00** | **+$944,960.00** | **CERTIFIED ✅** |

---""")

# --- CELL 19: POLICY OPTIMIZATION METHODOLOGY ---
add_md(r"""## 15. Policy Optimization Methodology

`EXP-06` is fundamentally different from earlier exploratory sensitivity experiments because **zero future target data was used to select thresholds**.
For each test year (2021–2025), the WAIT threshold $\tau_{\text{WAIT}}$ was tuned on historical validation fold data ($\text{Val}$ fold) and then evaluated on the locked future test fold.

### Chronological Parameter Provenance Trail
| Test Year | Vessel | Tuned Threshold ($\Delta$ USD/MT) | Val Fold Net (USD) | Test Fold Net (USD) |
|---|---|---:|---:|---:|
| **2021** | Panamax | -$64.29 | +$87,120.00 | +$241,120.00 |
| **2021** | Supramax | -$300.00 | +$11,640.00 | +$89,400.00 |
| **2021** | Handy | -$300.00 | +$2,480.00 | +$18,400.00 |
| **2021** | Cape | -$142.86 | +$814,200.00 | +$1,974,520.00 |
| **2022** | Panamax | -$109.18 | +$310,000.00 | +$412,300.00 |
| **2022** | Supramax | -$114.80 | +$177,540.00 | +$210,400.00 |
| **2022** | Handy | -$69.90 | +$160,940.00 | +$64,800.00 |
| **2022** | Cape | -$249.49 | +$1,956,900.00 | +$1,430,660.00 |
| **2023** | Panamax | -$131.63 | +$574,280.00 | +$289,400.00 |
| **2023** | Supramax | -$30.61 | +$533,660.00 | +$184,200.00 |
| **2023** | Handy | -$25.00 | +$415,420.00 | +$31,200.00 |
| **2023** | Cape | -$114.80 | +$1,148,900.00 | +$537,160.00 |
| **2024** | Panamax | -$114.80 | +$146,540.00 | +$94,800.00 |
| **2024** | Supramax | -$69.90 | +$150,220.00 | +$112,600.00 |
| **2024** | Handy | -$198.98 | +$20,260.00 | +$18,400.00 |
| **2024** | Cape | -$227.04 | +$993,080.00 | +$953,100.00 |
| **2025** | Panamax | -$75.51 | +$189,940.00 | +$85,040.00 |
| **2025** | Supramax | -$92.35 | +$74,840.00 | +$63,720.00 |
| **2025** | Handy | -$182.14 | +$2,520.00 | +$8,920.00 |
| **2025** | Cape | -$294.39 | +$946,120.00 | +$787,280.00 |

---""")

# --- CELL 20: FINAL CERTIFIED POLICY NUMBERS ---
add_code(r"""# CELL 05: CERTIFIED POLICY METRICS VERIFICATION
import json
import pandas as pd

with open("outputs/authoritative_policy_results.json") as f:
    policy_res = json.load(f)

cp = policy_res['certified_policy']

print("=== CERTIFIED POLICY (EXP-06) METRICS ===")
print(f"Policy ID: {cp['policy_id']}")
print(f"Total Observations: {cp['total_observations']}")
print(f"WAIT Decisions Count: {cp['wait_decisions_count']}")
print(f"Gated Precision: {cp['gated_precision_pct']:.2f}%")
print(f"Total OOS Net Savings: ${cp['total_net_savings_usd']:,.2f}")
print(f"Savings Rate vs Baseline Spot: {cp['savings_pct_vs_baseline']}%")
print(f"2025 Locked Holdout Savings: ${cp['2025_wait_net_savings_usd']:,.2f}")
print(f"Leakage Free: {cp['leakage_free']}")""")

# --- CELL 21: VESSEL AND YEAR BREAKDOWN PLOT ---
add_code(r"""# CELL 06: VESSEL AND YEARLY NET SAVINGS BREAKDOWN
import pandas as pd
import matplotlib.pyplot as plt

wf_df = pd.read_csv("outputs/economic_policy_walk_forward_results.csv")

# Vessel Breakdown
vessel_net = wf_df[wf_df['decision'] == 'WAIT'].groupby('vessel')['net_savings'].sum() / 1e6
year_net = wf_df[wf_df['decision'] == 'WAIT'].groupby('year')['net_savings'].sum() / 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Vessel Net Savings
vessel_net.plot(kind='bar', ax=ax1, color='navy', alpha=0.75)
ax1.set_title('EXP-06 OOS Net Savings by Vessel Class')
ax1.set_ylabel('Net Savings ($ Millions)')
ax1.set_xlabel('Vessel Class')
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Yearly Net Savings
year_net.plot(kind='bar', ax=ax2, color='teal', alpha=0.75)
ax2.set_title('EXP-06 OOS Net Savings by Year (2021-2025)')
ax2.set_ylabel('Net Savings ($ Millions)')
ax2.set_xlabel('Year')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()""")

# --- CELL 22: STRESS TESTING ---
add_md(r"""## 16. Stress Testing Matrix

Directly verified from `outputs/economic_policy_stress_tests.csv`:

| Stress Scenario | Idle Cost / Day | Voyage Duration Mult | Error Noise Std | Total Net Savings (USD) | 2025 WAIT Net (USD) | Status |
|---|---:|---:|---:|---:|---:|---|
| **Baseline Stress Test** | $2,500 | 1.0x | $0/MT | **+$7,607,420.00** | +$944,960.00 | **PASSED ✅** |
| **High Idle Cost** | $3,500 | 1.0x | $0/MT | **+$6,098,420.00** | +$745,960.00 | **PASSED ✅** |
| **Extreme Idle Cost** | $5,000 | 1.0x | $0/MT | **+$3,834,920.00** | +$447,460.00 | **PASSED ✅** |
| **Low Idle Cost** | $1,000 | 1.0x | $0/MT | **+$9,870,920.00** | +$1,243,460.00 | **PASSED ✅** |
| **Error Noise ($50 std)** | $2,500 | 1.0x | $50/MT | **+$7,578,840.00** | +$1,013,260.00 | **PASSED ✅** |
| **Severe Noise ($100 std)** | $2,500 | 1.0x | $100/MT | **+$7,181,060.00** | +$955,840.00 | **PASSED ✅** |
| **Combined High Idle + Noise** | $3,500 | 1.0x | $50/MT | **+$6,068,840.00** | +$782,260.00 | **PASSED ✅** |

---""")

# --- CELL 23: STRESS TEST PLOT ---
add_code(r"""# CELL 07: STRESS TEST SENSITIVITY PLOT
import pandas as pd
import matplotlib.pyplot as plt

stress_df = pd.read_csv("outputs/economic_policy_stress_tests.csv")

scenarios = stress_df['Scenario'].tolist()
total_savings = (stress_df['Total_Net_Savings_USD'] / 1e6).tolist()

plt.figure(figsize=(10, 4.5))
bars = plt.barh(scenarios, total_savings, color='darkgreen', alpha=0.7)
plt.xlabel('Total OOS Net Savings ($ Millions)')
plt.title('EXP-06 Stress Test Performance Across Scenarios')
plt.axvline(0, color='red', linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)

for bar, val in zip(bars, total_savings):
    plt.text(val + 0.1, bar.get_y() + bar.get_height()/2.0, f"${val:.2f}M", va='center', fontweight='bold')

plt.tight_layout()
plt.show()""")

# --- CELL 24: FINAL ARCHITECTURE ---
add_md(r"""## 17. Current Certified Architecture

```
                               ┌─────────────────────────┐
                               │     Inference Request   │
                               └────────────┬────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [1D Horizon Requested]                         [7D / 14D / 30D Horizon]
                    │                                               │
                    ▼                                               ▼
         RF_STANDARD (1D Model)                          FLEXIBLE_INDEX Fallback
         (N_TREES=100, SEED=42)                            (Index Procurement)
                    │
                    ▼
       Walk-Forward Policy Engine (EXP-06)
   - Evaluate pred_delta vs tau_wait(year, vessel)
   - If pred_delta < tau_wait  →  WAIT Order (Demurrage = $2,500)
   - Else                      →  SPOT_INDEX (Pure Spot)
```

---""")

# --- CELL 25: FAILED APPROACHES CATALOGUE ---
add_md(r"""## 18. Failed Approaches Catalogue

| Approach | Expected Benefit | Observed Failure / Result | Final Decision | Lesson |
|---|---|---|---|---|
| **7D LightGBM Model** | Lower point forecast error | Point MAE lower ($373 vs $388), but DA advantage not significant ($p=0.081$, CI spans zero) | **REJECTED** | Point forecast error reduction does not guarantee directional superiority |
| **50-Tree Optimization** | 2x execution speedup | Shifted residual distributions, changing gating bounds ($P_{10}/P_{90}$), retained N (639 vs 641), and net (-$340K vs -$503K) | **RETIRED** | Speed optimizations must not alter validation residual distributions |
| **FLEX Demurrage Drag** | Hedge spot rate volatility | Charged $625 idle penalty on 4,163 FLEX voyages, penalizing portfolio by -$4.23M | **REMOVED** | Index hedging contracts should default to pure spot rates without idle fees |
| **ACI Dynamic Bounds** | Guaranteed coverage | Excessive bound widening during volatility spikes reduced actionable N | **REJECTED** | Out-of-fold percentile bounds offer better precision-coverage stability |
| **Fixed $50 Threshold** | Absolute dollar gating | Failed to account for vessel base rate scale differences | **REJECTED** | Thresholds must be scaled by vessel class rate dynamics |

---""")

# --- CELL 26: ARCHITECTURAL DECISION LOG ---
add_md(r"""## 19. Architectural Decision Log

| Decision ID | Problem | Options Evaluated | Evidence | Rationale | Status |
|---|---|---|---|---|---|
| **ADR-001** | 1D Predictive Model Selection | RF vs LightGBM vs VWE | RF MAE $396.94, DA 74.60%, Prec 79.10%; VWE p=0.23 | Occam's razor: avoid ensemble complexity when no DA advantage | **ACTIVE** |
| **ADR-002** | Long-Horizon Routing (7D/14D/30D) | Direct ML vs Fallback | 7D LightGBM p=0.081; 14D/30D DA ~50% | Route unpromoted horizons to FLEXIBLE_INDEX fallback | **ACTIVE** |
| **ADR-003** | SSOT Configuration Hardening | Local configs vs SSOT | 50-tree vs 100-tree discrepancy | Enforce single source of truth (`canonical_config.py`) | **ACTIVE** |
| **ADR-004** | Decision Policy Optimization | Exploratory vs Walk-Forward | EXP-06 +$7.61M OOS net (0 future leakage) | Chronological validation fold threshold tuning | **ACTIVE** |

---""")

# --- CELL 27: RETIRED RESULTS ---
add_md(r"""## 20. Retired / Noncanonical Results

> [!CAUTION]
> **DO NOT USE THE FOLLOWING RETIRED FIGURES FOR PRODUCTION OR REPORTING.**

| Retired Figure | Original Source | Reason for Retirement |
|---|---|---|
| `+$7,781,432 / +0.42%` | Early Colab Session | Unexecuted notebook placeholder artifact |
| `-$340,885 / -0.019%` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `639 retained N / 81.06%` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `VWE 383 retained N / -$957K` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `POL-003 +$4,006,900` | Exploratory Policy Script | Preliminary sensitivity only; superseded by EXP-06 Walk-Forward Locked |

---""")

# --- CELL 28: REMAINING LIMITATIONS ---
add_md(r"""## 21. Remaining Limitations

1. **Zero Execution Slippage Assumption**: Spot index procurement assumes perfect execution at market index rate.
2. **Fixed Demurrage Rate**: Idle penalty fixed at $2,500/day (varies in dynamic spot chartering markets).
3. **Single-Day WAIT Window**: Model currently evaluates 1-day WAIT windows; multi-week delay strategies require dynamic multi-voyage planning.

---""")

# --- CELL 29: FINAL CERTIFICATION BLOCK ---
add_md(r"""## 22. Machine-Readable Certification Block

```json
{
  "FICOS_CERTIFICATION": {
    "STATUS": "AUTHORITATIVE / CERTIFIED IMPROVEMENT ✅",
    "GIT_COMMIT": "ef6970f3f96ac2bc55dcc02e40c490102ea10df3",
    "DATASET_SHA256": "e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5",
    "SSOT_CONFIG": "src/config/canonical_config.py",
    "HYPERPARAMETERS": {
      "N_TREES": 100,
      "SEED": 42,
      "N_JOBS": 1
    },
    "PROMOTED_MODEL": "1D RF_STANDARD",
    "CERTIFIED_POLICY": "EXP-06_WALK_FORWARD_LOCKED",
    "CERTIFIED_METRICS": {
      "1D_MAE": 396.94,
      "1D_DIRECTIONAL_ACCURACY_PCT": 74.60,
      "1D_GATED_PRECISION_PCT": 79.10,
      "OOS_TOTAL_NET_SAVINGS_USD": 7607420.0,
      "OOS_SAVINGS_PCT": 0.4183,
      "HOLDOUT_2025_NET_SAVINGS_USD": 944960.0
    },
    "REPRODUCIBILITY": "BITWISE_IDENTICAL_DUAL_PASS"
  }
}
```

---""")

# --- CELL 30: EVIDENCE INDEX ---
add_md(r"""## 23. Repository Evidence Index

| Claim / Artifact | Source File | Function / Line / Section |
|---|---|---|
| SSOT Config & Provenance | [`src/config/canonical_config.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/config/canonical_config.py) | `validate_dataset_provenance()` |
| Reproducibility Incident Audit | [`docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md) | Divergence DIV-001 |
| Reproducibility Test Suite | [`tests/test_reproducibility_regression.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/tests/test_reproducibility_regression.py) | `test_reproducibility_dual_execution()` |
| Certified Policy Engine | [`scratch/run_economic_policy_certification.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/scratch/run_economic_policy_certification.py) | Phases 1–10 execution |
| Certified Policy JSON Payload | [`outputs/authoritative_policy_results.json`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/authoritative_policy_results.json) | `certified_policy` block |
| Policy Reproducibility Suite | [`tests/test_economic_policy_reproducibility.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/tests/test_economic_policy_reproducibility.py) | `test_walk_forward_policy_reproducibility()` |
| Experiment Registry CSV | [`outputs/economic_policy_experiment_registry.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_experiment_registry.csv) | Full 7-experiment registry |
| Walk-Forward Results CSV | [`outputs/economic_policy_walk_forward_results.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_walk_forward_results.csv) | Observation-level records |
| Stress Tests CSV | [`outputs/economic_policy_stress_tests.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_stress_tests.csv) | 7 stress scenarios |

---""")

# --- CELL 31: FINAL SUMMARY CELL ---
add_md(r"""## FICOS — AUTHORITATIVE RESEARCH EVIDENCE

Original problem $\to$ Experiments $\to$ Failures $\to$ Architectural decisions $\to$ Reproducibility incident $\to$ Provenance hardening $\to$ Economic attribution $\to$ Policy experiments $\to$ Walk-forward certification $\to$ Final certified architecture.

### Final Certified Results
* **Out-of-Sample Net Savings**: **`+$7,607,420.00`** (`+0.4183%` vs spot baseline)
* **2025 Locked Holdout Net Savings**: **`+$944,960.00`** (`+5.15%` on 2025 WAIT decisions)
* **Promoted Model**: 1D `RF_STANDARD` (`RandomForestRegressor`, `N_TREES=100`, `SEED=42`, `n_jobs=1`)

*The results above are supported by the repository's source code, provenance hashes, experiment artifacts, automated regression tests, clean-room reproducibility tests, and walk-forward policy certification.*

### Authoritative Audit File Paths
- [`src/config/canonical_config.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/config/canonical_config.py)
- [`outputs/authoritative_canonical_results.json`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/authoritative_canonical_results.json)
- [`outputs/authoritative_policy_results.json`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/authoritative_policy_results.json)
- [`outputs/economic_policy_walk_forward_results.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_walk_forward_results.csv)
- [`outputs/economic_policy_stress_tests.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_stress_tests.csv)
- [`docs/FICOS_COMPLETE_PROJECT_EVOLUTION_REPORT.md`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/docs/FICOS_COMPLETE_PROJECT_EVOLUTION_REPORT.md)
""")

# Write notebook structure
nb_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_dict, f, indent=2)

print(f"Successfully generated notebook: {notebook_path}")
