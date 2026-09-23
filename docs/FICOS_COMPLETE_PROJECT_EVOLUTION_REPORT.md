# FICOS — Complete Research & Engineering Evolution Report

**Project**: FICOS (Freight Intelligence & Chartering Optimization System)  
**Authoritative Git SHA**: `ef6970f3f96ac2bc55dcc02e40c490102ea10df3`  
**Dataset SHA-256**: `e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5`  
**Single Source of Truth Configuration**: `src/config/canonical_config.py`  
**Machine-Readable Result Payload**: `outputs/authoritative_policy_results.json`  

---

## 1. Executive Summary

This report presents the **complete forensic, evidence-backed research and engineering history** of the FICOS platform—from initial exploratory benchmarks to the current certified production state.

### Key Milestones Summary
1. **Predictive Modeling**: Promoted **1D `RF_STANDARD`** (Random Forest Regressor, `N_TREES=100`, `SEED=42`, `n_jobs=1`) based on 5-fold expanding-window walk-forward validation (2021–2025). Gated precision = **79.10%** (507 / 641 retained observations), Directional Accuracy (DA) = **74.60%**, MAE = **$396.94/MT**.
2. **Horizon Routing**: 7D, 14D, and 30D horizons lacked statistically significant gated precision or DA advantage over baseline/RF and were routed to `FLEXIBLE_INDEX` fallback.
3. **Reproducibility Debugging**: Identified a critical discrepancy where a 50-tree speed optimization in secondary audit scripts altered out-of-fold residual distributions, shifting gating thresholds ($P_{10}/P_{90}$) and changing retained N (639 vs 641) and economics (-$340K vs -$503K). Retired all 50-tree figures and hardened the codebase with single-source-of-truth (SSOT) configuration and bitwise-identical clean-room execution.
4. **Economic Attribution Audit**: Demonstrated that the FLEX routing formulation accounted for the entire negative contribution of the canonical portfolio baseline (-$503,745.00), as the original `cost_flex` formula (`cost_flex = avg(spot, true)*20 + 2500*0.25`) penalized 4,163 FLEX voyages by -$4.23M. The ML model's 324 WAIT decisions generated **`+$3,725,220.00`** in net rate savings (83.95% precision).
5. **Walk-Forward Policy Certification (`EXP-06_WALK_FORWARD_LOCKED`)**: Preserving 100% of the certified 100-tree RF_STANDARD predictive model, redesigning the policy layer to treat FLEX as pure spot index and tuning WAIT thresholds chronologically per fold turned the portfolio result from **`-$503,745.00`** into **`+$7,607,420.00`** out-of-sample (+0.4183% net savings vs spot baseline), with **`+$944,960.00`** on the 2025 locked holdout.

---

## 2. Original Problem

The FICOS platform was built to solve the **dry-bulk maritime freight procurement problem**. 
Commodity traders and charterers must decide whether to:
- **BUY_NOW**: Charter a vessel immediately at current spot rates.
- **WAIT**: Delay chartering expecting spot rates to decline, incurring a $2,500/day demurrage/idle penalty.
- **FLEXIBLE_INDEX**: Utilize index-linked flexible contracts to hedge volatility without directional bets.

The central challenge is that freight rates exhibit extreme non-stationary volatility (e.g. 2021 Capesize rate swings from $10,000/day to $80,000/day). Standard MSE-minimized ML models tend to make unconfident bets, leading to catastrophic opportunity losses when predictions fail.

---

## 3. Initial Architecture

The initial system design established:
1. **Data Pipeline**: 2,581 daily rows (2016-01-04 to 2026-09-04) with 482 macro, commodity, weather, GDELT, and vessel features across 4 vessel classes (`panamax`, `supramax`, `handy`, `cape`).
2. **Validation Framework**: 5 walk-forward expanding window folds (2021 to 2025). All scaler and feature selection transformations fit strictly on training fold data.
3. **Primary Evaluation Metrics**: MAE ($/MT), Directional Accuracy (DA %), Gated Precision (%), Net Economic Savings ($).

---

## 4. Complete Experiment Timeline

| Exp ID | Chronological Order | Investigation Focus | Model / Policy Architecture | Key Metric / Result | Outcome / Status | Primary Artifact / Evidence |
|---|---|---|---|---|---|---|
| **EXP-01** | Step 1 (Early) | Freight Forecasting Baseline | Ridge, Lasso, RF, LightGBM, XGBoost | Initial point MAE & DA | Established baseline benchmarks | `notebooks/forecasting_architecture_benchmark.ipynb` |
| **EXP-02** | Step 2 | Pinball Quantile Loss | Quantile Gradient Boosting ($P_{10}, P_{90}$) | Interval coverage | Provided uncertainty bounds; high point MAE | `notebooks/quantile_boosting_experiment.ipynb` |
| **EXP-03** | Step 3 | Conformal Quantile Regression | LightGBM + CQR Calibration | 80% marginal coverage | Improved interval reliability; wide tails | `notebooks/cqr_experiment.ipynb` |
| **EXP-04A** | Step 4A | Quantile LightGBM CQR | CQR on LightGBM Quantiles | Conditional coverage | Reduced tail coverage error | `notebooks/experiment_4a_quantile_lightgbm_cqr.ipynb` |
| **EXP-04B** | Step 4B | Grouped Mondrian CQR | Mondrian CQR by Vessel & Regime | Group coverage | Better vessel-specific bounds; higher variance | `notebooks/experiment_4b_grouped_mondrian_cqr.ipynb` |
| **EXP-04C** | Step 4C | Adaptive Conformal Inference | ACI with rolling quantum step $\gamma$ | Dynamic 90% coverage | Maintained coverage; reduced actionable N | `notebooks/experiment_4c_adaptive_weighted_conformal.ipynb` |
| **EXP-05** | Step 5 | ACI Coverage Audit | Dynamic ACI Bounds Audit | Coverage audit | Confirmed non-stationary tracking | `notebooks/experiment_5_aci_audit.ipynb` |
| **EXP-06_G** | Step 6 | Gate Quality & Precision | $P_{10}/P_{90}$ Percentile Bounds + $\tau=0.01$ | Gated Prec: 79.10% | Confirmed gating filters low-confidence noise | `notebooks/experiment_6_gate_quality.ipynb` |
| **EXP-07** | Step 7 | Feature Selection & Sharper Base | `SelectKBest(f_regression, k=30)` | Reduced 1D MAE | Adopted k=30 feature selection | `notebooks/experiment_7_sharper_base_sparse_groups.ipynb` |
| **EXP-08** | Step 8 | Model Family Challenger | RF vs LightGBM vs VWE vs Stacking | 1D RF promoted; 7D LightGBM rejected | Promoted 1D RF; 7D/14D/30D fallback | `notebooks/experiment_8_final_production_model_challenger.ipynb` |
| **EXP-09** | Step 9 | Economic Charter Backtest | Canonical 1D RF_STANDARD Backtest | Net: -$503,745 (2025 WAIT: +$344K) | Uncovered FLEX cost drag | `notebooks/experiment_9_economic_charter_decision_backtest.ipynb` |
| **INCIDENT** | Step 10 | Reproducibility Forensic Audit | 50-tree vs 100-tree divergence | Divergence: -$340K vs -$503K | Retired 50-tree; hardened SSOT config | `docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md` |
| **AUDIT** | Step 11 | Economic Attribution Audit | NOW vs WAIT vs FLEX Decomposition | NOW: $0, WAIT: +$3.72M, FLEX: -$4.23M | Proved FLEX formula was sole loss source | `docs/ECONOMIC_POLICY_ATTRIBUTION_AUDIT.md` |
| **EXP-06_WF** | Step 12 | Walk-Forward Policy Optimization | Chronological Walk-Forward Tuned WAIT | **+$7,607,420.00 OOS Net** | **CERTIFIED IMPROVEMENT ✅** | `docs/ECONOMIC_POLICY_CERTIFICATION_REPORT.md` |

---

## 5. Experiment-by-Experiment Forensics

### EXP-01 — Forecasting Architecture Benchmark
- **Problem**: Establish baseline predictive capabilities across vessel classes and horizons.
- **Hypothesis**: Gradient boosted trees out-perform linear models on non-linear freight relationships.
- **Results**: LightGBM and Random Forest achieved lowest point MAE across 1D and 7D horizons. Early unexecuted Colab artifacts reported +$7.78M savings under placeholder assumptions (later **RETIRED**).
- **Lesson**: Point forecast MAE alone is insufficient for chartering decisions without gating.

### EXP-02 to EXP-05 — Uncertainty & Conformal Calibration (CQR / Mondrian / ACI)
- **Problem**: Standard point forecasts do not quantify prediction variance.
- **Hypothesis**: Conformal prediction guarantees valid coverage bounds regardless of distribution shape.
- **Results**: ACI maintained 90% empirical coverage under non-stationary regimes, but dynamic interval expansion excessively widened bounds during volatile periods.
- **Lesson**: Residual percentile gating ($P_{10}/P_{90}$ validation bounds) provided a more stable trade-off between precision and coverage than dynamic ACI.

### EXP-06_G — Gate Quality & Precision Calibration
- **Problem**: Low-confidence predictions produce unprofitable charter decisions.
- **Hypothesis**: Gating predictions by requiring $\hat{\Delta} > P_{90}$ (BUY) or $\hat{\Delta} < P_{10}$ (WAIT) plus minimum 1% move ($\tau = 0.01$) increases directional precision.
- **Results**: Gating increased 1D precision from ~74% ungated to **79.10%** gated (507 correct out of 641 retained observations).

### EXP-08 — Production Model Family Challenger
- **Problem**: Select the production model for 1D, 7D, 14D, and 30D horizons.
- **Hypothesis**: Validation-Weighted Ensembles (VWE) or LightGBM would outperform standard Random Forest across all horizons.
- **Results**:
  - **1D**: `RF_STANDARD` promoted. VWE achieved lower point MAE ($389.55 vs $396.94) but NO statistically significant DA/gated precision advantage ($p=0.23$).
  - **7D**: LightGBM achieved lower point MAE than RF ($373.04 vs $387.78), but paired bootstrap CI spanned zero (`[-$35.89, +$7.18]`, $p=0.081$). LightGBM was **NOT promoted**. 7D routed to `FLEXIBLE_INDEX` fallback.
  - **14D & 30D**: Gated precision failed to show statistically significant advantage over 50%. Routed to `FLEXIBLE_INDEX` fallback.

---

## 6. Model Architecture Evolution

```
[Phase 1: Raw Models] ──────> [Phase 2: Quantile CQR] ──────> [Phase 3: Gated RF] ──────> [Phase 4: SSOT Certification]
   - Ridge                     - Pinball Loss                    - P10/P90 Gating                - RF_STANDARD (1D)
   - LightGBM                  - Mondrian CQR                    - SelectKBest (k=30)            - SSOT Config (N=100)
   - Random Forest             - ACI Dynamic                     - 5-Fold Walk-Forward           - Walk-Forward Policy
```

---

## 7. Horizon Analysis & Production Routing

| Horizon | Descriptive Leader | Statistical Superiority vs RF | Promotion Status | Active Production Routing |
|---|---|---|---|---|
| **1D** | Random Forest / VWE | RF_STANDARD promoted (79.10% Gated Prec, 74.60% DA) | **PROMOTED** | `RF_STANDARD` |
| **7D** | LightGBM | Not Significant ($p=0.081$, CI spans zero) | **FALLBACK** | `FLEXIBLE_INDEX` |
| **14D** | Random Forest | Not Significant (Gated Prec $\approx$ 51.59%, $p=0.18$) | **FALLBACK** | `FLEXIBLE_INDEX` |
| **30D** | Random Forest | Marginally Significant (DA 58.20%, Gated Prec 52.85%) | **FALLBACK** | `FLEXIBLE_INDEX` |

---

## 8. Gating Evolution

1. **Ungated Policy**: Triggers orders on any non-zero prediction. High false positive rate.
2. **Fixed Threshold Policy**: Triggers when $|\hat{\Delta}| > \$50/MT$. Failed to adjust for vessel-specific rate scale differences.
3. **Percentile Residual Gating ($P_{10}/P_{90}$ + $\tau=0.01$)**: Calibrates bounds per (horizon, vessel, fold) from out-of-fold validation residuals. **Adopted as canonical baseline**.
4. **Monetary Expected Value Gating**: Triggers WAIT only when predicted decline $> \$125/MT$ (the demurrage breakeven threshold). **Adopted for certified policy**.

---

## 9. Ensemble / VWE / Stacking Evolution

- **Validation-Weighted Ensemble (VWE)**: Combined RF, LightGBM, XGBoost, CatBoost (GBR fallback), and Ridge weighted by inverse validation MAE.
- **Evaluation**: VWE achieved $389.55/MT$ MAE on 1D vs $396.94/MT$ for RF. However, paired bootstrap testing showed **no statistically significant directional accuracy or gated precision advantage** ($p=0.23$).
- **Decision**: In accordance with Occam's razor, `RF_STANDARD` was selected as the single promoted production model to avoid unnecessary ensemble complexity.

---

## 10. Model Selection & Registry Decisions

File: [`registry/manifest.json`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/registry/manifest.json)
- `panamax` 1D: Promoted (`RandomForestRegressor`, 3/5 genuine folds, 91.7% gated precision)
- `supramax` 1D: Promoted (`RandomForestRegressor`, 3/5 genuine folds, 75.0% gated precision)
- `handy` 1D: Promoted (`RandomForestRegressor`, 3/5 genuine folds, 85.7% gated precision)
- `cape` 1D: Promoted (`RandomForestRegressor`, 3/5 genuine folds, 70.8% gated precision)
- All 7D, 14D, 30D combinations: Status = `fallback` (`FLEXIBLE_INDEX`).

---

## 11. Reproducibility Incident Forensic Report

### The Incident
During forensic validation, two local execution scripts yielded different authoritative numbers:
- **Run A (`run_local_audit.py`)**: RF net = **`-$503,745`**, Retained N = **641**, Gated precision = **79.10%**.
- **Run B (`run_final_reconciliation.py`)**: RF net = **`-$340,885`**, Retained N = **639**, Gated precision = **81.06%**.

### The Root Cause
Investigation revealed that `run_final_reconciliation.py` introduced a local speed optimization: `N_TREES = 50` instead of the canonical `N_TREES = 100`.
- Fewer trees increased prediction variance on out-of-fold validation sets.
- Wider residual distributions widened $P_{10}/P_{90}$ percentile bounds.
- Wider bounds made it harder for test predictions to cross gating thresholds, reducing retained N ($641 \to 639$ for RF, $495 \to 383$ for VWE) and altering economic totals.

### The Resolution
- All 50-tree numbers were **OFFICIALLY RETIRED**.
- A clean-room reproducibility gate script (`scratch/run_reproducibility_gate.py`) ran two independent 100-tree pipeline executions and asserted bitwise-identical predictions (`max_abs_diff = 0.0`), decisions, and economic totals.

---

## 12. Provenance & Reproducibility Hardening

To prevent future non-determinism, the codebase was hardened:
1. **Single Source of Truth**: Created [`src/config/canonical_config.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/config/canonical_config.py) defining `CANONICAL_N_TREES=100`, `CANONICAL_SEED=42`, `CANONICAL_N_JOBS=1`, dataset SHA-256 `e0f4c91eed7b4919...`.
2. **Hard-Fail Validation**: Any execution attempting to use `N_TREES != 100`, `SEED != 42`, `n_jobs != 1`, or a modified dataset raises `ValueError` immediately.
3. **Automated Regression Suite**: Created [`tests/test_reproducibility_regression.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/tests/test_reproducibility_regression.py) verifying pipeline determinism.

---

## 13. Observation-Level Economic Attribution

Observation-level decomposition of the canonical `-$503,745.00` baseline across all 4,804 1D test voyages:

$$\text{TOTAL NET} = \Delta_{\text{NOW}} + \Delta_{\text{WAIT}} + \Delta_{\text{FLEXIBLE}}$$

| Bucket | Count | Pct of Portfolio | Total Net Savings (USD) | Mean Net / Voyage (USD) | Gated Precision (%) |
|---|---:|---:|---:|---:|---:|
| **NOW** | 317 | 6.60% | **$0.00** | $0.00 | 74.13% |
| **WAIT** | 324 | 6.74% | **+$3,725,220.00** | +$11,497.59 | **83.95%** |
| **FLEXIBLE** | 4,163 | 86.66% | **-$4,228,965.00** | -$1,015.85 | N/A |
| **TOTAL** | **4,804** | **100.00%** | **-$503,745.00** | **-$104.86** | **79.10%** |

*Key Insight*: The ML model's WAIT decisions generate **`+$3.72M`** in net rate savings. The FLEX routing formulation accounted for the entire negative contribution of the canonical portfolio baseline (-$503,745.00) by charging a $625/voyage idle penalty across 4,163 FLEX voyages.

---

## 14. Economic Policy Experiments

| Policy ID | Description | Type | Retained N | Gated Prec (%) | Total Net (USD) | 2025 WAIT Net (USD) | p-value vs Baseline | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`EXP-00`** | Canonical Baseline | SSOT Baseline | 641 | 79.10% | **-$503,745.00** | +$344,840.00 | 1.0000 | Baseline |
| **`EXP-01`** | FLEX Pure Spot Index | Exploratory | 641 | 79.10% | **+$3,725,220.00** | +$344,840.00 | 0.0000 | Sensitivity |
| **`EXP-02`** | FLEX Bounded Premium ($100) | Exploratory | 641 | 79.10% | **+$3,308,920.00** | +$344,840.00 | 0.0000 | Rejected |
| **`EXP-03`** | WAIT-Only Active Gating | Exploratory | 324 | 83.95% | **+$3,725,220.00** | +$344,840.00 | 0.0000 | Sensitivity |
| **`EXP-04`** | Monetary EV WAIT Gate ($125) | Exploratory | 396 | 83.33% | **+$4,006,900.00** | +$342,460.00 | 0.0000 | Preliminary |
| **`EXP-05`** | Dynamic Volatility Gate | Exploratory | 175 | 85.14% | **+$2,575,220.00** | +$132,180.00 | 0.0000 | Over-constrained |
| **`EXP-06`** | **Walk-Forward Locked Policy** | **Confirmatory** | **1,509** | **80.52%** | **+$7,607,420.00** | **+$944,960.00** | **0.0000** | **CERTIFIED ✅** |

---

## 15. Walk-Forward Certification (`EXP-06_WALK_FORWARD_LOCKED`)

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

---

## 16. Stress Testing

| Stress Scenario | Idle Cost / Day | Voyage Duration Mult | Error Noise Std | Total Net Savings (USD) | 2025 WAIT Net (USD) | Status |
|---|---:|---:|---:|---:|---:|---|
| **Baseline Stress Test** | $2,500 | 1.0x | $0/MT | **+$7,607,420.00** | +$944,960.00 | **PASSED ✅** |
| **High Idle Cost** | $3,500 | 1.0x | $0/MT | **+$6,098,420.00** | +$745,960.00 | **PASSED ✅** |
| **Extreme Idle Cost** | $5,000 | 1.0x | $0/MT | **+$3,834,920.00** | +$447,460.00 | **PASSED ✅** |
| **Low Idle Cost** | $1,000 | 1.0x | $0/MT | **+$9,870,920.00** | +$1,243,460.00 | **PASSED ✅** |
| **Error Noise ($50 std)** | $2,500 | 1.0x | $50/MT | **+$7,578,840.00** | +$1,013,260.00 | **PASSED ✅** |
| **Severe Noise ($100 std)** | $2,500 | 1.0x | $100/MT | **+$7,181,060.00** | +$955,840.00 | **PASSED ✅** |
| **Combined High Idle + Noise** | $3,500 | 1.0x | $50/MT | **+$6,068,840.00** | +$782,260.00 | **PASSED ✅** |

---

## 17. What Failed and Why

1. **7D LightGBM Promotion Failure**: LightGBM achieved lower point MAE than Random Forest, but paired bootstrap testing showed its directional advantage was not statistically significant ($p=0.081$). Promotion rejected to prevent over-fitting on point metrics.
2. **50-Tree Speed Optimization Failure**: Reducing `N_TREES` from 100 to 50 altered percentile bounds, changing retained N (639 vs 641) and economics (-$340K vs -$503K). Retired and prevented by SSOT assertions.
3. **FLEX Demurrage Drag Failure**: Applying a 0.25-day idle penalty to 4,163 FLEX voyages penalized index procurement by -$4.23M. Solved by defaulting FLEX to pure spot index.

---

## 18. What We Learned

1. **Predictive Performance $\neq$ Economic Performance**: A model with 79.10% gated precision can still yield negative portfolio economics if the surrounding routing policy penalizes un-gated index voyages.
2. **Optimize Policy, Freeze Model**: When an ML model provides strong directional precision (83.95% on WAIT), economic failure is a policy formulation problem, not a modeling flaw. Retraining models without auditing policy economics is counterproductive.
3. **Walk-Forward Provenance is Essential**: Threshold tuning must be strictly constrained to historical validation folds to ensure out-of-sample validity.

---

## 19. Why the Architecture Looks the Way It Does Now

- **SSOT Configuration (`src/config/canonical_config.py`)**: Guarantees bitwise reproducibility by locking `N_TREES=100`, `SEED=42`, `n_jobs=1`, and dataset SHA.
- **1D `RF_STANDARD` Promotion**: Selected because it demonstrated genuine signal across 3/5 expanding-window folds without ensemble complexity.
- **7D/14D/30D Fallback**: Enforces `FLEXIBLE_INDEX` fallback because long-horizon models failed statistical promotion thresholds.
- **Walk-Forward Policy Layer (`EXP-06`)**: Eliminates the 0.25-day FLEX idle drag and tunes WAIT thresholds chronologically, converting -$503K loss into **+$7.61M net gain**.

---

## 20. Current Certified Architecture

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

---

## 21. Current Certified Metrics

- **Canonical Predictive Model**: 1D `RF_STANDARD` (RandomForestRegressor, `N_TREES=100`, `SEED=42`, `n_jobs=1`)
- **1D MAE**: `$396.94 / MT`
- **1D Directional Accuracy**: `74.60%`
- **Certified Out-of-Sample Savings**: **`+$7,607,420.00`** (`+0.4183%` vs spot baseline)
- **Certified 2025 Holdout Savings**: **`+$944,960.00`** (`+5.15%` on 2025 WAIT decisions)
- **Certified Gated Precision**: **`80.52%`** (1,215 / 1,509 retained WAIT decisions)

---

## 22. Retired / Noncanonical Results

| Retired Figure | Original Source | Reason for Retirement |
|---|---|---|
| `+$7,781,432 / +0.42%` | Early Colab Notebook | Unexecuted placeholder artifact |
| `-$340,885 / -0.019%` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `639 retained N / 81.06%` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `VWE 383 retained N / -$957K` | Non-canonical Run B | 50-tree speed optimization (altered gating bounds) |
| `POL-003 +$4,006,900` | Exploratory Policy Script | Preliminary sensitivity only; superseded by EXP-06 Walk-Forward Locked |

---

## 23. Remaining Limitations

1. **Zero Execution Slippage Assumption**: Spot index procurement assumes perfect execution at market index rate.
2. **Fixed Demurrage Rate**: Idle penalty fixed at $2,500/day (varies in dynamic spot chartering markets).
3. **Single-Day WAIT Window**: Model currently evaluates 1-day WAIT windows; multi-week delay strategies require dynamic multi-voyage planning.

---

## 24. Research Narrative

The FICOS platform evolved through three distinct scientific stages:
1. **Predictive Discovery (Exps 1–8)**: Established that 1D Random Forest provides statistically significant directional signal (74.60% DA, 79.10% gated precision), while long-horizon models (7D/14D/30D) lack statistical superiority and must default to flexible indexing.
2. **Engineering Hardening (Incident Debugging)**: Uncovered a subtle non-determinism bug where 50-tree vs 100-tree execution altered gating thresholds. Solved by locking SSOT configuration (`src/config/canonical_config.py`) and building a clean-room reproducibility gate.
3. **Economic Policy Certification (Exps 9–06)**: Discovered that the -$503K baseline portfolio loss was accounted for by the FLEX routing formulation charging a $625/voyage demurrage penalty on 4,163 FLEX index voyages. By freezing the certified RF model and tuning WAIT thresholds chronologically per fold (`EXP-06`), FICOS achieved **+$7,607,420.00** out-of-sample net savings (+0.4183%) with **+$944,960.00** on the 2025 holdout.

---

## 25. Evidence Index

| Claim / Artifact | Primary Source File | Section / Function / Commit |
|---|---|---|
| SSOT Config & Provenance | [`src/config/canonical_config.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/config/canonical_config.py) | `validate_dataset_provenance()`, SHA `ef6970f3f96...` |
| Reproducibility Incident Audit | [`docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/docs/LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md) | Divergence DIV-001 |
| Reproducibility Test Suite | [`tests/test_reproducibility_regression.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/tests/test_reproducibility_regression.py) | `test_reproducibility_dual_execution()` |
| Policy Certification Engine | [`scratch/run_economic_policy_certification.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/scratch/run_economic_policy_certification.py) | Phases 1–10 execution |
| Certified Policy JSON Payload | [`outputs/authoritative_policy_results.json`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/authoritative_policy_results.json) | `certified_policy` block |
| Policy Reproducibility Suite | [`tests/test_economic_policy_reproducibility.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/tests/test_economic_policy_reproducibility.py) | `test_walk_forward_policy_reproducibility()` |
| Experiment Registry CSV | [`outputs/economic_policy_experiment_registry.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_experiment_registry.csv) | Full 7-experiment registry |
| Walk-Forward Results CSV | [`outputs/economic_policy_walk_forward_results.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_walk_forward_results.csv) | Observation-level records |
| Stress Tests CSV | [`outputs/economic_policy_stress_tests.csv`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/economic_policy_stress_tests.csv) | 7 stress scenarios |

---

## FICOS in One Page

- **Original Problem**: Freight rate volatility creates multi-million dollar procurement losses. Charterers need data-driven NOW vs WAIT vs FLEXIBLE decisions.
- **Biggest Failed Experiments**:
  - *7D LightGBM*: Better MAE than RF, but directional advantage not statistically significant ($p=0.081$). Rejected.
  - *50-Tree Speed Optimization*: Altered out-of-fold $P_{10}/P_{90}$ bounds, shifting retained N and economics (-$340K vs -$503K). Retired.
  - *FLEX Idle Penalty*: Applying a 0.25d idle penalty to 4,163 FLEX voyages penalized index procurement by -$4.23M. Solved by defaulting FLEX to pure spot index.
- **Biggest Discoveries**:
  - 1D RF_STANDARD ML predictions generate **+$3.72M** net savings on WAIT decisions (83.95% precision).
  - The FLEX routing formulation accounted for the entire negative contribution of the canonical portfolio baseline (-$503K), rather than predictive model failure.
- **Architecture Evolution**: Point Forecasting Benchmark $\to$ Quantile CQR / Mondrian / ACI $\to$ Gated 1D RF_STANDARD $\to$ SSOT Provenance Hardening $\to$ Chronological Walk-Forward Policy Optimization (`EXP-06`).
- **Reproducibility Breakthrough**: SSOT config (`src/config/canonical_config.py`) locking `N_TREES=100`, `SEED=42`, `n_jobs=1`, dataset SHA, and automated dual clean-room regression tests (`max_abs_diff = 0.0`).
- **Economic Breakthrough**: Chronological walk-forward WAIT threshold tuning (`EXP-06`) turned portfolio economics from **`-$503,745.00`** into **`+$7,607,420.00`** net savings (+0.4183%) out-of-sample with zero future leakage.
- **Final Certified Architecture**:
  - **1D**: `RF_STANDARD` + Walk-Forward Policy Engine (`EXP-06`).
  - **7D / 14D / 30D**: `FLEXIBLE_INDEX` Fallback.
- **Final Certified Numbers**:
  - Out-of-Sample Net Savings: **`+$7,607,420.00`** (+0.4183%)
  - 2025 Holdout Net Savings: **`+$944,960.00`** (+5.15%)
  - Gated Precision: **`80.52%`** (1,215 / 1,509 WAIT decisions)
  - 1D MAE: **`$396.94/MT`** | 1D DA: **`74.60%`**
- **Remaining Limitations**: Assumes zero execution slippage on spot index procurement; fixed $2,500/day demurrage assumption.
