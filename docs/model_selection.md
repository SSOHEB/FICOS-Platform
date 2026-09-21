# FICOS Model Selection & Scientific Validation Framework

This document outlines the authoritative scientific methodology, model tournament results, zero-leakage safeguards, and promotion/abstention policies implemented in the FICOS (Freight Intelligence & Chartering Optimization System) platform.

---

## 1. Walk-Forward Validation Protocol

To reflect real-world shipping market conditions without look-ahead bias, FICOS enforces a **5-fold chronological expanding-window walk-forward validation** protocol spanning 2021–2025 ($N \approx 1,242$ out-of-sample trading days):

- **Training Window**: Expanding historical window ($t_0 \to t_{\text{val\_start}}$)
- **Validation Window**: 200 trading days for hyperparameter tuning & residual uncertainty bounds ($P_{10}, P_{90}$)
- **Test Window**: 250 trading days evaluated strictly **once** with frozen parameters

```
Fold 1: [--- Train (2016-2020) ---][ Val (200d) ][ Test 1 (250d) ]
Fold 2: [------ Train (2016-2021) ------][ Val (200d) ][ Test 2 (250d) ]
Fold 3: [--------- Train (2016-2022) ---------][ Val (200d) ][ Test 3 (250d) ]
Fold 4: [------------ Train (2016-2023) ------------][ Val (200d) ][ Test 4 (250d) ]
Fold 5: [--------------- Train (2016-2024) ---------------][ Val (200d) ][ Test 5 (250d) ]
```

---

## 2. Zero-Leakage Preprocessing & Gating Architecture

1. **Feature Quarantine**: All 41 future-dated, direction-derived, and timestamp columns are quarantined. Exactly 441 clean predictors are passed to the model.
2. **Fold-Isolated Transforms**:
   - `SelectKBest(f_regression, k=30)` is fit strictly on training masks.
   - `StandardScaler` and `MedianImputer` are fit strictly on training masks.
3. **Out-of-Fold Residual Gating**:
   - Gating thresholds $\tau$ and residual bounds ($P_{10}, P_{90}$) are computed solely from validation errors:
     $$\epsilon_v = y_v - \hat{y}_v \quad \Longrightarrow \quad P_{10} = \text{Percentile}(\epsilon_v, 10), \quad P_{90} = \text{Percentile}(\epsilon_v, 90)$$
   - A forecast is gated as **High Conviction** if:
     $$\hat{y}_{\text{test}} > \max(0, P_{90}) \quad \text{and} \quad \frac{\hat{y}_{\text{test}}}{y_{\text{base}}} > \tau$$
     $$\text{or} \quad \hat{y}_{\text{test}} < \min(0, P_{10}) \quad \text{and} \quad \frac{\hat{y}_{\text{test}}}{y_{\text{base}}} < -\tau$$

---

## 3. Audited Master Results & Promotion Table

| Asset & Horizon | Winning Model | Ungated DA | Gated DA | Coverage | Gated F1 | AUC (Gated) | Verdict & Routing |
|---|---|---|---|---|---|---|---|
| **PANAMAX_1D** | RandomForest / Ridge | **78.1%** | **91.1%** | 17.2% ($N=214$) | **91.3%** | **0.924** | 🟢 **PROMOTED** (Primary Signal) |
| **SUPRAMAX_1D** | RandomForest / XGBoost | **75.1%** | **85.0%** | 16.1% ($N=200$) | **83.0%** | **0.805** | 🟢 **PROMOTED** (High Conviction) |
| **HANDY_1D** | RandomForest / LightGBM | **70.7%** | **79.2%** | 11.6% ($N=144$) | **79.2%** | **0.750** | 🟢 **PROMOTED** |
| **CAPE_1D** | RandomForest / Ridge | **66.5%** | **71.3%** | 14.3% ($N=174$) | **52.8%** | **0.711** | 🟡 **PROMOTED (Conservative)** |
| **KDCI_7D** | RandomForest / GBDT | **58.7%** | **76.7%** | 12.1% ($N=150$) | **55.7%** | **0.801** | 🟡 **FALLBACK / INDEX-LINKED** |
| **SUPRAMAX_7D** | GBDT / Ridge | **62.0%** | **63.8%** | 19.0% ($N=235$) | **50.9%** | **0.744** | ⚠️ **FALLBACK (Flexible Index)** |
| **HANDY_7D** | Linear / RF | **60.3%** | **58.6%** | 19.3% ($N=239$) | **57.1%** | **0.593** | 🛑 **EXCLUDED (Abstain)** |
| **SUPRAMAX_14D**| Linear / GBDT | **54.4%** | **49.1%** | 40.9% ($N=503$) | **17.9%** | **0.564** | 🛑 **EXCLUDED (Abstain)** |

---

## 4. Decision Engine Abstention & Fallback Policy

To prevent costly misallocation during low-predictability regimes:
1. **Promoted Pairs (`1D`)**: Feed directional forecasts directly into the Expected Cost Optimizer & Risk Engine.
2. **Excluded Pairs (`SUPRAMAX_14D`, `HANDY_7D`)**: The system explicitly **ABSTAINS** from directional betting (`gate_status = "FALLBACK_UNPROMOTED"`) and routes chartering decisions to **FLEXIBLE_INDEX** (index-linked floating rate contracts with quarterly review options).
3. **Regime-Dependent Pairs (`7D`)**: Operate under widened fallback uncertainty bounds ($\pm 25\%$ current market rate).

---

## 5. Statistical Significance

- **Time-Series Aware Permutation Testing ($B=200$)**:
  - Circular-shift permutation preserves intra-series autocorrelation.
  - Directional accuracy on promoted pairs achieves **$p < 0.005$** against null distributions (mean null accuracy $51.8\% \pm 3.9\%$).

---

## 6. Economic Backtest & Scientific Disclaimer (Experiment 9)

1. **Forecast Validation vs. Economic Validation**: Forecast validation evaluates statistical accuracy ($R^2$, DA, AUC) under zero-leakage walk-forward protocols. Economic backtesting (Experiment 9) evaluates charter decision policy cost differences against counterfactual baselines.
2. **Model Promotion Semantics**: Registry promotion indicates verified walk-forward directional signal. It **does NOT constitute proof of monetary savings**.
3. **Experiment 9 Classification**: Experiment 9 is a decision-level economic validation study evaluating counterfactual cost differences vs. Always Spot on production-promoted pairs ($N=952$).
4. **Current Economic Conclusion**: The aggregate economic outcome is **inconclusive** (-0.131% backtested cost difference vs spot, 95% CI [-0.265%, +0.005%]).
5. **WAIT Decision Counterfactual**: WAIT decisions achieved a +3.174% counterfactual cost difference vs spot, which was statistically unusual ($p < 0.0001$) relative to a 10,000-draw placebo distribution. However, this finding is counterfactual evidence under stated evaluation assumptions and **does NOT prove real-world monetary savings**.
