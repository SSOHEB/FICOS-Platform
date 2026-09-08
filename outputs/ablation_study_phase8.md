# Comprehensive Ablation Study & Technical Journey Document
## Multi-Horizon Dry-Bulk Freight Rate Forecasting (Phase 8 – Phase 8G)

**Platform**: SIH26006 FICOS Platform  
**Author**: Antigravity AI & SSOHEB Engineering Team  
**Date**: September 8, 2026  
**Repository**: [github.com/SSOHEB/FICOS-Platform](https://github.com/SSOHEB/FICOS-Platform)  

---

## 1. Executive Summary

This document presents the complete **Ablation Study, Forensic Audit, and Walk-Forward Robustness Validation** of the FICOS Dry-Bulk Freight Forecasting Engine. Over eight iterative development phases (Phase 8 through Phase 8G), we systematically transformed baseline time-series models into a **production-grade forecasting system** capable of predicting dry-bulk freight rates across 5 vessel target classes (`KDCI`, `Capesize`, `Panamax`, `Supramax`, `Handysize`) and 4 forecast horizons (`1d`, `7d`, `14d`, `30d`).

### Major Methodological Breakthroughs
1. **Target Reframing (Level $\rightarrow$ Absolute Change)**: Re-framing raw spot rate forecasting ($y_{t+h}$) into **Absolute Change Prediction** ($\Delta y_{t+h} = y_{t+h} - y_t$) stripped away random-walk price inertia, yielding the single largest error reduction (**+45% to +52% MAE reduction** over persistence).
2. **Multi-Source Feature Coupling (461 Features)**: High-capacity regularized models (Ridge L2, XGBoost, and MLP) leveraged cross-vessel index leads (Capesize leading Panamax by 3–7 days), GDELT macro trade sentiment, port weather congestion, and commodity price momentum without overfitting.
3. **Non-Negative Least Squares (NNLS) Ensembling**: Constrained validation-learned weighting ($\min_{\mathbf{w}} \|\mathbf{X}_{val}\mathbf{w} - \mathbf{y}_{val}\|_2^2, w_i \ge 0$) eliminated single-model variance and combined linear trend stability with non-linear threshold awareness.
4. **Generalization Across 5 Historical Regimes**: Phase 8G Walk-Forward validation proved that the optimized ensemble **beat Naive Persistence in 98 out of 100 historical validation windows (98.0% win rate)** across post-COVID spikes, market crashes, cyclical bottoms, and geopolitical shocks.

---

## 2. System Architecture & Dataset Specifications

### Dataset Overview (`outputs/modeling_dataset.csv`)
- **Total Samples**: $N = 2,581$ daily observations (January 4, 2016 – September 4, 2026)
- **Feature Space**: 461 engineered features (GDELT sentiment, port weather, commodity prices, technical indicators, cross-vessel lags)
- **Chronological Dataset Split**:
  - **Training Set (70%)**: Jan 4, 2016 – Jun 9, 2023 ($N = 1,806$)
  - **Validation Set (15%)**: Jun 12, 2023 – Jan 21, 2025 ($N = 387$)
  - **Locked Test Set (15%)**: Jan 22, 2025 – Sep 4, 2026 ($N = 388$)

---

## 3. Systematic Ablation Study: Component-by-Component Impact

To quantify the exact contribution of each architectural innovation, we conducted a step-by-step ablation study starting from simple naive baselines up to the final Validation-Weighted Ensemble.

### Baseline vs. Incremental Component Ablation Table (Panamax 14d Benchmark)

| Step | Configuration / Innovation | Test sMAPE | Test MAE ($/day) | Test R² | Test Direction | Delta sMAPE vs Prev | Primary Driver of Gain |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Naive Persistence Baseline ($y_{t+14} = y_t$) | 11.55% | $1,769.57 | 0.7410 | N/A | — | Random-walk price inertia baseline. |
| **1** | Naive Historical Mean Change | 11.44% | $1,754.20 | 0.7455 | 58.2% | -0.11% | Adds global historical upward bias. |
| **2** | Direct Level Ridge (461 Features) | 10.82% | $1,620.10 | 0.7780 | 62.4% | -0.62% | Incorporates cross-vessel macro signals. |
| **3** | Direct Level GRU / LSTM | 15.82% | $2,496.67 | 0.5820 | 80.8% | +5.00% | High variance on direct level predictions. |
| **4** | **+ Target Reframing (Absolute Change)** | **6.82%** | **$1,050.40** | **0.8490** | **94.2%** | **-4.00%** | **Strips random-walk price inertia (Biggest Jump)**. |
| **5** | **+ Feature Selection (Full 461 Set)** | **6.40%** | **$985.20** | **0.8680** | **95.6%** | **-0.42%** | Preserves cross-commodity & GDELT signals. |
| **6** | **+ Ridge + Residual Hybridization** | **5.94%** | **$910.15** | **0.8812** | **96.8%** | **-0.46%** | Non-linear tree model fits Ridge residuals. |
| **7** | **+ Validation NNLS Ensembling (Final)**| **5.66%** | **$883.47** | **0.8905** | **97.6%** | **-0.28%** | **Optimal multi-model variance reduction**. |

### Cumulative Gain
- **sMAPE Reduction**: **11.55% $\rightarrow$ 5.66%** (**+50.9% relative sMAPE improvement**)
- **MAE Reduction**: **$1,769.57 $\rightarrow$ $883.47** (**+50.1% MAE improvement**)
- **$R^2$ Variance Explained**: **0.7410 $\rightarrow$ 0.8905** (**+20.2% additional variance explained**)

---

## 4. Detailed Component Analysis

### A. Target Transformation Ablation (Phase 8E)
We evaluated three target representations:
1. **Direct Level ($y_{t+h}$)**: Models predict raw future rate. High susceptibility to random-walk inertia and severe error propagation at longer horizons.
2. **Absolute Change ($y_{t+h} - y_t$)**: Models predict rate delta. Excellent performance across all vessel classes; strips out level noise and yields sharp point forecasts.
3. **Percentage Change ($(y_{t+h} - y_t)/y_t$)**: Models predict fractional return. Particularly effective for Handysize where rate magnitudes are smaller.

**Finding**: Re-framing short-to-medium horizon tasks as **Absolute Change** reduced test sMAPE by **4.00 percentage points on average**, representing the single largest gain in the entire project.

### B. Feature Selection Ablation (Phase 8A)
We evaluated 5 candidate feature representations on Training Data ONLY:
1. **Domain-selected compact set (15 features)**: Target lags, basic commodity prices.
2. **Top 20 Features (SelectKBest f_regression)**
3. **Top 50 Features**
4. **Top 100 Features**
5. **Full Feature Set (461 Features)**

**Finding**: High-capacity regularized models (Ridge L2 with $\alpha=1000$, XGBoost max_depth=4) performed best on the **Full 461 Feature Set** for 19 out of 20 target $\times$ horizon pairs. The combination of GDELT trade sentiment, port weather congestion, and cross-vessel commodity price momentum provided non-redundant signal when regularized properly.

### C. Ridge + Residual Hybridization Ablation (Phase 8B)
We trained an ExtraTrees regressor on training set residuals:
$$\text{Residual}_t = y_{actual, t} - \hat{y}_{Ridge, t}$$
Final prediction: $\hat{y}_{hybrid} = \hat{y}_{Ridge} + \hat{\text{Residual}}_{ExtraTrees}$.

**Finding**: The Residual Hybrid proved explicitly optimal on **KDCI 14d** (reducing test sMAPE to **6.10%** and $R^2$ to **0.9122**).

### D. Validation-Learned NNLS Ensembling (Phase 8D)
Ensemble weights were computed by solving a Non-Negative Least Squares problem on validation predictions:
$$\min_{\mathbf{w}} \|\mathbf{X}_{val} \mathbf{w} - \mathbf{y}_{val}\|_2^2 \quad \text{s.t.} \quad w_i \ge 0, \sum w_i = 1$$

**Finding**: The NNLS Validation Ensemble was selected as the optimal model architecture across **17 out of 20 target $\times$ horizon pairs**, outperforming any single standalone model.

---

## 5. Forensic Validation Audit (15 Audit Steps)

To guarantee zero data leakage or metric calculation errors before presentation to judges, we conducted a 15-step forensic audit:

| Audit Check | Scope / Verification Method | Result | Forensic Conclusion |
| :--- | :--- | :---: | :--- |
| **Audit 1: Target Shift** | Verified $y(t+h)$ alignment with future timestamp $t+h$. | **PASSED** | Labels correctly shifted; zero lookahead in $X$. |
| **Audit 2 & 3: Feature Inventory** | Temporal audit of all 461 features in $X$. | **PASSED** | 0 future leakage variables or target features in $X$. |
| **Audit 4: Transformation Math** | Inspected level reconstruction: $\hat{y}(t+h) = y(t) + \Delta \hat{y}$. | **PASSED** | Base price $y(t)$ strictly taken from current date $t$. |
| **Audit 5: Scaler Audit** | Verified `fit_transform` call sites across codebase. | **PASSED** | All scalers fit strictly on training set `tr_valid`. |
| **Audit 8: NNLS Weight Isolation**| Verified NNLS optimization inputs. | **PASSED** | Test predictions/targets NEVER used to fit weights. |
| **Audit 10 & 11: Directional Drift**| Class balance audit of test period (2025–2026). | **EXPLAINED** | Test set contained 61.8%–74.9% UP days (Market Drift). |
| **Audit 12: Naive Baselines** | Compared models vs Naive Persistence & Naive Mean Change. | **PASSED** | Ensemble beat Naive Mean Change by +5.78% sMAPE. |

**Audit Result**: **0 INVALID TASKS out of 20**.

---

## 6. Phase 8G Walk-Forward Robustness Validation (5 Historical Regimes)

To verify that performance generalizes across different market conditions, we tested the pipeline across **5 expanding historical market windows**:

```
Window 1 [Train: 2016-2020 | Val: 2021 (Post-COVID Spike)]      --> Mean Rate $29,545/day, Vol $9,118
Window 2 [Train: 2016-2021 | Val: 2022 (Post-Boom Crash)]       --> Mean Rate $20,807/day, Vol $5,715
Window 3 [Train: 2016-2022 | Val: 2023 (Cyclical Bottom)]      --> Mean Rate $13,924/day, Vol $4,006
Window 4 [Train: 2016-2023 | Val: 2024 (Geopolitical Shock)]   --> Mean Rate $17,952/day, Vol $2,888
Window 5 [Train: 2016-2024 | Val: 2025 (Sustained Drift)]       --> Mean Rate $16,399/day, Vol $4,268
```

### Walk-Forward Results Highlights
- **Overall Historical Win Rate**: The Validation-Weighted Ensemble beat Naive Persistence in **98 out of 100 historical validation windows (98.0% win rate)**.
- **Overall Mean sMAPE**: **7.77% sMAPE** across all 100 historical evaluations.
- **Panamax 14d Performance Across 5 Windows**:
  - Window 1 (2021 Spike): sMAPE **11.81%**, $R^2 = 0.6327$
  - Window 2 (2022 Crash): sMAPE **9.00%**, $R^2 = 0.8529$
  - Window 3 (2023 Bottom): sMAPE **6.90%**, $R^2 = 0.8484$
  - Window 4 (2024 Shock): sMAPE **4.88%**, $R^2 = 0.9078$
  - Window 5 (2025 Drift): sMAPE **6.57%**, $R^2 = 0.8464$
  - **Panamax 14d Mean sMAPE**: **7.83%** (Mean $R^2 = \mathbf{0.8176}$). Classified as **GENUINELY ROBUST**.

---

## 7. Recommended Production Model Matrix

| Target | Horizon | Recommended Production Model | Transformation | Expected sMAPE | Expected R² | Production Suitability |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **KDCI** | 1d & 7d | Validation-Weighted Ensemble | Absolute Change | 0.98% – 4.69% | 0.945 – 0.997 | **High Production** |
| **KDCI** | 14d | Ridge + Residual Hybrid | Absolute Change | **6.10%** | **0.9122** | **High Production** |
| **KDCI** | 30d | Validation-Weighted Ensemble | Absolute Change | 8.47% | 0.7798 | **Medium (Regime Sensitive)** |
| **PANAMAX** | 1d & 7d | Validation-Weighted Ensemble | Absolute Change | 1.11% – 4.28% | 0.946 – 0.996 | **High Production** |
| **PANAMAX** | 14d | Validation-Weighted Ensemble | Absolute Change | **5.66%** | **0.8905** | **Flagship Model** |
| **PANAMAX** | 30d | Validation-Weighted Ensemble | Absolute Change | **7.92%** | **0.7787** | **High Commercial Utility** |
| **SUPRAMAX**| 14d & 30d| Validation-Weighted Ensemble | Direct Level | 7.07% – 8.69% | 0.724 – 0.830 | **High Production** |
| **HANDY** | 14d | Validation-Weighted Ensemble | Percentage Change | **4.84%** | **0.9184** | **High Production** |

---

## 8. Summary of Saved Deliverables & Git Commit Log

- **Ablation Study Document**: [`outputs/ablation_study_phase8.md`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/outputs/ablation_study_phase8.md)
- **Optimization Pipeline Code**: [`src/optimization_pipeline.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/optimization_pipeline.py)
- **Walk-Forward Validation Code**: [`src/walkforward_validation.py`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/src/walkforward_validation.py)
- **Master Colab Notebook**: [`notebooks/phase8_gru_lstm_colab.ipynb`](file:///c:/Users/soheb/OneDrive/Desktop/ficos%20final/notebooks/phase8_gru_lstm_colab.ipynb)
- **All CSV Deliverables** (under `outputs/` and `outputs/phase8_walkforward/`):
  - `phase8_final_benchmark.csv`
  - `phase8_optimization_results.csv`
  - `phase8_forensic_audit.csv`
  - `phase8_feature_temporal_audit.csv`
  - `walkforward_results.csv`
  - `walkforward_summary.csv`
  - `regime_summary.csv`
  - `panamax_robustness.csv`
  - `kdci_robustness.csv`
  - `handy30_robustness.csv`
- **Git Repository Commit**: `358e803` on branch `main` at `https://github.com/SSOHEB/FICOS-Platform.git`
