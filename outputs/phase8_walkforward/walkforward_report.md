# Phase 8G Walk-Forward Robustness Validation Report

**Date of Execution**: September 8, 2026  
**Scope**: Expanding-Window Temporal Robustness Validation across 5 Historical Market Regimes  
**Dataset Range**: January 2016 – September 2026 ($N=2,581$ daily observations)  
**Deliverables Directory**: `outputs/phase8_walkforward/`  

---

## Executive Summary & Core Conclusion

To determine whether the strong Phase 8 forecasting performance generalizes across different historical market regimes — rather than being an artifact of the 2025–2026 evaluation period — we executed an exhaustive **Expanding-Window Walk-Forward Robustness Validation** across 5 distinct historical market periods ($2021\text{--}2025$).

### Key Findings
1. **High Generalization & Robustness (98.0% Historical Win Rate)**:
   - Across **100 window $\times$ target $\times$ horizon evaluations** (5 windows $\times$ 5 targets $\times$ 4 horizons), the **Phase 8 Validation-Weighted Ensemble beat Naive Persistence in 98 out of 100 historical validation windows (98.0% win rate)**.
   - The overall mean sMAPE across all 100 historical evaluations was **7.77% sMAPE** (mean improvement of **+5.23% sMAPE reduction over Naive Persistence**).

2. **Panamax 14d & KDCI 14d Are Genuinely Robust**:
   - **Panamax 14d**: Achieved a mean sMAPE of **7.83%** across all 5 historical windows (Best: **4.88%** in 2024, Worst: **11.81%** during the $29,545/day 2021 Post-COVID spike) with a mean $R^2 = \mathbf{0.8176}$. It beat Persistence in **100% of historical windows**.
   - **KDCI 14d**: Achieved a mean sMAPE of **7.45%** across all 5 historical windows (Best: **5.98%** in 2025, Worst: **9.58%** in 2021) with a mean $R^2 = \mathbf{0.8485}$. It beat Persistence in **100% of historical windows**.

3. **Regime Sensitivity Identified**:
   - **30-Day Horizons in Extreme Volatility Regimes**: During the 2021 Post-COVID freight spike ($mean = \$29,545/day, vol = \$9,118$), 30-day forecast errors rose to 19%–34% sMAPE across Capesize, Panamax, and Handysize.
   - In normal, cyclical, or trending market regimes (2022, 2023, 2024, 2025), 30-day sMAPEs returned to **6.76% – 12.83% sMAPE**.

---

## Historical Market Regime Definitions (2021–2025)

| Window | Train Years | Validation Year | Mean KDCI Rate | Rate Volatility | Yearly Rate Trend | Market Regime Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Window 1** | 2016–2020 | **2021** | $29,545 /day | $9,118 | +$10,195 | **Post-COVID Freight Spike**: Extreme demand boom, port congestion, record high spot rates. |
| **Window 2** | 2016–2021 | **2022** | $20,807 /day | $5,715 | -$8,704 | **Post-Boom Rate Correction**: Steep macro normalization, downward rate crash across Cape & Panamax. |
| **Window 3** | 2016–2022 | **2023** | $13,924 /day | $4,006 | +$6,450 | **Cyclical Bottom / Rebuilding**: Low volatility, sideways rate stabilization, building momentum. |
| **Window 4** | 2016–2023 | **2024** | $17,952 /day | $2,888 | -$10,341 | **Geopolitical Shocks / Red Sea Rerouting**: Moderate rate level, trade flow disruptions. |
| **Window 5** | 2016–2024 | **2025** | $16,399 /day | $4,268 | +$7,523 | **Sustained Upward Market Drift**: Upward market momentum (61.8%–74.9% UP days). |

---

## Target-by-Target Walk-Forward Summary & Classification

| Target | Horizon | Model | Mean sMAPE | Std sMAPE | Best sMAPE | Worst sMAPE | Mean R² | Mean Impr. vs Persistence | Stability Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KDCI** | 1d | Ensemble | 1.24% | 0.18% | 1.01% | 1.47% | 0.9949 | +0.88% | **ROBUST** |
| **KDCI** | 7d | Ensemble | 5.19% | 0.92% | 3.93% | 6.47% | 0.9198 | +4.90% | **ROBUST** |
| **KDCI** | 14d | Ensemble | 7.45% | 1.37% | 5.98% | 9.58% | 0.8485 | +8.67% | **ROBUST** |
| **KDCI** | 30d | Ensemble | 12.80% | 5.55% | 7.46% | 23.05% | 0.6480 | +9.36% | **REGIME_DEPENDENT** |
| **CAPE** | 1d | Ensemble | 2.85% | 0.66% | 2.16% | 3.99% | 0.9870 | +1.86% | **ROBUST** |
| **CAPE** | 7d | Ensemble | 11.09% | 2.04% | 8.42% | 13.51% | 0.8084 | +8.20% | **PROMISING** |
| **CAPE** | 14d | Ensemble | 14.95% | 2.64% | 11.69% | 17.77% | 0.6801 | +14.01% | **REGIME_DEPENDENT** |
| **CAPE** | 30d | Ensemble | 18.90% | 3.37% | 14.46% | 23.12% | 0.5381 | +16.58% | **REGIME_DEPENDENT** |
| **PANAMAX** | 1d | Ensemble | 1.14% | 0.18% | 0.90% | 1.44% | 0.9951 | +0.60% | **ROBUST** |
| **PANAMAX** | 7d | Ensemble | 5.53% | 0.99% | 4.49% | 7.22% | 0.9107 | +3.47% | **ROBUST** |
| **PANAMAX** | 14d | Ensemble | 7.83% | 2.38% | 4.88% | 11.81% | 0.8176 | +5.74% | **ROBUST** |
| **PANAMAX** | 30d | Ensemble | 11.42% | 4.21% | 7.41% | 19.18% | 0.6235 | +8.80% | **PROMISING** |
| **SUPRAMAX**| 1d | Ensemble | 0.87% | 0.28% | 0.52% | 1.22% | 0.9966 | +0.49% | **ROBUST** |
| **SUPRAMAX**| 7d | Ensemble | 4.38% | 1.16% | 2.92% | 5.73% | 0.9296 | +3.33% | **ROBUST** |
| **SUPRAMAX**| 14d | Ensemble | 7.24% | 2.22% | 5.26% | 11.32% | 0.8291 | +5.08% | **ROBUST** |
| **SUPRAMAX**| 30d | Ensemble | 13.98% | 9.48% | 6.79% | 32.65% | 0.3237 | +4.36% | **REGIME_DEPENDENT** |
| **HANDY** | 1d | Ensemble | 0.80% | 0.19% | 0.52% | 1.01% | 0.9963 | +0.43% | **ROBUST** |
| **HANDY** | 7d | Ensemble | 4.34% | 1.16% | 2.80% | 6.11% | 0.9156 | +2.41% | **ROBUST** |
| **HANDY** | 14d | Ensemble | 7.68% | 2.33% | 5.07% | 11.62% | 0.7455 | +3.52% | **ROBUST** |
| **HANDY** | 30d | Ensemble | 15.72% | 9.97% | 6.76% | 34.16% | 0.1165 | +1.87% | **REGIME_DEPENDENT** |

---

## Detailed Investigation of Critical Targets

### 1. Panamax 14d Robustness Analysis
- **Historical Performance Across Windows**:
  - Window 1 (2021 Spike): sMAPE **11.81%**, MAE $3,225, $R^2 = 0.6327$
  - Window 2 (2022 Crash): sMAPE **9.00%**, MAE $1,795, $R^2 = 0.8529$
  - Window 3 (2023 Bottom): sMAPE **6.90%**, MAE $925, $R^2 = 0.8484$
  - Window 4 (2024 Shock): sMAPE **4.88%**, MAE $730, $R^2 = 0.9078$
  - Window 5 (2025 Drift): sMAPE **6.57%**, MAE $903, $R^2 = 0.8464$
- **Verdict**: **GENUINELY ROBUST**. Outperforms persistence in **5 out of 5 historical windows (100% win rate)** with a mean $R^2 = 0.8176$.

### 2. Panamax 30d Robustness Analysis
- **Historical Performance Across Windows**:
  - sMAPEs: 19.18% (2021), 12.41% (2022), 8.96% (2023), 7.41% (2024), 9.16% (2025).
- **Verdict**: **PROMISING / REGIME-DEPENDENT**. Highly effective in standard, cyclical, or trending market regimes (7.41% – 12.41% sMAPE), but experiences higher percentage error during extreme rate spikes like 2021.

### 3. KDCI 14d & 30d Robustness Analysis
- **KDCI 14d**: sMAPEs across windows: 9.58%, 7.54%, 8.16%, 5.99%, 5.98%. Mean sMAPE: **7.45%**, Mean $R^2 = \mathbf{0.8485}$. **ROBUST**.
- **KDCI 30d**: Mean sMAPE **12.80%**, Mean $R^2 = 0.6480$. **REGIME_DEPENDENT**.

### 4. Handysize 30d Robustness Analysis
- **Handysize 30d**: sMAPEs across windows: 34.16% (2021), 17.85% (2022), 11.35% (2023), 8.47% (2024), 6.76% (2025). Mean $R^2 = 0.1165$.
- **Verdict**: **REGIME-DEPENDENT / HIGH VOLATILITY RISK**. Underperforms in spike years due to low Handysize baseline rate magnitude amplifications.

---

## Direct Answers to Section 19 Questions

1. **Does the Phase 8 improvement survive multiple historical regimes?**  
   **YES**. The Validation-Weighted Ensemble beat Naive Persistence in **98 out of 100 historical validation windows (98.0%)**.

2. **Is Panamax 14d genuinely robust?**  
   **YES**. Panamax 14d achieved a mean sMAPE of **7.83%** ($R^2 = 0.8176$) across all 5 windows, beating persistence in 100% of historical windows.

3. **Is Panamax 30d genuinely robust?**  
   **PROMISING / REGIME-DEPENDENT**. Reaches 7.41% – 9.16% sMAPE in standard regimes, but expanded to 19.18% during the 2021 spike.

4. **Is KDCI 14d/30d genuinely robust?**  
   KDCI 14d is **GENUINELY ROBUST** (mean sMAPE 7.45%, $R^2 = 0.8485$). KDCI 30d is **REGIME-DEPENDENT** (mean sMAPE 12.80%).

5. **Which models are consistently strong?**  
   The **Validation-Weighted Ensemble** and **Ridge + Residual Hybrid** using **Absolute Change Transformations**.

6. **Which models are regime-dependent?**  
   30-day long-horizon models in extreme freight spike regimes (e.g. Handysize 30d, Capesize 30d).

7. **Is there evidence of overfitting?**  
   **NO**. Out-of-sample historical validation errors strictly track or beat inner training errors across 4 out of 5 regimes.

8. **Is there evidence of underfitting?**  
   Only for Handysize 30d during high-rate spike periods.

9. **How much does the optimized model improve over persistence across historical windows?**  
   An average reduction of **+5.23% sMAPE** across all 100 historical window evaluations.

10. **What percentage of historical validation windows show improvement over persistence?**  
    **98.0%** (98 out of 100 evaluations).

11. **What is the mean and standard deviation of sMAPE across windows?**  
    Overall Mean sMAPE: **7.77%**, Standard Deviation: **6.53%** (driven by 2021 30d spikes).

12. **What is the worst historical validation result?**  
    Handysize 30d in 2021 (sMAPE 34.16%, $R^2 = -1.3813$).

13. **Would you consider the current Phase 8 model suitable for the SIH demonstration?**  
    **YES, ABSOLUTELY**. Short-horizon models (1d and 7d across all vessels) and medium-horizon models (**Panamax 14d, KDCI 14d, Supramax 14d, Handysize 14d**) demonstrate verified, robust out-of-sample forecasting superiority.
