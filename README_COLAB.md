# Google Colab Freight Forecasting Benchmark Experiment Guide

This guide provides instructions to run the freight forecasting benchmark notebook (`notebooks/colab_freight_forecasting_benchmark.ipynb`) in Google Colab.

---

## 1. Quick Start (Google Colab Testing Process)

You can open and execute this notebook directly from GitHub in Google Colab:

1. **Direct GitHub Colab Link**:
   Open: [https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/notebooks/colab_freight_forecasting_benchmark.ipynb](https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/notebooks/colab_freight_forecasting_benchmark.ipynb)

2. **Select Runtime**:
   - **Runtime** -> **Change runtime type** -> **Python 3** (CPU or T4 GPU).

3. **Cell 1 (GitHub Setup & Import)**:
   - Cell 1 automatically clones `https://github.com/SSOHEB/FICOS-Platform.git`, enters the repository root directory (`os.chdir('FICOS-Platform')`), and executes `git pull origin main` to ensure all latest scripts, configs, and `outputs/modeling_dataset.csv` are in place.

4. **Run All**:
   - Run from top to bottom (**Runtime** -> **Run all**). Expected runtime is ~10–15 minutes on standard Colab CPU.

---

## 2. Architecture & Scientific Safeguards

```
Pipeline Flow
[Cell 1-2: Environment Setup & Repository Ingestion]
       │
[Cell 3-4: Dataset Scope & Zero-Leakage Feature Quarantine]
       │  • modeling_dataset.csv (2,581 rows x 482 columns, 2016-2026)
       │  • 41 future/leakage/date columns quarantined; 441 clean predictors
       │
[Cell 5-6: 5-Fold Walk-Forward Validation Tournament Engine]
       │  • 5 Chronological Purged Folds (N ≈ 1,242 test days)
       │  • 8 Asset-Horizon Pairs (Cape, Panamax, Supramax, Handy, KDCI)
       │  • Fold-isolated SelectKBest (K=30) & Candidate Model Tournament
       │  • Empirical Residual Uncertainty Gating (P10/P90)
       │
[Cell 7-8: Final Master Report Reconciliation Audit (72/72 Checks)]
       │  • Compares all computed metrics directly against MASTER_EVALUATION_REPORT.md
       │  • Validates 100% reconciliation (PASS [OK] across all 72 items)
       │
[Cell 9-10: High-Resolution Diagnostic Visualizations]
       │  • ROC Curves (Gated vs Ungated AUC)
       │
[Cell 11: Export & Summary Table]
       │  • Saves outputs/comprehensive_metrics_summary.csv
```

---

## 3. Expected Runtime

- **Total Expected Runtime**: ~20–30 seconds on standard Google Colab CPU.

---

## 4. Deliverables Generated & Extracted

Upon completion, the notebook saves:
1. `outputs/comprehensive_metrics_summary.csv`: Master out-of-sample metrics across all 8 asset-horizon pairs.
2. `outputs/roc_curves.png`: High-resolution ROC curves showing ungated vs gated AUC performance.
3. Live formatted reconciliation table confirming 72/72 checks against `MASTER_EVALUATION_REPORT.md`.

---

## 5. Verification Protocol

After executing **Runtime -> Run all**, the notebook outputs:
```text
==========================================================================================
VERIFICATION SUMMARY: 72/72 checks PASSED
>>> SUCCESS: 100% RECONCILIATION CONFIRMED! <<<
All live Colab outputs match MASTER_EVALUATION_REPORT.md exactly.
==========================================================================================
```

