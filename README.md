# FICOS — Freight Index Decision & Optimization System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/notebooks/colab_freight_forecasting_benchmark.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-audited%20%26%20validated-brightgreen.svg)](reports/MASTER_EVALUATION_REPORT.md)
[![Reconciliation](https://img.shields.io/badge/reconciliation-72%2F72%20PASS%20(100%25)-success.svg)](reports/MASTER_EVALUATION_REPORT.md)

Production-grade dry-bulk freight rate forecasting, uncertainty gating, and automated chartering decision engine for Capesize, Panamax, Supramax, and Handysize shipping asset classes.

---

## 📁 Modular Repository Architecture

```text
FICOS-Platform/
├── configs/                   # Central configuration YAML & parameters
│   ├── config.yaml            # Environment and dataset path mappings
│   └── threshold_config.yaml  # Decision engine threshold parameters
├── data/                      # Local datasets (Excluded from git via .gitignore)
│   ├── dataset_a_final_clean.csv
│   ├── DATASET_B_FINAL_FIXED (1).xlsx
│   ├── DATASET_C_FINAL.xlsx
│   └── modeling_dataset.csv
├── images/                    # Master evaluation plots & diagnostic curves
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── confusion_matrices.png
│   ├── residual_distribution.png
│   ├── feature_importance.png
│   ├── fold_variance.png
│   ├── metrics_comparison.png
│   └── regression_scatter.png
├── notebooks/                 # Google Colab benchmarks & interactive notebooks
│   ├── colab_freight_forecasting_benchmark.ipynb  (Official 72/72 Colab Benchmark)
│   └── phase8_gru_lstm_colab.ipynb
├── outputs/                   # Audited quantitative results
│   └── comprehensive_metrics_summary.csv
├── reports/                   # Audit reports, schemas, and master evaluation report
│   ├── MASTER_EVALUATION_REPORT.md
│   ├── LEAKAGE_AUDIT_CHECKLIST.md
│   └── RESULTS_SCHEMA.md
├── src/                       # Modular Python package
│   ├── __init__.py            # Package facade & clean exports
│   ├── data_loader.py         # Multi-source dataset loaders (A, B, C)
│   ├── decision_engine.py     # P10/P90 residual uncertainty gate & chartering logic
│   ├── evaluation.py          # Metric calculations (sMAPE, DA, R2, MAE)
│   ├── feasibility_engine.py  # Port & berth feasibility constraints
│   ├── features.py            # Feature engineering & zero-leakage quarantine
│   ├── optimization_pipeline.py
│   ├── shock_response.py
│   ├── validation.py
│   └── walkforward_validation.py
├── tests/                     # Test suite & validation runners
│   └── test_final_report_reconciliation.py
├── .gitignore                 # Strict rules: ignores data/, *.xlsx, *.csv, *.pkl, *.json logs
├── LICENSE
├── README.md
├── README_COLAB.md            # Detailed Google Colab user guide
└── requirements_colab.txt
```

---

## 🔒 Security & Data Privacy Policy

> **Strict Data Protection:** In accordance with production best practices, raw market fixtures, sensitive vessel logs, and large dataset artifacts (`*.csv`, `*.xlsx`, `*.parquet`, `*.pkl`) are **strictly excluded via `.gitignore`** and are never pushed to public repositories.

---

## 🚀 Quick Start

### 1. Google Colab (Recommended)
Open and run the official 5-fold walk-forward validation and report reconciliation benchmark directly in Google Colab:  
👉 **[Open In Colab](https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/notebooks/colab_freight_forecasting_benchmark.ipynb)**  
Select **Runtime → Run all**. Runtime is ~20–30 seconds.

### 2. Local Verification Test
Run the automated 72-point verification test locally:
```bash
python tests/test_final_report_reconciliation.py
```
Outputs:
```text
==========================================================================================
VERIFICATION SUMMARY: 72/72 checks PASSED
>>> SUCCESS: 100% RECONCILIATION CONFIRMED! <<<
All live outputs match MASTER_EVALUATION_REPORT.md exactly.
==========================================================================================
```

---

## 📊 Summary of Out-of-Sample Performance (5 Purged Folds)

| Asset & Horizon | Classification | Gated Acc (%) | Coverage (%) | UP Precision (%) | Recall (%) | F1 (%) | ROC-AUC | Gated Signals ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PANAMAX 1D** | 🟢 **PRIMARY PROMOTE** | **91.1%** | 17.2% | **91.7%** | **90.9%** | **91.3%** | **0.924** | 214 |
| **SUPRAMAX 1D** | 🟢 **PRIMARY PROMOTE** | **85.0%** | 16.1% | **76.0%** | **91.2%** | **83.0%** | **0.805** | 200 |
| **HANDY 1D** | 🟢 **PRIMARY PROMOTE** | **79.2%** | 11.6% | **72.2%** | **87.7%** | **79.2%** | **0.750** | 144 |
| **CAPE 1D** | 🟢 **PRIMARY PROMOTE** | **71.3%** | 14.3% | **58.3%** | **48.3%** | **52.8%** | **0.711** | 174 |
| **SUPRAMAX 7D** | 🟢 **SECONDARY PROMOTE** | **63.8%** | 19.0% | **66.7%** | **41.1%** | **50.9%** | **0.744** | 235 |
| **HANDY 7D** | 🟢 **SECONDARY PROMOTE** | **58.6%** | 19.3% | **52.4%** | **62.9%** | **57.1%** | **0.593** | 239 |
| *SUPRAMAX 14D* | 🔴 **EXCLUDE** | **49.1%** | 40.9% | 49.1% | 11.0% | 17.9% | 0.564 | 503 |
| *KDCI 7D* | 🔴 **EXCLUDE** | **76.7%** | 12.1% | 59.5% | 52.4% | 55.7% | **0.801** | 150 |

For full scientific documentation, see [MASTER_EVALUATION_REPORT.md](reports/MASTER_EVALUATION_REPORT.md).
