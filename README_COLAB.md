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

## 3. Architecture & Strict Scientific Safeguards

```
Pipeline Flow
[Cell 1-3: Setup & Data Loading]
       │
[Cell 4-5: Data & Leakage Audits] ──(Fatal Assertion if Violation)──> STOP
       │
[Cell 6-8: Feature & Target Pipeline + Strict Date Splits]
       │  • Train:        date < 2023-01-01 (1,699 rows)
       │  • Dev/Val:      2023-01-01 <= date < 2025-01-01 (480 rows)
       │  • Locked Test:  date >= 2025-01-01 (402 rows)
       │
[Cell 9-12: Models Architecture (Ridge, ElasticNet, Trees, GBDT, Classifiers)]
       │
[Cell 13-14: Walk-Forward Validation & Pre-Test Tournament]
       │  • 5-Fold expanding walk-forward inside pre-2025 development period
       │  • Select & freeze finalist per pair on validation metrics ONLY
       │
[Cell 15: ===== TEST SET LOCKED =====] ──(Irreversible state barrier)
       │
[Cell 16-19: Locked-Test Evaluation, Bootstrap CIs, Stability, Regimes]
       │
[Cell 20-22: Consolidated Results Table, Diagnostic Plots, CSV/JSON Export]
```

---

## 4. Expected Runtime

- **Cells 1–8** (Setup, Audits, Features, Targets): ~30 seconds
- **Cells 9–12** (Model definitions & Baselines): ~1 minute
- **Cell 13–14** (Walk-Forward Validation Tournament across Anchor Pairs): ~5–8 minutes on CPU (faster on T4)
- **Cell 15–19** (Locked Test Evaluation, 1,000 Bootstrap Resamples, Regime Analysis): ~3–5 minutes
- **Cell 20–22** (Summary Tables, Visualizations, Export): ~30 seconds
- **Total Expected Runtime**: ~10–15 minutes.

---

## 5. Deliverables Generated & Extracted

Upon completion, the notebook saves:
1. `outputs/colab_benchmark_results.csv`: Master quantitative comparison across all 12 pairs.
2. `outputs/colab_benchmark_summary.json`: Complete machine-readable experiment state and parameters.
3. `outputs/plots/benchmark_actual_vs_predicted.png`: Out-of-sample forecast trajectories.
4. `outputs/plots/benchmark_regularization_stability.png`: Multi-panel alpha/C sensitivity curves.
5. `outputs/plots/benchmark_confusion_matrices.png`: Directional classification heatmaps.

---

## 6. What Output to Send Back After Running

After running the notebook, copy and send back:
1. The **Final Report text output printed by Cell 22** (containing the data, model, target, baseline comparisons, and final verdict per pair).
2. The **Master Consolidated Results Table from Cell 20** (printed in markdown / CSV format).
3. Any unexpected assertion failures or warnings produced by Cell 4 (Data Audit) or Cell 5 (Leakage Audit).
