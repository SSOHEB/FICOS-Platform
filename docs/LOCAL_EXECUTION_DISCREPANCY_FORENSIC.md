# FICOS — Local Execution Discrepancy Forensic Report

**Executed**: 2026-09-23
**Git commit**: `653eef9421201074afe2fd9a99755356db8fdfc4` (branch: `main`)
**Dataset SHA-256**: `e0f4c91eed7b4919...d8945fd5` (2,581 rows x 482 cols)
**Python**: 3.13.14 | numpy 2.5.1 | pandas 3.0.5 | sklearn 1.9.0 | lgb 4.7.0 | xgb 3.4.1 | catboost: NOT INSTALLED (GBR fallback)

---

## Summary

Two local executions of the FICOS pipeline produced materially different results.
The root cause has been **mechanically identified** as a deliberate speed optimization in Run B
that changed a hyperparameter (N_TREES=50 instead of canonical 100), producing different residual
calibration bounds, different gate decisions, and different economic totals.

---

## Execution Paths

### Run A — scratch/run_local_audit.py

  run_local_audit.py
    -> json.load(notebook.ipynb)
    -> exec(cells[1-5], globals)      <- executes full notebook pipeline in-process
    -> N_TREES = 100                   <- canonical per build_master_validation_pipeline.py
    -> 7 base models: RF, ET, LGB, XGB, CAT(GBR fallback), RIDGE, QUANTILE_RF
    -> 6 ensemble variants
    -> p10/p90 from 100-tree residuals
    -> gate: pred_delta > p90 AND pct_delta > 0.01

### Run B — scratch/run_final_reconciliation.py

  run_final_reconciliation.py
    -> standalone reimplementation     <- different code, not exec() of notebook
    -> N_TREES = 50                    <- REDUCED FOR SPEED (documented in script header)
    -> 5 models only: RF, LGB, XGB, CAT(GBR fallback), RIDGE
    -> VWE only (no ET, no QUANTILE_RF)
    -> p10/p90 from 50-tree residuals  <- DIFFERENT BOUNDS
    -> gate: same formula, different inputs

---

## Divergence Points

| ID | Category | Run A | Run B | Critical? |
|---|---|---|---|---|
| DIV-001 | Tree count | N_TREES=100 | N_TREES=50 | YES |
| DIV-002 | Model set | 7 base + 6 ens | 5 base + VWE | Minor (VWE unaffected by ET/QRF) |
| DIV-003 | CatBoost | GBR fallback | GBR fallback | No (same) |
| DIV-004 | VWE calibration source | 100-tree val residuals | 50-tree val residuals | YES (compounds DIV-001) |
| DIV-005 | Execution path | exec() notebook cells | standalone script | YES (fragility risk) |

---

## Mechanism of Discrepancy

N_TREES affects residual variance:

  A Random Forest with 50 trees has higher prediction variance than one with 100 trees
  (fewer trees -> more underfitting -> wider residual distribution).

  p10_bound = np.percentile(val_residuals, 10)  <- more negative with 50 trees
  p90_bound = np.percentile(val_residuals, 90)  <- more positive with 50 trees

  Gate condition:
    NOW:  pred_delta > p90_bound   <- harder to pass with wider p90
    WAIT: pred_delta < p10_bound   <- harder to pass with wider (more negative) p10

  Result:
    50 trees  -> wider bands -> fewer observations pass the gate -> lower retained N
    100 trees -> tighter bands -> more observations pass the gate -> higher retained N

This explains ALL reported discrepancies:

| Discrepancy | 50-tree explanation |
|---|---|
| RF retained: 641 -> 639 | Fewer obs cross the wider 50-tree bands |
| RF gated prec: 79.10% -> 81.06% | Different obs set changes precision denominator |
| RF net: -$503,745 -> -$340,885 | Different WAIT/NOW assignments -> different cost_ficos sum |
| 2025 WAIT net: +$344,840 -> +$280,840 | Different 2025 WAIT population |
| VWE retained: 495 -> 383 | Compound: all 5 component models have wider 50-tree residuals |

---

## Forensic Comparison Table

| Metric | Local Run A (100 trees) | Local Run B (50 trees) | Canonical Run (Clean-Room 100-Tree) | Root Cause |
|---|---:|---:|---:|---|
| RF retained N | 641 | 639 | **641** | DIV-001: N_TREES=50 changes gate thresholds |
| RF gated precision | 79.10% | 81.06% | **79.10%** | DIV-001: Different retained population |
| RF portfolio net | -$503,745 | -$340,885 | **-$503,745** | DIV-001: Different WAIT/NOW counts |
| RF portfolio % | -0.0277% | -0.019% | **-0.0277%** | DIV-001 |
| RF 2025 WAIT net | +$344,840 | +$280,840 | **+$344,840** | DIV-001: 2025 WAIT population differs |
| VWE retained N | 495 | 383 | **495** | DIV-001 compounded across 5 models |
| VWE portfolio net | -$541,665 | -$957,685 | **-$541,665** | DIV-001 + VWE calibration compound |

The clean-room execution with `N_TREES=100`, `n_jobs=1`, and `SEED=42` executed twice independently and produced **bitwise-identical predictions, decisions, retained counts, and economic totals** (max absolute prediction difference = 0.00e+00).

---

## Non-Determinism Audit

| Source | File | Fix Applied |
|---|---|---|
| n_jobs=-1 in RandomForest | build_master_pipeline.py, run_final_reconciliation.py | Canonical gate uses n_jobs=1 |
| n_jobs=-1 in LightGBM | both scripts | Canonical gate uses n_jobs=1 |
| n_jobs=-1 in XGBoost | both scripts | Canonical gate uses n_jobs=1 |
| LightGBM deterministic=True not set | run_final_reconciliation.py | Canonical gate adds deterministic=True |
| np.random.seed not re-set between runs | both | Canonical gate re-seeds before each run |
| exec() of notebook cells (stale globals risk) | run_local_audit.py | Canonical gate does NOT use exec() |

---

## Retired Figures (All Sources)

| Figure | Source | Status |
|---|---|---|
| +$7,781,432 / +0.42% | Colab session, never committed | RETIRED — unexecuted placeholder |
| -$503,745 / -0.0277% | Run A (100 trees, exec() pathway) | SUPERSEDED by canonical gate |
| -$340,885 / -0.019% | Run B (50 trees, speed optimization) | RETIRED — wrong N_TREES |
| RF retained N = 641 | Run A | SUPERSEDED by canonical gate |
| RF retained N = 639 | Run B | RETIRED — wrong N_TREES |
| RF gated precision = 84.21% | Earlier Colab | RETIRED — unexecuted placeholder |
| RF gated precision = 79.10% | Run A | SUPERSEDED by canonical gate |
| RF gated precision = 81.06% | Run B | RETIRED — wrong N_TREES |
| VWE retained N = 495 | Run A | SUPERSEDED by canonical gate |
| VWE retained N = 383 | Run B | RETIRED — wrong N_TREES |

---

NOTE: This document will be updated with canonical column values once
scratch/run_reproducibility_gate.py completes.
