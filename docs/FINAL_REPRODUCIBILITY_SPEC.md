# FICOS — Final Reproducibility Specification

**Canonical Execution Document — SIH26006**
**Git commit**: `653eef9421201074afe2fd9a99755356db8fdfc4`
**Branch**: `main`
**Generated**: 2026-09-23

---

## 1. Dataset Identity

| Property | Value |
|---|---|
| File | `data/modeling_dataset.csv` |
| SHA-256 | `e0f4c91eed7b4919...d8945fd5` |
| Rows | 2,581 |
| Columns | 482 |
| Date range | 2016-01-04 to 2026-09-04 |
| Modified | 2026-09-08T14:35:27Z |

**Verification**:
```bash
python -c "import hashlib; h=hashlib.sha256(); [h.update(c) for c in iter(lambda: open('data/modeling_dataset.csv','rb').read(65536), b'')]; print(h.hexdigest())"
```

---

## 2. Git Commit

```
commit: 653eef9421201074afe2fd9a99755356db8fdfc4
date:   2026-09-23 12:23:49 +0530
msg:    feat(notebook): final authoritative unrounded economic breakdown and certified verdicts
```

**Verification**:
```bash
git rev-parse HEAD
git status --porcelain   # must show clean tree (untracked outputs/ dirs acceptable)
```

---

## 3. Python & Package Versions

| Package | Version |
|---|---|
| Python | 3.13.14 |
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.7.0 |
| xgboost | 3.4.1 |
| catboost | NOT INSTALLED (GradientBoostingRegressor fallback used) |

---

## 4. Random Seeds

| Seed | Usage |
|---|---|
| `SEED = 42` | All model `random_state` parameters |
| `np.random.seed(42)` | Global NumPy seed (set at script top and before each clean-room run) |
| No other random calls | Bootstrap uses `np.random.default_rng(42)` |

---

## 5. Fold Definitions

5 walk-forward folds. Train end dates are exclusive of val/test sets.

| Fold | Test Year | Train End | Val Start | Val End | Test Start | Test End |
|---|---|---|---|---|---|---|
| 1 | 2021 | 2019-12-24 | 2020-01-03 | 2020-12-24 | 2021-01-05 | 2021-12-31 |
| 2 | 2022 | 2020-12-24 | 2021-01-05 | 2021-12-24 | 2022-01-03 | 2022-12-30 |
| 3 | 2023 | 2021-12-24 | 2022-01-03 | 2022-12-23 | 2023-01-03 | 2023-12-29 |
| 4 | 2024 | 2022-12-23 | 2023-01-03 | 2023-12-22 | 2024-01-02 | 2024-12-31 |
| 5 | 2025 | 2023-12-22 | 2024-01-02 | 2024-12-24 | 2025-01-02 | 2025-12-31 |

Vessels: `['panamax', 'supramax', 'handy', 'cape']`
Horizons: `[1, 7, 14, 30]` days

---

## 6. Feature Construction

Features = all columns in `modeling_dataset.csv` EXCEPT:
- `date` (index column)
- columns starting with `target_` (forecast targets)
- columns starting with `dir_` (direction labels)

Total features: 480 (482 cols minus `date` and one rate column per vessel used as `y_base`)

NaN handling: `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`

Target construction (per vessel, per horizon):
```python
y_tr = df_raw.loc[tr_mask, tgt_col].values - df_raw.loc[tr_mask, rate_col].values
```
This predicts the **price delta** (future - current), not the absolute future price.

---

## 7. Model Hyperparameters (Canonical)

| Model | Class | Key Parameters |
|---|---|---|
| RF_STANDARD | `RandomForestRegressor` | `n_estimators=100, max_depth=5, random_state=42, n_jobs=1` |
| LIGHTGBM | `LGBMRegressor` | `n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=1, deterministic=True` |
| XGBOOST | `XGBRegressor` | `n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=1` |
| CATBOOST | `GradientBoostingRegressor` (fallback) | `n_estimators=100, max_depth=5, learning_rate=0.03, random_state=42` |
| RIDGE | `Ridge` | `alpha=100.0` |

**CRITICAL**: `N_TREES=100` is canonical. Any execution using `N_TREES=50` is a speed approximation and its results are NOT authoritative.

---

## 8. Preprocessing

Applied **strictly within each fold** on training data only:

```python
# 1. Standardization
scaler = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)    # fit on train only
X_val_sc = scaler.transform(X_val)       # transform val
X_te_sc  = scaler.transform(X_te)        # transform test

# 2. Feature selection
selector = SelectKBest(f_regression, k=min(30, X_tr_sc.shape[1]))
X_tr_fit  = selector.fit_transform(X_tr_sc, y_tr)
X_val_fit = selector.transform(X_val_sc)
X_te_fit  = selector.transform(X_te_sc)
```

No leakage: scaler and selector fitted on train fold only, applied to val and test.

---

## 9. Uncertainty Calibration (Gate Bounds)

Calibrated **per (horizon, vessel, fold)** from validation fold residuals:

```python
val_residuals = y_val - val_preds_delta
p10_bound = float(np.percentile(val_residuals, 10))
p90_bound = float(np.percentile(val_residuals, 90))
```

These bounds are model-specific and fold-specific. They are NOT global constants.
The bounds depend on `N_TREES` — this is the source of Run A vs Run B discrepancy.

---

## 10. Gating Logic (Actionability)

```python
tau = 0.01  # 1% minimum percentage move threshold
pct_delta = pred_delta / (y_base + 1e-8)

is_buy  = (pred_delta > p90_bound) and (pct_delta >  tau)  # confident price rise
is_wait = (pred_delta < p10_bound) and (pct_delta < -tau)  # confident price fall

decision = 'NOW' if is_buy else ('WAIT' if is_wait else 'FLEXIBLE')
retained = decision in ('NOW', 'WAIT')
```

---

## 11. Economic Formulas

```python
VOYAGE_DURATION = 20.0      # days per voyage
DAILY_IDLE      = 2500.0    # USD per idle day (demurrage rate)

cost_spot  = y_base * VOYAGE_DURATION
cost_wait  = y_true * VOYAGE_DURATION + DAILY_IDLE * 1         # 1 idle day
cost_flex  = avg(y_base, y_true) * VOYAGE_DURATION + DAILY_IDLE * 0.25  # partial idle

cost_ficos = cost_spot  if decision == 'NOW'
           = cost_wait  if decision == 'WAIT'
           = cost_flex  if decision == 'FLEXIBLE'
```

**Portfolio economics** (1D horizon, RF_STANDARD):
```python
net_savings = sum(cost_spot) - sum(cost_ficos)
savings_pct = net_savings / sum(cost_spot) * 100
```

---

## 12. Demurrage Assumptions

| Parameter | Value |
|---|---|
| Voyage duration | 20 days |
| WAIT idle days | 1 day |
| FLEXIBLE idle fraction | 0.25 days |
| Idle day rate | $2,500/day |

---

## 13. Exact Population Definitions

| Subset | Definition |
|---|---|
| Full evaluation | All (model, horizon, vessel, fold) combinations with non-null target |
| Retained | decision in ['NOW', 'WAIT'] |
| 2025 holdout | year == 2025 |
| 2025 WAIT | year == 2025 AND decision == 'WAIT' |

Primary key: `(model, horizon, vessel, date)` — must be unique (verified by assertion).

---

## 14. Production Registry

File: `registry/manifest.json`
SHA-256: `9f74203df75386a5...05dc13c2`
Modified: 2026-09-21T20:21:45Z

| Horizon | Model | Status |
|---|---|---|
| 1D | RandomForestRegressor (4 vessels) | promoted |
| 7D | All models | fallback |
| 14D | (none registered) | fallback via decision_policy.yaml |
| 30D | (none registered) | fallback via decision_policy.yaml |

---

## 15. Deterministic Execution Instructions

To reproduce canonical results:

```bash
# 1. Verify git state
git checkout 653eef9421201074afe2fd9a99755356db8fdfc4
git status  # must show clean (untracked outputs/ acceptable)

# 2. Verify dataset hash
python -c "
import hashlib
h = hashlib.sha256()
with open('data/modeling_dataset.csv','rb') as f:
    for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
print(h.hexdigest())
# Expected: e0f4c91eed7b4919...d8945fd5
"

# 3. Run the reproducibility gate (runs pipeline TWICE, asserts identical results)
python scratch/run_reproducibility_gate.py

# 4. Verify FINAL STATUS: REPRODUCIBLE
```

**Must NOT use**:
- `run_local_audit.py` (exec() fragility, stale globals risk)
- `run_final_reconciliation.py` (N_TREES=50, non-canonical)
- Colab (uncontrolled environment, outputs never committed)

---

## 16. Canonical Results

*Generated from `scratch/run_reproducibility_gate.py` clean-room dual execution.*

| Metric | Canonical Value | Status / Notes |
|---|---:|---|
| RF retained N (1D) | **641** | 13.34% of 4,804 observations retained |
| RF gated precision (1D) | **79.10%** | Directional accuracy on gated population |
| RF net savings (1D) | **-$503,745** | -$503,745.00 vs baseline ($1,818,608,140) |
| RF net savings % (1D) | **-0.0277%** | -0.0277% portfolio impact |
| RF 2025 WAIT net (1D) | **+$344,840** | +$344,840.00 (+1.8805%) on 64 WAIT decisions |
| VWE retained N (1D) | **495** | 10.30% of 4,804 observations retained |
| VWE gated precision (1D) | **84.24%** | Directional accuracy on gated population |
| VWE net savings (1D) | **-$541,665** | -$541,665.00 vs baseline |
| VWE 2025 WAIT net (1D) | **+$329,440** | +$329,440.00 (+1.9888%) on 53 WAIT decisions |

All metrics verified bitwise-identical across two independent clean-room runs.

---

## 17. Final Certification

```
======================================================================
FICOS AUTHORITATIVE CERTIFICATION
======================================================================
STATUS:           AUTHORITATIVE / CERTIFIED ✅
GIT COMMIT:       653eef9421201074afe2fd9a99755356db8fdfc4
DATASET SHA-256:  e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5
CONFIG SSOT:     src/config/canonical_config.py
HYPERPARAMETERS:  N_TREES=100, SEED=42, n_jobs=1

CANONICAL RESULTS SUMMARY:
  - RF Retained N (1D):      641 / 4,804 (13.34%)
  - RF Gated Precision:      79.10%
  - RF Portfolio Net:        -$503,745.00 (-0.0277%)
  - RF 2025 WAIT Net:        +$344,840.00 (+1.8805%)
  - VWE Retained N (1D):     495 / 4,804 (10.30%)
  - VWE Gated Precision:     84.24%
  - VWE Portfolio Net:       -$541,665.00 (-0.0298%)
  - VWE 2025 WAIT Net:       +$329,440.00 (+1.9888%)

OFFICIALLY RETIRED NUMBERS (NON-CANONICAL / UNEXECUTED / 50-TREE):
  - +$7,781,432 / +0.42%      [RETIRED — unexecuted Colab output]
  - -$340,885 / -0.019%       [RETIRED — 50-tree speed approximation]
  - 639 retained / 81.06%     [RETIRED — 50-tree speed approximation]
  - VWE 383 retained / -$957K [RETIRED — 50-tree speed approximation]
======================================================================
```
