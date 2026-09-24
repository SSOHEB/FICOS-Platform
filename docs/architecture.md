# FICOS — Technical Architecture Reference

## 1. Executive Overview

**FICOS (Freight Intelligence & Chartering Optimization System)** is a reproducibility-hardened freight procurement decision platform for bulk ocean shipping.

The core architectural principle:

> **Separation of prediction from decision-making.** The model forecasts freight movement. The uncertainty layer determines actionability. The policy layer determines economic action. The evaluation layer proves whether the complete system creates value.

---

## 2. System-Level Data Flow

```
                 ┌───────────────────┐
                 │     DATA          │   Dataset A: 2,581 daily observations
                 │  (git-ignored)    │   482 features · SHA-256 locked
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ FEATURE ENGINE    │   441 predictors (freight indices,
                 │  ml/features/     │   macro, bunker, cyclone, weather,
                 └─────────┬─────────┘   GDELT geopolitical risk)
                           ↓
                 ┌───────────────────┐
                 │  RF_STANDARD 1D   │   RandomForestRegressor
                 │  100 trees        │   n_estimators=100, seed=42, n_jobs=1
                 │  FROZEN           │   ← canonical_config.py SSOT
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │  FORECAST +       │   MAE: $396.94/MT
                 │  UNCERTAINTY      │   DA: 74.60% (ungated)
                 │  P10/P90 GATING   │   Retained: 641/4,804
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │  EXP-06 POLICY    │   Walk-Forward Locked Policy
                 │  WALK-FORWARD     │   Tuned chronologically
                 │  LOCKED           │   Evaluated out-of-sample
                 └─────────┬─────────┘
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
          BUY_NOW         WAIT      FLEXIBLE_INDEX
             │             │             │
             └─────────────┼─────────────┘
                           ↓
                 ┌───────────────────┐
                 │ ECONOMIC          │   +$7,607,420 net savings
                 │ EVALUATION        │   +0.4183% vs ~$1.818B baseline
                 └───────────────────┘
```

---

## 3. Multi-Horizon Routing

FICOS applies per-horizon routing. Longer-horizon models are only promoted if their directional improvement meets the statistical promotion criterion:

```
                    FICOS
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
      1D             7D            14D / 30D
       │              │              │
  RF_STANDARD   FLEXIBLE_INDEX  FLEXIBLE_INDEX
       │
  EXP-06 Policy
```

For example: the 7D LightGBM candidate had lower MAE, but the directional improvement did not meet the statistical promotion criterion. The architecture explicitly says: **if the longer-horizon signal isn't sufficiently demonstrated, don't pretend it is actionable.**

---

## 4. Configuration & Reproducibility Layer

### `configs/canonical_config.py` — Single Source of Truth

This file is the reproducibility anchor. No script may alter or locally redefine the values below:

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `CANONICAL_N_TREES` | `100` | RF n_estimators — hardware-locked |
| `CANONICAL_SEED` | `42` | random_state — deterministic |
| `CANONICAL_N_JOBS` | `1` | single-thread — eliminates parallelism variance |
| `CANONICAL_DATASET_SHA256` | `e0f4c91...d5` | Dataset identity fingerprint |
| `CANONICAL_DATASET_EXPECTED_ROWS` | `2,581` | Shape assertion |
| `CANONICAL_DATASET_EXPECTED_COLS` | `482` | Shape assertion |

**Why 100 trees is locked:**
Changing 100 → 50 trees altered the residual distribution → P10/P90 bounds → gating → retained observations → economic result. The canonical population shifted from 641 to 639 with materially different economics. This became the most significant engineering finding in the project.

```python
# Enforcement chain
canonical_config.validate_hyperparameters(n_trees, seed, n_jobs)
canonical_config.validate_dataset_provenance()
canonical_config.validate_cost_model(cost_params)
canonical_config.validate_gating_policy(policy_params)
canonical_config.assert_authoritative_certification(...)  # → "AUTHORITATIVE / CERTIFIED ✅"
```

### `configs/` YAML Files

| File | Purpose |
| :--- | :--- |
| `cost_model.yaml` | Bunker prices (VLSFO/MGO), port dues, canal transit fees, demurrage rates |
| `decision_policy.yaml` | BUY_NOW/WAIT/FLEX thresholds, fallback rules, variance penalties |
| `ports.yaml` | Port draft, beam, LOA, DWT constraints |
| `risk_policy.yaml` | Chokepoint & weather risk weights (Red Sea, Suez, monsoon) |
| `vessels.yaml` | Vessel DWT, speed, draft, loading/discharge rates, fuel specs |

---

## 5. ML Layer (`ml/`)

### 5.1 Data Ingestion (`ml/data/`)

`data_loader.py` — handles raw dataset ingestion, missing value treatment, and preprocessing. All transformations (scaling, imputation, feature selection) are fit **strictly on training masks** with no look-ahead.

### 5.2 Feature Engineering (`ml/features/`)

| Module | Predictor Type | Count |
| :--- | :--- | :--- |
| `features.py` | Freight indices, macro, bunker, spreads, lagged targets | ~400 |
| `cyclone_features.py` | Cyclone/typhoon track proximity, intensity signals | ~15 |
| `weather_features.py` | Monsoon seasonality, wave height, wind signals | ~15 |
| `gdelt_features.py` | GDELT geopolitical risk events, chokepoint stress | ~11 |

Total: **441 clean predictors** with fold-isolated `SelectKBest(k=30)` selection.

### 5.3 Forecasting & Uncertainty (`ml/forecasting/`)

**`service.py`** — Forecast orchestration service. Loads the model registry manifest, selects the appropriate model for `(vessel, horizon)`, runs inference, and assembles a `ForecastResult`.

**`uncertainty.py`** — P10/P90 empirical residual uncertainty gating:
```
τ = 0.01  (1% minimum directional move)
q_low  = P10 of out-of-sample validation residuals
q_high = P90 of out-of-sample validation residuals
```
A prediction is only marked **actionable** if the predicted delta exceeds τ and falls within the calibrated residual confidence band.

### 5.4 Models (`ml/models/`)

| Sub-directory | Contents | Status |
| :--- | :--- | :--- |
| `xgboost/` | 40 files: `xgb_<vessel>_<horizon>.json` + `_meta.pkl` | **Production API inference models** — 5 vessels × 4 horizons |
| `ridge/` | *(reserved — empty)* | Experimental track |
| `gru_lstm/` | *(empty — Colab GPU-trained)* | Research track |
| `experimental/` | `ridge/`, `xgboost/` sub-tracks | Challenger experiments |
| `production/` | *(empty — RF_STANDARD via config)* | RF_STANDARD is promoted via `canonical_config.py`, not file artefact |
| `registry/manifest.json` | Model manifest | Status, metrics, provenance per model |

**Vessels covered:** Capesize (`cape`), Panamax (`panamax`), Supramax (`supramax`), Handysize (`handy`), KDCI (`kdci`)
**Horizons:** 1D, 7D, 14D, 30D

### 5.5 Evaluation (`ml/evaluation/`)

| Module | Purpose |
| :--- | :--- |
| `metrics.py` | Directional Accuracy, F1-score, ROC-AUC calculation |
| `validation.py` | Base walk-forward cross-validation framework |
| `walkforward_validation.py` | 5-fold expanding-window validator (2021–2025, ~1,242 OOS days) |
| `decision_backtest.py` | Charter decision backtest engine — counterfactual cost vs spot |

**5-fold structure:**

| Fold | Test Year | Train End | Val End |
| :---: | :---: | :--- | :--- |
| 1 | 2021 | 2019-12-24 | 2020-12-24 |
| 2 | 2022 | 2020-12-24 | 2021-12-24 |
| 3 | 2023 | 2021-12-24 | 2022-12-23 |
| 4 | 2024 | 2022-12-23 | 2023-12-22 |
| 5 | 2025 | 2023-12-22 | 2024-12-24 |

### 5.6 Audit (`ml/audit/`)

| Module | Purpose |
| :--- | :--- |
| `evaluation_audit.py` | Leakage audit — verifies no preprocessing leakage across fold boundaries |
| `gate_aware_comparison.py` | Compares gated vs ungated performance — proves the gating filter adds value |
| `regime_analysis.py` | Market regime detection — identifies structural breaks and regime shifts |

### 5.7 Notebooks (`ml/notebooks/`)

The notebooks are the **research evidence chain**:

#### `02_modeling/` — Core Training & Benchmarking ← Start Here
| Notebook | Purpose |
| :--- | :--- |
| `colab_freight_forecasting_benchmark.ipynb` | Main Colab benchmark — RF/Ridge/XGBoost/LightGBM tournament |
| `forecasting_architecture_benchmark.ipynb` | Architecture-level comparison (GRU/LSTM vs tree models) |
| `experiment_8_final_production_model_challenger.ipynb` | Final challenger: confirms RF_STANDARD as production model |
| `final_model_selection_audit_corrected.ipynb` | Corrected leakage-free model selection audit |
| `model_selection_da_reconciliation.ipynb` | DA metric reconciliation across model runs |
| `phase8_gru_lstm_colab.ipynb` | GRU/LSTM experiment (GPU-accelerated, Colab) |
| `stacked_blend_experiment.ipynb` | Stacked ensemble blending experiment |

#### `03_uncertainty/` — Uncertainty Quantification Lineage
| Notebook | Purpose |
| :--- | :--- |
| `cqr_experiment.ipynb` | Conformal Quantile Regression baseline |
| `experiment_4a_quantile_lightgbm_cqr.ipynb` | Quantile LightGBM + CQR |
| `experiment_4b_grouped_mondrian_cqr.ipynb` | Grouped/Mondrian CQR |
| `experiment_4c_adaptive_weighted_conformal.ipynb` | Adaptive weighted conformal approach |
| `experiment_5_aci_audit.ipynb` | Adaptive Conformal Inference audit |
| `experiment_6_gate_quality.ipynb` | Gate quality evaluation |
| `experiment_7_sharper_base_sparse_groups.ipynb` | Sharper gating with sparse group structure |
| `quantile_boosting_experiment.ipynb` | Quantile gradient boosting experiment |

#### `04_policy/` — Economic Policy Research
| Notebook | Purpose |
| :--- | :--- |
| `experiment_9_economic_charter_decision_backtest.ipynb` | EXP-00 through EXP-06 economic backtest |
| `final_model_family_decision_audit.ipynb` | Model family × decision policy cross-audit |
| `final_model_selection_decision.ipynb` | Final model selection rationale |
| `sih26006_procurement_decision_layer_audit.ipynb` | Procurement layer audit (SIH26006 objective) |

#### `05_proof/` — Authoritative Evidence
| Notebook | Purpose |
| :--- | :--- |
| `01_live_proof_of_outcome.ipynb` | Live, re-runnable proof — reproduces certified result |
| `02_authoritative_research_evidence.ipynb` | Full research evidence trail with reconciliation |

---

## 6. Backend Layer (`backend/`)

### 6.1 API (`backend/api/`)

**`api.py`** — FastAPI REST application. Exposes all HTTP endpoints consumed by the frontend and CLI. Orchestrates the full decision pipeline on each request.

**`recommendation_service.py`** — High-level orchestrator. Calls forecast service → feasibility engine → risk engine → cost model → policy → decision engine in sequence.

**`__main__.py`** — Uvicorn server entry point.

### 6.2 Decision Engine (`backend/decision/`)

| Module | Purpose |
| :--- | :--- |
| `engine.py` | Core chartering decision engine — receives forecast + feasibility + risk + cost inputs, emits `NOW/WAIT/FLEXIBLE` + charter type |
| `explanation.py` | Generates step-by-step human-readable rationale for each decision |
| `procurement_engine.py` | Procurement strategy optimizer — evaluates Spot vs TC vs COA vs Flex under uncertainty |
| `contract_comparison.py` | 4-way charter structure comparator |
| `multi_voyage_planner.py` | Multi-leg voyage scheduler — plans N-voyage sequences vs independent spot |
| `schemas.py` | Pydantic request/response schemas |

### 6.3 Cost Model (`backend/cost/`)

`model.py` — Full voyage cost accounting:
- Charter freight (Spot / Time Charter / COA / Flexible Index)
- Bunker fuel consumption (VLSFO/MGO)
- Port disbursement fees
- Canal transit fees
- Idle/waiting day penalties

`idle_assessment.py` — Opportunity cost computation for idle vessel scenarios.

### 6.4 Operational Feasibility (`backend/operational/`)

Non-negotiable physical gate. Checks:
- Draft limit vs port tidal restrictions
- Length-Over-All (LOA) vs berth length
- Beam vs port width constraints
- DWT vs cargo quantity
- Vessel cargo compatibility

### 6.5 Risk Engine (`backend/risk/`)

`engine.py` — Multi-factor risk scorer producing a 0–100 composite score:
- Red Sea / Suez chokepoint disruption probability
- Monsoon / weather disruption signals
- Port congestion waiting time estimates
- GDELT geopolitical risk index

### 6.6 Policy Layer (`backend/policy/`)

`expected_cost_policy.py` — Risk-adjusted expected cost policy. Selects the optimal charter structure by minimizing risk-weighted expected cost given forecast, uncertainty bounds, and operational constraints.

### 6.7 Scenario Engine (`backend/scenario/`)

`engine.py` — Stress-tests decisions under alternate conditions: fuel price shocks, port congestion spikes, weather scenario replays.

---

## 7. Economic Policy Architecture (EXP-06)

### 7.1 Attribution Discovery

Before EXP-06, the canonical baseline showed `-$503,745` net portfolio result. Observation-level attribution decomposed this:

| Decision | Contribution |
| :--- | ---: |
| BUY_NOW | $0 |
| WAIT | **+$3,725,220** |
| FLEXIBLE_INDEX | **-$4,228,965** |
| **Total** | **-$503,745** |

**Diagnosis:** The signal (WAIT class) was generating real value. The FLEXIBLE_INDEX cost-routing was destroying it. This is a **policy problem, not a model problem.**

**Decision:** Freeze the predictive model. Optimize the policy layer.

### 7.2 Policy Experimentation Path

| Experiment | Description |
| :--- | :--- |
| EXP-00 | Canonical baseline (RF_STANDARD default policy) |
| EXP-01 | FLEX pure spot routing |
| EXP-02 | FLEX bounded premium |
| EXP-03 | WAIT-only gating |
| EXP-04 | Monetary EV gate |
| EXP-05 | Dynamic volatility gate |
| **EXP-06** | **Walk-forward locked policy ← CERTIFIED PRODUCTION** |

### 7.3 EXP-06 Certified Result

| Metric | Value |
| :--- | :--- |
| Policy ID | `EXP-06_WALK_FORWARD_LOCKED` |
| WAIT decisions | 1,509 (31.41% of observations) |
| Gated precision | 80.52% |
| **Net portfolio savings** | **+$7,607,420** |
| **Savings %** | **+0.4183%** on ~$1.818B baseline |
| 2025 locked holdout | **+$944,960** (+5.15% on WAIT decisions) |
| Leakage-free | ✅ |
| Walk-forward validated | ✅ |
| Certification | ✅ `AUTHORITATIVE / CERTIFIED IMPROVEMENT` |

---

## 8. Test Architecture (`tests/`)

```
tests/
├── unit/              # Fast, isolated component tests
│   ├── test_config.py           # canonical_config.py validation functions
│   ├── test_cost_model.py       # Cost arithmetic (Spot/TC/COA/Flex)
│   ├── test_domain.py           # Domain schema validation
│   └── test_feasibility.py      # Physical feasibility gate logic
├── integration/       # Full pipeline component integration
│   ├── test_api.py              # REST endpoint contracts
│   ├── test_decision_engine.py  # Decision engine round-trip
│   ├── test_forecast_service.py # Forecast service integration
│   ├── test_multi_voyage_planner.py
│   ├── test_procurement_decision.py
│   └── test_scenario_engine.py
├── evaluation/        # Statistical validation
│   ├── test_backtest.py         # Backtest correctness
│   └── run_permutation_test.py  # Permutation significance test
└── reproducibility/   # Canonical reproducibility regression
    ├── test_reproducibility_regression.py   # Identical output across runs
    ├── test_economic_policy_reproducibility.py
    └── test_final_report_reconciliation.py  # 72-check reconciliation
```

Run with:
```bash
python -m pytest               # all 38 tests
python -m pytest tests/unit/   # unit only
python -m pytest tests/reproducibility/ -v  # reproducibility checks
```

---

## 9. Proof Infrastructure

Independent evidence layers that collectively constitute the certification:

| Layer | Artefact | Location |
| :--- | :--- | :--- |
| Production implementation | Codebase | `backend/`, `ml/` |
| Canonical configuration | SSOT | `configs/canonical_config.py` |
| Authoritative result payload | JSON | `outputs/authoritative/` |
| Observation-level results | JSON / CSV | `reports/` |
| Automated reconciliation | Tests | `tests/reproducibility/` |
| Reproducibility tests | Pytest | `tests/reproducibility/` |
| Certification report | Markdown | `docs/ECONOMIC_POLICY_CERTIFICATION_REPORT.md` |
| Proof notebooks | Jupyter | `ml/notebooks/05_proof/` |

---

## 10. What Is and Is Not Finalized

### ✅ Finalized

| Dimension | Value |
| :--- | :--- |
| 1D predictive model | RF_STANDARD, 100 trees, seed 42, n_jobs=1 |
| Horizon routing | 1D→ML, 7D→FLEXIBLE_INDEX, 14D→FLEXIBLE_INDEX, 30D→FLEXIBLE_INDEX |
| Economic policy | EXP-06 Walk-Forward Locked |
| Economic result | +$7,607,420 net savings (+0.4183%) |
| 2025 holdout | +$944,960 (+5.15% on WAIT decisions) |
| Provenance | Dataset SHA-256, canonical config, Git provenance, reproducibility gate |
| Repository | Modular domain-driven structure, research/evidence separation |

### 🔲 Extensions (Not Yet Finalized)

These are planned extensions, not part of the currently certified core:

- Multi-voyage portfolio optimizer (joint contract optimization)
- CVaR risk term integration
- Fully validated 7D/14D/30D ML models (currently FLEXIBLE_INDEX fallback)
- Live data ingestion pipeline (AIS, GDELT, weather APIs)

---

## 11. CLI & API Reference

### CLI
```bash
# Run decision evaluation
python scripts/run_cli.py evaluate \
  --asset PANAMAX_1D \
  --quantity 75000 \
  --origin TUBARAO \
  --dest QINGDAO

# Start API server
python scripts/run_api.py

# Run canonical verification
python scripts/run_verification.py

# Generate diagnostic plots
python scripts/generate_plots.py
```

### Test Suite
```bash
python -m pytest                        # all 38 tests
python -m pytest -v                     # verbose
python -m pytest tests/reproducibility/ # reproducibility only
```
