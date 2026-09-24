# FICOS — Freight Intelligence & Chartering Optimization System

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/ml/notebooks/02_modeling/colab_freight_forecasting_benchmark.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: RF_STANDARD](https://img.shields.io/badge/model-RF__STANDARD%20(100%20trees)-orange.svg)](configs/canonical_config.py)
[![Policy: EXP-06](https://img.shields.io/badge/policy-EXP--06%20Walk--Forward-green.svg)](outputs/authoritative/authoritative_policy_results.json)
[![Certified Savings](https://img.shields.io/badge/certified%20savings-%2B%247.6M-brightgreen.svg)](outputs/authoritative/authoritative_policy_results.json)
[![Tests](https://img.shields.io/badge/tests-38%20passed-success.svg)](tests/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-32%2F32%20%7C%204%2F4-success.svg)](tests/reproducibility/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Reproducibility-Hardened Freight Forecasting → Uncertainty → Economic-Policy → Procurement-Decision System**

*RF_STANDARD + EXP-06 certified 1D production path · FLEXIBLE_INDEX fallback for horizons without demonstrated predictive superiority · +$7,607,420 certified net savings vs spot baseline.*

</div>

---

## 🏛️ System Architecture

FICOS is a **freight procurement decision system**. Its architectural principle is the strict separation of prediction from decision-making:

```
Market / Operational Data
          ↓
  Feature Engineering
          ↓
  Freight Forecasting          ← RF_STANDARD (1D) | FLEXIBLE_INDEX (7D/14D/30D)
          ↓
Uncertainty / Confidence       ← P10/P90 Empirical Residual Gating
          ↓
   Decision Policy             ← EXP-06 Walk-Forward Locked Policy
          ↓
BUY_NOW / WAIT / FLEXIBLE_INDEX
          ↓
  Economic Evaluation          ← Observation-level attribution · Walk-forward validated
```

> **The model forecasts. The uncertainty layer determines actionability. The policy layer determines economic action. The evaluation layer proves whether the complete system creates value.**

### Multi-Horizon Routing

FICOS does **not** force the same model onto every horizon. Longer-horizon models are only promoted if their directional improvement meets the statistical promotion criterion:

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
       │
  BUY_NOW / WAIT / FLEXIBLE_INDEX
```

---

## ⚡ Current Certified Production State

### Production Predictive Model (Frozen)

| Parameter | Value |
| :--- | :--- |
| **Algorithm** | `RandomForestRegressor` — `RF_STANDARD` |
| **n_estimators** | `100` ← canonical, hardware-locked |
| **random_state** | `42` |
| **n_jobs** | `1` (deterministic single-thread) |
| **MAE** | **$396.94 / MT** |
| **Directional Accuracy** | **74.60%** (ungated full population) |
| **Gated Precision** | **79.10%** (on actionable gate) |
| **Retained (Actionable) N** | **641 / 4,804** observations |
| **Status** | 🔒 **FROZEN — do not alter** |

> **Why frozen?** A seemingly harmless change from 100 → 50 trees altered the residual distribution → P10/P90 bounds → gating → retained observations → economic result. The canonical population shifted from 641 to 639, with materially different economics. The configuration is hardened as SSOT in [`configs/canonical_config.py`](configs/canonical_config.py).

### Certified Economic Policy (EXP-06)

| Metric | Value |
| :--- | :--- |
| **Policy** | EXP-06 — Walk-Forward Locked |
| **Total Observations** | 4,804 |
| **WAIT Decisions** | 1,509 (31.41% coverage) |
| **Gated Precision** | 80.52% |
| **Net Portfolio Savings** | **+$7,607,420** vs spot baseline |
| **Savings %** | **+0.4183%** on ~$1.818B baseline portfolio |
| **2025 Locked Holdout** | **+$944,960** (+5.15% on WAIT decisions) |
| **Walk-Forward Validated** | ✅ Leakage-free |
| **Certification Status** | ✅ **AUTHORITATIVE / CERTIFIED IMPROVEMENT** |

> **Canonical baseline context:** The RF_STANDARD canonical model with the default policy layer produced `-$503,745`. EXP-06 supersedes this by freezing the predictive model and optimizing only the policy layer — the correct architectural diagnosis when the signal is real but the cost-routing is destroying value.

### Dataset Provenance

| Property | Value |
| :--- | :--- |
| **SHA-256** | `e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5` |
| **Rows** | 2,581 |
| **Columns** | 482 |
| **Validation Folds** | 5 (expanding-window, 2021–2025) |

---

## 🔬 Uncertainty & Gating Architecture

The uncertainty layer evolved through multiple stages before reaching the canonical residual-based gating approach:

```
Point prediction
      ↓
Quantile models
      ↓
CQR (Conformal Quantile Regression)
      ↓
Grouped / Mondrian approaches
      ↓
Adaptive conformal approaches
      ↓
Residual-based gating           ← current canonical approach
      ↓
Canonical RF gating (P10/P90)
      ↓
Economic policy gating (EXP-06)
```

| Parameter | Value |
| :--- | :--- |
| `tau` | 0.01 (1% directional move threshold) |
| `q_low` | P10 empirical residual bound |
| `q_high` | P90 empirical residual bound |

The system only treats a prediction as actionable when the predicted movement clears these empirical thresholds. High-conviction gated observations jump from 74.60% → 79.10%+ directional accuracy.

---

## 📊 Economic Attribution Decomposition

The canonical RF baseline decomposition revealed the source of the -$503,745 result and guided EXP-06 design:

| Decision Class | Portfolio Contribution |
| :--- | ---: |
| BUY_NOW | $0 |
| WAIT | **+$3,725,220** |
| FLEXIBLE_INDEX | **-$4,228,965** |
| **Total (Canonical Baseline)** | **-$503,745** |

This established that the predictive signal (WAIT class) was generating real value, but the FLEXIBLE_INDEX cost-routing was destroying it. **The fix was a policy problem, not a model problem.**

---

## 📈 Experiment Timeline: Policy Layer

| Experiment | Description | Outcome |
| :--- | :--- | :--- |
| EXP-00 | Canonical baseline (RF_STANDARD default policy) | -$503,745 |
| EXP-01 | FLEX pure spot | Evaluated |
| EXP-02 | FLEX bounded premium | Evaluated |
| EXP-03 | WAIT-only gating | Evaluated |
| EXP-04 | Monetary EV gate | Evaluated |
| EXP-05 | Dynamic volatility gate | Evaluated |
| **EXP-06** | **Walk-forward locked policy** | **✅ +$7,607,420 CERTIFIED** |

EXP-06 was tuned chronologically and evaluated out-of-sample. This is not a backfit result.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements/requirements.txt
```
For Google Colab:
```bash
pip install -r requirements/requirements_colab.txt
```

### 2. Configure Environment
```bash
cp .env.example .env   # fill in your API keys
```

### 3. Run Benchmark in Colab (Recommended)
👉 **[Open In Google Colab](https://colab.research.google.com/github/SSOHEB/FICOS-Platform/blob/main/ml/notebooks/02_modeling/colab_freight_forecasting_benchmark.ipynb)**
Select **Runtime → Run all** (`Ctrl+F9`). Runtime: ~20–30 seconds.

### 4. Start the API Server
```bash
python scripts/run_api.py
```

### 5. Run the CLI Decision Evaluator
```bash
python scripts/run_cli.py evaluate --asset PANAMAX_1D --quantity 75000 --origin Tubarao --dest Qingdao
```

Example output:
```
=================================================================
 FICOS FREIGHT DECISION RECOMMENDATION
=================================================================
Recommendation ID : REC-A5B07063
Asset / Cargo     : PANAMAX_1D (75,000 MT)
Route             : Tubarao -> Qingdao
Vessel Class      : Panamax
-----------------------------------------------------------------
RECOMMENDED ACTION: WAIT
Expected Total Cost: $769,340.70   |   Cost per MT: $10.26/MT
Freight Forecast:    $25.00/MT     |   P10: $18.75  P90: $31.25
Physical Feasibility: PASSED [OK]
Overall Risk Score:   10.0/100 (LOW Weather / LOW Disruption)
=================================================================
```

### 6. Run Automated Test Suite
```bash
python -m pytest
```
*Output: `38 passed` — unit + integration + evaluation + reproducibility.*

---

## 📁 Repository Structure

> **Read order:** `frontend/` → `backend/` + configs & scripts → `ml/` + outputs → `tests/` → support files.

```
FICOS-Platform/
│
│  ── PRESENTATION LAYER ─────────────────────────────────────────────────────
│
├── frontend/                          ← React/TypeScript Web Dashboard
│   ├── src/
│   │   ├── App.tsx                    # Root router & layout
│   │   ├── Dashboard.tsx              # Main analytics dashboard
│   │   ├── VesselIntelligence.tsx     # Vessel-level market intelligence
│   │   ├── IdleIntelligence.tsx       # Idle vessel cost analysis
│   │   ├── Services.tsx               # Services & API status page
│   │   ├── apiClient.ts               # Typed HTTP client → backend API
│   │   └── dashboardData.ts           # Dashboard data & chart configs
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── netlify.toml                   # Netlify deployment config
│
│  ── APPLICATION LAYER ──────────────────────────────────────────────────────
│
├── backend/                           ← Python Application & Business Logic
│   ├── api/
│   │   ├── api.py                     # FastAPI REST endpoints
│   │   ├── recommendation_service.py  # Decision pipeline orchestrator
│   │   └── __main__.py                # Uvicorn server entry point
│   ├── config/
│   │   ├── settings.py                # .env loader & app settings
│   │   └── canonical_config.py        # Pointer to root canonical SSOT
│   ├── cost/
│   │   ├── model.py                   # Spot / TC / COA / Flexible Index cost model
│   │   └── idle_assessment.py         # Idle vessel opportunity cost engine
│   ├── decision/
│   │   ├── engine.py                  # Core chartering decision engine
│   │   ├── explanation.py             # Human-readable rationale generator
│   │   ├── procurement_engine.py      # Procurement strategy optimizer
│   │   ├── contract_comparison.py     # 4-way charter contract evaluator
│   │   ├── multi_voyage_planner.py    # Multi-leg voyage planner
│   │   └── schemas.py                 # Pydantic request/response schemas
│   ├── domain/
│   │   └── schemas.py                 # Typed domain models (vessel, port, cargo)
│   ├── operational/
│   │   ├── feasibility_engine.py      # Physical feasibility checker
│   │   ├── port_repository.py         # Port draft/beam/LOA constraint data
│   │   └── vessel_repository.py       # Vessel spec & capability data
│   ├── policy/
│   │   └── expected_cost_policy.py    # Risk-adjusted expected cost policy
│   ├── risk/
│   │   └── engine.py                  # Multi-factor risk scorer (0–100)
│   └── scenario/
│       └── engine.py                  # Scenario simulation engine
│
├── configs/                           ← ★ SSOT — All Production Configurations
│   ├── canonical_config.py            # Locked: n_trees=100, seed=42, SHA-256
│   ├── cost_model.yaml                # Bunker prices, port dues, canal fees
│   ├── decision_policy.yaml           # BUY_NOW/WAIT/FLEX thresholds
│   ├── ports.yaml                     # Port draft, beam, LOA constraints
│   ├── risk_policy.yaml               # Chokepoint & weather risk weights
│   └── vessels.yaml                   # Vessel DWT, speed, draft limits
│
├── scripts/                           ← Backend Entry Points
│   ├── run_api.py                     # Launch FastAPI backend server
│   ├── run_cli.py                     # Launch CLI decision evaluator
│   ├── run_verification.py            # Run canonical verification checks
│   └── generate_plots.py              # Generate diagnostic plots
│
│  ── MACHINE LEARNING LAYER ─────────────────────────────────────────────────
│
├── ml/                                ← Machine Learning — Research & Production
│   ├── data/
│   │   └── data_loader.py             # Dataset ingestion & preprocessing
│   ├── features/
│   │   ├── features.py                # Core feature engineering (441 predictors)
│   │   ├── cyclone_features.py        # Cyclone/typhoon signals
│   │   ├── weather_features.py        # Monsoon & seasonal weather signals
│   │   └── gdelt_features.py          # GDELT geopolitical risk signals
│   ├── forecasting/
│   │   ├── service.py                 # Forecast orchestration service
│   │   └── uncertainty.py             # P10/P90 empirical residual gating
│   ├── models/
│   │   ├── xgboost/                   # 40 XGBoost API inference models
│   │   │   └── xgb_<vessel>_<horizon>.json + _meta.pkl
│   │   ├── ridge/                     # Ridge weights (reserved)
│   │   ├── gru_lstm/                  # GRU/LSTM research checkpoints
│   │   ├── experimental/              # Challenger experiments
│   │   ├── production/                # Promoted production snapshots
│   │   └── registry/manifest.json     # Model version & status manifest
│   ├── evaluation/
│   │   ├── metrics.py                 # DA, F1, AUC metric calculations
│   │   ├── validation.py              # Walk-forward cross-validation
│   │   ├── walkforward_validation.py  # 5-fold expanding-window validator
│   │   └── decision_backtest.py       # Charter decision backtest engine
│   ├── audit/
│   │   ├── evaluation_audit.py        # Leakage audit framework
│   │   ├── gate_aware_comparison.py   # Gate-aware performance comparison
│   │   └── regime_analysis.py         # Market regime analysis
│   ├── registry/
│   │   ├── registry.py                # Manifest reader & version manager
│   │   └── manifest.json
│   └── notebooks/                     # Research evidence chain
│       ├── 01_data/                   # Data exploration
│       ├── 02_modeling/               # ← Start here: model tournament & selection
│       │   ├── colab_freight_forecasting_benchmark.ipynb
│       │   ├── forecasting_architecture_benchmark.ipynb
│       │   ├── experiment_8_final_production_model_challenger.ipynb
│       │   └── (+ 4 more modeling notebooks)
│       ├── 03_uncertainty/            # CQR → Mondrian → ACI → Residual gating
│       │   └── (8 uncertainty quantification notebooks)
│       ├── 04_policy/                 # EXP-00 through EXP-06 policy research
│       │   └── (4 policy & decision notebooks)
│       └── 05_proof/                  # Authoritative evidence — final numbers
│           ├── 01_live_proof_of_outcome.ipynb
│           └── 02_authoritative_research_evidence.ipynb
│
├── outputs/                           ← Authoritative Result Payloads
│   ├── authoritative/
│   │   ├── authoritative_canonical_results.json  # RF_STANDARD certified metrics
│   │   └── authoritative_policy_results.json     # EXP-06 certified economics
│   ├── backtests/
│   ├── figures/
│   ├── experiment_6_gate_quality/
│   ├── experiment_7_sharper_base_sparse_groups/
│   └── experiment_9_economic_backtest/
│
│  ── QUALITY & SUPPORT ──────────────────────────────────────────────────────
│
├── tests/                             ← Automated Test Suite (38 tests)
│   ├── unit/                          # Config, cost, domain, feasibility (4)
│   ├── integration/                   # API, decision, forecast, scenario (6)
│   ├── evaluation/                    # Backtest & permutation significance (2)
│   └── reproducibility/               # Canonical reproducibility regression (3)
│
├── requirements/                      ← Dependency Management
│   ├── requirements.txt               # Local development & production
│   └── requirements_colab.txt         # Google Colab (lighter)
│
├── docs/                              ← Scientific & Technical Documentation
│   ├── architecture.md                # ← Full technical architecture spec
│   ├── model_selection.md
│   ├── ECONOMIC_POLICY_CERTIFICATION_REPORT.md
│   ├── ECONOMIC_POLICY_ATTRIBUTION_AUDIT.md
│   ├── FICOS_COMPLETE_PROJECT_EVOLUTION_REPORT.md
│   ├── FINAL_REPRODUCIBILITY_SPEC.md
│   └── LOCAL_EXECUTION_DISCREPANCY_FORENSIC.md
│
├── reports/                           ← Audit Reports & Result Artefacts
│   ├── MASTER_EVALUATION_REPORT.md    # 72-point reconciliation benchmark
│   ├── FINAL_BACKEND_AUDIT_REPORT.md
│   ├── FINAL_MODEL_SELECTION_REPORT.md
│   ├── LEAKAGE_AUDIT_CHECKLIST.md
│   ├── RESULTS_SCHEMA.md
│   └── final_model_family_*.csv       # Per-dimension scoring CSVs
│
├── data/                              ← Raw & processed data (git-ignored)
├── images/                            ← Publication-grade plots (300 DPI)
├── registry/manifest.json             ← Shared root model registry
├── docs-site/                         ← Documentation site source
├── archive/                           ← Historical research lineage
│
├── main.py                            # Top-level entry point
├── .env.example                       # Environment variable template
├── pytest.ini                         # pytest configuration
└── .gitignore                         # Data privacy rules
```

---


## 🧩 Component Breakdown

### 🖥️ Frontend (`frontend/`)
A **React + TypeScript + Vite** single-page application providing the analyst-facing UI.

| Component | Purpose |
| :--- | :--- |
| `Dashboard.tsx` | Main operational dashboard — freight rate charts, market overview, KPIs |
| `VesselIntelligence.tsx` | Vessel-level deep-dive — route performance, rate comparisons, suitability |
| `IdleIntelligence.tsx` | Idle vessel cost analytics — opportunity cost calculations, layup scenarios |
| `Services.tsx` | Backend service health, API status, system diagnostics |
| `apiClient.ts` | Typed HTTP client connecting UI → backend REST API |
| `dashboardData.ts` | Dashboard seed data and static chart configurations |

---

### ⚙️ Backend (`backend/`)
The **Python application layer** — all business logic, API endpoints, decision orchestration.

| Module | Key Files | Purpose |
| :--- | :--- | :--- |
| `api/` | `api.py`, `recommendation_service.py` | FastAPI REST application — all HTTP endpoints, decision pipeline orchestrator |
| `config/` | `settings.py`, `canonical_config.py` | `.env` loader + canonical reproducibility pointer |
| `cost/` | `model.py`, `idle_assessment.py` | Four-structure maritime cost model: Spot Voyage, TC, COA, Flexible Index |
| `decision/` | `engine.py`, `procurement_engine.py`, `contract_comparison.py`, `multi_voyage_planner.py` | Core chartering decision engine; procurement optimizer; 4-way contract comparator; multi-leg voyage planner |
| `domain/` | `schemas.py` | Strongly-typed Pydantic domain models (vessel, port, cargo, recommendation) |
| `operational/` | `feasibility_engine.py`, `port_repository.py`, `vessel_repository.py` | Physical feasibility checker — draft, LOA, beam, DWT vs port limits |
| `policy/` | `expected_cost_policy.py` | Risk-adjusted expected cost policy — selects optimal charter strategy |
| `risk/` | `engine.py` | Multi-factor risk scorer — geopolitical events, chokepoints, weather, port delays |
| `scenario/` | `engine.py` | Scenario simulation — stress-tests decisions under alternate market conditions |

---

### 🤖 ML (`ml/`)
All **machine learning** code: data ingestion, feature engineering, forecasting, uncertainty quantification, evaluation, and the research evidence trail.

| Module | Key Files | Purpose |
| :--- | :--- | :--- |
| `data/` | `data_loader.py` | Dataset loader and preprocessing pipeline — missing values, scaling |
| `features/` | `features.py`, `cyclone_features.py`, `weather_features.py`, `gdelt_features.py` | 441-predictor feature engineering: freight indices, macro, bunker, cyclone/weather, GDELT geopolitical risk |
| `forecasting/` | `service.py`, `uncertainty.py` | Forecast orchestration service + P10/P90 empirical residual gating |
| `models/xgboost/` | 40 × `.json` + `.pkl` | **40 XGBoost models** — 5 vessels (Cape/Panamax/Supramax/Handy/KDCI) × 4 horizons (1D/7D/14D/30D); these are the **API inference models** |
| `models/ridge/` | *(reserved)* | Ridge regression weights (experimental track) |
| `models/gru_lstm/` | *(empty — Colab-trained)* | GRU/LSTM deep learning architecture (research track, trained on Colab GPU) |
| `models/experimental/` | ridge/, xgboost/ sub-tracks | Challenger model experiment artefacts |
| `models/production/` | *(empty — SSOT in canonical_config)* | Promoted production snapshots (RF_STANDARD promoted via configuration) |
| `models/registry/` | `manifest.json` | Model version & status manifest |
| `evaluation/` | `metrics.py`, `validation.py`, `walkforward_validation.py`, `decision_backtest.py` | Walk-forward validation, DA/F1/AUC metrics, charter decision backtest |
| `audit/` | `evaluation_audit.py`, `gate_aware_comparison.py`, `regime_analysis.py` | Leakage audit framework, gate-aware performance comparison, regime analysis |
| `registry/` | `registry.py`, `manifest.json` | Manifest reader & version manager |

#### Notebooks (`ml/notebooks/`)

| Section | Notebooks | Purpose |
| :--- | :--- | :--- |
| `02_modeling/` (7 notebooks) | `colab_freight_forecasting_benchmark.ipynb`, `forecasting_architecture_benchmark.ipynb`, `experiment_8_final_production_model_challenger.ipynb`, + 4 more | Core training & benchmarking — **start here**; RF vs Ridge vs XGBoost vs LightGBM vs GRU/LSTM tournaments |
| `03_uncertainty/` (8 notebooks) | `cqr_experiment.ipynb`, `experiment_4a/4b/4c`, `experiment_5_aci_audit.ipynb`, `experiment_6/7` | Full uncertainty quantification lineage: CQR → Mondrian → ACI → Adaptive → Residual gating |
| `04_policy/` (4 notebooks) | `experiment_9_economic_charter_decision_backtest.ipynb`, `final_model_family_decision_audit.ipynb`, + 2 more | Economic policy & charter decision research: EXP-00 through EXP-06 |
| `05_proof/` (2 notebooks) | `01_live_proof_of_outcome.ipynb`, `02_authoritative_research_evidence.ipynb` | Live reproducible proof & authoritative research evidence trail |

---

### 🧪 Tests (`tests/`)

| Suite | Tests | Purpose |
| :--- | :---: | :--- |
| `unit/` | 4 | Config validation, cost arithmetic, domain schema checks, feasibility gate logic |
| `integration/` | 6 | Full API contract, decision engine, forecast service, scenario, multi-voyage planner, procurement decision |
| `evaluation/` | 2 | Backtest correctness, permutation significance test |
| `reproducibility/` | 3 | Canonical reproducibility regression — verifies identical predictions/decisions/economics |
| **Total** | **38** | **All passing** |

---

### 📦 Requirements (`requirements/`)

| File | Used For |
| :--- | :--- |
| `requirements.txt` | Local development, production server |
| `requirements_colab.txt` | Google Colab (lighter; pre-installed libs stripped) |

---

### 🔧 Configs (`configs/`)

| File | Purpose |
| :--- | :--- |
| `canonical_config.py` | **SSOT** — locks `N_TREES=100`, `SEED=42`, `n_jobs=1`, dataset SHA-256, gating policy, cost model |
| `cost_model.yaml` | Bunker prices (VLSFO/MGO), port dues, canal fees, demurrage rates |
| `decision_policy.yaml` | BUY_NOW/WAIT/FLEX thresholds, fallback rules |
| `ports.yaml` | Port draft, beam, LOA, DWT constraints |
| `risk_policy.yaml` | Chokepoint & weather risk weights |
| `vessels.yaml` | Vessel DWT, speed, draft, fuel specs |
| `app_config.yaml` / `config.yaml` | Application-level environment settings |

---

### 📂 Outputs (`outputs/`)

| Directory | Contents |
| :--- | :--- |
| `authoritative/` | **Certified result payloads** — `authoritative_canonical_results.json` (RF metrics) + `authoritative_policy_results.json` (EXP-06 economics) |
| `backtests/` | Decision backtest run outputs |
| `figures/` | Generated diagnostic plots |
| `experiment_6_gate_quality/` | Gate quality experiment artefacts |
| `experiment_7_sharper_base_sparse_groups/` | Sharper gating experiment artefacts |
| `experiment_9_economic_backtest/` | EXP-00 through EXP-06 economic backtest artefacts |

---

## 🔍 Proof Infrastructure

FICOS has moved beyond "here is a notebook with results." The system has layered, independent evidence:

```
Production implementation
        ↓
Canonical configuration (SSOT)
        ↓
Authoritative result payload (JSON)
        ↓
Observation-level results
        ↓
Automated reconciliation
        ↓
Reproducibility tests
        ↓
Certification report
        ↓
Proof notebook (05_proof/)
```

Latest certified state:
- ✅ **32/32** unit + integration tests passing
- ✅ **4/4** reproducibility tests passing
- ✅ **72/72** verification checks passing
- ✅ Clean Git working tree

---

## 🔒 Security & Data Privacy

> **Proprietary Data Protection:** All raw maritime fixtures, AIS vessel tracking records, and dataset files (`data/`, `*.csv`, `*.xlsx`, `*.parquet`, `*.pkl`) are **strictly excluded via `.gitignore`** and are not tracked in public version control.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
