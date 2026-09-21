# FICOS — Finalized Scalable Platform Architecture

## Executive Overview
The **Freight Intelligence & Chartering Optimization System (FICOS)** is a configuration-first, modular freight forecasting and decision engine designed for bulk ocean shipping.

The platform decouples freight rate forecasting (Dataset A machine learning models) from operational vessel-port feasibility constraints (Dataset B) and environmental/weather disruption risks (Dataset C).

---

## High-Level Component Diagram

```
                             DOMAIN KNOWLEDGE
                                    |
                                    v
       +-----------------------------------------------------------+
       |                     CONFIGURATION LAYER                   |
       |  ports.yaml | vessels.yaml | cost_model.yaml | policy...  |
       +-----------------------------------------------------------+
                                    |
                                    v
       +-----------------------------------------------------------+
       |                  INPUT & DOMAIN SCHEMAS                   |
       |  CargoRequirement | VesselClass | Port | Route | ...     |
       +-----------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
 +--------------------+   +--------------------+   +--------------------+
 |  DATASET A (ML)    |   |  DATASET B (PHYS)  |   |  DATASET C (RISK)  |
 |  Walk-Forward ML   |   |  Port & Vessel     |   |  Weather, GDELT,   |
 |  5-Fold Benchmark  |   |  Constraints Gate  |   |  Disruption Risk   |
 +--------------------+   +--------------------+   +--------------------+
            |                       |                       |
            v                       v                       v
 +--------------------+   +--------------------+   +--------------------+
 | FORECAST SERVICE   |   | FEASIBILITY ENGINE |   |    RISK ENGINE     |
 | Model Registry     |   | Physical Draft/GT  |   | Weather/Chokepoint |
 | Calibration P10/90 |   | Non-negotiable gate|   | Risk Scoring 0-100 |
 +--------------------+   +--------------------+   +--------------------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
       +-----------------------------------------------------------+
       |               COST MODEL & EXPECTED COST POLICY           |
       | Charter | Bunker | Port Calls | Canal | Demurrage Penalties|
       +-----------------------------------------------------------+
                                    |
                                    v
       +-----------------------------------------------------------+
       |                    DECISION ENGINE v2                     |
       | Evaluates SPOT vs TIME_CHARTER vs COA | Min Expected Cost |
       | Generates Audit Trail & Step-by-Step Explanation Rationale|
       +-----------------------------------------------------------+
                                    |
                                    v
       +-----------------------------------------------------------+
       |             RECOMMENDATION APPLICATION SERVICE            |
       | High-level API & CLI (`ficos_cli.py`) | Ready for REST/UI |
       +-----------------------------------------------------------+
```

---

## Core System Modules

### 1. Configuration Layer (`configs/`)
- `ports.yaml`: Draft limits, LOA, beam, air draft, quay length, and congestion parameters.
- `vessels.yaml`: Vessel dimensions, DWT, loading/discharge rates, fuel consumption rates.
- `cost_model.yaml`: Bunker fuel prices (VLSFO/MGO), port disbursement fees, canal transit costs, demurrage rates.
- `decision_policy.yaml`: Variance penalties, risk weighting factors, policy thresholds.

### 2. Domain Schemas (`src/domain/schemas.py`)
- Strongly typed Python dataclasses for all core entities: `CargoRequirement`, `VesselClass`, `Port`, `Route`, `ForecastResult`, `FeasibilityResult`, `RiskResult`, `CostBreakdown`, `PolicyEvaluation`.

### 3. Model Registry (`registry/manifest.json`, `src/registry/`)
- Centralized model manifest mapping `(asset_type, horizon_days)` to promoted model artifacts and empirical performance metrics (F1, Accuracy, P10/P90 residual calibration).

### 4. Forecast Service & Uncertainty Engine (`src/forecast/`)
- Obtains calibrated freight rate predictions and empirical P10/P90 confidence intervals without leaking operational assumptions into feature matrices.

### 5. Operational Feasibility Layer (`src/operational/`)
- Gated physical constraint checker verifying vessel draft, LOA, beam, air draft, and DWT capacity against origin and destination port specifications.

### 6. Risk Engine & Scenario Engine (`src/risk/`, `src/scenario/`)
- Evaluates Dataset C environmental, weather, and geopolitical disruption signals to construct a composite 0-100 risk score and severity alerts.
- Supports scenario replaying under alternative fuel prices, congestion delays, or weather conditions.

### 7. Cost Model & Expected Cost Policy (`src/cost/`, `src/policy/`)
- Comprehensive voyage cost accounting combining charter freight, bunker fuel consumption, port disbursements, canal transit fees, and idle delay penalties.
- Evaluates candidate chartering strategies (SPOT, TIME_CHARTER, COA) under uncertainty.

### 8. Decision Engine v2 & Recommendation Service (`src/decision/`, `src/application/`)
- Orchestrates all components, enforces physical gating, selects optimal strategy minimizing risk-adjusted cost, and generates human-readable audit explanations.

---

## Research & Economic Validation Scope (Experiment 9)
- **Forecast Validation vs. Economic Validation**: Forecast validation evaluates statistical accuracy ($R^2$, DA, AUC) under zero-leakage walk-forward protocols. Economic validation (Experiment 9) evaluates decision policy cost differences on historical holdout data.
- **Model Promotion Scope**: Model promotion in the registry signifies directional accuracy and walk-forward stability, **not proof of monetary savings**.
- **Experiment 9 Classification**: Experiment 9 is a decision-level economic validation study evaluating counterfactual cost differences vs. Always Spot on production-promoted pairs ($N=952$).
- **Current Economic Findings**: Aggregate economic cost difference is **inconclusive** (-0.131% backtested cost difference vs spot, 95% CI [-0.265%, +0.005%]).
- **WAIT Decision Sub-population**: WAIT decisions showed a +3.174% counterfactual cost difference vs spot, which is statistically unusual ($p < 0.0001$) relative to a 10,000-draw placebo distribution, but this does NOT establish proven real-world monetary savings.
- **Isolation**: Experiment 9 is strictly research/validation evidence and is completely isolated from production API inference.

---

## Verification & Execution

### CLI Runner
```bash
# Run freight decision evaluation
python ficos_cli.py evaluate --asset PANAMAX_1D --quantity 75000 --origin TUBARAO --dest QINGDAO

# Inspect model registry
python ficos_cli.py registry
```

### Unit & Integration Testing
```bash
python -m pytest tests/ -v
```
