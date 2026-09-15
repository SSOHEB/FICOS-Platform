# FICOS Architecture Implementation — Task Tracker

## Phase 1 — Inspection & Baseline
- [x] Read decision_engine.py
- [x] Read feasibility_engine.py  
- [x] Read walkforward_validation.py
- [x] Read features.py
- [x] Read data_loader.py
- [x] Read evaluation.py
- [x] Read config.yaml
- [x] Read __init__.py
- [x] Read tests/test_final_report_reconciliation.py

## Phase 2 — Typed Domain Schemas & Config Layer
- [x] configs/ports.yaml
- [x] configs/vessels.yaml
- [x] configs/cost_model.yaml
- [x] configs/decision_policy.yaml
- [x] configs/risk_policy.yaml
- [x] src/domain/schemas.py

## Phase 3 — Model & Config Registry
- [x] registry/manifest.json
- [x] src/registry/registry.py

## Phase 4 — Forecast Service & Uncertainty
- [x] src/forecast/service.py
- [x] src/forecast/uncertainty.py
- [x] src/forecast/__init__.py

## Phase 5 — Operational Layer (Dataset B)
- [x] src/operational/port_repository.py
- [x] src/operational/vessel_repository.py
- [x] src/operational/feasibility_engine.py

## Phase 6 — Risk Engine (Dataset C)
- [x] src/risk/engine.py
- [x] src/scenario/engine.py

## Phase 7 — Cost Model & Idle-Time
- [x] src/cost/model.py
- [x] src/cost/idle_assessment.py

## Phase 8 — Expected-Cost Policy
- [x] src/policy/expected_cost_policy.py

## Phase 9 — Decision Engine v2
- [x] src/decision/schemas.py
- [x] src/decision/engine.py
- [x] src/decision/explanation.py

## Phase 10 — Recommendation Service
- [x] src/application/recommendation_service.py

## Phase 11 — CLI Integration
- [x] src/application/__main__.py
- [x] ficos_cli.py

## Phase 12 — Historical Decision Backtest
- [x] src/evaluation/decision_backtest.py

## Phase 13 — Permutation Test Restoration
- [x] tests/run_permutation_test.py

## Phase 14 — Tests
- [x] tests/test_domain.py
- [x] tests/test_config.py
- [x] tests/test_feasibility.py
- [x] tests/test_forecast_service.py
- [x] tests/test_uncertainty.py
- [x] tests/test_cost_model.py
- [x] tests/test_risk_engine.py
- [x] tests/test_decision_engine.py

## Phase 15 — Documentation
- [x] docs/architecture.md
- [x] README.md
