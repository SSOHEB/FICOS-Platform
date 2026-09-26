# FICOS Architectural Stress-Grid Report

This experiment keeps the canonical forecasting model and fresh OOS construction unchanged. It varies only explicit scenario assumptions to test whether portfolio coupling becomes decision-relevant when constraints bind.

## Design

The grid varies:

- 4, 8, or 12 selected opportunities;
- contract discounts of 1%, 4.5%, or 10%;
- contract capacity equal to 25%, 50%, or 100% of the actionable volume;
- budget multipliers of 0.90, 1.00, or 1.10 times actionable spot cost.

There are 81 scenario cells. Each cell compares unconstrained independent contract choice with deterministic joint optimization over the same actionable opportunities. Forecasts, rows, timing decisions, costs, and information are otherwise shared.

## Findings

- 59 cells solved optimally.
- 22 cells were infeasible under the imposed budget/capacity combination. These are recorded rather than silently removed.
- In the original easy scenario, coupling made no change because capacity was not binding.
- Under binding capacity, coupling changed decisions. For example, with 12 opportunities, 4 actionable rows, and 75,000 MT capacity, independent choice selected 4 contracts while the joint optimizer selected 1, changing 3 decisions and avoiding a 225,000 MT capacity excess.
- The joint plan cost more than the unconstrained independent counterfactual in those cells. This is expected: the independent policy violates the shared resource constraint, so its lower cost is not a feasible portfolio result.

## Interpretation

The stress test changes the conclusion in a precise way:

> Cross-voyage coupling is not economically beneficial merely because it is mathematically sophisticated. It becomes operationally necessary when shared capacity or budget binds, because it enforces portfolio feasibility and changes allocations. A separate claim of incremental cost savings requires a feasible independent comparator or observed procurement outcomes.

This is stronger and more defensible than claiming that coupling “saves money.” The current evidence supports **constraint-handling value**, not causal commercial savings.

## Limitations

Volume, contract discount, capacity, and budget remain scenario assumptions. The grid is a sensitivity analysis, not a replacement for a procurement ledger. The unconstrained independent baseline is intentionally retained as a counterfactual to reveal violations; it must not be described as a feasible competing portfolio.

## Pre-registered activation policy

The dormant layers remain part of FICOS. They are activated by the policy in `configs/portfolio_activation_policy.yaml`:

- **Coupling activates** when the independently preferred portfolio reaches or exceeds shared contract capacity, shared budget, or maximum contract count. The trigger is `utilization >= 1.0`; it is not selected after inspecting economic outcomes.
- **Robust/CVaR activates** when the independent plan's scenario P90 cost is at least 5% above its median scenario cost. Otherwise the deterministic plan remains active and the risk layer is recorded as dormant for that case.

This creates a falsifiable regime definition. The current easy ablation is below the coupling trigger, which explains its zero incremental value. Binding stress cells cross the capacity trigger, and their outputs record activation state and reason in `stress_grid_results.csv`.

## Evidence

Generated files are under `outputs/experiments/architectural_stress_grid/`:

- `fresh_oos_predictions.csv`
- `stress_grid_results.csv`
- `stress_grid_manifest.json`

The experiment is reproducible with `python scripts/run_architectural_stress_grid.py`.
