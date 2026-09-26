# FICOS Architectural Ablation Report

Generated from a fresh run of `scripts/run_architectural_ablation.py` on the canonical modeling dataset. Every system received the same fresh OOS predictions, uncertainty bounds, 12 selected opportunities, scenarios, cost formula, and constraints.

## Result

| System | Eligible | WAIT | Actionable | SPOT | Contract | Modeled cost | Savings vs spot | Worst-case cost | Solver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Always spot | 12 | 8 | 4 | 12 | 0 | $4.030M | $0 | $4.168M | N/A |
| Independent contract | 12 | 8 | 4 | 0 | 12 | $3.867M | $162,984 | $3.998M | N/A |
| Forecast + WHEN | 12 | 8 | 4 | 4 | 0 | $3.943M | $87,320 | $4.015M | N/A |
| WHEN + independent HOW | 12 | 8 | 4 | 0 | 4 | $3.859M | $170,752 | $3.928M | N/A |
| FICOS deterministic | 12 | 8 | 4 | 0 | 4 | $3.859M | $170,752 | $3.928M | OPTIMAL |
| FICOS robust | 12 | 8 | 4 | 0 | 4 | $3.859M | $170,752 | $3.928M | OPTIMAL |
| FICOS mean-CVaR | 12 | 8 | 4 | 0 | 4 | $3.859M | $170,752 | $3.928M | OPTIMAL |

## Incremental interpretation

The literal requested ladder is retained in `incremental_value_table.csv`. Because Baseline 1 and Baseline 2 change more than one behavior at once, the controlled comparisons in `controlled_incremental_value_table.csv` are the more defensible attribution. The corrected economic unit is daily rate multiplied by the 20-day duration; cargo volume is used only for capacity constraints:

| Layer | Controlled incremental result |
|---|---:|
| Timing versus always spot | +$87,320 modeled savings; 8 decisions changed |
| Independent HOW after WHEN | +$83,431.70 modeled savings; 4 decisions changed |
| Cross-voyage coupling after WHEN/HOW | $0; no decisions changed |
| Robust objective | $0; no decisions changed |
| Mean-CVaR objective | $0; no decisions changed |

The current scenario has only four actionable opportunities after eight WAIT decisions. Contract capacity is not binding, and the assumed contract discount makes the contract independently preferable for every actionable opportunity. Consequently, this run cannot demonstrate a coupling benefit: the joint optimizer converges to the independent solution. That is a valid null result, not evidence that coupling is universally useless.

## Data and assumptions

- Fresh OOS rows: 4,804.
- Canonical gate replay: 641 retained rows, 13.34% coverage, 79.10% gated directional accuracy.
- Dataset SHA-256: `e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5`.
- Seed: 42; scenarios: 100; solver: SciPy MILP / HiGHS.
- Observed: historical rate and canonical row data.
- Derived: fold-fitted forecast and residual uncertainty bounds.
- Scenario assumptions: volume, duration, capacity, contract discount, budget, and contract capacity.

The dollar values are modeled scenario results, not causal commercial savings. No procurement ledger is present to validate those assumptions against realized contracts.

## Conclusion

FICOS demonstrates measurable architectural value in this run through selective timing and contract-structure decisions on a shared fresh OOS opportunity set. It does **not** demonstrate incremental value from cross-voyage coupling, robust optimization, or mean-CVaR under the current non-binding 12-voyage scenario. The strongest defensible claim is therefore architectural and reproducibility-focused: the system is more than a collection of forecasts, but the value of its portfolio-risk layers remains unproven until the experiment includes binding, defensible portfolio constraints or observed procurement data.

Do not claim causal commercial savings, guaranteed improvement, or universal superiority from this experiment.

## Evidence files

All generated evidence is under `outputs/experiments/architectural_ablation/`, including the fresh row-level predictions, shared opportunities, master table, both incremental tables, allocation comparison, assumption audit, manifest, and conclusion JSON.
