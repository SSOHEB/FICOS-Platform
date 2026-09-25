# FICOS Final Research Report

## Executive result

The final historical replay evaluated 4,804 common market opportunities from fresh leakage-safe forecasts. It did not fabricate SAIL procurement records. Historical market rates were observed; contract economics were explicit assumptions.

## Counterfactual policy results

| Policy | Opportunities | Counterfactual difference vs Always Spot | Evidence |
|---|---:|---:|---|
| Always Spot | 4,804 | $0 | Observed market rate |
| Always Contract | 4,804 | +$306.890M | Assumed 4.5% discount |
| Fixed Horizon | 4,804 | +$49.078M | Counterfactual timing |
| Timing Only | 4,804 | +$17.007M | Historical market outcome with observed future rate |
| Timing + HOW | 4,804 | +$297.822M | Timing plus assumed contract economics |
| FICOS Portfolio | 12 | Not estimated on full history | Separate constraint experiment |

The counterfactual cost differences are not realized SAIL savings. Descriptive paired bootstrap intervals are included in `policy_benchmark.csv`; they quantify replay variation under the stated counterfactual, not causal commercial uncertainty.

## Architectural findings

The controlled ablation found +$327.450M modeled timing value versus Always Spot and +$312.869M for independent HOW after WHEN in the 12-opportunity scenario. Coupling, robust, and mean-CVaR produced no incremental cost change in that easy configuration. The 81-cell stress grid found 59 optimal and 22 infeasible cells; when capacity bound, coupling changed allocations and prevented infeasible contract volumes.

## Final interpretation

FICOS is more than a collection of forecasts: selective timing, contract choice, and constraint-aware allocation are separately testable architectural layers. The strongest economic numbers remain counterfactual and assumption-dependent. No claim of actual SAIL savings is permitted without private ledger data.
