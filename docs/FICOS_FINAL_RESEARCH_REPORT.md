# FICOS Final Research Report

## Executive result

The final historical replay evaluated 4,804 common market opportunities from fresh leakage-safe forecasts. It did not fabricate SAIL procurement records. Historical market rates were observed; contract economics were explicit assumptions.

## Counterfactual policy results

| Policy | Opportunities | Counterfactual difference vs Always Spot | Evidence |
|---|---:|---:|---|
| Always Spot | 4,804 | $0 | Observed market rate |
| Always Contract | 4,804 | +$81.837M | Assumed 4.5% discount |
| Fixed Horizon | 4,804 | +$13.088M | Counterfactual timing |
| Timing Only | 4,804 | +$4.535M | Historical market outcome with observed future rate |
| Timing + HOW | 4,804 | +$79.419M | Timing plus assumed contract economics |
| FICOS Portfolio | 12 | Not estimated on full history | Separate constraint experiment |

The counterfactual cost differences are not realized SAIL savings. The corrected economic unit is daily rate multiplied by the assumed 20-day voyage duration; volume is used for capacity only. Descriptive paired bootstrap intervals are included in `policy_benchmark.csv`; they quantify replay variation under the stated counterfactual, not causal commercial uncertainty.

## Architectural findings

The corrected controlled ablation found +$87,320 modeled timing value versus Always Spot and +$83,431.70 for independent HOW after WHEN in the 12-opportunity scenario. Coupling, robust, and mean-CVaR produced no incremental cost change in that easy configuration. The 81-cell stress grid found 59 optimal and 22 infeasible cells; when capacity bound, coupling changed allocations and prevented infeasible contract volumes.

## Registry reconciliation

The current authoritative registry is `ml/registry/manifest.json`: four promoted 1D Random Forest pairs (`panamax`, `supramax`, `handy`, and `cape`), `supramax` 7D excluded, and the other 7D entries represented as fallback/non-promoted. The historical `docs-site` model-evaluation narrative contains older benchmark classifications and is not the current production registry. No claim of promoted 14D or KDCI 7D models is made.

## Final interpretation

FICOS is more than a collection of forecasts: selective timing, contract choice, and constraint-aware allocation are separately testable architectural layers. The strongest economic numbers remain counterfactual and assumption-dependent. No claim of actual SAIL savings is permitted without private ledger data.
