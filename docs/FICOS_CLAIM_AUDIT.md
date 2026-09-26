# FICOS Claim Audit

| Claim | Status | Reason |
|---|---|---|
| Fresh OOS forecast population is 4,804 | PROVEN | Reproduced from canonical dataset and folds |
| Canonical gate retains 641 rows at 79.10% gated accuracy | PROVEN | Exact gate replay |
| Timing changes historical counterfactual cost | COUNTERFACTUAL | Uses observed market outcomes and locked policy logic |
| Contract selection creates commercial savings | SCENARIO-DEPENDENT | Discount and contract rate are assumed |
| Coupling improves feasibility under binding constraints | STRONGLY SUPPORTED | Stress grid shows allocation changes and avoided violations |
| Coupling always lowers cost | UNPROVEN | Independent comparator can be infeasible |
| Robust optimization adds economic value | UNPROVEN | Zero incremental value in current ablation |
| Mean-CVaR adds economic value | UNPROVEN | Zero incremental value in current ablation |
| Forecast results prove actual SAIL impact | PRIVATE-DATA-BLOCKED | No SAIL decisions or realized costs |
| FICOS is universally novel | UNPROVEN | No universal novelty claim made |
| FICOS combines leakage-safe forecasting, WHEN/HOW separation, and counterfactual attribution for freight procurement | STRONGLY SUPPORTED | Implemented and documented architectural combination |
| Current registry promotes 7D/14D models | PRIVATE-DATA-BLOCKED | Current `ml/registry/manifest.json` promotes four 1D pairs only; older benchmark documents are historical |
