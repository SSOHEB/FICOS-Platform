# FICOS Decision-State Formulation

This document records the final decision precedence and the quantities exposed in the procurement recommendation.

## Confidence and uncertainty

Let `w = (P90 - P10) / current_rate * 100`.

- `w < 5%`: LOW uncertainty, HIGH confidence.
- `5% <= w < 15%`: MEDIUM uncertainty, MEDIUM confidence.
- `w >= 15%`: HIGH uncertainty, LOW confidence.

The P10/P90 values are empirical walk-forward validation-residual bounds. They are not guaranteed probability intervals.

## WHEN

For a promoted and feasible forecast, let `delta = expected_forecast - current_rate` and `tau = 0.01 * current_rate`:

- `delta > tau`: `NOW`.
- `delta < -tau`: `WAIT`.
- Otherwise: `FLEXIBLE`.

If the model is unpromoted, uncertainty is high, or physical feasibility fails, the state is `FLEXIBLE`.

If the risk state is HIGH and the preliminary state is `WAIT`, the configured risk override changes the state to `FLEXIBLE`.

## HOW

The comparator evaluates `SPOT`, `TIME_CHARTER`, `COA`, and `FLEXIBLE_INDEX`. For each strategy:

```text
adjusted_cost = expected_cost + variance_penalty + risk_penalty
risk_penalty = risk_score / 100 * expected_cost * 0.2
variance_penalty = uncertainty_spread * cargo_quantity * 0.3
```

For a firm `NOW` or `WAIT` state, the selected HOW is the feasible strategy with minimum adjusted cost. For `FLEXIBLE`, the system selects `FLEXIBLE_INDEX` to preserve optionality.

## Portfolio state

For independently selected strategies:

```text
capacity_utilization = contract_volume / contract_capacity
budget_utilization = portfolio_cost / budget
contract_utilization = contract_count / max_contracts
```

The portfolio layer activates when any utilization is at least `1.0`. The MILP then selects one strategy per voyage while satisfying the shared constraints.

