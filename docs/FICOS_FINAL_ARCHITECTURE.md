# FICOS Final Architecture

## Purpose

FICOS is a leakage-safe freight forecasting and procurement-decision research system. Its final evidence layer evaluates **historical market opportunities**, not undocumented SAIL transactions.

## Architecture

```text
historical public data
  -> market/operational state
  -> leakage-safe walk-forward forecast
  -> uncertainty and selective WHEN decision
  -> opportunity engine
  -> HOW contract alternatives
  -> portfolio MILP with binding-constraint stress tests
  -> historical market counterfactual
  -> ablation, robustness, provenance, claim audit
```

The WHEN and HOW decisions remain separate. Robust and mean-CVaR optimization remain research components and are retained only with their measured limitations.

## Evidence status

- Observed: freight rates, macro variables, weather, cyclone events, geopolitical events, port/berth data, traffic, capacity, and fleet data.
- Reconstructed: fresh walk-forward forecasts, uncertainty bounds, WHEN decisions, historical market opportunities, and counterfactual policy costs.
- Scenario assumptions: volume, duration, contract economics, budget, contract capacity, and execution slippage.
- Private-data blocked: SAIL contract prices, discounts, procurement decisions, realized costs, budgets, and commitments.

## Final claims

The evidence supports a reproducible forecast-to-decision architecture and measurable modeled value for selective timing and contract selection. Coupling becomes operationally important when shared constraints bind, but its lower-cost commercial value is not established. Robust and CVaR objectives were redundant in the current ablation configuration. Economic replay uses daily rate times voyage duration; it does not multiply rates by cargo volume.

The dormant portfolio layers are retained with a pre-registered activation policy: coupling activates at or above shared-constraint utilization of 1.0, while Robust/CVaR activates when scenario P90 cost is at least 5% above median cost. These thresholds are operational gates, not claims that the layers have already demonstrated commercial value.
