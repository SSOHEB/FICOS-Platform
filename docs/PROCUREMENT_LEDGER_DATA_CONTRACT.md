# Procurement Ledger Data Contract

FICOS currently uses scenario assumptions for several economic inputs. This contract defines the minimum historical ledger needed to replace them with observed data. The template is `data/procurement_ledger_template.csv`.

## Required row grain

One row must represent one voyage procurement decision and its realized execution. `voyage_id` must be stable and unique within the source system. The decision timestamp must be at or before the execution/contract period; future fields must not be joined into forecasting features.

## Field classes

| Field | Required | Classification | Validation |
|---|---:|---|---|
| `voyage_id` | yes | observed | non-empty, unique per voyage |
| `decision_timestamp` | yes | observed | parseable timestamp |
| `vessel_class` | yes | observed | non-empty |
| `route`, `origin`, `destination` | yes | observed | non-empty |
| `actual_volume_mt` | yes | observed | positive and <= `vessel_capacity_mt` |
| `vessel_capacity_mt` | yes | observed | positive |
| `duration_days` | yes | observed | positive |
| `procurement_strategy` | yes | observed | `SPOT`, `SHORT_TERM`, `MEDIUM_TERM`, or `MULTI_VOYAGE_CONTRACT` |
| `contract_id` | conditional | observed | required for non-`SPOT`; blank for `SPOT` |
| `contract_start_date`, `contract_end_date` | conditional | observed | required for contract strategies; end >= start |
| `spot_rate_usd_per_mt` | yes | observed/derived | positive, source definition documented |
| `contract_rate_usd_per_mt` | conditional | observed | required for contract strategies and positive |
| `realized_procurement_cost_usd` | yes | observed/derived | non-negative; reconciliation target |
| `execution_slippage_usd` | yes | observed/derived | non-negative or signed only with documented convention |
| `budget_period` | yes | observed | stable period key |
| `budget_limit_usd` | yes | observed | non-negative |
| `contract_capacity_limit_mt` | yes | observed | non-negative |
| `source_record_id`, `source_system` | yes | provenance | traceable back to source |

## What this unlocks

With populated rows, FICOS can replace assumed volume, duration, discount, budget, and contract-capacity values; calculate realized procurement cost from the ledger; compare feasible independent and joint policies; and evaluate whether coupling creates realized value rather than only constraint-handling value.

The validation script does not fabricate missing values. Run:

```text
python scripts/validate_procurement_ledger.py data/procurement_ledger_template.csv
```

Until a populated ledger passes validation, economic outputs remain scenario analyses and must not be described as causal commercial savings.
