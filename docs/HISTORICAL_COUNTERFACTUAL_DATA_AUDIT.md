# Historical Counterfactual Data Audit

## Available repository evidence

| Source | Status | Use |
|---|---|---|
| `data/modeling_dataset.csv` | OBSERVED/RECONSTRUCTED | 2,581 dated rows, 482 columns, freight and engineered features |
| `data/interim/cleaned_spot_prices.csv` | OBSERVED | Historical freight and macro observations |
| `data/interim/weather_features.csv` | RECONSTRUCTED FROM HISTORICAL WEATHER | Port weather and engineered exposure features |
| `data/interim/cyclone_features.csv` | RECONSTRUCTED FROM HISTORICAL EVENTS | Cyclone exposure features |
| `data/interim/geopolitical_features.csv` | RECONSTRUCTED FROM HISTORICAL EVENTS | Event-derived risk features |
| `data/raw/maritime_macro_data.xlsx` | OBSERVED SOURCE WORKBOOK | Macro and maritime source material |
| `data/raw/baltic_freight_indices.xlsx` | OBSERVED SOURCE WORKBOOK | Ports, berths, traffic, capacity, turnaround, and fleet tables |

## Counterfactual boundary

The replay produces `HISTORICAL_MARKET_OPPORTUNITY` rows from observable market conditions. It does not claim that any row is a SAIL voyage or procurement event.

The following seven fields remain `PRIVATE_DATA_UNAVAILABLE`: contract prices, negotiated discounts, voyage-specific SAIL volumes, realized procurement costs, historical SAIL procurement decisions, SAIL budget allocations, and SAIL contract-capacity commitments.

## Temporal controls

Forecasts are regenerated using the frozen canonical expanding folds. Training preprocessing is fitted inside each fold. Future realized rates are retained only as outcomes for counterfactual evaluation and are marked `OUTCOME_ONLY_NOT_INPUT` in the replay. The state object rejects provenance entries explicitly labelled `FUTURE_INFORMATION`.

## Result

The repository supports a historical public-data counterfactual study and a transparent sensitivity analysis. It does not support a ledger-backed claim about actual SAIL economics.
