# FICOS — Economic Policy Attribution Audit Report

**Executed**: 2026-09-24  
**Git Commit**: `ef6970f3f96ac2bc55dcc02e40c490102ea10df3`  
**Dataset SHA-256**: `e0f4c91eed7b4919200472c3fe7e0735e4fd12433383727b58f73c2fd8945fd5`  
**SSOT Config**: `N_TREES=100`, `SEED=42`, `n_jobs=1`  

---

## 1. Executive Summary

This audit establishes the **exact mechanical cause** of the **`-$503,745.00`** net portfolio loss for the canonical 1D `RF_STANDARD` model across 4,804 observations (2021–2025).

### Key Finding
> **The `-$503,745.00` portfolio loss is NOT caused by poor ML model predictions or WAIT decisions.**  
> The 1D RF_STANDARD model produces **`+$3,725,220.00`** net economic savings across its 324 WAIT decisions (83.95% precision).  
>  
> Instead, **`100%` of the portfolio loss (and more)** is caused by the **FLEXIBLE routing formula** (`cost_flex = avg(spot, true) * 20 + 2500 * 0.25`), which creates a **`-$4,228,965.00`** cost premium vs pure spot indexing across 4,163 FLEX voyages (~86.66% of the portfolio).

---

## 2. Exact Portfolio Decomposition (-$503,745.00 Reconciled)

$$\text{TOTAL NET} = \Delta_{\text{NOW}} + \Delta_{\text{WAIT}} + \Delta_{\text{FLEXIBLE}}$$

| Decision Bucket | Observation Count | % of Portfolio | Total Net Savings (USD) | Mean Net / Voyage (USD) | Gated Precision (%) |
|---|---:|---:|---:|---:|---:|
| **NOW** | 317 | 6.60% | **$0.00** | $0.00 | 74.13% |
| **WAIT** | 324 | 6.74% | **+$3,725,220.00** | +$11,497.59 | **83.95%** |
| **FLEXIBLE** | 4,163 | 86.66% | **-$4,228,965.00** | -$1,015.85 | N/A |
| **TOTAL / PORTFOLIO** | **4,804** | **100.00%** | **-$503,745.00** | **-$104.86** | **79.10%** |

*Reconciliation Verification*: `$0.00 + $3,725,220.00 - $4,228,965.00 = -$503,745.00` (Exact Match ✅)

---

## 3. FLEXIBLE Routing Audit (The Single Source of Portfolio Loss)

- **Total FLEX Observations**: 4,163 (86.66% of portfolio)
- **FLEX Pricing Formula**: `cost_flex = (base + 0.5 * true_delta) * 20 + 2500 * 0.25`
- **FLEX Idle Penalty**: Every FLEX decision incurs a fixed $625.00 idle penalty ($2,500/day * 0.25 days).
- **Cumulative Idle Penalty**: `4,163 * $625.00 = $2,601,875.00`
- **Rate Drift Penalty**: Market rates rose on average during FLEX periods, adding another `$1,627,090.00` in cost, bringing the total FLEX premium to **`+$4,228,965.00`**.
- **Vessel Distribution**:
  - `cape` (1,055 voyages): -$2,359,215.00 loss
  - `supramax` (1,028 voyages): -$765,430.00 loss
  - `handy` (1,041 voyages): -$578,695.00 loss
  - `panamax` (1,039 voyages): -$525,625.00 loss

> **Conclusion**: Because 86.66% of all voyages are routed to FLEX, the portfolio economics are overwhelmingly dominated by the `cost_flex` formula assumption.

---

## 4. NOW vs WAIT Decision Quality & Asymmetry Audit

| Action | Total Count | Correct Direction (N) | Incorrect Direction (N) | Net Gain on Correct (USD) | Net Loss on Incorrect (USD) | Asymmetry Ratio |
|---|---:|---:|---:|---:|---:|---:|
| **WAIT** | 324 | 272 (83.95%) | 52 (16.05%) | +$4,577,780.00 | -$852,560.00 | **0.97x** |
| **NOW** | 317 | 235 (74.13%) | 82 (25.87%) | $0.00 (spot rate) | $0.00 (spot rate) | N/A |

### Key Findings:
1. **WAIT decisions generate substantial value**: 272 correct WAIT decisions generate **`+$4,577,780.00`** in net rate savings, easily overcoming the **`-$852,560.00`** loss on 52 incorrect WAITs (which includes demurrage). Net WAIT contribution = **`+$3,725,220.00`**.
2. **NOW decisions execute at baseline spot**: Locking spot rate immediately produces `$0.00` delta vs baseline spot.

---

## 5. Offline Policy Sensitivity Analysis (Leakage-Safe Walk-Forward)

| Policy ID | Description | Gated Prec (%) | Coverage (%) | 2025 WAIT Net (USD) | Total 5-Yr Portfolio Net (USD) | Net Delta vs Canonical |
|---|---|---:|---:|---:|---:|---:|
| `POL-000_CANONICAL` | Canonical Policy (p10/p90 bounds, 0.25d flex idle) | 79.10% | 13.34% | +$344,840.00 | **-$503,745.00** | $0.00 |
| `POL-001_FLEX_PURE_SPOT` | FLEX = Pure Spot Index (cost_flex = cost_spot) | 79.10% | 13.34% | +$344,840.00 | **+$3,725,220.00** | **+$4,228,965.00** |
| `POL-002_WAIT_ONLY_SPOT_FLEX` | WAIT-Only Active Gating; NOW & FLEX default to Spot | 83.95% | 6.74% | +$344,840.00 | **+$3,725,220.00** | **+$4,228,965.00** |
| `POL-003_EMV_MONETARY_WAIT_GATE` | Monetary EV Gating (WAIT only if pred decline > $125/MT) | 83.33% | 8.24% | +$342,460.00 | **+$4,006,900.00** | **+$4,510,645.00** |

---

## 6. Root Cause Summary & Recommended Interventions

### Diagnostic Summary
1. **Model Quality**: **EXCELLENT.** 1D RF_STANDARD generates **`+$3.72M`** net savings on WAIT decisions.
2. **Decision Policy**: **SOLID.** Gated precision is 79.10% overall and 83.95% on WAITs.
3. **FLEXIBLE Cost Formula**: **SINGLE SOURCE OF LOSS.** Applying an artificial 0.25-day idle penalty plus rate drift formula to 86.66% of the portfolio creates a **`-$4.23M`** penalty vs spot index.

### Recommended Evidence-Backed Cure (Next Iteration)
1. **Default FLEX routing to pure spot index**: Set `cost_flex = cost_spot` (index procurement).
2. **Apply Monetary EV Gating for WAIT (`POL-003`)**: Require predicted rate decline $> \$125$/MT (the demurrage breakeven point) before triggering a WAIT order.

These two policy adjustments convert the portfolio net result from **`-$503,745.00`** into **`+$4,006,900.00`** (+834.08 $/voyage) out-of-sample without modifying the certified 100-tree RF_STANDARD model.
