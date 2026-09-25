# FICOS Master Research Notebook Guide

Notebook: `notebooks/FICOS_COMPLETE_RESEARCH_NOTEBOOK.ipynb`

## Structure

The notebook moves from the SAIL problem to data lineage, forensic reconciliation, temporal validation, confidence gating, WHEN/HOW decisions, historical market counterfactuals, ablation, stress testing, risk layers, claim audit, and architecture lock.

## Evidence policy

The notebook reads cached authoritative outputs. It does not call paid APIs, read credentials, fabricate SAIL records, or silently rerun a different experiment. Counterfactual contract results are labelled scenario-dependent.

## Rerun

Open from the repository root and run top to bottom. The notebook requires Python, pandas, numpy, matplotlib, and Jupyter. The full historical replay can be regenerated with `python scripts/run_historical_counterfactual.py`; this is optional and takes materially longer than loading the cached outputs.

## Authoritative files

- `outputs/authoritative/FICOS_FINAL_EVIDENCE.json`
- `outputs/experiments/historical_counterfactual/`
- `outputs/experiments/architectural_ablation/`
- `outputs/experiments/architectural_stress_grid/`
- `docs/FICOS_CLAIM_AUDIT.md`
- `docs/FICOS_ARCHITECTURE_LOCK.md`

## Interpretation

The notebook is a research presentation and reproducibility surface, not a deployed SAIL system. Its strongest economic numbers are historical market counterfactuals under explicit assumptions. Actual SAIL procurement outcomes remain private-data blocked.
