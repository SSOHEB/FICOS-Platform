"""Make the master notebook regenerate all evidence before displaying it."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "FICOS_COMPLETE_RESEARCH_NOTEBOOK.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

fresh_source = '''from pathlib import Path
import json, runpy, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
assert (ROOT / "data" / "modeling_dataset.csv").exists(), "Run from the repository root."
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "modeling_dataset.csv"
OUT = ROOT / "outputs" / "experiments"
CF = OUT / "historical_counterfactual"
ABL = OUT / "architectural_ablation"
STRESS = OUT / "architectural_stress_grid"
AUTH = ROOT / "outputs" / "authoritative"

# FRESH RAW EXECUTION: these scripts read only the canonical raw dataset and
# frozen configuration. They overwrite their own evidence during this run.
runpy.run_path(str(ROOT / "scripts" / "run_historical_counterfactual.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "run_architectural_ablation.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "run_architectural_stress_grid.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "build_final_evidence_package.py"), run_name="__main__")

evidence = json.loads((AUTH / "FICOS_FINAL_EVIDENCE.json").read_text())
bench = pd.read_csv(CF / "policy_benchmark.csv")
master = pd.read_csv(ABL / "master_ablation_table.csv")
controlled = pd.read_csv(ABL / "controlled_incremental_value_table.csv")
stress = pd.read_csv(STRESS / "stress_grid_results.csv")
print("Fresh raw execution complete. Cached outputs were not used as inputs.")
print("Raw input:", RAW)
print("Fresh OOS rows:", evidence["forecast_population"]["fresh_oos_rows"])
'''

for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and "ROOT = Path.cwd()" in "".join(cell.get("source", [])):
        cell["source"] = fresh_source.splitlines(True)
        cell["outputs"] = []
        cell["execution_count"] = None
        break
else:
    raise RuntimeError("setup cell not found")

# Ensure the notebook's final manifest records its raw-run status.
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and "master_results = pd.DataFrame" in "".join(cell.get("source", [])):
        source = "".join(cell["source"])
        source = source.replace("'notebook_status': 'COMPLETE_RESEARCH_SYNTHESIS'", "'notebook_status': 'FRESH_RAW_EXECUTION'")
        source = source.replace("'private_sail_data': 'UNAVAILABLE'", "'private_sail_data': 'UNAVAILABLE', 'cached_outputs_used_as_inputs': False")
        cell["source"] = source.splitlines(True)
        cell["outputs"] = []
        cell["execution_count"] = None
        break

nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(nb_path)
