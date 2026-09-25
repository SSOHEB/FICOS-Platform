"""Make the master notebook regenerate all evidence before displaying it."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "notebooks" / "FICOS_COMPLETE_RESEARCH_NOTEBOOK.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

fresh_source = '''from pathlib import Path
import contextlib, hashlib, io, json, runpy, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "data" / "modeling_dataset.csv").exists():
    import subprocess
    clone_root = Path("/content/FICOS-Platform")
    if not (clone_root / "data" / "modeling_dataset.csv").exists():
        subprocess.run(["git", "clone", "https://github.com/SSOHEB/FICOS-Platform.git", str(clone_root)], check=True)
    else:
        subprocess.run(["git", "-C", str(clone_root), "pull", "--ff-only"], check=True)
    ROOT = clone_root
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "modeling_dataset.csv"
OUT = ROOT / "outputs" / "experiments"
CF = OUT / "historical_counterfactual"
ABL = OUT / "architectural_ablation"
STRESS = OUT / "architectural_stress_grid"
AUTH = ROOT / "outputs" / "authoritative"

# FRESH RAW EXECUTION: these scripts read only the canonical raw dataset and
# frozen configuration. They overwrite their own evidence during this run.
df = pd.read_csv(RAW, parse_dates=["date"])
with contextlib.redirect_stdout(io.StringIO()):
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

inventory_source = '''source_specs = [
    ("data/modeling_dataset.csv", "canonical matrix", "raw input used by this run"),
    ("data/interim/cleaned_spot_prices.csv", "spot provenance", "historical source"),
    ("data/interim/weather_features.csv", "weather provenance", "historical source"),
    ("data/interim/cyclone_features.csv", "cyclone provenance", "historical source"),
    ("data/interim/geopolitical_features.csv", "geopolitical provenance", "historical source"),
    ("data/raw/maritime_macro_data.xlsx", "macro provenance", "historical source"),
    ("data/raw/baltic_freight_indices.xlsx", "port/fleet provenance", "historical source"),
]
inventory = pd.DataFrame([
    [path, "AVAILABLE" if (ROOT / path).exists() else "NOT_IN_CLONE", purpose, "RAW/SOURCE"]
    for path, purpose, _ in source_specs
], columns=["Source", "Availability", "Purpose", "Classification"])
display(inventory)
print("Canonical dataset SHA-256:", hashlib.sha256(RAW.read_bytes()).hexdigest())
print("Only modeling_dataset.csv is required for the fresh canonical computation.")'''

for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and "inventory = pd.DataFrame" in "".join(cell.get("source", [])):
        cell["source"] = inventory_source.splitlines(True)
        cell["outputs"] = []
        cell["execution_count"] = None
        break

visual_source = '''families = pd.Series({"Market/commodity derived":160, "Freight derived":105, "Weather":61, "GDELT":53, "Route freight":22, "Raw macro/commodity":19, "Cyclone":16, "Raw freight":5})
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
families.sort_values().plot.barh(ax=axes[0], color="#2f6f8f", title="Canonical feature families")
axes[0].set_xlabel("Feature count")
missing = df.isna().sum().sort_values(ascending=False).head(12)
missing[missing.gt(0)].sort_values().plot.barh(ax=axes[1], color="#d97706", title="Top missing-value counts")
axes[1].set_xlabel("Missing cells")
plt.tight_layout(); plt.show()
rate_cols = [c for c in ["cape", "panamax", "supramax", "handy"] if c in df.columns]
df[rate_cols].plot(figsize=(12, 4), title="Historical freight-rate series used by the canonical dataset")
plt.ylabel("Rate index"); plt.tight_layout(); plt.show()'''

for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and "families = pd.Series" in "".join(cell.get("source", [])):
        cell["source"] = visual_source.splitlines(True)
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
