import json
import matplotlib
matplotlib.use('Agg')

target_nb = "notebooks/FICOS_Authoritative_Proof_and_Research_Evidence_Notebook.ipynb"

with open(target_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

for idx, cell in enumerate(code_cells):
    code = "".join(cell['source'])
    print(f"\n--- Executing Code Cell {idx+1} ---")
    exec(code, globals())
print("\n✅ ALL CODE CELLS IN PROOF NOTEBOOK EXECUTED WITH ZERO ERRORS!")
