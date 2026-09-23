import json
import matplotlib
matplotlib.use('Agg')

with open('notebooks/FICOS_Authoritative_Research_Evidence_Notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

for idx, cell in enumerate(code_cells):
    code = "".join(cell['source'])
    print(f"\n--- Executing Code Cell {idx+1} ---")
    exec(code, globals())
print("\n✅ ALL CODE CELLS EXECUTED WITH ZERO ERRORS!")
