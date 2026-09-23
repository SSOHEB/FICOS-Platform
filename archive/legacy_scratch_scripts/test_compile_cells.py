import json

with open('notebooks/experiment_9_economic_charter_decision_backtest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_code = ''.join(cell['source'])
        print(f"--- Cell {idx} ---")
        try:
            # Test compiling code
            compile(source_code, f'<cell_{idx}>', 'exec')
            print(f"Cell {idx} compiled successfully.")
        except Exception as e:
            print(f"Cell {idx} COMPILE ERROR: {e}")
