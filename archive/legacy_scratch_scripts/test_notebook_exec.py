"""
Test script to run all code cells from notebooks/final_model_family_investigation_and_validation.ipynb
in a clean runtime environment to guarantee 100% pass on all automated validation gates.
"""
import json
import sys

with open('notebooks/final_model_family_investigation_and_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Starting clean execution test of all notebook code cells...")

def dummy_display(*args, **kwargs):
    pass

globs = {'display': dummy_display}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        print(f"--> Executing Cell {i}...")
        # filter out ipython magics / shell commands if testing locally
        lines = []
        for line in src.splitlines():
            if line.strip().startswith('!') or line.strip().startswith('%'):
                continue
            lines.append(line)
        clean_src = "\n".join(lines)
        exec(clean_src, globs)

print("\n[SUCCESS] ALL NOTEBOOK CELLS EXECUTED CLEANLY WITH 100% ASSERTION PASS!")
