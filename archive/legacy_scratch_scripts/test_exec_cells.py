import os, sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('notebooks/experiment_9_economic_charter_decision_backtest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

g_vars = {'os': os, 'sys': sys, 'json': json, 'pd': pd, 'np': np, 'warnings': warnings, 'plt': plt, 'sns': sns, 'SEED': 42}

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        exec(src, g_vars)
        print(f"Code cell {idx} executed successfully.")

print("\nALL NOTEBOOK CELLS EXECUTED 100% CLEANLY WITH ZERO ERRORS!")
