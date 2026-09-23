import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_csv('data/modeling_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

vessels = ['panamax', 'supramax', 'handy', 'cape']
FOLDS = [
    {'year': 2021, 'train_end': '2019-12-24', 'val_start': '2020-01-03', 'val_end': '2020-12-24', 'test_start': '2021-01-05', 'test_end': '2021-12-31'},
    {'year': 2022, 'train_end': '2020-12-24', 'val_start': '2021-01-05', 'val_end': '2021-12-24', 'test_start': '2022-01-03', 'test_end': '2022-12-30'},
    {'year': 2023, 'train_end': '2021-12-24', 'val_start': '2022-01-03', 'val_end': '2022-12-23', 'test_start': '2023-01-03', 'test_end': '2023-12-29'},
    {'year': 2024, 'train_end': '2022-12-23', 'val_start': '2023-01-03', 'val_end': '2023-12-22', 'test_start': '2024-01-02', 'test_end': '2024-12-31'},
    {'year': 2025, 'train_end': '2023-12-22', 'val_start': '2024-01-02', 'val_end': '2024-12-24', 'test_start': '2025-01-02', 'test_end': '2025-12-31'}
]

feature_cols = [c for c in df.columns if c not in ['date'] and not c.startswith('target_') and not c.startswith('dir_')]

cases = []
for vessel in vessels:
    rate_col = vessel
    tgt_col = f'target_{vessel}_1d'
    valid_row = df[rate_col].notnull() & df[tgt_col].notnull()
    
    for f in FOLDS:
        tr_mask = (df['date'] <= f['train_end']) & valid_row
        val_mask = (df['date'] >= f['val_start']) & (df['date'] <= f['val_end']) & valid_row
        te_mask = (df['date'] >= f['test_start']) & (df['date'] <= f['test_end']) & valid_row
        
        X_tr = np.nan_to_num(df.loc[tr_mask, feature_cols].values, nan=0.0)
        y_tr = df.loc[tr_mask, tgt_col].values - df.loc[tr_mask, rate_col].values
        
        X_val = np.nan_to_num(df.loc[val_mask, feature_cols].values, nan=0.0)
        y_val = df.loc[val_mask, tgt_col].values - df.loc[val_mask, rate_col].values
        
        X_te = np.nan_to_num(df.loc[te_mask, feature_cols].values, nan=0.0)
        y_te_base = df.loc[te_mask, rate_col].values
        y_te_true = df.loc[te_mask, tgt_col].values
        dates_te = df.loc[te_mask, 'date'].values
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        X_te_sc = scaler.transform(X_te)
        
        selector = SelectKBest(f_regression, k=min(30, X_tr_sc.shape[1]))
        X_tr_fit = selector.fit_transform(X_tr_sc, y_tr)
        X_val_fit = selector.transform(X_val_sc)
        X_te_fit = selector.transform(X_te_sc)
        
        mdl = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        mdl.fit(X_tr_fit, y_tr)
        
        preds_delta = mdl.predict(X_te_fit)
        val_preds_delta = mdl.predict(X_val_fit)
        
        val_res = y_val - val_preds_delta
        p10 = float(np.percentile(val_res, 10))
        p90 = float(np.percentile(val_res, 90))
        
        for i in range(len(y_te_true)):
            cases.append({
                'vessel': vessel,
                'year': f['year'],
                'date': dates_te[i],
                'y_base': y_te_base[i],
                'y_true': y_te_true[i],
                'y_pred': y_te_base[i] + preds_delta[i],
                'pred_delta': preds_delta[i],
                'actual_delta': y_te_true[i] - y_te_base[i],
                'p10': p10,
                'p90': p90
            })

df_res = pd.DataFrame(cases)
pct_delta = df_res['pred_delta'] / (df_res['y_base'] + 1e-8)
tau = 0.01

is_buy = (df_res['pred_delta'] > df_res['p90']) & (pct_delta > tau)
is_wait = (df_res['pred_delta'] < df_res['p10']) & (pct_delta < -tau)

df_res['decision'] = np.where(is_buy, 'NOW', np.where(is_wait, 'WAIT', 'FLEXIBLE'))
df_res['retained'] = df_res['decision'].isin(['NOW', 'WAIT'])

df_res['dir_correct'] = (np.sign(df_res['pred_delta']) == np.sign(df_res['actual_delta'])).astype(int)

sub_ret = df_res[df_res['retained']]
n_ret = len(sub_ret)
n_now = (sub_ret['decision'] == 'NOW').sum()
n_wait = (sub_ret['decision'] == 'WAIT').sum()
now_corr = sub_ret[sub_ret['decision'] == 'NOW']['dir_correct'].sum()
wait_corr = sub_ret[sub_ret['decision'] == 'WAIT']['dir_correct'].sum()
n_corr = sub_ret['dir_correct'].sum()
prec = n_corr / n_ret * 100.0

print(f"Total Retained N: {n_ret}")
print(f"NOW N: {n_now}, Correct: {now_corr}, Prec: {now_corr/n_now*100:.2f}%")
print(f"WAIT N: {n_wait}, Correct: {wait_corr}, Prec: {wait_corr/n_wait*100:.2f}%")
print(f"Overall Correct: {n_corr} / {n_ret}")
print(f"Calculated Gated Precision: {prec:.4f}% ({prec:.2f}%)")
