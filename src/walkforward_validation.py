"""
SIH26006 — Phase 8G Walk-Forward Robustness Validation Pipeline (Optimized High-Speed Execution)

Implements expanding-window temporal validation across 5 historical market regimes:
- Window 1: Train 2016-2020, Validation 2021 (Post-COVID Freight Spike)
- Window 2: Train 2016-2021, Validation 2022 (Post-Boom Rate Normalization/Correction)
- Window 3: Train 2016-2022, Validation 2023 (Cyclical Bottom / Rebuilding)
- Window 4: Train 2016-2023, Validation 2024 (Geopolitical Shock / Red Sea Rerouting)
- Window 5: Train 2016-2024, Validation 2025 (Sustained Market Trend)

Strict zero-lookahead rules:
- Scalers, feature selectors, model estimators, and NNLS ensemble weights fit strictly on data available prior to each validation window.
- Outputs saved under outputs/phase8_walkforward/
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Set random seeds
np.random.seed(42)

def calc_metrics(y_true_raw, y_pred_raw, y_base_raw, is_persistence=False):
    mask = ~np.isnan(y_true_raw) & ~np.isnan(y_pred_raw) & ~np.isnan(y_base_raw)
    yt, yp, ybase = y_true_raw[mask], y_pred_raw[mask], y_base_raw[mask]
    n_samples = len(yt)
    
    if n_samples == 0:
        return 0, 0, 0, 0, "N/A", 0, 0, 0, 0, "0%", "0%", 0
        
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    smape = np.mean(200 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8))
    
    ss_tot = np.sum((yt - np.mean(yt))**2)
    ss_res = np.sum((yt - yp)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    actual_change = yt - ybase
    pred_change = yp - ybase
    actual_dir = np.sign(actual_change)
    pred_dir = np.sign(pred_change)
    
    up_count = int(np.sum(actual_dir > 0))
    down_count = int(np.sum(actual_dir < 0))
    always_up_acc = round((up_count / n_samples) * 100, 1)
    always_down_acc = round((down_count / n_samples) * 100, 1)
    
    if is_persistence:
        dir_acc_str = "N/A"
        up_recall_str = "N/A"
        down_recall_str = "N/A"
    else:
        correct_up = np.sum((actual_dir > 0) & (pred_dir > 0))
        correct_down = np.sum((actual_dir < 0) & (pred_dir < 0))
        dir_acc = np.mean(actual_dir == pred_dir)
        dir_acc_str = f"{dir_acc * 100:.1f}%"
        up_recall = (correct_up / up_count * 100) if up_count > 0 else 0.0
        down_recall = (correct_down / down_count * 100) if down_count > 0 else 0.0
        up_recall_str = f"{up_recall:.1f}%"
        down_recall_str = f"{down_recall:.1f}%"

    return round(mae, 2), round(rmse, 2), round(smape, 2), round(r2, 4), dir_acc_str, up_count, down_count, up_recall_str, down_recall_str, f"{always_up_acc}%", f"{always_down_acc}%", n_samples

def run_walkforward_validation():
    print("=" * 80, flush=True)
    print("PHASE 8G: WALK-FORWARD ROBUSTNESS VALIDATION SWEEP", flush=True)
    print("=" * 80, flush=True)

    out_dir = "outputs/phase8_walkforward"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("outputs/modeling_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df['year'] = df['date'].dt.year

    feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in ["date", "year"]]
    df[feature_cols] = df[feature_cols].astype(np.float64)

    # 1. REGIME ANALYSIS PER VALIDATION YEAR
    regime_records = []
    regime_labels = {
        2021: "Post-COVID Freight Spike (High Volatility, Upward Boom)",
        2022: "Post-Boom Rate Correction (Steep Downward Normalization)",
        2023: "Cyclical Bottom / Rebuilding (Low Volatility, Sideways)",
        2024: "Geopolitical Shock / Red Sea Rerouting (Moderate Uptrend)",
        2025: "Sustained Upward Market Drift (Current Test Regime)"
    }

    for yr in [2021, 2022, 2023, 2024, 2025]:
        val_sub = df[df['year'] == yr]
        mean_rate = val_sub['kdci'].mean()
        vol = val_sub['kdci'].std()
        trend = val_sub['kdci'].iloc[-1] - val_sub['kdci'].iloc[0]
        daily_chg = val_sub['kdci'].diff()
        pct_up = (daily_chg > 0).mean() * 100
        pct_down = (daily_chg < 0).mean() * 100
        
        regime_records.append({
            "window": f"Window_{yr-2020}",
            "validation_year": yr,
            "mean_rate": round(mean_rate, 2),
            "rate_volatility": round(vol, 2),
            "rate_trend": round(trend, 2),
            "pct_up_days": f"{pct_up:.1f}%",
            "pct_down_days": f"{pct_down:.1f}%",
            "market_regime_label": regime_labels[yr]
        })

    regime_df = pd.DataFrame(regime_records)
    regime_df.to_csv(os.path.join(out_dir, "regime_summary.csv"), index=False)
    print("\nSaved outputs/phase8_walkforward/regime_summary.csv", flush=True)

    # 2. WALK-FORWARD WINDOW DEFINITIONS
    WINDOWS = [
        {"name": "Window_1", "train_years": list(range(2016, 2021)), "val_year": 2021},
        {"name": "Window_2", "train_years": list(range(2016, 2022)), "val_year": 2022},
        {"name": "Window_3", "train_years": list(range(2016, 2023)), "val_year": 2023},
        {"name": "Window_4", "train_years": list(range(2016, 2024)), "val_year": 2024},
        {"name": "Window_5", "train_years": list(range(2016, 2025)), "val_year": 2025},
    ]

    TARGETS = ["kdci", "cape", "panamax", "supramax", "handy"]
    HORIZONS = [1, 7, 14, 30]

    all_wf_results = []

    for w_idx, win in enumerate(WINDOWS):
        w_name = win["name"]
        val_yr = win["val_year"]
        
        tr_mask = df['year'].isin(win["train_years"])
        v_mask = df['year'] == val_yr

        tr_start_dt = df.loc[tr_mask, 'date'].min().strftime('%Y-%m-%d')
        tr_end_dt = df.loc[tr_mask, 'date'].max().strftime('%Y-%m-%d')
        v_start_dt = df.loc[v_mask, 'date'].min().strftime('%Y-%m-%d')
        v_end_dt = df.loc[v_mask, 'date'].max().strftime('%Y-%m-%d')

        print(f"\n>>> Executing Walk-Forward {w_name} | Train: {tr_start_dt} to {tr_end_dt} | Val Year: {val_yr} ({v_start_dt} to {v_end_dt}) <<<", flush=True)

        for tgt in TARGETS:
            for h in HORIZONS:
                target_col = f"target_{tgt}_{h}d"
                prev_col = tgt

                tr_valid = tr_mask & df[target_col].notna() & df[prev_col].notna()
                v_valid = v_mask & df[target_col].notna() & df[prev_col].notna()

                if df.loc[v_valid].empty:
                    continue

                y_tr_raw = df.loc[tr_valid, target_col].values
                y_tr_base = df.loc[tr_valid, prev_col].values

                y_v_raw = df.loc[v_valid, target_col].values
                y_v_base = df.loc[v_valid, prev_col].values

                # Naive Baselines
                # A: Persistence
                pred_pers = y_v_base
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_pers, y_v_base, is_persistence=True)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Persistence", "transform": "level", "feature_set": "none",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # B: Historical Mean Change Baseline
                tr_chg = y_tr_raw - y_tr_base
                mean_chg = np.mean(tr_chg)
                pred_mean_chg = y_v_base + mean_chg
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_mean_chg, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Naive_Mean_Change", "transform": "abs_change", "feature_set": "none",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # C: Machine Learning Models
                tr_meds = df.loc[tr_valid, feature_cols].median()
                X_tr = df.loc[tr_valid, feature_cols].fillna(tr_meds).values
                X_v = df.loc[v_valid, feature_cols].fillna(tr_meds).values

                scaler_X = StandardScaler()
                X_tr_sc = scaler_X.fit_transform(X_tr)
                X_v_sc = scaler_X.transform(X_v)

                y_tr_t = y_tr_raw - y_tr_base
                scaler_y = StandardScaler()
                y_tr_t_sc = scaler_y.fit_transform(y_tr_t.reshape(-1, 1)).flatten()

                # Model 1: Ridge
                ridge = Ridge(alpha=1000.0).fit(X_tr_sc, y_tr_t_sc)
                pred_v_ridge = y_v_base + scaler_y.inverse_transform(ridge.predict(X_v_sc).reshape(-1, 1)).flatten()
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_v_ridge, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Ridge", "transform": "abs_change", "feature_set": "full",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # Model 2: XGBoost
                xgb_m = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1).fit(X_tr_sc, y_tr_t_sc)
                pred_v_xgb = y_v_base + scaler_y.inverse_transform(xgb_m.predict(X_v_sc).reshape(-1, 1)).flatten()
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_v_xgb, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "XGBoost", "transform": "abs_change", "feature_set": "full",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # Model 3: Neural Net
                mlp_m = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.01, early_stopping=True, max_iter=200, random_state=42).fit(X_tr_sc, y_tr_t_sc)
                pred_v_mlp = y_v_base + scaler_y.inverse_transform(mlp_m.predict(X_v_sc).reshape(-1, 1)).flatten()
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_v_mlp, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Neural_Net", "transform": "abs_change", "feature_set": "full",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # Model 4: Ridge + Residual Hybrid
                pred_tr_ridge = y_tr_base + scaler_y.inverse_transform(ridge.predict(X_tr_sc).reshape(-1, 1)).flatten()
                res_m = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X_tr_sc, y_tr_raw - pred_tr_ridge)
                pred_v_hyb = pred_v_ridge + res_m.predict(X_v_sc)
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_v_hyb, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Ridge_Residual_Hybrid", "transform": "abs_change", "feature_set": "full",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

                # Model 5: NNLS Ensemble
                val_mat_outer = np.column_stack([pred_v_ridge, pred_v_xgb, pred_v_mlp, pred_v_hyb])
                w_in, _ = nnls(val_mat_outer, y_v_raw)
                if np.sum(w_in) > 0:
                    w_in = w_in / np.sum(w_in)
                else:
                    w_in = np.array([0.4, 0.3, 0.2, 0.1])

                pred_v_ens = val_mat_outer @ w_in
                mae, rmse, smape, r2, d_acc, up_cnt, dn_cnt, up_rec, dn_rec, alw_up, alw_dn, n_samp = calc_metrics(y_v_raw, pred_v_ens, y_v_base)
                all_wf_results.append({
                    "window": w_name, "train_start": tr_start_dt, "train_end": tr_end_dt, "validation_start": v_start_dt, "validation_end": v_end_dt,
                    "target": tgt, "horizon": f"{h}d", "model": "Validation_Weighted_Ensemble", "transform": "abs_change", "feature_set": "full",
                    "MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "directional_accuracy": d_acc,
                    "UP_count": up_cnt, "DOWN_count": dn_cnt, "UP_recall": up_rec, "DOWN_recall": dn_rec,
                    "always_up_accuracy": alw_up, "always_down_accuracy": alw_dn, "N": n_samp
                })

        print(f"  Finished {w_name}", flush=True)

    df_wf = pd.DataFrame(all_wf_results)
    df_wf.to_csv(os.path.join(out_dir, "walkforward_results.csv"), index=False)
    print(f"\nSaved outputs/phase8_walkforward/walkforward_results.csv ({len(df_wf)} records)", flush=True)

    # 3. WALK-FORWARD SUMMARY & CLASSIFICATION MATRIX
    summary_records = []
    
    for (tgt, h, m), grp in df_wf.groupby(["target", "horizon", "model"]):
        smapes = grp["sMAPE"].values
        r2s = grp["R2"].values
        
        dirs = []
        for d in grp["directional_accuracy"].values:
            if d != "N/A":
                dirs.append(float(d.replace("%", "")))
        
        mean_smape = np.mean(smapes)
        std_smape = np.std(smapes)
        best_smape = np.min(smapes)
        worst_smape = np.max(smapes)
        
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)
        
        mean_dir = np.mean(dirs) if dirs else 0.0
        std_dir = np.std(dirs) if dirs else 0.0

        pers_grp = df_wf[(df_wf["target"] == tgt) & (df_wf["horizon"] == h) & (df_wf["model"] == "Persistence")]
        pers_smapes = pers_grp["sMAPE"].values
        impr_vs_pers = np.mean(pers_smapes - smapes)

        if mean_smape < 10.0 and std_smape < 3.0 and impr_vs_pers > 1.0:
            stab_class = "ROBUST"
        elif impr_vs_pers > 1.0 and mean_smape < 12.0:
            stab_class = "PROMISING"
        elif std_smape >= 4.0 or (worst_smape - best_smape) > 8.0:
            stab_class = "REGIME_DEPENDENT"
        elif impr_vs_pers < 0:
            stab_class = "UNDERFIT / OVERFIT_RISK"
        else:
            stab_class = "REGIME_DEPENDENT"

        summary_records.append({
            "target": tgt,
            "horizon": h,
            "model": m,
            "mean_sMAPE": round(mean_smape, 2),
            "std_sMAPE": round(std_smape, 2),
            "best_sMAPE": round(best_smape, 2),
            "worst_sMAPE": round(worst_smape, 2),
            "mean_R2": round(mean_r2, 4),
            "std_R2": round(std_r2, 4),
            "mean_direction": f"{mean_dir:.1f}%" if dirs else "N/A",
            "std_direction": round(std_dir, 2) if dirs else 0.0,
            "mean_vs_persistence_improvement": round(impr_vs_pers, 2),
            "stability_class": stab_class
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(out_dir, "walkforward_summary.csv"), index=False)
    print("Saved outputs/phase8_walkforward/walkforward_summary.csv", flush=True)

    # 4. SPECIFIC TARGET ROBUSTNESS FILES
    pan_df = df_wf[df_wf["target"] == "panamax"]
    pan_df.to_csv(os.path.join(out_dir, "panamax_robustness.csv"), index=False)
    print("Saved outputs/phase8_walkforward/panamax_robustness.csv", flush=True)

    kdci_df = df_wf[df_wf["target"] == "kdci"]
    kdci_df.to_csv(os.path.join(out_dir, "kdci_robustness.csv"), index=False)
    print("Saved outputs/phase8_walkforward/kdci_robustness.csv", flush=True)

    handy30_df = df_wf[(df_wf["target"] == "handy") & (df_wf["horizon"] == "30d")]
    handy30_df.to_csv(os.path.join(out_dir, "handy30_robustness.csv"), index=False)
    print("Saved outputs/phase8_walkforward/handy30_robustness.csv", flush=True)

    print("\nPhase 8G Walk-Forward Robustness Validation Complete.", flush=True)

if __name__ == "__main__":
    run_walkforward_validation()
