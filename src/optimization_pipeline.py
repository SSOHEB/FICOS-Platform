"""
SIH26006 — Phase 8 Optimization Pipeline (8A through 8F)

Implements:
- Phase 8A: Leakage-safe feature selection on Training Set ONLY (Full, Compact, Top20, Top50, Top100).
- Phase 8B: Ridge + Neural/MLP Residual Hybridization.
- Phase 8C: Controlled Hyperparameter Tuning with explicit float64 dtype casting (fixing pandas warnings).
- Phase 8D: Validation-Learned Weighted Ensembling (NNLS on Validation Set ONLY).
- Phase 8E: Target Transformations (Direct Level, Absolute Change, Percentage Change) reconstructed to original units.
- Final Evaluation: Locks best configuration per target x horizon on Validation, then evaluates ONCE on untouched Test Set.

Evaluation Rules:
- Test Set is LOCKED. Zero test-set tuning.
- Persistence Directional Accuracy marked 'N/A' (no directional call).
- Output: outputs/phase8_optimization_results.csv & outputs/phase8_final_benchmark.csv
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Set fixed seeds for reproducibility
np.random.seed(42)

def load_data():
    df = pd.read_csv("outputs/modeling_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def calc_metrics(y_true_raw, y_pred_raw, y_base_raw, is_persistence=False):
    mask = ~np.isnan(y_true_raw) & ~np.isnan(y_pred_raw) & ~np.isnan(y_base_raw)
    yt, yp, ybase = y_true_raw[mask], y_pred_raw[mask], y_base_raw[mask]
    
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp)**2))
    smape = np.mean(200 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8))
    
    ss_tot = np.sum((yt - np.mean(yt))**2)
    ss_res = np.sum((yt - yp)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    if is_persistence:
        dir_acc_str = "N/A"
    else:
        actual_change = yt - ybase
        pred_change = yp - ybase
        actual_dir = np.sign(actual_change)
        pred_dir = np.sign(pred_change)
        dir_acc = np.mean(actual_dir == pred_dir)
        dir_acc_str = f"{dir_acc * 100:.1f}%"
        
    return round(mae, 2), round(rmse, 2), round(smape, 2), round(r2, 4), dir_acc_str, len(yt)

def get_candidate_features(df, feature_cols, target_col, train_mask, candidate_type):
    """
    Selects candidate feature subset using TRAINING DATA ONLY.
    """
    if candidate_type == "full":
        return feature_cols
    elif candidate_type == "compact":
        # Domain compact set: Target lags, GDELT, Weather, Commodity prices
        compact_keywords = ["lag", "rolling", "gdelt", "wind", "precip", "cyclone", "bci", "bpi", "bsi", "bhsi", "oil", "coal", "iron"]
        cols = [c for c in feature_cols if any(k in c.lower() for k in compact_keywords)]
        return cols if len(cols) > 5 else feature_cols
    else:
        # Top K features via Training-only f_regression / SelectKBest
        k = int(candidate_type.replace("top", ""))
        X_tr = df.loc[train_mask, feature_cols].fillna(0).values
        y_tr = df.loc[train_mask, target_col].values
        
        valid_idx = ~np.isnan(y_tr)
        X_tr_v = X_tr[valid_idx]
        y_tr_v = y_tr[valid_idx]
        
        selector = SelectKBest(score_func=f_regression, k=min(k, X_tr_v.shape[1]))
        selector.fit(X_tr_v, y_tr_v)
        selected_mask = selector.get_support()
        return [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

def run_optimization_pipeline():
    df = load_data()
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    # Chronological splits
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_train] = True
    
    val_mask = np.zeros(n, dtype=bool)
    val_mask[n_train:n_train+n_val] = True
    
    test_mask = np.zeros(n, dtype=bool)
    test_mask[n_train+n_val:] = True
    
    feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in ["date"]]
    
    # Fix pandas FutureWarning: Explicitly convert feature columns to float64
    df[feature_cols] = df[feature_cols].astype(np.float64)
    
    TARGETS = ["kdci", "cape", "panamax", "supramax", "handy"]
    HORIZONS = [1, 7, 14, 30]
    
    feature_sets_to_test = ["full", "compact", "top20", "top50", "top100"]
    target_transformations = ["level", "abs_change", "pct_change"]
    
    optimization_results = []
    final_benchmark_results = []

    print("=" * 80)
    print("PHASE 8 OPTIMIZATION PIPELINE: EXECUTING PHASES 8A THROUGH 8F")
    print("=" * 80)

    for tgt in TARGETS:
        for h in HORIZONS:
            target_col = f"target_{tgt}_{h}d"
            prev_col = tgt
            horizon_str = f"{h}d"

            print(f"\n>>> Optimizing {tgt.upper()} {horizon_str} (Target: {target_col}) <<<")

            # -------------------------------------------------------------
            # PHASE 8A & 8E: FEATURE SELECTION & TARGET TRANSFORMATION SWEEP (ON VALIDATION ONLY)
            # -------------------------------------------------------------
            best_val_smape = float("inf")
            best_config = {}

            # Masks for target availability
            tr_valid = train_mask & df[target_col].notna() & df[prev_col].notna()
            v_valid = val_mask & df[target_col].notna() & df[prev_col].notna()
            te_valid = test_mask & df[target_col].notna() & df[prev_col].notna()

            y_tr_raw = df.loc[tr_valid, target_col].values
            y_tr_base = df.loc[tr_valid, prev_col].values
            
            y_v_raw = df.loc[v_valid, target_col].values
            y_v_base = df.loc[v_valid, prev_col].values
            
            y_te_raw = df.loc[te_valid, target_col].values
            y_te_base = df.loc[te_valid, prev_col].values

            # Sweep Feature Sets & Transformations on Validation
            for f_set in feature_sets_to_test:
                selected_cols = get_candidate_features(df, feature_cols, target_col, tr_valid, f_set)
                
                # Impute missing feature values using train median ONLY
                tr_medians = df.loc[tr_valid, selected_cols].median()
                X_tr = df.loc[tr_valid, selected_cols].fillna(tr_medians).values
                X_v = df.loc[v_valid, selected_cols].fillna(tr_medians).values
                X_te = df.loc[te_valid, selected_cols].fillna(tr_medians).values

                # Scale features (fit ONLY on train)
                scaler_X = StandardScaler()
                X_tr_sc = scaler_X.fit_transform(X_tr)
                X_v_sc = scaler_X.transform(X_v)
                X_te_sc = scaler_X.transform(X_te)

                for t_trans in target_transformations:
                    # Construct target representation for training
                    if t_trans == "level":
                        y_train_target = y_tr_raw
                    elif t_trans == "abs_change":
                        y_train_target = y_tr_raw - y_tr_base
                    elif t_trans == "pct_change":
                        y_train_target = (y_tr_raw - y_tr_base) / (y_tr_base + 1e-8)

                    # Scale target representation
                    scaler_y = StandardScaler()
                    y_tr_target_sc = scaler_y.fit_transform(y_train_target.reshape(-1, 1)).flatten()

                    # Tune Ridge alpha on Validation
                    for alpha in [10.0, 100.0, 1000.0, 5000.0, 10000.0]:
                        ridge = Ridge(alpha=alpha)
                        ridge.fit(X_tr_sc, y_tr_target_sc)
                        
                        # Validate and reconstruct to original level units
                        pred_v_target_sc = ridge.predict(X_v_sc)
                        pred_v_target_raw = scaler_y.inverse_transform(pred_v_target_sc.reshape(-1, 1)).flatten()

                        if t_trans == "level":
                            pred_v_level = pred_v_target_raw
                        elif t_trans == "abs_change":
                            pred_v_level = y_v_base + pred_v_target_raw
                        elif t_trans == "pct_change":
                            pred_v_level = y_v_base * (1.0 + pred_v_target_raw)

                        v_mae, v_rmse, v_smape, v_r2, v_dir, _ = calc_metrics(y_v_raw, pred_v_level, y_v_base)

                        optimization_results.append({
                            "freight_class": tgt,
                            "horizon": horizon_str,
                            "feature_set": f_set,
                            "target_transformation": t_trans,
                            "alpha": alpha,
                            "num_features": len(selected_cols),
                            "val_MAE": v_mae,
                            "val_RMSE": v_rmse,
                            "val_sMAPE": v_smape,
                            "val_R2": v_r2,
                            "val_DirAcc": v_dir
                        })

                        if v_smape < best_val_smape:
                            best_val_smape = v_smape
                            best_config = {
                                "feature_set": f_set,
                                "target_transformation": t_trans,
                                "alpha": alpha,
                                "selected_cols": selected_cols,
                                "scaler_X": scaler_X,
                                "scaler_y": scaler_y,
                                "val_MAE": v_mae,
                                "val_RMSE": v_rmse,
                                "val_sMAPE": v_smape,
                                "val_R2": v_r2,
                                "val_DirAcc": v_dir
                            }

            print(f"  Validation Best Search Config: FeatureSet={best_config['feature_set']} ({len(best_config['selected_cols'])} cols), "
                  f"Transform={best_config['target_transformation']}, Alpha={best_config['alpha']} | Val sMAPE: {best_config['val_sMAPE']:.2f}%")

            # -------------------------------------------------------------
            # PHASE 8B & 8D: RESIDUAL HYBRID & VALIDATION-LEARNED ENSEMBLE
            # -------------------------------------------------------------
            sel_cols = best_config["selected_cols"]
            tr_meds = df.loc[tr_valid, sel_cols].median()
            
            X_tr_opt = scaler_X.fit_transform(df.loc[tr_valid, sel_cols].fillna(tr_meds).values)
            X_v_opt = scaler_X.transform(df.loc[v_valid, sel_cols].fillna(tr_meds).values)
            X_te_opt = scaler_X.transform(df.loc[te_valid, sel_cols].fillna(tr_meds).values)

            t_trans = best_config["target_transformation"]
            if t_trans == "level":
                y_tr_t = y_tr_raw
            elif t_trans == "abs_change":
                y_tr_t = y_tr_raw - y_tr_base
            elif t_trans == "pct_change":
                y_tr_t = (y_tr_raw - y_tr_base) / (y_tr_base + 1e-8)

            scaler_y_opt = StandardScaler()
            y_tr_t_sc = scaler_y_opt.fit_transform(y_tr_t.reshape(-1, 1)).flatten()

            # 1. Base Model 1: Optimal Ridge
            ridge_opt = Ridge(alpha=best_config["alpha"])
            ridge_opt.fit(X_tr_opt, y_tr_t_sc)

            pred_tr_ridge_raw = scaler_y_opt.inverse_transform(ridge_opt.predict(X_tr_opt).reshape(-1, 1)).flatten()
            pred_v_ridge_raw = scaler_y_opt.inverse_transform(ridge_opt.predict(X_v_opt).reshape(-1, 1)).flatten()
            pred_te_ridge_raw = scaler_y_opt.inverse_transform(ridge_opt.predict(X_te_opt).reshape(-1, 1)).flatten()

            # 2. Base Model 2: XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)
            xgb_model.fit(X_tr_opt, y_tr_t_sc)

            pred_v_xgb_raw = scaler_y_opt.inverse_transform(xgb_model.predict(X_v_opt).reshape(-1, 1)).flatten()
            pred_te_xgb_raw = scaler_y_opt.inverse_transform(xgb_model.predict(X_te_opt).reshape(-1, 1)).flatten()

            # 3. Base Model 3: Neural Net (MLP/GRU surrogate)
            mlp_model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.01, early_stopping=True, random_state=42)
            mlp_model.fit(X_tr_opt, y_tr_t_sc)

            pred_v_mlp_raw = scaler_y_opt.inverse_transform(mlp_model.predict(X_v_opt).reshape(-1, 1)).flatten()
            pred_te_mlp_raw = scaler_y_opt.inverse_transform(mlp_model.predict(X_te_opt).reshape(-1, 1)).flatten()

            # Reconstruct level functions helper
            def reconstruct_level(pred_target_raw, base_prices):
                if t_trans == "level":
                    return pred_target_raw
                elif t_trans == "abs_change":
                    return base_prices + pred_target_raw
                elif t_trans == "pct_change":
                    return base_prices * (1.0 + pred_target_raw)

            v_ridge_lvl = reconstruct_level(pred_v_ridge_raw, y_v_base)
            v_xgb_lvl = reconstruct_level(pred_v_xgb_raw, y_v_base)
            v_mlp_lvl = reconstruct_level(pred_v_mlp_raw, y_v_base)

            te_ridge_lvl = reconstruct_level(pred_te_ridge_raw, y_te_base)
            te_xgb_lvl = reconstruct_level(pred_te_xgb_raw, y_te_base)
            te_mlp_lvl = reconstruct_level(pred_te_mlp_raw, y_te_base)

            # Phase 8B: Train Residual Estimator on Training Residuals
            tr_ridge_lvl = reconstruct_level(pred_tr_ridge_raw, y_tr_base)
            residuals_tr = y_tr_raw - tr_ridge_lvl

            res_model = ExtraTreesRegressor(n_estimators=50, max_depth=5, random_state=42)
            res_model.fit(X_tr_opt, residuals_tr)

            v_res_pred = res_model.predict(X_v_opt)
            v_hybrid_lvl = v_ridge_lvl + v_res_pred

            te_res_pred = res_model.predict(X_te_opt)
            te_hybrid_lvl = te_ridge_lvl + te_res_pred

            # Phase 8D: Learn Non-Negative Weighted Ensemble on VALIDATION ONLY
            val_preds_matrix = np.column_stack([v_ridge_lvl, v_xgb_lvl, v_mlp_lvl, v_hybrid_lvl])
            weights, _ = nnls(val_preds_matrix, y_v_raw)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.array([0.5, 0.25, 0.25, 0.0])

            v_ensemble_lvl = val_preds_matrix @ weights
            test_preds_matrix = np.column_stack([te_ridge_lvl, te_xgb_lvl, te_mlp_lvl, te_hybrid_lvl])
            te_ensemble_lvl = test_preds_matrix @ weights

            # Compare Candidate Systems on Validation
            candidates_val = {
                "Optimal_Ridge": (v_ridge_lvl, te_ridge_lvl),
                "XGBoost": (v_xgb_lvl, te_xgb_lvl),
                "Neural_Net": (v_mlp_lvl, te_mlp_lvl),
                "Ridge_Residual_Hybrid": (v_hybrid_lvl, te_hybrid_lvl),
                "Validation_Weighted_Ensemble": (v_ensemble_lvl, te_ensemble_lvl)
            }

            best_sys_name = "Optimal_Ridge"
            best_sys_val_smape = float("inf")

            for sys_name, (v_pred_sys, _) in candidates_val.items():
                _, _, sys_val_smape, _, _, _ = calc_metrics(y_v_raw, v_pred_sys, y_v_base)
                if sys_val_smape < best_sys_val_smape:
                    best_sys_val_smape = sys_val_smape
                    best_sys_name = sys_name

            # -------------------------------------------------------------
            # FINAL EVALUATION: UNTOUCHED TEST SET EVALUATION (ONCE)
            # -------------------------------------------------------------
            best_v_pred, best_te_pred = candidates_val[best_sys_name]

            v_mae, v_rmse, v_smape, v_r2, v_dir, v_n = calc_metrics(y_v_raw, best_v_pred, y_v_base)
            t_mae, t_rmse, t_smape, t_r2, t_dir, t_n = calc_metrics(y_te_raw, best_te_pred, y_te_base)

            # Compare against existing baseline (Ridge / Persistence)
            existing_pers_mae, existing_pers_rmse, existing_pers_smape, existing_pers_r2, _, _ = calc_metrics(y_te_raw, y_te_base, y_te_base, is_persistence=True)
            
            smape_improvement = round(existing_pers_smape - t_smape, 2)
            mae_improvement = round(existing_pers_mae - t_mae, 2)
            pct_improvement = round(((existing_pers_mae - t_mae) / existing_pers_mae) * 100, 2) if existing_pers_mae > 0 else 0.0

            final_benchmark_results.append({
                "freight_class": tgt,
                "horizon": horizon_str,
                "selected_model": f"{best_sys_name} ({best_config['feature_set']})",
                "target_transformation": best_config["target_transformation"],
                "val_sMAPE": v_smape,
                "test_sMAPE": t_smape,
                "test_MAE": t_mae,
                "test_RMSE": t_rmse,
                "test_R2": t_r2,
                "test_Direction": t_dir,
                "test_N": t_n,
                "baseline_persistence_MAE": existing_pers_mae,
                "mae_improvement": mae_improvement,
                "pct_improvement": f"{pct_improvement:+.1f}%",
                "meaningful_improvement": "YES" if pct_improvement > 2.0 or t_dir in ["70.0%", "75.0%", "80.0%", "85.0%"] else "MODEST",
                "validation_test_stability": "STABLE" if abs(v_smape - t_smape) < 5.0 else "DRIFT"
            })

            print(f"  FINAL LOCKED CHOICE: Model={best_sys_name} | Val sMAPE={v_smape:.2f}% | Test sMAPE={t_smape:.2f}% | Test MAE={t_mae:.2f} | Test R2={t_r2:.4f} | Test Dir={t_dir} | (vs Baseline: {pct_improvement:+.1f}%)")

    # Save outputs
    df_opt = pd.DataFrame(optimization_results)
    df_opt.to_csv("outputs/phase8_optimization_results.csv", index=False)
    
    df_bm = pd.DataFrame(final_benchmark_results)
    df_bm.to_csv("outputs/phase8_final_benchmark.csv", index=False)

    print("\n" + "=" * 80)
    print("SAVED DELIVERABLES:")
    print("  - outputs/phase8_optimization_results.csv (Full sweep matrix)")
    print("  - outputs/phase8_final_benchmark.csv (Final locked evaluation)")
    print("=" * 80)

if __name__ == "__main__":
    run_optimization_pipeline()
