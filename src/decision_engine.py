"""
SIH26006 — Phase 10: Procurement & Vessel Chartering Decision Engine

Features:
1. Multi-Factor Decision Rules: BUY NOW, WAIT, FLEXIBLE / INDEX-LINKED.
2. Point-in-Time No-Lookahead Audit Architecture: Decision logic at date t relies strictly on past/present inputs.
3. Separate Post-Hoc Evaluation Labels: Realized future rate y_(t+30) is logged strictly after recommendation for performance audit.
4. Indicative Cost & Risk Score (0-100): Combines normalized Freight Index Score, Historical Port Detention Risk, and Disruption Risk Exposure.
5. Feasibility Integration: Uses Dataset B 5 MVP ports (Paradip, Visakhapatnam, Gangavaram, Dhamra, Haldia) with candidate-first vessel class validation.
6. Structured Audit Log (JSON & CSV): Every recommendation includes complete rationale and rejected vessel classes with reasons.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from feasibility_engine import FeasibilityEngine

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMOTED_PAIRS = {
    ('cape', '7d'): {
        'optimal_tau': 0.01,
        'p10': -4745.0,
        'p90': 7321.0,
        'k': 30,
        'alpha': 1.0,
        'historical_precision': 63.3
    },
    ('kdci', '7d'): {
        'optimal_tau': 0.01,
        'p10': -2190.0,
        'p90': 2258.0,
        'k': 10,
        'alpha': 10.0,
        'historical_precision': 95.5
    },
    ('supramax', '7d'): {
        'optimal_tau': 0.01,
        'p10': -1315.0,
        'p90': 1380.0,
        'k': 50,
        'alpha': 10.0,
        'historical_precision': 100.0
    },
    ('supramax', '14d'): {
        'optimal_tau': 0.01,
        'p10': -2200.0,
        'p90': 1965.0,
        'k': 50,
        'alpha': 1.0,
        'historical_precision': 89.8
    },
    ('supramax', '30d'): {
        'optimal_tau': 0.01,
        'p10': -2326.0,
        'p90': 3124.0,
        'k': 20,
        'alpha': 10.0,
        'historical_precision': 100.0
    },
}

class ProcurementDecisionEngine:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = load_config(config_path)
        self.outputs_dir = self.config["paths"]["outputs_dir"]
        self.feasibility_engine = FeasibilityEngine(self.config["paths"]["dataset_b"])
        self.promoted_pairs = PROMOTED_PAIRS
        
        # Load Model Comparison to get Test RMSE (uncertainty sigma_30d) for top model
        comp_path = os.path.join(self.outputs_dir, "model_comparison.csv")
        if os.path.exists(comp_path):
            self.model_comp = pd.read_csv(comp_path)
        else:
            self.model_comp = pd.DataFrame()

    def get_forecast_uncertainty(self, freight_class, horizon="30d"):
        """Get forecast uncertainty bounds derived from validation residuals or RMSE."""
        pair_key = (freight_class.lower(), horizon.lower())
        if pair_key in self.promoted_pairs:
            cfg = self.promoted_pairs[pair_key]
            return {"p10": cfg["p10"], "p90": cfg["p90"], "is_promoted": True}
        
        if self.model_comp.empty:
            return {"p10": -1500.0, "p90": 1500.0, "is_promoted": False}
        subset = self.model_comp[(self.model_comp["freight_class"] == freight_class) & 
                                 (self.model_comp["horizon"] == horizon)]
        if not subset.empty:
            ridge_row = subset[subset["model"] == "Ridge"]
            rmse = float(ridge_row.iloc[0]["RMSE"]) if not ridge_row.empty else float(subset["RMSE"].min())
            return {"p10": -rmse, "p90": rmse, "is_promoted": False}
        return {"p10": -1500.0, "p90": 1500.0, "is_promoted": False}

    def evaluate_decision(self, decision_date_str, freight_class, current_rate_y0, forecast_30d_yhat=None,
                          gdelt_burst_active=False, weather_disruption_active=False, cargo_volume_mt=55000, destination_port="Paradip",
                          realized_30d_y30=None, horizon="7d", forecast_delta=None):
        """
        Generates a point-in-time recommendation at date t.
        STRICT B3 ARCHITECTURE:
          1. Promoted in registry? -> NO -> Fallback (FLEXIBLE / INDEX-LINKED)
          2. Inside uncertainty [P10, P90]? -> YES -> FLEXIBLE / INDEX-LINKED
          3. Threshold exceeded? -> NO -> FLEXIBLE, + -> BUY NOW, - -> WAIT
        STRICT NO-LOOKAHEAD: Uses ONLY inputs available at or before date t.
        """
        # 1. Feasibility Validation at Decision Time
        feasible_vessels, rejected_vessels, port_risk_score = self.feasibility_engine.validate_feasibility(
            cargo_volume_mt, destination_port
        )

        if not feasible_vessels:
            return {
                "decision_date_t": decision_date_str,
                "recommendation": "REJECT SHIPMENT / ALTERNATIVE PORT",
                "recommended_vessel_class": None,
                "audit_rationale": f"No feasible vessel class available for cargo volume {cargo_volume_mt} MT at {destination_port}. All candidates rejected."
            }

        # Select primary vessel class
        vessel_priority = ["SUPRA", "PANA", "HANDY", "CAPE"]
        rec_vessel = next((v for v in vessel_priority if v in feasible_vessels), feasible_vessels[0])

        # 2. Forecasting Signals at Decision Time
        if forecast_delta is not None:
            delta_val = float(forecast_delta)
            yhat = current_rate_y0 + delta_val
        elif forecast_30d_yhat is not None:
            yhat = float(forecast_30d_yhat)
            delta_val = yhat - current_rate_y0
        else:
            delta_val = 0.0
            yhat = current_rate_y0

        pair_key = (freight_class.lower(), horizon.lower())
        unc_info = self.get_forecast_uncertainty(freight_class, horizon)
        p10 = unc_info["p10"]
        p90 = unc_info["p90"]
        is_promoted = unc_info["is_promoted"]

        # 3. Disruption Risk Scores at Decision Time
        gdelt_risk = 75.0 if gdelt_burst_active else 10.0
        weather_risk = 75.0 if weather_disruption_active else 10.0
        disruption_risk_score = max(gdelt_risk, weather_risk)

        # 4. Financial Indicative Cost & Risk Score (0-100)
        pct_delta = delta_val / (current_rate_y0 + 1e-8)
        freight_score = np.clip(50.0 + (pct_delta * 200.0), 5.0, 95.0)
        indicative_cost_risk_score = (0.50 * freight_score) + (0.25 * port_risk_score) + (0.25 * disruption_risk_score)

        # 5. B3 Decision Rules (Selective Registry + P10/P90 Uncertainty Gate + Optimal Threshold)
        recommendation = "FLEXIBLE / INDEX-LINKED"
        rationale = ""
        gate_status = "UNKNOWN"

        if not is_promoted:
            # Fallback for unpromoted pairs (e.g. 1d pairs, or non-viable horizons)
            gate_status = "FALLBACK_UNPROMOTED"
            recommendation = "FLEXIBLE / INDEX-LINKED"
            rationale = (f"Pair ({freight_class} {horizon}) is not in the high-conviction promoted registry. "
                         f"Naive baseline is superior; defaulting to index-linked floating rate contract.")
        else:
            cfg = self.promoted_pairs[pair_key]
            tau = cfg["optimal_tau"]
            hist_prec = cfg["historical_precision"]

            # Uncertainty Gate: is delta inside [P10, P90]?
            if p10 <= delta_val <= p90:
                gate_status = "INSIDE_UNCERTAINTY"
                recommendation = "FLEXIBLE / INDEX-LINKED"
                rationale = (f"Predicted rate move (${delta_val:+.1f}/day, {pct_delta*100:+.1f}%) falls inside "
                             f"empirical 80% validation noise band [{p10:+.0f}, {p90:+.0f} $/day]. "
                             f"Uncertainty too high for directional commitment; maintain floating index terms.")
            elif delta_val > p90 and pct_delta > tau:
                gate_status = "CONFIDENT_BUY"
                recommendation = "BUY NOW"
                rationale = (f"Strong upward Δ-forecast (+${delta_val:.1f}/day, +{pct_delta*100:.1f}%) clears "
                             f"P90 uncertainty (+${p90:.0f}) and optimal threshold ±{tau*100:.0f}%. "
                             f"Historical edge precision: {hist_prec:.1f}%. Lock forward charter rate now.")
            elif delta_val < p10 and pct_delta < -tau:
                gate_status = "CONFIDENT_WAIT"
                recommendation = "WAIT"
                rationale = (f"Strong downward Δ-forecast (${delta_val:.1f}/day, {pct_delta*100:.1f}%) clears "
                             f"P10 uncertainty (${p10:.0f}) and optimal threshold ±{tau*100:.0f}%. "
                             f"Historical edge precision: {hist_prec:.1f}%. Delay procurement to float on falling spot rates.")
            else:
                gate_status = "THRESHOLD_NOT_MET"
                recommendation = "FLEXIBLE / INDEX-LINKED"
                rationale = (f"Predicted move (${delta_val:+.1f}/day) does not satisfy optimal decision threshold (±{tau*100:.0f}%). "
                             f"Maintain flexible chartering terms.")

        # 6. Build Structured Audit Trace
        audit_trace = {
            "decision_date_t": decision_date_str,
            "cargo_request": {
                "cargo_type": "Thermal Coal",
                "cargo_volume_mt": cargo_volume_mt,
                "destination_port": destination_port
            },
            "decision_time_inputs": {
                "freight_class": freight_class,
                "horizon": horizon,
                "is_promoted": bool(is_promoted),
                "current_rate_y0": round(current_rate_y0, 2),
                "forecast_yhat": round(yhat, 2),
                "forecast_trend_delta": round(delta_val, 2),
                "forecast_pct_delta": round(pct_delta * 100, 2),
                "uncertainty_p10": round(p10, 2),
                "uncertainty_p90": round(p90, 2),
                "gate_status": gate_status,
                "gdelt_event_burst_active": bool(gdelt_burst_active),
                "weather_disruption_active": bool(weather_disruption_active),
                "historical_port_pbdt_risk_score": round(port_risk_score, 1),
                "disruption_risk_score": round(disruption_risk_score, 1)
            },
            "feasibility_assessment": {
                "feasible_vessel_classes": feasible_vessels,
                "rejected_vessel_classes": rejected_vessels
            },
            "financial_indicative_scores": {
                "indicative_freight_score": round(freight_score, 1),
                "indicative_delay_score": round(port_risk_score, 1),
                "indicative_disruption_score": round(disruption_risk_score, 1),
                "combined_indicative_cost_risk_score": round(indicative_cost_risk_score, 1)
            },
            "decision_output": {
                "recommendation": recommendation,
                "recommended_vessel_class": rec_vessel,
                "audit_rationale": rationale
            }
        }

        # 7. Post-Hoc Evaluation Labels (Strictly separated)
        realized_rate = realized_30d_y30
        if realized_rate is not None and not np.isnan(realized_rate):
            realized_return_pct = (realized_rate - current_rate_y0) / current_rate_y0 * 100.0
            actual_delta = realized_rate - current_rate_y0
            
            # Evaluate outcome and economic PnL ($/day saved)
            if recommendation == "BUY NOW":
                pnl_dollars = actual_delta
                eval_str = (f"SUCCESS (BUY NOW avoided +{realized_return_pct:.2f}% price rise, saved +${pnl_dollars:.0f}/day)"
                            if actual_delta > 0 else
                            f"TIMING_PENALTY (Market dropped {realized_return_pct:.2f}%, penalty -${abs(pnl_dollars):.0f}/day)")
            elif recommendation == "WAIT":
                pnl_dollars = -actual_delta
                eval_str = (f"SUCCESS (WAIT captured {realized_return_pct:.2f}% price drop, saved +${pnl_dollars:.0f}/day)"
                            if actual_delta < 0 else
                            f"TIMING_PENALTY (Market rose +{realized_return_pct:.2f}%, missed saving -${abs(pnl_dollars):.0f}/day)")
            else:
                pnl_dollars = 0.0
                eval_str = f"NEUTRAL_FLOATING (Index-linked contract adjusted with market {realized_return_pct:+.2f}%)"

            audit_trace["post_hoc_evaluation_labels"] = {
                "realized_future_rate": round(realized_rate, 2),
                "realized_return_pct": round(realized_return_pct, 2),
                "realized_delta": round(actual_delta, 2),
                "economic_pnl_dollars_per_day": round(pnl_dollars, 2),
                "decision_success_evaluation": eval_str
            }

        return audit_trace

    def run_historical_decision_simulation(self):
        """Runs the decision engine across test period dates using B3 promoted registry and Δ forecasts."""
        model_df = pd.read_csv(os.path.join(self.outputs_dir, "modeling_dataset.csv"))
        model_df["date"] = pd.to_datetime(model_df["date"])
        model_df = model_df.sort_values("date").reset_index(drop=True)

        pred_file = os.path.join(self.outputs_dir, "delta_forecast", "b2_test_predictions.csv")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Missing {pred_file}. Run B2 delta forecasting first.")

        df_preds = pd.read_csv(pred_file)
        df_preds["date"] = pd.to_datetime(df_preds["date"])

        # Merge with exogenous disruption variables from modeling dataset
        risk_cols = [c for c in ["date", "gdelt_event_count", "cyclone_dist_1"] if c in model_df.columns]
        merged = df_preds.merge(model_df[risk_cols], on="date", how="left")

        all_audits = []
        csv_rows = []

        print(f"Phase 10 / B3: Running historical decision simulation across {len(merged)} prediction instances...")

        for _, row in merged.iterrows():
            d_str = row["date"].strftime("%Y-%m-%d")
            asset = str(row["asset"]).lower()
            horizon = str(row["horizon"]).lower()
            y0 = float(row["y_base"])
            y_lvl_pred = float(row["delta_pred"])
            delta_val = y_lvl_pred - y0
            y_realized = float(row["y_true"]) if pd.notna(row["y_true"]) else None

            gdelt_burst = bool(row.get("gdelt_event_count", 0) > 50)
            weather_disruption = bool(row.get("cyclone_dist_1", 999) < 500)

            # Map asset to cargo and port for feasibility test
            cargo_vol = 55000 if asset == "supramax" else (75000 if asset == "panamax" else 150000)
            dest_port = "Paradip" if asset in ["supramax", "panamax"] else "Visakhapatnam"

            audit = self.evaluate_decision(
                decision_date_str=d_str,
                freight_class=asset,
                current_rate_y0=y0,
                forecast_delta=delta_val,
                gdelt_burst_active=gdelt_burst,
                weather_disruption_active=weather_disruption,
                cargo_volume_mt=cargo_vol,
                destination_port=dest_port,
                realized_30d_y30=y_realized,
                horizon=horizon
            )

            all_audits.append(audit)

            # Flatten row for CSV deliverable
            dt_in = audit["decision_time_inputs"]
            d_out = audit["decision_output"]
            scores = audit["financial_indicative_scores"]
            post = audit.get("post_hoc_evaluation_labels", {})

            csv_rows.append({
                "date": d_str,
                "freight_class": dt_in["freight_class"],
                "horizon": dt_in["horizon"],
                "is_promoted": dt_in["is_promoted"],
                "gate_status": dt_in["gate_status"],
                "destination_port": audit["cargo_request"]["destination_port"],
                "current_rate_y0": dt_in["current_rate_y0"],
                "forecast_yhat": dt_in["forecast_yhat"],
                "forecast_trend_delta": dt_in["forecast_trend_delta"],
                "forecast_pct_delta": dt_in["forecast_pct_delta"],
                "uncertainty_p10": dt_in["uncertainty_p10"],
                "uncertainty_p90": dt_in["uncertainty_p90"],
                "indicative_cost_risk_score": scores["combined_indicative_cost_risk_score"],
                "feasible_vessel_classes": ", ".join(audit["feasibility_assessment"]["feasible_vessel_classes"]),
                "recommendation": d_out["recommendation"],
                "recommended_vessel_class": d_out["recommended_vessel_class"],
                "audit_rationale": d_out["audit_rationale"],
                "realized_future_rate": post.get("realized_future_rate", np.nan),
                "realized_return_pct": post.get("realized_return_pct", np.nan),
                "economic_pnl_dollars_per_day": post.get("economic_pnl_dollars_per_day", 0.0),
                "decision_success_evaluation": post.get("decision_success_evaluation", "")
            })

        # Save deliverables
        json_path = os.path.join(self.outputs_dir, "decision_engine_audit_log.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_audits, f, indent=2)
        print(f"Saved: {json_path} ({len(all_audits)} JSON audit logs)")

        df_csv = pd.DataFrame(csv_rows)
        csv_path = os.path.join(self.outputs_dir, "decision_engine_recommendations.csv")
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(df_csv)} recommendation rows)")

        # Summary breakdown
        print("\nDecision Recommendations Distribution:")
        print(df_csv.groupby(["is_promoted", "recommendation"]).size())

        promoted_df = df_csv[df_csv["is_promoted"]]
        total_pnl = promoted_df["economic_pnl_dollars_per_day"].sum()
        print(f"\nTotal Economic PnL across Promoted Pairs: ${total_pnl:+,.0f} ($/day-sum)")

        return df_csv

if __name__ == "__main__":
    engine = ProcurementDecisionEngine()
    engine.run_historical_decision_simulation()
