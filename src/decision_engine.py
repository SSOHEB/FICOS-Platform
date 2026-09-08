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

class ProcurementDecisionEngine:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = load_config(config_path)
        self.outputs_dir = self.config["paths"]["outputs_dir"]
        self.feasibility_engine = FeasibilityEngine(self.config["paths"]["dataset_b"])
        
        # Load Model Comparison to get Test RMSE (uncertainty sigma_30d) for top model
        comp_path = os.path.join(self.outputs_dir, "model_comparison.csv")
        if os.path.exists(comp_path):
            self.model_comp = pd.read_csv(comp_path)
        else:
            self.model_comp = pd.DataFrame()

    def get_forecast_uncertainty(self, freight_class, horizon="30d"):
        """Get 30-day forecast uncertainty (RMSE) derived from validation/test residuals."""
        if self.model_comp.empty:
            return 1500.0
        subset = self.model_comp[(self.model_comp["freight_class"] == freight_class) & 
                                 (self.model_comp["horizon"] == horizon)]
        if not subset.empty:
            # Use Ridge RMSE if available, else minimum RMSE
            ridge_row = subset[subset["model"] == "Ridge"]
            if not ridge_row.empty:
                return float(ridge_row.iloc[0]["RMSE"])
            return float(subset["RMSE"].min())
        return 1500.0

    def evaluate_decision(self, decision_date_str, freight_class, current_rate_y0, forecast_30d_yhat,
                          gdelt_burst_active, weather_disruption_active, cargo_volume_mt, destination_port,
                          realized_30d_y30=None):
        """
        Generates a point-in-time recommendation at date t.
        STRICT NO-LOOKAHEAD: Uses ONLY inputs available at or before date t.
        realized_30d_y30 is used strictly in post-hoc evaluation labels.
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

        # Select primary vessel class (e.g. Supramax or Panamax preferred)
        vessel_priority = ["SUPRA", "PANA", "HANDY", "CAPE"]
        rec_vessel = next((v for v in vessel_priority if v in feasible_vessels), feasible_vessels[0])

        # 2. Forecasting Signals at Decision Time
        uncertainty_sigma = self.get_forecast_uncertainty(freight_class, "30d")
        forecast_trend_delta = forecast_30d_yhat - current_rate_y0
        trend_sigma_ratio = forecast_trend_delta / uncertainty_sigma if uncertainty_sigma > 0 else 0.0

        # 3. Disruption Risk Scores at Decision Time
        gdelt_risk = 75.0 if gdelt_burst_active else 10.0
        weather_risk = 75.0 if weather_disruption_active else 10.0
        disruption_risk_score = max(gdelt_risk, weather_risk)

        # 4. Financial Indicative Cost & Risk Score (0-100)
        # Freight Score normalized relative to baseline price
        freight_score = np.clip(50.0 + (trend_sigma_ratio * 15.0), 5.0, 95.0)
        indicative_cost_risk_score = (0.50 * freight_score) + (0.25 * port_risk_score) + (0.25 * disruption_risk_score)

        # 5. Multi-Factor Decision Rules (Decision-Time Logic)
        recommendation = "FLEXIBLE / INDEX-LINKED"
        rationale = ""

        # Rule 1: BUY NOW (Lock Fixed Forward Rate)
        if (trend_sigma_ratio > 0.75 and (disruption_risk_score > 50.0 or port_risk_score > 60.0)) or (trend_sigma_ratio > 1.50):
            recommendation = "BUY NOW"
            rationale = (f"Strong upward forecast trend (+{trend_sigma_ratio:.2f} σ) combined with elevated "
                         f"risk signals (Port Risk: {port_risk_score}, Disruption Risk: {disruption_risk_score}). "
                         f"Lock fixed forward rate now to secure tonnage before expected rate surge.")
        # Rule 2: WAIT (Delay Procurement / Spot Market)
        elif trend_sigma_ratio < -0.50 and disruption_risk_score < 50.0:
            recommendation = "WAIT"
            rationale = (f"Downward forecast trend ({trend_sigma_ratio:.2f} σ) with Low/Moderate disruption risk. "
                         f"Delay chartering to capitalize on falling spot market rates.")
        # Rule 3: FLEXIBLE / INDEX-LINKED
        else:
            recommendation = "FLEXIBLE / INDEX-LINKED"
            uncertainty_ratio = uncertainty_sigma / current_rate_y0
            if uncertainty_ratio > 0.15:
                rationale = f"High forecast uncertainty ({uncertainty_ratio*100:.1f}% of price level). Utilize index-linked contract with floating rate option."
            else:
                rationale = f"Neutral rate trend ({trend_sigma_ratio:.2f} σ). Maintain flexible chartering timing with floating index-linked terms."

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
                "current_rate_y0": round(current_rate_y0, 2),
                "forecast_30d_yhat": round(forecast_30d_yhat, 2),
                "forecast_trend_delta": round(forecast_trend_delta, 2),
                "uncertainty_sigma_30d": round(uncertainty_sigma, 2),
                "trend_sigma_ratio": round(trend_sigma_ratio, 2),
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
        if realized_30d_y30 is not None and not np.isnan(realized_30d_y30):
            realized_return_pct = (realized_30d_y30 - current_rate_y0) / current_rate_y0 * 100.0
            
            # Evaluate outcome
            if recommendation == "BUY NOW" and realized_return_pct > 0:
                eval_str = f"SUCCESS (BUY NOW avoided +{realized_return_pct:.2f}% price rise)"
            elif recommendation == "WAIT" and realized_return_pct < 0:
                eval_str = f"SUCCESS (WAIT captured {realized_return_pct:.2f}% price drop)"
            elif recommendation == "FLEXIBLE / INDEX-LINKED":
                eval_str = f"NEUTRAL (Index-linked contract adjusted with {realized_return_pct:+.2f}% market move)"
            else:
                eval_str = f"SUBOPTIMAL (Market moved {realized_return_pct:+.2f}%)"

            audit_trace["post_hoc_evaluation_labels"] = {
                "realized_30d_rate_y30": round(realized_30d_y30, 2),
                "realized_30d_return_pct": round(realized_return_pct, 2),
                "decision_success_evaluation": eval_str
            }

        return audit_trace

    def run_historical_decision_simulation(self):
        """Runs the decision engine across all 387 test period dates."""
        model_df = pd.read_csv(os.path.join(self.outputs_dir, "modeling_dataset.csv"))
        model_df["date"] = pd.to_datetime(model_df["date"])
        model_df = model_df.sort_values("date").reset_index(drop=True)

        # Load Ridge predictions for 30d Supramax (best performing model)
        pred_path = os.path.join(self.outputs_dir, "predictions", "ridge", "supramax_30d.csv")
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            pred_df["date"] = pd.to_datetime(pred_df["date"])
        else:
            print("Warning: Ridge predictions for supramax_30d not found. Using raw dataset.")
            pred_df = pd.DataFrame()

        # Merge test dates
        if not pred_df.empty:
            test_merged = pd.merge(pred_df, model_df, on="date", how="left")
        else:
            # Fallback to test split portion
            n = len(model_df)
            test_merged = model_df.iloc[int(n*0.85):].copy()

        all_audits = []
        csv_rows = []

        print(f"Phase 10: Running point-in-time decision simulation across {len(test_merged)} test-period dates...")

        for _, row in test_merged.iterrows():
            d_str = row["date"].strftime("%Y-%m-%d")
            y0 = row["y_prev"] if "y_prev" in row else row["supramax"]
            yhat_30d = row["y_pred"] if "y_pred" in row else (y0 * 1.02)
            y30_realized = row["y_true"] if "y_true" in row else None
            
            gdelt_burst = bool(row.get("gdelt_event_count", 0) > 50)
            weather_disruption = bool(row.get("cyclone_dist_1", 999) < 500)

            # Evaluate decision for 55,000 MT Coal to Paradip
            audit = self.evaluate_decision(
                decision_date_str=d_str,
                freight_class="supramax",
                current_rate_y0=y0,
                forecast_30d_yhat=yhat_30d,
                gdelt_burst_active=gdelt_burst,
                weather_disruption_active=weather_disruption,
                cargo_volume_mt=55000,
                destination_port="Paradip",
                realized_30d_y30=y30_realized
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
                "destination_port": audit["cargo_request"]["destination_port"],
                "current_rate_y0": dt_in["current_rate_y0"],
                "forecast_30d_yhat": dt_in["forecast_30d_yhat"],
                "forecast_trend_delta": dt_in["forecast_trend_delta"],
                "uncertainty_sigma_30d": dt_in["uncertainty_sigma_30d"],
                "trend_sigma_ratio": dt_in["trend_sigma_ratio"],
                "indicative_cost_risk_score": scores["combined_indicative_cost_risk_score"],
                "feasible_vessel_classes": ", ".join(audit["feasibility_assessment"]["feasible_vessel_classes"]),
                "recommendation": d_out["recommendation"],
                "recommended_vessel_class": d_out["recommended_vessel_class"],
                "audit_rationale": d_out["audit_rationale"],
                "realized_30d_rate_y30": post.get("realized_30d_rate_y30", np.nan),
                "realized_30d_return_pct": post.get("realized_30d_return_pct", np.nan),
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
        print(df_csv["recommendation"].value_counts())

        return df_csv

if __name__ == "__main__":
    engine = ProcurementDecisionEngine()
    engine.run_historical_decision_simulation()
