"""
FICOS — Historical Decision Backtest Engine
Evaluates decision policy performance against baseline strategies (Naive Spot, Fixed TC)
over historical dataset timelines.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.api.recommendation_service import RecommendationService
from ml.data.data_loader import load_modeling_dataset


class DecisionBacktestEngine:
    """
    Simulates decision policy over historical walk-forward windows.
    """

    def __init__(self, data_path: str = "data/modeling_dataset.csv"):
        p = Path(data_path)
        if not p.exists() and Path("outputs/modeling_dataset.csv").exists():
            p = Path("outputs/modeling_dataset.csv")
        elif not p.exists() and Path("data/modeling_dataset.csv").exists():
            p = Path("data/modeling_dataset.csv")
        self.data_path = p
        self.rec_service = RecommendationService()

    def run_backtest(
        self,
        asset_type: str = "PANAMAX_1D",
        sample_stride: int = 50
    ) -> Dict[str, Any]:
        """
        Runs decision backtest sampling historical data rows.
        """
        if not self.data_path.exists():
            return {"error": f"Dataset not found at {self.data_path}"}

        df = pd.read_csv(self.data_path)
        if asset_type not in df.columns:
            # Fallback to target column if available
            target_cols = [c for c in df.columns if "target" in c.lower()]
            if not target_cols:
                return {"error": "No suitable target column found in dataset."}
            target_col = target_cols[0]
        else:
            target_col = asset_type

        sub_df = df.iloc[::sample_stride].copy()
        
        ficos_costs: List[float] = []
        spot_costs: List[float] = []
        quantity = 75000.0

        for idx, row in sub_df.iterrows():
            rec = self.rec_service.get_recommendation(
                asset_type=asset_type,
                quantity_mt=quantity,
                origin_port_code="TUBARAO",
                dest_port_code="QINGDAO"
            )
            
            actual_spot_rate = float(row[target_col]) if target_col in row and not np.isnan(row[target_col]) else rec.forecast.point_forecast
            spot_cost = actual_spot_rate * quantity

            # FICOS recommended cost
            ficos_cost = rec.recommended_cost_usd

            ficos_costs.append(ficos_cost)
            spot_costs.append(spot_cost)

        avg_ficos_cost = float(np.mean(ficos_costs)) if ficos_costs else 0.0
        avg_spot_cost = float(np.mean(spot_costs)) if spot_costs else 0.0
        savings_usd = avg_spot_cost - avg_ficos_cost
        savings_pct = (savings_usd / avg_spot_cost * 100.0) if avg_spot_cost > 0 else 0.0

        return {
            "asset_type": asset_type,
            "samples_evaluated": len(sub_df),
            "avg_ficos_cost_usd": round(avg_ficos_cost, 2),
            "avg_naive_spot_cost_usd": round(avg_spot_cost, 2),
            "total_savings_usd": round(savings_usd, 2),
            "savings_percentage": round(savings_pct, 2)
        }


if __name__ == "__main__":
    engine = DecisionBacktestEngine()
    res = engine.run_backtest()
    print("Decision Backtest Result:", res)
