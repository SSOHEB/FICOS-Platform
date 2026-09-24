"""
FICOS — Regime Analysis
========================
Analyzes forecast performance across macroeconomic and market regimes:
  - Volatility regimes: High Volatility vs Low Volatility (realized 30d rolling vol)
  - Trend regimes: Bull Market vs Bear Market vs Sideways (SMA-50 vs SMA-200 / rolling return)
  - Macro/Structural regimes: Pre-2022 vs Post-2022 / War & Red Sea disruption periods
  - Gated vs Ungated breakdown per regime

Provides empirical answers to:
  1. Does the model only work in specific market regimes?
  2. Does confidence gating help protect against drawdown during turbulent regimes?
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.data_loader import DataLoader
from ml.features.features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FICOS-RegimeAnalysis")


class RegimeAnalyzer:
    """Classifies market periods into economic/statistical regimes and audits model behavior."""

    def __init__(self, df: pd.DataFrame, asset: str = "BDI"):
        self.df = df.copy()
        self.asset = asset
        self._classify_regimes()

    def _classify_regimes(self):
        """Classify daily records into volatility and trend regimes."""
        if self.asset not in self.df.columns:
            target_cols = [c for c in self.df.columns if self.asset in c]
            price_col = target_cols[0] if target_cols else self.df.columns[0]
        else:
            price_col = self.asset

        prices = self.df[price_col].astype(float)
        ret_1d = prices.pct_change()

        # Volatility regime (30-day annualized rolling volatility)
        rolling_vol = ret_1d.rolling(30).std() * np.sqrt(252)
        median_vol = rolling_vol.median()
        self.df["vol_regime"] = np.where(rolling_vol >= median_vol, "HIGH_VOLATILITY", "LOW_VOLATILITY")

        # Trend regime (SMA 50 vs SMA 200 / 60d return)
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        
        conditions = [
            (prices > sma_50) & (sma_50 > sma_200),
            (prices < sma_50) & (sma_50 < sma_200)
        ]
        choices = ["BULL_TREND", "BEAR_TREND"]
        self.df["trend_regime"] = np.select(conditions, choices, default="SIDEWAYS")

        # Structural era
        if "Date" in self.df.columns:
            dates = pd.to_datetime(self.df["Date"])
            self.df["structural_regime"] = np.where(dates >= pd.Timestamp("2022-01-01"), "POST_2022_CRISIS", "PRE_2022_STABLE")
        elif isinstance(self.df.index, pd.DatetimeIndex):
            self.df["structural_regime"] = np.where(self.df.index >= pd.Timestamp("2022-01-01"), "POST_2022_CRISIS", "PRE_2022_STABLE")
        else:
            self.df["structural_regime"] = "ALL"

    def evaluate_predictions_by_regime(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.Series,
        tau: float = 0.0,
        pred_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate directional accuracy, gating rate, and coverage across regimes.
        """
        eval_df = pd.DataFrame({
            "Date": pd.to_datetime(dates).values,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if pred_returns is not None:
            eval_df["pred_return"] = pred_returns
        else:
            eval_df["pred_return"] = y_pred

        # Merge regime tags
        regimes_subset = self.df[["Date", "vol_regime", "trend_regime", "structural_regime"]].drop_duplicates(subset=["Date"])
        eval_df = eval_df.merge(regimes_subset, on="Date", how="left")
        eval_df["correct"] = (np.sign(eval_df["y_pred"]) == np.sign(eval_df["y_true"])).astype(int)

        # Gated filter
        eval_df["gated_pass"] = np.abs(eval_df["pred_return"]) >= tau
        eval_df["gated_correct"] = np.where(
            eval_df["gated_pass"],
            eval_df["correct"],
            np.nan
        )

        results = {}
        for regime_type in ["vol_regime", "trend_regime", "structural_regime"]:
            regime_res = {}
            for val, group in eval_df.groupby(regime_type):
                n_total = len(group)
                n_gated = group["gated_pass"].sum()
                raw_da = group["correct"].mean() if n_total > 0 else 0.0
                gated_da = group["gated_correct"].dropna().mean() if n_gated > 0 else 0.0
                coverage = n_gated / n_total if n_total > 0 else 0.0

                regime_res[str(val)] = {
                    "count": int(n_total),
                    "raw_da": float(raw_da),
                    "gated_da": float(gated_da) if not np.isnan(gated_da) else None,
                    "coverage": float(coverage),
                }
            results[regime_type] = regime_res

        return results


def run_regime_audit():
    """CLI runner to analyze historical regime resilience."""
    logger.info("Starting Regime Audit across historical data...")
    loader = DataLoader()
    raw_df = loader.load_all()
    fe = FeatureEngineer()
    df_feat = fe.create_all_features(raw_df)
    
    if "Date" not in df_feat.columns and isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat = df_feat.reset_index()

    analyzer = RegimeAnalyzer(df_feat, asset="BDI")
    logger.info("Regime classification completed.")
    vol_counts = analyzer.df["vol_regime"].value_counts().to_dict()
    trend_counts = analyzer.df["trend_regime"].value_counts().to_dict()
    struct_counts = analyzer.df["structural_regime"].value_counts().to_dict()

    logger.info(f"Volatility breakdown: {vol_counts}")
    logger.info(f"Trend breakdown: {trend_counts}")
    logger.info(f"Structural breakdown: {struct_counts}")
    return analyzer


if __name__ == "__main__":
    run_regime_audit()
