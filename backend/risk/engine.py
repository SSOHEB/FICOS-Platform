"""
FICOS — Risk Engine (Dataset C)

Aggregates weather, geopolitical, and disruption signals from Dataset C
(and GDELT features from modeling_dataset.csv) into a structured RiskResult.

Key design rules:
  - Dataset C signals are evaluated separately from Dataset A freight ML features.
  - Risk scores inform the DECISION ECONOMICS (expected cost of waiting).
  - Risk does NOT directly enter the freight-rate prediction model.
  - All thresholds are configuration-driven (configs/risk_policy.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import yaml

from src.domain.schemas import RiskLevel, RiskResult, RiskAlert


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_risk_policy() -> dict:
    path = _project_root() / "configs" / "risk_policy.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_cost_model() -> dict:
    path = _project_root() / "configs" / "cost_model.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RiskEngine:
    """
    Evaluates disruption/event/weather risk from Dataset C signals.
    Returns a RiskResult that the DecisionEngine uses to adjust economics.

    The risk score has a real economic effect:
      - It adds a risk premium to expected_cost_wait (see CostModel).
      - HIGH risk can convert a WAIT to FLEXIBLE_INDEX (see DecisionPolicy).
    """

    def __init__(self):
        self._policy = _load_risk_policy()
        self._cost_cfg = _load_cost_model()
        self._levels = self._policy.get("risk_levels", {})
        self._weather_cfg = self._policy.get("weather_risk", {})
        self._geo_cfg = self._policy.get("geopolitical_risk", {})
        self._premium_cfg = self._cost_cfg.get("risk_premium", {})

    def _classify_level(self, score: float) -> RiskLevel:
        low_thresh = self._levels.get("low_threshold", 30.0)
        med_thresh = self._levels.get("medium_threshold", 65.0)
        if score < low_thresh:
            return RiskLevel.LOW
        elif score < med_thresh:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def _weather_score(
        self,
        cyclone_dist_km: Optional[float],
        high_wind_active: bool,
        heavy_precip_active: bool,
    ) -> tuple[float, list]:
        alerts = []
        cyclone_thresh = self._weather_cfg.get("cyclone_proximity_km", 500)

        if cyclone_dist_km is not None and cyclone_dist_km < cyclone_thresh:
            score = self._weather_cfg.get("cyclone_active_score", 85.0)
            alerts.append(RiskAlert(
                alert_type="CYCLONE_PROXIMITY",
                description=f"Cyclone within {cyclone_dist_km:.0f} km (threshold: {cyclone_thresh} km).",
                severity=RiskLevel.HIGH,
            ))
        elif high_wind_active:
            score = self._weather_cfg.get("high_wind_score", 60.0)
            alerts.append(RiskAlert(
                alert_type="HIGH_WIND",
                description=f"High wind indicator active (>{self._weather_cfg.get('high_wind_kmh', 50)} km/h).",
                severity=RiskLevel.MEDIUM,
            ))
        elif heavy_precip_active:
            score = self._weather_cfg.get("heavy_precip_score", 45.0)
            alerts.append(RiskAlert(
                alert_type="HEAVY_PRECIPITATION",
                description=f"Heavy precipitation indicator active (>{self._weather_cfg.get('heavy_precip_mm', 50)} mm).",
                severity=RiskLevel.MEDIUM,
            ))
        else:
            score = self._weather_cfg.get("normal_score", 10.0)

        return score, alerts

    def _geopolitical_score(
        self,
        gdelt_event_count: Optional[float],
    ) -> tuple[float, list]:
        alerts = []
        burst_thresh = self._geo_cfg.get("burst_count_threshold", 50)

        if gdelt_event_count is not None and gdelt_event_count > burst_thresh:
            score = self._geo_cfg.get("burst_active_score", 75.0)
            alerts.append(RiskAlert(
                alert_type="GDELT_BURST",
                description=f"GDELT conflict event count ({gdelt_event_count:.0f}) exceeds burst threshold ({burst_thresh}).",
                severity=RiskLevel.HIGH,
            ))
        elif gdelt_event_count is not None and gdelt_event_count > 0:
            score = self._geo_cfg.get("elevated_score", 40.0)
        else:
            score = self._geo_cfg.get("normal_score", 10.0)

        return score, alerts

    def _risk_cost_premium(self, level: RiskLevel) -> float:
        if level == RiskLevel.HIGH:
            return float(self._premium_cfg.get("high_risk_premium_usd", {}).get("value", 15000))
        elif level == RiskLevel.MEDIUM:
            return float(self._premium_cfg.get("medium_risk_premium_usd", {}).get("value", 5000))
        return 0.0

    def evaluate_risk(
        self,
        origin_port: str,
        dest_port: str,
        asset_type: str = "PANAMAX_1D"
    ) -> RiskResult:
        """Convenience accessor for risk evaluation."""
        return self.evaluate()

    def evaluate(
        self,
        port_risk_score: float = 35.0,
        cyclone_dist_km: Optional[float] = None,
        high_wind_active: bool = False,
        heavy_precip_active: bool = False,
        gdelt_event_count: Optional[float] = None,
    ) -> RiskResult:

        """
        Evaluate all risk signals and return a structured RiskResult.

        Parameters
        ----------
        port_risk_score : float
            Historical port detention risk from Dataset B PBDT (0-100).
        cyclone_dist_km : float, optional
            Distance of nearest active cyclone to East Coast India (km).
        high_wind_active : bool
            Whether the high-wind engineering indicator is active.
        heavy_precip_active : bool
            Whether the heavy-precipitation indicator is active.
        gdelt_event_count : float, optional
            GDELT daily conflict event count from modeling_dataset.csv.
        """
        weather_score, weather_alerts = self._weather_score(
            cyclone_dist_km, high_wind_active, heavy_precip_active
        )
        geo_score, geo_alerts = self._geopolitical_score(gdelt_event_count)

        agg_method = self._policy.get("disruption_aggregation", {}).get("method", "max")
        if agg_method == "max":
            disruption_score = max(weather_score, geo_score)
        else:  # weighted_mean
            w = self._policy.get("disruption_aggregation", {}).get("weights", {})
            ww = w.get("weather", 0.5)
            wg = w.get("geopolitical", 0.5)
            disruption_score = ww * weather_score + wg * geo_score

        overall_level = self._classify_level(disruption_score)

        # Adjust if port itself is high risk
        if port_risk_score >= 65.0:
            overall_level = RiskLevel.HIGH

        all_alerts = weather_alerts + geo_alerts
        risk_premium = self._risk_cost_premium(overall_level)

        # Decision adjustment text
        if overall_level == RiskLevel.HIGH:
            adj = "HIGH disruption risk: converts WAIT to FLEXIBLE_INDEX; increases expected waiting cost."
        elif overall_level == RiskLevel.MEDIUM:
            adj = "MEDIUM disruption risk: moderate risk premium applied to expected cost of waiting."
        else:
            adj = "LOW disruption risk: no material adjustment to decision economics."

        explanation = (
            f"Weather score: {weather_score:.0f}/100. "
            f"Geopolitical score: {geo_score:.0f}/100. "
            f"Port risk: {port_risk_score:.0f}/100. "
            f"Overall: {overall_level.value}."
        )

        return RiskResult(
            overall_level=overall_level,
            weather_score=round(weather_score, 1),
            geopolitical_score=round(geo_score, 1),
            disruption_score=round(disruption_score, 1),
            operational_risk_score=round(port_risk_score, 1),
            active_alerts=all_alerts,
            decision_adjustment=adj,
            risk_cost_premium_usd=risk_premium,
            explanation=explanation,
        )
