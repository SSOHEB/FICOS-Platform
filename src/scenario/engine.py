"""
FICOS — Scenario Engine
Replays the decision under modified assumptions without touching ML training.

Scenarios modify inputs to the risk engine, forecast, or cost model.
The DecisionEngine is re-run on each scenario to show how decisions change.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from src.domain.schemas import (
    CargoRequirement, ForecastResult, RiskResult, ScenarioType, ScenarioResult, DecisionType
)


class ScenarioEngine:
    """
    Replays chartering decisions under alternative scenarios.
    Does NOT retrain models. Modifies input parameters only.

    Scenarios:
      NORMAL             — baseline inputs unchanged
      FREIGHT_SPIKE      — delta boosted by +20%
      FREIGHT_DROP       — delta reduced by -20%
      PORT_DISRUPTION    — port risk score set to HIGH
      WEATHER_DISRUPTION — weather risk activated
      GEOPOLITICAL_SHOCK — GDELT burst activated
      HIGH_UNCERTAINTY   — uncertainty widened; fallback triggered
    """

    # Scenario parameter modifications
    SCENARIO_OVERRIDES: Dict[ScenarioType, Dict[str, Any]] = {
        ScenarioType.NORMAL: {},
        ScenarioType.FREIGHT_SPIKE: {
            "forecast_delta_multiplier": 1.20,
            "description": "Freight rates spike 20% above forecast.",
        },
        ScenarioType.FREIGHT_DROP: {
            "forecast_delta_multiplier": -0.80,
            "description": "Freight rates drop 20% below forecast.",
        },
        ScenarioType.PORT_DISRUPTION: {
            "port_risk_score_override": 80.0,
            "description": "Port disruption (operational/congestion risk = HIGH).",
        },
        ScenarioType.WEATHER_DISRUPTION: {
            "cyclone_dist_km_override": 200.0,
            "description": "Active cyclone within 200 km of East Coast India ports.",
        },
        ScenarioType.GEOPOLITICAL_SHOCK: {
            "gdelt_event_count_override": 200.0,
            "description": "Major geopolitical shock; GDELT conflict events at 4x burst threshold.",
        },
        ScenarioType.HIGH_UNCERTAINTY: {
            "force_fallback": True,
            "description": "Forecast uncertainty too high; forced to FLEXIBLE_INDEX fallback.",
        },
    }

    def run(
        self,
        scenario: ScenarioType,
        decision_engine,        # DecisionEngine instance (avoid circular import)
        cargo: CargoRequirement,
        current_rate: float,
        baseline_forecast_delta: Optional[float] = None,
        baseline_decision: Optional[DecisionType] = None,
        **kwargs,
    ) -> ScenarioResult:
        """
        Run a scenario and return the comparison to baseline.

        Parameters
        ----------
        scenario : ScenarioType
        decision_engine : DecisionEngine
            The live DecisionEngine instance to re-run.
        cargo : CargoRequirement
        current_rate : float
        baseline_forecast_delta : float, optional
        baseline_decision : DecisionType, optional
            The baseline decision (from NORMAL scenario) to compare against.
        """
        overrides = self.SCENARIO_OVERRIDES.get(scenario, {})
        description = overrides.get("description", "No description.")

        # ── Apply scenario overrides to kwargs ──
        if "forecast_delta_multiplier" in overrides and baseline_forecast_delta is not None:
            kwargs["forecast_delta"] = baseline_forecast_delta * overrides["forecast_delta_multiplier"]
        elif "forecast_delta_multiplier" in overrides:
            kwargs["forecast_delta"] = 0.0  # no baseline delta — no change

        if "port_risk_score_override" in overrides:
            kwargs["port_risk_score"] = overrides["port_risk_score_override"]

        if "cyclone_dist_km_override" in overrides:
            kwargs["cyclone_dist_km"] = overrides["cyclone_dist_km_override"]

        if "gdelt_event_count_override" in overrides:
            kwargs["gdelt_event_count"] = overrides["gdelt_event_count_override"]

        if overrides.get("force_fallback"):
            kwargs["force_fallback"] = True

        # ── Re-run the decision engine under this scenario ──
        scenario_charter = decision_engine.decide(cargo=cargo, current_rate=current_rate, **kwargs)
        scenario_decision = scenario_charter.decision

        # ── Compute differences ──
        changed = (baseline_decision is not None) and (scenario_decision != baseline_decision)

        cost_diff = None
        if (
            scenario_charter.expected_cost_now is not None
            and scenario_charter.expected_cost_wait is not None
            and baseline_decision is not None
        ):
            cost_diff = scenario_charter.expected_cost_wait - scenario_charter.expected_cost_now

        r_val = scenario_charter.risk_level.value if hasattr(scenario_charter.risk_level, 'value') else str(scenario_charter.risk_level)
        risk_diff = (
            f"Scenario risk: {r_val}."
            if scenario_charter.risk_result
            else "Risk unchanged."
        )

        base_val = baseline_decision.value if hasattr(baseline_decision, 'value') else str(baseline_decision) if baseline_decision else 'N/A'
        scen_val = scenario_decision.value if hasattr(scenario_decision, 'value') else str(scenario_decision)

        explanation = (
            f"Scenario: {scenario.value if hasattr(scenario, 'value') else str(scenario)}. {description} "
            f"Baseline decision: {base_val}. "
            f"Scenario decision: {scen_val}. "
            f"{'Decision CHANGED.' if changed else 'Decision unchanged.'}"
        )


        return ScenarioResult(
            scenario=scenario,
            baseline_decision=baseline_decision or scenario_decision,
            scenario_decision=scenario_decision,
            changed=changed,
            cost_difference_usd=cost_diff,
            risk_difference=risk_diff,
            explanation=explanation,
        )

    def run_all(
        self,
        decision_engine,
        cargo: CargoRequirement,
        current_rate: float,
        baseline_forecast_delta: Optional[float] = None,
        baseline_decision: Optional[DecisionType] = None,
        **kwargs,
    ) -> Dict[str, ScenarioResult]:
        """Run all defined scenarios and return results keyed by scenario name."""
        results = {}
        for scenario in ScenarioType:
            results[scenario.value] = self.run(
                scenario=scenario,
                decision_engine=decision_engine,
                cargo=cargo,
                current_rate=current_rate,
                baseline_forecast_delta=baseline_forecast_delta,
                baseline_decision=baseline_decision,
                **kwargs,
            )
        return results
