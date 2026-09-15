"""
FICOS — Decision Explanation Generator
Builds human-readable audit trails and explanations for decision engine outputs.
"""

from __future__ import annotations

from typing import Dict, Any, List
from src.domain.schemas import ForecastResult, FeasibilityResult, RiskResult, PolicyEvaluation


class ExplanationGenerator:
    """
    Generates step-by-step audit trace explaining why a specific decision was chosen.
    """

    def generate_explanation(
        self,
        recommended_strategy: str,
        forecast: ForecastResult,
        feasibility: FeasibilityResult,
        risk: RiskResult,
        candidate_evaluations: List[PolicyEvaluation]
    ) -> Dict[str, Any]:
        """
        Creates a structured breakdown explaining the decision rationale.
        """
        steps: List[str] = []

        # 1. Forecast summary
        steps.append(
            f"Forecast Step: Promoted model '{forecast.model_name}' (Version {forecast.model_version}) "
            f"predicted freight rate of ${forecast.point_forecast:.2f}/MT for asset {forecast.asset_type} "
            f"(P10=${forecast.uncertainty.p10:.2f}, P90=${forecast.uncertainty.p90:.2f})."
        )

        # 2. Feasibility gate
        if feasibility.is_feasible:
            steps.append(
                f"Feasibility Gate: PASSED. All physical port constraints satisfied for vessel "
                f"at origin ({feasibility.checks[0].reason if feasibility.checks else 'OK'}) and destination."
            )

        else:
            violations = [c.message for c in feasibility.checks if not c.passed]
            steps.append(f"Feasibility Gate: FAILED. Violations: {'; '.join(violations)}")

        # 3. Risk gate
        steps.append(
            f"Risk Gate: Overall risk score {risk.overall_risk_score:.1f}/100 "
            f"(Weather: {risk.weather_risk_level}, Disruption: {risk.disruption_risk_level}). "
            f"Key drivers: {', '.join(risk.key_risk_drivers) if risk.key_risk_drivers else 'None'}."
        )

        # 4. Strategy comparison
        strat_summaries = []
        for eval_item in candidate_evaluations:
            status = "Feasible" if eval_item.is_feasible else "Infeasible/Gated"
            strat_summaries.append(
                f"- {eval_item.strategy_name}: Adjusted Cost = ${eval_item.adjusted_expected_cost_usd:,.0f} "
                f"(Base: ${eval_item.expected_cost_usd:,.0f}, Status: {status})"
            )
        steps.append("Strategy Comparison:\n" + "\n".join(strat_summaries))

        # 5. Final Recommendation Rationale
        if recommended_strategy == "REJECT":
            rationale = "Candidate route/vessel rejected due to physical constraint violations or critical risk level."
        else:
            rationale = (
                f"Selected '{recommended_strategy}' strategy because it achieved the lowest risk-adjusted "
                f"expected cost while satisfying all physical feasibility gates and risk thresholds."
            )
        steps.append(f"Final Decision Rationale: {rationale}")

        return {
            "summary_rationale": rationale,
            "decision_steps": steps,
            "gating_status": {
                "physical_feasibility": feasibility.is_feasible,
                "risk_within_threshold": risk.overall_risk_score <= 85.0
            }
        }
