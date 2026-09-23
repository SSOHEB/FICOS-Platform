"""
FICOS — Idle-Time Assessment Engine
Estimates expected idle days and risk penalties at origin/destination ports
or canal transit points based on weather, congestion, and port constraints.
"""

from __future__ import annotations

from typing import Dict, Any, List
from src.domain.schemas import Port, RiskResult, IdleTimeAssessment


class IdleAssessmentEngine:
    """
    Assesses idle-time risk based on operational constraints and environmental/weather signals.
    """

    def __init__(self, demurrage_rate_usd_day: float = 20000.0):
        self.demurrage_rate_usd_day = demurrage_rate_usd_day

    def assess_idle_time(
        self,
        origin_port: Port,
        dest_port: Port,
        risk_result: RiskResult,
        base_wait_days: float = 1.0
    ) -> IdleTimeAssessment:
        """
        Calculates total expected idle time and cost impact.
        """
        driver_factors: List[str] = []
        expected_days = base_wait_days

        # Port congestion contributions
        if origin_port.avg_congestion_delay_hours > 0:
            origin_delay_days = origin_port.avg_congestion_delay_hours / 24.0
            expected_days += origin_delay_days
            driver_factors.append(f"Origin congestion ({origin_port.name}: {origin_delay_days:.1f} days)")

        if dest_port.avg_congestion_delay_hours > 0:
            dest_delay_days = dest_port.avg_congestion_delay_hours / 24.0
            expected_days += dest_delay_days
            driver_factors.append(f"Dest congestion ({dest_port.name}: {dest_delay_days:.1f} days)")

        # Risk signals impact on delay
        if risk_result.weather_risk_level in ["HIGH", "CRITICAL"]:
            weather_delay = 2.0 if risk_result.weather_risk_level == "HIGH" else 4.0
            expected_days += weather_delay
            driver_factors.append(f"Weather alert delay ({risk_result.weather_risk_level}: +{weather_delay} days)")

        if risk_result.disruption_risk_level in ["HIGH", "CRITICAL"]:
            disruption_delay = 1.5 if risk_result.disruption_risk_level == "HIGH" else 3.5
            expected_days += disruption_delay
            driver_factors.append(f"Chokepoint disruption delay ({risk_result.disruption_risk_level}: +{disruption_delay} days)")

        estimated_cost_usd = expected_days * self.demurrage_rate_usd_day

        return IdleTimeAssessment(
            expected_idle_days=round(expected_days, 2),
            estimated_idle_cost_usd=round(estimated_cost_usd, 2),
            demurrage_rate_usd_per_day=self.demurrage_rate_usd_day,
            primary_drivers=driver_factors if driver_factors else ["Standard port laytime"]
        )
