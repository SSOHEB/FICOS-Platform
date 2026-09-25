"""Explicit research voyage and procurement opportunity objects."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

class Strategy(str, Enum):
    SPOT = "SPOT"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    MULTI_VOYAGE_CONTRACT = "MULTI_VOYAGE_CONTRACT"

@dataclass(frozen=True)
class Voyage:
    voyage_id: str
    vessel_class: str
    route: str
    origin: str
    destination: str
    laycan_start: str
    laycan_end: str
    expected_duration_days: int
    volume_mt: float
    capacity_mt: float
    timestamp: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class VoyageOpportunity:
    voyage: Voyage
    current_rate: float
    forecast_rate: float
    lower_rate: float
    upper_rate: float
    strategy_costs: Mapping[str, float]
    signal: str
    uncertainty_method: str
    assumptions: tuple[str, ...] = ()

    @property
    def forecast_delta(self):
        return self.forecast_rate - self.current_rate

    @property
    def uncertainty_width(self):
        return self.upper_rate - self.lower_rate

    def to_dict(self):
        return {"voyage_id": self.voyage.voyage_id, "vessel_class": self.voyage.vessel_class, "route": self.voyage.route, "timestamp": self.voyage.timestamp, "current_rate": self.current_rate, "forecast_rate": self.forecast_rate, "forecast_delta": self.forecast_delta, "lower_rate": self.lower_rate, "upper_rate": self.upper_rate, "strategy_costs": dict(self.strategy_costs), "signal": self.signal, "uncertainty_method": self.uncertainty_method, "assumptions": list(self.assumptions)}
