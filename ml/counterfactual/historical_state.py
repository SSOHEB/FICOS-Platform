"""Leakage-aware historical market state representation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HistoricalMarketState:
    """What was observable at a decision timestamp.

    Forecast and uncertainty fields are reconstructed from training-only folds;
    private procurement fields are deliberately represented as unavailable.
    """

    date: str
    vessel: str
    current_rate: float
    actual_future_rate: float | None
    forecast_delta: float | None
    lower_bound: float | None
    upper_bound: float | None
    when_decision: str | None
    observed_features: Mapping[str, Any]
    provenance: Mapping[str, str]

    def assert_no_future_features(self) -> None:
        for key, source in self.provenance.items():
            if source == "FUTURE_INFORMATION":
                raise ValueError(f"future information attached to state: {key}")

    def to_dict(self) -> dict[str, Any]:
        self.assert_no_future_features()
        return {
            "date": self.date,
            "vessel": self.vessel,
            "current_rate": self.current_rate,
            "actual_future_rate": self.actual_future_rate,
            "forecast_delta": self.forecast_delta,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "when_decision": self.when_decision,
            "observed_features": dict(self.observed_features),
            "provenance": dict(self.provenance),
        }
