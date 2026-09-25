"""Convert historical market states into explicitly counterfactual opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .historical_state import HistoricalMarketState


@dataclass(frozen=True)
class HistoricalMarketOpportunity:
    opportunity_id: str
    state: HistoricalMarketState
    status: str
    evidence_type: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "status": self.status,
            "evidence_type": self.evidence_type,
            "assumptions": list(self.assumptions),
            **self.state.to_dict(),
        }


def build_opportunity(index: int, state: HistoricalMarketState) -> HistoricalMarketOpportunity:
    state.assert_no_future_features()
    return HistoricalMarketOpportunity(
        opportunity_id=f"HIST-MARKET-{index:06d}",
        state=state,
        status="HISTORICAL_MARKET_OPPORTUNITY",
        evidence_type="COUNTERFACTUAL",
        assumptions=(
            "PRIVATE_DATA_UNAVAILABLE: SAIL procurement decision",
            "PRIVATE_DATA_UNAVAILABLE: voyage-specific volume",
            "SCENARIO_ASSUMPTION: contract economics",
        ),
    )
