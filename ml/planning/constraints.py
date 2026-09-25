"""Auditable coupling and portfolio constraints."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
from .voyage import Voyage, VoyageOpportunity

@dataclass
class CouplingGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def from_voyages(cls, voyages: list[Voyage]):
        return cls(tuple(v.voyage_id for v in voyages), tuple((a.voyage_id, b.voyage_id, "same_route_sequence") for a, b in zip(voyages, voyages[1:])))

@dataclass(frozen=True)
class PlanningConstraints:
    budget_usd: float = 1_000_000_000.0
    contract_capacity_mt: float = 10_000_000.0
    max_contracts: int = 999
    max_multi_voyage_contracts: int = 999
    max_route_concentration: float = 1.0
    assumptions: Mapping[str, str] = field(default_factory=lambda: {"budget_usd": "SCENARIO_ASSUMPTION", "contract_capacity_mt": "SCENARIO_ASSUMPTION", "max_contracts": "SCENARIO_ASSUMPTION", "max_route_concentration": "SCENARIO_ASSUMPTION"})

    def validate(self, opportunities: list[VoyageOpportunity]):
        if not opportunities:
            raise ValueError("At least one voyage opportunity is required")
        for opportunity in opportunities:
            if opportunity.voyage.volume_mt > opportunity.voyage.capacity_mt:
                raise ValueError(f"{opportunity.voyage.voyage_id} exceeds vessel capacity")
