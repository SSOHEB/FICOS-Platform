"""
FICOS — Recommendation Application Service
High-level application service interface for generating freight charter recommendations.
Designed to support future REST API / UI integrations cleanly without logic coupling.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime

from backend.domain.schemas import CargoRequirement, Route
from backend.operational.port_repository import PortRepository
from backend.operational.vessel_repository import VesselRepository
from backend.decision.engine import DecisionEngine
from backend.decision.schemas import RecommendationOutput


class RecommendationService:
    """
    Application Service entrypoint for decision requests.
    Handles loading entity defaults, invoking DecisionEngine, and returning structured outputs.
    """

    def __init__(
        self,
        port_repo: Optional[PortRepository] = None,
        vessel_repo: Optional[VesselRepository] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.port_repo = port_repo or PortRepository()
        self.vessel_repo = vessel_repo or VesselRepository()
        self.decision_engine = decision_engine or DecisionEngine()

    def get_recommendation(
        self,
        asset_type: str,
        quantity_mt: float,
        origin_port_code: str,
        dest_port_code: str,
        laycan_days: int = 14,
        vessel_class_name: Optional[str] = None,
        distance_nm: float = 5000.0,
        requires_suez: bool = False,
        requires_panama: bool = False
    ) -> RecommendationOutput:
        """
        Submits cargo and route request to generate a structured recommendation.
        """
        # Resolve Ports
        origin_port = self.port_repo.get_port(origin_port_code)
        if not origin_port:
            origin_port = self.port_repo.get_port("TUBARAO")  # fallback default

        dest_port = self.port_repo.get_port(dest_port_code)
        if not dest_port:
            dest_port = self.port_repo.get_port("QINGDAO")  # fallback default

        # Resolve Vessel
        if not vessel_class_name:
            vessel_class_name = asset_type  # e.g. PANAMAX_1D -> PANAMAX

        vessel = self.vessel_repo.get_vessel(vessel_class_name)
        if not vessel:
            vessel = self.vessel_repo.get_vessel("PANAMAX")  # fallback default

        # Build CargoRequirement entity
        cargo = CargoRequirement(
            cargo_type=asset_type,
            quantity_mt=quantity_mt,
            origin=origin_port.code,
            destination=dest_port.code,
            cargo_id=f"CARGO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            asset_type=asset_type,
            origin_port_code=origin_port.code,
            dest_port_code=dest_port.code,
            laycan_days=laycan_days
        )

        # Build Route entity
        route = Route(
            origin=origin_port.code,
            destination=dest_port.code,
            origin_code=origin_port.code,
            dest_code=dest_port.code,
            distance_nautical_miles=distance_nm,
            estimated_speed_knots=vessel.speed_knots,
            requires_suez=requires_suez,
            requires_panama=requires_panama
        )


        # Run Decision Engine
        return self.decision_engine.evaluate(
            cargo=cargo,
            vessel=vessel,
            origin_port=origin_port,
            dest_port=dest_port,
            route=route
        )
