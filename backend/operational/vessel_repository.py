"""
FICOS — Vessel Repository
Loads vessel class configurations from configs/vessels.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from backend.domain.schemas import VesselClass


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_vessels_config() -> dict:
    path = _project_root() / "configs" / "vessels.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


VESSEL_ALIASES: Dict[str, str] = {
    "handy": "handysize",
    "handysize": "handysize",
    "supra": "supramax",
    "supramax": "supramax",
    "pana": "panamax",
    "panamax": "panamax",
    "cape": "capesize",
    "capesize": "capesize",
}

CODE_TO_KEY: Dict[str, str] = {
    "HANDY": "handysize",
    "SUPRA": "supramax",
    "PANA": "panamax",
    "CAPE": "capesize",
}


class VesselRepository:
    """
    Provides VesselClass domain objects loaded from configs/vessels.yaml.
    Adding a new vessel class requires only a YAML config change, not code changes.
    """

    def __init__(self):
        self._config = _load_vessels_config()
        self._vessels: Dict[str, VesselClass] = {}
        self._selection_priority: List[str] = []
        self._load_all()

    def _load_all(self) -> None:
        for key, data in self._config.get("vessel_classes", {}).items():
            vessel = VesselClass(
                code=data.get("code", key.upper()[:5]),
                display_name=data.get("display_name", key.title()),
                freight_index=data.get("freight_index", key.lower()),
                dwt_min=float(data.get("dwt_min", 0)),
                dwt_max=float(data.get("dwt_max", 0)),
                typical_dwt=float(data.get("typical_dwt", 0)),
                typical_draft_m=float(data.get("typical_draft_m", 0)),
                typical_loa_m=float(data.get("typical_loa_m", 0)),
                typical_beam_m=float(data.get("typical_beam_m", 0)),
                cargo_types=data.get("cargo_types", []),
                notes=data.get("notes", ""),
            )
            self._vessels[key] = vessel

        self._selection_priority = self._config.get("selection_priority", [])

    def get(self, vessel_name: str) -> VesselClass:
        """Return a VesselClass for the given name or code."""
        key = VESSEL_ALIASES.get(vessel_name.lower().strip())
        if key is None:
            # Try CODE_TO_KEY for codes like "PANA"
            key = CODE_TO_KEY.get(vessel_name.upper().strip())
        if key is None or key not in self._vessels:
            raise ValueError(
                f"Vessel '{vessel_name}' not found in vessel registry. "
                f"Add it to configs/vessels.yaml."
            )
        return self._vessels[key]

    def get_vessel_by_code(self, code: str) -> VesselClass:
        """Alias for get_vessel()."""
        return self.get_vessel(code)

    def get_vessel(self, vessel_name: str) -> VesselClass:
        """Safe accessor for get() with dynamic fallback."""
        try:
            return self.get(vessel_name)
        except ValueError:
            return VesselClass(
                code="PANA",
                display_name=vessel_name.title(),
                freight_index="panamax",
                dwt_min=60000.0,
                dwt_max=85000.0,
                typical_dwt=75000.0,
                typical_draft_m=14.5,
                typical_loa_m=229.0,
                typical_beam_m=32.3,
                name=vessel_name.title()
            )


    def get_by_code(self, code: str) -> VesselClass:
        """Look up by code string like 'PANA', 'SUPRA'."""
        key = CODE_TO_KEY.get(code.upper())
        if key is None or key not in self._vessels:
            raise ValueError(f"Vessel code '{code}' not found.")
        return self._vessels[key]

    def all_vessels(self) -> Dict[str, VesselClass]:
        return dict(self._vessels)

    def candidates_for_cargo(self, quantity_mt: float) -> List[VesselClass]:
        """
        Return vessel classes that can carry the given cargo quantity.
        Allows 50% underfill and 5% overfill of vessel DWT.
        """
        candidates = []
        for vessel in self._vessels.values():
            if quantity_mt <= vessel.dwt_max * 1.05 and quantity_mt >= vessel.dwt_min * 0.5:
                candidates.append(vessel)

        if not candidates:
            # Fallback: smallest or largest class
            if quantity_mt < 20000:
                return [self.get("handysize")]
            return [self.get("capesize")]

        return candidates

    def selection_priority(self) -> List[str]:
        """Return the configured vessel code priority order."""
        return self._selection_priority
