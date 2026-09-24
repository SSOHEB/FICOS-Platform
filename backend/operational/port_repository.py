"""
FICOS — Port Repository (Dataset B Abstraction)
Loads port configurations from configs/ports.yaml and optionally
enriches constraints from the Dataset B Excel file.

Dataset B must NOT be merged into the freight ML feature matrix.
Its role is strictly: operational feasibility and risk scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import yaml

from backend.domain.schemas import Port, PortConstraints


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_ports_config() -> dict:
    path = _project_root() / "configs" / "ports.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Port key aliases (lowercase → canonical config key)
PORT_ALIASES: Dict[str, str] = {
    "para": "paradip",
    "paradip": "paradip",
    "vizag": "visakhapatnam",
    "visakhapatnam": "visakhapatnam",
    "gang": "gangavaram",
    "gangavaram": "gangavaram",
    "dham": "dhamra",
    "dhamra": "dhamra",
    "haldia": "haldia",
    "haldia_dock_complex": "haldia",
    "hdc": "haldia",
}


class PortRepository:
    """
    Provides Port domain objects loaded from configs/ports.yaml.
    Optionally cross-references Dataset B for berth-level detail.
    """

    def __init__(self, dataset_b_path: Optional[str] = None):
        self._config = _load_ports_config()
        self._ports: Dict[str, Port] = {}
        self._dataset_b_path = dataset_b_path
        self._load_all()

    def _load_all(self) -> None:
        for key, data in self._config.get("ports", {}).items():
            constraints_data = data.get("constraints", {})
            constraints = PortConstraints(
                max_draft_m=constraints_data.get("max_draft_m", 0.0),
                max_loa_m=constraints_data.get("max_loa_m", 0.0),
                max_dwt_mt=constraints_data.get("max_dwt_mt", 0.0),
            )
            port = Port(
                code=data.get("code", key.upper()),
                display_name=data.get("display_name", key.title()),
                key=key,
                country=data.get("country", "India"),
                state=data.get("state", ""),
                constraints=constraints,
                cargo_types_supported=data.get("cargo_types_supported", []),
                tidal_port=data.get("tidal_port", False),
                monsoon_restrictions=data.get("monsoon_restrictions", False),
                operational_notes=data.get("operational_notes", ""),
            )
            self._ports[key] = port

    def get(self, port_name: str) -> Port:
        """
        Return a Port object for the given name/alias.
        """
        key = PORT_ALIASES.get(port_name.lower().strip())
        if key is not None and key in self._ports:
            return self._ports[key]
        if port_name.lower() in self._ports:
            return self._ports[port_name.lower()]
        
        # Fallback default port object
        constraints = PortConstraints(max_draft_m=18.0, max_loa_m=300.0, max_dwt_mt=150000.0)
        return Port(
            code=port_name.upper()[:6],
            display_name=port_name.title(),
            key=port_name.lower(),
            country="International",
            state="",
            constraints=constraints,
            name=port_name.title()
        )

    def get_port(self, port_name: str) -> Port:
        """Safe accessor for get()."""
        return self.get(port_name)



    def all_ports(self) -> Dict[str, Port]:
        return dict(self._ports)

    def get_dataset_b_pbdt(self, port_code: str) -> float:
        """
        Get historical pre-berthing detention (PBDT) risk score from Dataset B.
        Returns a normalised 0-100 score. Falls back to 35.0 if data unavailable.
        """
        if self._dataset_b_path is None:
            return 35.0  # default baseline if no Dataset B

        try:
            import pandas as pd
            import numpy as np
            xl = pd.ExcelFile(self._dataset_b_path)
            pbdt_df = pd.read_excel(xl, "pbdt")
            subset = pbdt_df[pbdt_df["port"].str.contains(port_code, case=False, na=False)]
            if subset.empty:
                return 35.0
            avg_pbdt = subset["pre_berthing_detention_hours"].mean()
            return float(np.clip((avg_pbdt / 50.0) * 100.0, 10.0, 95.0))
        except Exception:
            return 35.0
