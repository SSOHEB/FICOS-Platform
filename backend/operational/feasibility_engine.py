"""
FICOS — Feasibility Engine (Refactored, Configuration-Driven)

Checks whether a proposed vessel-port combination is physically feasible.
Uses PortRepository and VesselRepository (config-driven).

Key principles:
  - Returns a structured FeasibilityResult (not just True/False).
  - Every constraint is explicitly pass/fail with value comparison.
  - New ports/vessels require only config changes, not code changes.
  - Dataset B is the authoritative source for berth constraints where available.
"""

from __future__ import annotations

from typing import List, Optional
import yaml
from pathlib import Path

from backend.domain.schemas import (
    CargoRequirement, VesselClass, Port, FeasibilityResult, ConstraintCheck
)
from backend.operational.port_repository import PortRepository
from backend.operational.vessel_repository import VesselRepository


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class FeasibilityEngine:
    """
    Checks vessel-port feasibility for a cargo requirement.

    Usage:
        engine = FeasibilityEngine()
        result = engine.check(cargo, vessel_class, destination_port)
    """

    def __init__(self, port_repo: Optional[PortRepository] = None, vessel_repo: Optional[VesselRepository] = None, dataset_b_path: Optional[str] = None):
        if isinstance(port_repo, str):
            dataset_b_path = port_repo
            port_repo = None

        self._port_repo = port_repo or PortRepository(dataset_b_path=dataset_b_path)
        self._vessel_repo = vessel_repo or VesselRepository()

    def check_feasibility(
        self,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        cargo: CargoRequirement
    ) -> FeasibilityResult:
        """
        Check physical feasibility of route and vessel for cargo.
        """
        return self.check(
            cargo=cargo,
            vessel=vessel,
            destination_port_name=dest_port.name or dest_port.display_name or dest_port.code
        )

    def check(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        destination_port_name: str,
    ) -> FeasibilityResult:
        """
        Check feasibility of shipping cargo on vessel to destination_port.


        Returns a FeasibilityResult with detailed constraint pass/fail list.
        Never returns bare True/False.
        """
        port = self._port_repo.get(destination_port_name)
        passed: List[ConstraintCheck] = []
        failed: List[ConstraintCheck] = []

        # ── 1. Draft Check ──
        draft_ok = port.constraints.max_draft_m >= vessel.typical_draft_m
        draft_check = ConstraintCheck(
            name="DraftCheck",
            passed=draft_ok,
            required_value=vessel.typical_draft_m,
            actual_value=port.constraints.max_draft_m,
            reason=(
                "OK" if draft_ok else
                f"DraftExceeded: vessel requires {vessel.typical_draft_m}m, port max is {port.constraints.max_draft_m}m"
            ),
        )
        (passed if draft_ok else failed).append(draft_check)

        # ── 2. LOA Check ──
        loa_ok = port.constraints.max_loa_m >= vessel.typical_loa_m
        loa_check = ConstraintCheck(
            name="LOACheck",
            passed=loa_ok,
            required_value=vessel.typical_loa_m,
            actual_value=port.constraints.max_loa_m,
            reason=(
                "OK" if loa_ok else
                f"LOAExceeded: vessel is {vessel.typical_loa_m}m, port max is {port.constraints.max_loa_m}m"
            ),
        )
        (passed if loa_ok else failed).append(loa_check)

        # ── 3. DWT / Cargo Capacity Check ──
        dwt_ok = port.constraints.max_dwt_mt >= cargo.quantity_mt
        dwt_check = ConstraintCheck(
            name="DWTCapacityCheck",
            passed=dwt_ok,
            required_value=cargo.quantity_mt,
            actual_value=port.constraints.max_dwt_mt,
            reason=(
                "OK" if dwt_ok else
                f"CapacityInsufficient: cargo {cargo.quantity_mt} MT > port max {port.constraints.max_dwt_mt} MT"
            ),
        )
        (passed if dwt_ok else failed).append(dwt_check)

        # ── 4. Cargo Type Compatibility ──
        cargo_compatible = (
            not port.cargo_types_supported or
            cargo.cargo_type in port.cargo_types_supported
        )
        cargo_check = ConstraintCheck(
            name="CargoTypeCompatibility",
            passed=cargo_compatible,
            required_value=cargo.cargo_type,
            actual_value=port.cargo_types_supported,
            reason=(
                "OK" if cargo_compatible else
                f"CargoNotSupported: '{cargo.cargo_type}' not in {port.cargo_types_supported}"
            ),
        )
        (passed if cargo_compatible else failed).append(cargo_check)

        # ── 5. Vessel DWT vs. Cargo Size ──
        vessel_cargo_ok = (
            cargo.quantity_mt <= vessel.dwt_max * 1.05
            and cargo.quantity_mt >= vessel.dwt_min * 0.5
        )
        vessel_cargo_check = ConstraintCheck(
            name="VesselCargoFit",
            passed=vessel_cargo_ok,
            required_value=f"{vessel.dwt_min}-{vessel.dwt_max} MT",
            actual_value=f"{cargo.quantity_mt} MT",
            reason=(
                "OK" if vessel_cargo_ok else
                f"CargoVesselMismatch: cargo {cargo.quantity_mt} MT outside vessel DWT range "
                f"{vessel.dwt_min}-{vessel.dwt_max} MT"
            ),
        )
        (passed if vessel_cargo_ok else failed).append(vessel_cargo_check)

        # ── Result ──
        feasible = len(failed) == 0
        limiting = failed[0].name if failed else None
        explanation_parts = [c.reason for c in failed]
        explanation = " | ".join(explanation_parts) if explanation_parts else "All constraints satisfied."

        # Port risk score from Dataset B PBDT
        port_risk_score = self._port_repo.get_dataset_b_pbdt(port.code)

        return FeasibilityResult(
            vessel_code=vessel.code,
            port_code=port.code,
            feasible=feasible,
            passed_constraints=passed,
            failed_constraints=failed,
            limiting_constraint=limiting,
            port_risk_score=port_risk_score,
            explanation=explanation,
        )

    # ──────────────────────────────────────────────────────────
    # BATCH CHECK — All Candidate Vessels
    # ──────────────────────────────────────────────────────────

    def check_all_candidates(
        self,
        cargo: CargoRequirement,
        destination_port_name: str,
        vessel_codes: Optional[List[str]] = None,
    ) -> List[FeasibilityResult]:
        """
        Check feasibility for all candidate vessel classes.
        Returns a list of FeasibilityResult in vessel priority order.
        """
        if vessel_codes is not None:
            candidates = [self._vessel_repo.get_by_code(c) for c in vessel_codes]
        else:
            candidates = self._vessel_repo.candidates_for_cargo(cargo.quantity_mt)

        results = []
        for vessel in candidates:
            result = self.check(cargo, vessel, destination_port_name)
            results.append(result)
        return results

    # ──────────────────────────────────────────────────────────
    # ALTERNATIVE PORT SEARCH
    # ──────────────────────────────────────────────────────────

    def find_alternative_port(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        failed_port_name: str,
    ) -> Optional[str]:
        """
        Search all configured ports for a feasible alternative.
        Returns the name of the first feasible port, or None.
        """
        all_ports = self._port_repo.all_ports()
        for port_key, port in all_ports.items():
            if port_key == failed_port_name.lower():
                continue
            try:
                result = self.check(cargo, vessel, port_key)
                if result.feasible:
                    return port.display_name
            except Exception:
                continue
        return None
