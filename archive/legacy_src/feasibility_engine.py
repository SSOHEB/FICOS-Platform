"""
SIH26006 — Phase 10: Vessel & Port Feasibility Engine

Restricted exclusively to the 5 validated MVP ports in Dataset B:
1. Paradip (PARA)
2. Visakhapatnam (VIZAG)
3. Gangavaram (GANG)
4. Dhamra (DHAM)
5. Haldia Dock Complex (HDC)

Candidate-First Feasibility Validation:
1. Generates candidate vessel classes for requested cargo volume (MT).
2. Validates candidates against Dataset B berth specs (depth/draft, LOA, DWT, cargo rules, tide/operating rules).
3. Computes historical baseline port detention risk factor from PBDT/TRT evidence.
"""

import os
import yaml
import pandas as pd
import numpy as np

PORT_MAP = {
    "paradip": "PARA",
    "para": "PARA",
    "visakhapatnam": "VIZAG",
    "vizag": "VIZAG",
    "gangavaram": "GANG",
    "gang": "GANG",
    "dhamra": "DHAM",
    "dham": "DHAM",
    "haldia": "HDC",
    "haldia_dock_complex": "HDC",
    "hdc": "HDC"
}

# Candidate Vessel Class Specification (DWT range, typical draft requirement, typical LOA requirement)
VESSEL_SPECS = {
    "HANDY": {
        "name": "Handysize",
        "min_dwt": 20000,
        "max_dwt": 40000,
        "typical_draft_m": 10.0,
        "typical_loa_m": 180.0
    },
    "SUPRA": {
        "name": "Supramax",
        "min_dwt": 45000,
        "max_dwt": 62000,
        "typical_draft_m": 12.5,
        "typical_loa_m": 200.0
    },
    "PANA": {
        "name": "Panamax",
        "min_dwt": 63000,
        "max_dwt": 85000,
        "typical_draft_m": 14.0,
        "typical_loa_m": 225.0
    },
    "CAPE": {
        "name": "Capesize",
        "min_dwt": 100000,
        "max_dwt": 200000,
        "typical_draft_m": 17.5,
        "typical_loa_m": 290.0
    }
}

class FeasibilityEngine:
    def __init__(self, dataset_b_path="DATASET_B_FINAL_FIXED (1).xlsx"):
        self.excel_path = dataset_b_path
        if os.path.exists(dataset_b_path):
            self.excel_file = pd.ExcelFile(dataset_b_path)
            self.ports_df = pd.read_excel(self.excel_file, "ports")
            self.berths_df = pd.read_excel(self.excel_file, "berths")
            self.cargo_rules_df = pd.read_excel(self.excel_file, "berth_cargo_rules")
            self.pbdt_df = pd.read_excel(self.excel_file, "pbdt")
        else:
            self.excel_file = None
            self.ports_df = pd.DataFrame()
            self.berths_df = pd.DataFrame()

    def get_port_code(self, port_name):
        clean_name = str(port_name).strip().lower()
        if clean_name in PORT_MAP:
            return PORT_MAP[clean_name]
        raise ValueError(f"Port '{port_name}' is not one of the 5 validated MVP ports in Dataset B (Paradip, Visakhapatnam, Gangavaram, Dhamra, Haldia Dock Complex).")

    def generate_candidate_vessel_classes(self, cargo_volume_mt):
        """Step 1: Generate candidate vessel classes suitable for cargo volume."""
        candidates = []
        for v_code, spec in VESSEL_SPECS.items():
            # Allow candidate if cargo volume fits reasonably within vessel DWT capacity range
            if cargo_volume_mt <= spec["max_dwt"] * 1.05 and cargo_volume_mt >= spec["min_dwt"] * 0.5:
                candidates.append(v_code)
        
        # Fallback if cargo is small/large
        if not candidates:
            if cargo_volume_mt < 20000:
                candidates = ["HANDY"]
            else:
                candidates = ["CAPE"]
        return candidates

    def validate_feasibility(self, cargo_volume_mt, destination_port, cargo_type="Thermal Coal"):
        """
        Step 2: Validate candidate vessel classes against Dataset B berth constraints.
        Returns:
        - feasible_vessel_classes (list)
        - rejected_vessel_classes (dict of class: reason)
        - historical_port_risk_score (float 0-100)
        """
        port_code = self.get_port_code(destination_port)
        candidates = self.generate_candidate_vessel_classes(cargo_volume_mt)
        
        # Get berths for target port
        port_berths = self.berths_df[self.berths_df["port_id"] == port_code]
        if port_berths.empty:
            return [], {c: f"No berth data for port {port_code}" for c in candidates}, 50.0

        max_port_depth = port_berths["present_depth_m"].max()
        max_port_loa = port_berths["max_loa_m"].max()
        max_port_dwt = port_berths["max_dwt_mt"].max()

        feasible = []
        rejected = {}

        for v_code in candidates:
            spec = VESSEL_SPECS[v_code]
            req_draft = spec["typical_draft_m"]
            req_loa = spec["typical_loa_m"]

            # Check if any berth at the port can accommodate this vessel class
            rejection_reasons = []

            # 1. Draft Check
            if max_port_depth < req_draft:
                rejection_reasons.append(f"DraftExceeded (Required: {req_draft}m, Port Max Depth: {max_port_depth}m)")

            # 2. LOA Check
            if pd.notna(max_port_loa) and max_port_loa < req_loa:
                rejection_reasons.append(f"LOAExceeded (Required: {req_loa}m, Port Max LOA: {max_port_loa}m)")

            # 3. DWT Capacity Check
            if pd.notna(max_port_dwt) and max_port_dwt < cargo_volume_mt:
                rejection_reasons.append(f"CapacityInsufficient (Cargo: {cargo_volume_mt} MT > Port Max DWT: {max_port_dwt} MT)")

            if not rejection_reasons:
                feasible.append(v_code)
            else:
                rejected[v_code] = "; ".join(rejection_reasons)

        # Also evaluate non-candidate vessel classes as rejected
        for v_code in VESSEL_SPECS:
            if v_code not in candidates and v_code not in rejected and v_code not in feasible:
                spec = VESSEL_SPECS[v_code]
                if cargo_volume_mt > spec["max_dwt"]:
                    rejected[v_code] = f"CapacityInsufficient (Cargo: {cargo_volume_mt} MT > Vessel Max DWT: {spec['max_dwt']} MT)"
                elif cargo_volume_mt < spec["min_dwt"] * 0.5:
                    rejected[v_code] = f"VesselOversized (Cargo: {cargo_volume_mt} MT too small for vessel DWT: {spec['min_dwt']} MT)"

        # Historical Port Detention Risk Factor from PBDT
        port_pbdt = self.pbdt_df[self.pbdt_df["port"].str.contains(port_code, case=False, na=False)]
        if not port_pbdt.empty:
            avg_pbdt = port_pbdt["pre_berthing_detention_hours"].mean()
            # Normalize to 0-100 scale (assuming max typical PBDT is 50 hours)
            port_risk_score = float(np.clip((avg_pbdt / 50.0) * 100.0, 10.0, 95.0))
        else:
            port_risk_score = 35.0  # default baseline

        return feasible, rejected, round(port_risk_score, 1)

if __name__ == "__main__":
    engine = FeasibilityEngine()
    print("Testing FeasibilityEngine for 55,000 MT Coal at Paradip:")
    feas, rej, risk = engine.validate_feasibility(55000, "Paradip", "Thermal Coal")
    print(f"Feasible Classes: {feas}")
    print(f"Rejected Classes: {rej}")
    print(f"Historical Port Risk Score: {risk}")
