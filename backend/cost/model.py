"""
FICOS — Cost Model Engine
Calculates voyage and charter costs, bunker costs, port fees, canal transit costs,
and idle-time risk penalties based on configs/cost_model.yaml.

DISCLAIMER:
  Default parameters (e.g. VLSFO $650/MT, port fees $15,000/call) are simplified
  industry benchmarks for structural decision logic. Real-world execution must
  override these with live market parameters via cost_model.yaml or direct arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from src.domain.schemas import (
    CargoRequirement, VesselClass, Port, Route, CostBreakdown, CostComponent
)


class CostModel:
    """
    Evaluates total expected cost for executing a cargo requirement on a vessel class
    under spot or time charter strategy.
    """

    def __init__(self, config_path: str | Path = "configs/cost_model.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {
                "bunker_prices": {"vlsfo_usd_ton": 650.0, "mgo_usd_ton": 850.0},
                "port_costs": {"fixed_call_usd": 15000.0, "per_gt_usd": 0.15},
                "canal_fees": {"suez_usd": 300000.0, "panama_usd": 250000.0},
                "idle_time_penalties": {
                    "demurrage_per_day_usd": 20000.0,
                    "opportunity_cost_per_day_usd": 15000.0
                }
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def calculate_cost(
        self,
        cargo: CargoRequirement,
        vessel: VesselClass,
        origin_port: Port,
        dest_port: Port,
        route: Route,
        freight_rate_usd_ton: float,
        strategy: str = "SPOT",
        expected_wait_days: float = 0.0,
        fuel_price_override: Optional[float] = None
    ) -> CostBreakdown:
        """
        Calculates total voyage cost breakdown.

        Args:
            cargo: Cargo requirement (quantity in MT, asset type, dates)
            vessel: Assigned vessel class specifications
            origin_port: Origin port details
            dest_port: Destination port details
            route: Route details (distance NM, speed, transit time, canal flags)
            freight_rate_usd_ton: Forecasted or actual freight rate per MT
            strategy: 'SPOT' or 'TIME_CHARTER'
            expected_wait_days: Expected congestion/idle delay in days
            fuel_price_override: Optional VLSFO price per MT override
        """
        quantity_mt = float(cargo.quantity_mt) if isinstance(cargo.quantity_mt, (int, float)) else 75000.0
        speed_kts = float(vessel.speed_knots) if hasattr(vessel, 'speed_knots') and isinstance(vessel.speed_knots, (int, float)) else 14.0
        distance_nm = float(route.distance_nautical_miles) if hasattr(route, 'distance_nautical_miles') and isinstance(route.distance_nautical_miles, (int, float)) else 5000.0

        # Voyage duration
        steaming_days = distance_nm / (speed_kts * 24.0) if speed_kts > 0 else 0.0
        loading_rate = float(vessel.loading_rate_tpd) if hasattr(vessel, 'loading_rate_tpd') and isinstance(vessel.loading_rate_tpd, (int, float)) and vessel.loading_rate_tpd > 0 else 25000.0
        discharge_rate = float(vessel.discharge_rate_tpd) if hasattr(vessel, 'discharge_rate_tpd') and isinstance(vessel.discharge_rate_tpd, (int, float)) and vessel.discharge_rate_tpd > 0 else 20000.0

        loading_days = quantity_mt / loading_rate
        discharge_days = quantity_mt / discharge_rate
        total_voyage_days = steaming_days + loading_days + discharge_days + expected_wait_days


        # 1. Base Charter / Freight Cost
        if strategy == "SPOT":
            base_charter_cost = quantity_mt * freight_rate_usd_ton
            charter_desc = f"Spot Freight ({quantity_mt:,.0f} MT @ ${freight_rate_usd_ton:.2f}/MT)"
        else: # TIME_CHARTER
            tc_daily_rate = freight_rate_usd_ton  # assuming rate passed is daily TC rate or equivalent
            base_charter_cost = total_voyage_days * tc_daily_rate
            charter_desc = f"Time Charter ({total_voyage_days:.1f} days @ ${tc_daily_rate:,.0f}/day)"

        # 2. Bunker Cost
        bunker_prices = self.config.get("bunker_prices", {})
        vlsfo_price = fuel_price_override if fuel_price_override is not None else bunker_prices.get("vlsfo_usd_ton", 650.0)
        
        steaming_fuel_used = steaming_days * vessel.fuel_consumption_laden_tpd
        idle_port_fuel_used = (loading_days + discharge_days + expected_wait_days) * vessel.fuel_consumption_port_tpd
        total_fuel_mt = steaming_fuel_used + idle_port_fuel_used
        bunker_cost = total_fuel_mt * vlsfo_price

        # 3. Port Costs
        port_cfg = self.config.get("port_costs", {})
        fixed_port_fee = port_cfg.get("fixed_call_usd", 15000.0)
        gt_rate = port_cfg.get("per_gt_usd", 0.15)
        
        origin_port_fee = fixed_port_fee + (vessel.gross_tonnage * gt_rate)
        dest_port_fee = fixed_port_fee + (vessel.gross_tonnage * gt_rate)
        total_port_cost = origin_port_fee + dest_port_fee

        # 4. Canal Transit Fees
        canal_fees_cfg = self.config.get("canal_fees", {})
        canal_cost = 0.0
        if route.requires_suez:
            canal_cost += canal_fees_cfg.get("suez_usd", 300000.0)
        if route.requires_panama:
            canal_cost += canal_fees_cfg.get("panama_usd", 250000.0)

        # 5. Idle / Congestion Delay Cost
        penalty_cfg = self.config.get("idle_time_penalties", {})
        demurrage_rate = penalty_cfg.get("demurrage_per_day_usd", 20000.0)
        opp_cost_rate = penalty_cfg.get("opportunity_cost_per_day_usd", 15000.0)
        idle_time_cost = expected_wait_days * (demurrage_rate + opp_cost_rate)

        # Build Components
        components = [
            CostComponent(
                name="Charter Freight Cost",
                amount_usd=base_charter_cost,
                description=charter_desc
            ),
            CostComponent(
                name="Bunker Fuel Cost",
                amount_usd=bunker_cost,
                description=f"VLSFO {total_fuel_mt:.1f} MT @ ${vlsfo_price:.2f}/MT"
            ),
            CostComponent(
                name="Port Disbursements",
                amount_usd=total_port_cost,
                description=f"Origin: ${origin_port_fee:,.0f}, Dest: ${dest_port_fee:,.0f}"
            ),
            CostComponent(
                name="Canal Transit Fees",
                amount_usd=canal_cost,
                description=f"Suez: {route.requires_suez}, Panama: {route.requires_panama}"
            ),
            CostComponent(
                name="Idle / Delay Risk Penalty",
                amount_usd=idle_time_cost,
                description=f"Expected Wait: {expected_wait_days:.1f} days @ ${demurrage_rate+opp_cost_rate:,.0f}/day"
            )
        ]

        total_usd = sum(c.amount_usd for c in components)
        unit_cost_usd_ton = total_usd / quantity_mt if quantity_mt > 0 else 0.0

        return CostBreakdown(
            total_cost_usd=total_usd,
            cost_per_mt=unit_cost_usd_ton,
            components=components,
            currency="USD",
            assumptions={
                "strategy": strategy,
                "freight_rate_usd_ton": freight_rate_usd_ton,
                "vlsfo_price_usd_ton": vlsfo_price,
                "steaming_days": round(steaming_days, 2),
                "total_voyage_days": round(total_voyage_days, 2),
                "expected_wait_days": expected_wait_days
            }
        )
