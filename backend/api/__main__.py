"""
FICOS CLI Entrypoint
Provides command-line interface for requesting recommendations, inspecting registry,
and running scenario analyses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.api.recommendation_service import RecommendationService
from ml.registry.registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(
        description="FICOS — Freight Intelligence & Chartering Optimization System CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate cargo requirement & get decision recommendation")
    eval_parser.add_argument("--asset", type=str, default="PANAMAX_1D", help="Asset type (e.g. PANAMAX_1D, CAPESIZE_10D)")
    eval_parser.add_argument("--quantity", type=float, default=75000.0, help="Cargo quantity in metric tons")
    eval_parser.add_argument("--origin", type=str, default="TUBARAO", help="Origin port name or code")
    eval_parser.add_argument("--dest", type=str, default="QINGDAO", help="Destination port name or code")
    eval_parser.add_argument("--laycan", type=int, default=14, help="Laycan horizon days")
    eval_parser.add_argument("--distance", type=float, default=11000.0, help="Route distance in nautical miles")
    eval_parser.add_argument("--suez", action="store_true", help="Flag if route requires Suez canal")
    eval_parser.add_argument("--panama", action="store_true", help="Flag if route requires Panama canal")
    eval_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Command: registry
    reg_parser = subparsers.add_parser("registry", help="Inspect promoted models in Model Registry")

    args = parser.parse_args()

    if args.command == "evaluate":
        service = RecommendationService()
        result = service.get_recommendation(
            asset_type=args.asset,
            quantity_mt=args.quantity,
            origin_port_code=args.origin,
            dest_port_code=args.dest,
            laycan_days=args.laycan,
            distance_nm=args.distance,
            requires_suez=args.suez,
            requires_panama=args.panama
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "="*65)
            print(" FICOS FREIGHT DECISION RECOMMENDATION")
            print("="*65)
            print(f"Recommendation ID : {result.recommendation_id}")
            print(f"Asset / Cargo     : {result.cargo.asset_type} ({result.cargo.quantity_mt:,.0f} MT)")
            print(f"Route             : {result.origin_port.name} -> {result.dest_port.name}")
            print(f"Vessel Class      : {result.selected_vessel.name}")
            print("-" * 65)
            print(f"RECOMMENDED ACTION: {result.recommended_strategy}")
            print(f"Expected Total Cost: ${result.recommended_cost_usd:,.2f}")
            print(f"Cost per MT       : ${result.recommended_cost_per_mt:.2f}/MT")
            print("-" * 65)
            print(f"Freight Rate Forecast: ${result.forecast.point_forecast:.2f}/MT (P10: ${result.forecast.uncertainty.p10:.2f}, P90: ${result.forecast.uncertainty.p90:.2f})")
            print(f"Physical Feasibility : {'PASSED [OK]' if result.feasibility.is_feasible else 'FAILED [VIOLATION]'}")
            print(f"Overall Risk Score   : {result.risk.overall_risk_score:.1f}/100 ({result.risk.weather_risk_level} Weather / {result.risk.disruption_risk_level} Disruption)")
            print("-" * 65)
            if result.warnings_and_alerts:
                print("WARNINGS & ALERTS:")
                for w in result.warnings_and_alerts:
                    print(f"  * {w}")
                print("-" * 65)
            print("DECISION RATIONALE:")
            print(f"  {result.explanation['summary_rationale']}")
            print("="*65 + "\n")

    elif args.command == "registry":
        reg = ModelRegistry()
        promoted = reg.list_promoted_models()
        print("\n" + "="*65)
        print(" FICOS MODEL REGISTRY — PROMOTED MODELS")
        print("="*65)
        for p in promoted:
            asset = p.get("asset", "?")
            h = p.get("horizon_days", "?")
            mtype = p.get("model_type", "?")
            f1 = p.get("validation_metrics", {}).get("f1", 0.0)
            status = p.get("status", "?")
            print(f"Asset: {asset:<15} | Horizon: {h}d | Model: {mtype:<25} | F1: {f1:.3f} | status={status}")
        print("="*65 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
