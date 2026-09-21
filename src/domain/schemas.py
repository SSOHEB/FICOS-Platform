"""
FICOS — Domain Schemas
Typed dataclasses for all core domain entities.

These are the stable domain objects used across the entire platform.
DO NOT use plain dicts for domain entities — always use these classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


# ──────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class UncertaintyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DecisionType(str, Enum):
    BUY_NOW = "BUY_NOW"
    WAIT = "WAIT"
    FLEXIBLE_INDEX = "FLEXIBLE_INDEX"
    ALTERNATIVE_PORT = "ALTERNATIVE_PORT"
    REJECT_SHIPMENT = "REJECT_SHIPMENT"


class ModelStatus(str, Enum):
    PROMOTED = "promoted"
    SECONDARY = "secondary"
    FALLBACK = "fallback"
    EXCLUDED = "excluded"


class ScenarioType(str, Enum):
    NORMAL = "NORMAL"
    FREIGHT_SPIKE = "FREIGHT_SPIKE"
    FREIGHT_DROP = "FREIGHT_DROP"
    PORT_DISRUPTION = "PORT_DISRUPTION"
    WEATHER_DISRUPTION = "WEATHER_DISRUPTION"
    GEOPOLITICAL_SHOCK = "GEOPOLITICAL_SHOCK"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"


# ──────────────────────────────────────────────────────────────
# CARGO & ROUTE
# ──────────────────────────────────────────────────────────────

@dataclass
class CargoRequirement:
    """Describes what needs to be shipped."""
    cargo_type: str = "Thermal Coal"         # e.g. "Thermal Coal", "Coking Coal"
    quantity_mt: float = 75000.0             # metric tonnes
    origin: str = "TUBARAO"                  # e.g. "australia", "indonesia"
    destination: str = "QINGDAO"             # e.g. "paradip", "dhamra"
    decision_date: str = "2026-09-14"        # ISO date string "YYYY-MM-DD"
    delivery_deadline_days: Optional[int] = None  # max days before delivery required
    candidate_vessel_classes: Optional[List[str]] = None  # restrict to these codes
    cargo_id: Optional[str] = None
    asset_type: Optional[str] = None
    origin_port_code: Optional[str] = None
    dest_port_code: Optional[str] = None
    laycan_days: Optional[int] = None

    def __post_init__(self):
        if not self.asset_type and self.cargo_type:
            self.asset_type = self.cargo_type
        elif not self.cargo_type and self.asset_type:
            self.cargo_type = self.asset_type
        if not self.cargo_id:
            self.cargo_id = f"CARGO-{uuid.uuid4().hex[:8].upper()}" if 'uuid' in globals() else "CARGO-DEFAULT"
        if not self.origin_port_code and self.origin:
            self.origin_port_code = self.origin
        if not self.dest_port_code and self.destination:
            self.dest_port_code = self.destination
        if self.laycan_days is None:
            self.laycan_days = self.delivery_deadline_days or 14


@dataclass
class Route:
    """Represents a shipping route."""
    origin: str = "TUBARAO"
    destination: str = "QINGDAO"
    typical_voyage_days: Optional[int] = None  # None if unknown
    via_suez: bool = False
    via_panama: bool = False
    notes: str = ""
    distance_nautical_miles: float = 5000.0
    estimated_speed_knots: float = 14.0
    requires_suez: bool = False
    requires_panama: bool = False
    origin_code: str = ""
    dest_code: str = ""

    def __post_init__(self):
        if not self.origin_code:
            self.origin_code = self.origin
        if not self.dest_code:
            self.dest_code = self.destination
        if self.via_suez:
            self.requires_suez = True
        if self.via_panama:
            self.requires_panama = True


# ──────────────────────────────────────────────────────────────
# VESSEL
# ──────────────────────────────────────────────────────────────

@dataclass
class VesselClass:
    """Represents a vessel class specification loaded from config."""
    code: str = "PANA"                 # e.g. "SUPRA", "PANA"
    display_name: str = "Panamax"      # e.g. "Supramax"
    freight_index: str = "panamax"     # e.g. "supramax" (Dataset A column)
    dwt_min: float = 60000.0
    dwt_max: float = 85000.0
    typical_dwt: float = 75000.0
    typical_draft_m: float = 14.5
    typical_loa_m: float = 229.0
    typical_beam_m: float = 32.3
    cargo_types: List[str] = field(default_factory=list)
    notes: str = ""
    name: str = ""
    max_dwt: float = 0.0
    max_draft_m: float = 0.0
    max_beam_m: float = 0.0
    max_loa_m: float = 0.0
    speed_knots: float = 14.0
    fuel_consumption_laden_tpd: float = 30.0
    fuel_consumption_port_tpd: float = 3.0
    loading_rate_tpd: float = 25000.0
    discharge_rate_tpd: float = 20000.0
    gross_tonnage: float = 43000.0


    def __post_init__(self):
        if not self.name:
            self.name = self.display_name or self.code
        if self.max_dwt == 0.0:
            self.max_dwt = self.typical_dwt or self.dwt_max
        if self.max_draft_m == 0.0:
            self.max_draft_m = self.typical_draft_m
        if self.max_beam_m == 0.0:
            self.max_beam_m = self.typical_beam_m
        if self.max_loa_m == 0.0:
            self.max_loa_m = self.typical_loa_m


# ──────────────────────────────────────────────────────────────
# PORT
# ──────────────────────────────────────────────────────────────

@dataclass
class PortConstraints:
    max_draft_m: float
    max_loa_m: float
    max_dwt_mt: float


@dataclass
class Port:
    """Represents a port loaded from config."""
    code: str                    # e.g. "PARA", "VIZAG"
    display_name: str
    key: str                     # e.g. "paradip" (config key)
    country: str
    state: str
    constraints: PortConstraints
    cargo_types_supported: List[str] = field(default_factory=list)
    tidal_port: bool = False
    monsoon_restrictions: bool = False
    operational_notes: str = ""
    name: str = ""
    max_draft_m: float = 0.0
    max_loa_m: float = 0.0
    max_beam_m: float = 45.0
    max_dwt: float = 0.0
    avg_congestion_delay_hours: float = 12.0

    def __post_init__(self):
        if not self.name:
            self.name = self.display_name or self.code
        if isinstance(self.constraints, PortConstraints):
            if self.max_draft_m == 0.0:
                self.max_draft_m = self.constraints.max_draft_m
            if self.max_loa_m == 0.0:
                self.max_loa_m = self.constraints.max_loa_m
            if self.max_dwt == 0.0:
                self.max_dwt = self.constraints.max_dwt_mt



# ──────────────────────────────────────────────────────────────
# FEASIBILITY
# ──────────────────────────────────────────────────────────────

@dataclass
class ConstraintCheck:
    """Single constraint pass/fail result."""
    name: str
    passed: bool
    required_value: Any
    actual_value: Any
    reason: str = ""


@dataclass
class FeasibilityResult:
    """Structured result from FeasibilityEngine.check()."""
    vessel_code: str
    port_code: str
    feasible: bool
    passed_constraints: List[ConstraintCheck] = field(default_factory=list)
    failed_constraints: List[ConstraintCheck] = field(default_factory=list)
    limiting_constraint: Optional[str] = None
    port_risk_score: float = 35.0   # 0-100, from Dataset B PBDT
    explanation: str = ""
    is_feasible: bool = True
    checks: List[ConstraintCheck] = field(default_factory=list)

    def __post_init__(self):
        self.is_feasible = self.feasible
        if not self.checks:
            self.checks = self.passed_constraints + self.failed_constraints


@dataclass
class VesselFeasibilityRanking:
    """Ranked vessel option with cost estimate."""
    vessel_code: str
    vessel_name: str
    feasible: bool
    feasibility_result: Optional[FeasibilityResult]
    estimated_freight_cost_usd: Optional[float]
    expected_total_cost_usd: Optional[float]
    rank: Optional[int]
    rationale: str = ""


# ──────────────────────────────────────────────────────────────
# FORECAST
# ──────────────────────────────────────────────────────────────

@dataclass
class UncertaintyResult:
    p10: float
    p50: float
    p90: float
    confidence_interval_width: float = 0.0


@dataclass
class ForecastResult:
    """Structured output from ForecastService."""
    asset: str
    forecast_date: str
    horizon_days: int
    current_rate: float
    point_forecast: float             # best estimate of future rate
    p10: float                        # 10th percentile (pessimistic)
    p50: float                        # 50th percentile (median)
    p90: float                        # 90th percentile (optimistic)
    expected_delta: float             # point_forecast - current_rate
    expected_pct_change: float        # delta / current_rate
    model_name: str
    model_version: str
    confidence: UncertaintyLevel
    uncertainty_width: float          # p90 - p10
    is_promoted: bool = True
    fallback_used: bool = False
    feature_metadata: Dict[str, Any] = field(default_factory=dict)
    asset_type: str = ""
    production_status: str = "promoted"
    validation_status: str = "production_validated"
    economic_evidence: str = "inconclusive"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.asset_type:
            self.asset_type = self.asset.upper()

    @property
    def uncertainty(self) -> UncertaintyResult:
        return UncertaintyResult(
            p10=self.p10,
            p50=self.p50,
            p90=self.p90,
            confidence_interval_width=self.uncertainty_width or (self.p90 - self.p10)
        )

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Expose full forecast provenance metadata."""
        if self.provenance:
            return self.provenance
        conf_str = self.confidence.value if hasattr(self.confidence, "value") else str(self.confidence)
        return {
            "vessel_class": self.asset_type,
            "horizon": f"{self.horizon_days}d",
            "forecast_value": self.point_forecast,
            "uncertainty": {
                "p10": self.p10,
                "p50": self.p50,
                "p90": self.p90,
                "spread": round(self.p90 - self.p10, 2),
                "level": conf_str
            },
            "model_used": self.model_name,
            "model_version": self.model_version,
            "production_status": self.production_status,
            "validation_status": self.validation_status,
            "economic_evidence": self.economic_evidence,
            "fallback_used": self.fallback_used
        }


# ──────────────────────────────────────────────────────────────
# RISK
# ──────────────────────────────────────────────────────────────

@dataclass
class RiskAlert:
    alert_type: str     # e.g. "CYCLONE_PROXIMITY", "GDELT_BURST"
    description: str
    severity: RiskLevel
    category: str = "GENERAL"
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = self.description


@dataclass
class RiskResult:
    """Structured output from RiskEngine."""
    overall_level: RiskLevel
    weather_score: float          # 0-100
    geopolitical_score: float     # 0-100
    disruption_score: float       # 0-100 (max of weather + geopolitical)
    operational_risk_score: float # from Dataset B port PBDT
    active_alerts: List[RiskAlert] = field(default_factory=list)
    decision_adjustment: str = ""  # human-readable adjustment description
    risk_cost_premium_usd: float = 0.0
    explanation: str = ""
    overall_risk_score: float = 0.0
    weather_risk_level: str = "LOW"
    disruption_risk_level: str = "LOW"
    key_risk_drivers: List[str] = field(default_factory=list)
    risk_alerts: List[RiskAlert] = field(default_factory=list)

    def __post_init__(self):
        if self.overall_risk_score == 0.0:
            self.overall_risk_score = max(self.weather_score, self.geopolitical_score, self.disruption_score)
        self.weather_risk_level = self.overall_level.value if isinstance(self.overall_level, RiskLevel) else str(self.overall_level)
        self.disruption_risk_level = self.weather_risk_level
        if not self.risk_alerts:
            self.risk_alerts = self.active_alerts



# ──────────────────────────────────────────────────────────────
# COST MODEL
# ──────────────────────────────────────────────────────────────

@dataclass
class CostComponent:
    """A single cost item with source and assumption flag."""
    name: str
    value_usd: float = 0.0
    amount_usd: float = 0.0
    unit: str = "USD"
    description: str = ""
    assumption: bool = True
    source: str = ""
    confidence: str = "LOW"

    def __post_init__(self):
        if self.amount_usd == 0.0 and self.value_usd != 0.0:
            self.amount_usd = self.value_usd
        elif self.value_usd == 0.0 and self.amount_usd != 0.0:
            self.value_usd = self.amount_usd
        if not self.description:
            self.description = f"{self.name}: ${self.amount_usd:,.2f}"



@dataclass
class PolicyEvaluation:
    """Evaluation result for a candidate chartering policy strategy."""
    strategy_name: str
    expected_cost_usd: float
    adjusted_expected_cost_usd: float
    cost_per_mt: float
    cost_breakdown: CostBreakdown
    risk_score: float
    is_feasible: bool
    policy_reasoning: str = ""


@dataclass
class CostBreakdown:
    """Full cost breakdown from CostModel."""
    freight_cost_usd: float = 0.0
    waiting_cost_usd: float = 0.0
    delay_cost_usd: float = 0.0
    demurrage_cost_usd: float = 0.0
    operational_risk_cost_usd: float = 0.0
    risk_premium_usd: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_mt: float = 0.0
    currency: str = "USD"
    components: List[CostComponent] = field(default_factory=list)
    assumptions_applied: List[str] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)



@dataclass
class IdleTimeAssessment:
    """Idle / waiting economics assessment."""
    expected_idle_days: float = 0.0
    waiting_cost_usd: float = 0.0
    delay_cost_usd: float = 0.0
    alternative_employment_value_usd: float = 0.0
    net_idle_cost_usd: float = 0.0
    recommendation: str = ""
    assumptions: List[str] = field(default_factory=list)
    estimated_idle_cost_usd: float = 0.0
    demurrage_rate_usd_per_day: float = 20000.0
    primary_drivers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.estimated_idle_cost_usd == 0.0 and self.waiting_cost_usd != 0.0:
            self.estimated_idle_cost_usd = self.waiting_cost_usd
        elif self.waiting_cost_usd == 0.0 and self.estimated_idle_cost_usd != 0.0:
            self.waiting_cost_usd = self.estimated_idle_cost_usd



# ──────────────────────────────────────────────────────────────
# CHARTER DECISION
# ──────────────────────────────────────────────────────────────

@dataclass
class CharterDecision:
    """Final output from DecisionEngine."""
    decision: DecisionType
    recommended_vessel: Optional[str]          # vessel code
    feasible_vessels: List[str]                # all feasible vessel codes
    rejected_vessels: Dict[str, str]           # vessel_code -> rejection reason

    current_rate: float
    forecast: Optional[ForecastResult]
    risk_result: Optional[RiskResult]

    expected_cost_now: Optional[float]         # USD
    expected_cost_wait: Optional[float]        # USD
    expected_savings_from_waiting: Optional[float]  # cost_now - cost_wait
    waiting_cost_usd: Optional[float]
    delay_cost_usd: Optional[float]
    operational_risk_cost_usd: Optional[float]

    risk_level: RiskLevel
    confidence: UncertaintyLevel

    rationale: str
    reasoning_trace: List[str] = field(default_factory=list)
    alternative_port: Optional[str] = None
    alternative_port_reason: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    vessel_rankings: List[VesselFeasibilityRanking] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# SCENARIO REPLAY
# ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Output from ScenarioEngine replay."""
    scenario: ScenarioType
    baseline_decision: DecisionType
    scenario_decision: DecisionType
    changed: bool
    cost_difference_usd: Optional[float]
    risk_difference: str
    explanation: str


# ──────────────────────────────────────────────────────────────
# BACKTEST
# ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Summary metrics from the historical decision backtest."""
    total_baseline_cost_usd: float
    total_ficos_cost_usd: float
    total_savings_usd: float
    savings_pct: float
    avg_savings_per_shipment_usd: float
    median_savings_usd: float
    n_buy_now: int
    n_wait: int
    n_flexible: int
    n_infeasible: int
    n_total: int
    coverage_pct: float
    worst_loss_usd: float
    best_saving_usd: float
    assumptions_disclaimer: str = (
        "Scenario-based economic backtest under stated assumptions. "
        "Results do not represent actual SAIL or shipper procurement outcomes."
    )


# ──────────────────────────────────────────────────────────────
# MODEL REGISTRY ENTRY
# ──────────────────────────────────────────────────────────────

@dataclass
class ModelRegistryEntry:
    """A single entry in the model registry manifest."""
    asset: str
    horizon_days: int
    model_type: str                    # exact class, e.g. "RandomForestRegressor"
    status: ModelStatus
    model_artifact_path: Optional[str]
    training_start: Optional[str]
    training_end: Optional[str]
    feature_list_path: Optional[str]
    uncertainty_path: Optional[str]
    preprocessing_metadata: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    walkforward_metrics: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "1.0.0"
    created_at: str = ""
    promoted_at: Optional[str] = None
    notes: str = ""
    p10_bound: Optional[float] = None
    p90_bound: Optional[float] = None
    optimal_tau: Optional[float] = None
    historical_precision: Optional[float] = None
    production_status: Optional[str] = None
    validation_status: Optional[str] = None
    economic_evidence: Optional[str] = "inconclusive"
