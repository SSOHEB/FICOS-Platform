import React, { useEffect, useMemo, useState } from 'react';
import { Info, ArrowRight, ShieldAlert, Activity, TrendingUp, Sliders, CheckCircle2, BarChart3 } from 'lucide-react';
import {
  MOCK_INPUTS,
  MOCK_MARKET_SUMMARY,
  MOCK_FORECAST_DATA,
  MOCK_HORIZON_COMPARISON,
  MOCK_RECOMMENDATION,
  MOCK_RISK_BREAKDOWN,
  MOCK_CONFIDENCE_METRICS,
  Timeframe,
} from './dashboardData';
import { apiGet, ApiStatus } from './apiClient';

type EndpointState<T> = {
  data: T | null;
  status: ApiStatus;
  loading: boolean;
  error: string | null;
};

type ForecastResponse = {
  vessel_class: string;
  horizon: string;
  origin: string;
  destination: string;
  cargo_qty_mt: number;
  current_rate: number;
  p10: number;
  p50: number;
  p90: number;
  expected_delta: number;
  expected_pct_change: number;
  recommended_action: string;
  action_rationale: string;
  confidence_tier: string;
  historical_precision: number;
  coverage_status: string;
  is_promoted: boolean;
  fallback_used: boolean;
  model_type: string;
  inference_status?: string;
};

type RiskCategory = {
  name: string;
  percent: number;
  score?: number;
  color?: string;
  impact: string;
  description: string;
};

type RiskResponse = {
  destination_port: string;
  route: string;
  overall_risk_score: number;
  overall_level: string;
  caption: string;
  driver_summary: string;
  categories: RiskCategory[];
};

type ShockPoint = {
  day: number;
  surge_pct: number;
  label: string;
};

type ShockResponse = {
  event_details?: {
    title: string;
    date_range: string;
    surge_peak: string;
    peak_day: string;
    recovery_time: string;
    description: string;
    mitigation_advice: string;
    impulse_curve: ShockPoint[];
  };
  comparison_events?: string[];
  available_events?: string[];
  message?: string;
};

const riskColors = ['#1D3045', '#E05A47', '#486581', '#829AB1'];
const dashboardTimeframes: Timeframe[] = ['1D', '7D', '14D', '30D', '60D', '90D'];
const shockComparisonPaths = [
  'M 30 180 L 100 130 L 200 25 L 350 90 L 550 145 L 750 175',
  'M 30 180 L 140 160 L 250 85 L 420 135 L 620 168 L 750 178',
];

const emptyEndpoint = <T,>(): EndpointState<T> => ({
  data: null,
  status: 'idle',
  loading: false,
  error: null,
});

const parseQuantity = (value: string) => {
  const parsed = Number(value.replace(/[^0-9.]/g, ''));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 75000;
};

const inferVesselClass = (quantityMt: number) => {
  if (quantityMt >= 120000) return 'cape';
  if (quantityMt >= 65000) return 'panamax';
  if (quantityMt >= 45000) return 'supramax';
  return 'handy';
};

const horizonToApi = (tf: Timeframe) => tf.toLowerCase();
const formatMoney = (value: number) => `$${value.toFixed(2)}`;
const formatPct = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
const isFallbackStatus = (status: ApiStatus) => status === 'fallback' || status === 'unavailable';
const isDirectionalAction = (action?: string) => action === 'BUY NOW' || action === 'WAIT';

const LoadingPill = ({ label }: { label: string }) => (
  <span className="inline-flex items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-[#1D3045]/60 font-semibold">
    <span className="w-2 h-2 rounded-full bg-[#E05A47] animate-pulse" />
    {label}
  </span>
);

const InlineNotice = ({ tone = 'navy', children }: { tone?: 'navy' | 'coral'; children: React.ReactNode }) => (
  <div className={`rounded-xl border px-4 py-3 text-xs font-mono leading-relaxed ${
    tone === 'coral'
      ? 'bg-[#E05A47]/10 border-[#E05A47]/20 text-[#8f372b]'
      : 'bg-[#1D3045]/5 border-[#1D3045]/12 text-[#1D3045]/75'
  }`}>
    {children}
  </div>
);

interface DashboardProps {
  onBackToLanding: () => void;
  onNavigate?: (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => void;
}

export default function Dashboard({ onBackToLanding, onNavigate }: DashboardProps) {
  const [timeframe, setTimeframe] = useState<Timeframe>('30D');
  const [hoveredRiskCategory, setHoveredRiskCategory] = useState<string | null>(null);
  const [hoveredSparklineIndex, setHoveredSparklineIndex] = useState<number | null>(null);
  const [hoveredHorizon, setHoveredHorizon] = useState<string | null>(null);

  const goTo = (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => {
    if (onNavigate) {
      onNavigate(view);
    } else if (view === 'landing') {
      onBackToLanding();
    }
  };

  // Cargo input states initialized from centralized mockData config
  const [cargo, setCargo] = useState(MOCK_INPUTS.cargo);
  const [quantity, setQuantity] = useState(MOCK_INPUTS.quantity);
  const [origin, setOrigin] = useState(MOCK_INPUTS.origin);
  const [destination, setDestination] = useState(MOCK_INPUTS.destination);
  const [windowPeriod, setWindowPeriod] = useState(MOCK_INPUTS.windowPeriod);
  const [analysisRun, setAnalysisRun] = useState(0);
  const [forecastState, setForecastState] = useState<EndpointState<ForecastResponse>>(emptyEndpoint);
  const [riskState, setRiskState] = useState<EndpointState<RiskResponse>>(emptyEndpoint);
  const [shockState, setShockState] = useState<EndpointState<ShockResponse>>(emptyEndpoint);
  const [horizonForecasts, setHorizonForecasts] = useState<Partial<Record<Timeframe, ForecastResponse>>>({});

  const quantityMt = useMemo(() => parseQuantity(quantity), [quantity]);
  const vesselClass = useMemo(() => inferVesselClass(quantityMt), [quantityMt]);
  const route = useMemo(() => `${origin}-${destination}`, [origin, destination]);

  useEffect(() => {
    const controller = new AbortController();
    const commonParams = {
      cargo,
      origin,
      destination,
      quantity: quantityMt,
      cargo_qty: quantityMt,
      route,
    };

    setForecastState((prev) => ({ ...prev, loading: true, error: null }));
    apiGet<ForecastResponse>('/forecast', {
      ...commonParams,
      vessel_class: vesselClass,
      horizon: horizonToApi(timeframe),
    }, controller.signal)
      .then((response) => setForecastState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setForecastState((prev) => ({ ...prev, loading: false, error: error.message || 'Forecast unavailable' }));
        }
      });

    Promise.all(
      dashboardTimeframes.map((tf) =>
        apiGet<ForecastResponse>('/forecast', {
          ...commonParams,
          vessel_class: vesselClass,
          horizon: horizonToApi(tf),
        }, controller.signal).then((response) => [tf, response.data] as const)
      )
    )
      .then((entries) => setHorizonForecasts(Object.fromEntries(entries) as Partial<Record<Timeframe, ForecastResponse>>))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setHorizonForecasts({});
        }
      });

    setRiskState((prev) => ({ ...prev, loading: true, error: null }));
    apiGet<RiskResponse>('/risk-breakdown', {
      ...commonParams,
      destination_port: destination,
    }, controller.signal)
      .then((response) => setRiskState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setRiskState((prev) => ({ ...prev, loading: false, error: error.message || 'Risk profile unavailable' }));
        }
      });

    setShockState((prev) => ({ ...prev, loading: true, error: null }));
    apiGet<ShockResponse>('/shock-response', {
      ...commonParams,
      event_type: 'red_sea',
    }, controller.signal)
      .then((response) => setShockState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setShockState((prev) => ({ ...prev, loading: false, error: error.message || 'Shock response unavailable' }));
        }
      });

    return () => controller.abort();
  }, [analysisRun, cargo, destination, origin, quantityMt, route, timeframe, vesselClass]);

  const isAnalyzing = forecastState.loading || riskState.loading || shockState.loading;

  const handleAnalyze = () => {
    setAnalysisRun((run) => run + 1);
  };

  const forecastData = forecastState.data;
  const currentForecast = useMemo(() => {
    const base = MOCK_FORECAST_DATA[timeframe];
    if (!forecastData) return base;

    return {
      ...base,
      p10Value: formatMoney(forecastData.p90),
      p50Value: formatMoney(forecastData.p50),
      p90Value: formatMoney(forecastData.p10),
      currentRate: formatMoney(forecastData.current_rate),
    };
  }, [forecastData, timeframe]);

  const forecastAction = forecastData?.recommended_action;
  const hasDirectionalAction = isDirectionalAction(forecastAction);
  const unpromotedFallback = isFallbackStatus(forecastState.status) || forecastData?.coverage_status === 'FALLBACK_UNPROMOTED' || forecastData?.is_promoted === false;
  const flexibleNoDirection = !!forecastData && !hasDirectionalAction;
  const forecastFallback = unpromotedFallback;
  const forecastPct = forecastData?.expected_pct_change ?? Number(MOCK_MARKET_SUMMARY.forecastChange.replace(/[^0-9.-]/g, ''));
  const flexibleRationale = unpromotedFallback
    ? 'Insufficient directional confidence at this horizon. Use flexible or index-linked coverage until a promoted signal is available.'
    : 'Promoted coverage is available, but the live model has no directional commitment at this horizon. Use flexible or index-linked execution.';
  const marketSummary = {
    currentRate: forecastData ? formatMoney(forecastData.current_rate) : MOCK_MARKET_SUMMARY.currentRate,
    currentRateUnit: MOCK_MARKET_SUMMARY.currentRateUnit,
    route: `${origin} → ${destination} route`,
    forecastChange: flexibleNoDirection ? 'FLEXIBLE' : forecastData ? formatPct(forecastData.expected_pct_change) : MOCK_MARKET_SUMMARY.forecastChange,
    forecastSubtext: flexibleNoDirection ? 'No directional commitment at this horizon' : MOCK_MARKET_SUMMARY.forecastSubtext,
    expectedCost: forecastData ? `$${Math.round(forecastData.p50 * quantityMt).toLocaleString()}` : MOCK_MARKET_SUMMARY.expectedCost,
    expectedCostBasis: `${quantityMt.toLocaleString()} MT ${forecastData?.vessel_class || vesselClass.toUpperCase()} basis`,
    marketState: flexibleNoDirection ? 'FLEXIBLE' : forecastPct >= 0 ? 'RISING' : 'SOFTENING',
    marketStateText: flexibleNoDirection ? 'Index-linked execution recommended.' : MOCK_MARKET_SUMMARY.marketStateText,
  };

  const recommendation = {
    badge: 'POLICY DECISION',
    decision: flexibleNoDirection ? 'FLEXIBLE' : forecastData?.recommended_action || MOCK_RECOMMENDATION.decision,
    confidence: unpromotedFallback ? 'FALLBACK' : forecastData?.confidence_tier || 'HIGH CONVICTION',
    rationale: flexibleNoDirection
      ? flexibleRationale
      : forecastData?.action_rationale || MOCK_RECOMMENDATION.rationale,
    vesselClass: forecastData ? `${forecastData.vessel_class} (${quantityMt.toLocaleString()} MT parcel)` : MOCK_RECOMMENDATION.vesselClass,
    targetPort: forecastData?.destination || destination,
    optimalWindow: hasDirectionalAction ? MOCK_RECOMMENDATION.optimalWindow : 'Flexible / Index-linked',
    potentialSavings: hasDirectionalAction ? 'Not computed by backend' : 'Directional savings not claimed',
  };

  const horizonComparison = useMemo(() => {
    return dashboardTimeframes
      .map((tf) => {
        const horizonForecast = horizonForecasts[tf];
        if (!horizonForecast) return null;
        const spread = Math.abs(horizonForecast.p90 - horizonForecast.p10);
        const p50Height = Math.max(12, Math.min(95, 52 + horizonForecast.expected_pct_change * 2));
        const p10Height = Math.max(p50Height + 8, Math.min(108, p50Height + spread * 2));
        const p90Height = Math.max(8, Math.min(p50Height - 8, p50Height - spread * 2));
      return {
          horizon: tf,
          p10: horizonForecast.p90,
          p50: horizonForecast.p50,
          p90: horizonForecast.p10,
          p10Height,
          p50Height,
          p90Height,
          date: horizonForecast.horizon,
        spread: formatMoney(spread),
          note: horizonForecast.coverage_status === 'FALLBACK_UNPROMOTED'
          ? 'Fallback coverage: insufficient directional confidence for this vessel horizon.'
            : horizonForecast.action_rationale,
      };
      })
      .filter(Boolean) as typeof MOCK_HORIZON_COMPARISON;
  }, [horizonForecasts]);

  const horizonChartDomain = useMemo(() => {
    const values = horizonComparison.flatMap((item) => [item.p90, item.p50, item.p10]);
    const minValue = values.length ? Math.min(...values) : 10;
    const maxValue = values.length ? Math.max(...values) : 20;
    const padding = Math.max(0.5, (maxValue - minValue) * 0.18);
    const min = Math.max(0, Math.floor((minValue - padding) * 2) / 2);
    const max = Math.ceil((maxValue + padding) * 2) / 2;
    const span = Math.max(max - min, 1);

    return {
      min,
      max,
      span,
      ticks: Array.from({ length: 5 }, (_, index) => max - (span / 4) * index),
    };
  }, [horizonComparison]);

  const riskBreakdown = useMemo(() => {
    if (!riskState.data) return MOCK_RISK_BREAKDOWN;
    return {
      overallLevel: riskState.data.overall_level,
      score: Math.round(riskState.data.overall_risk_score),
      caption: riskState.data.caption,
      driverSummary: riskState.data.driver_summary,
      categories: riskState.data.categories.map((cat, index) => ({
        ...cat,
        color: riskColors[index % riskColors.length],
      })),
    };
  }, [riskState.data]);

  const confidenceMetrics = {
    ...MOCK_CONFIDENCE_METRICS,
    scorePercent: forecastData?.historical_precision ?? MOCK_CONFIDENCE_METRICS.scorePercent,
    label: unpromotedFallback ? 'FALLBACK COVERAGE MODE' : forecastData?.coverage_status === 'COVERED_NO_DIRECTIONAL_DELTA' ? 'PROMOTED COVERAGE - NO DIRECTIONAL DELTA' : MOCK_CONFIDENCE_METRICS.label,
    uncertaintySpread: forecastData ? `${formatMoney(Math.abs(forecastData.p90 - forecastData.p10))} / MT` : MOCK_CONFIDENCE_METRICS.uncertaintySpread,
    bandWidthScore: forecastData
      ? Math.max(0, Math.min(100, Math.round(100 - (Math.abs(forecastData.p90 - forecastData.p10) / Math.max(forecastData.current_rate, 1)) * 100)))
      : MOCK_CONFIDENCE_METRICS.bandWidthScore,
    statusBadge: unpromotedFallback ? 'BASELINE' : flexibleNoDirection ? 'COVERED' : 'CALIBRATED',
    convictionLabel: unpromotedFallback ? 'LOW CONVICTION' : flexibleNoDirection ? 'NO DIRECTIONAL DELTA' : 'HIGH CONVICTION',
    intervalLabel: unpromotedFallback ? 'Fallback / unpromoted horizon' : flexibleNoDirection ? 'Promoted pair, persistence forecast' : '95% Confidence Interval',
    varianceStatus: unpromotedFallback ? 'Wide / fallback band' : flexibleNoDirection ? 'Neutral alignment' : 'Tight alignment',
    detailedExplanation: unpromotedFallback
      ? 'The backend returned a fallback forecast because this vessel-class and horizon pair is not promoted. FICOS keeps the decision capital-preserving and avoids overstating directional certainty.'
      : flexibleNoDirection
        ? 'This pair is in the promoted registry, but the production inference layer returned a persistence forecast. FICOS shows historical precision while keeping the action flexible until a real directional delta is available.'
      : MOCK_CONFIDENCE_METRICS.detailedExplanation,
  };

  const shockResponse = useMemo(() => {
    const details = shockState.data?.event_details;
    if (!details) {
      return {
        primaryEvent: {
          title: 'Shock response unavailable',
          dateRange: 'API pending',
          surgePeak: '-',
          peakDay: '-',
          recoveryTime: '-',
          description: 'Shock response data will appear here when the backend returns a calibrated event model.',
        },
        comparisonEvents: shockComparisonPaths.map((path, index) => ({
          title: `Comparison event ${index + 1}`,
          path,
        })),
        mitigationAdvice: 'No mitigation recommendation is shown until the backend returns shock-response data.',
      };
    }
    return {
      primaryEvent: {
        title: details.title,
        dateRange: details.date_range,
        surgePeak: details.surge_peak,
        peakDay: details.peak_day,
        recoveryTime: details.recovery_time,
        description: details.description,
      },
      comparisonEvents: shockComparisonPaths.map((path, index) => ({
        title: shockState.data?.comparison_events?.[index] || `Comparison event ${index + 1}`,
        path,
      })),
      mitigationAdvice: details.mitigation_advice,
    };
  }, [shockState.data]);

  // Donut chart parameters (Large 210px diameter, r=105)
  const donutR = 105;
  const donutC = 2 * Math.PI * donutR;
  let cumulativePercent = 0;
  const donutSegments = riskBreakdown.categories.map((cat) => {
    const dashLength = (cat.percent / 100) * donutC;
    const dashOffset = -(cumulativePercent / 100) * donutC;
    cumulativePercent += cat.percent;
    return {
      ...cat,
      dashLength,
      dashOffset,
    };
  });

  // Shock Response Data Points for Interactive Chart
  const shockPoints = useMemo(() => {
    const apiPoints = shockState.data?.event_details?.impulse_curve;
    if (!apiPoints?.length) {
      return [
        { day: 'Day 0', label: 'Baseline', surge: '0.0%', x: 30, y: 180 },
        { day: 'Day 3', label: 'Initial Impact', surge: '+6.2%', x: 120, y: 150 },
        { day: 'Day 7', label: 'Peak Surge', surge: '+18.4%', x: 220, y: 50 },
        { day: 'Day 14', label: 'Decay Phase', surge: '+9.1%', x: 400, y: 110 },
        { day: 'Day 22', label: 'Full Recovery', surge: '+1.5%', x: 750, y: 170 },
      ];
    }

    const maxDay = Math.max(...apiPoints.map((pt) => pt.day), 1);
    const maxSurge = Math.max(...apiPoints.map((pt) => pt.surge_pct), 25);
    return apiPoints.map((pt) => ({
      day: `Day ${pt.day}`,
      label: pt.label,
      surge: `${pt.surge_pct >= 0 ? '+' : ''}${pt.surge_pct.toFixed(1)}%`,
      x: 30 + (pt.day / maxDay) * 720,
      y: 200 - (pt.surge_pct / maxSurge) * 165,
    }));
  }, [shockState.data]);

  const primaryShockPath = shockPoints.map((pt, index) => `${index === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`).join(' ');
  const primaryShockAreaPath = `${primaryShockPath} L ${shockPoints[shockPoints.length - 1]?.x || 750} 200 L ${shockPoints[0]?.x || 30} 200 Z`;

  return (
    <div
      className="min-h-screen text-[#1D3045] font-sans antialiased selection:bg-[#E05A47] selection:text-white pb-20 relative bg-cover bg-center bg-no-repeat bg-fixed"
      style={{
        backgroundImage: 'url(/dashboard-bg.png)',
        backgroundColor: '#E2E8F0',
      }}
    >
      {/* 1. TOP NAVIGATION BAR */}
      <header className="w-full px-6 sm:px-8 md:px-12 pt-7 sm:pt-10 pb-6 flex items-center justify-between z-50">
        <nav className="hidden lg:flex items-center gap-6 xl:gap-8" aria-label="Main Navigation">
          <button
            onClick={() => goTo('landing')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045] hover:opacity-70 transition-opacity"
          >
            FICOS
          </button>
          <button
            onClick={() => goTo('landing')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            FREIGHT INTELLIGENCE
          </button>
          <button
            onClick={() => goTo('vessel-intelligence')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            VESSEL INTELLIGENCE
          </button>
          <button
            onClick={() => goTo('idle-intelligence')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            IDLE INTELLIGENCE
          </button>
          <span className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]">
            DASHBOARD
            <span className="absolute -bottom-3 left-0 right-0 h-[2px] w-full bg-[#1D3045]" />
          </span>
          <button
            onClick={() => goTo('services')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            HOW IT WORKS
          </button>
        </nav>

        <div className="lg:hidden flex items-center gap-4">
          <button
            onClick={() => goTo('landing')}
            className="text-base font-bold tracking-[0.15em] text-[#1D3045]"
          >
            FICOS
          </button>
          <span className="text-xs tracking-[0.15em] uppercase font-semibold text-[#1D3045] border-b-2 border-[#1D3045] pb-0.5">
            DASHBOARD
          </span>
        </div>

        <div className="flex items-center gap-6">
          <button
            onClick={() => goTo('landing')}
            className="text-xs tracking-[0.2em] uppercase font-medium hover:opacity-80 transition-opacity text-[#1D3045]"
          >
            ← BACK TO SITE
          </button>

          <div className="hidden sm:flex items-center gap-2">
            <span className="text-xs tracking-[0.2em] uppercase font-medium text-[#1D3045]">
              INSIGHTS
            </span>
            <div className="w-5 h-5 rounded-full bg-[#1D3045] text-white flex items-center justify-center shadow-xs">
              <Info size={10} color="#FFFFFF" />
            </div>
          </div>
        </div>
      </header>

      {/* 2. MAIN DASHBOARD CONTENT */}
      <main className="w-full px-6 sm:px-8 md:px-12 py-3 sm:py-5 space-y-7 max-w-[1824px] mx-auto">
        
        {/* Top Heading & Market State */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <p className="text-xs tracking-[0.3em] uppercase mb-1 font-semibold text-[#1D3045]/70">
              OPTIMIZATION TERMINAL
            </p>
            <h1 className="text-2xl sm:text-[2.55rem] font-light uppercase tracking-tight text-[#1D3045] drop-shadow-[0_1px_1px_rgba(255,255,255,0.8)]">
              FREIGHT INTELLIGENCE
            </h1>
            <p className="mt-1.5 text-sm sm:text-base text-[#1D3045]/80 tracking-wide font-normal">
              Forecast freight rates, evaluate vessel constraints, and turn market uncertainty into a clear chartering decision.
            </p>
          </div>

          <div className="inline-flex items-center gap-3 px-4 py-2.5 rounded-xl bg-white/95 backdrop-blur-md border border-[#1D3045]/15 shadow-[0_4px_16px_rgba(29,48,69,0.08)]">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E05A47] shadow-[0_0_10px_rgba(224,90,71,0.7)] animate-pulse" />
              <span className="text-[11px] uppercase tracking-[0.2em] font-semibold text-[#1D3045]/70">
                MARKET STATE:
              </span>
              <strong className="text-xs font-bold uppercase tracking-wider text-[#E05A47]">
                {marketSummary.marketState}
              </strong>
            </div>
            <span className="text-xs text-[#1D3045]/30">•</span>
            <span className="text-xs text-[#1D3045]/85 font-medium">
              {marketSummary.marketStateText}
            </span>
          </div>
        </div>

        {/* ZONE 1 — INPUT STRIP: Cargo move parameters */}
        <div className="bg-white/95 backdrop-blur-md rounded-2xl p-5 sm:p-6 border border-[#1D3045]/15 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] relative">
          <div className="flex items-center gap-2 mb-3 pb-2 border-b border-[#1D3045]/10">
            <Sliders size={14} className="text-[#1D3045]" />
            <span className="text-[11px] uppercase tracking-[0.25em] font-semibold text-[#1D3045]">
              CARGO MOVE PARAMETERS
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            <div className="border-r border-[#1D3045]/10 pr-3 last:border-r-0">
              <label className="block text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/60 font-semibold mb-1">
                Cargo
              </label>
              <input
                type="text"
                value={cargo}
                onChange={(e) => setCargo(e.target.value)}
                className="w-full text-sm font-semibold text-[#1D3045] bg-transparent focus:outline-none focus:border-b-2 focus:border-[#1D3045] pb-0.5"
              />
            </div>

            <div className="border-r border-[#1D3045]/10 pr-3 last:border-r-0">
              <label className="block text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/60 font-semibold mb-1">
                Quantity
              </label>
              <input
                type="text"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className="w-full text-sm font-semibold text-[#1D3045] bg-transparent focus:outline-none focus:border-b-2 focus:border-[#1D3045] pb-0.5"
              />
            </div>

            <div className="border-r border-[#1D3045]/10 pr-3 last:border-r-0">
              <label className="block text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/60 font-semibold mb-1">
                Origin
              </label>
              <input
                type="text"
                value={origin}
                onChange={(e) => setOrigin(e.target.value)}
                className="w-full text-sm font-semibold text-[#1D3045] bg-transparent focus:outline-none focus:border-b-2 focus:border-[#1D3045] pb-0.5"
              />
            </div>

            <div className="border-r border-[#1D3045]/10 pr-3 last:border-r-0">
              <label className="block text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/60 font-semibold mb-1">
                Destination
              </label>
              <input
                type="text"
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                className="w-full text-sm font-semibold text-[#1D3045] bg-transparent focus:outline-none focus:border-b-2 focus:border-[#1D3045] pb-0.5"
              />
            </div>

            <div>
              <label className="block text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/60 font-semibold mb-1">
                Shipment Window
              </label>
              <input
                type="text"
                value={windowPeriod}
                onChange={(e) => setWindowPeriod(e.target.value)}
                className="w-full text-sm font-semibold text-[#1D3045] bg-transparent focus:outline-none focus:border-b-2 focus:border-[#1D3045] pb-0.5"
              />
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-[#1D3045]/10 flex justify-end">
            <button
              onClick={handleAnalyze}
              className="px-7 py-3 rounded-full bg-[#1D3045] hover:opacity-90 text-white text-xs tracking-[0.2em] uppercase font-semibold transition-all shadow-sm active:translate-y-0.5 flex items-center gap-2"
            >
              {isAnalyzing ? 'Analyzing Move...' : 'ANALYZE MOVE'}
              <ArrowRight size={14} />
            </button>
          </div>
        </div>

        {/* SUMMARY METRIC TILES */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 items-stretch">
          <div className="bg-white/95 backdrop-blur-md p-5 sm:p-6 rounded-2xl border border-[#1D3045]/15 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.08)] flex flex-col justify-between">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/65 font-semibold">
              Current Freight
            </div>
            <div className="text-2xl sm:text-3xl font-light text-[#1D3045] my-2 flex items-baseline gap-1">
              {forecastState.loading ? <LoadingPill label="Loading" /> : marketSummary.currentRate} <span className="text-xs text-[#1D3045]/60 font-normal">{marketSummary.currentRateUnit}</span>
            </div>
            <div className="text-[11px] text-[#1D3045]/65">{marketSummary.route}</div>
          </div>

          <div className="bg-white/95 backdrop-blur-md p-5 sm:p-6 rounded-2xl border border-[#1D3045]/15 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.08)] flex flex-col justify-between">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/65 font-semibold">
              Forecast ({timeframe})
            </div>
            <div className={`text-2xl sm:text-3xl font-light my-2 flex items-center gap-1 font-semibold ${flexibleNoDirection ? 'text-[#1D3045]' : 'text-[#E05A47]'}`}>
              {forecastState.loading ? <LoadingPill label="Loading" /> : marketSummary.forecastChange}
            </div>
            <div className="text-[11px] text-[#1D3045]/65">{marketSummary.forecastSubtext}</div>
          </div>

          <div className="bg-white/95 backdrop-blur-md p-5 sm:p-6 rounded-2xl border border-[#1D3045]/15 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.08)] flex flex-col justify-between">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/65 font-semibold">
              Expected Cost
            </div>
            <div className="text-2xl sm:text-3xl font-light text-[#1D3045] my-2">
              {forecastState.loading ? <LoadingPill label="Loading" /> : marketSummary.expectedCost}
            </div>
            <div className="text-[11px] text-[#1D3045]/65">{marketSummary.expectedCostBasis}</div>
          </div>

          <div className="bg-white/95 backdrop-blur-md p-5 sm:p-6 rounded-2xl border border-[#1D3045]/15 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.08)] flex flex-col justify-between">
            <div className="text-[10px] uppercase tracking-[0.2em] text-[#1D3045]/65 font-semibold">
              Risk Profile
            </div>
            <div className="text-2xl sm:text-3xl font-light text-[#1D3045] my-2 flex items-center gap-2 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E05A47]" />
              {riskState.loading ? <LoadingPill label="Loading" /> : riskBreakdown.overallLevel}
            </div>
            <div className="text-[11px] text-[#1D3045]/65">{riskBreakdown.caption}</div>
          </div>
        </div>

        {/* ZONE 2 — HERO DECISION ZONE: 68/32 SPLIT (8 cols Chart / 4 cols Recommendation Card) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-7 items-stretch">
          
          {/* LEFT: FREIGHT FORECAST & UNCERTAINTY (68% width - lg:col-span-8 grid grid-rows-[1fr]) */}
          <div className="lg:col-span-8 grid grid-rows-[1fr]">
            <div className="bg-white/95 backdrop-blur-md p-6 sm:p-7 rounded-2xl border border-[#1D3045]/15 shadow-[0_12px_36px_-4px_rgba(29,48,69,0.12)] flex flex-col justify-between gap-4">
              <div className="flex flex-wrap items-center justify-between border-b border-[#1D3045]/10 pb-3 gap-3">
                <div>
                  <h2 className="text-xs tracking-[0.2em] uppercase font-semibold text-[#1D3045]">
                    FREIGHT FORECAST & UNCERTAINTY CONE
                  </h2>
                  <p className="text-xs text-[#1D3045]/65 mt-0.5">
                    High-precision scenario trajectory: P10 (Upper Surge), P50 (Median), P90 (Lower) bounds
                  </p>
                </div>

                <div className="inline-flex rounded-full p-1 bg-[#E8EEF5] border border-[#1D3045]/15">
                  {dashboardTimeframes.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setTimeframe(tf)}
                      className={`px-3.5 py-1 rounded-full text-xs font-semibold tracking-wider uppercase transition-all ${
                        timeframe === tf
                          ? 'bg-[#1D3045] text-white shadow-xs'
                          : 'text-[#1D3045]/70 hover:text-[#1D3045]'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              {forecastState.error && (
                <InlineNotice tone="coral">
                  Forecast API unavailable. Showing the last local visual baseline until the backend responds.
                </InlineNotice>
              )}

              {!forecastState.error && forecastFallback && (
                <InlineNotice>
                  FLEXIBLE — Insufficient directional confidence at this horizon. The backend returned fallback coverage for {vesselClass.toUpperCase()} {timeframe}.
                </InlineNotice>
              )}

              {/* HIGH-RESOLUTION FORECAST CHART WITH CLEAR PAST/FUTURE BOUNDARY */}
              <div className="relative w-full h-80 sm:h-[360px] select-none bg-[#F7F9FC] rounded-xl p-4 sm:p-5 border border-[#1D3045]/10 flex flex-col justify-between">
              <svg className="w-full h-full overflow-visible" viewBox="0 0 720 280" fill="none">
                <defs>
                  {/* Soft Navy Gradient under P50 Curve */}
                  <linearGradient id="p50NavyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1D3045" stopOpacity="0.16" />
                    <stop offset="100%" stopColor="#1D3045" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Light Background Shading for Future Forecast Zone (Right of NOW line X=360) */}
                <rect x="360" y="22" width="240" height="223" fill="#1D3045" fillOpacity="0.02" rx="4" />

                {/* PAST / FUTURE ZONE ANNOTATION TEXT AT TOP OF CANVAS */}
                <text x="200" y="34" fontSize="9" fontWeight="bold" fill="#1D3045" fillOpacity="0.45" textAnchor="middle" letterSpacing="0.12em" fontFamily="monospace">
                  ← HISTORICAL ACTUALS
                </text>
                <text x="480" y="34" fontSize="9" fontWeight="bold" fill="#E05A47" fillOpacity="0.75" textAnchor="middle" letterSpacing="0.12em" fontFamily="monospace">
                  FORECAST SCENARIOS →
                </text>

                {/* Horizontal Gridlines */}
                <line x1="45" y1="45" x2="600" y2="45" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
                <line x1="45" y1="95" x2="600" y2="95" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
                <line x1="45" y1="145" x2="600" y2="145" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
                <line x1="45" y1="195" x2="600" y2="195" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
                <line x1="45" y1="245" x2="600" y2="245" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />

                {/* Vertical Time Guidelines */}
                <line x1="120" y1="22" x2="120" y2="245" stroke="#1D3045" strokeOpacity="0.05" strokeDasharray="3 3" />
                <line x1="200" y1="22" x2="200" y2="245" stroke="#1D3045" strokeOpacity="0.05" strokeDasharray="3 3" />
                <line x1="280" y1="22" x2="280" y2="245" stroke="#1D3045" strokeOpacity="0.05" strokeDasharray="3 3" />
                <line x1="440" y1="22" x2="440" y2="245" stroke="#1D3045" strokeOpacity="0.05" strokeDasharray="3 3" />
                <line x1="520" y1="22" x2="520" y2="245" stroke="#1D3045" strokeOpacity="0.05" strokeDasharray="3 3" />

                {/* Y-axis Ticks */}
                <text x="36" y="49" fontSize="10" fontWeight="bold" fill="#1D3045" fillOpacity="0.6" textAnchor="end" fontFamily="monospace">$19.50</text>
                <text x="36" y="99" fontSize="10" fontWeight="bold" fill="#1D3045" fillOpacity="0.6" textAnchor="end" fontFamily="monospace">$18.20</text>
                <text x="36" y="149" fontSize="10" fontWeight="bold" fill="#1D3045" fillOpacity="0.6" textAnchor="end" fontFamily="monospace">$16.90</text>
                <text x="36" y="199" fontSize="10" fontWeight="bold" fill="#1D3045" fillOpacity="0.6" textAnchor="end" fontFamily="monospace">$15.60</text>
                <text x="36" y="249" fontSize="10" fontWeight="bold" fill="#1D3045" fillOpacity="0.6" textAnchor="end" fontFamily="monospace">$14.30</text>

                {/* Uncertainty Band Shading */}
                <path d={currentForecast.bandPath} fill="#1D3045" fillOpacity="0.08" />

                {/* P50 Gradient Fill Under Median Curve */}
                <path d={`${currentForecast.p50Path} L 600 245 L 360 245 Z`} fill="url(#p50NavyGrad)" />

                {/* VISUALLY DISTINCT SOLID HISTORICAL LINE (Thicker Stroke, Confirmed Past) */}
                <path d={currentForecast.historicalPath} stroke="#1D3045" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />

                {/* P10 Upper Scenario Line */}
                <path d={currentForecast.p10Path} stroke="#E05A47" strokeWidth="2.2" strokeDasharray="4 3" strokeOpacity="0.9" />

                {/* P50 Median Forecast Line */}
                <path d={currentForecast.p50Path} stroke="#1D3045" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />

                {/* P90 Lower Scenario Line */}
                <path d={currentForecast.p90Path} stroke="#486581" strokeWidth="2" strokeDasharray="6 3" strokeOpacity="0.8" />

                {/* CLEAR UNAMBIGUOUS "NOW" REFERENCE LINE & TOP BADGE */}
                <g transform="translate(360, 0)">
                  {/* Vertical Dashed Reference Line from Top (Y=20) to Bottom (Y=245) */}
                  <line x1="0" y1="20" x2="0" y2="245" stroke="#1D3045" strokeOpacity="0.4" strokeWidth="1.5" strokeDasharray="4 3" />
                  
                  {/* Clean NOW Badge Positioned Above Chart Plot Area at Y=14 */}
                  <g transform="translate(0, 14)">
                    <rect x="-45" y="-10" width="90" height="20" fill="#1D3045" rx="5" stroke="#00E5FF" strokeWidth="1" />
                    <text x="0" y="3" fontSize="9" fontWeight="bold" fill="#FFFFFF" textAnchor="middle" letterSpacing="0.05em">
                      NOW ({currentForecast.currentRate})
                    </text>
                  </g>
                </g>

                {/* DIVERGENCE POINT DOT AT (360, 162) - CLEAN ANCHOR WITHOUT CLUTTER */}
                <g transform="translate(360, 162)">
                  <circle cx="0" cy="0" r="6" fill="#1D3045" fillOpacity="0.2" className="animate-ping" />
                  <circle cx="0" cy="0" r="4" fill="#1D3045" stroke="#FFFFFF" strokeWidth="2" />
                </g>

                {/* ENDPOINT BADGES */}
                <g transform={`translate(600, ${currentForecast.p10Y})`}>
                  <circle cx="0" cy="0" r="3.5" fill="#E05A47" />
                  <line x1="0" y1="0" x2="6" y2="0" stroke="#E05A47" strokeWidth="1.5" />
                  <g transform="translate(6, -9)">
                    <rect x="0" y="0" width="108" height="18" fill="#E05A47" rx="4" />
                    <text x="6" y="12" fontSize="9" fontWeight="bold" fill="#FFFFFF" fontFamily="monospace">
                      P10: {currentForecast.p10Value}
                    </text>
                  </g>
                </g>

                <g transform={`translate(600, ${currentForecast.p50Y})`}>
                  <circle cx="0" cy="0" r="4" fill="#1D3045" stroke="#00E5FF" strokeWidth="1.5" />
                  <line x1="0" y1="0" x2="6" y2="0" stroke="#1D3045" strokeWidth="2" />
                  <g transform="translate(6, -9)">
                    <rect x="0" y="0" width="108" height="18" fill="#1D3045" rx="4" />
                    <text x="6" y="12" fontSize="9" fontWeight="bold" fill="#00E5FF" fontFamily="monospace">
                      P50: {currentForecast.p50Value}
                    </text>
                  </g>
                </g>

                <g transform={`translate(600, ${currentForecast.p90Y})`}>
                  <circle cx="0" cy="0" r="3.5" fill="#486581" />
                  <line x1="0" y1="0" x2="6" y2="0" stroke="#486581" strokeWidth="1.5" />
                  <g transform="translate(6, -9)">
                    <rect x="0" y="0" width="108" height="18" fill="#486581" rx="4" />
                    <text x="6" y="12" fontSize="9" fontWeight="bold" fill="#FFFFFF" fontFamily="monospace">
                      P90: {currentForecast.p90Value}
                    </text>
                  </g>
                </g>
              </svg>

              {/* Legend */}
              <div className="flex flex-wrap items-center justify-between pt-2.5 border-t border-[#1D3045]/10 text-[10px] text-[#1D3045]/80 font-mono">
                <div className="flex items-center gap-2">
                  <span className="w-4 h-0.5 bg-[#1D3045]" />
                  <span><strong>P50</strong> Median Forecast (Navy)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-4 h-0.5 bg-[#E05A47] border-t border-dashed" />
                  <span><strong>P10</strong> High Scenario (Coral)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-4 h-0.5 bg-[#486581] border-t border-dashed" />
                  <span><strong>P90</strong> Low Scenario (Slate)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-[#1D3045]/10 rounded-sm" />
                  <span>P10–P90 Cone</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: RECOMMENDATION ENGINE (PROPORTIONATE HEADING SIZE & BALANCED VISUAL WEIGHT) */}
          <div className="lg:col-span-4 grid grid-rows-[1fr]">
            <div className="bg-[#1D3045] text-white p-5 sm:p-6 rounded-2xl border border-[#1D3045]/20 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.12)] flex flex-col justify-between gap-4 relative overflow-hidden">
              
              <div className="absolute top-0 right-0 w-48 h-48 bg-[#E05A47]/10 rounded-full blur-2xl pointer-events-none" />

            <div className="space-y-3 relative z-10">
              <div className="flex items-center justify-between border-b border-white/15 pb-2.5">
                <span className="text-[10px] uppercase tracking-[0.25em] text-white/70 font-semibold flex items-center gap-2">
                  <CheckCircle2 size={13} className="text-[#E05A47]" />
                  {recommendation.badge}
                </span>
                <span className="text-[9px] font-mono text-cyan-300 uppercase tracking-widest bg-white/10 px-2 py-0.5 rounded">
                  {recommendation.confidence}
                </span>
              </div>

              {/* PROPORTIONATE CONFIDENT HEADING (text-2xl sm:text-3xl) */}
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] text-white/60 mb-0.5 font-semibold">
                  RECOMMENDED ACTION
                </p>
                <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#E05A47] uppercase leading-tight drop-shadow-sm">
                  {forecastState.loading ? 'LOADING' : recommendation.decision}
                </h2>
              </div>

              <p className="text-xs sm:text-sm text-white/90 leading-relaxed font-normal">
                {forecastState.loading ? 'Fetching policy decision from the FICOS backend.' : recommendation.rationale}
              </p>

              <div className="space-y-2 pt-2.5 text-xs border-t border-white/10 font-mono">
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Vessel Class:</span>
                  <strong className="text-white font-semibold">{recommendation.vesselClass}</strong>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Target Port:</span>
                  <strong className="text-white font-semibold">{recommendation.targetPort}</strong>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Optimal Window:</span>
                  <strong className="text-[#E05A47] font-semibold">{recommendation.optimalWindow}</strong>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Est. Savings:</span>
                  <strong className="text-emerald-400 font-semibold">{recommendation.potentialSavings}</strong>
                </div>
              </div>
            </div>

            <div className="relative z-10 pt-1">
              <button
                onClick={() => goTo('vessel-intelligence')}
                className="w-full py-3 rounded-full bg-[#E05A47] hover:bg-[#c84e3c] text-white text-xs tracking-[0.2em] uppercase font-semibold transition-all shadow-md active:translate-y-0.5 text-center flex items-center justify-center gap-2"
              >
                <span>EXECUTE CHARTER IN VESSEL INTELLIGENCE</span>
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        </div>
      </div>

        {/* ========================================================================= */}
        {/* RESTRUCTURED FULL-WIDTH DEEP DIVE SECTIONS BELOW HERO DECISION ZONE       */}
        {/* ========================================================================= */}

        {/* SECTION 1 — HORIZON COMPARISON (FULL-WIDTH ENLARGED BAR CHART) */}
        <section className="bg-white/95 backdrop-blur-md p-6 sm:p-9 rounded-2xl border border-[#1D3045]/15 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] space-y-7">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#1D3045]/10 pb-5 gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold tracking-[0.2em] uppercase text-[#E05A47] bg-[#E05A47]/10 px-2.5 py-0.5 rounded">
                  SECTION 01
                </span>
                <span className="text-xs uppercase tracking-[0.25em] font-semibold text-[#1D3045]/60">
                  // DEEP DIVE ANALYSIS
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-light uppercase tracking-tight text-[#1D3045] flex items-center gap-2">
                <BarChart3 size={24} className="text-[#1D3045]" />
                HORIZON COMPARISON & SCENARIO TRAJECTORY
              </h2>
              <p className="text-sm sm:text-base text-[#1D3045]/80 mt-1 font-normal">
                Detailed rate scenario distribution across 5 evaluation horizons comparing P10 (Upper Surge), P50 (Median), and P90 (Lower) bounds.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-[#1D3045]">
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-3.5 rounded-xs bg-[#E05A47]" />
                <span><strong>P10</strong> Upper Surge</span>
              </div>
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-3.5 rounded-xs bg-[#1D3045]" />
                <span><strong>P50</strong> Median Forecast</span>
              </div>
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-3.5 rounded-xs bg-[#486581]" />
                <span><strong>P90</strong> Lower Bound</span>
              </div>
            </div>
          </div>

          {/* LARGE LEGIBLE BAR CHART CANVAS (360px Height, 15 Bars Across 5 Horizon Groups) */}
          <div className="relative w-full h-[340px] sm:h-[400px] bg-[#F7F9FC] rounded-xl p-6 border border-[#1D3045]/10 select-none flex flex-col justify-between">
            <svg className="w-full h-full overflow-visible" viewBox="0 0 900 280">
              {/* Dynamic gridlines keep bars valid for any corridor rate range. */}
              {horizonChartDomain.ticks.map((tick, tickIndex) => {
                const y = 35 + tickIndex * (205 / 4);
                return (
                  <g key={tick.toFixed(2)}>
                    <line
                      x1="55"
                      y1={y}
                      x2="865"
                      y2={y}
                      stroke="#1D3045"
                      strokeOpacity={tickIndex === 4 ? 0.2 : 0.08}
                      strokeDasharray={tickIndex === 4 ? undefined : '3 3'}
                    />
                    <text x="46" y={y + 4} fontSize="12" fontWeight="bold" fill="#1D3045" fillOpacity="0.65" textAnchor="end" fontFamily="monospace">
                      ${tick.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* 5 Horizon Groups */}
              {horizonComparison.map((item, idx) => {
                const groupX = 80 + idx * 135;
                const isHovered = hoveredHorizon === item.horizon;

                const chartTop = 35;
                const chartBase = 240;
                const chartHeight = chartBase - chartTop;
                const getY = (val: number) => {
                  const normalized = (val - horizonChartDomain.min) / horizonChartDomain.span;
                  return Math.max(chartTop, Math.min(chartBase, chartBase - normalized * chartHeight));
                };
                const getH = (val: number) => Math.max(2, chartBase - getY(val));

                const p90Y = getY(item.p90);
                const p90H = getH(item.p90);

                const p50Y = getY(item.p50);
                const p50H = getH(item.p50);

                const p10Y = getY(item.p10);
                const p10H = getH(item.p10);

                return (
                  <g
                    key={item.horizon}
                    onMouseEnter={() => setHoveredHorizon(item.horizon)}
                    onMouseLeave={() => setHoveredHorizon(null)}
                    className="cursor-pointer transition-all duration-200"
                  >
                    {/* Group Highlight Background on Hover */}
                    <rect
                      x={groupX - 18}
                      y="18"
                      width="112"
                      height="252"
                      fill={isHovered ? '#1D3045' : 'transparent'}
                      fillOpacity={isHovered ? 0.05 : 0}
                      rx="8"
                    />

                    {/* Bar 1: P90 Lower Bound (Slate #486581) */}
                    <rect
                      x={groupX}
                      y={p90Y}
                      width="24"
                      height={p90H}
                      fill="#486581"
                      rx="3.5"
                      className="transition-all duration-300 hover:opacity-100 opacity-90"
                    />
                    <text
                      x={groupX + 12}
                      y={p90Y - 7}
                      fontSize="10"
                      fontWeight="bold"
                      fill="#486581"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      ${item.p90.toFixed(2)}
                    </text>

                    {/* Bar 2: P50 Median Forecast (Navy #1D3045) */}
                    <rect
                      x={groupX + 35}
                      y={p50Y}
                      width="24"
                      height={p50H}
                      fill="#1D3045"
                      rx="3.5"
                      className="transition-all duration-300 hover:opacity-100 opacity-95"
                    />
                    <text
                      x={groupX + 47}
                      y={p50Y - 7}
                      fontSize="10"
                      fontWeight="bold"
                      fill="#1D3045"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      ${item.p50.toFixed(2)}
                    </text>

                    {/* Bar 3: P10 Upper Scenario (Coral #E05A47) */}
                    <rect
                      x={groupX + 70}
                      y={p10Y}
                      width="24"
                      height={p10H}
                      fill="#E05A47"
                      rx="3.5"
                      className="transition-all duration-300 hover:opacity-100 opacity-90"
                    />
                    <text
                      x={groupX + 82}
                      y={p10Y - 7}
                      fontSize="10"
                      fontWeight="bold"
                      fill="#E05A47"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      ${item.p10.toFixed(2)}
                    </text>

                    {/* Group Label */}
                    <text
                      x={groupX + 47}
                      y="260"
                      fontSize="12"
                      fontWeight="bold"
                      fill="#1D3045"
                      textAnchor="middle"
                      fontFamily="sans-serif"
                      letterSpacing="0.05em"
                    >
                      {item.horizon}
                    </text>
                    <text
                      x={groupX + 47}
                      y="273"
                      fontSize="10"
                      fontWeight="600"
                      fill="#1D3045"
                      fillOpacity="0.55"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      ({item.date})
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* SPREAD NARRATIVE TILES BELOW CHART */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 pt-2">
            {horizonComparison.slice(0, 3).map((item) => (
              <div
                key={item.horizon}
                className="bg-[#F7F9FC] p-5 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between shadow-xs"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-[#1D3045] uppercase tracking-wider">
                    {item.horizon} Horizon ({item.date})
                  </span>
                  <span className="text-xs font-mono font-bold text-[#E05A47] bg-[#E05A47]/10 px-2.5 py-0.5 rounded">
                    Spread: {item.spread}
                  </span>
                </div>
                <p className="text-sm text-[#1D3045]/80 leading-relaxed">
                  {item.note}
                </p>
                <div className="mt-4 pt-3 border-t border-[#1D3045]/10 grid grid-cols-3 gap-2 text-[11px] font-mono">
                  <span className="text-[#486581] font-semibold truncate">P90 ${item.p90.toFixed(2)}</span>
                  <span className="text-[#1D3045] font-extrabold text-center truncate">P50 ${item.p50.toFixed(2)}</span>
                  <span className="text-[#E05A47] font-bold text-right truncate">P10 ${item.p10.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 2 — RISK BREAKDOWN (BALANCED 50/50 MULTI-COLUMN WITH ENLARGED DONUT) */}
        <section className="bg-white/95 backdrop-blur-md p-6 sm:p-9 rounded-2xl border border-[#1D3045]/15 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] space-y-7">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#1D3045]/10 pb-5 gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold tracking-[0.2em] uppercase text-[#E05A47] bg-[#E05A47]/10 px-2.5 py-0.5 rounded">
                  SECTION 02
                </span>
                <span className="text-xs uppercase tracking-[0.25em] font-semibold text-[#1D3045]/60">
                  // CATEGORICAL PROFILE
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-light uppercase tracking-tight text-[#1D3045] flex items-center gap-2">
                <ShieldAlert size={24} className="text-[#E05A47]" />
                RISK BREAKDOWN & DRIVER ANALYSIS
              </h2>
              <p className="text-sm sm:text-base text-[#1D3045]/80 mt-1 font-normal">
                Quantified risk distribution across weather, port infrastructure, geopolitics, and market supply.
              </p>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#E05A47]/10 border border-[#E05A47]/20">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E05A47] animate-pulse" />
              <span className="text-xs sm:text-sm font-mono font-bold uppercase tracking-wider text-[#E05A47]">
                {riskState.loading ? 'LOADING RISK PROFILE' : `OVERALL RATING: ${riskBreakdown.overallLevel} (SCORE: ${riskBreakdown.score}/100)`}
              </span>
            </div>
          </div>

          {riskState.error && (
            <InlineNotice tone="coral">
              Risk API unavailable. Keeping a readable baseline risk profile on screen instead of leaving the panel blank.
            </InlineNotice>
          )}

          {/* EQUAL 50/50 MULTI-COLUMN LAYOUT FOR VISUAL BALANCE */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            
            {/* LEFT COLUMN: ENLARGED SVG DONUT CHART (lg:col-span-6 grid grid-rows-[1fr]) */}
            <div className="lg:col-span-6 grid grid-rows-[1fr]">
              <div className="bg-[#F7F9FC] p-6 sm:p-8 rounded-2xl border border-[#1D3045]/10 flex flex-col items-center justify-between shadow-xs relative">
                
                <div className="w-full flex justify-between items-center border-b border-[#1D3045]/10 pb-3 mb-4">
                  <span className="text-xs font-mono font-bold uppercase tracking-widest text-[#1D3045]/70 flex items-center gap-2">
                    <ShieldAlert size={14} className="text-[#E05A47]" />
                    RISK WEIGHT PROPORTIONS
                  </span>
                  <span className="text-xs font-mono font-bold text-[#E05A47] bg-[#E05A47]/10 px-2 py-0.5 rounded">
                    MEDIUM LEVEL
                  </span>
                </div>

                {/* Scaled-Up 280x280 SVG Donut */}
                <div className="relative w-72 h-72 sm:w-80 sm:h-80 flex items-center justify-center my-auto py-2">
                  <svg className="w-full h-full transform -rotate-90" viewBox="0 0 280 280">
                    {donutSegments.map((seg) => {
                      const isHovered = hoveredRiskCategory === seg.name;
                      return (
                        <circle
                          key={seg.name}
                          cx="140"
                          cy="140"
                          r={donutR}
                          fill="transparent"
                          stroke={seg.color}
                          strokeWidth={isHovered ? 36 : 30}
                          strokeDasharray={`${seg.dashLength} ${donutC - seg.dashLength}`}
                          strokeDashoffset={seg.dashOffset}
                          className="transition-all duration-300 cursor-pointer"
                          onMouseEnter={() => setHoveredRiskCategory(seg.name)}
                          onMouseLeave={() => setHoveredRiskCategory(null)}
                        />
                      );
                    })}
                  </svg>

                  {/* Scaled Big Center Readout */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center p-4">
                    <span className="text-xs uppercase tracking-widest text-[#1D3045]/65 font-bold max-w-[160px] truncate">
                      {hoveredRiskCategory || 'OVERALL RISK'}
                    </span>
                    <span className="text-4xl sm:text-5xl font-black text-[#1D3045] leading-none my-1">
                      {hoveredRiskCategory
                        ? `${riskBreakdown.categories.find((c) => c.name === hoveredRiskCategory)?.percent}%`
                        : riskBreakdown.overallLevel}
                    </span>
                    <span className="text-xs font-mono font-bold text-[#1D3045]/60 uppercase tracking-wider">
                      {hoveredRiskCategory ? 'Category Weight' : `Score: ${riskBreakdown.score}/100`}
                    </span>
                  </div>
                </div>

                {/* Bottom Quick Category Pills */}
                <div className="w-full grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-[#1D3045]/10 text-xs font-mono mt-auto">
                  {riskBreakdown.categories.map((cat) => (
                    <div
                      key={cat.name}
                      onMouseEnter={() => setHoveredRiskCategory(cat.name)}
                      onMouseLeave={() => setHoveredRiskCategory(null)}
                      className={`flex items-center justify-between p-2 rounded-lg transition-colors cursor-pointer ${
                        hoveredRiskCategory === cat.name ? 'bg-[#1D3045] text-white' : 'bg-white border border-[#1D3045]/10'
                      }`}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: cat.color }} />
                        <span className="truncate">{cat.name.split(' ')[0]}</span>
                      </div>
                      <span className="font-bold">{cat.percent}%</span>
                    </div>
                  ))}
                </div>

              </div>
            </div>

            {/* RIGHT COLUMN: EXPANDED RISK CATEGORIES & DRIVER EXPLANATION (lg:col-span-6 grid grid-rows-[auto_1fr] gap-5) */}
            <div className="lg:col-span-6 grid grid-rows-[auto_1fr] gap-5">
              
              {/* Explanatory Callout Box */}
              <div className="bg-[#1D3045] text-white p-6 rounded-2xl border-l-4 border-[#E05A47] shadow-sm space-y-2">
                <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-[#E05A47]">
                  PRIMARY DRIVER SUMMARY
                </h3>
                <p className="text-sm sm:text-base text-white/95 leading-relaxed font-normal">
                  {riskState.loading ? 'Fetching route and destination risk drivers from the FICOS backend.' : riskBreakdown.driverSummary}
                </p>
              </div>

              {/* Category Breakdown List */}
              <div className="space-y-3.5 flex-1">
                {riskBreakdown.categories.map((cat) => {
                  const isHovered = hoveredRiskCategory === cat.name;
                  return (
                    <div
                      key={cat.name}
                      onMouseEnter={() => setHoveredRiskCategory(cat.name)}
                      onMouseLeave={() => setHoveredRiskCategory(null)}
                      className={`p-5 rounded-2xl border transition-all cursor-pointer ${
                        isHovered
                          ? 'bg-white border-[#1D3045] shadow-md'
                          : 'bg-[#F7F9FC] border-[#1D3045]/12 hover:border-[#1D3045]/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2.5">
                          <span className="w-3.5 h-3.5 rounded-full flex-shrink-0" style={{ backgroundColor: cat.color }} />
                          <span className="text-base font-bold text-[#1D3045]">{cat.name}</span>
                          <span className="text-xs font-mono font-bold text-[#1D3045]/60 bg-[#1D3045]/8 px-2 py-0.5 rounded">
                            {cat.impact}
                          </span>
                        </div>
                        <span className="text-lg font-mono font-bold text-[#1D3045]">{cat.percent}%</span>
                      </div>

                      {/* Progress Bar */}
                      <div className="w-full h-2.5 bg-[#1D3045]/10 rounded-full overflow-hidden mb-2.5">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${cat.percent}%`, backgroundColor: cat.color }}
                        />
                      </div>

                      <p className="text-sm text-[#1D3045]/80 leading-relaxed font-normal">
                        {cat.description}
                      </p>
                    </div>
                  );
                })}
              </div>

            </div>

          </div>
        </section>

        {/* SECTION 3 — CONFIDENCE GAUGE (BALANCED 50/50 MULTI-COLUMN WITH SCALED BIG NUMBERS & PIXEL-PERFECT EQUAL HEIGHTS) */}
        <section className="bg-white/95 backdrop-blur-md p-6 sm:p-9 rounded-2xl border border-[#1D3045]/15 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] space-y-7">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#1D3045]/10 pb-5 gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold tracking-[0.2em] uppercase text-[#E05A47] bg-[#E05A47]/10 px-2.5 py-0.5 rounded">
                  SECTION 03
                </span>
                <span className="text-xs uppercase tracking-[0.25em] font-semibold text-[#1D3045]/60">
                  // MODEL ACCURACY & CONVICTION
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-light uppercase tracking-tight text-[#1D3045] flex items-center gap-2">
                <Activity size={24} className="text-[#1D3045]" />
                MODEL CONFIDENCE & UNCERTAINTY GAUGE
              </h2>
              <p className="text-sm sm:text-base text-[#1D3045]/80 mt-1 font-normal">
                Statistical validation of forecast conviction based on historical backtests and real-time market data ingestion.
              </p>
            </div>

            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${
              unpromotedFallback
                ? 'bg-[#E05A47]/10 border-[#E05A47]/20'
                : 'bg-emerald-500/10 border-emerald-500/20'
            }`}>
              <span className={`w-2.5 h-2.5 rounded-full animate-pulse ${unpromotedFallback ? 'bg-[#E05A47]' : 'bg-emerald-600'}`} />
              <span className={`text-xs sm:text-sm font-mono font-bold uppercase tracking-wider ${unpromotedFallback ? 'text-[#E05A47]' : 'text-emerald-700'}`}>
                {forecastState.loading ? 'FETCHING MODEL STATUS' : confidenceMetrics.label}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            
            {/* LEFT / CENTER: SLEEK LINEAR CONVICTION METER (lg:col-span-6 grid grid-rows-[1fr]) */}
            <div className="lg:col-span-6 grid grid-rows-[1fr]">
              <div className="bg-[#F7F9FC] p-6 sm:p-8 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between shadow-xs select-none">
                
                <div className="w-full flex justify-between items-center border-b border-[#1D3045]/10 pb-4 mb-4">
                  <span className="text-xs font-mono font-bold uppercase tracking-widest text-[#1D3045]/70 flex items-center gap-2">
                    <Activity size={14} className="text-[#1D3045]" />
                    STATISTICAL CONVICTION METER
                  </span>
                  <span className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded border ${
                    unpromotedFallback
                      ? 'text-[#E05A47] bg-[#E05A47]/10 border-[#E05A47]/20'
                      : 'text-emerald-600 bg-emerald-50 border-emerald-500/20'
                  }`}>
                    {confidenceMetrics.statusBadge}
                  </span>
                </div>

                {/* Big Score Readout + Linear Progress Meter */}
                <div className="w-full space-y-6 my-auto py-2">
                  <div className="flex items-baseline justify-between">
                    <div>
                      <span className="text-xs font-mono uppercase tracking-widest text-[#1D3045]/60 font-semibold block mb-0.5">
                        Model Confidence Score
                      </span>
                      <div className="text-5xl sm:text-6xl font-black text-[#1D3045] font-sans tracking-tight">
                        {confidenceMetrics.scorePercent}%
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`text-xs font-mono uppercase tracking-wider font-bold px-3 py-1 rounded-lg border inline-block ${
                        unpromotedFallback
                          ? 'text-[#E05A47] bg-[#E05A47]/10 border-[#E05A47]/20'
                          : 'text-emerald-700 bg-emerald-500/10 border-emerald-500/20'
                      }`}>
                        {confidenceMetrics.convictionLabel}
                      </span>
                      <span className="text-xs text-[#1D3045]/60 block mt-1 font-mono">
                        {confidenceMetrics.intervalLabel}
                      </span>
                    </div>
                  </div>

                  {/* High-Contrast Multi-Segment Linear Progress Bar */}
                  <div className="space-y-2">
                    <div className="w-full h-5 bg-[#1D3045]/10 rounded-full overflow-hidden p-1 relative border border-[#1D3045]/15 shadow-inner">
                      <div
                        className="h-full bg-gradient-to-r from-[#1D3045] via-[#2C4866] to-[#E05A47] rounded-full transition-all duration-1000 shadow-sm"
                        style={{ width: `${confidenceMetrics.scorePercent}%` }}
                      />
                    </div>

                    {/* Scale Ticks (0%, 25%, 50%, 75%, 100%) */}
                    <div className="flex justify-between text-[11px] font-mono font-bold text-[#1D3045]/60 px-0.5">
                      <span>0%</span>
                      <span>25%</span>
                      <span>50%</span>
                      <span>75%</span>
                      <span className="text-[#1D3045] font-extrabold">100%</span>
                    </div>
                  </div>
                </div>

                {/* Bottom Footer Specs */}
                <div className="w-full grid grid-cols-2 gap-4 pt-4 border-t border-[#1D3045]/10 mt-auto text-xs font-mono">
                  <div>
                    <span className="text-[#1D3045]/60 block mb-0.5">Band Width Index:</span>
                    <strong className="text-[#1D3045] font-bold text-sm">{confidenceMetrics.bandWidthScore}/100</strong>
                  </div>
                  <div>
                    <span className="text-[#1D3045]/60 block mb-0.5">Variance Status:</span>
                    <strong className={`font-bold text-sm ${unpromotedFallback ? 'text-[#E05A47]' : 'text-emerald-600'}`}>
                      {confidenceMetrics.varianceStatus}
                    </strong>
                  </div>
                </div>

              </div>
            </div>

            {/* RIGHT: SCALED STAT CARDS & NARRATIVE (lg:col-span-6 grid grid-rows-[auto_1fr] gap-5) */}
            <div className="lg:col-span-6 grid grid-rows-[auto_1fr] gap-5">
              
              {/* Stat Cards with Scaled Big Numbers */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-[#F7F9FC] p-5 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between shadow-xs">
                  <span className="text-xs uppercase tracking-wider text-[#1D3045]/65 font-semibold block mb-1">
                    Uncertainty Spread
                  </span>
                  <span className="text-3xl sm:text-4xl font-black font-mono text-[#1D3045] my-1">
                    {confidenceMetrics.uncertaintySpread}
                  </span>
                  <span className="text-xs text-[#1D3045]/70 block">
                    P10–P90 band at 30D
                  </span>
                </div>

                <div className="bg-[#F7F9FC] p-5 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between shadow-xs">
                  <span className="text-xs uppercase tracking-wider text-[#1D3045]/65 font-semibold block mb-1">
                    Backtest Accuracy
                  </span>
                  <span className="text-3xl sm:text-4xl font-black font-mono text-emerald-600 my-1">
                    {confidenceMetrics.historicalAccuracy}
                  </span>
                  <span className="text-xs text-[#1D3045]/70 block">
                    1,200 simulated voyages
                  </span>
                </div>

                <div className="bg-[#F7F9FC] p-5 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between shadow-xs">
                  <span className="text-xs uppercase tracking-wider text-[#1D3045]/65 font-semibold block mb-1">
                    Telemetry Stream
                  </span>
                  <span className="text-3xl sm:text-4xl font-black font-mono text-[#1D3045] my-1">
                    {confidenceMetrics.signalsIngested} Signals
                  </span>
                  <span className="text-xs text-[#1D3045]/70 block">
                    AIS & FFA live feeds
                  </span>
                </div>
              </div>

              {/* Explanatory Box */}
              <div className="bg-[#F7F9FC] p-6 rounded-2xl border border-[#1D3045]/12 space-y-2.5 flex flex-col justify-center">
                <h3 className="text-sm font-mono font-bold uppercase tracking-widest text-[#1D3045] flex items-center gap-2">
                  <Info size={16} className="text-[#1D3045]" />
                  WHAT DOES THIS CONFIDENCE SCORE MEAN?
                </h3>
                <p className="text-sm sm:text-base text-[#1D3045]/85 leading-relaxed font-normal">
                  {forecastState.loading ? 'Fetching calibration and uncertainty details from /forecast.' : confidenceMetrics.detailedExplanation}
                </p>
              </div>

            </div>

          </div>
        </section>

        {/* SECTION 4 — SHOCK RESPONSE (ENLARGED EVENT ANALYSIS & STRATEGY) */}
        <section className="bg-white/95 backdrop-blur-md p-6 sm:p-9 rounded-2xl border border-[#1D3045]/15 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] space-y-7">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#1D3045]/10 pb-5 gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold tracking-[0.2em] uppercase text-[#E05A47] bg-[#E05A47]/10 px-2.5 py-0.5 rounded">
                  SECTION 04
                </span>
                <span className="text-xs uppercase tracking-[0.25em] font-semibold text-[#1D3045]/60">
                  // EVENT SENSITIVITY
                </span>
              </div>
              <h2 className="text-2xl sm:text-3xl font-light uppercase tracking-tight text-[#1D3045] flex items-center gap-2">
                <TrendingUp size={24} className="text-[#E05A47]" />
                SHOCK RESPONSE & HISTORICAL EVENT ANALYSIS
              </h2>
              <p className="text-sm sm:text-base text-[#1D3045]/80 mt-1 font-normal">
                Stress testing rate trajectory against major geopolitical disruptions and historical supply shocks.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-0.5 bg-[#E05A47]" />
                <span><strong>{shockResponse.primaryEvent.title}</strong> {shockResponse.primaryEvent.surgePeak !== '-' ? `(${shockResponse.primaryEvent.surgePeak})` : ''}</span>
              </div>
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-0.5 bg-[#486581] border-t border-dashed" />
                <span><strong>{shockResponse.comparisonEvents[0].title}</strong></span>
              </div>
              <div className="flex items-center gap-2 bg-[#F7F9FC] px-3.5 py-2 rounded-lg border border-[#1D3045]/10">
                <span className="w-3.5 h-0.5 bg-[#829AB1] border-t border-dashed" />
                <span><strong>{shockResponse.comparisonEvents[1].title}</strong></span>
              </div>
            </div>
          </div>

          {(shockState.error || shockState.status === 'unavailable') && (
            <InlineNotice tone="coral">
              {shockState.status === 'unavailable'
                ? shockState.data?.message || 'Shock response model unavailable for this event.'
                : 'Shock API unavailable. Showing the calibrated baseline event response until the backend responds.'}
            </InlineNotice>
          )}

          {/* ENLARGED MULTI-EVENT ANALYSIS CHART (360px Height) */}
          <div className="relative w-full h-[340px] sm:h-[380px] bg-[#F7F9FC] rounded-xl p-6 border border-[#1D3045]/10 select-none flex flex-col justify-between">
            <svg className="w-full h-full overflow-visible" viewBox="0 0 850 250" fill="none">
              <defs>
                <linearGradient id="primaryEventAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#E05A47" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#E05A47" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Gridlines */}
              <line x1="45" y1="35" x2="815" y2="35" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
              <line x1="45" y1="90" x2="815" y2="90" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
              <line x1="45" y1="145" x2="815" y2="145" stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="3 3" />
              <line x1="45" y1="200" x2="815" y2="200" stroke="#1D3045" strokeOpacity="0.2" />

              {/* Y-Axis Labels (Surge %) */}
              <text x="36" y="39" fontSize="11" fontWeight="bold" fill="#1D3045" fillOpacity="0.65" textAnchor="end" fontFamily="monospace">+25%</text>
              <text x="36" y="94" fontSize="11" fontWeight="bold" fill="#1D3045" fillOpacity="0.65" textAnchor="end" fontFamily="monospace">+18%</text>
              <text x="36" y="149" fontSize="11" fontWeight="bold" fill="#1D3045" fillOpacity="0.65" textAnchor="end" fontFamily="monospace">+9%</text>
              <text x="36" y="204" fontSize="11" fontWeight="bold" fill="#1D3045" fillOpacity="0.65" textAnchor="end" fontFamily="monospace">0%</text>

              {/* Historical Comparison Curve 1: Suez Canal Blockage 2021 */}
              <path
                d={shockResponse.comparisonEvents[0].path}
                stroke="#486581"
                strokeWidth="2.2"
                strokeDasharray="5 4"
                strokeOpacity="0.8"
              />

              {/* Historical Comparison Curve 2: Bunker Fuel Spike 2022 */}
              <path
                d={shockResponse.comparisonEvents[1].path}
                stroke="#829AB1"
                strokeWidth="2.2"
                strokeDasharray="3 3"
                strokeOpacity="0.8"
              />

              {/* Primary event gradient area from backend response */}
              <path
                d={primaryShockAreaPath}
                fill="url(#primaryEventAreaGrad)"
              />

              {/* Primary event curve from backend response */}
              <path
                d={primaryShockPath}
                stroke="#E05A47"
                strokeWidth="3.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Interactive Data Points along Primary Curve */}
              {shockPoints.map((pt, idx) => {
                const isHovered = hoveredSparklineIndex === idx;
                return (
                  <g key={pt.day} transform={`translate(${pt.x}, ${pt.y})`}>
                    <circle
                      cx="0"
                      cy="0"
                      r={isHovered ? 7 : 4.5}
                      fill={isHovered ? '#1D3045' : '#E05A47'}
                      stroke="#FFFFFF"
                      strokeWidth="2"
                      className="cursor-pointer transition-all duration-200"
                      onMouseEnter={() => setHoveredSparklineIndex(idx)}
                      onMouseLeave={() => setHoveredSparklineIndex(null)}
                    />
                    
                    {/* Floating Tooltip / Label */}
                    <g transform="translate(0, -14)">
                      <text
                        x="0"
                        y="0"
                        fontSize="11"
                        fontWeight="bold"
                        fill={isHovered ? '#1D3045' : '#E05A47'}
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        {pt.surge}
                      </text>
                    </g>
                  </g>
                );
              })}

              {/* X-Axis Ticks */}
              {shockPoints.map((pt) => (
                <g key={pt.day} transform={`translate(${pt.x}, 222)`}>
                  <text x="0" y="0" fontSize="12" fontWeight="bold" fill="#1D3045" textAnchor="middle" fontFamily="sans-serif">
                    {pt.day}
                  </text>
                  <text x="0" y="14" fontSize="10" fontWeight="600" fill="#1D3045" fillOpacity="0.55" textAnchor="middle" fontFamily="monospace">
                    {pt.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>

          {/* EVENT CONTEXT & MITIGATION PANELS */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            
            {/* Event Overview Card (lg:col-span-6 grid grid-rows-[1fr]) */}
            <div className="lg:col-span-6 grid grid-rows-[1fr]">
              <div className="bg-[#F7F9FC] p-6 sm:p-7 rounded-2xl border border-[#1D3045]/10 flex flex-col justify-between gap-4 shadow-xs">
                <div className="space-y-4">
                  <h3 className="text-sm font-mono font-bold uppercase tracking-widest text-[#1D3045]">
                    EVALUATED EVENT: {shockState.loading ? 'LOADING EVENT MODEL' : shockResponse.primaryEvent.title}
                  </h3>
                  <p className="text-sm sm:text-base text-[#1D3045]/85 leading-relaxed font-normal">
                    {shockState.loading ? 'Fetching historical shock response data from the FICOS backend.' : shockResponse.primaryEvent.description}
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-3 pt-3 text-xs sm:text-sm font-mono border-t border-[#1D3045]/10 mt-auto">
                  <div>
                    <span className="text-[#1D3045]/60 block mb-0.5">Date Range:</span>
                    <strong className="text-[#1D3045] font-bold">{shockResponse.primaryEvent.dateRange}</strong>
                  </div>
                  <div>
                    <span className="text-[#1D3045]/60 block mb-0.5">Peak Surge:</span>
                    <strong className="text-[#E05A47] font-bold">{shockResponse.primaryEvent.surgePeak}</strong>
                  </div>
                  <div>
                    <span className="text-[#1D3045]/60 block mb-0.5">Recovery Time:</span>
                    <strong className="text-[#1D3045] font-bold">{shockResponse.primaryEvent.recoveryTime}</strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Mitigation Strategy Card (lg:col-span-6 grid grid-rows-[1fr]) */}
            <div className="lg:col-span-6 grid grid-rows-[1fr]">
              <div className="bg-[#1D3045] text-white p-6 sm:p-7 rounded-2xl border border-[#1D3045]/20 flex flex-col justify-between gap-4 shadow-md">
                <div className="space-y-2">
                  <h3 className="text-sm font-mono font-bold uppercase tracking-widest text-[#E05A47] flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-[#E05A47]" />
                    CHARTERING RISK MITIGATION STRATEGY
                  </h3>
                  <p className="text-sm sm:text-base text-white/95 leading-relaxed font-normal pt-1">
                    {shockState.status === 'unavailable'
                      ? shockState.data?.message || 'Shock response model unavailable for the selected event. Choose a calibrated scenario such as red_sea, suez_canal, or bunker_spike.'
                      : shockResponse.mitigationAdvice}
                  </p>
                </div>

                <div className="pt-3 border-t border-white/15 flex justify-between items-center text-xs font-mono text-white/70 mt-auto">
                  <span>Hedge Activation: <strong className="text-white font-bold">Within 72 Hours</strong></span>
                  <span>Max Exposure Cap: <strong className="text-emerald-400 font-bold">14.8% Saved</strong></span>
                </div>
              </div>
            </div>

          </div>

        </section>

      </main>
    </div>
  );
}
