import React, { useState } from 'react';
import {
  Info,
  Sparkles,
  AlertTriangle,
  ArrowUpRight,
  MapPin,
  TrendingDown,
  CheckCircle2,
  Ship,
  DollarSign
} from 'lucide-react';
import { apiGet, ApiStatus } from './apiClient';

interface IdleIntelligenceProps {
  onNavigate: (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => void;
}

type EndpointState<T> = {
  data: T | null;
  status: ApiStatus;
  loading: boolean;
  error: string | null;
};

type IdleVessel = {
  id: string;
  code: string;
  name: string;
  vesselClass: string;
  dwt: string;
  currentPort: string;
  riskLevel: string;
  idleDaysForecast: number;
  dailyIdleCost: number;
  activeDays: number[];
  idleWindow: number[];
  repositionWindow: number[];
  recommendedRoute: string;
  repositionSavings: number;
};

type IdleRiskResponse = {
  fleet_count: number;
  total_projected_idle_cost_usd: number;
  total_potential_reposition_savings_usd: number;
  high_risk_vessel_count: number;
  vessels: Array<{
    id: string;
    code: string;
    name: string;
    vessel_class: string;
    dwt: string;
    current_port: string;
    risk_level: string;
    idle_days_forecast: number;
    daily_idle_cost_usd: number;
    active_days: number[];
    idle_window: number[];
    reposition_window: number[];
    recommended_route: string;
    reposition_savings_usd: number;
  }>;
};

const emptyEndpoint = <T,>(): EndpointState<T> => ({ data: null, status: 'idle', loading: false, error: null });
const formatUsd = (value: number) => `$${Math.round(value).toLocaleString()}`;

const IDLE_FLEET_VESSELS = [
  {
    id: 'v5', code: '05', name: 'MV BENGAL PHOENIX', vesselClass: 'Panamax', dwt: '72,000 DWT',
    currentPort: 'Haldia Anchorage', riskLevel: 'HIGH', idleDaysForecast: 10, dailyIdleCost: 14500,
    activeDays: [1, 14], idleWindow: [15, 25], repositionWindow: [26, 35],
    recommendedRoute: 'Paradip / Dhamra', repositionSavings: 147600,
  },
  {
    id: 'v6', code: '06', name: 'MV MARITIME PRIDE', vesselClass: 'Supramax', dwt: '55,000 DWT',
    currentPort: 'Gangavaram Outer', riskLevel: 'HIGH', idleDaysForecast: 8, dailyIdleCost: 12800,
    activeDays: [1, 10], idleWindow: [11, 19], repositionWindow: [20, 28],
    recommendedRoute: 'Port Hedland (Iron Ore)', repositionSavings: 112000,
  },
  {
    id: 'v3', code: '03', name: 'MV IRON TRADER', vesselClass: 'Capesize', dwt: '175,000 DWT',
    currentPort: 'Vizag Outer', riskLevel: 'MEDIUM', idleDaysForecast: 6, dailyIdleCost: 21500,
    activeDays: [1, 22], idleWindow: [23, 29], repositionWindow: [30, 40],
    recommendedRoute: 'Dampier (Bauxite Return)', repositionSavings: 94500,
  },
  {
    id: 'v2', code: '02', name: 'MV EASTERN WIND', vesselClass: 'Supramax', dwt: '58,000 DWT',
    currentPort: 'Paradip Roadstead', riskLevel: 'MEDIUM', idleDaysForecast: 5, dailyIdleCost: 13200,
    activeDays: [1, 18], idleWindow: [19, 24], repositionWindow: [25, 32],
    recommendedRoute: 'Dhamra Coal Berth', repositionSavings: 68000,
  },
  {
    id: 'v4', code: '04', name: 'MV PACIFIC VOYAGER', vesselClass: 'Handysize', dwt: '38,000 DWT',
    currentPort: 'Port Hedland', riskLevel: 'MEDIUM', idleDaysForecast: 4, dailyIdleCost: 10500,
    activeDays: [1, 28], idleWindow: [29, 33], repositionWindow: [34, 42],
    recommendedRoute: 'Gopalpur Silica', repositionSavings: 45000,
  },
  {
    id: 'v1', code: '01', name: 'MV OCEAN STAR', vesselClass: 'Panamax', dwt: '74,500 DWT',
    currentPort: 'En Route Dhamra', riskLevel: 'LOW', idleDaysForecast: 2, dailyIdleCost: 15100,
    activeDays: [1, 35], idleWindow: [36, 38], repositionWindow: [39, 48],
    recommendedRoute: 'Australia Coking Coal', repositionSavings: 32000,
  },
  {
    id: 'v7', code: '07', name: 'MV SOUTHERN CROSS', vesselClass: 'Panamax', dwt: '76,000 DWT',
    currentPort: 'Dampier Port', riskLevel: 'LOW', idleDaysForecast: 1, dailyIdleCost: 14800,
    activeDays: [1, 42], idleWindow: [43, 44], repositionWindow: [45, 52],
    recommendedRoute: 'Paradip Thermal', repositionSavings: 18000,
  },
  {
    id: 'v8', code: '08', name: 'MV CORAL SEA', vesselClass: 'Capesize', dwt: '180,000 DWT',
    currentPort: 'Gladstone Outer', riskLevel: 'LOW', idleDaysForecast: 2, dailyIdleCost: 22000,
    activeDays: [1, 40], idleWindow: [41, 43], repositionWindow: [44, 55],
    recommendedRoute: 'Vizag Iron Ore', repositionSavings: 28000,
  },
];

const CLUSTER_DATA = [
  { rank: '01', cluster: 'Haldia Anchorage', vessels: 'MV Bengal Phoenix / MV Maritime Pride', count: 2, dailyLoss: '$27,300', risk: 'CRITICAL', bar: 100, vesselId: 'v5', riskBg: 'bg-[#E05A47]', riskText: 'text-[#E05A47]' },
  { rank: '02', cluster: 'Vizag Outer Roads', vessels: 'MV Iron Trader', count: 1, dailyLoss: '$21,500', risk: 'HIGH', bar: 79, vesselId: 'v3', riskBg: 'bg-amber-500', riskText: 'text-amber-600' },
  { rank: '03', cluster: 'Paradip Roadstead', vessels: 'MV Eastern Wind', count: 1, dailyLoss: '$13,200', risk: 'MODERATE', bar: 48, vesselId: 'v2', riskBg: 'bg-amber-400', riskText: 'text-amber-500' },
  { rank: '04', cluster: 'Port Hedland Outer', vessels: 'MV Pacific Voyager', count: 1, dailyLoss: '$10,500', risk: 'MODERATE', bar: 38, vesselId: 'v4', riskBg: 'bg-emerald-500', riskText: 'text-emerald-600' },
];

export default function IdleIntelligence({ onNavigate }: IdleIntelligenceProps) {
  const [selectedVesselId, setSelectedVesselId] = useState<string>('v5');
  const [timelineFilter, setTimelineFilter] = useState<'ALL' | 'HIGH_RISK' | 'PANAMAX' | 'CAPESIZE'>('ALL');
  const [idleState, setIdleState] = useState<EndpointState<IdleRiskResponse>>(emptyEndpoint);

  React.useEffect(() => {
    const controller = new AbortController();
    setIdleState((prev) => ({ ...prev, loading: true, error: null }));

    apiGet<IdleRiskResponse>('/idle-risk', {}, controller.signal)
      .then((response) => setIdleState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setIdleState((prev) => ({ ...prev, loading: false, error: error.message || 'Idle risk unavailable' }));
        }
      });

    return () => controller.abort();
  }, []);

  const idleFleetVessels: IdleVessel[] = React.useMemo(() => {
    if (!idleState.data?.vessels?.length) return IDLE_FLEET_VESSELS;
    return idleState.data.vessels.map((v) => ({
      id: v.id,
      code: v.code,
      name: v.name,
      vesselClass: v.vessel_class,
      dwt: v.dwt,
      currentPort: v.current_port,
      riskLevel: v.risk_level,
      idleDaysForecast: v.idle_days_forecast,
      dailyIdleCost: v.daily_idle_cost_usd,
      activeDays: v.active_days,
      idleWindow: v.idle_window,
      repositionWindow: v.reposition_window,
      recommendedRoute: v.recommended_route,
      repositionSavings: v.reposition_savings_usd,
    }));
  }, [idleState.data]);

  React.useEffect(() => {
    if (!idleFleetVessels.some((v) => v.id === selectedVesselId)) {
      setSelectedVesselId(idleFleetVessels[0]?.id || 'v5');
    }
  }, [idleFleetVessels, selectedVesselId]);

  const selectedVessel = idleFleetVessels.find((v) => v.id === selectedVesselId) || idleFleetVessels[0];

  const filteredVessels = idleFleetVessels.filter((v) => {
    if (timelineFilter === 'HIGH_RISK') return v.riskLevel === 'HIGH';
    if (timelineFilter === 'PANAMAX') return v.vesselClass === 'Panamax';
    if (timelineFilter === 'CAPESIZE') return v.vesselClass === 'Capesize';
    return true;
  });

  const totalIdleDays = idleFleetVessels.reduce((sum, v) => sum + v.idleDaysForecast, 0);
  const totalIdleCost = idleState.data?.total_projected_idle_cost_usd || idleFleetVessels.reduce((sum, v) => sum + v.idleDaysForecast * v.dailyIdleCost, 0);
  const totalSavings = idleState.data?.total_potential_reposition_savings_usd || idleFleetVessels.reduce((sum, v) => sum + v.repositionSavings, 0);
  const optimizedIdleCost = Math.max(0, totalIdleCost - totalSavings);
  const savingsRate = totalIdleCost > 0 ? (totalSavings / totalIdleCost) * 100 : 0;
  const highRiskCount = idleState.data?.high_risk_vessel_count || idleFleetVessels.filter((v) => v.riskLevel === 'HIGH').length;
  const avgIdleCost = idleFleetVessels.length ? totalIdleCost / Math.max(totalIdleDays, 1) : 0;
  const previewMode = true;
  const clusterData = idleFleetVessels.slice(0, 4).map((v, index) => {
    const dailyLoss = v.dailyIdleCost;
    const risk = v.riskLevel === 'HIGH' && index === 0 ? 'CRITICAL' : v.riskLevel === 'HIGH' ? 'HIGH' : 'MODERATE';
    return {
      rank: String(index + 1).padStart(2, '0'),
      cluster: v.currentPort,
      vessels: v.name,
      count: 1,
      dailyLoss: formatUsd(dailyLoss),
      risk,
      bar: Math.min(100, Math.max(30, Math.round((dailyLoss / Math.max(...idleFleetVessels.map((item) => item.dailyIdleCost), 1)) * 100))),
      vesselId: v.id,
      riskBg: risk === 'CRITICAL' || risk === 'HIGH' ? 'bg-[#E05A47]' : 'bg-amber-400',
      riskText: risk === 'CRITICAL' || risk === 'HIGH' ? 'text-[#E05A47]' : 'text-amber-500',
    };
  });

  return (
    <div
      className="min-h-screen text-[#1D3045] antialiased selection:bg-[#E05A47] selection:text-white pb-20 relative flex flex-col"
      style={{ fontFamily: "'Inter', 'Plus Jakarta Sans', sans-serif", backgroundImage: 'url(/dashboard-bg.png)', backgroundColor: '#E2E8F0', backgroundSize: 'cover', backgroundAttachment: 'fixed' }}
    >
      {/* NAV */}
      <header className="w-full px-6 sm:px-8 md:px-12 pt-7 sm:pt-10 pb-6 flex items-center justify-between z-50 shrink-0">
        <nav className="hidden lg:flex items-center gap-6 xl:gap-8">
          <button onClick={() => onNavigate('landing')} className="text-[11px] tracking-[0.16em] uppercase font-semibold text-[#1D3045] hover:opacity-60 transition-opacity">FICOS</button>
          <button onClick={() => onNavigate('landing')} className="text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/60 hover:opacity-100 transition-opacity">FREIGHT INTELLIGENCE</button>
          <button onClick={() => onNavigate('vessel-intelligence')} className="text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/60 hover:opacity-100 transition-opacity">VESSEL INTELLIGENCE</button>
          <span className="relative text-[11px] tracking-[0.16em] uppercase font-semibold text-[#1D3045]">
            IDLE INTELLIGENCE
            <span className="absolute -bottom-3 left-0 right-0 h-[2px] w-full bg-[#1D3045]" />
          </span>
          <button onClick={() => onNavigate('dashboard')} className="text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/60 hover:opacity-100 transition-opacity">DASHBOARD</button>
          <button onClick={() => onNavigate('services')} className="text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/60 hover:opacity-100 transition-opacity">HOW IT WORKS</button>
        </nav>
        <div className="lg:hidden flex items-center gap-4">
          <button onClick={() => onNavigate('landing')} className="text-base font-bold tracking-[0.15em] text-[#1D3045]">FICOS</button>
          <span className="text-xs tracking-[0.15em] uppercase font-semibold text-[#1D3045] border-b-2 border-[#1D3045] pb-0.5">IDLE INTELLIGENCE</span>
        </div>
        <div className="flex items-center gap-6">
          <button onClick={() => onNavigate('landing')} className="text-xs tracking-[0.2em] uppercase font-medium hover:opacity-70 transition-opacity text-[#1D3045]">BACK</button>
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-xs tracking-[0.2em] uppercase font-medium text-[#1D3045]">INSIGHTS</span>
            <div className="w-5 h-5 rounded-full bg-[#1D3045] text-white flex items-center justify-center">
              <Info size={10} color="#FFFFFF" />
            </div>
          </div>
        </div>
      </header>

      <main className="w-full px-6 sm:px-8 md:px-12 space-y-5 flex-1 flex flex-col max-w-[1824px] mx-auto">

        {/* HERO ZONE */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 shrink-0">
          <div>
            <p className="text-[11px] tracking-[0.3em] uppercase font-bold text-[#E05A47] mb-2">PS DELIVERABLE (C) - IDLE SCENARIO MANAGEMENT</p>
            <h1 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-3xl sm:text-[2.8rem] font-extrabold uppercase tracking-tight text-[#1D3045] leading-none">
              IDLE &amp; REPOSITIONING<br className="hidden sm:block" /> INTELLIGENCE
            </h1>
            <p className="mt-3 text-sm text-[#1D3045]/65 font-medium max-w-xl">
              Predictive vessel idle risk forecasting / Repositioning cost-benefit optimization / Fleet capacity management
            </p>
          </div>
          <div className="inline-flex items-center gap-3 px-5 py-3 rounded-2xl bg-white/90 border border-[#E05A47]/30 shadow-lg shrink-0">
            <span className="w-3 h-3 rounded-full bg-[#E05A47] shadow-[0_0_12px_rgba(224,90,71,0.8)] animate-pulse" />
            <div>
              <p className="text-[10px] uppercase tracking-[0.2em] font-bold text-[#1D3045]/50">FLEET IDLE ALERT</p>
              <p className="text-sm font-bold text-[#E05A47]">
                {idleState.loading ? 'LOADING IDLE RISK...' : `${idleFleetVessels.length} VESSELS AT RISK · ${totalIdleDays} PROJECTED DAYS LOSS`}
              </p>
            </div>
          </div>
        </div>

        {/* â”€â”€ KPI STAT TILES â”€â”€ */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 shrink-0">
          {[
            { label: 'VESSELS AT IDLE RISK', value: String(idleFleetVessels.length), sub: `${highRiskCount} high risk`, accent: '#E05A47', tag: idleState.loading ? 'LOADING' : 'BACKEND' },
            { label: 'AVG IDLE COST / DAY', value: formatUsd(avgIdleCost), sub: 'Fuel · anchorage · port', accent: '#E05A47', tag: 'PER IDLE DAY' },
            { label: 'REPOSITION OPPORTUNITIES', value: String(idleFleetVessels.filter((v) => v.repositionSavings > 0).length), sub: 'Backend idle-risk routes', accent: '#059669', tag: `+${formatUsd(totalSavings)} POTENTIAL` },
            { label: 'PROJECTED IDLE DAYS (30D)', value: String(totalIdleDays), sub: 'Without intervention', accent: '#E05A47', tag: `=${formatUsd(totalIdleCost)} LOSS` },
          ].map((tile) => (
            <div key={tile.label} className="bg-white/95 backdrop-blur-md p-5 sm:p-6 rounded-2xl border border-[#1D3045]/10 shadow-[0_8px_24px_-4px_rgba(29,48,69,0.1)]">
              <div className="flex items-start justify-between gap-2 mb-3">
                <p style={{ fontFamily: "'Inter', sans-serif" }} className="text-[10px] tracking-[0.22em] uppercase font-bold text-[#1D3045]/40 leading-tight">{tile.label}</p>
                <span className="text-[9px] font-bold tracking-wider px-2 py-0.5 rounded-full border whitespace-nowrap shrink-0" style={{ color: tile.accent, borderColor: tile.accent + '40', backgroundColor: tile.accent + '12' }}>{tile.tag}</span>
              </div>
              <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", color: tile.accent === '#E05A47' ? '#1D3045' : tile.accent, fontSize: tile.value.length > 3 ? '2rem' : '3rem', fontWeight: 800, lineHeight: 1 }} className="tabular-nums">{tile.value}</p>
              <p className="mt-2 text-xs text-[#1D3045]/50 font-medium">{tile.sub}</p>
            </div>
          ))}
        </div>

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            SECTION 01 â€” IDLE RISK TIMELINE (Gantt)
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        <section className="bg-white/95 backdrop-blur-md rounded-2xl border border-[#1D3045]/10 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] overflow-hidden">
          {/* Section Header */}
          <div className="px-6 sm:px-8 py-5 border-b border-[#1D3045]/10 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#1D3045]/5">
            <div>
              <p className="text-[10px] tracking-[0.28em] uppercase font-bold text-[#E05A47] mb-1">SECTION 01 // IDLE RISK TIMELINE</p>
              <h2 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-xl sm:text-2xl font-bold uppercase tracking-tight text-[#1D3045]">30-60 DAY FLEET IDLE &amp; REPOSITIONING SCHEDULE</h2>
            </div>
            {/* Filter Pills */}
            <div className="flex items-center gap-1.5 bg-[#1D3045]/5 p-1 rounded-xl border border-[#1D3045]/10 shrink-0">
              {[{ label: 'ALL', value: 'ALL' }, { label: 'HIGH RISK', value: 'HIGH_RISK' }, { label: 'PANAMAX', value: 'PANAMAX' }, { label: 'CAPESIZE', value: 'CAPESIZE' }].map((f) => (
                <button key={f.value} onClick={() => setTimelineFilter(f.value as any)}
                  className={`px-3 py-1.5 text-[10px] uppercase tracking-wider font-bold rounded-lg transition-all ${timelineFilter === f.value ? 'bg-[#1D3045] text-white shadow-sm' : 'text-[#1D3045]/60 hover:text-[#1D3045]'}`}>
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          <div className="px-6 sm:px-8 py-6 space-y-3">
            {/* Timeline header */}
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-3 text-[10px] font-bold tracking-[0.15em] uppercase text-[#1D3045]/40">VESSEL / CLASS</div>
              <div className="col-span-9 grid grid-cols-6 text-center text-[10px] font-bold tracking-wider uppercase text-[#1D3045]/40">
              {['1-10', '11-20', '21-30', '31-40', '41-50', '51-60'].map(d => <span key={d}>DAY {d}</span>)}
              </div>
            </div>

            {/* Gantt rows */}
            <div className="space-y-2">
              {filteredVessels.map((v) => {
                const isSelected = v.id === selectedVesselId;
                const riskColor = v.riskLevel === 'HIGH' ? '#E05A47' : v.riskLevel === 'MEDIUM' ? '#F59E0B' : '#10B981';
                return (
                  <div key={v.id} onClick={() => setSelectedVesselId(v.id)}
                    className={`grid grid-cols-12 gap-2 items-center p-3 rounded-xl border transition-all cursor-pointer ${isSelected ? 'bg-[#1D3045]/5 border-[#1D3045]/40 shadow-sm' : 'bg-white/60 border-[#1D3045]/10 hover:border-[#1D3045]/20 hover:bg-white/80'}`}>
                    <div className="col-span-3 flex items-center gap-2.5">
                      <div className="w-2.5 h-2.5 rounded-full shrink-0 ring-2 ring-white" style={{ backgroundColor: riskColor, boxShadow: v.riskLevel === 'HIGH' ? `0 0 8px ${riskColor}` : 'none' }} />
                      <div className="min-w-0">
                        <p className="text-[11px] font-bold text-[#1D3045] truncate leading-none">{v.name}</p>
                        <p className="text-[9px] text-[#1D3045]/50 font-medium mt-0.5 font-mono">{v.vesselClass} / {v.dwt}</p>
                      </div>
                    </div>
                    <div className="col-span-9 h-8 bg-[#1D3045]/5 rounded-lg border border-[#1D3045]/10 overflow-hidden flex items-center px-1 gap-0.5">
                      <div className="h-6 bg-[#1D3045] rounded-sm flex items-center justify-center text-[8px] font-bold text-white px-1.5 truncate" style={{ width: `${(v.activeDays[1] / 60) * 100}%` }}>
                        Active {v.activeDays[1]}d
                      </div>
                      <div className="h-6 rounded-sm flex items-center justify-center text-[8px] font-bold text-white px-1.5 truncate" style={{ width: `${((v.idleWindow[1] - v.idleWindow[0] + 1) / 60) * 100}%`, backgroundColor: '#E05A47', boxShadow: '0 0 12px rgba(224,90,71,0.5)' }}>
                        IDLE {v.idleWindow[1] - v.idleWindow[0] + 1}d
                      </div>
                      <div className="h-6 bg-amber-500 rounded-sm flex items-center justify-center text-[8px] font-bold text-white px-1.5 truncate" style={{ width: `${((v.repositionWindow[1] - v.repositionWindow[0] + 1) / 60) * 100}%` }}>
                        Repos. {v.repositionWindow[1] - v.repositionWindow[0] + 1}d
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-between pt-3 border-t border-[#1D3045]/10 text-[10px] font-bold tracking-wide text-[#1D3045]/60">
              <div className="flex items-center gap-5">
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#1D3045]" /> Active / Cargo</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[#E05A47]" /> Idle Risk Window</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-amber-500" /> Reposition Window</span>
              </div>
              <span className="italic text-[#1D3045]/40">Click a row to see arbitrage in Section 03</span>
            </div>
          </div>
        </section>

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            SECTION 02 â€” DEMAND CHART + CLUSTER RANKING
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        <section className="bg-white/95 backdrop-blur-md rounded-2xl border border-[#1D3045]/10 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] overflow-hidden">
          <div className="px-6 sm:px-8 py-5 border-b border-[#1D3045]/10 bg-[#1D3045]/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="text-[10px] tracking-[0.28em] uppercase font-bold text-[#E05A47] mb-1">SECTION 02 // BACKEND IDLE-RISK EXPOSURE</p>
              <h2 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-xl sm:text-2xl font-bold uppercase tracking-tight text-[#1D3045]">IDLE COST EXPOSURE vs VULNERABILITY CLUSTERS</h2>
            </div>
            <div className="inline-flex items-center gap-2 bg-[#E05A47]/10 text-[#E05A47] border border-[#E05A47]/30 px-4 py-2 rounded-xl font-bold text-[11px] tracking-wide shrink-0">
              <AlertTriangle size={14} />
              LIVE /idle-risk DATA
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-x divide-[#1D3045]/8">
            {/* LEFT: DEMAND CHART */}
            <div className="lg:col-span-7 p-6 sm:p-8 space-y-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-base font-bold text-[#1D3045] uppercase tracking-wide">VESSEL IDLE EXPOSURE FROM BACKEND</p>
                  <p className="text-xs text-[#1D3045]/50 mt-1">No route-demand curve is shown until a real demand endpoint exists.</p>
                </div>
                <span className="text-[10px] font-bold bg-[#E05A47]/15 text-[#E05A47] px-3 py-1 rounded-lg border border-[#E05A47]/30 shrink-0">{highRiskCount} HIGH RISK</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-xl bg-[#1D3045] text-white p-4">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-white/50 font-bold">Projected Idle Cost</p>
                  <p className="mt-2 text-3xl font-black tabular-nums">{formatUsd(totalIdleCost)}</p>
                </div>
                <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-emerald-700/70 font-bold">Reposition Savings</p>
                  <p className="mt-2 text-3xl font-black text-emerald-700 tabular-nums">{formatUsd(totalSavings)}</p>
                </div>
                <div className="rounded-xl bg-[#E05A47]/10 border border-[#E05A47]/20 p-4">
                  <p className="text-[10px] uppercase tracking-[0.2em] text-[#E05A47]/80 font-bold">Idle Days</p>
                  <p className="mt-2 text-3xl font-black text-[#E05A47] tabular-nums">{totalIdleDays}</p>
                </div>
              </div>

              <div className="space-y-3">
                {idleFleetVessels.slice(0, 5).map((v) => {
                  const exposure = v.idleDaysForecast * v.dailyIdleCost;
                  const pct = Math.max(8, Math.min(100, (exposure / Math.max(totalIdleCost, 1)) * 100));
                  return (
                    <div key={v.id} className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] font-bold">
                        <span className="text-[#1D3045]">{v.name}</span>
                        <span className="font-mono text-[#E05A47]">{formatUsd(exposure)}</span>
                      </div>
                      <div className="h-4 rounded-lg bg-[#1D3045]/8 border border-[#1D3045]/10 overflow-hidden">
                        <div className="h-full bg-[#E05A47]" style={{ width: `${pct}%` }} />
                      </div>
                      <p className="text-[10px] text-[#1D3045]/50">{v.currentPort} · {v.idleDaysForecast} days · {formatUsd(v.dailyIdleCost)}/day</p>
                    </div>
                  );
                })}
              </div>

              {/* SVG Chart */}
              <div className="hidden relative h-72 w-full">
                {/* Idle risk band */}
                <div className="absolute inset-y-0 bottom-10 bg-[#E05A47]/10 border-l-2 border-r-2 border-[#E05A47]/40 rounded-md pointer-events-none" style={{ left: '22%', width: '24%' }}>
                  <span className="absolute top-3 left-2 text-[9px] font-bold text-[#E05A47] uppercase tracking-wider bg-white/90 px-2 py-0.5 rounded shadow-sm">IDLE RISK BAND</span>
                </div>

                {/* Annotation callout */}
                <div className="absolute top-8 left-[27%] z-20 bg-[#1D3045] text-white p-3 rounded-xl border border-[#E05A47] shadow-xl pointer-events-none max-w-[200px]">
                  <div className="flex items-center gap-1.5 text-[#E05A47] text-[11px] font-bold uppercase tracking-wide">
                    <TrendingDown size={13} /> DEMAND -32%
                  </div>
                  <p className="text-[10px] text-white/70 leading-snug mt-1">Day 15-28: port maintenance + steel mill slowdown. 10-day idle bottleneck.</p>
                </div>

                <svg viewBox="0 0 500 200" className="w-full h-full overflow-visible">
                  <defs>
                    <linearGradient id="navyGrad2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#1D3045" stopOpacity="0.15" />
                      <stop offset="100%" stopColor="#1D3045" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {/* Grid */}
                  {[35, 80, 125, 165].map(y => <line key={y} x1="0" y1={y} x2="500" y2={y} stroke="#1D3045" strokeOpacity="0.08" strokeDasharray="4 3" />)}
                  {/* Y axis labels */}
                  <text x="4" y="38" fontSize="9" fill="#1D3045" opacity="0.4" fontFamily="Inter">HIGH</text>
                  <text x="4" y="83" fontSize="9" fill="#1D3045" opacity="0.4" fontFamily="Inter">MED</text>
                  <text x="4" y="128" fontSize="9" fill="#1D3045" opacity="0.4" fontFamily="Inter">LOW</text>
                  {/* Route 1: Dampier â†” Dhamra â€” steep dip, fast recovery */}
                  <path d="M 20 38 C 80 42, 120 95, 145 138 C 170 162, 205 158, 245 122 C 295 82, 365 44, 490 22" fill="none" stroke="#1D3045" strokeWidth="3.5" strokeLinecap="round" />
                  {/* Route 2: Gladstone â†” Vizag â€” deepest drop */}
                  <path d="M 20 58 C 70 74, 108 132, 135 168 C 162 188, 215 183, 262 153 C 335 112, 415 76, 490 48" fill="none" stroke="#E05A47" strokeWidth="3" strokeLinecap="round" />
                  {/* Route 3: Newcastle â†” Gangavaram â€” steady */}
                  <path d="M 20 78 C 95 74, 165 88, 225 83 C 285 78, 385 67, 490 52" fill="none" stroke="#F59E0B" strokeWidth="2.5" strokeDasharray="7 4" strokeLinecap="round" />
                  {/* Dip marker */}
                  <circle cx="145" cy="138" r="6" fill="#E05A47" opacity="0.3" />
                  <circle cx="145" cy="138" r="4.5" fill="#E05A47" stroke="#fff" strokeWidth="1.5" />
                </svg>

                {/* X axis */}
                <div className="flex items-center justify-between text-[10px] font-bold text-[#1D3045]/50 pt-1 border-t border-[#1D3045]/10 mt-1" style={{ fontFamily: "'Inter', monospace" }}>
                  <span>DAY 1</span>
                  <span className="text-[#E05A47]">DAY 15 DIP</span>
                  <span className="text-[#E05A47]">DAY 28 RECOVERY</span>
                  <span>DAY 45</span>
                  <span>DAY 60</span>
                </div>
              </div>

              {/* Legend */}
              <div className="hidden flex-wrap items-center gap-5 text-[11px] font-bold pt-3 border-t border-[#1D3045]/10">
                <span className="flex items-center gap-2"><span className="w-5 h-1.5 bg-[#1D3045] rounded-full inline-block" /> Dampier / Dhamra (Iron Ore)</span>
                <span className="flex items-center gap-2 text-[#E05A47]"><span className="w-5 h-1.5 bg-[#E05A47] rounded-full inline-block" /> Gladstone / Vizag (Coal -45%)</span>
                <span className="flex items-center gap-2 text-amber-600"><span className="w-5 h-1.5 bg-amber-500 rounded-full inline-block" /> Newcastle / Gangavaram</span>
              </div>
            </div>

            {/* RIGHT: VULNERABILITY CLUSTER RANKING */}
            <div className="lg:col-span-5 flex flex-col">
              {/* Cluster header */}
              <div className="px-6 py-4 bg-[#1D3045] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MapPin size={16} className="text-[#E05A47]" />
                  <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-sm font-bold text-white uppercase tracking-wider">VULNERABILITY CLUSTERS</p>
                </div>
                <span className="text-[10px] font-bold text-[#E05A47] bg-[#E05A47]/20 px-2.5 py-1 rounded-lg tracking-wider">4 AT RISK</span>
              </div>

              {/* Cluster rows */}
              <div className="flex-1 divide-y divide-[#1D3045]/8 bg-white">
                {clusterData.map((item) => (
                  <div key={item.cluster} onClick={() => setSelectedVesselId(item.vesselId)}
                    className="px-6 py-5 hover:bg-[#1D3045]/5 transition-colors cursor-pointer group">
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex items-start gap-3">
                        <span style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-4xl font-black text-[#1D3045]/12 leading-none select-none">{item.rank}</span>
                        <div>
                          <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-[15px] font-bold text-[#1D3045] leading-tight group-hover:text-[#E05A47] transition-colors">{item.cluster}</p>
                          <p className="text-[11px] text-[#1D3045]/50 font-medium mt-0.5">{item.vessels}</p>
                          <span className={`inline-block mt-1.5 text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded text-white ${item.riskBg}`}>{item.risk}</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", color: item.riskText.replace('text-', '').includes('E05') ? '#E05A47' : '' }} className={`text-2xl font-black tabular-nums ${item.riskText}`}>{item.dailyLoss}</p>
                        <p className="text-[10px] text-[#1D3045]/40 font-medium">/day idle cost</p>
                      </div>
                    </div>
                    {/* Risk bar */}
                    <div className="h-1.5 w-full bg-[#1D3045]/10 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${item.riskBg}`} style={{ width: `${item.bar}%`, transition: 'width 0.6s ease' }} />
                    </div>
                    <div className="flex justify-between text-[9px] text-[#1D3045]/35 font-medium mt-1">
                      <span>{item.count} vessel{item.count > 1 ? 's' : ''}</span>
                      <span>{item.bar}% exposure</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="px-6 py-3 bg-[#1D3045]/5 border-t border-[#1D3045]/10">
                <p className="text-[10px] text-[#1D3045]/40 font-medium italic text-center">Click any cluster to view repositioning plan in Section 03</p>
              </div>
            </div>
          </div>
        </section>

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            SECTION 03 â€” REPOSITIONING ARBITRAGE
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        <section className="bg-white/95 backdrop-blur-md rounded-2xl border border-[#1D3045]/10 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] overflow-hidden">
          <div className="px-6 sm:px-8 py-5 border-b border-[#1D3045]/10 bg-[#1D3045]/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="text-[10px] tracking-[0.28em] uppercase font-bold text-[#E05A47] mb-1">SECTION 03 // OPTIMAL REPOSITIONING STRATEGY</p>
              <h2 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-xl sm:text-2xl font-bold uppercase tracking-tight text-[#1D3045]">COST-BENEFIT REPOSITIONING ARBITRAGE</h2>
            </div>
            <div className="flex items-center gap-2 bg-[#1D3045]/5 px-4 py-2 rounded-xl border border-[#1D3045]/10 shrink-0">
              <span className="text-[10px] uppercase font-bold text-[#1D3045]/50 tracking-wider">TARGET VESSEL:</span>
              <select value={selectedVesselId} onChange={(e) => setSelectedVesselId(e.target.value)}
                className="bg-transparent text-[#1D3045] font-bold text-xs focus:outline-none cursor-pointer">
                {idleFleetVessels.map((v) => (
                  <option key={v.id} value={v.id}>{v.name} ({v.vesselClass} - {v.riskLevel})</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-x divide-[#1D3045]/8">
            {/* LEFT: AI PLAN CARD */}
            <div className="lg:col-span-5 bg-[#1D3045] p-7 sm:p-8 flex flex-col justify-between relative overflow-hidden">
              <div className="absolute -top-10 -right-10 w-48 h-48 bg-[#E05A47]/10 rounded-full blur-3xl pointer-events-none" />
              <div className="absolute bottom-0 left-0 w-32 h-32 bg-[#E05A47]/5 rounded-full blur-2xl pointer-events-none" />

              <div className="relative">
                <div className="flex items-center justify-between mb-5">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-[#E05A47] bg-[#E05A47]/20 px-3 py-1 rounded-lg">AI REPOSITIONING PLAN</span>
                  <span className="text-[10px] font-bold text-white/50">CONFIDENCE 94.8%</span>
                </div>

                <p className="text-[11px] font-bold uppercase tracking-widest text-white/40 mb-1">RECOMMENDED ACTION</p>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-2xl sm:text-3xl font-black text-white leading-tight uppercase">
                  REPOSITION TO<br />{selectedVessel.recommendedRoute.toUpperCase()}
                </h3>
                <p className="text-sm text-white/50 mt-2 font-medium">
                  Avoiding {selectedVessel.idleDaysForecast} days idle loss at {selectedVessel.currentPort}
                </p>

                {/* Financial grid */}
                <div className="mt-6 grid grid-cols-2 gap-4">
                  <div className="bg-white/10 rounded-xl p-4 border border-white/10">
                    <p className="text-[10px] uppercase tracking-wider text-white/40 font-bold mb-1.5">IDLE LOSS IF HOLDING</p>
                    <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-3xl font-black text-[#E05A47] tabular-nums">
                      -${(selectedVessel.idleDaysForecast * selectedVessel.dailyIdleCost).toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-white/10 rounded-xl p-4 border border-white/10">
                    <p className="text-[10px] uppercase tracking-wider text-white/40 font-bold mb-1.5">REPOSITIONING COST</p>
                    <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-3xl font-black text-amber-300 tabular-nums">-$38,400</p>
                  </div>
                </div>

                <div className="mt-4 bg-white/10 rounded-xl p-5 border border-emerald-400/30">
                  <p className="text-[10px] uppercase tracking-wider text-white/40 font-bold mb-1.5">NET ARBITRAGE GAIN</p>
                  <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-4xl font-black text-emerald-400 tabular-nums">
                    +${selectedVessel.repositionSavings.toLocaleString()}
                  </p>
                  <p className="text-[11px] text-emerald-300/70 font-medium mt-1">After all costs deducted</p>
                </div>
              </div>

              <button onClick={() => onNavigate('dashboard')}
                className="relative mt-6 w-full py-4 rounded-xl bg-[#E05A47] hover:bg-[#c94937] text-white text-sm uppercase tracking-[0.18em] font-black flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-[#E05A47]/40">
                EXECUTE REPOSITIONING PLAN
                <ArrowUpRight size={16} />
              </button>
            </div>

            {/* RIGHT: COMPARISON BARS */}
            <div className="lg:col-span-7 p-6 sm:p-8 flex flex-col gap-5">
              <div>
                <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-base font-bold uppercase tracking-wide text-[#1D3045]">FINANCIAL OUTCOME COMPARISON</p>
                <p className="text-xs text-[#1D3045]/50 mt-1">{selectedVessel.name} / 4 strategic scenarios ranked by net impact</p>
              </div>

              <div className="space-y-5 flex-1 flex flex-col justify-center">
                {[
                  { option: 'REPOSITION TO PARADIP / DHAMRA', amount: selectedVessel.repositionSavings, label: `+$${selectedVessel.repositionSavings.toLocaleString()}`, isOptimal: true, positive: true },
                  { option: 'REPOSITION TO VIZAG OUTER', amount: 98200, label: '+$98,200', isOptimal: false, positive: true },
                  { option: 'SLOW STEAMING EN ROUTE', amount: 52000, label: '-$52,000', isOptimal: false, positive: false },
                  { option: `HOLD IDLE AT ${selectedVessel.currentPort.toUpperCase()}`, amount: selectedVessel.idleDaysForecast * selectedVessel.dailyIdleCost, label: `-$${(selectedVessel.idleDaysForecast * selectedVessel.dailyIdleCost).toLocaleString()}`, isOptimal: false, positive: false },
                ].map((item) => {
                  const maxVal = 180000;
                  const pct = Math.max(8, (item.amount / maxVal) * 100);
                  return (
                    <div key={item.option} className="space-y-2">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 min-w-0">
                          {item.isOptimal && <span className="text-[9px] font-black bg-[#E05A47] text-white px-2 py-0.5 rounded uppercase tracking-wider shrink-0">BEST</span>}
                          <span className={`text-[11px] font-bold uppercase tracking-wide truncate ${item.isOptimal ? 'text-[#E05A47]' : 'text-[#1D3045]/70'}`}>{item.option}</span>
                        </div>
                        <span style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className={`text-xl font-black tabular-nums shrink-0 ${item.positive ? 'text-emerald-700' : 'text-[#E05A47]'}`}>{item.label}</span>
                      </div>
                      <div className="h-5 bg-[#1D3045]/5 rounded-lg overflow-hidden border border-[#1D3045]/10">
                        <div className={`h-full rounded-lg flex items-center justify-end pr-2 text-[9px] font-black text-white transition-all duration-700 ${item.isOptimal ? 'bg-[#E05A47] shadow-[0_0_12px_rgba(224,90,71,0.4)]' : item.positive ? 'bg-[#1D3045]' : 'bg-red-400'}`}
                          style={{ width: `${pct}%` }}>{item.label}</div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <p className="text-[10px] text-[#1D3045]/40 italic font-medium pt-3 border-t border-[#1D3045]/10 text-right">
                *Repositioning to Paradip yields +${selectedVessel.repositionSavings.toLocaleString()} net vs -${(selectedVessel.idleDaysForecast * selectedVessel.dailyIdleCost).toLocaleString()} idle loss
              </p>
            </div>
          </div>
        </section>

        {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            SECTION 04 â€” FLEET-WIDE IDLE COST TREND
        â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
        <section className="bg-white/95 backdrop-blur-md rounded-2xl border border-[#1D3045]/10 shadow-[0_10px_32px_-4px_rgba(29,48,69,0.12)] overflow-hidden">
          <div className="px-6 sm:px-8 py-5 border-b border-[#1D3045]/10 bg-[#1D3045]/5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="text-[10px] tracking-[0.28em] uppercase font-bold text-[#E05A47] mb-1">SECTION 04 // FLEET-WIDE FINANCIAL EXPOSURE</p>
              <h2 style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-xl sm:text-2xl font-bold uppercase tracking-tight text-[#1D3045]">60-DAY AGGREGATE IDLE COST vs AI REPOSITIONING SAVINGS</h2>
            </div>
            <div className="flex items-center gap-2 bg-emerald-100 text-emerald-900 border border-emerald-300 px-4 py-2 rounded-xl font-bold text-[11px] shrink-0">
              <CheckCircle2 size={14} className="text-emerald-700" />
              AI REPOSITIONING SAVES {formatUsd(totalSavings)} ({savingsRate.toFixed(1)}% LOSS REDUCTION) OVER 60 DAYS
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-x divide-[#1D3045]/8">
            {/* CHART */}
            <div className="lg:col-span-8 p-6 sm:p-8 space-y-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-base font-bold uppercase tracking-wide text-[#1D3045]">PROJECTED CUMULATIVE IDLE LOSS ($)</p>
                  <p className="text-xs text-[#1D3045]/50 mt-1">Unmitigated baseline vs AI-guided repositioning trajectory</p>
                </div>
                <div className="flex flex-col gap-1.5 text-[11px] font-bold text-right">
                  <span className="flex items-center gap-2 text-[#E05A47]"><span className="w-4 h-1.5 bg-[#E05A47] rounded-full" /> Unmitigated ($1.42M)</span>
                  <span className="flex items-center gap-2 text-emerald-700"><span className="w-4 h-1.5 bg-emerald-500 rounded-full" /> AI-Optimized ($340K)</span>
                </div>
              </div>

              <div className="relative h-72 w-full">
                {/* In-chart savings badge */}
                <div className="absolute top-10 right-[16%] z-20 bg-emerald-900 text-white p-3 rounded-xl border-2 border-emerald-400 shadow-2xl pointer-events-none text-center" style={{ animation: 'bounce 3s infinite' }}>
                  <p className="text-[13px] font-black text-emerald-300 tracking-wider">{formatUsd(totalSavings)} SAVED</p>
                  <p className="text-[9px] text-white/70 font-medium mt-0.5">Gap between curves = Net Profit Preserved</p>
                </div>
                {/* Day 30 badge */}
                <div className="absolute bottom-14 left-[46%] z-20 bg-[#1D3045] text-white px-3 py-1.5 rounded-lg border border-emerald-400/40 shadow-md text-[10px] font-bold flex items-center gap-1.5">
                  <Sparkles size={11} className="text-emerald-400" /> {savingsRate.toFixed(1)}% LOSS AVOIDANCE RATE
                </div>

                <svg viewBox="0 0 600 200" className="w-full h-full overflow-visible">
                  <defs>
                    <linearGradient id="savingsGradV2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10B981" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="#10B981" stopOpacity="0.04" />
                    </linearGradient>
                    <filter id="coralGlowV2">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                  </defs>
                  {/* Savings gap area */}
                  <polygon points="0,180 120,150 240,115 360,75 480,38 600,10 600,145 480,150 360,158 240,165 120,172 0,180" fill="url(#savingsGradV2)" stroke="rgba(16,185,129,0.25)" strokeDasharray="4 2" />
                  {/* Grid */}
                  {[40, 90, 140].map(y => <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#1D3045" strokeOpacity="0.07" strokeDasharray="3 3" />)}
                  {/* Unmitigated â€” coral, thick, glowing */}
                  <path d="M 0 180 C 140 145, 280 100, 420 50 L 600 10" fill="none" stroke="#E05A47" strokeWidth="4.5" filter="url(#coralGlowV2)" strokeLinecap="round" />
                  {/* AI Optimized â€” emerald, calm */}
                  <path d="M 0 180 C 140 174, 280 166, 420 156 L 600 145" fill="none" stroke="#10B981" strokeWidth="3.5" strokeLinecap="round" />
                  {/* Gap pointer line */}
                  <line x1="590" y1="12" x2="590" y2="143" stroke="#10B981" strokeWidth="2" strokeDasharray="3 3" />
                  <circle cx="600" cy="10" r="6" fill="#E05A47" stroke="#fff" strokeWidth="2" />
                  <circle cx="600" cy="145" r="6" fill="#10B981" stroke="#fff" strokeWidth="2" />
                </svg>

                {/* X axis */}
                <div className="flex items-center justify-between text-[10px] font-bold text-[#1D3045]/40 pt-2 border-t border-[#1D3045]/10" style={{ fontFamily: "'Inter', monospace" }}>
                  <span>DAY 1 ($0)</span>
                  <span>DAY 15 ($350K)</span>
                  <span className="font-black text-[#1D3045]/70">DAY 30 ($720K vs $165K)</span>
                  <span>DAY 45 ($1.1M)</span>
                  <span className="text-[#E05A47] font-black">DAY 60 ($1.42M vs $340K)</span>
                </div>
              </div>
            </div>

            {/* RIGHT: MACRO STATS */}
            <div className="lg:col-span-4 flex flex-col divide-y divide-[#1D3045]/8">
              {/* Stat 1 */}
              <div className="p-6 sm:p-7 flex flex-col justify-center flex-1">
                <p className="text-[10px] font-black uppercase tracking-[0.22em] text-[#1D3045]/40 mb-3">UNMITIGATED IDLE LOSS (60D)</p>
                <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-4xl font-black text-[#E05A47] tabular-nums leading-none">{formatUsd(totalIdleCost)}</p>
                <p className="text-xs text-[#1D3045]/50 font-medium mt-2">Without vessel repositioning execution</p>
              </div>
              {/* Stat 2 */}
              <div className="p-6 sm:p-7 flex flex-col justify-center flex-1">
                <p className="text-[10px] font-black uppercase tracking-[0.22em] text-[#1D3045]/40 mb-3">AI OPTIMIZED IDLE LOSS (60D)</p>
                <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-4xl font-black text-emerald-600 tabular-nums leading-none">{formatUsd(optimizedIdleCost)}</p>
                <p className="text-xs text-[#1D3045]/50 font-medium mt-2">With AI repositioning to high-demand corridors</p>
              </div>
              {/* Stat 3 â€” hero */}
              <div className="p-6 sm:p-7 bg-[#1D3045] flex flex-col justify-center flex-1">
                <p className="text-[10px] font-black uppercase tracking-[0.22em] text-white/30 mb-3">NET FLEET SAVINGS RATE</p>
                <p style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }} className="text-5xl font-black text-emerald-400 leading-none tabular-nums">{savingsRate.toFixed(1)}%</p>
                <p className="text-base font-bold text-emerald-300 mt-2">+{formatUsd(totalSavings)} saved</p>
                <p className="text-xs text-white/40 font-medium mt-1">Net bottom-line preservation over 60 days</p>
              </div>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
