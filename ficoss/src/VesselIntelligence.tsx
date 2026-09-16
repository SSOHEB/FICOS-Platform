import React, { useState, useEffect, useRef } from 'react';
import { 
  Home, 
  LayoutGrid, 
  Network, 
  CreditCard, 
  Table2, 
  Settings, 
  User, 
  Search, 
  Filter, 
  Layers, 
  Plus, 
  Minus, 
  Compass, 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronDown, 
  X, 
  ArrowRight, 
  Anchor, 
  ShieldCheck, 
  Gauge, 
  Navigation2, 
  Radio, 
  Sparkles,
  Info,
  Ship,
  TrendingUp,
  AlertTriangle,
  ArrowUpRight
} from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { apiGet, ApiStatus } from './apiClient';

interface VesselIntelligenceProps {
  onNavigate: (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => void;
}

export interface Vessel {
  id: string;
  code: string;
  name: string;
  vesselClass: 'Handysize' | 'Supramax' | 'Panamax' | 'Capesize';
  origin: string;
  originCoords: [number, number];
  destination: string;
  destCoords: [number, number];
  status: 'EN ROUTE' | 'AVAILABLE' | 'AT PORT' | 'IDLE';
  speed: string;
  eta: string;
  remainingNm: string;
  cargo: string;
  charterStatus: string;
  progress: number;
  currentHeading: number;
  rate: string;
  feasibilityScore: string;
  waypoints: [number, number][];
}

type EndpointState<T> = {
  data: T | null;
  status: ApiStatus;
  loading: boolean;
  error: string | null;
};

type FeasibilityClass = {
  vessel_class: string;
  code: string;
  typical_dwt: number;
  is_feasible: boolean;
  reasons: string[];
};

type VesselFeasibilityResponse = {
  destination_port: string;
  cargo_quantity_mt: number;
  feasibility_percentage: number;
  recommended_vessel_class: string;
  forecast_delta_pct: string;
  per_class_comparison: FeasibilityClass[];
};

type FleetStatusResponse = {
  summary: Record<string, number>;
  fleet: Array<{
    id: string;
    code: string;
    name: string;
    vessel_class: string;
    dwt: number;
    status: string;
    origin: string;
    destination: string;
    speed_knots: number;
    eta: string;
    remaining_nm: number;
    cargo: string;
    charter_status: string;
    feasibility_score: string;
  }>;
};

const emptyEndpoint = <T,>(): EndpointState<T> => ({ data: null, status: 'idle', loading: false, error: null });

const classRates: Record<string, number> = {
  Handysize: 18.5,
  Supramax: 17.8,
  Panamax: 16.9,
  Capesize: 22.4,
};

const classSubtitles: Record<string, string> = {
  Handysize: '28,000 - 38,000 DWT',
  Supramax: '50,000 - 60,000 DWT',
  Panamax: '65,000 - 85,000 DWT',
  Capesize: '100,000+ DWT',
};

// Bulk Carrier SVG Graphics styled in FICOS Navy & Coral/Teal
const BulkCarrierGraphic: React.FC<{ vesselClass: string; active?: boolean }> = ({ vesselClass, active }) => {
  return (
    <svg viewBox="0 0 240 56" className="w-full h-11 transition-all duration-300">
      <defs>
        <linearGradient id={`hullGradLight-${vesselClass}`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={active ? "#1D3045" : "#334E68"} />
          <stop offset="60%" stopColor={active ? "#243B53" : "#486581"} />
          <stop offset="100%" stopColor={active ? "#102A43" : "#243B53"} />
        </linearGradient>
        <linearGradient id="keelGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#C53030" />
          <stop offset="100%" stopColor="#9B2C2C" />
        </linearGradient>
      </defs>
      
      {/* Upper superstructure & bridge */}
      <rect x="25" y="14" width="22" height="18" fill="#FFFFFF" stroke="#1D3045" strokeWidth="1" rx="1" />
      <rect x="28" y="17" width="16" height="4" fill="#1D3045" opacity="0.85" />
      <rect x="33" y="8" width="6" height="8" fill="#E05A47" />
      <line x1="36" y1="4" x2="36" y2="8" stroke="#1D3045" strokeWidth="1.5" />
      
      {/* Cranes / Cargo holds */}
      {vesselClass === 'Handysize' && (
        <>
          <line x1="70" y1="22" x2="85" y2="12" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="120" y1="22" x2="135" y2="12" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="170" y1="22" x2="185" y2="12" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <rect x="55" y="22" width="30" height="6" fill="#627D98" rx="1" />
          <rect x="105" y="22" width="30" height="6" fill="#627D98" rx="1" />
          <rect x="155" y="22" width="30" height="6" fill="#627D98" rx="1" />
        </>
      )}
      {vesselClass === 'Supramax' && (
        <>
          <line x1="65" y1="22" x2="78" y2="13" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="105" y1="22" x2="118" y2="13" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="145" y1="22" x2="158" y2="13" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="185" y1="22" x2="198" y2="13" stroke="#1D3045" strokeWidth="1.8" strokeLinecap="round" />
          <rect x="52" y="22" width="24" height="6" fill="#627D98" rx="1" />
          <rect x="92" y="22" width="24" height="6" fill="#627D98" rx="1" />
          <rect x="132" y="22" width="24" height="6" fill="#627D98" rx="1" />
          <rect x="172" y="22" width="24" height="6" fill="#627D98" rx="1" />
        </>
      )}
      {vesselClass === 'Panamax' && (
        <>
          <rect x="55" y="20" width="26" height="8" fill="#486581" rx="1" />
          <rect x="88" y="20" width="26" height="8" fill="#486581" rx="1" />
          <rect x="121" y="20" width="26" height="8" fill="#486581" rx="1" />
          <rect x="154" y="20" width="26" height="8" fill="#486581" rx="1" />
          <rect x="187" y="20" width="24" height="8" fill="#486581" rx="1" />
        </>
      )}
      {vesselClass === 'Capesize' && (
        <>
          <rect x="50" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="73" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="96" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="119" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="142" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="165" y="18" width="18" height="10" fill="#334E68" rx="1" />
          <rect x="188" y="18" width="18" height="10" fill="#334E68" rx="1" />
        </>
      )}

      {/* Main Hull */}
      <path
        d="M 15 28 L 215 28 Q 228 28 232 38 L 222 44 L 20 44 Q 14 40 15 28 Z"
        fill={`url(#hullGradLight-${vesselClass})`}
      />
      {/* Red Keel / Underwater Hull */}
      <path
        d="M 20 44 L 222 44 L 212 51 Q 190 54 120 54 Q 40 54 25 51 Z"
        fill="url(#keelGrad)"
      />
      {/* Draft Marks */}
      <line x1="220" y1="34" x2="228" y2="34" stroke="#FFFFFF" strokeWidth="1" opacity="0.8" />
      <line x1="218" y1="40" x2="224" y2="40" stroke="#FFFFFF" strokeWidth="1" opacity="0.8" />
      <circle cx="218" cy="36" r="1.5" fill="#E05A47" />
    </svg>
  );
};

// Realistic Waypoint Paths from Australia to East Coast India Ports
const ROUTE_AUSTRALIA_DHAMRA: [number, number][] = [
  [-20.31, 118.57],
  [-16.50, 112.00],
  [-11.20, 104.50],
  [-6.00, 95.00],
  [5.50, 88.00],
  [12.00, 85.00],
  [16.50, 84.50],
  [20.80, 86.95],
];

const ROUTE_AUSTRALIA_PARADIP: [number, number][] = [
  [-20.31, 118.57],
  [-15.00, 110.00],
  [-8.50, 100.00],
  [-2.00, 93.00],
  [6.00, 87.50],
  [14.00, 85.00],
  [20.26, 86.67],
];

const ROUTE_AUSTRALIA_VIZAG: [number, number][] = [
  [-20.31, 118.57],
  [-14.00, 108.00],
  [-7.00, 96.00],
  [3.00, 89.00],
  [11.00, 84.00],
  [17.68, 83.21],
];

const ROUTE_AUSTRALIA_GANGAVARAM: [number, number][] = [
  [-20.31, 118.57],
  [-13.50, 106.00],
  [-5.00, 94.00],
  [4.50, 87.00],
  [13.00, 83.50],
  [17.63, 83.23],
];

const ROUTE_AUSTRALIA_GOPALPUR: [number, number][] = [
  [-20.31, 118.57],
  [-12.00, 105.00],
  [-4.00, 93.50],
  [7.00, 86.50],
  [15.50, 85.00],
  [19.30, 84.96],
];

const ROUTE_AUSTRALIA_HALDIA: [number, number][] = [
  [-20.31, 118.57],
  [-10.00, 102.00],
  [-1.00, 92.00],
  [8.00, 88.00],
  [17.00, 87.50],
  [22.02, 88.06],
];

const DESTINATION_META: Record<string, { coords: [number, number]; waypoints: [number, number][] }> = {
  DHAMRA: { coords: [20.80, 86.95], waypoints: ROUTE_AUSTRALIA_DHAMRA },
  PARADIP: { coords: [20.26, 86.67], waypoints: ROUTE_AUSTRALIA_PARADIP },
  VIZAG: { coords: [17.68, 83.21], waypoints: ROUTE_AUSTRALIA_VIZAG },
  VISAKHAPATNAM: { coords: [17.68, 83.21], waypoints: ROUTE_AUSTRALIA_VIZAG },
  GANGAVARAM: { coords: [17.63, 83.23], waypoints: ROUTE_AUSTRALIA_GANGAVARAM },
  GOPALPUR: { coords: [19.30, 84.96], waypoints: ROUTE_AUSTRALIA_GOPALPUR },
  HALDIA: { coords: [22.02, 88.06], waypoints: ROUTE_AUSTRALIA_HALDIA },
  QINGDAO: { coords: [36.07, 120.38], waypoints: ROUTE_AUSTRALIA_VIZAG },
};

const vesselClassName = (value: string): Vessel['vesselClass'] => {
  const normalized = value.toLowerCase();
  if (normalized.includes('cape')) return 'Capesize';
  if (normalized.includes('supra')) return 'Supramax';
  if (normalized.includes('handy')) return 'Handysize';
  return 'Panamax';
};

const routeMetaFor = (destination: string) => {
  const key = Object.keys(DESTINATION_META).find((name) => destination.toUpperCase().includes(name));
  return key ? DESTINATION_META[key] : DESTINATION_META.DHAMRA;
};

const INITIAL_VESSELS: Vessel[] = [
  {
    id: 'v1',
    code: '01',
    name: 'MV OCEAN STAR',
    vesselClass: 'Panamax',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'DHAMRA',
    destCoords: [20.80, 86.95],
    status: 'EN ROUTE',
    speed: '12.4 KN',
    eta: '18 SEP 2026',
    remainingNm: '2,840 NM',
    cargo: 'COKING COAL (74,500 MT)',
    charterStatus: 'AVAILABLE',
    progress: 0.65,
    currentHeading: 325,
    rate: '$16.90/t',
    feasibilityScore: '96.4%',
    waypoints: ROUTE_AUSTRALIA_DHAMRA,
  },
  {
    id: 'v2',
    code: '02',
    name: 'MV EASTERN WIND',
    vesselClass: 'Supramax',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'PARADIP',
    destCoords: [20.26, 86.67],
    status: 'EN ROUTE',
    speed: '11.8 KN',
    eta: '20 SEP 2026',
    remainingNm: '3,120 NM',
    cargo: 'THERMAL COAL (58,000 MT)',
    charterStatus: 'CHARTERED',
    progress: 0.42,
    currentHeading: 318,
    rate: '$17.80/t',
    feasibilityScore: '92.1%',
    waypoints: ROUTE_AUSTRALIA_PARADIP,
  },
  {
    id: 'v3',
    code: '03',
    name: 'MV IRON TRADER',
    vesselClass: 'Capesize',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'VIZAG',
    destCoords: [17.68, 83.21],
    status: 'AVAILABLE',
    speed: '13.1 KN',
    eta: '22 SEP 2026',
    remainingNm: '1,950 NM',
    cargo: 'IRON ORE (175,000 MT)',
    charterStatus: 'AVAILABLE',
    progress: 0.78,
    currentHeading: 330,
    rate: '$22.40/t',
    feasibilityScore: '98.0%',
    waypoints: ROUTE_AUSTRALIA_VIZAG,
  },
  {
    id: 'v4',
    code: '04',
    name: 'MV PACIFIC VOYAGER',
    vesselClass: 'Handysize',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'GANGAVARAM',
    destCoords: [17.63, 83.23],
    status: 'EN ROUTE',
    speed: '12.0 KN',
    eta: '24 SEP 2026',
    remainingNm: '3,480 NM',
    cargo: 'BAUXITE (38,000 MT)',
    charterStatus: 'AVAILABLE',
    progress: 0.28,
    currentHeading: 310,
    rate: '$18.50/t',
    feasibilityScore: '88.5%',
    waypoints: ROUTE_AUSTRALIA_GANGAVARAM,
  },
  {
    id: 'v5',
    code: '05',
    name: 'MV BENGAL PHOENIX',
    vesselClass: 'Panamax',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'GOPALPUR',
    destCoords: [19.30, 84.96],
    status: 'EN ROUTE',
    speed: '11.5 KN',
    eta: '25 SEP 2026',
    remainingNm: '2,640 NM',
    cargo: 'MET COAL (72,000 MT)',
    charterStatus: 'IN BIDDING',
    progress: 0.54,
    currentHeading: 320,
    rate: '$16.90/t',
    feasibilityScore: '94.2%',
    waypoints: ROUTE_AUSTRALIA_GOPALPUR,
  },
  {
    id: 'v6',
    code: '06',
    name: 'MV MARITIME PRIDE',
    vesselClass: 'Supramax',
    origin: 'AUSTRALIA',
    originCoords: [-20.31, 118.57],
    destination: 'HALDIA',
    destCoords: [22.02, 88.06],
    status: 'AT PORT',
    speed: '0.0 KN',
    eta: 'DISCHARGING',
    remainingNm: '0 NM',
    cargo: 'LIMESTONE (55,000 MT)',
    charterStatus: 'AVAILABLE',
    progress: 1.0,
    currentHeading: 0,
    rate: '$17.80/t',
    feasibilityScore: '90.7%',
    waypoints: ROUTE_AUSTRALIA_HALDIA,
  },
];

function interpolateRoute(waypoints: [number, number][], t: number): { lat: number; lng: number; heading: number } {
  if (waypoints.length === 0) return { lat: 0, lng: 0, heading: 0 };
  if (waypoints.length === 1 || t <= 0) return { lat: waypoints[0][0], lng: waypoints[0][1], heading: 0 };
  if (t >= 1) return { lat: waypoints[waypoints.length - 1][0], lng: waypoints[waypoints.length - 1][1], heading: 0 };

  const totalSegments = waypoints.length - 1;
  const scaledT = t * totalSegments;
  const segIndex = Math.min(Math.floor(scaledT), totalSegments - 1);
  const segProgress = scaledT - segIndex;

  const p1 = waypoints[segIndex];
  const p2 = waypoints[segIndex + 1];

  const lat = p1[0] + (p2[0] - p1[0]) * segProgress;
  const lng = p1[1] + (p2[1] - p1[1]) * segProgress;

  const dLat = p2[0] - p1[0];
  const dLng = p2[1] - p1[1];
  let angle = (Math.atan2(dLng, dLat) * 180) / Math.PI;
  if (angle < 0) angle += 360;

  return { lat, lng, heading: angle };
}

export default function VesselIntelligence({ onNavigate }: VesselIntelligenceProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersRef = useRef<Record<string, L.Marker>>({});
  const polylinesRef = useRef<Record<string, L.Polyline>>({});
  const [mapInstance, setMapInstance] = useState<L.Map | null>(null);
  const [vessels, setVessels] = useState<Vessel[]>(INITIAL_VESSELS);
  const [selectedVesselId, setSelectedVesselId] = useState<string>('v1');
  const [selectedClassFilter, setSelectedClassFilter] = useState<string>('ALL');
  const [activeTab, setActiveTab] = useState<'CLASSES' | 'FLEET'>('CLASSES');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1.0);
  const [showFeasibilityMatrix, setShowFeasibilityMatrix] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [aisConnected, setAisConnected] = useState<boolean>(false);
  const [feasibilityState, setFeasibilityState] = useState<EndpointState<VesselFeasibilityResponse>>(emptyEndpoint);
  const [fleetState, setFleetState] = useState<EndpointState<FleetStatusResponse>>(emptyEndpoint);

  const selectedVessel = vessels.find((v) => v.id === selectedVesselId) || vessels[0];

  useEffect(() => {
    const controller = new AbortController();

    setFeasibilityState((prev) => ({ ...prev, loading: true, error: null }));
    apiGet<VesselFeasibilityResponse>('/vessel-feasibility', {
      cargo_qty: 75000,
      origin: 'Australia',
      destination: 'Dhamra',
    }, controller.signal)
      .then((response) => setFeasibilityState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setFeasibilityState((prev) => ({ ...prev, loading: false, error: error.message || 'Feasibility unavailable' }));
        }
      });

    setFleetState((prev) => ({ ...prev, loading: true, error: null }));
    apiGet<FleetStatusResponse>('/fleet-status', {}, controller.signal)
      .then((response) => setFleetState({ data: response.data, status: response.status, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setFleetState((prev) => ({ ...prev, loading: false, error: error.message || 'Fleet status unavailable' }));
        }
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!fleetState.data?.fleet?.length) return;

    const mapped = fleetState.data.fleet.map((item, index) => {
      const vesselClass = vesselClassName(item.vessel_class);
      const route = routeMetaFor(item.destination);
      const status = item.status as Vessel['status'];
      const progress = status === 'AT PORT' || status === 'AVAILABLE' || status === 'IDLE'
        ? status === 'AVAILABLE' ? 0.08 : 1
        : Math.max(0.12, Math.min(0.9, 0.3 + index * 0.08));

      return {
        id: item.id,
        code: item.code,
        name: item.name,
        vesselClass,
        origin: item.origin.toUpperCase(),
        originCoords: [-20.31, 118.57] as [number, number],
        destination: item.destination.toUpperCase(),
        destCoords: route.coords,
        status,
        speed: `${item.speed_knots.toFixed(1)} KN`,
        eta: item.eta.toUpperCase(),
        remainingNm: `${item.remaining_nm.toLocaleString()} NM`,
        cargo: item.cargo.toUpperCase(),
        charterStatus: item.charter_status,
        progress,
        currentHeading: 320,
        rate: `$${(classRates[vesselClass] || 16.9).toFixed(2)}/t`,
        feasibilityScore: item.feasibility_score,
        waypoints: route.waypoints,
      };
    });

    setVessels(mapped);
    if (!mapped.some((v) => v.id === selectedVesselId)) {
      setSelectedVesselId(mapped[0].id);
    }
  }, [fleetState.data, selectedVesselId]);

  const feasibilityRows = (feasibilityState.data?.per_class_comparison || []).map((item) => {
    const name = vesselClassName(item.vessel_class);
    const recommended = feasibilityState.data?.recommended_vessel_class.toLowerCase().includes(name.toLowerCase()) || false;
    const feasibility = item.is_feasible ? 100 : 42;
    const rate = classRates[name] || 16.9;
    return {
      name,
      shortName: name === 'Handysize' ? 'Handy' : name === 'Supramax' ? 'Supra' : name === 'Capesize' ? 'Cape' : name,
      subtitle: classSubtitles[name] || `${item.typical_dwt.toLocaleString()} DWT`,
      capacity: `${item.typical_dwt.toLocaleString()} DWT`,
      feasibility,
      rate,
      label: `$${rate.toFixed(2)}/t`,
      status: recommended ? 'Recommended Class' : item.is_feasible ? 'Feasible' : 'Draft Constrained',
      vesselId: vessels.find((v) => v.vesselClass === name)?.id || vessels[0]?.id || 'v1',
      recommended,
      forecast: Number((feasibilityState.data?.forecast_delta_pct || '+0').replace(/[^0-9.-]/g, '')) || 0,
      restricted: !item.is_feasible,
    };
  });

  const classCards = feasibilityRows.length ? feasibilityRows : [
    { name: 'Handysize', shortName: 'Handy', rate: 18.50, label: '$18.50/t', vesselId: 'v4', subtitle: '28,000 - 38,000 DWT', capacity: '38,000 DWT', status: 'Optimal Draft', recommended: false, feasibility: 88.5, forecast: 8.8, restricted: false },
    { name: 'Supramax', shortName: 'Supra', rate: 17.80, label: '$17.80/t', vesselId: 'v2', subtitle: '50,000 - 60,000 DWT', capacity: '60,000 DWT', status: 'High Demand', recommended: false, feasibility: 92.1, forecast: 8.0, restricted: false },
    { name: 'Panamax', shortName: 'Panamax', rate: 16.90, label: '$16.90/t', vesselId: 'v1', subtitle: '65,000 - 85,000 DWT', capacity: '75,000 DWT', status: 'Recommended Class', recommended: true, feasibility: 96.4, forecast: 8.9, restricted: false },
    { name: 'Capesize', shortName: 'Cape', rate: 22.40, label: '$22.40/t', vesselId: 'v3', subtitle: '100,000+ DWT', capacity: '175,000 DWT', status: 'Draft Constrained', recommended: false, feasibility: 42.0, forecast: -9.8, restricted: true },
  ];

  const lowestRate = Math.min(...classCards.map((item) => item.rate));
  const recommendedClass = classCards.find((item) => item.recommended)?.name || 'Panamax';
  const fleetSummary = fleetState.data?.summary;
  const fleetTotal = fleetSummary?.total_fleet || vessels.length;
  const enRouteCount = fleetSummary?.active || vessels.filter((v) => v.status === 'EN ROUTE').length;
  const availableCount = fleetSummary?.available || vessels.filter((v) => v.status === 'AVAILABLE').length;
  const atPortCount = (fleetSummary?.at_port || 0) + (fleetSummary?.idle || 0) || vessels.filter((v) => v.status === 'AT PORT' || v.status === 'IDLE').length;
  const donutC = 238;
  const enRouteDash = (enRouteCount / Math.max(fleetTotal, 1)) * donutC;
  const availableDash = (availableCount / Math.max(fleetTotal, 1)) * donutC;
  const atPortDash = Math.max(0, donutC - enRouteDash - availableDash);
  const pctLabel = (count: number) => `${Math.round((count / Math.max(fleetTotal, 1)) * 100)}%`;

  // Initialize Light Cartography Map matching FICOS palette
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [5.0, 96.0],
      zoom: 4,
      minZoom: 3,
      maxZoom: 9,
      zoomControl: false,
      attributionControl: false,
    });

    // High precision dark maritime cartography tiles (Esri World Dark Gray Base - No Watermark / No API Key required)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 16,
        attribution: '&copy; Esri, HERE, Garmin, NGA, USGS',
      }
    ).addTo(map);

    mapInstanceRef.current = map;
    setMapInstance(map);

    setTimeout(() => {
      map.invalidateSize();
    }, 150);

    // Dark Port Waypoint Markers matching FICOS Dark Palette
    const ports = [
      { name: 'Australia (Port Hedland)', coords: [-20.31, 118.57] as [number, number], tag: '18.6% Risk' },
      { name: 'Dhamra Port', coords: [20.80, 86.95] as [number, number], tag: '98.6% Opt' },
      { name: 'Paradip Port', coords: [20.26, 86.67] as [number, number], tag: '11.5m' },
      { name: 'Vizag Port', coords: [17.68, 83.21] as [number, number], tag: '94.0%' },
      { name: 'Risk Australia', coords: [-12.50, 130.80] as [number, number], tag: '88.0% Risk' },
    ];

    ports.forEach((port) => {
      const icon = L.divIcon({
        className: 'custom-port-pin-dark !bg-transparent !border-none',
        html: `
          <div class="relative group cursor-pointer -translate-x-1/2 -translate-y-1/2">
            <div class="flex items-center gap-1.5 bg-[#0A1628]/95 backdrop-blur-md px-2.5 py-1 rounded-md border border-cyan-400/50 shadow-[0_4px_16px_rgba(0,229,255,0.2)] text-white">
              <div class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
              <span class="text-[11px] font-bold tracking-wider uppercase font-mono text-white">${port.name}</span>
              ${port.tag ? `<span class="text-[10px] text-cyan-300 font-mono ml-1 font-bold">${port.tag}</span>` : ''}
            </div>
            <div class="w-1.5 h-1.5 mx-auto bg-cyan-400 rounded-full mt-0.5 shadow-sm"></div>
          </div>
        `,
        iconSize: [150, 32],
        iconAnchor: [75, 16],
      });

      L.marker(port.coords, { icon }).addTo(map);
    });

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      setMapInstance(null);
      polylinesRef.current = {};
      markersRef.current = {};
    };
  }, []);

  // Update polylines and ship markers
  useEffect(() => {
    const map = mapInstance || mapInstanceRef.current;
    if (!map) return;

    vessels.forEach((vessel) => {
      const isSelected = vessel.id === selectedVesselId;
      const routeColor = isSelected ? '#00E5FF' : '#38BDF8';
      const routeWeight = isSelected ? 3.5 : 2;
      const routeDash = isSelected ? undefined : '5, 5';

      // 1. Draw / Update Polyline
      const existingPolyline = polylinesRef.current[vessel.id];
      if (!existingPolyline || !map.hasLayer(existingPolyline)) {
        if (existingPolyline) {
          try { map.removeLayer(existingPolyline); } catch (e) {}
        }
        const polyline = L.polyline(vessel.waypoints, {
          color: routeColor,
          weight: routeWeight,
          opacity: isSelected ? 1.0 : 0.65,
          dashArray: routeDash,
          lineCap: 'round',
          lineJoin: 'round',
        }).addTo(map);
        polylinesRef.current[vessel.id] = polyline;
      } else {
        existingPolyline.setLatLngs(vessel.waypoints);
        existingPolyline.setStyle({
          color: routeColor,
          weight: routeWeight,
          opacity: isSelected ? 1.0 : 0.65,
          dashArray: routeDash,
        });
      }

      // 2. Compute Position & Heading
      const currentPos = interpolateRoute(vessel.waypoints, vessel.progress);

      // 3. Draw / Update Compact Vessel Marker (26x12)
      const shipIcon = L.divIcon({
        className: 'custom-vessel-marker-dark !bg-transparent !border-none',
        html: `
          <div class="relative cursor-pointer transition-transform duration-300 group" style="transform: rotate(${currentPos.heading}deg);">
            <div class="relative flex items-center justify-center ${
              isSelected ? 'scale-110 filter drop-shadow-[0_0_10px_rgba(0,229,255,0.95)]' : 'opacity-85 filter drop-shadow-[0_0_4px_rgba(0,229,255,0.5)]'
            }">
              <svg width="26" height="12" viewBox="0 0 44 20" fill="none">
                <!-- Hull silhouette -->
                <path d="M 4 10 L 32 3 Q 42 6 42 10 Q 42 14 32 17 L 4 10 Z" fill="${isSelected ? '#0A1628' : '#142436'}" stroke="#00E5FF" stroke-width="1.8" />
                <!-- Deckhouse / Bridge -->
                <rect x="7" y="6" width="6" height="8" fill="#00E5FF" rx="1" />
                <rect x="9" y="8" width="2" height="4" fill="#E05A47" />
                <!-- Cargo holds -->
                <rect x="16" y="7" width="4" height="6" fill="#00E5FF" opacity="0.8" />
                <rect x="23" y="7" width="4" height="6" fill="#00E5FF" opacity="0.8" />
                <rect x="30" y="8" width="3" height="4" fill="#00E5FF" opacity="0.8" />
                <!-- Bow light -->
                <circle cx="40" cy="10" r="2" fill="#00E5FF" class="animate-pulse" />
              </svg>
            </div>
            ${
              isSelected
                ? `<div class="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap bg-[#0A1628]/95 border border-cyan-400 text-cyan-300 text-[9px] font-mono px-1.5 py-0.5 rounded shadow-md uppercase tracking-wider font-bold" style="transform: rotate(-${currentPos.heading}deg);">
                    ${vessel.name} (${vessel.speed})
                   </div>`
                : ''
            }
          </div>
        `,
        iconSize: [26, 12],
        iconAnchor: [13, 6],
      });

      const existingMarker = markersRef.current[vessel.id];
      if (!existingMarker || !map.hasLayer(existingMarker)) {
        if (existingMarker) {
          try { map.removeLayer(existingMarker); } catch (e) {}
        }
        const marker = L.marker([currentPos.lat, currentPos.lng], { icon: shipIcon }).addTo(map);
        marker.on('click', () => {
          setSelectedVesselId(vessel.id);
        });
        markersRef.current[vessel.id] = marker;
      } else {
        existingMarker.setLatLng([currentPos.lat, currentPos.lng]);
        existingMarker.setIcon(shipIcon);
      }
    });
  }, [mapInstance, vessels, selectedVesselId]);

  // Simulation Loop - realistic drifting speed
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setVessels((prevVessels) =>
        prevVessels.map((v) => {
          if (v.status === 'AT PORT') return v;
          const delta = 0.00004 * simSpeed;
          let newProgress = v.progress + delta;
          if (newProgress >= 1.0) {
            newProgress = 0.05;
          }
          return {
            ...v,
            progress: newProgress,
          };
        })
      );
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, simSpeed]);

  // Live AISStream Satellite / Terrestrial WebSocket Stream using .env APIKey
  useEffect(() => {
    const aisApiKey = import.meta.env.VITE_AISSTREAM_API_KEY;
    if (!aisApiKey) return;

    let socket: WebSocket | null = null;
    try {
      socket = new WebSocket('wss://stream.aisstream.io/v0/stream');

      socket.onopen = () => {
        setAisConnected(true);
        const subscriptionMessage = {
          APIKey: aisApiKey,
          BoundingBoxes: [
            [[-25.0, 70.0], [25.0, 125.0]]
          ]
        };
        socket?.send(JSON.stringify(subscriptionMessage));
      };

      socket.onmessage = (event) => {
        try {
          const aisMsg = JSON.parse(event.data);
          if (aisMsg.MessageType === 'PositionReport') {
            const report = aisMsg.Message.PositionReport;
            const liveSog = report.Sog || 12.4;

            // Live update telemetry from real satellite AIS
            setVessels((prevVessels) =>
              prevVessels.map((v, i) => {
                if (i === 0) {
                  return {
                    ...v,
                    speed: `${liveSog.toFixed(1)} KN`,
                  };
                }
                return v;
              })
            );
          }
        } catch (e) {
          // fallback to simulation loop
        }
      };

      socket.onerror = () => {
        setAisConnected(false);
      };
      socket.onclose = () => {
        setAisConnected(false);
      };
    } catch (e) {
      setAisConnected(false);
    }

    return () => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, []);

  const handleZoomIn = () => mapInstanceRef.current?.zoomIn();
  const handleZoomOut = () => mapInstanceRef.current?.zoomOut();
  const handleResetView = () => mapInstanceRef.current?.setView([5.0, 96.0], 4);

  const filteredVessels = vessels.filter((v) => {
    const matchesClass = selectedClassFilter === 'ALL' || v.vesselClass.toUpperCase() === selectedClassFilter.toUpperCase();
    const matchesQuery = searchQuery === '' || v.name.toLowerCase().includes(searchQuery.toLowerCase()) || v.destination.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesClass && matchesQuery;
  });

  return (
    <div
      className="min-h-screen text-[#1D3045] font-sans antialiased selection:bg-[#E05A47] selection:text-white pb-12 relative bg-cover bg-center bg-no-repeat bg-fixed flex flex-col"
      style={{
        backgroundImage: 'url(/dashboard-bg.png)',
        backgroundColor: '#E2E8F0',
      }}
    >
      
      {/* 1. TOP HEADER - Synchronized with Dashboard & Landing */}
      <header className="w-full px-6 sm:px-8 md:px-12 pt-7 sm:pt-10 pb-6 flex items-center justify-between z-50 shrink-0">
        
        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center gap-6 xl:gap-8" aria-label="Main Navigation">
          <button
            onClick={() => onNavigate('landing')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045] hover:opacity-70 transition-opacity"
          >
            FICOS
          </button>
          <button
            onClick={() => onNavigate('landing')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            FREIGHT INTELLIGENCE
          </button>
          <span className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]">
            VESSEL INTELLIGENCE
            <span className="absolute -bottom-3 left-0 right-0 h-[2px] w-full bg-[#1D3045]" />
          </span>
          <button
            onClick={() => onNavigate('idle-intelligence')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            IDLE INTELLIGENCE
          </button>
          <button
            onClick={() => onNavigate('dashboard')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            DASHBOARD
          </button>
          <button
            onClick={() => onNavigate('services')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            HOW IT WORKS
          </button>
          <button
            onClick={() => onNavigate('landing')}
            className="relative text-[11px] tracking-[0.16em] uppercase font-medium text-[#1D3045]/70 hover:opacity-100 transition-opacity"
          >
            ABOUT
          </button>
        </nav>

        {/* Mobile Brand */}
        <div className="lg:hidden flex items-center gap-4">
          <button
            onClick={() => onNavigate('landing')}
            className="text-base font-bold tracking-[0.15em] text-[#1D3045]"
          >
            FICOS
          </button>
          <span className="text-xs tracking-[0.15em] uppercase font-semibold text-[#1D3045] border-b-2 border-[#1D3045] pb-0.5">
            VESSEL INTELLIGENCE
          </span>
        </div>

        {/* Right Cluster */}
        <div className="flex items-center gap-6">
          <button
            onClick={() => onNavigate('landing')}
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

          <button
            onClick={() => onNavigate('landing')}
            className="hidden sm:inline-block text-xs tracking-[0.2em] uppercase font-medium focus:outline-none hover:opacity-80 text-[#1D3045]"
          >
            MENU
          </button>
        </div>
      </header>

      {/* 2. MAIN WORKSPACE CONTAINER */}
      <main className="w-full px-6 sm:px-8 md:px-12 space-y-5 flex-1 flex flex-col max-w-[1824px] mx-auto">
        
        {/* Top Heading & Fleet State Banner */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 shrink-0">
          <div>
            <h1 className="text-2xl sm:text-[2.55rem] font-light uppercase tracking-tight text-[#1D3045] drop-shadow-[0_1px_1px_rgba(255,255,255,0.8)]">
              VESSEL INTELLIGENCE
            </h1>
            <p className="mt-1.5 text-sm sm:text-base text-[#1D3045]/80 tracking-wide font-normal">
              Real-time dry bulk fleet tracking, route optimization, and port draft analytics.
            </p>
          </div>

          {/* FLEET STATE PILL matching Dashboard */}
          <div className="inline-flex items-center gap-3 px-4 py-2.5 rounded-xl bg-white border border-[#1D3045]/15 shadow-[0_4px_16px_rgba(29,48,69,0.09),inset_0_1px_0_rgba(255,255,255,1)]">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E05A47] shadow-[0_0_10px_rgba(224,90,71,0.7)] animate-pulse" />
              <span className="text-[11px] uppercase tracking-[0.2em] font-semibold text-[#1D3045]/70">
                FLEET STATUS:
              </span>
              <strong className="text-xs font-bold uppercase tracking-wider text-[#1D3045]">
                24 ACTIVE
              </strong>
            </div>
            <span className="text-xs text-[#1D3045]/30">•</span>
            <span className="text-xs text-[#1D3045]/85 font-medium">
              18 En Route · 4 Available · 2 Idle
            </span>
          </div>
        </div>

        {/* 3. FOUR METRIC SUMMARY CARDS - Synchronized with Dashboard cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-5 shrink-0">
          <div className="bg-white/90 backdrop-blur-md p-5 rounded-2xl border border-[#1D3045]/10 shadow-[0_4px_16px_rgba(29,48,69,0.05)]">
            <div className="flex items-center justify-between">
              <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#1D3045]/50">
                MONITORED FLEET
              </p>
              <span className="text-[10px] font-mono text-emerald-700 font-bold">100% ONLINE</span>
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-light text-[#1D3045]">24</span>
              <span className="text-xs text-[#1D3045]/60 font-medium">Vessels Total</span>
            </div>
            {/* Multi-segment fleet status bar */}
            <div className="mt-2.5 space-y-1">
              <div className="h-2 w-full bg-[#1D3045]/10 rounded-full overflow-hidden flex">
                <div className="h-full bg-[#1D3045]" style={{ width: '75.0%' }} title="18 En Route (75%)" />
                <div className="h-full bg-emerald-500" style={{ width: '16.7%' }} title="4 Available (16.7%)" />
                <div className="h-full bg-amber-500" style={{ width: '8.3%' }} title="2 Idle/Maint (8.3%)" />
              </div>
              <div className="flex items-center justify-between text-[9px] text-[#1D3045]/60 font-mono">
                <span>18 En Route</span>
                <span>4 Avail</span>
                <span>2 Maint</span>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-md p-5 rounded-2xl border border-[#1D3045]/10 shadow-[0_4px_16px_rgba(29,48,69,0.05)]">
            <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#1D3045]/50">
              EN ROUTE
            </p>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-light text-[#1D3045]">18</span>
              <span className="text-xs text-[#E05A47] font-semibold font-mono">12.2 KN avg</span>
            </div>
            <p className="mt-1 text-xs text-[#1D3045]/60">Bay of Bengal & Indian Ocean</p>
          </div>

          <div className="bg-white/90 backdrop-blur-md p-5 rounded-2xl border border-[#1D3045]/10 shadow-[0_4px_16px_rgba(29,48,69,0.05)]">
            <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#1D3045]/50">
              AVAILABLE CAPACITY
            </p>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-light text-[#1D3045]">4</span>
              <span className="text-xs text-emerald-600 font-medium">Prompt</span>
            </div>
            <p className="mt-1 text-xs text-[#1D3045]/60">Panamax & Supramax ready</p>
          </div>

          <div className="bg-white/90 backdrop-blur-md p-5 rounded-2xl border border-[#1D3045]/10 shadow-[0_4px_16px_rgba(29,48,69,0.05)]">
            <p className="text-[10px] tracking-[0.2em] uppercase font-semibold text-[#1D3045]/50">
              MAINTENANCE / IDLE
            </p>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-light text-[#1D3045]">2</span>
              <span className="text-xs text-amber-600 font-semibold font-mono">8.3% Fleet</span>
            </div>
            <p className="mt-1 text-xs text-[#1D3045]/60">Discharging at Haldia / Yard</p>
          </div>
        </div>

        {/* 4. MAIN OPERATIONAL WORKSPACE (Left Side Classes + Right Map) */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[580px]">
          
          {/* LEFT SIDEBAR: PORT FREIGHT CLASSES / FLEET (4 cols) */}
          <div className="lg:col-span-4 bg-white/90 backdrop-blur-md rounded-2xl border border-[#1D3045]/10 shadow-[0_4px_20px_rgba(29,48,69,0.06)] flex flex-col overflow-hidden">
            
            {/* Header with Class / Fleet toggle */}
            <div className="p-5 border-b border-[#1D3045]/10">
              <div className="flex items-center justify-between mb-1">
                <p className="text-[10px] tracking-[0.25em] uppercase font-semibold text-[#1D3045]/60">
                  PORT FREIGHT CLASSES
                </p>
                <span className="text-[10px] tracking-[0.15em] uppercase font-mono text-[#1D3045]/40">
                  LIVE 24
                </span>
              </div>

              <div className="flex items-center justify-between mt-1">
                <h2 className="text-xl font-light tracking-tight text-[#1D3045] flex items-center gap-1.5">
                  Decisions
                  <ChevronDown size={16} className="text-[#1D3045]/40" />
                </h2>

                <div className="flex items-center gap-1 bg-[#1D3045]/5 p-0.5 rounded-lg border border-[#1D3045]/10">
                  <button
                    onClick={() => setActiveTab('CLASSES')}
                    className={`px-3 py-1 text-[10px] uppercase tracking-wider font-semibold rounded-md transition-colors ${
                      activeTab === 'CLASSES' ? 'bg-[#1D3045] text-white' : 'text-[#1D3045]/70 hover:text-[#1D3045]'
                    }`}
                  >
                    Classes
                  </button>
                  <button
                    onClick={() => setActiveTab('FLEET')}
                    className={`px-3 py-1 text-[10px] uppercase tracking-wider font-semibold rounded-md transition-colors ${
                      activeTab === 'FLEET' ? 'bg-[#1D3045] text-white' : 'text-[#1D3045]/70 hover:text-[#1D3045]'
                    }`}
                  >
                    Fleet ({filteredVessels.length})
                  </button>
                </div>
              </div>

              {/* Search input */}
              <div className="mt-3 relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#1D3045]/40" />
                <input
                  type="text"
                  placeholder="Filter vessel, class, or destination..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-[#1D3045]/5 border border-[#1D3045]/10 rounded-lg pl-8 pr-3 py-1.5 text-xs text-[#1D3045] placeholder:text-[#1D3045]/40 focus:outline-none focus:border-[#1D3045]/30"
                />
              </div>
            </div>

            {/* List container */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {activeTab === 'CLASSES' ? (
                <>
                  <div className="space-y-3">
                    {classCards.map((item) => {
                      const isSelected = selectedVessel?.vesselClass === item.name;
                      return (
                        <div
                          key={item.name}
                          onClick={() => {
                            setSelectedVesselId(item.vesselId);
                            setSelectedClassFilter(item.name.toUpperCase());
                          }}
                          className={`p-4 rounded-xl border transition-all cursor-pointer relative ${
                            isSelected
                              ? 'bg-[#1D3045]/5 border-[#1D3045] shadow-sm ring-1 ring-[#1D3045]/20'
                              : 'bg-white/80 border-[#1D3045]/10 hover:border-[#1D3045]/20 hover:bg-white'
                          }`}
                        >
                          {item.recommended && (
                            <span className="absolute -top-2.5 right-3 bg-[#E05A47] text-white text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider shadow-sm flex items-center gap-1">
                              <Sparkles size={10} /> RECOMMENDED
                            </span>
                          )}
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="text-sm font-semibold text-[#1D3045] flex items-center gap-2">
                                {item.name}
                              </h3>
                              <p className="text-[10px] tracking-wider uppercase text-[#1D3045]/50">{item.subtitle}</p>
                            </div>
                            <div className="text-right">
                              <span className={`text-sm font-bold font-mono ${item.recommended ? 'text-[#E05A47]' : 'text-[#1D3045]'}`}>
                                {item.label}
                              </span>
                              <p className="text-[9px] uppercase tracking-wider text-[#1D3045]/60 font-medium">{item.status}</p>
                            </div>
                          </div>

                          <div className="mt-2 pt-1 border-t border-[#1D3045]/5">
                            <BulkCarrierGraphic vesselClass={item.name} active={isSelected} />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* 2. VESSEL CLASS COST COMPARISON - BAR CHART */}
                  <div className="mt-4 p-4 bg-white/95 rounded-xl border border-[#1D3045]/15 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider text-[#1D3045]">
                          VESSEL CLASS FREIGHT RATE ($/MT)
                        </h4>
                        <p className="text-[10px] text-[#1D3045]/60">
                          Backend feasibility plus class rate comparison
                        </p>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-[#E05A47] bg-[#E05A47]/10 px-2 py-0.5 rounded">
                        {recommendedClass.toUpperCase()} OPTIMAL
                      </span>
                    </div>

                    <div className="space-y-2.5 pt-1">
                      {classCards.map((item) => {
                        const maxRate = 25.0;
                        const pct = (item.rate / maxRate) * 100;
                        const isSelected = selectedVessel?.vesselClass === item.name;
                        return (
                          <div
                            key={item.name}
                            onClick={() => setSelectedClassFilter(item.name.toUpperCase())}
                            className="group cursor-pointer space-y-1"
                          >
                            <div className="flex items-center justify-between text-[11px] font-mono">
                              <span className={`font-semibold flex items-center gap-1.5 ${item.recommended ? 'text-[#E05A47]' : 'text-[#1D3045]'}`}>
                                {item.name}
                                {item.rate === lowestRate && (
                                  <span className="text-[9px] bg-[#E05A47]/15 text-[#E05A47] px-1.5 py-0.2 rounded font-mono font-bold">
                                    LOWEST $/T
                                  </span>
                                )}
                              </span>
                              <span className={`font-bold ${item.recommended ? 'text-[#E05A47]' : 'text-[#1D3045]'}`}>
                                {item.label}
                              </span>
                            </div>

                            <div className="h-4 bg-[#1D3045]/5 rounded-md p-0.5 overflow-hidden flex items-center border border-[#1D3045]/10">
                              <div
                                className={`h-full rounded-sm transition-all duration-500 flex items-center justify-end pr-2 text-[9px] font-mono font-bold text-white ${
                                  item.recommended
                                    ? 'bg-[#E05A47] shadow-[0_0_10px_rgba(224,90,71,0.4)]'
                                    : isSelected
                                    ? 'bg-[#1D3045]'
                                    : 'bg-[#486581]'
                                }`}
                                style={{ width: `${pct}%` }}
                              >
                                {item.rate.toFixed(2)}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <p className="text-[9px] text-[#1D3045]/50 italic text-right pt-1">
                      *Rates are displayed against backend feasibility constraints for the Dhamra cargo move.
                    </p>
                  </div>
                </>
              ) : (
                <div className="space-y-4">
                  {/* 4. FLEET STATUS BREAKDOWN - DONUT CHART */}
                  <div className="p-4 bg-white/95 rounded-xl border border-[#1D3045]/15 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-[#1D3045]">
                        FLEET COMPOSITION & STATUS
                      </h4>
                      <span className="text-[10px] font-mono text-[#1D3045]/60 font-semibold">{fleetTotal} TOTAL</span>
                    </div>

                    <div className="flex items-center gap-4">
                      {/* Donut SVG */}
                      <div className="relative w-28 h-28 shrink-0">
                        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                          {/* Segment 1: En Route (75% = 18 vessels) */}
                          <circle
                            cx="50"
                            cy="50"
                            r="38"
                            fill="transparent"
                            stroke="#1D3045"
                            strokeWidth="14"
                            strokeDasharray={`${enRouteDash} ${donutC}`}
                            strokeDashoffset="0"
                          />
                          {/* Segment 2: Available (16.7% = 4 vessels) */}
                          <circle
                            cx="50"
                            cy="50"
                            r="38"
                            fill="transparent"
                            stroke="#10B981"
                            strokeWidth="14"
                            strokeDasharray={`${availableDash} ${donutC}`}
                            strokeDashoffset={-enRouteDash}
                          />
                          {/* Segment 3: Maintenance (8.3% = 2 vessels) */}
                          <circle
                            cx="50"
                            cy="50"
                            r="38"
                            fill="transparent"
                            stroke="#F59E0B"
                            strokeWidth="14"
                            strokeDasharray={`${atPortDash} ${donutC}`}
                            strokeDashoffset={-(enRouteDash + availableDash)}
                          />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
                          <span className="text-lg font-bold text-[#1D3045] leading-none">{fleetTotal}</span>
                          <span className="text-[8px] uppercase tracking-wider text-[#1D3045]/60 font-semibold mt-0.5">
                            FLEET
                          </span>
                        </div>
                      </div>

                      {/* Donut Legend */}
                      <div className="space-y-2 flex-1 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium text-[#1D3045]">
                            <span className="w-2.5 h-2.5 rounded-sm bg-[#1D3045]" />
                            En Route
                          </span>
                          <span className="font-mono font-bold text-[#1D3045]">{enRouteCount} ({pctLabel(enRouteCount)})</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium text-[#1D3045]">
                            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                            Prompt Available
                          </span>
                          <span className="font-mono font-bold text-emerald-700">{availableCount} ({pctLabel(availableCount)})</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium text-[#1D3045]">
                            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
                            Maintenance / Idle
                          </span>
                          <span className="font-mono font-bold text-amber-700">{atPortCount} ({pctLabel(atPortCount)})</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {filteredVessels.map((vessel) => {
                      const isSelected = vessel.id === selectedVesselId;
                      return (
                        <div
                          key={vessel.id}
                          onClick={() => setSelectedVesselId(vessel.id)}
                          className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                            isSelected
                              ? 'bg-[#1D3045]/5 border-[#1D3045] shadow-sm'
                              : 'bg-white/80 border-[#1D3045]/10 hover:border-[#1D3045]/20 hover:bg-white'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] font-mono tracking-widest text-[#1D3045]/50">
                              {vessel.code}
                            </span>
                            <span
                              className={`text-[9px] px-2 py-0.5 rounded-full font-mono uppercase tracking-wider font-semibold ${
                                vessel.status === 'EN ROUTE'
                                  ? 'bg-[#1D3045]/10 text-[#1D3045]'
                                  : vessel.status === 'AVAILABLE'
                                  ? 'bg-emerald-100 text-emerald-800'
                                  : 'bg-amber-100 text-amber-800'
                              }`}
                            >
                              {vessel.status}
                            </span>
                          </div>

                          <h4 className="text-sm font-semibold text-[#1D3045] mt-1">
                            {vessel.name}
                          </h4>

                          <div className="flex items-center justify-between mt-1 text-[11px] text-[#1D3045]/70">
                            <span className="uppercase tracking-wider font-mono text-[10px] text-[#1D3045]/60">
                              {vessel.vesselClass}
                            </span>
                            <span className="font-mono text-[11px] text-[#1D3045] font-medium">
                              {vessel.origin} → {vessel.destination}
                            </span>
                          </div>

                          <div className="mt-2 w-full bg-[#1D3045]/10 h-1.5 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#1D3045] transition-all duration-300"
                              style={{ width: `${Math.round(vessel.progress * 100)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar Footer */}
            <div className="p-4 border-t border-[#1D3045]/10 bg-white/60 flex items-center justify-between text-[11px] text-[#1D3045]/60">
              <span className="flex items-center gap-1.5 font-medium">
                <Ship size={13} className="text-[#1D3045]" />
                AIS FLEET ENGINE
              </span>
              <span className="font-mono text-[10px]">v2.4 REALTIME</span>
            </div>
          </div>

          {/* RIGHT MAP DISPLAY & FLOATING PANELS (8 cols) */}
          <div className="lg:col-span-8 bg-white rounded-2xl border border-[#1D3045]/15 shadow-[0_4px_24px_rgba(29,48,69,0.08)] relative overflow-hidden flex flex-col">
            
            {/* Map Canvas */}
            <div ref={mapContainerRef} className="w-full h-full min-h-[500px] z-0 flex-1" />

            {/* TOP MAP CONTROLS OVERLAY */}
            <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none z-10">
              <div className="flex items-center gap-2 pointer-events-auto">
                {/* Class selector */}
                <div className="flex items-center gap-1 bg-white/95 backdrop-blur-md border border-[#1D3045]/15 px-3 py-1.5 rounded-lg shadow-sm text-xs text-[#1D3045]">
                  <Filter size={12} className="text-[#1D3045]/60" />
                  <span className="text-[#1D3045]/60 text-[10px] uppercase tracking-wider font-semibold">CLASS:</span>
                  <select
                    value={selectedClassFilter}
                    onChange={(e) => setSelectedClassFilter(e.target.value)}
                    className="bg-transparent text-[#1D3045] font-semibold text-xs focus:outline-none cursor-pointer"
                  >
                    <option value="ALL">ALL CLASSES</option>
                    <option value="HANDYSIZE">HANDYSIZE</option>
                    <option value="SUPRAMAX">SUPRAMAX</option>
                    <option value="PANAMAX">PANAMAX</option>
                    <option value="CAPESIZE">CAPESIZE</option>
                  </select>
                </div>

                <button
                  onClick={() => setShowFeasibilityMatrix(!showFeasibilityMatrix)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border backdrop-blur-md shadow-sm transition-colors ${
                    showFeasibilityMatrix
                      ? 'bg-[#1D3045] text-white border-[#1D3045]'
                      : 'bg-white/95 text-[#1D3045] border-[#1D3045]/15 hover:bg-white'
                  }`}
                >
                  <Table2 size={12} />
                  <span className="uppercase tracking-wider text-[10px] font-semibold">Decisions Matrix</span>
                </button>
              </div>

              {/* Right Zoom Controls */}
              <div className="flex items-center gap-2 pointer-events-auto">
                <div className="flex flex-col bg-white/95 backdrop-blur-md border border-[#1D3045]/15 rounded-lg shadow-sm overflow-hidden">
                  <button
                    onClick={handleZoomIn}
                    title="Zoom In"
                    className="w-7 h-7 flex items-center justify-center text-[#1D3045] hover:bg-[#1D3045]/5 transition-colors border-b border-[#1D3045]/10"
                  >
                    <Plus size={13} />
                  </button>
                  <button
                    onClick={handleZoomOut}
                    title="Zoom Out"
                    className="w-7 h-7 flex items-center justify-center text-[#1D3045] hover:bg-[#1D3045]/5 transition-colors"
                  >
                    <Minus size={13} />
                  </button>
                </div>

                <button
                  onClick={handleResetView}
                  title="Reset Map Orientation"
                  className="w-7 h-7 bg-white/95 backdrop-blur-md border border-[#1D3045]/15 rounded-lg flex items-center justify-center text-[#1D3045] hover:bg-[#1D3045]/5 transition-colors shadow-sm"
                >
                  <Compass size={14} />
                </button>
              </div>
            </div>

            {/* 3. FLOATING FEASIBILITY MATRIX & CHART MODAL */}
            {showFeasibilityMatrix && (
              <div className="absolute top-16 left-4 z-10 w-[440px] max-w-[calc(100vw-2rem)] bg-white/95 backdrop-blur-lg border border-[#1D3045]/20 rounded-xl p-4 shadow-2xl transition-all pointer-events-auto space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-[#1D3045]/10">
                  <div>
                    <span className="text-[10px] tracking-[0.2em] uppercase font-bold text-[#1D3045] block">
                      Feasibility & Forecast Matrix
                    </span>
                    <span className="text-[10px] text-[#1D3045]/60">
                      Draft clearance & 30-day forecast delta per class
                    </span>
                  </div>
                  <button
                    onClick={() => setShowFeasibilityMatrix(false)}
                    className="text-[#1D3045]/40 hover:text-[#1D3045] p-1"
                  >
                    <X size={14} />
                  </button>
                </div>

                {/* Table */}
                <table className="hidden">
                  <thead>
                    <tr className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-mono border-b border-[#1D3045]/10">
                      <th className="pb-1.5 font-semibold">Class</th>
                      <th className="pb-1.5 font-semibold">Capacity</th>
                      <th className="pb-1.5 font-semibold">Feasibility</th>
                      <th className="pb-1.5 font-semibold text-right">Forecast</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1D3045]/5 text-xs">
                    <tr className="hover:bg-[#1D3045]/5 cursor-pointer" onClick={() => setSelectedClassFilter('HANDYSIZE')}>
                      <td className="py-1.5 font-medium text-[#1D3045]">Handysize</td>
                      <td className="py-1.5 text-[#1D3045]/70 font-mono text-[11px]">38,000 DWT</td>
                      <td className="py-1.5 font-mono text-[11px] font-semibold text-emerald-700">88.5%</td>
                      <td className="py-1.5 text-right font-mono font-semibold text-[#E05A47]">+8.8%</td>
                    </tr>
                    <tr className="hover:bg-[#1D3045]/5 cursor-pointer" onClick={() => setSelectedClassFilter('SUPRAMAX')}>
                      <td className="py-1.5 font-medium text-[#1D3045]">Supramax</td>
                      <td className="py-1.5 text-[#1D3045]/70 font-mono text-[11px]">60,000 DWT</td>
                      <td className="py-1.5 font-mono text-[11px] font-semibold text-emerald-700">92.1%</td>
                      <td className="py-1.5 text-right font-mono font-semibold text-[#E05A47]">+8.0%</td>
                    </tr>
                    <tr className="hover:bg-[#1D3045]/5 cursor-pointer" onClick={() => setSelectedClassFilter('PANAMAX')}>
                      <td className="py-1.5 font-medium text-[#1D3045] flex items-center gap-1 font-bold text-[#E05A47]">
                        Panamax ★
                      </td>
                      <td className="py-1.5 text-[#1D3045]/70 font-mono text-[11px]">75,000 DWT</td>
                      <td className="py-1.5 font-mono text-[11px] font-bold text-[#E05A47]">96.4%</td>
                      <td className="py-1.5 text-right font-mono font-bold text-[#E05A47]">+8.9%</td>
                    </tr>
                    <tr className="hover:bg-[#1D3045]/5 cursor-pointer bg-red-50/50" onClick={() => setSelectedClassFilter('CAPESIZE')}>
                      <td className="py-1.5 font-medium text-red-900 flex items-center gap-1">
                        Capesize
                        <AlertTriangle size={11} className="text-[#E05A47]" />
                      </td>
                      <td className="py-1.5 text-[#1D3045]/70 font-mono text-[11px]">175,000 DWT</td>
                      <td className="py-1.5 font-mono text-[11px] font-semibold text-amber-700">42.0%</td>
                      <td className="py-1.5 text-right font-mono font-bold text-[#E05A47]">-9.8%</td>
                    </tr>
                  </tbody>
                </table>

                <table className="w-full text-left">
                  <thead>
                    <tr className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-mono border-b border-[#1D3045]/10">
                      <th className="pb-1.5 font-semibold">Class</th>
                      <th className="pb-1.5 font-semibold">Capacity</th>
                      <th className="pb-1.5 font-semibold">Feasibility</th>
                      <th className="pb-1.5 font-semibold text-right">Forecast</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1D3045]/5 text-xs">
                    {classCards.map((item) => (
                      <tr
                        key={item.name}
                        className={`hover:bg-[#1D3045]/5 cursor-pointer ${item.restricted ? 'bg-red-50/50' : ''}`}
                        onClick={() => setSelectedClassFilter(item.name.toUpperCase())}
                      >
                        <td className={`py-1.5 font-medium flex items-center gap-1 ${item.recommended ? 'text-[#E05A47] font-bold' : item.restricted ? 'text-red-900' : 'text-[#1D3045]'}`}>
                          {item.name} {item.recommended ? '*' : ''}
                          {item.restricted && <AlertTriangle size={11} className="text-[#E05A47]" />}
                        </td>
                        <td className="py-1.5 text-[#1D3045]/70 font-mono text-[11px]">{item.capacity}</td>
                        <td className={`py-1.5 font-mono text-[11px] font-semibold ${item.recommended ? 'text-[#E05A47] font-bold' : item.restricted ? 'text-amber-700' : 'text-emerald-700'}`}>
                          {item.feasibility.toFixed(1)}%
                        </td>
                        <td className="py-1.5 text-right font-mono font-semibold text-[#E05A47]">
                          {item.forecast >= 0 ? `+${item.forecast.toFixed(1)}%` : `${item.forecast.toFixed(1)}%`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* VISUAL CHART VERSION: GROUPED BAR / DOT-PLOT CHART */}
                <div className="pt-2 border-t border-[#1D3045]/10 space-y-2">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="font-bold uppercase tracking-wider text-[#1D3045]">
                      FEASIBILITY VS FORECAST VISUAL MATRIX
                    </span>
                    <div className="flex items-center gap-3 font-mono text-[9px]">
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-sm bg-[#1D3045]" /> Feasibility %
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-[#E05A47]" /> Forecast %
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2 pt-1 bg-[#1D3045]/5 p-3 rounded-lg border border-[#1D3045]/10">
                    {classCards.map((item) => (
                      <div key={item.name} className="space-y-1">
                        <div className="flex items-center justify-between text-[10px] font-mono">
                          <span className={`font-semibold ${item.recommended ? 'text-[#E05A47] font-bold' : item.restricted ? 'text-red-700 font-bold' : 'text-[#1D3045]'}`}>
                            {item.shortName} {item.recommended ? '(Opt)' : item.restricted ? '(Restricted)' : ''}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-[#1D3045]/70">{item.feasibility.toFixed(1)}% feas</span>
                            <span className={`font-bold ${item.forecast >= 0 ? 'text-emerald-700' : 'text-[#E05A47]'}`}>
                              {item.forecast >= 0 ? `+${item.forecast}%` : `${item.forecast}%`}
                            </span>
                          </div>
                        </div>

                        {/* Dual Track Visual */}
                        <div className="relative h-4 bg-white rounded flex items-center px-1 border border-[#1D3045]/10 overflow-hidden">
                          {/* Feasibility Bar */}
                          <div
                            className={`h-2.5 rounded-sm transition-all duration-500 ${
                              item.recommended ? 'bg-[#E05A47]' : item.restricted ? 'bg-amber-500' : 'bg-[#1D3045]'
                            }`}
                            style={{ width: `${item.feasibility}%` }}
                          />

                          {/* Forecast Indicator Dot */}
                          <div
                            className={`absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full border-2 border-white shadow-sm flex items-center justify-center ${
                              item.forecast >= 0 ? 'bg-emerald-600' : 'bg-[#E05A47] animate-pulse'
                            }`}
                            style={{ left: `${Math.max(8, Math.min(92, item.feasibility))}%` }}
                            title={`Forecast Delta: ${item.forecast}%`}
                          >
                            <span className="w-1 h-1 bg-white rounded-full" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <p className="text-[9px] text-[#1D3045]/60 font-mono text-center">
                    Backend feasibility matrix for {feasibilityState.data?.destination_port || 'Dhamra'} cargo constraints.
                  </p>
                </div>
              </div>
            )}

            {/* FLOATING VESSEL DETAIL INSPECTION CARD */}
            {selectedVessel && (
              <div className="absolute top-16 right-4 z-10 w-80 bg-white/95 backdrop-blur-xl border border-[#1D3045]/20 rounded-2xl p-4 shadow-xl pointer-events-auto">
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-mono tracking-widest text-[#E05A47] uppercase font-semibold">
                      {selectedVessel.vesselClass} · {selectedVessel.status}
                    </span>
                    <h3 className="text-base font-bold text-[#1D3045] tracking-wide mt-0.5">
                      {selectedVessel.name}
                    </h3>
                  </div>
                  <div className="w-7 h-7 rounded-full bg-[#1D3045]/10 border border-[#1D3045]/20 flex items-center justify-center text-[#1D3045]">
                    <Anchor size={14} />
                  </div>
                </div>

                <div className="my-3 p-3 rounded-xl bg-[#1D3045]/5 border border-[#1D3045]/10 flex items-center justify-between">
                  <div>
                    <span className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-semibold block">ORIGIN</span>
                    <span className="text-xs font-bold text-[#1D3045]">{selectedVessel.origin}</span>
                  </div>
                  <div className="flex flex-col items-center px-2">
                    <ArrowRight size={14} className="text-[#E05A47]" />
                    <span className="text-[9px] font-mono text-[#1D3045]/60">{selectedVessel.remainingNm}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-semibold block">DESTINATION</span>
                    <span className="text-xs font-bold text-[#1D3045]">{selectedVessel.destination}</span>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center py-2 border-t border-b border-[#1D3045]/10 font-mono">
                  <div>
                    <span className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-semibold block">SPEED</span>
                    <span className="text-xs text-[#1D3045] font-bold">{selectedVessel.speed}</span>
                  </div>
                  <div>
                    <span className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-semibold block">ETA</span>
                    <span className="text-xs text-[#1D3045] font-bold">{selectedVessel.eta}</span>
                  </div>
                  <div>
                    <span className="text-[9px] uppercase tracking-wider text-[#1D3045]/50 font-semibold block">FEASIBILITY</span>
                    <span className="text-xs text-emerald-700 font-bold">{selectedVessel.feasibilityScore}</span>
                  </div>
                </div>

                <div className="mt-2.5 space-y-1 text-xs">
                  <div className="flex items-center justify-between text-[#1D3045]/80">
                    <span className="text-[10px] uppercase tracking-wider text-[#1D3045]/50 font-semibold">CARGO</span>
                    <span className="font-medium text-[#1D3045]">{selectedVessel.cargo}</span>
                  </div>
                  <div className="flex items-center justify-between text-[#1D3045]/80">
                    <span className="text-[10px] uppercase tracking-wider text-[#1D3045]/50 font-semibold">CHARTER STATUS</span>
                    <span className="font-mono text-[11px] text-emerald-700 font-bold">{selectedVessel.charterStatus}</span>
                  </div>
                  <div className="flex items-center justify-between text-[#1D3045]/80">
                    <span className="text-[10px] uppercase tracking-wider text-[#1D3045]/50 font-semibold">ESTIMATED FREIGHT</span>
                    <span className="font-mono text-[11px] text-[#1D3045] font-bold">{selectedVessel.rate}</span>
                  </div>
                </div>

                <button
                  onClick={() => onNavigate('dashboard')}
                  className="w-full mt-3 py-2 rounded-xl bg-[#1D3045] hover:bg-[#253d58] text-xs uppercase tracking-[0.15em] font-semibold text-white flex items-center justify-center gap-1.5 transition-colors shadow-sm"
                >
                  OPTIMIZE IN DASHBOARD
                  <ArrowUpRight size={13} />
                </button>
              </div>
            )}

            {/* BOTTOM PLAYBACK CONTROLS & STATUS STRIP */}
            <div className="h-10 bg-white/95 backdrop-blur-md border-t border-[#1D3045]/10 px-4 flex items-center justify-between text-[11px] font-mono z-10 shrink-0">
              <div className="flex items-center gap-4 text-[#1D3045]/70">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  title={isPlaying ? 'Pause Simulation' : 'Play Live Motion'}
                  className="w-6 h-6 rounded bg-[#1D3045]/5 hover:bg-[#1D3045]/10 flex items-center justify-center text-[#1D3045] transition-colors"
                >
                  {isPlaying ? <Pause size={11} /> : <Play size={11} />}
                </button>
                <button
                  onClick={() => setSimSpeed((prev) => (prev === 1.0 ? 2.0 : prev === 2.0 ? 5.0 : 1.0))}
                  className="font-bold text-[#1D3045] hover:underline"
                >
                  {simSpeed.toFixed(1)}x
                </button>
                <span className="hidden sm:inline text-[#1D3045]/40">•</span>
                <span className="hidden sm:inline font-sans text-xs font-semibold text-[#1D3045]">
                  LIVE FLEET SIMULATION
                </span>
              </div>

              <div className="flex items-center gap-3 text-[#1D3045]/60 text-[10px]">
                <span className="flex items-center gap-1.5 font-semibold">
                  <span className={`w-1.5 h-1.5 rounded-full ${aisConnected ? 'bg-emerald-500' : 'bg-[#E05A47]'} animate-pulse`} />
                  {aisConnected ? 'LIVE AISSTREAM (CONNECTED)' : 'AISSTREAM READY'}
                </span>
                <span className="hidden md:inline">LAST UPDATED 12:42 UTC</span>
              </div>
            </div>

          </div>
        </div>

      </main>
    </div>
  );
}
