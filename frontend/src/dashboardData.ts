// Centralized Mock Data Configuration for Freight Intelligence Dashboard
// All hardcoded numbers and forecast trajectories are defined here to enable 1-line backend API integration.

export interface ForecastPoint {
  x: number;
  y: number;
  rate: number;
}

export interface TimeframeForecast {
  historicalPath: string;
  p10Path: string; // Upper bound (P10 percentile - high rate scenario)
  p50Path: string; // Median forecast (P50 percentile)
  p90Path: string; // Lower bound (P90 percentile - low rate scenario)
  bandPath: string; // Polygon path between P10 and P90 for shading
  p10Value: string;
  p50Value: string;
  p90Value: string;
  p10Y: number; // Exact Y endpoint coordinate of P10 line at X=600
  p50Y: number; // Exact Y endpoint coordinate of P50 line at X=600
  p90Y: number; // Exact Y endpoint coordinate of P90 line at X=600
  currentRate: string;
}

export type Timeframe = '1D' | '7D' | '14D' | '30D' | '60D' | '90D';

export const MOCK_INPUTS = {
  cargo: 'Coking Coal',
  quantity: '75,000 MT',
  origin: 'Australia',
  destination: 'Dhamra',
  windowPeriod: 'Sep 2026',
};

export const MOCK_MARKET_SUMMARY = {
  currentRate: '$14.85',
  currentRateUnit: '/ MT',
  route: 'Hay Point → Dhamra route',
  forecastChange: '↑ 10.2%',
  forecastDirection: 'Rising',
  forecastSubtext: 'Rising rate momentum',
  expectedCost: '$418,775',
  expectedCostBasis: '75,000 MT Panamax basis',
  marketState: 'RISING',
  marketStateText: 'Freight momentum remains upward.',
};

// Tightened Y domain: $14.00 (Y=240) to $19.50 (Y=40)
export const MOCK_FORECAST_DATA: Record<Timeframe, TimeframeForecast> = {
  '1D': {
    historicalPath: 'M 40 205 L 120 192 L 200 180 L 280 172 L 360 162',
    p10Path: 'M 360 162 L 440 132 L 520 118 L 600 105',
    p50Path: 'M 360 162 L 440 154 L 520 148 L 600 142',
    p90Path: 'M 360 162 L 440 170 L 520 168 L 600 166',
    bandPath: 'M 360 162 L 440 132 L 520 118 L 600 105 L 600 166 L 520 168 L 440 170 Z',
    p10Value: '$16.20',
    p50Value: '$15.05',
    p90Value: '$14.40',
    p10Y: 105,
    p50Y: 142,
    p90Y: 166,
    currentRate: '$14.85',
  },
  '7D': {
    historicalPath: 'M 40 210 L 120 195 L 200 178 L 280 182 L 360 162',
    p10Path: 'M 360 162 L 440 135 L 520 120 L 600 110',
    p50Path: 'M 360 162 L 440 152 L 520 142 L 600 135',
    p90Path: 'M 360 162 L 440 168 L 520 164 L 600 160',
    bandPath: 'M 360 162 L 440 135 L 520 120 L 600 110 L 600 160 L 520 164 L 440 168 Z',
    p10Value: '$16.10',
    p50Value: '$15.20',
    p90Value: '$14.50',
    p10Y: 110,
    p50Y: 135,
    p90Y: 160,
    currentRate: '$14.85',
  },
  '14D': {
    historicalPath: 'M 40 220 L 120 200 L 200 182 L 280 178 L 360 162',
    p10Path: 'M 360 162 L 440 125 L 520 102 L 600 88',
    p50Path: 'M 360 162 L 440 142 L 520 128 L 600 118',
    p90Path: 'M 360 162 L 440 165 L 520 158 L 600 152',
    bandPath: 'M 360 162 L 440 125 L 520 102 L 600 88 L 600 152 L 520 158 L 440 165 Z',
    p10Value: '$16.80',
    p50Value: '$15.75',
    p90Value: '$14.70',
    p10Y: 88,
    p50Y: 118,
    p90Y: 152,
    currentRate: '$14.85',
  },
  '30D': {
    historicalPath: 'M 40 230 L 110 210 L 180 190 L 250 180 L 320 170 L 360 162',
    p10Path: 'M 360 162 L 430 115 L 500 90 L 570 74 L 600 64',
    p50Path: 'M 360 162 L 430 138 L 500 120 L 570 106 L 600 96',
    p90Path: 'M 360 162 L 430 162 L 500 152 L 570 144 L 600 138',
    bandPath: 'M 360 162 L 430 115 L 500 90 L 570 74 L 600 64 L 600 138 L 570 144 L 500 152 L 430 162 Z',
    p10Value: '$17.40',
    p50Value: '$16.36',
    p90Value: '$15.10',
    p10Y: 64,
    p50Y: 96,
    p90Y: 138,
    currentRate: '$14.85',
  },
  '60D': {
    historicalPath: 'M 40 238 L 120 220 L 200 200 L 280 182 L 360 162',
    p10Path: 'M 360 162 L 440 102 L 520 70 L 600 48',
    p50Path: 'M 360 162 L 440 128 L 520 100 L 600 82',
    p90Path: 'M 360 162 L 440 158 L 520 142 L 600 130',
    bandPath: 'M 360 162 L 440 102 L 520 70 L 600 48 L 600 130 L 520 142 L 440 158 Z',
    p10Value: '$18.20',
    p50Value: '$16.90',
    p90Value: '$15.40',
    p10Y: 48,
    p50Y: 82,
    p90Y: 130,
    currentRate: '$14.85',
  },
  '90D': {
    historicalPath: 'M 40 245 L 120 228 L 200 208 L 280 185 L 360 162',
    p10Path: 'M 360 162 L 440 88 L 520 52 L 600 30',
    p50Path: 'M 360 162 L 440 120 L 520 88 L 600 68',
    p90Path: 'M 360 162 L 440 152 L 520 134 L 600 122',
    bandPath: 'M 360 162 L 440 88 L 520 52 L 600 30 L 600 122 L 520 134 L 440 152 Z',
    p10Value: '$19.10',
    p50Value: '$17.50',
    p90Value: '$15.80',
    p10Y: 30,
    p50Y: 68,
    p90Y: 122,
    currentRate: '$14.85',
  },
};

export const MOCK_HORIZON_COMPARISON = [
  { horizon: '7D', p10: 16.10, p50: 15.20, p90: 14.50, p10Height: 65, p50Height: 52, p90Height: 42, date: 'Sep 16', spread: '$1.60', note: 'Immediate spot window; lowest variance.' },
  { horizon: '14D', p10: 16.80, p50: 15.75, p90: 14.70, p10Height: 74, p50Height: 60, p90Height: 45, date: 'Sep 23', spread: '$2.10', note: 'Recommended charter window before escalation.' },
  { horizon: '30D', p10: 17.40, p50: 16.36, p90: 15.10, p10Height: 82, p50Height: 68, p90Height: 50, date: 'Oct 09', spread: '$2.30', note: 'Target evaluation window; +10.2% median growth.' },
  { horizon: '60D', p10: 18.20, p50: 16.90, p90: 15.40, p10Height: 92, p50Height: 76, p90Height: 54, date: 'Nov 08', spread: '$2.80', note: 'Q4 seasonal demand buildup pushing upper tail.' },
  { horizon: '90D', p10: 19.10, p50: 17.50, p90: 15.80, p10Height: 104, p50Height: 84, p90Height: 60, date: 'Dec 08', spread: '$3.30', note: 'Long-range horizon with maximum confidence cone width.' },
];

export const MOCK_RECOMMENDATION = {
  decision: 'CHARTER NOW',
  badge: 'POLICY DECISION',
  subheading: 'HIGH CONVICTION RECOMMENDATION',
  rationale: 'Lock Panamax spot vessel before anticipated 30D rate surge (+10.2%).',
  vesselClass: 'Panamax (74,000 DWT)',
  targetPort: 'Dhamra (Berth #1)',
  optimalWindow: 'Next 7–14 Days',
  potentialSavings: '$42,500 vs waiting',
};

export const MOCK_RISK_BREAKDOWN = {
  overallLevel: 'MEDIUM',
  score: 48,
  caption: 'Weather & port draft limits dominant',
  driverSummary: 'Weather constraints at Hay Point coupled with tight draft windows at Dhamra represent 65% of total move risk. Geopolitical risk on the Australia-India corridor remains subdued.',
  categories: [
    {
      name: 'Weather & Sea State',
      percent: 35,
      color: '#1D3045',
      impact: 'HIGH',
      description: 'High swell at Hay Point loading terminal and early monsoon surge in Bay of Bengal.',
    },
    {
      name: 'Port Congestion & Draft',
      percent: 30,
      color: '#E05A47',
      impact: 'MEDIUM-HIGH',
      description: '14.5m max draft limitation at Dhamra Berth #1 requires strict tide window synchronization.',
    },
    {
      name: 'Geopolitical (GPR)',
      percent: 20,
      color: '#486581',
      impact: 'MEDIUM',
      description: 'Indirect bunker surcharge inflation due to Red Sea rerouting affecting Pacific fleet fuel costs.',
    },
    {
      name: 'Market Supply Volatility',
      percent: 15,
      color: '#829AB1',
      impact: 'LOW-MEDIUM',
      description: 'Panamax spot availability in Eastern Australia tightening slightly over next 10 days.',
    },
  ],
};

export const MOCK_CONFIDENCE_METRICS = {
  scorePercent: 94.2,
  uncertaintySpread: '$2.30 / MT',
  bandWidthScore: 88, // 0-100 gauge score (higher = tighter band = higher confidence)
  historicalAccuracy: '91.8%',
  signalsIngested: 48,
  label: 'HIGH MODEL CALIBRATION',
  subtext: 'Narrow P10–P90 uncertainty band indicates strong forecast stability.',
  detailedExplanation: 'A 94.2% calibration score indicates high statistical conviction. 3 of 4 primary indicators (bunker futures, Pacific vessel availability, and import demand) demonstrate tight alignment. The $2.30/MT spread represents maximum expected variance under normal operating conditions.',
};

export const MOCK_SHOCK_RESPONSE = {
  primaryEvent: {
    title: 'Red Sea Crisis & Cape Rerouting (2024)',
    dateRange: 'Nov 2023 – Mar 2024',
    surgePeak: '+18.4%',
    peakDay: 'Day 7',
    recoveryTime: '22 Days',
    description: 'Sudden regional conflict diverted global tonnage around Cape of Good Hope, spiking Pacific spot rates before market equilibrium restored.',
    path: 'M 30 180 L 120 150 L 220 50 L 400 110 L 600 155 L 750 170',
    areaPath: 'M 30 180 L 120 150 L 220 50 L 400 110 L 600 155 L 750 170 L 750 200 L 30 200 Z',
  },
  comparisonEvents: [
    {
      title: 'Suez Canal Blockage (2021)',
      dateRange: 'Mar 2021',
      surgePeak: '+24.1%',
      recoveryTime: '18 Days',
      path: 'M 30 180 L 100 130 L 200 25 L 350 90 L 550 145 L 750 175',
      color: '#486581',
    },
    {
      title: 'Global Bunker Fuel Spike (2022)',
      dateRange: 'Feb 2022',
      surgePeak: '+14.2%',
      recoveryTime: '12 Days',
      path: 'M 30 180 L 140 160 L 250 85 L 420 135 L 620 168 L 750 178',
      color: '#829AB1',
    },
  ],
  mitigationAdvice: 'If a geopolitical shock occurs during active chartering, executing FFA hedges within 72 hours caps peak freight rate inflation by up to 14.8%.',
};
