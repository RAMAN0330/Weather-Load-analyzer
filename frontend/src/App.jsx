import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { API_BASE, TRAINING_API_BASE, getApiUrl, getTrainingApiUrl } from './apiConfig';
import { useSimulatorStore } from './features/simulator/store';
import { useAuthStore } from './features/auth/authStore';

import CommandStrip from './components/CommandStrip';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
} from './components/page/PagePrimitives.jsx';
import PipelineProgress from './components/PipelineProgress';
import AlertRibbon from './components/AlertRibbon';
import WeatherStrip from './components/WeatherStrip';
import ContextPanel from './components/ContextPanel';
import ForecastTable from './components/ForecastTable';
import {
  Activity,
  ArrowRight,
  BarChart3,
  CloudRain,
  Database,
  Download,
  Droplets,
  GitCompare,
  Layers,
  LayoutGrid,
  Loader2,
  LogOut,
  MapPin,
  Play,
  Settings,
  Sun,
  Thermometer,
  TrendingUp,
  Wind,
  Zap,
  ChevronsLeft,
  ChevronsRight,
  Sparkles
} from 'lucide-react';

const ReactECharts = lazy(() => import('echarts-for-react'));
const SimulatorPage = lazy(() => import('./features/simulator/SimulatorPage').then((mod) => ({ default: mod.SimulatorPage })));
const ForecastPage = lazy(() => import('./features/forecast/ForecastPage'));
const AnalysisPage = lazy(() => import('./features/analysis/AnalysisPage'));
const WeatherDeepPage = lazy(() => import('./features/weather/WeatherDeepPage'));
const LoadAnalysisPage = lazy(() => import('./features/load/LoadAnalysisPage'));
const SettingsPage = lazy(() => import('./features/settings/SettingsPage'));
const SimilarDaysPage = lazy(() => import('./features/pipeline/SimilarDaysPage'));
const WeatherLocPage = lazy(() => import('./features/pipeline/WeatherLocPage'));
const AccuracyMonitorPage = lazy(() => import('./features/monitor/AccuracyMonitorPage'));
const BacktestPage = lazy(() => import('./features/backtest/BacktestPage'));
const ForecastStudioPage = lazy(() => import('./features/studio/ForecastStudioPage'));

// API URL builders moved to apiConfig.ts
const API_URL = getApiUrl;
const TRAINING_API_URL = getTrainingApiUrl;


axios.defaults.timeout = 45000;

const APP_TITLE = 'VidyutPragya';
const FORECAST_REQUEST_TIMEOUT_MS = 180000;

const NAV_ITEMS = [
  { key: 'studio', label: 'Forecast Studio', icon: Sparkles },
  { key: 'load_analysis', label: 'Load Analysis', icon: Activity },
  { key: 'weather_analysis', label: 'Weather Analysis', icon: Sun },
  { key: 'simulator', label: 'Simulator', icon: Play },
  { key: 'analysis', label: 'Analysis', icon: BarChart3 },
  { key: 'forecast', label: 'Forecast', icon: LayoutGrid },
  { key: 'monitor', label: 'Monitor', icon: TrendingUp },
  { key: 'backtest', label: 'Backtest', icon: BarChart3 },
  { key: 'similar_days', label: 'Similar Days', icon: Database },
  { key: 'weather_loc', label: 'Weather Locations', icon: MapPin },
];

const OPTIMIZER_TABS = [
  { key: 'window', label: 'Window Selection', icon: Layers },
  { key: 'pattern', label: 'Pattern Fit', icon: Activity },
  { key: 'errors', label: 'Error Diagnostics', icon: BarChart3 }
];


const LIVE_INSIGHT_OPTIONS = [
  { key: 'similar_day_matches', label: 'Similar Day Matches', subtitle: 'Top-N ranked by similarity score' },
  { key: 'sudden_dip_explanations', label: 'Sudden Dip Explanations', subtitle: 'Largest forecast drops with primary driver' },
  { key: 'top_block_contributors', label: 'Top Block Contributors', subtitle: 'Highest absolute slot impact with confidence' },
  { key: 'decision_signals', label: 'Decision Signals', subtitle: 'Risk-ranked recommendations for slot-level actions' }
];

const fmt = (val, digits = 2) => {
  if (val === null || val === undefined) return '—';
  if (typeof val === 'string') return val;
  if (!Number.isFinite(val)) return '—';
  return val.toLocaleString(undefined, { maximumFractionDigits: digits });
};

const pct = (val) => `${fmt(val)}%`;

const getNumericStats = (values = []) => {
  const clean = values.filter((v) => Number.isFinite(v));
  if (!clean.length) return null;
  const count = clean.length;
  const sum = clean.reduce((a, b) => a + b, 0);
  const mean = sum / count;
  const variance = clean.reduce((acc, v) => acc + (v - mean) ** 2, 0) / count;
  const std = Math.sqrt(variance);
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const mae = clean.reduce((acc, v) => acc + Math.abs(v), 0) / count;
  const maxAbs = Math.max(Math.abs(min), Math.abs(max));
  return {
    count,
    mean,
    std,
    min,
    max,
    mae,
    maxAbs
  };
};

const buildHistogram = (values = [], binCount = 10) => {
  const clean = values.filter((v) => Number.isFinite(v));
  if (!clean.length) return null;
  const maxAbs = Math.max(...clean.map((v) => Math.abs(v)));
  const range = maxAbs === 0 ? 1 : maxAbs * 2;
  const step = range / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => {
    const start = -maxAbs + i * step;
    const end = start + step;
    return { start, end, mid: (start + end) / 2, count: 0 };
  });
  clean.forEach((val) => {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((val + maxAbs) / step)));
    bins[idx].count += 1;
  });
  return { bins, maxAbs, step };
};

const dayType = (dateStr) => {
  if (!dateStr) return '—';
  const d = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(d.getTime())) return '—';
  const isWeekend = d.getDay() === 0 || d.getDay() === 6;
  return isWeekend ? 'Weekend' : 'Weekday';
};

const detectSeason = (dateStr) => {
  const raw = String(dateStr || '').trim();
  const dt = raw ? new Date(`${raw}T00:00:00`) : new Date();
  const month = Number.isNaN(dt.getTime()) ? (new Date().getMonth() + 1) : (dt.getMonth() + 1);
  if (month >= 6 && month <= 9) return 'monsoon';
  if (month >= 11 || month <= 2) return 'winter';
  return 'summer';
};

const blockTime = (block) => {
  const minutes = (block - 1) * 15;
  const h = String(Math.floor(minutes / 60)).padStart(2, '0');
  const m = String(minutes % 60).padStart(2, '0');
  return `${h}:${m}`;
};

const titleize = (key) =>
  key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (m) => m.toUpperCase());

const KPI_META = {
  mape: { label: 'MAPE', unit: '%' },
  peak_accuracy_pct: { label: 'Peak Load Accuracy', unit: '%' },
  daily_energy_error_pct: { label: 'Daily Energy Error', unit: '%' },
  rmse_kw: { label: 'RMSE', unit: 'MW' },
  bias_kw: { label: 'Forecast Bias', unit: 'MW' },
  block_accuracy_within_2pct: { label: 'Blocks Within ±2%', unit: '%' },
  block_accuracy_within_5pct: { label: 'Blocks Within ±5%', unit: '%' },
  block_over_10pct: { label: 'Blocks > ±10%', unit: '%' },
  temperature_impact_kw: { label: 'Temperature Impact', unit: 'MW' },
  humidity_impact_kw: { label: 'Humidity Impact', unit: 'MW' },
  precipitation_impact_kw: { label: 'Precipitation Impact', unit: 'MW' },
  calendar_effect_kw: { label: 'Calendar Effect', unit: 'MW' },
  unexplained_variance_kw: { label: 'Unexplained Variance', unit: 'MW' },
  load_dip_count: { label: 'Load Dip Count' },
  load_rise_count: { label: 'Load Rise Count' },
  max_ramp_kw_per_15min: { label: 'Max Ramp Rate', unit: 'MW/15min' },
  cooling_sensitivity_kw_per_c: { label: 'Cooling Sensitivity', unit: 'MW/°C' },
  heating_sensitivity_kw_per_c: { label: 'Heating Sensitivity', unit: 'MW/°C' },
  humidity_amplification_x: { label: 'Humidity Amplification', unit: 'x' },
  precipitation_response_pct: { label: 'Precipitation Response', unit: '%' },
  day_type: { label: 'Day Type' },
  weekend_load_change_pct: { label: 'Weekend Load Change', unit: '%' },
  holiday_impact_pct: { label: 'Holiday Impact', unit: '%' },
  weekend_to_weekday_transition_pct: { label: 'Weekend?Weekday', unit: '%' },
  day_after_holiday_pct: { label: 'Day-After-Holiday', unit: '%' },
  day_type_match_quality: { label: 'Day-Type Match Quality', unit: '%' },
  baseline_confidence_score: { label: 'Baseline Confidence', unit: '%' },
  baseline_std_dev: { label: 'Baseline Std Dev', unit: 'MW' },
  baseline_completeness_pct: { label: 'Baseline Completeness', unit: '%' },
  baseline_outlier_days: { label: 'Baseline Outlier Days' },
  reserve_margin_adequacy_pct: { label: 'Reserve Margin', unit: '%' },
  generator_commitment_alignment_pct: { label: 'Commitment Alignment', unit: '%' },
  dr_accuracy_pct: { label: 'DR Accuracy', unit: '%' },
  peak_time_prediction_category: { label: 'Peak Time Prediction' },
  peak_time_block_error: { label: 'Peak Time Error', unit: 'blocks' },
  peak_variance_attribution_temp_kw: { label: 'Peak Temp Attribution', unit: 'MW' },
  peak_variance_attribution_hum_kw: { label: 'Peak Humidity Attribution', unit: 'MW' },
  peak_variance_attribution_precip_kw: { label: 'Peak Precip Attribution', unit: 'MW' },
  daily_energy_variance_temp_kw: { label: 'Daily Energy Temp', unit: 'MW' },
  daily_energy_variance_hum_kw: { label: 'Daily Energy Humidity', unit: 'MW' },
  daily_energy_variance_precip_kw: { label: 'Daily Energy Precip', unit: 'MW' },
  morning_ramp_delta_kw: { label: 'Morning Ramp Delta', unit: 'MW' },
  midday_plateau_delta_kw: { label: 'Midday Plateau Delta', unit: 'MW' },
  evening_peak_delta_kw: { label: 'Evening Peak Delta', unit: 'MW' },
  night_valley_delta_kw: { label: 'Night Valley Delta', unit: 'MW' },
  heat_index_effect: { label: 'Heat Index Effect', unit: 'corr' },
  weekend_weather_interaction: { label: 'Weekend Weather Interaction', unit: 'corr' },
  rain_temperature_interaction: { label: 'Rain × Temp Interaction', unit: 'corr' },
  mape_improvement_pct: { label: 'MAPE Improvement', unit: '%' },
  bias_trend_improving: { label: 'Bias Trend Improving' },
  problematic_block_identification_rate: { label: 'Identified High-Error Blocks', unit: '%' },
  historical_completeness_pct: { label: 'Historical Completeness', unit: '%' },
  weather_data_accuracy: { label: 'Weather Data Accuracy' }
};

const KPI_TARGETS = {
  mape: { type: 'max', good: 3, warn: 5 },
  peak_accuracy_pct: { type: 'max', good: 2, warn: 3 },
  daily_energy_error_pct: { type: 'max', good: 2.5, warn: 4 },
  rmse_kw: { type: 'max', good: 0.5, warn: 0.7 },
  bias_kw: { type: 'abs', good: 0.1, warn: 0.2 },
  block_accuracy_within_2pct: { type: 'min', good: 70, warn: 60 },
  block_accuracy_within_5pct: { type: 'min', good: 90, warn: 80 },
  block_over_10pct: { type: 'max', good: 2, warn: 5 },
  reserve_margin_adequacy_pct: { type: 'range', good: [15, 20], warn: [10, 25] },
  dr_accuracy_pct: { type: 'min', good: 90, warn: 80 },
  baseline_confidence_score: { type: 'min', good: 85, warn: 70 },
  historical_completeness_pct: { type: 'min', good: 98, warn: 95 },
  unexplained_variance_kw: { type: 'max', good: 0, warn: 2 }
};

const REGION_DETAILS = {
  odisha: {
    label: 'Odisha',
    subtitle: 'Coastal industry, monsoon swings, and heavy-grid draw clusters.',
    signal: 'Cyclone-aware operations',
    tags: ['Port + metals', 'Rain volatility'],
    icon: Layers,
    footerNote: 'Best for monsoon-sensitive load shifts, industrial clusters, and coastal weather stress.'
  },
  rajasthan: {
    label: 'Rajasthan',
    subtitle: 'Solar-heavy daytime behavior with desert heat and ramp transitions.',
    signal: 'Solar ramp watch',
    tags: ['Solar swing', 'Dry heat load'],
    icon: Sun,
    footerNote: 'Best for solar ramp analysis, weather-driven cooling load, and daylight peak planning.'
  },
  haryana: {
    label: 'Haryana',
    subtitle: 'Industrial and agricultural demand with sharp urban evening acceleration.',
    signal: 'Evening peak acceleration',
    tags: ['Mixed demand', 'Fast ramps'],
    icon: TrendingUp,
    footerNote: 'Best for mixed urban-industrial behavior, short-term stress windows, and evening peak readiness.'
  },
  chhattisgarh: {
    label: 'Chhattisgarh',
    subtitle: 'Coal-heavy generation mix with industrial load clusters and seasonal demand swings.',
    signal: 'Thermal dispatch watch',
    tags: ['Industrial load', 'Coal-heavy'],
    icon: Activity,
    footerNote: 'Best for thermal-dominant dispatch analysis, industrial demand patterns, and monsoon-season variability.'
  }
};

const downloadBlob = (data, filename, type) => {
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

const Toast = ({ tone, message, onClose }) => (
  <div className={`toast ${tone}`} onClick={onClose}>
    <span>{message}</span>
    <button type="button">×</button>
  </div>
);

const SkeletonPanel = ({ lines = 4 }) => (
  <div className="skeleton-card">
    {Array.from({ length: lines }).map((_, i) => (
      <div key={i} className="skeleton-line" />
    ))}
  </div>
);

const ChartLoading = ({ label = 'Loading chart...' }) => (
  <div className="chart-loading">
    <div className="spin" />
    <span>{label}</span>
  </div>
);

const ViewLoading = ({ label = 'Loading view...' }) => (
  <div className="glass-panel p-4">
    <ChartLoading label={label} />
  </div>
);

const RegionSelectionModal = ({ regions, selected, onSelect, onConfirm, dateRange, onDateRange }) => {
  const selectedMeta = selected ? (REGION_DETAILS[selected] || null) : null;

  // Default: last 120 days up to today
  const todayStr = new Date().toISOString().slice(0, 10);
  const presetFrom = (days) => new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
  const defaultFrom = presetFrom(120);
  const isActivePreset = (days) =>
    (dateRange.from || defaultFrom) === presetFrom(days) && (dateRange.to || todayStr) === todayStr;

  return (
    <div className="overlay region-overlay">
      <div className="modal region-modal">
        <div className="modal-header region-modal__header">
          <p className="region-kicker">State Workspace</p>
          <h2>Select Your State</h2>
          <p>Choose a state to load localized power grid analytics and weather-aware forecasting.</p>
        </div>

        <div className="region-grid">
          {regions.map((r) => {
            const meta = REGION_DETAILS[r] || {
              label: titleize(r),
              subtitle: 'Localized load analytics and weather-aware forecasting.',
              signal: 'State-specific forecasting',
              tags: ['Grid analytics', 'Weather aware'],
              icon: Layers,
              footerNote: 'State-specific dashboard with localized forecasting and grid intelligence.'
            };
            const Icon = meta.icon;

            return (
              <button
                type="button"
                key={r}
                className={`region-tile ${selected === r ? 'active' : ''}`}
                onClick={() => onSelect(r)}
              >
                <div className="region-tile__top">
                  <div className="region-icon">
                    <Icon size={22} />
                  </div>
                </div>

                <div className="region-copy">
                  <span className="region-name">{meta.label}</span>
                  <span className="region-subtitle">{meta.subtitle}</span>
                </div>

                <span className="region-signal">{meta.signal}</span>
              </button>
            );
          })}
        </div>

        {/* ── Date range picker ─────────────────────────────────────────── */}
        <div className="region-date-range">
          <div className="region-date-range__label">
            <span>Data Range</span>
            <span className="region-date-range__hint">Fetch historical data between these dates</span>
          </div>
          <div className="region-date-range__inputs">
            <div className="region-date-field">
              <label>From</label>
              <input
                type="date"
                value={dateRange.from || defaultFrom}
                max={dateRange.to || todayStr}
                onChange={e => onDateRange({ ...dateRange, from: e.target.value })}
              />
            </div>
            <div className="region-date-range__sep">→</div>
            <div className="region-date-field">
              <label>To</label>
              <input
                type="date"
                value={dateRange.to || todayStr}
                min={dateRange.from || defaultFrom}
                max={todayStr}
                onChange={e => onDateRange({ ...dateRange, to: e.target.value })}
              />
            </div>
          </div>
          <div className="region-date-presets">
            <button type="button" className={`region-date-preset ${isActivePreset(120) ? 'active' : ''}`} title="Last 120 days"
              onClick={() => onDateRange({ from: defaultFrom, to: todayStr })}>120d</button>
            <button type="button" className={`region-date-preset ${isActivePreset(60) ? 'active' : ''}`} title="Last 60 days"
              onClick={() => onDateRange({ from: presetFrom(60), to: todayStr })}>60d</button>
            <button type="button" className={`region-date-preset ${isActivePreset(30) ? 'active' : ''}`} title="Last 30 days"
              onClick={() => onDateRange({ from: presetFrom(30), to: todayStr })}>30d</button>
          </div>
        </div>

        <div className="modal-footer region-modal__footer">
          <div className="region-selection-summary">
            <span className="region-selection-summary__label">Selected state</span>
            <strong>{selectedMeta?.label || 'Choose a state to continue'}</strong>
            <p>
              {selectedMeta?.footerNote || 'Your launch will open a tailored dashboard with localized baselines, weather-impact analytics, and operator-ready forecasting signals.'}
            </p>
          </div>
          <button
            className="primary-btn start-btn region-launch-btn"
            disabled={!selected}
            onClick={onConfirm}
          >
            <span>Launch Dashboard</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

const LoadChart = ({
  blocks = [],
  baseline = [],
  forecast = [],
  actual = [],
  p10 = [],
  p90 = [],
  highlight = null,
  onSelectBlock,
  peakBlock,
  peakValue,
  peakMarkers,
  residuals = [],
  residualThreshold,
  variance = [],
  thresholds,
  dateLabel,
  showForecast = true
}) => {
  if (!blocks.length) return <div className="chart-empty">No data available.</div>;
  const axisInterval = blocks.length >= 90 ? 3 : blocks.length >= 64 ? 2 : blocks.length >= 48 ? 1 : 0;
  const forecastSeries = (Array.isArray(forecast) && forecast.length === blocks.length) ? forecast : baseline;
  const alertSeries = showForecast ? forecastSeries : baseline;

  const scatterData = variance
    .map((v, idx) => ({ idx, v }))
    .filter(({ v }) => thresholds && Math.abs(v) >= thresholds.warning)
    .map(({ idx, v }) => ({
      value: [blocks[idx], alertSeries[idx] ?? null, v],
      itemStyle: { color: Math.abs(v) >= thresholds.critical ? '#c7655f' : '#d2a46f' }
    }));

  const residualAreas = [];
  if (Array.isArray(residuals) && residuals.length && blocks.length && Number.isFinite(residualThreshold) && residualThreshold > 0) {
    let start = null;
    const lastIdx = Math.min(residuals.length, blocks.length) - 1;
    for (let i = 0; i <= lastIdx; i += 1) {
      const val = residuals[i];
      const isHigh = Number.isFinite(val) && Math.abs(val) >= residualThreshold;
      if (isHigh && start === null) start = i;
      if (!isHigh && start !== null) {
        residualAreas.push([start, i - 1]);
        start = null;
      }
    }
    if (start !== null) residualAreas.push([start, lastIdx]);
  }

  const markAreas = [];
  if (highlight?.start && highlight?.end) {
    markAreas.push([
      { xAxis: highlight.start, itemStyle: { color: 'rgba(199, 101, 95, 0.18)' } },
      { xAxis: highlight.end }
    ]);
  }
  residualAreas.forEach(([start, end]) => {
    if (blocks[start] === undefined || blocks[end] === undefined) return;
    markAreas.push([
      { xAxis: blocks[start], itemStyle: { color: 'rgba(210, 164, 111, 0.16)' } },
      { xAxis: blocks[end] }
    ]);
  });

  const markArea = markAreas.length ? { data: markAreas } : undefined;

  const baselineMarkPoint = peakMarkers?.baseline
    ? {
      data: [{ coord: [peakMarkers.baseline.block, peakMarkers.baseline.value] }],
      symbolSize: 10,
      itemStyle: { color: '#d2a46f' },
      label: { show: true, color: '#d2a46f', formatter: 'Peak' }
    }
    : undefined;

  const actualMarkPoint = peakMarkers?.actual
    ? {
      data: [{ coord: [peakMarkers.actual.block, peakMarkers.actual.value] }],
      symbolSize: 10,
      itemStyle: { color: '#4f7d5c' },
      label: { show: true, color: '#4f7d5c', formatter: 'Peak' }
    }
    : undefined;

  // Compute y-axis bounds from actual data so the axis never starts at 0 when data is far above it
  const _allVals = [...forecast, ...baseline, ...actual, ...(p10 || []), ...(p90 || [])]
    .filter(v => v != null && Number.isFinite(v) && v > 0);
  const _dataMin = _allVals.length ? Math.min(..._allVals) : 0;
  const _dataMax = _allVals.length ? Math.max(..._allVals) : 12000;
  const _pad = (_dataMax - _dataMin) * 0.06;
  const _yMin = _dataMin > 500 ? Math.floor((_dataMin - _pad) / 500) * 500 : 0;
  const _yMax = Math.ceil((_dataMax + _pad) / 500) * 500;

  const option = {
    backgroundColor: 'transparent',
    grid: { left: 40, right: 20, top: 30, bottom: 55, containLabel: true },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1C1F28',
      borderColor: '#262A35',
      borderWidth: 1,
      padding: [10, 14],
      borderRadius: 4,
      textStyle: { color: '#E2E4E9', fontSize: 12, fontFamily: 'IBM Plex Mono, monospace' },
      formatter: (params) => {
        const block = params?.[0]?.axisValue;
        const bIdx = blocks.indexOf(Number(block));
        const f = forecastSeries[bIdx];
        const b = baseline[bIdx];
        const a = actual[bIdx];
        const lo = p10[bIdx];
        const hi = p90[bIdx];
        const lines = [
          `<div style="font-family:'IBM Plex Mono';font-weight:600;font-size:11px;margin-bottom:6px;border-bottom:1px solid #262A35;padding-bottom:4px;color:#8B8FA3;">BLOCK ${block} • ${blockTime(Number(block))}</div>`,
          `<div style="display:flex;justify-content:space-between;gap:20px;color:#565B6B;"><span>Baseline</span> <span style="font-weight:600;">${fmt(b)} MW</span></div>`,
          `<div style="display:flex;justify-content:space-between;gap:20px;color:#E2E4E9;"><span>Actual</span> <span style="font-weight:600;">${fmt(a)} MW</span></div>`
        ];
        if (showForecast) {
          lines.splice(2, 0, `<div style="display:flex;justify-content:space-between;gap:20px;color:#4A90D9;"><span>Forecast</span> <span style="font-weight:600;">${fmt(f)} MW</span></div>`);
          if (lo != null && hi != null && lo > 0) {
            lines.push(`<div style="display:flex;justify-content:space-between;gap:20px;color:#565B6B;font-size:11px;"><span>P10–P90</span> <span>${fmt(lo)} – ${fmt(hi)} MW</span></div>`);
          }
        }
        return lines.join('');
      }
    },
    xAxis: {
      type: 'category',
      data: blocks,
      nameLocation: 'middle',
      nameGap: 30,
      axisLine: { lineStyle: { color: '#262A35' } },
      axisTick: { show: false },
      axisLabel: { color: '#565B6B', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace', interval: axisInterval, margin: 15 },
      splitLine: { show: false }
    },
    yAxis: {
      type: 'value',
      min: _yMin,
      max: _yMax,
      axisLine: { lineStyle: { color: '#262A35' } },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: '#1C1F28', type: 'solid' } },
      axisLabel: { color: '#565B6B', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace',
        formatter: (v) => Math.abs(v) >= 1000 ? `${(v/1000).toFixed(1)}k` : `${v}` }
    },
    series: [
      // P10-P90 confidence band (stacked area — transparent floor + colored band)
      ...(showForecast && p10.length === blocks.length ? [
        {
          name: 'P10',
          type: 'line',
          data: p10,
          smooth: true,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          stack: 'confidence',
          areaStyle: { opacity: 0 },
          z: 1
        },
        {
          name: 'P10–P90 Band',
          type: 'line',
          data: p90.map((v, i) => Math.max(0, v - (p10[i] || 0))),
          smooth: true,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          stack: 'confidence',
          areaStyle: { color: 'rgba(74, 144, 217, 0.08)' },
          z: 1
        }
      ] : []),
      {
        name: 'Baseline',
        type: 'line',
        data: baseline,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#3B4050', type: 'dashed', width: 1 },
        markArea: showForecast ? undefined : markArea,
        markPoint: baselineMarkPoint,
        z: 2
      },
      ...(showForecast ? [{
        name: 'Forecast',
        type: 'line',
        data: forecastSeries,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: '#4A90D9',
          width: 2
        },
        itemStyle: { color: '#4A90D9' },
        markArea,
        markPoint: peakBlock && peakValue !== undefined ? {
          data: [{ coord: [peakBlock, peakValue] }],
          symbolSize: 8,
          itemStyle: { color: '#D4952A' }
        } : undefined,
        z: 3
      }] : []),
      {
        name: 'Actual',
        type: 'line',
        data: actual,
        smooth: true,
        showSymbol: false,
        lineStyle: {
          color: '#E2E4E9',
          width: 1.5
        },
        itemStyle: { color: '#E2E4E9' },
        markPoint: actualMarkPoint
      },
      {
        name: 'Variance Alerts',
        type: 'scatter',
        data: scatterData,
        symbolSize: 8,
        itemStyle: {
          borderWidth: 2,
          borderColor: '#0F1117'
        },
        tooltip: { show: false }
      }
    ]
  };

  const onEvents = {
    click: (params) => {
      if (params?.dataIndex != null && onSelectBlock) {
        onSelectBlock(params.dataIndex + 1);
      }
    }
  };

  return (
    <div className="chart-wrap" style={{ minHeight: 320 }}>
      <Suspense fallback={<ChartLoading />}>
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          onEvents={onEvents}
          notMerge
          lazyUpdate
        />
      </Suspense>
    </div>
  );
};

const KpiCard = ({ title, value, unit, status, subtitle, delta, footer, children }) => (
  <div className={`kpi-card ${status || ''}`}>
    <div className="kpi-title">{title}</div>
    <div className="kpi-value">
      <span>{value}</span>
      {unit && <span className="kpi-unit">{unit}</span>}
    </div>
    {subtitle && <div className="kpi-subtitle">{subtitle}</div>}
    {delta && <div className="kpi-delta">{delta}</div>}
    {children}
    {footer && <div className="kpi-footer">{footer}</div>}
  </div>
);

const AttributionBar = ({ label, value, percent, color }) => (
  <div className="attrib-row">
    <div className="attrib-label">{label}</div>
    <div className="attrib-bar">
      <div className="attrib-fill" style={{ width: `${Math.min(Math.abs(percent), 100)}%`, background: color }}></div>
    </div>
    <div className="attrib-meta">
      {fmt(value)} MW
      <span>{fmt(percent)}%</span>
    </div>
  </div>
);

const ImpactMatrix = ({ blocks = [], impacts = [] }) => {
  if (!blocks.length || !impacts?.length) return <div className="muted">No block-level KPI impacts available.</div>;
  const columns = [
    { key: 'temperature', label: 'Temperature', color: '#d19069' },
    { key: 'humidity', label: 'Humidity', color: '#c8b39b' },
    { key: 'precipitation', label: 'Precipitation', color: '#7f8b75' },
    { key: 'unexplained', label: 'Unexplained', color: '#736f68' }
  ];
  return (
    <div className="impact-matrix">
      <div className="impact-head">
        <span>Block</span>
        <span>Time</span>
        <span>Total ?</span>
        {columns.map((col) => (
          <span key={col.key}>{col.label}</span>
        ))}
      </div>
      {impacts.map((row) => (
        <div className="impact-row" key={row.block}>
          <span>{row.block}</span>
          <span>{row.time}</span>
          <span>{fmt(row.total_variance)} MW</span>
          {columns.map((col) => (
            <span key={col.key} style={{ color: col.color }}>
              {fmt(row[col.key])} MW
            </span>
          ))}
        </div>
      ))}
    </div>
  );
};

const CausalImpactMatrix = ({ impacts = [] }) => {
  if (!impacts?.length) return <div className="muted">No block-level impact data.</div>;
  return (
    <div className="impact-matrix causal">
      <div className="impact-head">
        <span>Block</span>
        <span>Time</span>
        <span>Total ?</span>
        <span>Weather ?</span>
        <span>System ?</span>
        <span>Model ?</span>
        <span>Primary</span>
        <span>Confidence</span>
      </div>
      {impacts.map((row) => {
        const weather = (row.temperature || 0) + (row.humidity || 0) + (row.precipitation || 0);
        const model = row.unexplained || 0;
        const system = (row.total_variance || 0) - weather - model;
        const primary = [
          { key: 'Weather', val: weather },
          { key: 'System', val: system },
          { key: 'Model', val: model }
        ].sort((a, b) => Math.abs(b.val) - Math.abs(a.val))[0]?.key;
        const conf = Math.max(0, 100 - (Math.abs(model) / Math.max(Math.abs(row.total_variance || 1), 1)) * 100);
        return (
          <div className="impact-row" key={`causal-${row.block}`}>
            <span>{row.block}</span>
            <span>{row.time}</span>
            <span>{fmt(row.total_variance)} MW</span>
            <span className="impact-weather">{fmt(weather)} MW</span>
            <span className="impact-system">{fmt(system)} MW</span>
            <span className="impact-model">{fmt(model)} MW</span>
            <span>{primary}</span>
            <span>{fmt(conf)}%</span>
          </div>
        );
      })}
    </div>
  );
};

const ProgressBar = ({ value, max = 100 }) => {
  const pctVal = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${pctVal}%` }} />
    </div>
  );
};

const Ring = ({ value = 0 }) => {
  const size = 44;
  const stroke = 6;
  const radius = (size - stroke) / 2;
  const circ = 2 * Math.PI * radius;
  const pctVal = Math.max(0, Math.min(100, value));
  const offset = circ - (pctVal / 100) * circ;
  return (
    <svg width={size} height={size} className="ring">
      <circle cx={size / 2} cy={size / 2} r={radius} stroke="rgba(240, 221, 199, 0.16)" strokeWidth={stroke} fill="none" />
      <circle cx={size / 2} cy={size / 2} r={radius} stroke="var(--accent)" strokeWidth={stroke} fill="none" strokeDasharray={circ} strokeDashoffset={offset} />
      <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle" className="ring-text">
        {Math.round(pctVal)}%
      </text>
    </svg>
  );
};

const KPI_GROUPS = [
  { key: 'forecast_accuracy', label: 'Forecast Accuracy', keys: ['mape', 'peak_accuracy_pct', 'daily_energy_error_pct', 'rmse_kw', 'bias_kw', 'block_accuracy_within_2pct', 'block_accuracy_within_5pct', 'block_over_10pct'] },
  { key: 'attribution', label: 'Attribution & Root Cause', keys: ['temperature_impact_kw', 'humidity_impact_kw', 'precipitation_impact_kw', 'calendar_effect_kw', 'unexplained_variance_kw'] },
  { key: 'pattern', label: 'Load Pattern Analysis', keys: ['load_dip_count', 'load_rise_count', 'max_ramp_kw_per_15min'] },
  { key: 'weather', label: 'Weather Sensitivity', keys: ['cooling_sensitivity_kw_per_c', 'heating_sensitivity_kw_per_c', 'humidity_amplification_x', 'precipitation_response_pct'] },
  { key: 'calendar', label: 'Day Type & Calendar', keys: ['day_type', 'weekend_load_change_pct', 'holiday_impact_pct', 'weekend_to_weekday_transition_pct', 'day_after_holiday_pct', 'day_type_match_quality'] },
  { key: 'baseline', label: 'Baseline Quality', keys: ['baseline_confidence_score', 'baseline_std_dev', 'baseline_completeness_pct', 'baseline_outlier_days'] },
  { key: 'operational', label: 'Operational Decision', keys: ['reserve_margin_adequacy_pct', 'generator_commitment_alignment_pct', 'dr_accuracy_pct', 'peak_time_prediction_category', 'peak_time_block_error'] },
  { key: 'variance', label: 'Variance Attribution', keys: ['peak_variance_attribution_temp_kw', 'peak_variance_attribution_hum_kw', 'peak_variance_attribution_precip_kw', 'daily_energy_variance_temp_kw', 'daily_energy_variance_hum_kw', 'daily_energy_variance_precip_kw'] },
  { key: 'time_patterns', label: 'Time-Based Patterns', keys: ['morning_ramp_delta_kw', 'midday_plateau_delta_kw', 'evening_peak_delta_kw', 'night_valley_delta_kw'] },
  { key: 'interactions', label: 'Multi-Factor Interactions', keys: ['heat_index_effect', 'weekend_weather_interaction', 'rain_temperature_interaction'] },
  { key: 'improvement', label: 'Forecast Improvement', keys: ['mape_improvement_pct', 'bias_trend_improving', 'problematic_block_identification_rate'] },
  { key: 'data_quality', label: 'Data Quality', keys: ['historical_completeness_pct', 'weather_data_accuracy'] }
];

const getStatus = (key, value) => {
  const target = KPI_TARGETS[key];
  if (value === null || value === undefined || Number.isNaN(value)) return 'neutral';
  if (!target) return 'neutral';
  const val = typeof value === 'number' ? value : value;
  if (target.type === 'max') {
    if (val <= target.good) return 'good';
    if (val <= target.warn) return 'warning';
    return 'critical';
  }
  if (target.type === 'min') {
    if (val >= target.good) return 'good';
    if (val >= target.warn) return 'warning';
    return 'critical';
  }
  if (target.type === 'abs') {
    const abs = Math.abs(val);
    if (abs <= target.good) return 'good';
    if (abs <= target.warn) return 'warning';
    return 'critical';
  }
  if (target.type === 'range') {
    const [gmin, gmax] = target.good;
    const [wmin, wmax] = target.warn;
    if (val >= gmin && val <= gmax) return 'good';
    if (val >= wmin && val <= wmax) return 'warning';
    return 'critical';
  }
  return 'neutral';
};

class KpiErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <div className="muted">Unable to render KPI library. Check console for details.</div>;
    }
    return this.props.children;
  }
}

const KPI_TIERS = {
  tier1: new Set(['mape', 'peak_accuracy_pct', 'daily_energy_error_pct', 'reserve_margin_adequacy_pct', 'bias_kw']),
  tier2: new Set(['rmse_kw', 'block_accuracy_within_2pct', 'block_accuracy_within_5pct', 'block_over_10pct', 'temperature_impact_kw', 'humidity_impact_kw', 'unexplained_variance_kw', 'dr_accuracy_pct'])
};

const KPIGrid = ({ kpis, filterText = '', viewMode = 'all' }) => {
  if (!kpis) return null;
  const normalizedFilter = String(filterText || '').toLowerCase();
  return (
    <div className="kpi-library">
      {KPI_GROUPS.map((group) => {
        const items = group.keys
          .filter((key) => Object.prototype.hasOwnProperty.call(kpis, key))
          .map((key) => ({ key, value: kpis[key] }))
          .map((item) => {
            const meta = KPI_META[item.key] || {};
            const label = meta.label || titleize(item.key);
            const unit = meta.unit || '';
            const status = getStatus(item.key, item.value);
            const tier = KPI_TIERS.tier1.has(item.key) ? 'tier-1' : KPI_TIERS.tier2.has(item.key) ? 'tier-2' : 'tier-3';
            return { ...item, label, unit, status, tier };
          })
          .filter((item) => {
            if (!normalizedFilter) return true;
            return item.key.toLowerCase().includes(normalizedFilter) || String(item.label || '').toLowerCase().includes(normalizedFilter);
          })
          .filter((item) => viewMode === 'alerts' ? (item.status === 'warning' || item.status === 'critical') : true)
          .sort((a, b) => {
            const order = { 'tier-1': 0, 'tier-2': 1, 'tier-3': 2 };
            return order[a.tier] - order[b.tier];
          });
        if (!items.length) return null;
        return (
          <div key={group.key} className={`kpi-group group-${group.key}`}>
            <div className="kpi-group-title">{group.label}</div>
            <div className="kpi-group-grid">
              {items.map((item) => {
                const valRaw = typeof item.value === 'boolean' ? (item.value ? 'Yes' : 'No') : fmt(item.value);
                const isNA = valRaw === '—';
                const val = isNA ? 'N/A' : valRaw;
                const target = KPI_TARGETS[item.key];
                const targetText = target
                  ? target.type === 'range'
                    ? `Target: ${target.good[0]}–${target.good[1]}`
                    : target.type === 'min'
                      ? `Target: = ${target.good}`
                      : target.type === 'max'
                        ? `Target: = ${target.good}`
                        : target.type === 'abs'
                          ? `Target: ±${target.good}`
                          : null
                  : null;
                const rawValue = typeof item.value === 'number' ? item.value : Number(item.value);
                const safeValue = Number.isFinite(rawValue) ? rawValue : null;
                const progressValue = target?.type === 'max' && safeValue !== null
                  ? (target.good / Math.max(safeValue, target.good)) * 100
                  : target?.type === 'min' && safeValue !== null
                    ? Math.min(100, (safeValue / target.good) * 100)
                    : target?.type === 'abs' && safeValue !== null
                      ? Math.max(0, 100 - (Math.abs(safeValue) / target.good) * 100)
                      : null;
                return (
                  <div key={item.key} className={`kpi-chip ${item.tier} ${item.status}`}>
                    <div className="kpi-chip-head">
                      <span className="kpi-label">{item.label}</span>
                      {item.status !== 'neutral' && <span className={`status-pill ${item.status}`}>{item.status}</span>}
                    </div>
                    <strong className="kpi-value-big">{val}{!isNA && item.unit ? ` ${item.unit}` : ''}</strong>
                    {targetText && <div className="kpi-target">{targetText}</div>}
                    {progressValue !== null && (
                      <div className="kpi-target-bar">
                        <div className="kpi-target-fill" style={{ width: `${Math.max(0, Math.min(100, progressValue))}%` }} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const BlockVarianceChart = ({ blocks = [], variance = [], thresholds = { warning: 5, critical: 10 }, onSelect }) => {
  if (!blocks.length) return <div className="chart-empty">No variance data.</div>;
  const maxAbs = Math.max(...variance.map((v) => Math.abs(v)), 1) || 1;
  return (
    <div className="variance-chart">
      {blocks.map((block, idx) => {
        const v = variance[idx];
        const height = Math.max(6, (Math.abs(v) / maxAbs) * 100);
        const tone = Math.abs(v) >= thresholds.critical ? 'critical' : Math.abs(v) >= thresholds.warning ? 'warning' : 'ok';
        return (
          <button
            key={block}
            className={`variance-bar ${tone}`}
            style={{ height: `${height}%` }}
            onClick={() => onSelect && onSelect(block)}
            title={`Block ${block} • ${fmt(v)}%`}
            type="button"
          />
        );
      })}
    </div>
  );
};

// ── Raw Data Viewer Table with Excel-like filtering ──────────────
function DataViewerTable({ rows, keys, fmt }) {
  const [filters, setFilters] = React.useState({});
  const [sortCol, setSortCol] = React.useState(null);
  const [sortDir, setSortDir] = React.useState('asc');
  const [activeFilter, setActiveFilter] = React.useState(null);

  const setFilter = (col, value) => {
    setFilters(prev => {
      const next = { ...prev };
      if (!value && value !== 0) delete next[col];
      else next[col] = value;
      return next;
    });
  };

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('asc'); }
  };

  const filtered = React.useMemo(() => {
    let result = rows;
    Object.entries(filters).forEach(([col, val]) => {
      const v = String(val).toLowerCase();
      result = result.filter(row => {
        const cell = row[col];
        if (cell == null) return v === '';
        return String(cell).toLowerCase().includes(v);
      });
    });
    if (sortCol) {
      result = [...result].sort((a, b) => {
        const av = a[sortCol], bv = b[sortCol];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        const na = Number(av), nb = Number(bv);
        if (!isNaN(na) && !isNaN(nb)) return sortDir === 'asc' ? na - nb : nb - na;
        return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
    }
    return result;
  }, [rows, filters, sortCol, sortDir]);

  const activeFilterCount = Object.keys(filters).length;

  const formatCell = (val, col) => {
    if (val == null || val === '') return '—';
    if (typeof val === 'number') {
      if (col.includes('mw') || col.includes('MW') || col.includes('load') || col.includes('drawal'))
        return val.toFixed(1);
      if (col.includes('pct') || col.includes('percent') || col.includes('confidence'))
        return (val * 100).toFixed(1) + '%';
      return Number.isInteger(val) ? String(val) : val.toFixed(2);
    }
    return String(val);
  };

  const getCellClass = (val, col) => {
    if (typeof val !== 'number') return '';
    if (col === 'diff_mw' || col.includes('diff') || col.includes('deviation')) {
      if (Math.abs(val) > 100) return 'dv-cell-danger';
      if (Math.abs(val) > 50) return 'dv-cell-warn';
      if (val !== 0) return 'dv-cell-ok';
    }
    return '';
  };

  return (
    <div className="dv-body">
      {activeFilterCount > 0 && (
        <div className="dv-filter-bar">
          <span className="dv-filter-count">{activeFilterCount} filter{activeFilterCount > 1 ? 's' : ''} active</span>
          <span className="dv-filter-results">{filtered.length} of {rows.length} rows</span>
          <button className="dv-clear-filters" onClick={() => setFilters({})}>Clear all</button>
        </div>
      )}
      <div className="dv-table-wrap">
        <table className="dv-table">
          <thead>
            <tr className="dv-head-row">
              <th className="dv-row-num">#</th>
              {keys.map(k => (
                <th key={k} className="dv-th">
                  <div className="dv-th-content" onClick={() => handleSort(k)}>
                    <span className="dv-th-label">{k.replace(/_/g, ' ')}</span>
                    {sortCol === k && <span className="dv-sort-arrow">{sortDir === 'asc' ? '▲' : '▼'}</span>}
                  </div>
                  <button
                    className={`dv-filter-btn ${filters[k] ? 'active' : ''}`}
                    onClick={(e) => { e.stopPropagation(); setActiveFilter(activeFilter === k ? null : k); }}
                    title="Filter column"
                  >▼</button>
                  {activeFilter === k && (
                    <div className="dv-filter-dropdown" onClick={e => e.stopPropagation()}>
                      <input
                        className="dv-filter-input"
                        type="text"
                        placeholder={`Filter ${k.replace(/_/g, ' ')}...`}
                        value={filters[k] || ''}
                        onChange={e => setFilter(k, e.target.value)}
                        autoFocus
                      />
                      {filters[k] && (
                        <button className="dv-filter-clear" onClick={() => setFilter(k, '')}>×</button>
                      )}
                    </div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, i) => (
              <tr key={i} className={`dv-row ${i % 2 === 0 ? 'dv-row-even' : ''}`}>
                <td className="dv-row-num">{i + 1}</td>
                {keys.map(k => (
                  <td key={k} className={`dv-td ${getCellClass(row[k], k)}`}>
                    {formatCell(row[k], k)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MONITOR PAGE  –  Similar Days & Statistics
   ═══════════════════════════════════════════════════════════════ */
function MonitorPage({ live, dayAhead, effectiveDate, selectedRegion }) {
  const [activeKpi, setActiveKpi] = React.useState(null);
  const [statTab, setStatTab] = React.useState('central');
  const [simSort, setSimSort] = React.useState({ col: 'rank', dir: 'asc' });
  const [simFilter, setSimFilter] = React.useState('all');
  const [selectedSimRow, setSelectedSimRow] = React.useState(null);
  const [activeBin, setActiveBin] = React.useState(null);
  const [activeError, setActiveError] = React.useState(null);

  /* ── data ── */
  const monSeries = live?.series || dayAhead?.series;
  const monActual = monSeries?.actual || monSeries?.forecast || [];
  const monBaseline = monSeries?.baseline || monSeries?.hybrid_baseline || [];
  const monBlocks = monSeries?.blocks || Array.from({ length: 96 }, (_, i) => i + 1);
  const monLoads = monBlocks.map((_, i) => Number.isFinite(monActual[i]) ? Math.round(monActual[i]) : 0);
  const monYLoads = monBlocks.map((_, i) => Number.isFinite(monBaseline[i]) ? Math.round(monBaseline[i]) : 0);
  const monMeta = live?.metadata || dayAhead?.metadata || {};
  const monIntra = live?.weather_analysis?.intraday || dayAhead?.weather_analysis?.intraday || {};
  const monTemps = (monIntra.temperature?.actual || []).map(Number);

  /* ── similar days ── */
  const rawSim = monMeta?.similar_days || live?.similar_days || [];

  /* ── statistics ── */
  const validLoads = monLoads.filter(v => v > 0);
  const peak    = validLoads.length ? Math.max(...validLoads) : 0;
  const valley  = validLoads.length ? Math.min(...validLoads) : 0;
  const energy  = monLoads.reduce((s, v) => s + v * 0.25, 0);
  const avgLoad = validLoads.length ? validLoads.reduce((a, b) => a + b, 0) / validLoads.length : 0;
  const loadFactor = peak > 0 ? (avgLoad / peak) * 100 : 0;
  const sorted  = [...validLoads].sort((a, b) => a - b);
  const median  = sorted.length ? (sorted.length % 2 ? sorted[Math.floor(sorted.length / 2)] : (sorted[Math.floor(sorted.length / 2) - 1] + sorted[Math.floor(sorted.length / 2)]) / 2) : 0;
  const q1  = sorted.length >= 4 ? sorted[Math.floor(sorted.length * 0.25)] : 0;
  const q3  = sorted.length >= 4 ? sorted[Math.floor(sorted.length * 0.75)] : 0;
  const iqr = q3 - q1;
  const stdDev = validLoads.length > 1 ? Math.sqrt(validLoads.reduce((s, v) => s + (v - avgLoad) ** 2, 0) / validLoads.length) : 0;
  const cv   = avgLoad > 0 ? (stdDev / avgLoad) * 100 : 0;
  const ramps = monLoads.slice(1).map((v, i) => v - monLoads[i]);
  const maxRampUp   = ramps.length ? Math.max(...ramps) : 0;
  const maxRampDown = ramps.length ? Math.min(...ramps) : 0;
  const pvRatio  = valley > 0 ? peak / valley : null;
  const mape     = monMeta?.mape_live;
  const nowBlk   = (() => { const n = new Date(); return Math.min(95, n.getHours() * 4 + Math.floor(n.getMinutes() / 15)); })();
  const peakIdx  = monLoads.indexOf(peak);
  const peakTime = peakIdx >= 0 ? `${String(Math.floor(peakIdx / 4)).padStart(2,'0')}:${String((peakIdx % 4) * 15).padStart(2,'0')}` : '--';
  const valleyIdx  = monLoads.indexOf(valley);
  const valleyTime = valleyIdx >= 0 ? `${String(Math.floor(valleyIdx / 4)).padStart(2,'0')}:${String((valleyIdx % 4) * 15).padStart(2,'0')}` : '--';

  /* ── ramp profile ── */
  const topRamps = React.useMemo(() => {
    return ramps.map((v, i) => ({ v, t: `${String(Math.floor((i+1)/4)).padStart(2,'0')}:${String(((i+1)%4)*15).padStart(2,'0')}` }))
      .sort((a,b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 5);
  }, [ramps]);

  /* ── error vs baseline ── */
  const errorPairs = monLoads.map((v, i) => ({ a: v, f: monYLoads[i] })).filter(({ a, f }) => a > 100 && f > 0);
  const maeVal  = errorPairs.length ? errorPairs.reduce((s, { a, f }) => s + Math.abs(a - f), 0) / errorPairs.length : null;
  const rmseVal = errorPairs.length ? Math.sqrt(errorPairs.reduce((s, { a, f }) => s + (a - f) ** 2, 0) / errorPairs.length) : null;
  const mapeVal = errorPairs.length ? errorPairs.reduce((s, { a, f }) => s + Math.abs(a - f) / a, 0) / errorPairs.length * 100 : null;
  const biasVal = errorPairs.length ? errorPairs.reduce((s, { a, f }) => s + (f - a), 0) / errorPairs.length : null;

  const fmtN = (v, d = 0) => v == null || !Number.isFinite(v) ? '--' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });

  /* ── distribution buckets ── */
  const binSize = 250;
  const lo  = validLoads.length ? Math.floor(Math.min(...validLoads) / binSize) * binSize : 0;
  const hi  = validLoads.length ? Math.ceil(Math.max(...validLoads) / binSize) * binSize : 0;
  const bins = [];
  for (let b = lo; b < hi; b += binSize) bins.push({ label: `${(b / 1000).toFixed(1)}k`, lo: b, hi: b + binSize, count: validLoads.filter(v => v >= b && v < b + binSize).length });
  const maxBin = bins.length ? Math.max(...bins.map(b => b.count)) : 1;

  /* ── similar days: filter + sort ── */
  const dayOfWeek = (ds) => { try { return new Date(ds + 'T00:00:00').toLocaleDateString('en-IN', { weekday: 'short' }); } catch { return ''; } };
  const simColor  = (s) => s >= 85 ? '#34D399' : s >= 70 ? '#FBBF24' : '#F87171';

  const filteredSim = React.useMemo(() => {
    let rows = rawSim.map((d, i) => ({ ...d, _rank: i }));
    if (simFilter !== 'all') rows = rows.filter(d => (d.category || d.day_type || 'working') === simFilter);
    rows.sort((a, b) => {
      let va, vb;
      if (simSort.col === 'rank')      { va = a._rank; vb = b._rank; }
      else if (simSort.col === 'sim')  { va = (a.similarity_score ?? a.score ?? 0); vb = (b.similarity_score ?? b.score ?? 0); }
      else if (simSort.col === 'date') { va = a.date ?? ''; vb = b.date ?? ''; }
      else if (simSort.col === 'temp') { va = Math.abs(a.temp_diff ?? 0); vb = Math.abs(b.temp_diff ?? 0); }
      else if (simSort.col === 'hum')  { va = Math.abs(a.hum_diff ?? 0); vb = Math.abs(b.hum_diff ?? 0); }
      else { va = 0; vb = 0; }
      return simSort.dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    });
    return rows;
  }, [rawSim, simFilter, simSort]);

  /* ── selected similar day detail ── */
  const selDay = selectedSimRow != null ? rawSim.find(d => d.date === selectedSimRow) : null;

  /* ── stat tabs ── */
  const statGroups = {
    central:  [['Peak Load', fmtN(peak,0), 'MW', '#F07825', `Occurs at ${peakTime}`], ['Valley Load', fmtN(valley,0), 'MW', '#5B9FE4', `Occurs at ${valleyTime}`], ['Mean Load', fmtN(avgLoad,0), 'MW', '#ECEEF3', 'Simple arithmetic mean of all valid blocks'], ['Median Load', fmtN(median,0), 'MW', '#FBBF24', '50th percentile — robust to outliers'], ['Day Energy', fmtN(energy,0), 'MWh', '#ECEEF3', 'Sum of load × 0.25 h across all 96 blocks'], ['Load Factor', fmtN(loadFactor,2), '%', loadFactor>=70?'#34D399':'#FBBF24', 'Mean/Peak — higher = more efficient utilisation']],
    spread:   [['Std Deviation', fmtN(stdDev,0), 'MW', '#C084FC', 'Square root of variance — measures load volatility'], ['IQR (Q3−Q1)', fmtN(iqr,0), 'MW', '#C084FC', 'Middle 50% spread — resistant to extremes'], ['Q1 (25th pct)', fmtN(q1,0), 'MW', '#5B9FE4', 'Bottom quartile load threshold'], ['Q3 (75th pct)', fmtN(q3,0), 'MW', '#C084FC', 'Top quartile load threshold'], ['CV %', fmtN(cv,2), '%', '#C084FC', 'Std Dev / Mean × 100 — normalised volatility'], ['P/V Ratio', pvRatio!=null?pvRatio.toFixed(3):'--', '', '#ECEEF3', 'Peak-to-valley — higher = greater storage opportunity']],
    ramp:     [['Max Ramp Up', fmtN(maxRampUp,0), 'MW/15m', maxRampUp>200?'#F87171':'#FBBF24', 'Largest positive 15-min delta'], ['Max Ramp Down', fmtN(Math.abs(maxRampDown),0), 'MW/15m', '#5B9FE4', 'Largest negative 15-min delta (absolute)'], ['Valid Blocks', validLoads.length, '/ 96', '#ECEEF3', 'Blocks with non-zero load observed'], ['Peak Time', peakTime, '', '#F07825', 'Clock time of peak block'], ['Valley Time', valleyTime, '', '#5B9FE4', 'Clock time of lowest load block'], ['Temp Now', monTemps[nowBlk]!=null?`${monTemps[nowBlk].toFixed(1)}°C`:'--', '', '#FBBF24', 'Ambient temperature at current block']],
    perf:     [['MAPE (meta)', mape!=null?`${mape.toFixed(2)}%`:'--', '', mape!=null&&mape<3?'#34D399':'#F87171', 'Mean Absolute % Error from API metadata'], ['MAPE (calc)', mapeVal!=null?`${mapeVal.toFixed(2)}%`:'--', '', mapeVal!=null&&mapeVal<3?'#34D399':mapeVal!=null&&mapeVal<6?'#FBBF24':'#F87171', 'Computed from actual vs baseline pairs'], ['MAE', fmtN(maeVal,0), 'MW', '#5B9FE4', 'Mean Absolute Error in MW'], ['RMSE', fmtN(rmseVal,0), 'MW', '#C084FC', 'Root Mean Squared Error — penalises large errors'], ['Bias', biasVal!=null?`${biasVal>0?'+':''}${fmtN(biasVal,0)}`:'--', 'MW', biasVal!=null&&Math.abs(biasVal)<50?'#34D399':'#FBBF24', 'Signed mean error — positive = over-forecast'], ['Error Pairs', errorPairs.length, 'blocks', '#ECEEF3', 'Blocks used for error calculation']],
  };

  /* ── KPI card drill-down detail ── */
  const kpiDetail = {
    'Peak Load':     { title: 'Peak Load Deep-Dive', lines: [`Observed at ${peakTime}`, `Range spread: ${fmtN(peak-valley,0)} MW`, `Load Factor: ${fmtN(loadFactor,1)}%`, `Top 5 ramp events near peak shown in Ramp tab`] },
    'Valley Load':   { title: 'Valley Analysis', lines: [`Observed at ${valleyTime}`, `Night base demand: ${fmtN(valley,0)} MW`, `P/V Ratio: ${pvRatio!=null?pvRatio.toFixed(2):'--'}`, `Higher P/V ratio → more storage opportunity`] },
    'Day Energy':    { title: 'Energy Balance', lines: [`Total: ${fmtN(energy,0)} MWh`, `Average load: ${fmtN(avgLoad,0)} MW`, `Load factor: ${fmtN(loadFactor,1)}%`, `Valid blocks contributing: ${validLoads.length}/96`] },
    'Load Factor':   { title: 'Load Factor Explained', lines: [`Formula: Mean Load / Peak Load × 100`, `Mean: ${fmtN(avgLoad,0)} MW  Peak: ${fmtN(peak,0)} MW`, `Result: ${fmtN(loadFactor,1)}%`, loadFactor>=70?'✓ Good utilisation (≥70%)':'⚠ Low utilisation (<70%)'] },
    'Std Deviation': { title: 'Load Volatility', lines: [`Std Dev: ${fmtN(stdDev,0)} MW`, `CV: ${fmtN(cv,1)}% (Std/Mean)`, `IQR: ${fmtN(iqr,0)} MW`, cv>15?'⚠ High volatility — consider storage dispatch':'✓ Stable load profile'] },
    'Max Ramp Up':   { title: 'Ramp Events', lines: [`Steepest up: +${fmtN(maxRampUp,0)} MW/15m`, `Steepest down: −${fmtN(Math.abs(maxRampDown),0)} MW/15m`, `Top events listed in Ramp Stats tab`, maxRampUp>200?'⚠ High ramp — hydro / gas response needed':'✓ Within normal ramp range'] },
  };

  /* ── error explanations ── */
  const errorInfo = {
    MAPE:  'Mean Absolute % Error. Industry threshold: <3% excellent, <6% acceptable, >6% review needed.',
    MAE:   'Mean Absolute Error in MW. Direct measure of average miss magnitude. No penalty weighting.',
    RMSE:  'Root Mean Squared Error. Penalises large errors more heavily than MAE — critical for peak blocks.',
    Bias:  'Signed mean error. Positive = model over-forecasts. Negative = model under-forecasts. Target: near zero.',
  };

  /* ── styles ── */
  const cardS = { background: 'linear-gradient(180deg,rgba(26,25,30,.98),rgba(20,20,24,.96))', borderRadius: 14, border: '1px solid #2A292F', overflow: 'hidden', display: 'flex', flexDirection: 'column' };
  const headS = { padding: '12px 16px', borderBottom: '1px solid #2A292F', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 };
  const titleS = { fontSize: 12, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: '#B8BDCC', fontFamily: "'IBM Plex Mono',monospace" };
  const badge  = (c) => ({ fontSize: 11, fontWeight: 700, padding: '3px 9px', borderRadius: 999, background: `${c}18`, color: c, border: `1px solid ${c}33` });
  const tabBtn = (active) => ({ fontSize: 11, fontWeight: 600, letterSpacing: 0.5, padding: '5px 12px', borderRadius: 999, border: `1px solid ${active ? '#F07825' : '#2A292F'}`, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#8A90A6', cursor: 'pointer', fontFamily: 'inherit', textTransform: 'uppercase', transition: 'all 0.12s' });
  const sortTh = (col) => ({ padding: '8px 10px', textAlign: 'left', fontSize: 11, letterSpacing: 0.5, textTransform: 'uppercase', color: simSort.col === col ? '#F07825' : '#8A90A6', borderBottom: '1px solid #2A292F', fontWeight: 600, whiteSpace: 'nowrap', cursor: 'pointer', userSelect: 'none', transition: 'color 0.12s' });
  const onSort = (col) => setSimSort(s => ({ col, dir: s.col === col && s.dir === 'asc' ? 'desc' : 'asc' }));
  const sortIcon = (col) => simSort.col === col ? (simSort.dir === 'asc' ? ' ↑' : ' ↓') : '';

  return (
    <VpPageShell className="monitor-page">
      {/* ── HEADER ── */}
      <div style={{ display: 'none', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: '#F0F2F8' }}>Similar Days &amp; Statistics</div>
          <div style={{ fontSize: 12, color: '#8A90A6', marginTop: 3 }}>Click any KPI card · Sort table columns · Filter by day type · Click a bin to narrow similar days</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {mape != null && <span style={badge('#F07825')}>{`MAPE ${mape.toFixed(1)}%`}</span>}
          <span style={badge('#34D399')}>{effectiveDate || new Date().toISOString().slice(0,10)}</span>
        </div>
      </div>

      {/* ── KPI STRIP (clickable) ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, flexShrink: 0 }}>
        {[
          { label: 'Peak Load',     value: fmtN(peak),           unit: 'MW',     color: '#F07825', sub: `at ${peakTime}` },
          { label: 'Valley Load',   value: fmtN(valley),         unit: 'MW',     color: '#5B9FE4', sub: `at ${valleyTime}` },
          { label: 'Day Energy',    value: fmtN(energy,0),       unit: 'MWh',    color: '#ECEEF3', sub: `Avg ${fmtN(avgLoad,0)} MW` },
          { label: 'Load Factor',   value: fmtN(loadFactor,1),   unit: '%',      color: loadFactor>=70?'#34D399':'#FBBF24', sub: `P/V ${pvRatio!=null?pvRatio.toFixed(2):'--'}` },
          { label: 'Std Deviation', value: fmtN(stdDev,0),       unit: 'MW',     color: '#C084FC', sub: `CV ${fmtN(cv,1)}%` },
          { label: 'Max Ramp Up',   value: fmtN(maxRampUp,0),    unit: 'MW/15m', color: maxRampUp>200?'#F87171':'#FBBF24', sub: `Down ${fmtN(Math.abs(maxRampDown),0)}` },
        ].map(k => {
          const isActive = activeKpi === k.label;
          return (
            <button key={k.label} onClick={() => setActiveKpi(isActive ? null : k.label)}
              style={{ ...cardS, padding: '14px 14px 12px', cursor: 'pointer', textAlign: 'left', border: `1px solid ${isActive ? k.color+'66' : '#2A292F'}`, boxShadow: isActive ? `0 0 16px ${k.color}18` : 'none', transition: 'all 0.15s' }}>
              <div style={{ fontSize: 11, letterSpacing: 0.6, textTransform: 'uppercase', color: isActive ? k.color : '#8A90A6', marginBottom: 6 }}>{k.label}</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, marginBottom: 4 }}>
                <span style={{ fontSize: 22, fontWeight: 700, lineHeight: 1, color: k.color }}>{k.value}</span>
                <span style={{ fontSize: 12, color: '#8A90A6' }}>{k.unit}</span>
              </div>
              <div style={{ fontSize: 12, color: isActive ? k.color+'cc' : '#8A90A6' }}>{k.sub}</div>
              {isActive && <div style={{ marginTop: 6, width: 20, height: 2, borderRadius: 1, background: k.color }} />}
            </button>
          );
        })}
      </div>

      {/* ── KPI DRILL-DOWN PANEL ── */}
      {activeKpi && kpiDetail[activeKpi] && (
        <div style={{ ...cardS, flexShrink: 0, border: `1px solid ${['#F07825','#5B9FE4','#ECEEF3','#34D399','#C084FC','#FBBF24'].find((_,i)=>['Peak Load','Valley Load','Day Energy','Load Factor','Std Deviation','Max Ramp Up'][i]===activeKpi)||'#2A292F'}33` }}>
          <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 24 }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase', color: '#8A90A6', marginBottom: 8 }}>{kpiDetail[activeKpi].title}</div>
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                {kpiDetail[activeKpi].lines.map((l, i) => (
                  <div key={i} style={{ fontSize: 13, color: l.startsWith('⚠') ? '#F87171' : l.startsWith('✓') ? '#34D399' : '#F0F2F8', lineHeight: 1.6 }}>{l}</div>
                ))}
              </div>
            </div>
            <button onClick={() => setActiveKpi(null)} style={{ fontSize: 14, background: 'none', border: 'none', color: '#4A4D5E', cursor: 'pointer', padding: '0 4px', lineHeight: 1 }}>×</button>
          </div>
        </div>
      )}

      {/* ── MAIN GRID ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 14, flex: 1, minHeight: 400 }}>

        {/* LEFT: accuracy monitor */}
        <Suspense fallback={<ViewLoading label="Loading Accuracy Monitor..." />}>
          <AccuracyMonitorPage
            liveData={live}
            selectedRegion={selectedRegion}
            selectedDate={effectiveDate}
          />
        </Suspense>

        {/* RIGHT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>

          {/* Load Distribution (bins are clickable) */}
          <div style={{ ...cardS, flex: 1, minHeight: 0 }}>
            <div style={headS}>
              <div style={titleS}>Load Distribution</div>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>Click a bin to filter similar days</span>
            </div>
            <div style={{ flex: 1, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 4, overflow: 'auto' }}>
              {bins.length === 0 ? (
                <div style={{ color: '#8A90A6', fontSize: 12, textAlign: 'center', padding: '20px 0' }}>No data</div>
              ) : bins.map((b, i) => {
                const isActiveBin = activeBin === i;
                return (
                  <button key={b.label} onClick={() => setActiveBin(isActiveBin ? null : i)}
                    style={{ display: 'flex', alignItems: 'center', gap: 10, background: isActiveBin ? '#F0782508' : 'transparent', border: `1px solid ${isActiveBin ? '#F0782540' : 'transparent'}`, borderRadius: 6, padding: '3px 4px', cursor: 'pointer', transition: 'all 0.12s' }}>
                    <div style={{ fontSize: 12, color: isActiveBin ? '#F07825' : '#8A90A6', width: 38, textAlign: 'right', flexShrink: 0 }}>{b.label}</div>
                    <div style={{ flex: 1, height: 14, background: '#1A191E', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(b.count / maxBin) * 100}%`, background: isActiveBin ? '#F07825' : 'linear-gradient(90deg,#F0782550,#F07825)', borderRadius: 3 }} />
                    </div>
                    <div style={{ fontSize: 12, color: isActiveBin ? '#F07825' : '#B8BDCC', width: 24, textAlign: 'right', flexShrink: 0, fontWeight: isActiveBin ? 700 : 400 }}>{b.count}</div>
                  </button>
                );
              })}
              {validLoads.length > 0 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, padding: '8px 4px 0', borderTop: '1px solid #2A292F' }}>
                  {[['Q1', q1, '#5B9FE4'], ['Median', median, '#FBBF24'], ['Q3', q3, '#C084FC']].map(([l, v, c]) => (
                    <div key={l} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#8A90A6' }}>{l}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: c }}>{fmtN(v, 0)}</div>
                      <div style={{ fontSize: 11, color: '#8A90A6' }}>MW</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Error Diagnostics (click for explanation) */}
          <div style={cardS}>
            <div style={headS}>
              <div style={titleS}>Error Diagnostics</div>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>Click to explain</span>
            </div>
            <div style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[
                { label: 'MAPE', value: mapeVal, unit: '%', color: mapeVal!=null&&mapeVal<3?'#34D399':mapeVal!=null&&mapeVal<6?'#FBBF24':'#F87171', fmt: v => v.toFixed(2) },
                { label: 'MAE',  value: maeVal,  unit: 'MW', color: '#5B9FE4', fmt: v => fmtN(v, 0) },
                { label: 'RMSE', value: rmseVal, unit: 'MW', color: '#C084FC', fmt: v => fmtN(v, 0) },
                { label: 'Bias', value: biasVal, unit: 'MW', color: biasVal!=null&&Math.abs(biasVal)<50?'#34D399':'#FBBF24', fmt: v => `${v>0?'+':''}${fmtN(v,0)}` },
              ].map(e => {
                const isExpanded = activeError === e.label;
                return (
                  <div key={e.label}>
                    <button onClick={() => setActiveError(isExpanded ? null : e.label)}
                      style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: isExpanded ? '#ffffff04' : 'transparent', border: `1px solid ${isExpanded ? '#2A292F' : 'transparent'}`, borderRadius: 8, padding: '8px 10px', cursor: 'pointer', transition: 'all 0.12s' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 3, height: 14, borderRadius: 2, background: e.value != null ? e.color : '#2A292F' }} />
                        <span style={{ fontSize: 12, color: isExpanded ? '#F0F2F8' : '#8A90A6', letterSpacing: 0.4, fontFamily: 'inherit' }}>{e.label}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                        <span style={{ fontSize: 15, fontWeight: 700, color: e.value!=null ? e.color : '#555A6E', fontFamily: 'inherit' }}>{e.value!=null ? e.fmt(e.value) : '--'}</span>
                        <span style={{ fontSize: 11, color: '#8A90A6', fontFamily: 'inherit' }}>{e.unit}</span>
                        <span style={{ fontSize: 11, color: '#8A90A6', marginLeft: 4, fontFamily: 'inherit' }}>{isExpanded ? '▲' : '▾'}</span>
                      </div>
                    </button>
                    {isExpanded && (
                      <div style={{ padding: '6px 10px 8px 21px', fontSize: 12, color: '#B8BDCC', lineHeight: 1.7, borderBottom: '1px solid #2A292F40' }}>
                        {errorInfo[e.label]}
                      </div>
                    )}
                  </div>
                );
              })}
              {errorPairs.length === 0 && <div style={{ fontSize: 12, color: '#8A90A6', textAlign: 'center', padding: '6px 0' }}>Baseline required for error metrics</div>}
            </div>
          </div>

        </div>
      </div>

      {/* ── TABBED STATISTICS ── */}
      <div style={cardS}>
        <div style={headS}>
          <div style={titleS}>Descriptive Statistics</div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[['central','Central Tendency'],['spread','Spread'],['ramp','Ramp & Shape'],['perf','Performance']].map(([id, label]) => (
              <button key={id} onClick={() => setStatTab(id)} style={tabBtn(statTab === id)}>{label}</button>
            ))}
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 0 }}>
          {(statGroups[statTab] || []).map(([label, val, unit, color, hint], i) => (
            <div key={label} style={{ padding: '14px 16px', borderRight: i % 6 !== 5 ? '1px solid #2A292F22' : 'none', position: 'relative', overflow: 'hidden' }}>
              <div style={{ fontSize: 11, color: '#8A90A6', letterSpacing: 0.5, textTransform: 'uppercase', marginBottom: 6 }}>{label}</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, marginBottom: 5 }}>
                <span style={{ fontSize: 18, fontWeight: 700, color }}>{val}</span>
                {unit && <span style={{ fontSize: 11, color: '#8A90A6' }}>{unit}</span>}
              </div>
              <div style={{ fontSize: 11, color: '#8A90A6', lineHeight: 1.5 }}>{hint}</div>
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, background: `${color}20` }} />
            </div>
          ))}
        </div>

        {/* Ramp top-events (only in ramp tab) */}
        {statTab === 'ramp' && topRamps.length > 0 && (
          <div style={{ padding: '10px 16px 14px', borderTop: '1px solid #2A292F' }}>
            <div style={{ fontSize: 11, color: '#8A90A6', letterSpacing: 0.5, textTransform: 'uppercase', marginBottom: 8 }}>Top 5 Ramp Events</div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {topRamps.map((r, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 8, background: r.v > 0 ? '#F8717118' : '#34D39918', border: `1px solid ${r.v>0?'#F8717133':'#34D39933'}` }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: r.v>0?'#F87171':'#34D399' }}>{r.v>0?'+':''}{fmtN(r.v,0)}</span>
                  <span style={{ fontSize: 12, color: '#8A90A6' }}>MW at {r.t}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

    </VpPageShell>
  );
}

export default function App({ authUser, onLogout, onHome }) {
  const weatherComponentSliders = useSimulatorStore((s) => s.weatherComponentSliders);
  const setWeatherComponentSlider = useSimulatorStore((s) => s.setWeatherComponentSlider);
  const loadFromFileData = useSimulatorStore((s) => s.loadFromFileData);

  const defaultWeatherAdj = { temp: 0, humidity: 0, precip: 0, mode: 'all', start: 60, end: 72, block: 72 };
  const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
  const sanitizeWeatherAdj = (adj) => {
    const start = clamp(Number(adj?.start) || 1, 1, 96);
    const end = clamp(Number(adj?.end) || 1, 1, 96);
    const modeRaw = String(adj?.mode || 'all');
    const mode = modeRaw === 'range' || modeRaw === 'single' ? modeRaw : 'all';
    return {
      ...adj,
      mode,
      temp: clamp(Number(adj?.temp) || 0, -10, 10),
      humidity: clamp(Number(adj?.humidity) || 0, -40, 40),
      precip: clamp(Number(adj?.precip) || 0, 0, 100),
      start: Math.min(start, end),
      end: Math.max(start, end),
      block: clamp(Number(adj?.block) || 1, 1, 96),
    };
  };

  const [active, setActive] = useState('load_analysis');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem('vp-sidebar-collapsed') === '1'; }
    catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem('vp-sidebar-collapsed', sidebarCollapsed ? '1' : '0'); }
    catch { /* noop */ }
  }, [sidebarCollapsed]);
  const [config, setConfig] = useState(null);
  const [settings, setSettings] = useState(null);
  const [date, setDate] = useState('');
  const [baselineDays, setBaselineDays] = useState(7);
  const [loading, setLoading] = useState(false); // kept for fetchDayAhead/fetchLive/fetchAnalysis manual refreshes
  const [initPhase, setInitPhase] = useState('idle'); // 'idle'|'config'|'data'|'ready'|'training'|'done'
  const [trainingProgress, setTrainingProgress] = useState([]);
  const forecastWorkerRef = useRef(null);
  const [activeHorizon, setActiveHorizon] = useState('t1');
  const [dayAhead, setDayAhead] = useState(null);
  const [dayAheadT2, setDayAheadT2] = useState(null);
  const [live, setLive] = useState(null);
  const [liveT2, setLiveT2] = useState(null);
  const [liveT2Loading, setLiveT2Loading] = useState(false);
  const [actualBlocks, setActualBlocks] = useState(0);
  const [analysis, setAnalysis] = useState(null);

  const activeLive = activeHorizon === 't2' ? liveT2 : live;
  const activeDayAhead = activeHorizon === 't2' ? dayAheadT2 : dayAhead;
  const [toasts, setToasts] = useState([]);
  const [selectedBlock, setSelectedBlock] = useState(72);
  const [selectedKpiGroup, setSelectedKpiGroup] = useState('forecast_accuracy');
  const [selectedKpiImpact, setSelectedKpiImpact] = useState('temperature_impact_kw');
  const [weatherAdj, setWeatherAdj] = useState(defaultWeatherAdj);
  const [appliedWeatherAdj, setAppliedWeatherAdj] = useState(defaultWeatherAdj);
  const [calendarConfig, setCalendarConfig] = useState('weekday');
  const [activeLiveInsight, setActiveLiveInsight] = useState('similar_day_matches');
  const [liveInsightsOpen, setLiveInsightsOpen] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [showRegionPrompt, setShowRegionPrompt] = useState(true);
  const _today = new Date().toISOString().slice(0, 10);
  const _defaultFrom = new Date(Date.now() - 120 * 86400000).toISOString().slice(0, 10);
  const [regionDateRange, setRegionDateRange] = useState({ from: _defaultFrom, to: _today });
  const [viewDataOpen, setViewDataOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(true);
  const [pipelineDate, setPipelineDate] = useState(null);
  const [pipelineJobId, setPipelineJobId] = useState(null);

  // 3-Tab Load Analysis States
  const [loadTab, setLoadTab] = useState('benchmark');
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [multiDaySeries, setMultiDaySeries] = useState({});
  const [momentumChange, setMomentumChange] = useState([]);
  const [multiSelectedDates, setMultiSelectedDates] = useState([]);

  // Optimizer tabs
  const [optimizerTab, setOptimizerTab] = useState('window');

  const weatherRef = useRef(null);

  useEffect(() => {
    const activeLabel = NAV_ITEMS.find((item) => item.key === active)?.label
      || (active === 'settings' ? 'Settings' : 'Grid Intelligence');
    const regionLabel = selectedRegion ? titleize(selectedRegion) : 'Grid Intelligence';
    document.title = `${APP_TITLE} | ${activeLabel} - ${regionLabel}`;
  }, [active, selectedRegion]);

  // Sync store sliders into local adj for display/calculation
  // We keep mode/start/end/block in local state (weatherAdj), but override temp/humidity/precip from store.
  const safeWeatherAdj = useMemo(() => {
    const base = sanitizeWeatherAdj(weatherAdj);
    return {
      ...base,
      temp: weatherComponentSliders.temperature,
      humidity: weatherComponentSliders.humidity,
      precip: weatherComponentSliders.precipitation
    };
  }, [weatherAdj, weatherComponentSliders]);

  const selectionBlocksFromAdj = (adj, totalBlocks = 96) => {
    const total = clamp(Number(totalBlocks) || 96, 1, 96);
    // Use the safe/merged adj
    const safe = adj;
    if (safe.mode === 'single') return [safe.block];
    if (safe.mode === 'range') {
      const blocks = [];
      for (let b = safe.start; b <= safe.end; b += 1) blocks.push(b);
      return blocks;
    }
    return Array.from({ length: total }, (_, i) => i + 1);
  };

  const effectiveDate = date || config?.latest_date || dayAhead?.metadata?.effective_date || '';
  const t2Date = liveT2?.t2_date || '';

  const detectedSeason = useMemo(() => detectSeason(effectiveDate), [effectiveDate]);
  const liveEffectiveDate = config?.partial_latest_date || effectiveDate;
  const inferredDayType = dayType(effectiveDate).toLowerCase();
  const inferredCalendarConfig = useMemo(() => {
    if (!effectiveDate) return null;
    const current = new Date(`${effectiveDate}T00:00:00`);
    if (Number.isNaN(current.getTime())) return null;
    const next = new Date(current);
    next.setDate(next.getDate() + 1);
    const currIsWeekend = current.getDay() === 0 || current.getDay() === 6;
    const nextIsWeekend = next.getDay() === 0 || next.getDay() === 6;
    if (currIsWeekend && !nextIsWeekend) return 'weekend_to_weekday';
    if (!currIsWeekend && nextIsWeekend) return 'weekday_to_weekend';
    return currIsWeekend ? 'weekend' : 'weekday';
  }, [effectiveDate]);
  const inferredBaseConfig = inferredDayType || (inferredCalendarConfig?.includes('to')
    ? (inferredCalendarConfig.includes('weekday') ? 'weekday' : 'weekend')
    : inferredCalendarConfig);
  const inferredCalendarLabel = inferredBaseConfig
    ? inferredBaseConfig.replace(/_/g, ' ')
    : inferredDayType;
  const inferredTransitionConfig = inferredCalendarConfig?.includes('to') ? inferredCalendarConfig : null;
  const nextDayInfo = useMemo(() => {
    if (!effectiveDate) return { date: '--', dayType: '--', label: '--' };
    const current = new Date(`${effectiveDate}T00:00:00`);
    if (Number.isNaN(current.getTime())) return { date: '--', dayType: '--', label: '--' };
    const next = new Date(current);
    next.setDate(next.getDate() + 1);
    const y = next.getFullYear();
    const m = String(next.getMonth() + 1).padStart(2, '0');
    const d = String(next.getDate()).padStart(2, '0');
    const nextDate = `${y}-${m}-${d}`;
    const nextDayType = dayType(nextDate);
    return {
      date: nextDate,
      dayType: nextDayType,
      label: `${nextDate} (${nextDayType})`
    };
  }, [effectiveDate]);

  // ── Forecast Web Worker bootstrap ─────────────────────────────────────────
  const _startForecastWorker = (jobId) => {
    // Terminate any previous worker
    if (forecastWorkerRef.current) {
      forecastWorkerRef.current.postMessage({ type: 'stop' });
      forecastWorkerRef.current.terminate();
    }
    const worker = new Worker('/forecastWorker.js');
    forecastWorkerRef.current = worker;

    worker.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'progress') {
        setTrainingProgress(e.data.events || []);
      } else if (type === 'result') {
        const resultData = e.data.data;
        if (resultData && !resultData.error) {
          setDayAhead(resultData);
          // Simulator blocks: pipeline is now cached → fast
          const dateVal = resultData?.metadata?.effective_date || date;
          loadFromFileData(dateVal || date, baselineDays);
        }
        setInitPhase('done');
        setTrainingProgress([]);
        worker.terminate();
        forecastWorkerRef.current = null;
      } else if (type === 'error') {
        console.warn('Forecast worker error:', e.data.message);
        pushToast('error', `Model training failed: ${e.data.message}`);
        setInitPhase('ready'); // stay ready, forecast section shows retry
        setTrainingProgress([]);
        worker.terminate();
        forecastWorkerRef.current = null;
      }
    };

    // Poll the FastAPI training service; normal app data stays on Django.
    const workerApiBase = TRAINING_API_BASE;
    worker.postMessage({
      type: 'start',
      jobId,
      apiBase: workerApiBase,
      token: useAuthStore.getState().token,
    });
  };

  // ── Phased initialisation ──────────────────────────────────────────────────
  const initializeData = async (regionToUse) => {
    setInitPhase('config');
    setTrainingProgress([]);

    try {
      // ── Phase 1: Config + settings (blocking — need the date) ─────────────
      const [cfgRes, settingsRes] = await Promise.all([
        axios.get(API_URL('/v2/config'), { timeout: 15000 }),
        axios.get(API_URL('/v2/settings'), { timeout: 15000 }),
      ]);

      setAvailableRegions(cfgRes.data.available_regions || []);
      setConfig(cfgRes.data);
      setSettings(settingsRes.data);

      const d = cfgRes.data.default_date || cfgRes.data.latest_date || cfgRes.data.partial_latest_date || '';
      const bl = cfgRes.data.best_baseline_window || 7;
      if (d) setDate(d);
      setBaselineDays(bl);
      if (!d) { setInitPhase('ready'); return; }

      const prevDay = new Date(new Date(d).getTime() - 86400000).toISOString().slice(0, 10);

      // ── Phase 2: Fast historical/analytical data (no T+2 here — it's slow) ─
      setInitPhase('data');
      // momentum is slow — fire-and-forget so it never blocks init
      axios.post(API_URL('/v2/load_change'), { date1: d, date2: prevDay, dates: [] }, { timeout: 60000 })
        .then(r => setMomentumChange(r.data))
        .catch(e => console.warn('momentum failed:', e?.message));

      await Promise.allSettled([
        axios.post(API_URL('/v2/load_benchmarks'), { date: d }, { timeout: 15000 })
          .then(r => setBenchmarkData(r.data))
          .catch(e => console.warn('benchmarks failed:', e?.message)),

        axios.post(API_URL('/v2/analysis'), { date: d, region: regionToUse }, { timeout: 15000 })
          .then(r => { if (r.data && !r.data.error) setAnalysis(r.data); })
          .catch(e => console.warn('analysis failed:', e?.message)),
      ]);

      // ── Phase 3: UI ready — start forecast training immediately ───────────
      setInitPhase('ready');
      setInitPhase('training');
      try {
        const res = await axios.post(TRAINING_API_URL('/v2/forecast/submit'), {
          date: d, region: regionToUse, baseline_days: bl,
          calendar_config: calendarConfig,
        }, { timeout: 10000 });
        const { job_id } = res.data;
        setPipelineJobId(job_id);
        _startForecastWorker(job_id); // non-blocking — worker polls in background
      } catch (e) {
        console.warn('forecast submit failed:', e?.message);
        setInitPhase('ready');
      }

      // ── Phase 4 (background): T+2 forecast — fires after training is kicked ─
      // Runs independently so it never blocks the PipelineProgress tooltip.
      // liveT2 already contains MySQL-fetched weather for T+2 date — no need
      // to call /v2/dayahead again for a future date (returns zero deltas).
      setLiveT2Loading(true);
      axios.post(TRAINING_API_URL('/v2/forecast/t2'), {
        date: d, region: regionToUse, baseline_days: bl,
      }, { timeout: 0 })
        .then(r => {
          setLiveT2(r.data);
          setDayAheadT2(r.data);
          setLiveT2Loading(false);
        })
        .catch(e => {
          console.warn('T+2 background fetch failed:', e?.message);
          setLiveT2Loading(false);
        });

    } catch (e) {
      console.error('Initialize failed:', e);
      setInitPhase('ready');
      // Fallback: config-only, then trigger forecast directly
      try {
        const cfgRes = await axios.get(API_URL('/v2/config'), { timeout: 15000 });
        setConfig(cfgRes.data);
        const d = cfgRes.data.default_date || cfgRes.data.latest_date || '';
        if (d) {
          setDate(d);
          fetchDayAheadForInit(d, cfgRes.data.best_baseline_window || 7, regionToUse);
        }
      } catch { setConfig(null); }
    }
  };

  const [availableRegions, setAvailableRegions] = useState(['odisha', 'rajasthan', 'haryana', 'chhattisgarh']);

  // Startup: only fetch config if we have a region
  useEffect(() => {
    // If we already have a region (e.g. from localstorage later), we could auto-init
    // For now, wait for selection
  }, []);

  const handleRegionConfirm = async () => {
    if (selectedRegion) {
      setShowRegionPrompt(false);
      setInitPhase('config');
      try {
        await axios.post(TRAINING_API_URL('/switch-region'), {
          region: selectedRegion,
          from_date: regionDateRange.from || undefined,
          to_date: regionDateRange.to || undefined,
        }, { timeout: 60000 });
      } catch (e) {
        console.warn('switch-region failed, using existing engine data:', e?.message);
      }
      initializeData(selectedRegion);
    }
  };

  const fetchDayAheadForInit = async (targetDate, targetBaseline, targetRegion) => {
    setLoading(true);
    try {
      const res = await axios.post(TRAINING_API_URL('/v2/dayahead'), {
        date: targetDate,
        baseline_days: targetBaseline,
        calendar_config: calendarConfig,
        region: targetRegion
      }, { timeout: FORECAST_REQUEST_TIMEOUT_MS });
      setDayAhead(res.data);
    } catch (e) {
      console.error('Initial fetch failed:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const handler = (e) => {
      if (e.key.toLowerCase() === 'g') fetchDayAhead();
      if (e.key.toLowerCase() === 'w' && weatherRef.current) {
        weatherRef.current.scrollIntoView({ behavior: 'smooth' });
      }
      if (e.key === 'Escape') {
        setLiveInsightsOpen(false);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [date, baselineDays]);

  useEffect(() => {
    if (inferredTransitionConfig) {
      setCalendarConfig(inferredTransitionConfig);
    } else if (inferredBaseConfig) {
      setCalendarConfig(inferredBaseConfig);
    }
  }, [inferredTransitionConfig, inferredBaseConfig]);

  useEffect(() => {
    if (active !== 'live') {
      setLiveInsightsOpen(false);
    }
  }, [active]);

  const pushToast = (tone, message) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, tone, message }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
  };

  const fetchLoadAnalysis = async () => {
    if (!effectiveDate) return;
    try {
      if (loadTab === 'benchmark') {
        // Skip if already precomputed for this date
        if (benchmarkData?.dates?.today === effectiveDate) return;
        const res = await axios.post(API_URL('/v2/load_benchmarks'), { date: effectiveDate });
        setBenchmarkData(res.data);
      } else if (loadTab === 'momentum') {
        // Skip only if momentum data already loaded for this exact date
        if (momentumChange?._forDate === effectiveDate) return;
        const dt = new Date(effectiveDate);
        // Build last 30 days as dates list to avoid fetching entire dataset
        const dates30 = [];
        for (let i = 30; i >= 0; i--) {
          const d = new Date(effectiveDate);
          d.setDate(d.getDate() - i);
          dates30.push(d.toISOString().split('T')[0]);
        }
        const res = await axios.post(API_URL('/v2/load_change'), {
          date1: effectiveDate,
          date2: dates30[dates30.length - 2],
          dates: dates30
        });
        setMomentumChange({ ...res.data, _forDate: effectiveDate });
      } else if (loadTab === 'comparison' && multiSelectedDates.length > 0) {
        const res = await axios.post(API_URL('/v2/load_series'), { dates: multiSelectedDates });
        setMultiDaySeries(res.data.series);
      }
    } catch (e) {
      console.error('Failed to fetch load analysis:', e);
    }
  };

  useEffect(() => {
    if (active === 'load_analysis') {
      fetchLoadAnalysis();
    }
  }, [active, loadTab, effectiveDate, multiSelectedDates]);

  const fetchDayAhead = async () => {
    if (!effectiveDate) return;
    setInitPhase('training');
    setTrainingProgress([]);
    try {
      const res = await axios.post(TRAINING_API_URL('/v2/forecast/submit'), {
        date: effectiveDate,
        baseline_days: baselineDays,
        calendar_config: calendarConfig,
        region: selectedRegion,
      }, { timeout: 10000 });
      const { job_id } = res.data;
      setPipelineJobId(job_id);
      _startForecastWorker(job_id);
    } catch (e) {
      // Fallback: direct synchronous call (e.g. if submit endpoint fails)
      setLoading(true);
      try {
        const res = await axios.post(TRAINING_API_URL('/v2/dayahead'), {
          date: effectiveDate,
          baseline_days: baselineDays,
          calendar_config: calendarConfig,
          region: selectedRegion,
        }, { timeout: FORECAST_REQUEST_TIMEOUT_MS });
        setDayAhead(res.data);
        loadFromFileData(effectiveDate, baselineDays);
        pushToast('success', 'Forecast generated');
        setInitPhase('done');
      } catch (e2) {
        const msg = e2?.response?.data?.detail || e2?.message || 'Failed to load forecast';
        pushToast('error', `Forecast error: ${msg}`);
        setInitPhase('ready');
      } finally {
        setLoading(false);
      }
    }
  };

  const fetchLive = async () => {
    if (!selectedRegion) return; // never fire without a confirmed region
    setLoading(true); // live refresh is user-initiated; brief overlay is acceptable
    try {
      const payload = {
        date: liveEffectiveDate || undefined,
        calendar_config: calendarConfig,
        actual_blocks: actualBlocks || (live?.metadata?.actual_blocks ?? undefined),
        region: selectedRegion,
      };
      const [res] = await Promise.allSettled([
        axios.post(TRAINING_API_URL('/v2/live'), payload, { timeout: 180000 }),
      ]);
      if (res.status === 'fulfilled') {
        setLive(res.value.data);
        const availableBlocks = res.value.data?.metadata?.actual_blocks;
        if (Number.isFinite(availableBlocks) && actualBlocks === 0) {
          setActualBlocks(availableBlocks);
        }
        pushToast('info', 'Short-term forecast refreshed');
      }
    } finally {
      setLoading(false);
    }
  };

  const reloadData = async () => {
    setLoading(true);
    try {
      const res = await axios.post(API_URL('/v2/reload'));
      pushToast('success', `Data reloaded (${res.data?.rows || 0} rows)`);
      const cfg = await axios.get(API_URL('/v2/config'));
      setConfig(cfg.data);
      setDate(cfg.data.default_date || cfg.data.latest_date || cfg.data.partial_latest_date || '');
    } catch (e) {
      pushToast('error', 'Failed to reload data');
    } finally {
      setLoading(false);
    }
  };
  const fetchAnalysis = async () => {
    // Skip if already loaded for this date
    if (analysis?.date === effectiveDate) return;
    setLoading(true);
    try {
      const needsDayAhead = !dayAhead || dayAhead?.metadata?.effective_date !== effectiveDate;
      const [analysisRes, dayAheadRes] = await Promise.all([
        axios.post(API_URL('/v2/analysis'), {
          date: effectiveDate,
          region: selectedRegion
        }, { timeout: 15000 }),
        needsDayAhead
          ? axios.post(TRAINING_API_URL('/v2/dayahead'), {
            date: effectiveDate,
            baseline_days: baselineDays,
            calendar_config: calendarConfig
          }, { timeout: FORECAST_REQUEST_TIMEOUT_MS })
          : Promise.resolve(null)
      ]);
      setAnalysis(analysisRes.data);
      if (dayAheadRes?.data) {
        setDayAhead(dayAheadRes.data);
      }
    } catch (e) {
      const status = e?.response?.status;
      const msg = e?.response?.data?.detail || e?.message || 'Failed to load analysis';
      pushToast('error', `Analysis error (${status || 'network'}): ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  // Optimize fetching: Prevent duplicate reloads on tab switch
  useEffect(() => {
    // Guard: never fire during initialization — initPhase must be in dep array so the
    // closure always reads the current value (prevents stale-closure double-submission).
    if (initPhase === 'config' || initPhase === 'data' || initPhase === 'training') return;
    const isDayAheadActive = ['load_analysis', 'weather_analysis', 'simulator', 'analysis'].includes(active);
    if (isDayAheadActive && date) {
      const alreadyLoaded = dayAhead && dayAhead.metadata?.effective_date === date;
      if (!alreadyLoaded) {
        fetchDayAhead();
      }
    }
  }, [active, date, baselineDays, initPhase]);

  useEffect(() => {
    // Don't fire during initialization — initializeData handles the first load.
    // Also require a confirmed region so we never default to 'punjab'.
    if (
      (active === 'forecast' || active === 'load_analysis') &&
      !live &&
      selectedRegion &&
      initPhase === 'done'
    ) {
      fetchLive();
    }
  }, [active, selectedRegion, initPhase]);

  const applyWeatherAdjustment = (
    source,
    weatherImpact = null,
    featureDeltas = null,
    adjustment = appliedWeatherAdj
  ) => {
    if (!source?.length) return [];
    const forecast = [...source];
    const getNormProfile = (arr) => {
      if (!Array.isArray(arr) || arr.length !== forecast.length) return null;
      const parsed = arr.map((v) => Number(v) || 0);
      const maxAbs = Math.max(...parsed.map((v) => Math.abs(v)), 0);
      if (maxAbs <= 1e-6) return null;
      return parsed.map((v) => v / maxAbs);
    };
    const tempProfile = getNormProfile(featureDeltas?.temperature);
    const humProfile = getNormProfile(featureDeltas?.humidity);
    const precipProfile = getNormProfile(featureDeltas?.precipitation);
    const hasImpactProfile = Array.isArray(weatherImpact) && weatherImpact.length === forecast.length;
    const maxImpact = hasImpactProfile
      ? Math.max(...weatherImpact.map((v) => Math.abs(Number(v) || 0)), 0)
      : 0;
    const blockWeight = (idx) => {
      if (hasImpactProfile && maxImpact > 1e-6) {
        const normalized = Math.abs(Number(weatherImpact[idx]) || 0) / maxImpact;
        return 0.6 + (0.8 * normalized);
      }
      const theta = (2 * Math.PI * idx) / forecast.length;
      const diurnal = 1 + (0.35 * Math.sin(theta - (Math.PI / 2))) + (0.15 * Math.sin((2 * theta) - (Math.PI / 3)));
      return Math.max(0.45, Math.min(1.55, diurnal));
    };
    const safeAdj = sanitizeWeatherAdj(adjustment);
    const adjust = (idx) => {
      const profileWeight = blockWeight(idx);
      const tempW = tempProfile ? tempProfile[idx] : profileWeight;
      const humW = humProfile ? humProfile[idx] : profileWeight;
      const precipW = precipProfile ? precipProfile[idx] : profileWeight;
      const deltaMw = (safeAdj.temp * 8 * tempW)
        + (safeAdj.humidity * 2.5 * humW)
        + (safeAdj.precip * -4 * precipW);
      forecast[idx] += deltaMw;
    };

    if (safeAdj.mode === 'all') {
      for (let i = 0; i < forecast.length; i += 1) adjust(i);
    } else if (safeAdj.mode === 'range') {
      for (let i = safeAdj.start - 1; i <= safeAdj.end - 1; i += 1) {
        if (i >= 0 && i < forecast.length) adjust(i);
      }
    } else if (safeAdj.mode === 'single') {
      const idx = safeAdj.block - 1;
      if (idx >= 0 && idx < forecast.length) adjust(idx);
    }
    return forecast;
  };

  // safeWeatherAdj is now defined earlier and synced with store

  const safeAppliedWeatherAdj = useMemo(() => sanitizeWeatherAdj(appliedWeatherAdj), [appliedWeatherAdj]);

  // Keep weather adjustments live so the chart reflects slider changes instantly.
  useEffect(() => {
    setAppliedWeatherAdj(safeWeatherAdj);
  }, [safeWeatherAdj]);

  const highlightRange = useMemo(() => {
    if (safeWeatherAdj.mode === 'range') return { start: safeWeatherAdj.start, end: safeWeatherAdj.end };
    if (safeWeatherAdj.mode === 'single') return { start: safeWeatherAdj.block, end: safeWeatherAdj.block };
    return null;
  }, [safeWeatherAdj]);

  const thresholds = {
    warning: settings?.alert_thresholds?.warning_pct ?? 5,
    critical: settings?.alert_thresholds?.critical_pct ?? 10,
  };

  const liveActual = useMemo(() => {
    if (!live?.series?.actual?.length) return [];
    const available = live?.metadata?.actual_blocks ?? actualBlocks;
    return live.series.actual.map((val, idx) => (idx < available ? val : null));
  }, [live, actualBlocks]);

  const liveAdjustedForecast = useMemo(() => {
    if (!live?.series?.forecast?.length) return [];
    return applyWeatherAdjustment(
      live.series.forecast,
      live?.series?.weather_impact,
      live?.series?.weather_feature_deltas,
      safeAppliedWeatherAdj
    );
  }, [live, safeAppliedWeatherAdj]);

  const liveDataZipped = useMemo(() => {
    const bArr = live?.series?.blocks || [];
    const fArr = liveAdjustedForecast.length ? liveAdjustedForecast : (live?.series?.forecast || []);
    const aArr = liveActual || [];
    const baseArr = live?.series?.hybrid_baseline || [];

    return bArr.map((b, idx) => ({
      block_number: b,
      forecast_mw: fArr[idx] || 0,
      actual_mw: aArr[idx] ?? null,
      baseline_mw: baseArr[idx] || 0
    }));
  }, [live, liveAdjustedForecast, liveActual]);

  const t2DataZipped = useMemo(() => {
    const s = liveT2?.series;
    if (!s?.blocks?.length) return [];
    const fArr = s.forecast || [];
    const baseArr = s.hybrid_baseline || [];
    return s.blocks.map((b, idx) => ({
      block_number: b,
      forecast_mw: fArr[idx] || 0,
      actual_mw: null,
      baseline_mw: baseArr[idx] || 0,
    }));
  }, [liveT2]);

  const forecastPageDate = activeHorizon === 't2' ? t2Date : (liveEffectiveDate || effectiveDate);
  // For T+2: prefer activeLive (liveT2) series; fall back to t2DataZipped if series.blocks missing
  const forecastPageBlocks = (activeHorizon === 't2' && !activeLive?.series?.blocks?.length)
    ? (t2DataZipped.length ? t2DataZipped.map(r => r.block_number) : [])
    : (activeLive?.series?.blocks || []);
  const forecastPageBaseline = (activeHorizon === 't2' && !activeLive?.series?.hybrid_baseline?.length)
    ? (t2DataZipped.map(r => r.baseline_mw))
    : (activeLive?.series?.hybrid_baseline || []);
  const forecastPageForecast = useMemo(() => (
    activeHorizon === 't1'
      ? (liveAdjustedForecast.length ? liveAdjustedForecast : (live?.series?.forecast || []))
      : (activeLive?.series?.forecast?.length
          ? activeLive.series.forecast
          : t2DataZipped.map(r => r.forecast_mw))
  ), [activeHorizon, activeLive, live, liveAdjustedForecast, t2DataZipped]);

  const forecastPageRows = useMemo(() => {
    // T+2 with no activeLive series: fall back to pre-computed t2DataZipped rows
    if (activeHorizon === 't2' && !forecastPageBlocks.length && t2DataZipped.length) {
      return t2DataZipped.map(r => ({ ...r, actual_mw: null }));
    }
    return forecastPageBlocks.map((b, i) => ({
      block_number: b,
      actual_mw: activeHorizon === 't1' ? (liveActual[i] ?? null) : null,
      forecast_mw: forecastPageForecast?.[i] ?? 0,
      baseline_mw: forecastPageBaseline?.[i] ?? 0,
    }));
  }, [activeHorizon, forecastPageBaseline, forecastPageBlocks, forecastPageForecast, liveActual, t2DataZipped]);

  const dayAheadSeries = useMemo(() => {
    // If we are in T2 mode, prioritize T2 data
    if (activeHorizon === 't2') {
      if (liveT2?.series) {
        return {
          blocks: liveT2.series.blocks,
          baseline: liveT2.series.hybrid_baseline,
          forecast: liveT2.series.forecast,
          actual: [], // T2 has no actuals yet
          weather_impact: liveT2.series.weather_impact,
          weather_feature_deltas: liveT2.series.weather_feature_deltas,
          metadata: liveT2.metadata
        };
      }
      return undefined; // liveT2 branch above covers T+2; nothing else available
    }

    // Default T1 logic
    if (active === 'forecast' && live?.series && liveEffectiveDate && effectiveDate === liveEffectiveDate) {
      return {
        blocks: live.series.blocks,
        baseline: live.series.hybrid_baseline,
        forecast: live.series.forecast,
        actual: liveActual,
        weather_impact: live.series.weather_impact,
        weather_feature_deltas: live.series.weather_feature_deltas,
        metadata: live.metadata
      };
    }
    return dayAhead?.series;
  }, [active, live, liveEffectiveDate, effectiveDate, dayAhead, liveActual, activeHorizon, liveT2]);

  const dayAheadAdjustedBaseline = useMemo(() => {
    const base = dayAheadSeries?.baseline || [];
    if (!base.length) return [];
    return applyWeatherAdjustment(
      base,
      dayAheadSeries?.weather_impact,
      dayAheadSeries?.weather_feature_deltas,
      safeWeatherAdj
    );
  }, [dayAheadSeries, safeWeatherAdj]);

  const weatherScopeBlocks = useMemo(() => {
    const totalBlocks = Array.isArray(dayAheadSeries?.blocks) && dayAheadSeries.blocks.length
      ? dayAheadSeries.blocks.length
      : 96;
    return selectionBlocksFromAdj(safeWeatherAdj, totalBlocks);
  }, [dayAheadSeries, safeWeatherAdj]);

  const weatherScopeSummary = useMemo(() => {
    if (!weatherScopeBlocks.length) return 'No blocks';
    if (weatherScopeBlocks.length >= 96) return 'All Blocks (96)';
    if (weatherScopeBlocks.length === 1) return `Block ${weatherScopeBlocks[0]}`;
    return `Blocks ${weatherScopeBlocks[0]}-${weatherScopeBlocks[weatherScopeBlocks.length - 1]} (${weatherScopeBlocks.length})`;
  }, [weatherScopeBlocks]);

  const weatherPreviewStats = useMemo(() => {
    const base = dayAheadSeries?.baseline || [];
    if (!base.length) return { deltaMw: 0, deltaPct: 0 };
    const adjusted = applyWeatherAdjustment(
      base,
      dayAheadSeries?.weather_impact,
      dayAheadSeries?.weather_feature_deltas,
      safeWeatherAdj
    );
    let baseTotal = 0;
    let deltaTotal = 0;
    weatherScopeBlocks.forEach((block) => {
      const idx = Number(block) - 1;
      if (idx < 0 || idx >= base.length) return;
      const b = Number(base[idx]) || 0;
      const a = Number(adjusted[idx]) || 0;
      baseTotal += b;
      deltaTotal += (a - b);
    });
    const deltaPct = Math.abs(baseTotal) > 1e-9 ? (deltaTotal / baseTotal) * 100 : 0;
    return { deltaMw: deltaTotal, deltaPct };
  }, [dayAheadSeries, safeWeatherAdj, weatherScopeBlocks]);

  const baselineWindowMapes = useMemo(() => {
    return dayAhead?.metadata?.baseline_window_mapes || config?.baseline_window_mapes || [];
  }, [dayAhead, config]);

  // Only compute residuals up to actual blocks (future blocks have no real data)
  const availableActualBlocks = live?.metadata?.actual_blocks || actualBlocks || 0;

  const optimizerResiduals = useMemo(() => {
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!actual.length || !baseline.length) return [];
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    return actual.slice(0, limit).map((val, idx) => {
      const base = baseline[idx];
      if (!Number.isFinite(val) || val === 0 || !Number.isFinite(base)) return null;
      return val - base;
    });
  }, [dayAheadSeries, availableActualBlocks]);

  const optimizerResidualStats = useMemo(() => getNumericStats(optimizerResiduals), [optimizerResiduals]);

  const optimizerResidualThreshold = useMemo(() => {
    if (!optimizerResidualStats) return null;
    const threshold = optimizerResidualStats.std * 1.5;
    return Number.isFinite(threshold) && threshold > 0 ? threshold : null;
  }, [optimizerResidualStats]);

  const optimizerPeakMarkers = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!blocks.length) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    let actualIdx = -1;
    let actualPeak = -Infinity;
    actual.slice(0, limit).forEach((val, idx) => {
      if (Number.isFinite(val) && val > 0 && val > actualPeak) {
        actualPeak = val;
        actualIdx = idx;
      }
    });
    let baselineIdx = -1;
    let baselinePeak = -Infinity;
    baseline.forEach((val, idx) => {
      if (Number.isFinite(val) && val > baselinePeak) {
        baselinePeak = val;
        baselineIdx = idx;
      }
    });
    if (actualIdx < 0 && baselineIdx < 0) return null;
    return {
      actual: actualIdx >= 0 ? { block: blocks[actualIdx], value: actualPeak } : null,
      baseline: baselineIdx >= 0 ? { block: blocks[baselineIdx], value: baselinePeak } : null
    };
  }, [dayAheadSeries, availableActualBlocks]);

  const optimizerPeakError = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!blocks.length || !actual.length || !baseline.length) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    let actualIdx = -1;
    let actualPeak = -Infinity;
    actual.slice(0, limit).forEach((val, idx) => {
      if (Number.isFinite(val) && val > 0 && val > actualPeak) {
        actualPeak = val;
        actualIdx = idx;
      }
    });
    if (actualIdx < 0) return null;
    const baselineAtPeak = Number.isFinite(baseline[actualIdx]) ? baseline[actualIdx] : null;
    const errorMw = Number.isFinite(baselineAtPeak) ? actualPeak - baselineAtPeak : null;
    return { block: blocks[actualIdx], actualPeak, baselineAtPeak, errorMw };
  }, [dayAheadSeries, availableActualBlocks]);

  const optimizerRampSeries = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (blocks.length < 2) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    const rampBlocks = blocks.slice(1, limit);
    const actualRamp = rampBlocks.map((_, idx) => {
      const curr = actual[idx + 1];
      const prev = actual[idx];
      if (!Number.isFinite(curr) || curr === 0 || !Number.isFinite(prev) || prev === 0) return null;
      return curr - prev;
    });
    const baselineRamp = rampBlocks.map((_, idx) => {
      const curr = baseline[idx + 1];
      const prev = baseline[idx];
      if (!Number.isFinite(curr) || !Number.isFinite(prev)) return null;
      return curr - prev;
    });
    const rampError = actualRamp.map((val, idx) => {
      const base = baselineRamp[idx];
      if (!Number.isFinite(val) || !Number.isFinite(base)) return null;
      return val - base;
    });
    return { rampBlocks, actualRamp, baselineRamp, rampError };
  }, [dayAheadSeries, availableActualBlocks]);

  const optimizerStability = useMemo(() => {
    const values = baselineWindowMapes
      .map((row) => Number(row.baseline_mape))
      .filter((val) => Number.isFinite(val))
      .sort((a, b) => a - b);
    if (!values.length) return null;
    const min = values[0];
    const max = values[values.length - 1];
    const median = values[Math.floor(values.length / 2)];
    const q1 = values[Math.floor((values.length - 1) * 0.25)];
    const q3 = values[Math.floor((values.length - 1) * 0.75)];
    return { min, max, median, q1, q3 };
  }, [baselineWindowMapes]);

  const optimizerAccuracyOption = useMemo(() => {
    if (!baselineWindowMapes.length) return null;
    const rows = [...baselineWindowMapes].sort((a, b) => Number(a.window_days) - Number(b.window_days));
    const windows = rows.map((row) => row.window_days);
    const mapes = rows.map((row) => Number(row.baseline_mape) || 0);
    let bestIdx = 0;
    let bestVal = Infinity;
    mapes.forEach((val, idx) => {
      if (Number.isFinite(val) && val < bestVal) {
        bestVal = val;
        bestIdx = idx;
      }
    });
    return {
      grid: { left: 45, right: 20, top: 30, bottom: 40 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(10px);',
        formatter: '{b} Days • {c}%'
      },
      xAxis: {
        type: 'category',
        data: windows,
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 10, formatter: '{value}%' },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: [{
        name: 'MAPE',
        type: 'line',
        data: mapes,
        smooth: true,
        symbolSize: 6,
        lineStyle: { color: '#4A90D9', width: 2 },
        itemStyle: { color: '#4A90D9' },
        markPoint: Number.isFinite(bestVal) ? {
          data: [{ coord: [windows[bestIdx], bestVal] }],
          symbolSize: 8,
          itemStyle: { color: '#f43f5e', shadowBlur: 5, shadowColor: 'rgba(255, 0, 85, 0.5)' }
        } : undefined,
        markLine: Number.isFinite(baselineDays) ? {
          data: [{ xAxis: baselineDays }],
          lineStyle: { color: 'rgba(255, 255, 255, 0.2)', type: 'dashed' }
        } : undefined
      }]
    };
  }, [baselineWindowMapes, baselineDays]);

  const optimizerResidualHistogramOption = useMemo(() => {
    const hist = buildHistogram(optimizerResiduals, 10);
    if (!hist) return null;
    const labels = hist.bins.map((bin) => {
      const mid = Math.round(bin.mid);
      return `${mid}`;
    });
    return {
      grid: { left: 50, right: 20, top: 30, bottom: 40 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        extraCssText: 'backdrop-filter: blur(10px);',
        formatter: (params) => {
          const p = params?.[0];
          if (!p) return '';
          const bin = hist.bins[p.dataIndex];
          if (!bin) return '';
          return `<div style="font-weight:700;">Residual Impact</div><div>${fmt(bin.start)} to ${fmt(bin.end)} MW</div><div style="color:#38bdf8;">Count: ${bin.count}</div>`;
        }
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 0 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: [{
        type: 'bar',
        data: hist.bins.map((bin) => bin.count),
        itemStyle: {
          color: '#4A90D9',
          opacity: 0.8,
          borderRadius: [4, 4, 0, 0]
        },
        emphasis: { itemStyle: { opacity: 1, color: '#4A90D9' } }
      }]
    };
  }, [optimizerResiduals]);

  const optimizerErrorHeatmapOption = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    if (!blocks.length || !optimizerResiduals.length) return null;
    const values = optimizerResiduals.map((val) => (Number.isFinite(val) ? Math.abs(val) : null));
    const data = values
      .map((val, idx) => (Number.isFinite(val) ? [idx, 0, val] : null))
      .filter(Boolean);
    if (!data.length) return null;
    const maxVal = Math.max(...data.map((row) => row[2]), 1);
    return {
      grid: { left: 45, right: 20, top: 20, bottom: 40 },
      tooltip: {
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc' },
        formatter: (p) => `Block ${blocks[p.data[0]]}: <b>${fmt(p.data[2])} MW</b>`
      },
      xAxis: {
        type: 'category',
        data: blocks,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 5 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'category',
        data: ['Error'],
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      visualMap: {
        min: 0,
        max: maxVal,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: { color: ['#1e2028', '#4A90D9', '#f43f5e'] },
        textStyle: { color: '#64748b', fontSize: 10 }
      },
      series: [{
        type: 'heatmap',
        data,
        itemStyle: { borderColor: 'rgba(0,0,0,0.5)', borderWidth: 1 }
      }]
    };
  }, [dayAheadSeries, optimizerResiduals]);

  const optimizerRampComparisonOption = useMemo(() => {
    if (!optimizerRampSeries) return null;
    return {
      grid: { left: 50, right: 20, top: 30, bottom: 40 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc' },
        formatter: '{b}: {c} MW/15m'
      },
      legend: { data: ['Actual Ramp', 'Baseline Ramp'], textStyle: { color: '#64748b', fontSize: 10 }, top: 0, icon: 'circle' },
      xAxis: {
        type: 'category',
        data: optimizerRampSeries.rampBlocks,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 7 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: [
        {
          name: 'Actual Ramp',
          type: 'line',
          data: optimizerRampSeries.actualRamp,
          smooth: true,
          lineStyle: { color: '#34d399', width: 2, shadowBlur: 10, shadowColor: 'rgba(57, 255, 20, 0.5)' }
        },
        {
          name: 'Baseline Ramp',
          type: 'line',
          data: optimizerRampSeries.baselineRamp,
          smooth: true,
          lineStyle: { color: 'rgba(255, 255, 255, 0.3)', width: 1.5, type: 'dashed' }
        }
      ]
    };
  }, [optimizerRampSeries]);

  const optimizerPeakErrorOption = useMemo(() => {
    if (!optimizerPeakError || !Number.isFinite(optimizerPeakError.actualPeak) || !Number.isFinite(optimizerPeakError.baselineAtPeak)) {
      return null;
    }
    return {
      grid: { left: 50, right: 20, top: 30, bottom: 40 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc' },
        formatter: '{b}: <b>{c} MW</b>'
      },
      xAxis: {
        type: 'category',
        data: ['Baseline @ Peak', 'Actual Peak'],
        axisLabel: { color: '#64748b', fontSize: 11 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: [{
        type: 'bar',
        data: [
          { value: optimizerPeakError.baselineAtPeak, itemStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
          { value: optimizerPeakError.actualPeak, itemStyle: { color: '#34d399', shadowBlur: 10, shadowColor: 'rgba(57, 255, 20, 0.4)' } }
        ],
        barWidth: '40%',
        borderRadius: [4, 4, 0, 0]
      }]
    };
  }, [optimizerPeakError]);

  const optimizerRampErrorOption = useMemo(() => {
    if (!optimizerRampSeries) return null;
    const absErrors = optimizerRampSeries.rampError.map((val) => (Number.isFinite(val) ? Math.abs(val) : null));
    if (!absErrors.some((val) => Number.isFinite(val))) return null;
    return {
      grid: { left: 50, right: 20, top: 30, bottom: 40 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc' },
        formatter: '{b}: <b>{c} MW</b>'
      },
      xAxis: {
        type: 'category',
        data: optimizerRampSeries.rampBlocks,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 7 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: [{
        type: 'bar',
        data: absErrors,
        itemStyle: {
          color: '#f43f5e',
          opacity: 0.8,
          borderRadius: [4, 4, 0, 0]
        },
        emphasis: { itemStyle: { opacity: 1 } }
      }]
    };
  }, [optimizerRampSeries]);

  const variancePct = useMemo(() => {
    if (!dayAheadSeries?.actual?.length) return [];
    return dayAheadSeries.actual.map((val, idx) => {
      const base = dayAheadSeries.baseline?.[idx] || 0;
      if (!base || val === null) return 0;
      return ((val - base) / base) * 100;
    });
  }, [dayAheadSeries]);

  const liveTrendSlope = useMemo(() => {
    if (!live?.trend_curve?.length) return null;
    const curve = live.trend_curve;
    if (curve.length < 2) return null;
    return (curve[curve.length - 1] - curve[0]) / (curve.length - 1);
  }, [live]);

  const liveContributorRows = useMemo(() => {
    const rows = Array.isArray(live?.block_contributors) ? live.block_contributors : [];
    return rows
      .map((row) => {
        const ranked = Array.isArray(row?.contributors_ranked) ? row.contributors_ranked : [];
        return {
          block: row?.block,
          time: row?.time,
          netImpactPct: Number(row?.net_impact_pct || 0),
          netContributionMw: Number(row?.net_contribution_mw || 0),
          primary: ranked[0]?.feature || '—',
          primaryMw: Number(ranked[0]?.contribution_mw || 0),
          confidence: Number(row?.contributor_confidence || 0),
        };
      })
      .sort((a, b) => Math.abs(b.netContributionMw) - Math.abs(a.netContributionMw))
      .slice(0, 12);
  }, [live]);

  const liveDriverRows = useMemo(() => {
    const rows = Array.isArray(live?.driver_contributions) ? live.driver_contributions : [];
    const filtered = rows.filter((row) => {
      if (row.factor !== 'Trend MW') return true;
      const bias = rows.find((r) => r.factor === 'Bias MW');
      if (!bias) return true;
      const diff = Math.abs((bias.mw ?? 0) - (row.mw ?? 0));
      const base = Math.max(Math.abs(bias.mw ?? 0), Math.abs(row.mw ?? 0), 1);
      return diff / base > 0.02;
    });
    return filtered.map((row) => ({
      factor: row.factor,
      mw: Number(row.mw || 0),
      pct: Number(row.pct || 0),
    }));
  }, [live]);

  const liveDriverPctScale = useMemo(() => {
    return Math.max(...liveDriverRows.map((row) => Math.abs(Number(row.pct || 0))), 1);
  }, [liveDriverRows]);

  const liveDecisionRows = useMemo(() => {
    const rows = Array.isArray(live?.decision_signals) ? live.decision_signals : [];
    const priority = { high: 3, medium: 2, low: 1 };
    return rows
      .map((row) => ({
        block: row?.block,
        time: row?.time,
        risk: String(row?.risk_flag || 'low'),
        primary: row?.primary_driver || '—',
        action: row?.recommended_action || '—',
        uncertainty: Number(row?.uncertainty_pct || 0),
        confidence: Number(row?.confidence || 0),
      }))
      .sort((a, b) => (priority[b.risk] || 0) - (priority[a.risk] || 0) || (b.uncertainty - a.uncertainty))
      .slice(0, 12);
  }, [live]);

  const liveSensitivityRows = useMemo(() => {
    const rows = Array.isArray(live?.slot_sensitivity_profile) ? live.slot_sensitivity_profile : [];
    return rows
      .map((row) => ({
        block: row?.block,
        time: row?.time,
        dominant: row?.dominant_family || '—',
        score: Number(row?.sensitivity_score || 0),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }, [live]);

  const activeLiveInsightMeta = useMemo(() => {
    return LIVE_INSIGHT_OPTIONS.find((opt) => opt.key === activeLiveInsight) || LIVE_INSIGHT_OPTIONS[0];
  }, [activeLiveInsight]);

  const flaggedBlocks = useMemo(() => {
    return variancePct
      .map((v, idx) => ({ block: idx + 1, variance: v }))
      .filter((row) => Math.abs(row.variance) >= thresholds.warning)
      .slice(0, 12);
  }, [variancePct, thresholds.warning]);

  const selectedBlockDetail = useMemo(() => {
    const block = selectedBlock;
    const idx = block - 1;
    const actual = dayAhead?.series?.actual?.[idx];
    const forecast = dayAhead?.series?.forecast?.[idx];
    const baseline = dayAhead?.series?.baseline?.[idx];
    const variance = variancePct?.[idx];
    const blockAttrib = dayAhead?.attribution?.block_level?.find((b) => b.block === block);
    return { block, actual, forecast, baseline, variance, blockAttrib };
  }, [selectedBlock, dayAhead, variancePct]);

  const mean = (arr) => {
    if (!arr || !arr.length) return null;
    const clean = arr.filter((v) => Number.isFinite(v));
    if (!clean.length) return null;
    return clean.reduce((a, b) => a + b, 0) / clean.length;
  };

  const activeKpiGroup = useMemo(() => {
    return KPI_GROUPS.find((g) => g.key === selectedKpiGroup) || KPI_GROUPS[0];
  }, [selectedKpiGroup]);

  const activeKpiItems = useMemo(() => {
    if (!dayAhead?.kpis_full || !activeKpiGroup) return [];
    return activeKpiGroup.keys
      .filter((key) => Object.prototype.hasOwnProperty.call(dayAhead.kpis_full, key))
      .map((key) => ({ key, value: dayAhead.kpis_full[key] }))
      .map((item) => {
        const meta = KPI_META[item.key] || {};
        const label = meta.label || titleize(item.key);
        const unit = meta.unit || '';
        const status = getStatus(item.key, item.value);
        return { ...item, label, unit, status };
      });
  }, [dayAhead, activeKpiGroup]);

  const kpiImpactRows = useMemo(() => {
    const kpiVal = dayAhead?.kpis_full?.[selectedKpiImpact];
    if (!Number.isFinite(kpiVal)) return [];
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const forecast = dayAheadSeries?.forecast || [];
    if (!blocks.length || !forecast.length) return [];
    const errors = blocks.map((_, idx) => {
      const a = actual[idx];
      const f = forecast[idx];
      if (!Number.isFinite(a) || !Number.isFinite(f)) return 0;
      return Math.abs(a - f);
    });
    const errorSum = errors.reduce((a, b) => a + b, 0) || 1;
    const blockAttrib = dayAhead?.attribution?.block_level || [];
    return blocks.map((block, idx) => {
      const share = errors[idx] / errorSum;
      const totalImpact = kpiVal * share;
      const attrib = blockAttrib.find((b) => b.block === block) || {};
      const drivers = [
        { key: 'temperature', val: attrib.temperature || 0 },
        { key: 'humidity', val: attrib.humidity || 0 },
        { key: 'precipitation', val: attrib.precipitation || 0 },
        { key: 'unexplained', val: attrib.unexplained || 0 }
      ];
      const denom = drivers.reduce((s, d) => s + Math.abs(d.val), 0) || 1;
      const row = {
        block,
        time: attrib.time || blockTime(block),
        total_variance: totalImpact
      };
      drivers.forEach((d) => {
        const sign = d.val === 0 ? 1 : Math.sign(d.val);
        row[d.key] = sign * Math.abs(totalImpact) * (Math.abs(d.val) / denom);
      });
      return row;
    });
  }, [selectedKpiImpact, dayAhead, dayAheadSeries]);


  const analysisSummary = useMemo(() => {
    const actual = dayAheadSeries?.actual || [];
    const forecast = dayAheadSeries?.forecast || [];
    const baseline = dayAheadSeries?.baseline || [];
    const errors = actual.map((a, i) => (Number.isFinite(a) && Number.isFinite(forecast[i]) ? a - forecast[i] : 0));
    const absErrors = errors.map((e) => Math.abs(e));
    const totalVariancePct = variancePct.length ? (absErrors.reduce((a, b) => a + b, 0) / Math.max(actual.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0), 1e-6) * 100) : 0;
    const peakDeviation = absErrors.length ? Math.max(...absErrors) : 0;
    const netBias = errors.length ? errors.reduce((a, b) => a + b, 0) / Math.max(errors.length, 1) : 0;
    const accuracy = dayAhead?.kpis?.mape !== undefined ? Math.max(0, 100 - dayAhead.kpis.mape) : null;
    const baselineSum = baseline.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
    return { totalVariancePct, peakDeviation, netBias, accuracy, baselineSum };
  }, [dayAheadSeries, variancePct, dayAhead]);

  const alertStack = useMemo(() => {
    return variancePct
      .map((v, idx) => ({ block: idx + 1, variance: v }))
      .filter((row) => Math.abs(row.variance) >= 3)
      .map((row) => ({
        ...row,
        severity: Math.abs(row.variance) >= 10 ? 'critical' : Math.abs(row.variance) >= 5 ? 'major' : 'minor'
      }))
      .slice(0, 12);
  }, [variancePct]);

  const [exportLoading, setExportLoading] = useState(false);
  const handleExportLiveCsv = async () => {
    const exportDate = live?.date || effectiveDate;
    if (!exportDate) { alert('No forecast date available. Run a forecast first.'); return; }
    setExportLoading(true);
    try {
      const token = useAuthStore.getState().token;
      const res = await fetch(TRAINING_API_URL('/v2/forecast/export'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          date: exportDate,
          region: selectedRegion || 'haryana',
          baseline_days: baselineDays || 7,
        }),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`Server error ${res.status}: ${txt.slice(0, 200)}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `GNA_Forecast_${exportDate}_T1_T2.xlsx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
      alert(`Export failed: ${err.message}`);
    } finally {
      setExportLoading(false);
    }
  };

  const benchmarkOption = useMemo(() => {
    if (!benchmarkData) return null;
    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);
    const hasAny = (arr) => Array.isArray(arr) && arr.some((v) => v != null && Number.isFinite(Number(v)));
    const mkSeries = (name, data, extra) => {
      if (!hasAny(data)) return null;
      return {
        name,
        type: 'line',
        data,
        showSymbol: false,
        symbol: 'none',
        smooth: true,
        ...extra
      };
    };
    const series = [
      mkSeries('Today', benchmarkData.today, { itemStyle: { color: '#4A90D9' }, lineStyle: { width: 2 }, z: 10 }),
      mkSeries('Y\'day (T-1)', benchmarkData.t1, { itemStyle: { color: 'rgba(255,255,255,0.4)' }, lineStyle: { width: 1.5, type: 'dashed' } }),
      mkSeries('Last Week (T-7)', benchmarkData.t7, { itemStyle: { color: '#ffb300' }, opacity: 0.6 }),
      mkSeries('Last Year (T-365)', benchmarkData.t365, { itemStyle: { color: '#9d4edd' }, opacity: 0.5 }),
    ].filter(Boolean);

    return {
      grid: { left: 50, right: 22, top: 28, bottom: 44 },
      legend: { textStyle: { color: '#64748b', fontSize: 10 }, top: 0, icon: 'circle', type: series.length > 3 ? 'scroll' : 'plain' },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#f8fafc' },
        extraCssText: 'backdrop-filter: blur(10px);'
      },
      xAxis: {
        type: 'category',
        data: blocks,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 7 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series
    };
  }, [benchmarkData]);

  const comparisonOption = useMemo(() => {
    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);
    const colors = ['#4A90D9', '#34d399', '#f43f5e', '#ffe800', '#9d4edd', '#ffab00', '#4ecdc4'];
    const dates = Object.keys(multiDaySeries || {});
    return {
      grid: { left: 50, right: 22, top: 28, bottom: 44 },
      legend: { textStyle: { color: '#64748b', fontSize: 10 }, top: 0, type: 'scroll', icon: 'circle' },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(18, 20, 28, 0.85)',
        textStyle: { color: '#f8fafc' },
        extraCssText: 'backdrop-filter: blur(10px);'
      },
      xAxis: {
        type: 'category',
        data: blocks,
        axisLabel: { color: '#64748b', fontSize: 10, interval: 7 },
        axisLine: { show: false },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
      },
      series: dates.map((d, i) => ({
        name: d,
        type: 'line',
        data: multiDaySeries[d],
        itemStyle: { color: colors[i % colors.length] },
        lineStyle: { width: 2.5, shadowBlur: 8, shadowColor: `${colors[i % colors.length]}66` },
        smooth: true,
        symbol: 'none'
      }))
    };
  }, [multiDaySeries]);

  const momentumOption = useMemo(() => {
    if (!momentumChange) return null;

    if (momentumChange.type === 'daily') {
      const dates = momentumChange.data.map(d => d.date);
      const values = momentumChange.data.map(d => d.value);
      return {
        grid: { left: 50, right: 22, top: 28, bottom: 44 },
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(18, 20, 28, 0.85)',
          textStyle: { color: '#f8fafc' },
          formatter: '{b}: <b>{c}%</b>'
        },
        xAxis: {
          type: 'category',
          data: dates,
          axisLabel: { color: '#64748b', fontSize: 10, rotate: 35 },
          axisLine: { show: false },
          axisTick: { show: false }
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#64748b', fontSize: 10, formatter: '{value}%' },
          axisLine: { show: false },
          splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
        },
        series: [{
          name: 'Day-over-Day %',
          type: 'bar',
          data: values,
          itemStyle: {
            color: (params) => params.value >= 0 ? '#34d399' : '#f43f5e',
            borderRadius: [4, 4, 0, 0]
          }
        }]
      };
    } else if (momentumChange.type === 'blocks') {
      const blocks = Array.from({ length: 96 }, (_, i) => i + 1);
      return {
        grid: { left: 50, right: 22, top: 28, bottom: 44 },
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(18, 20, 28, 0.85)',
          textStyle: { color: '#f8fafc' },
          formatter: 'Block {b}: <b>{c}%</b>'
        },
        xAxis: {
          type: 'category',
          data: blocks,
          axisLabel: { color: '#64748b', fontSize: 10, interval: 7 },
          axisLine: { show: false },
          axisTick: { show: false }
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#64748b', fontSize: 10, formatter: '{value}%' },
          axisLine: { show: false },
          splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.03)', type: 'dashed' } }
        },
        series: [{
          name: 'Block Change %',
          type: 'bar',
          data: momentumChange.change_pct,
          itemStyle: {
            color: (params) => params.value >= 0 ? '#34d399' : '#f43f5e',
            borderRadius: [2, 2, 0, 0]
          }
        }]
      };
    }
    return null;
  }, [momentumChange]);


  const hasHighRisk = dayAhead?.metadata?.insights?.some(i => i.type === 'warning');
  const riskMessage = hasHighRisk ? 'High Variance Detected: Weather Anomaly' : 'System Nominal';

  // Forecast health for CommandStrip & AlertRibbon
  const forecastHealth = useMemo(() => {
    const data = liveDataZipped;
    if (!data?.length) return { mape: 0, status: 'unknown', color: 'var(--muted)', consecutiveHigh: 0 };
    const actuals = data.filter(d => d.actual_mw != null && d.actual_mw > 0 && d.forecast_mw > 0);
    if (actuals.length < 4) return { mape: 0, status: 'insufficient', color: 'var(--muted)', consecutiveHigh: 0 };
    const apes = actuals.map(d => Math.abs(d.actual_mw - d.forecast_mw) / Math.max(d.actual_mw, 1) * 100);
    const mape = apes.reduce((a, b) => a + b, 0) / apes.length;
    let consecutive = 0, maxConsecutive = 0;
    for (const ape of [...apes].reverse()) {
      if (ape > 5) { consecutive++; maxConsecutive = Math.max(maxConsecutive, consecutive); }
      else { consecutive = 0; }
    }
    let status = 'good', color = 'var(--success)';
    if (mape > 5) { status = 'critical'; color = 'var(--danger)'; }
    else if (mape > 2) { status = 'warning'; color = 'var(--warning)'; }
    return { mape: Math.round(mape * 100) / 100, status, color, consecutiveHigh: maxConsecutive, needsReforecast: maxConsecutive >= 8 };
  }, [liveDataZipped]);

  const commandStripStats = useMemo(() => {
    const data = liveDataZipped;
    if (!data?.length) return null;
    const actuals = data.filter(d => d.actual_mw != null && d.actual_mw > 0);
    const lastActual = actuals[actuals.length - 1];
    const peak = Math.max(...data.map(d => d.forecast_mw || 0), 0);
    const energy = data.reduce((acc, d) => acc + (d.forecast_mw || 0), 0) * 0.25;
    // Solar hours avg demand: blocks 33-64 (08:00-16:00)
    const solarBlocks = data.filter(d => d.block_number >= 33 && d.block_number <= 64);
    const solarAvg = solarBlocks.length > 0
      ? solarBlocks.reduce((s, d) => s + (d.forecast_mw || 0), 0) / solarBlocks.length
      : 0;
    return { lastActual: lastActual?.actual_mw || 0, lastBlock: lastActual?.block_number || 0, peak, energy, solarAvg };
  }, [liveDataZipped]);

  return (
    <div className="shell nexus-shell">
      {/* Data preparation progress — shown during phase 2 only, non-blocking */}
      {initPhase === 'data' && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 900,
          background: 'linear-gradient(90deg, var(--accent) 0%, transparent 100%)',
          height: 3,
          animation: 'progress-bar-indeterminate 1.4s ease-in-out infinite',
        }} />
      )}

      {/* Non-blocking forecast training status bar */}
      {initPhase === 'training' && (
        <div style={{
          position: 'fixed', bottom: 0, left: 64, right: 0, zIndex: 950,
          background: 'var(--bg-elevated)', borderTop: '1px solid var(--outline)',
          padding: '6px 16px', display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <Loader2 size={13} style={{ animation: 'spin 1s linear infinite', color: '#F07825', flexShrink: 0 }} />
          <span style={{ fontSize: 11, color: '#A0A5B8', flexShrink: 0 }}>Model training…</span>
          {trainingProgress.length > 0 && (
            <span style={{ fontSize: 10, color: '#6B7186', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {trainingProgress[trainingProgress.length - 1]?.message || ''}
            </span>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, flexShrink: 0 }}>
            {['Loading data', 'Feature engineering', 'Similarity search', 'ML training', 'Assembling forecast'].map((step, i) => {
              const stepPhrases = ['1/7', '2/7', '3/7', '4/7', '5/7'];
              const done = trainingProgress.some(p => p.message?.includes(stepPhrases[i]));
              return (
                <span key={step} style={{
                  fontSize: 11, padding: '2px 8px', borderRadius: 20,
                  background: done ? 'rgba(52,211,153,0.15)' : 'var(--bg-surface)',
                  color: done ? '#34D399' : '#8A90A6',
                  border: `1px solid ${done ? 'rgba(52,211,153,0.3)' : 'var(--outline)'}`,
                }}>{step}</span>
              );
            })}
          </div>
        </div>
      )}

      {/* Left Sidebar Navigation — redesigned, floating, collapsible */}
      <nav className={`sidebar-nav sidebar-nav--v2 ${sidebarCollapsed ? 'is-collapsed' : ''}`}>
        <div className="sb-brand">
          <div className="sb-brand-mark">
            <Zap size={15} />
          </div>
          <div className="sb-brand-text">
            <span className="sb-brand-title">VidyutPragya</span>
            <span className="sb-brand-sub">Forecast OS</span>
          </div>
          <button
            className="sb-collapse-btn"
            onClick={() => setSidebarCollapsed((v) => !v)}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? <ChevronsRight size={14} /> : <ChevronsLeft size={14} />}
          </button>
        </div>

        <div className="sb-section-label">Workspaces</div>
        <div className="sb-links">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = active === item.key;
            return (
              <button
                key={item.key}
                className={`sb-link ${isActive ? 'is-active' : ''}`}
                onClick={() => setActive(item.key)}
                title={item.label}
              >
                <span className="sb-link-rail" />
                <Icon size={15} className="sb-link-icon" />
                <span className="sb-link-label">{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="sb-bottom">
          {authUser && (
            <div className="sb-user" title={authUser.email}>
              <div className="sb-user-avatar">
                {(authUser.username || '?').slice(0, 1).toUpperCase()}
              </div>
              <div className="sb-user-meta">
                <span className="sb-user-name">{authUser.username}</span>
                <span className="sb-user-email">{authUser.email || 'signed in'}</span>
              </div>
            </div>
          )}
          <button
            className={`sb-link sb-link--small ${active === 'settings' ? 'is-active' : ''}`}
            onClick={() => setActive('settings')}
            title="Settings"
          >
            <span className="sb-link-rail" />
            <Settings size={14} className="sb-link-icon" />
            <span className="sb-link-label">Settings</span>
          </button>
          {onHome && (
            <button
              className="sb-link sb-link--small"
              onClick={onHome}
              title="Back to Home"
              style={{ opacity: 0.7 }}
            >
              <span className="sb-link-rail" />
              <LayoutGrid size={14} className="sb-link-icon" />
              <span className="sb-link-label">Home</span>
            </button>
          )}
          <button
            className="sb-link sb-link--small sb-link--danger"
            onClick={onLogout}
            title={`Sign out (${authUser?.username || ''})`}
          >
            <span className="sb-link-rail" />
            <LogOut size={14} className="sb-link-icon" />
            <span className="sb-link-label">Sign out</span>
          </button>
        </div>
      </nav>

      {/* Main content area */}
      <div className="main-area">

      {/* Command Strip — always visible top bar */}
      <CommandStrip
        effectiveDate={effectiveDate}
        selectedRegion={selectedRegion}
        date={date}
        setDate={setDate}
        loading={loading}
        onReload={() => { reloadData(); loadFromFileData(effectiveDate, baselineDays); }}
        onRefreshLive={fetchLive}
        onDownload={handleExportLiveCsv}
        canDownload={Boolean(live?.date || effectiveDate) && !exportLoading}
        activeView={active}
        setActiveView={setActive}
        forecastHealth={forecastHealth}
        stats={commandStripStats}
        fmt={fmt}
        liveMeta={live?.metadata}
        onToggleContext={() => setContextOpen(!contextOpen)}
        contextOpen={contextOpen}
        onViewData={() => setViewDataOpen(true)}
      />

      {/* Alert Ribbon — conditional */}
      <AlertRibbon
        forecastHealth={forecastHealth}
        alerts={alertStack}
        onReforecast={fetchLive}
      />

      {/* Main body: workspace + context panel */}
      <div className={`nexus-body ${contextOpen ? 'ctx-open' : ''}`}>
      <div className="workspace">

      {/* Page-scoped loading veil — keep app chrome visible */}
      {(loading || initPhase === 'config') && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="loading-brand loading-brand--pulse">
              <span className="loading-brand__title">VidyutPragya</span>
              <span className="loading-brand__subtitle">
                {selectedRegion ? `${titleize(selectedRegion)} analytics` : 'Loading analytics'}
              </span>
            </div>
            <div className="label">
              {initPhase === 'config' ? 'Loading configuration…' : 'Preparing data…'}
            </div>
          </div>
        </div>
      )}

      {viewDataOpen && (() => {
        const data = active === 'live' ? live : dayAhead;
        // Build raw rows from all available data sources
        let rawRows = [];
        let rawKeys = [];

        if (data) {
          // Strategy 1: forecast_df (full raw data)
          if (Array.isArray(data.forecast_df) && data.forecast_df.length > 0) {
            rawRows = data.forecast_df;
            rawKeys = Object.keys(data.forecast_df[0]);
          }
          // Strategy 2: series arrays → build rows
          else if (data.series && data.series.blocks) {
            const s = data.series;
            rawKeys = ['block', 'time', 'baseline_mw', 'forecast_mw', 'actual_mw', 'diff_mw', 'p10_mw', 'p90_mw'];
            rawRows = s.blocks.map((b, i) => {
              const base = s.baseline?.[i];
              const fore = s.forecast?.[i];
              const act = s.actual?.[i];
              const p10 = s.p10?.[i];
              const p90 = s.p90?.[i];
              const diff = (Number.isFinite(act) && Number.isFinite(fore)) ? act - fore : null;
              const mins = (b - 1) * 15;
              const time = `${String(Math.floor(mins / 60)).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}`;
              return { block: b, time, baseline_mw: base, forecast_mw: fore, actual_mw: act, diff_mw: diff, p10_mw: p10, p90_mw: p90 };
            });
          }
          // Strategy 3: raw object dump
          else {
            rawKeys = ['key', 'value'];
            rawRows = Object.entries(data).map(([k, v]) => ({
              key: k,
              value: typeof v === 'object' ? JSON.stringify(v) : v
            }));
          }
        }

        return (
        <div className="modal-overlay" onClick={() => setViewDataOpen(false)}>
          <div className="dv-panel" onClick={e => e.stopPropagation()}>
            <div className="dv-header">
              <div className="dv-title-area">
                <h3>Raw Data Viewer</h3>
                <span className="dv-subtitle">{effectiveDate} &bull; {selectedRegion} &bull; {rawRows.length} rows &times; {rawKeys.length} cols</span>
              </div>
              <div className="dv-actions">
                <button className="dv-close" onClick={() => setViewDataOpen(false)}>&times;</button>
              </div>
            </div>
            {/* Column Filters */}
            <DataViewerTable rows={rawRows} keys={rawKeys} fmt={fmt} />
          </div>
        </div>
        );
      })()}

      <div className="toast-stack">
        {toasts.map((toast) => (
          <Toast key={toast.id} tone={toast.tone} message={toast.message} onClose={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))} />
        ))}
      </div>

      {active === 'load_analysis' && (
        <Suspense fallback={<ViewLoading label="Loading load analysis..." />}>
          <LoadAnalysisPage
            effectiveDate={effectiveDate}
            benchmarkData={benchmarkData}
            momentumChange={momentumChange}
            multiDaySeries={multiDaySeries}
            multiSelectedDates={multiSelectedDates}
            setMultiSelectedDates={setMultiSelectedDates}
            detectedSeason={detectedSeason}
            liveData={live}
            dayAheadData={dayAhead}
            apiUrl={API_URL}
            horizon={activeHorizon}
            setHorizon={setActiveHorizon}
            t2Date={t2Date}
            liveT2={liveT2}
          />
        </Suspense>
      )}

      {
        active === 'weather_analysis' && (
          <Suspense fallback={<ViewLoading label="Loading weather analysis..." />}>
            <WeatherDeepPage
              effectiveDate={effectiveDate}
              dayAheadData={dayAhead}
              dayAheadSeries={dayAheadSeries}
              liveData={live}
              selectedRegion={selectedRegion}
              horizon={activeHorizon}
              setHorizon={setActiveHorizon}
              t2Date={t2Date}
              dayAheadT2={dayAheadT2}
            />
          </Suspense>
        )
      }


      {active === 'simulator' && (
        <Suspense fallback={<ViewLoading label="Loading simulator..." />}>
          <SimulatorPage requestedDate={effectiveDate} baselineDays={baselineDays} />
        </Suspense>
      )}

      {active === 'analysis' && (
        <Suspense fallback={<ViewLoading label="Loading analysis..." />}>
          <AnalysisPage
            effectiveDate={effectiveDate}
            dayAheadData={dayAhead}
            liveForecastData={live}
            liveT2={liveT2}
            horizon={activeHorizon}
            setHorizon={setActiveHorizon}
            t2Date={t2Date}
            dayAheadT2={dayAheadT2}
          />
        </Suspense>
      )}

      {
        active === 'forecast' && (
          <Suspense fallback={<ViewLoading label="Loading forecast workspace..." />}>
            <ForecastPage
              liveData={forecastPageRows}
              liveMeta={activeLive?.metadata}
              driverContributions={activeLive?.driver_contributions || []}
              decisionSignals={activeLive?.decision_signals || []}
              effectiveDate={forecastPageDate}
              selectedRegion={selectedRegion}
              onRefresh={fetchLive}
              onDownload={handleExportLiveCsv}
              canDownload={Boolean(activeLive?.date || effectiveDate) && !exportLoading}
              fmt={fmt}
              horizon={activeHorizon}
              setHorizon={setActiveHorizon}
              t2Date={t2Date}
              t2Loading={liveT2Loading}
              forecastQuality={activeLive?.forecast_quality}
              chart={
                <LoadChart
                  blocks={forecastPageBlocks}
                  baseline={forecastPageBaseline}
                  forecast={forecastPageForecast}
                  actual={activeHorizon === 't1' ? liveActual : []}
                  p10={activeLive?.series?.p10 || []}
                  p90={activeLive?.series?.p90 || []}
                  dateLabel={forecastPageDate || (activeHorizon === 't2' ? (activeLive?.t2_date || '') : (activeLive?.date || ''))}
                />
              }
              weatherStrip={
                <WeatherStrip
                  dayAhead={activeHorizon === 't2' ? activeLive : (activeDayAhead || activeLive)}
                  live={activeLive}
                  fmt={fmt}
                />
              }
              forecastTable={
                <ForecastTable
                  liveData={forecastPageRows}
                  forecastUncertainty={activeLive?.forecast_uncertainty}
                  actualBlocks={activeHorizon === 't2' ? 0 : (activeLive?.metadata?.actual_blocks || actualBlocks)}
                  fmt={fmt}
                  horizon={activeHorizon}
                />
              }
            />
          </Suspense>
        )
      }

      {/* Simulator and Analysis handled above */}

      {active === 'monitor' && <MonitorPage live={live} dayAhead={dayAhead} effectiveDate={effectiveDate} selectedRegion={selectedRegion} />}

      {active === 'studio' && (
        <Suspense fallback={<ViewLoading label="Loading Forecast Studio..." />}>
          <ForecastStudioPage selectedRegion={selectedRegion} />
        </Suspense>
      )}

      {active === 'backtest' && (
        <Suspense fallback={<ViewLoading label="Loading Backtest..." />}>
          <BacktestPage selectedRegion={selectedRegion} />
        </Suspense>
      )}

      {active === 'similar_days' && (
        <Suspense fallback={<ViewLoading label="Loading Similar Days..." />}>
          <SimilarDaysPage
            horizon={activeHorizon}
            setHorizon={setActiveHorizon}
            t2Date={t2Date}
            t1Forecast={live?.series?.forecast}
          />
        </Suspense>
      )}

      {active === 'weather_loc' && (
        <Suspense fallback={<ViewLoading label="Loading Weather Locations..." />}>
          <WeatherLocPage />
        </Suspense>
      )}

      {
        showRegionPrompt && (
          <RegionSelectionModal
            regions={availableRegions}
            selected={selectedRegion}
            onSelect={setSelectedRegion}
            onConfirm={handleRegionConfirm}
            dateRange={regionDateRange}
            onDateRange={setRegionDateRange}
          />
        )
      }

      {
        active === 'settings' && (
          <Suspense fallback={<ViewLoading label="Loading settings..." />}>
            <SettingsPage
              settings={settings}
              config={config}
              effectiveDate={effectiveDate}
              selectedRegion={selectedRegion}
              baselineDays={baselineDays}
              availableRegions={availableRegions}
              apiBaseUrl={API_BASE}
            />
          </Suspense>
        )
      }
      {/* Weather Strip & Forecast Table now inside ForecastPage */}

      </div>{/* end .workspace */}

      {/* Context Panel — collapsible right sidebar */}
      <ContextPanel
        open={contextOpen}
        onClose={() => setContextOpen(false)}
        liveMeta={live?.metadata}
        driverContributions={live?.driver_contributions || []}
        decisionSignals={live?.decision_signals || []}
        forecastUncertainty={live?.forecast_uncertainty}
        liveDecisionRows={liveDecisionRows}
        liveSensitivityRows={liveSensitivityRows}
        fmt={fmt}
      />
      </div>{/* end .nexus-body */}
      </div>{/* end .main-area */}
      {pipelineJobId && (
        <PipelineProgress
          jobId={pipelineJobId}
          apiBase={TRAINING_API_BASE}
          onResult={(data) => {
            if (data && !data.error) setDayAhead(data);
            setPipelineJobId(null);
          }}
        />
      )}
    </div>
  );
}
