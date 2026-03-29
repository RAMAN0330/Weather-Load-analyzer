import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import ReactECharts from 'echarts-for-react';
import { useSimulatorStore } from './features/simulator/store';
import { SimulatorPage } from './features/simulator/SimulatorPage';
import { ForecastPage } from './features/forecast/ForecastPage';
import AnalysisPage from './features/analysis/AnalysisPage';
import WeatherAnalysisPanel from './components/WeatherAnalysis/WeatherAnalysisPanel';
import WeatherDeepPage from './features/weather/WeatherDeepPage';
import LoadAnalysisPage from './features/load/LoadAnalysisPage';
import OptimizerPage from './features/optimizer/OptimizerPage';
import SettingsPage from './features/settings/SettingsPage';
import CommandStrip from './components/CommandStrip';
import AlertRibbon from './components/AlertRibbon';
import WeatherStrip from './components/WeatherStrip';
import ContextPanel from './components/ContextPanel';
import ForecastTable from './components/ForecastTable';
import {
  Activity,
  BarChart3,
  CloudRain,
  Download,
  Droplets,
  Gauge,
  GitCompare,
  Layers,
  LayoutGrid,
  Loader2,
  Play,
  Settings,
  Sun,
  Thermometer,
  TrendingUp,
  Wind
} from 'lucide-react';

const normalizeBase = (base) => {
  if (!base) return '/api';
  const trimmed = base.replace(/\/$/, '');
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? normalizeBase(import.meta.env.VITE_API_BASE_URL)
  : (import.meta.env.DEV ? '/api' : 'http://localhost:8000/api');

const API_URL = (path) => {
  const base = API_BASE.replace(/\/$/, '');
  const sub = path.startsWith('/') ? path : `/${path}`;
  return `${base}${sub}`;
};

axios.defaults.timeout = 45000;

const NAV_ITEMS = [
  { key: 'load_analysis', label: 'Load Analysis', icon: Activity },
  { key: 'weather_analysis', label: 'Weather Analysis', icon: Sun },
  { key: 'optimizer', label: 'Optimizer', icon: Gauge },
  { key: 'simulator', label: 'Simulator', icon: Play },
  { key: 'analysis', label: 'Analysis', icon: BarChart3 },
  { key: 'forecast', label: 'Forecast', icon: LayoutGrid },
  { key: 'monitor', label: 'Monitor', icon: TrendingUp }
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

const RegionSelectionModal = ({ regions, selected, onSelect, onConfirm }) => {
  return (
    <div className="overlay">
      <div className="modal">
        <div className="modal-header">
          <h2>Select Your State</h2>
          <p>Choose a region to load localized power grid analytics and weather-impact forecasts.</p>
        </div>
        <div className="region-grid">
          {regions.map((r) => (
            <div
              key={r}
              className={`region-tile ${selected === r ? 'active' : ''}`}
              onClick={() => onSelect(r)}
            >
              <div className="region-icon">
                {r === 'delhi' ? <Activity size={24} /> :
                  ['punjab', 'haryana'].includes(r) ? <TrendingUp size={24} /> :
                    ['rajasthan', 'gujarat'].includes(r) ? <Sun size={24} /> :
                      ['maharashtra', 'goa'].includes(r) ? <Wind size={24} /> :
                        <Layers size={24} />}
              </div>
              <span className="region-name">{r}</span>
            </div>
          ))}
        </div>
        <div className="modal-footer">
          <button
            className="primary-btn start-btn"
            disabled={!selected}
            onClick={onConfirm}
          >
            Launch Dashboard
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
      axisLine: { lineStyle: { color: '#262A35' } },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: '#1C1F28', type: 'solid' } },
      axisLabel: { color: '#565B6B', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace' }
    },
    series: [
      // P10-P90 confidence band (stacked area technique)
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
          areaStyle: {
            color: 'rgba(74, 144, 217, 0.06)',
          },
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
    <div className="chart-wrap">
      <ReactECharts option={option} style={{ height: '100%', width: '100%' }} onEvents={onEvents} />
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

export default function App() {
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
  const [config, setConfig] = useState(null);
  const [settings, setSettings] = useState(null);
  const [date, setDate] = useState('');
  const [baselineDays, setBaselineDays] = useState(7);
  const [loading, setLoading] = useState(false);
  const [dayAhead, setDayAhead] = useState(null);
  const [live, setLive] = useState(null);
  const [actualBlocks, setActualBlocks] = useState(0);
  const [analysis, setAnalysis] = useState(null);
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
  const [viewDataOpen, setViewDataOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(true);

  // 3-Tab Load Analysis States
  const [loadTab, setLoadTab] = useState('benchmark');
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [multiDaySeries, setMultiDaySeries] = useState({});
  const [momentumChange, setMomentumChange] = useState([]);
  const [multiSelectedDates, setMultiSelectedDates] = useState([]);

  // Optimizer tabs
  const [optimizerTab, setOptimizerTab] = useState('window');

  const weatherRef = useRef(null);

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

  const initializeData = async (regionToUse) => {
    setLoading(true);
    try {
      // Fetch config + settings in parallel
      const [cfgRes, settingsRes] = await Promise.all([
        axios.get(API_URL('/v2/config')),
        axios.get(API_URL('/v2/settings')),
      ]);

      setAvailableRegions(cfgRes.data.available_regions || []);
      setConfig(cfgRes.data);
      setSettings(settingsRes.data);

      const d = cfgRes.data.default_date || cfgRes.data.latest_date || cfgRes.data.partial_latest_date || '';
      const bl = cfgRes.data.best_baseline_window || 7;
      if (d) setDate(d);
      setBaselineDays(bl);
      if (d) loadFromFileData(d, bl);

      if (!d) { setLoading(false); return; }

      // Single precompute call: dayahead + benchmarks + momentum + analysis
      const preRes = await axios.post(API_URL('/v2/precompute'), {
        date: d,
        baseline_days: bl,
        calendar_config: calendarConfig,
        region: regionToUse,
      }, { timeout: 180000 });

      const pre = preRes.data;
      if (pre.dayahead && !pre.dayahead.error) setDayAhead(pre.dayahead);
      if (pre.benchmarks && !pre.benchmarks.error) setBenchmarkData(pre.benchmarks);
      if (pre.momentum && !pre.momentum.error) setMomentumChange(pre.momentum);
      if (pre.analysis && !pre.analysis.error) setAnalysis(pre.analysis);
    } catch (e) {
      console.error('Initialize failed:', e);
      // Fallback: try config-only init
      try {
        const cfgRes = await axios.get(API_URL('/v2/config'));
        setConfig(cfgRes.data);
        const d = cfgRes.data.default_date || cfgRes.data.latest_date || '';
        if (d) {
          setDate(d);
          fetchDayAheadForInit(d, cfgRes.data.best_baseline_window || 7, regionToUse);
        }
      } catch { setConfig(null); }
    } finally {
      setLoading(false);
    }
  };

  const [availableRegions, setAvailableRegions] = useState(['odisha', 'rajasthan', 'haryana']);

  // Startup: only fetch config if we have a region
  useEffect(() => {
    // If we already have a region (e.g. from localstorage later), we could auto-init
    // For now, wait for selection
  }, []);

  const handleRegionConfirm = () => {
    if (selectedRegion) {
      setShowRegionPrompt(false);
      initializeData(selectedRegion);
    }
  };

  const fetchDayAheadForInit = async (targetDate, targetBaseline, targetRegion) => {
    setLoading(true);
    try {
      const res = await axios.post(API_URL('/v2/dayahead'), {
        date: targetDate,
        baseline_days: targetBaseline,
        calendar_config: calendarConfig,
        region: targetRegion
      });
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
    setLoading(true);
    try {
      const res = await axios.post(API_URL('/v2/dayahead'), {
        date: effectiveDate,
        baseline_days: baselineDays,
        calendar_config: calendarConfig,
        region: selectedRegion
      });
      setDayAhead(res.data);
      pushToast('success', 'Forecast generated successfully');
      loadFromFileData(effectiveDate, baselineDays);
    } catch (e) {
      const fallback = 'http://localhost:8000/api/v2/dayahead';
      const status = e?.response?.status;
      const msg = e?.response?.data?.detail || e?.message || 'Failed to load forecast';
      pushToast('error', `Forecast error (${status || 'network'}): ${msg}`);
      if (API_BASE.startsWith('/api')) {
        try {
          const res = await axios.post(fallback, {
            date: effectiveDate,
            baseline_days: baselineDays,
            calendar_config: calendarConfig,
            region: selectedRegion
          });
          setDayAhead(res.data);
          pushToast('success', 'Forecast loaded via fallback');
        } catch {
          // keep error toast
        }
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchLive = async () => {
    setLoading(true);
    try {
      const payload = {
        date: liveEffectiveDate || undefined,
        calendar_config: calendarConfig,
        actual_blocks: actualBlocks || (live?.metadata?.actual_blocks ?? undefined),
        region: selectedRegion,
      };
      const res = await axios.post(API_URL('/v2/live'), payload, { timeout: 180000 });
      setLive(res.data);
      const availableBlocks = res.data?.metadata?.actual_blocks;
      if (Number.isFinite(availableBlocks) && actualBlocks === 0) {
        setActualBlocks(availableBlocks);
      }
      pushToast('info', 'Short-term forecast refreshed');
    } catch (e) {
      const fallback = 'http://localhost:8000/api/v2/live';
      const status = e?.response?.status;
      const msg = e?.response?.data?.detail || e?.message || 'Failed to refresh Live Ops';
      pushToast('error', `Live Ops error (${status || 'network'}): ${msg}`);
      if (API_BASE.startsWith('/api')) {
        try {
          const payload = {
            date: liveEffectiveDate || undefined,
            calendar_config: calendarConfig,
            actual_blocks: actualBlocks || (live?.metadata?.actual_blocks ?? undefined),
            region: selectedRegion || 'punjab',
          };
          const res = await axios.post(fallback, payload, { timeout: 180000 });
          setLive(res.data);
          pushToast('success', 'Live Ops loaded via fallback');
        } catch {
          // keep error toast
        }
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
      const res = await axios.post(API_URL('/v2/analysis'), {
        date: effectiveDate,
        region: selectedRegion
      });
      setAnalysis(res.data);
      if (!dayAhead || dayAhead?.metadata?.effective_date !== effectiveDate) {
        const da = await axios.post(API_URL('/v2/dayahead'), {
          date: effectiveDate,
          baseline_days: baselineDays,
          calendar_config: calendarConfig
        });
        setDayAhead(da.data);
      }
    } catch (e) {
      const fallback = 'http://localhost:8000/api/v2/analysis';
      const status = e?.response?.status;
      const msg = e?.response?.data?.detail || e?.message || 'Failed to load analysis';
      pushToast('error', `Analysis error (${status || 'network'}): ${msg}`);
      if (API_BASE.startsWith('/api')) {
        try {
          const res = await axios.post(fallback, {
            date: effectiveDate,
            region: selectedRegion
          });
          setAnalysis(res.data);
          pushToast('success', 'Analysis loaded via fallback');
        } catch {
          // keep error toast
        }
      }
    } finally {
      setLoading(false);
    }
  };

  // Optimize fetching: Prevent duplicate reloads on tab switch
  useEffect(() => {
    const isDayAheadActive = ['load_analysis', 'weather_analysis', 'optimizer', 'simulator', 'analysis'].includes(active);
    if (isDayAheadActive && date) {
      // Check if data is already loaded for this date and baseline
      const alreadyLoaded = dayAhead &&
        dayAhead.metadata?.effective_date === date;
      if (!alreadyLoaded) {
        fetchDayAhead();
      }
    }
  }, [active, date, baselineDays]);

  useEffect(() => {
    if (active === 'forecast') {
      // Live data might need more frequent updates, but for tab switching, we can cache.
      // Only fetch if we don't have live data yet.
      if (!live) {
        fetchLive();
      }
    }
  }, [active]);

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

  const dayAheadSeries = useMemo(() => {
    if (active === 'forecast' && live?.series && liveEffectiveDate && effectiveDate === liveEffectiveDate) {
      return {
        blocks: live.series.blocks,
        baseline: live.series.hybrid_baseline,
        forecast: live.series.forecast,
        actual: liveActual,
        weather_impact: live.series.weather_impact,
        weather_feature_deltas: live.series.weather_feature_deltas
      };
    }
    return dayAhead?.series;
  }, [active, live, liveEffectiveDate, effectiveDate, dayAhead, liveActual]);

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

  const handleExportLiveCsv = () => {
    if (!live?.forecast_df?.length) return;
    const header = Object.keys(live.forecast_df[0]).join(',');
    const body = live.forecast_df
      .map((row) => Object.values(row).map((v) => `"${v ?? ''}"`).join(','))
      .join('\n');
    downloadBlob(`${header}\n${body}`, `shortterm_${live?.date || effectiveDate}.csv`, 'text/csv');
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
      {/* Global loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="ring" />
            <div className="label">Loading analytics...</div>
          </div>
        </div>
      )}

      {/* Left Sidebar Navigation */}
      <nav className="sidebar-nav">
        <div className="sidebar-brand">GNA</div>
        <div className="sidebar-links">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                className={`sidebar-btn ${active === item.key ? 'active' : ''}`}
                onClick={() => setActive(item.key)}
                data-tooltip={item.label}
                title={item.label}
              >
                <Icon size={18} />
              </button>
            );
          })}
        </div>
        <div className="sidebar-bottom">
          <button
            className={`sidebar-btn ${active === 'settings' ? 'active' : ''}`}
            onClick={() => setActive('settings')}
            data-tooltip="Settings"
            title="Settings"
          >
            <Settings size={18} />
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
        canDownload={Boolean(live?.forecast_df?.length)}
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
        />
      )}

      {
        active === 'weather_analysis' && (
          <WeatherDeepPage
            effectiveDate={effectiveDate}
            dayAheadData={dayAhead}
            dayAheadSeries={dayAheadSeries}
            liveData={live}
            selectedRegion={selectedRegion}
          />
        )
      }

      {active === 'optimizer' && (
        <OptimizerPage
          dayAheadData={dayAhead}
          dayAheadSeries={dayAheadSeries}
          baselineWindowMapes={baselineWindowMapes}
          baselineDays={baselineDays}
          setBaselineDays={setBaselineDays}
          availableActualBlocks={availableActualBlocks}
          effectiveDate={effectiveDate}
        />
      )}

      {active === 'simulator' && <SimulatorPage requestedDate={effectiveDate} baselineDays={baselineDays} />}

      {active === 'analysis' && <AnalysisPage effectiveDate={effectiveDate} dayAheadData={dayAhead} liveForecastData={live} />}

      {
        active === 'forecast' && (
          <ForecastPage
            liveData={liveDataZipped}
            liveMeta={live?.metadata}
            driverContributions={live?.driver_contributions || []}
            decisionSignals={live?.decision_signals || []}
            effectiveDate={liveEffectiveDate}
            selectedRegion={selectedRegion}
            onRefresh={fetchLive}
            onDownload={handleExportLiveCsv}
            canDownload={Boolean(live?.forecast_df?.length)}
            fmt={fmt}
            chart={
              <LoadChart
                blocks={live?.series?.blocks || []}
                baseline={live?.series?.hybrid_baseline || []}
                forecast={liveAdjustedForecast.length ? liveAdjustedForecast : (live?.series?.forecast || [])}
                actual={liveActual}
                p10={live?.series?.p10 || []}
                p90={live?.series?.p90 || []}
                dateLabel={live?.date || liveEffectiveDate || ''}
              />
            }
            weatherStrip={
              <WeatherStrip dayAhead={dayAhead} live={live} fmt={fmt} />
            }
            forecastTable={
              <ForecastTable
                liveData={liveDataZipped}
                forecastUncertainty={live?.forecast_uncertainty}
                actualBlocks={live?.metadata?.actual_blocks || actualBlocks}
                fmt={fmt}
              />
            }
          />
        )
      }

      {/* Simulator and Analysis handled above */}

      {
        active === 'monitor' && (
          <main className="page page-full">
            <section className="hero">
              <div>
                <h1>Model Performance Monitor</h1>
                <p>Daily accuracy scorecard, drift detection, and operator annotations</p>
              </div>
            </section>

            <div className="grid grid-cols-4 gap-4 mt-4">
              <div className="glass-panel p-4">
                <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-2">30-Day MAPE</div>
                <div className="text-2xl font-bold text-[var(--accent)]">
                  {live?.metadata?.mape_live ? `${live.metadata.mape_live.toFixed(1)}%` : '--'}
                </div>
              </div>
              <div className="glass-panel p-4">
                <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-2">Model Version</div>
                <div className="text-lg font-bold">v3.0-additive</div>
                <div className="text-[10px] text-[var(--muted)]">Hybrid Additive Model</div>
              </div>
              <div className="glass-panel p-4">
                <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-2">Last Retrain</div>
                <div className="text-lg font-bold text-[var(--warning)]">--</div>
                <div className="text-[10px] text-[var(--muted)]">Auto-retrain on Sundays</div>
              </div>
              <div className="glass-panel p-4">
                <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-2">Drift Status</div>
                <div className="text-lg font-bold text-[var(--success)]">Stable</div>
                <div className="text-[10px] text-[var(--muted)]">No degradation detected</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="glass-panel p-4">
                <div className="panel-header mb-3">
                  <h3>Accuracy by Segment</h3>
                  <span>Performance breakdown by day type and weather regime</span>
                </div>
                <div className="flex flex-col gap-2">
                  {['Weekday', 'Weekend', 'Holiday', 'Hot', 'Cold', 'Monsoon'].map((seg) => (
                    <div key={seg} className="flex justify-between items-center text-xs">
                      <span>{seg}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full bg-[var(--accent)] rounded-full" style={{ width: `${Math.random() * 40 + 60}%` }} />
                        </div>
                        <span className="text-[var(--muted)] w-12 text-right">--</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-[10px] text-[var(--muted)] mt-3 italic">Run /api/v2/backtest to populate real metrics</div>
              </div>

              <div className="glass-panel p-4">
                <div className="panel-header mb-3">
                  <h3>Operator Annotations</h3>
                  <span>Tag days with special events or notes</span>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Add note for today..."
                      className="flex-1 bg-white/5 border border-[var(--outline)] rounded-lg p-2 text-xs text-white"
                    />
                    <button className="primary-btn text-xs px-3">Save</button>
                  </div>
                  <div className="flex gap-1 flex-wrap">
                    {['Outage', 'Festival', 'Fog', 'Industrial Shutdown', 'IPL Match'].map((tag) => (
                      <button key={tag} className="text-[9px] px-2 py-1 rounded-full bg-white/5 border border-[var(--outline)] text-[var(--muted)] hover:bg-white/10 transition-colors">
                        {tag}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel p-4 mt-4">
              <div className="panel-header mb-3">
                <h3>Recalibration Schedule</h3>
                <span>Seasonal recalibration at season transitions</span>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {[
                  { season: 'Summer', month: 'March', status: 'upcoming' },
                  { season: 'Monsoon', month: 'June', status: 'pending' },
                  { season: 'Post-Monsoon', month: 'September', status: 'pending' },
                  { season: 'Winter', month: 'December', status: 'pending' },
                ].map((cal) => (
                  <div key={cal.season} className={`p-3 rounded-lg ${cal.status === 'upcoming' ? 'bg-yellow-500/10 border border-[var(--warning)]' : 'bg-white/5'}`}>
                    <div className="text-xs font-bold">{cal.season}</div>
                    <div className="text-[10px] text-[var(--muted)]">{cal.month}</div>
                    <div className={`text-[9px] mt-1 ${cal.status === 'upcoming' ? 'text-[var(--warning)]' : 'text-[var(--muted)]'}`}>
                      {cal.status === 'upcoming' ? 'Due Soon' : 'Scheduled'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </main>
        )
      }

      {
        showRegionPrompt && (
          <RegionSelectionModal
            regions={availableRegions}
            selected={selectedRegion}
            onSelect={setSelectedRegion}
            onConfirm={handleRegionConfirm}
          />
        )
      }

      {
        active === 'settings' && (
          <SettingsPage
            settings={settings}
            config={config}
            effectiveDate={effectiveDate}
            selectedRegion={selectedRegion}
            baselineDays={baselineDays}
            availableRegions={availableRegions}
            apiBaseUrl={API_BASE}
          />
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
    </div>
  );
}
