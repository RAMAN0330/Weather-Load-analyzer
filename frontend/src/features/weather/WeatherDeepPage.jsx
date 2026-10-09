import React, { useState, useMemo, useCallback, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { useChartTokens, withAlpha } from '../../lib/chartTheme';
import axios from 'axios';
import HorizonToggle from '../../components/HorizonToggle';
import ContextPanel from '../../components/ContextPanel';
import { API_BASE, getApiUrl, getTrainingApiUrl } from '../../apiConfig';
import { Sun } from 'lucide-react';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Eyebrow as VpEyebrow,
  Pill as VpPill,
} from '../../components/page/PagePrimitives.jsx';

/* ═══════════════════════════════════════════════════════════════
   VidyutPragya — Weather Intelligence Page
   Real data from backend, ECharts, spacious dark layout
   ═══════════════════════════════════════════════════════════════ */

const C = {
  bg: 'var(--bg)',
  card: 'var(--bg-elevated)',
  surface: 'var(--bg-surface)',
  border: 'var(--outline)',
  accent: 'var(--accent)',
  accent2: 'var(--accent2)',
  text: 'var(--text)',
  sub: 'var(--text-secondary)',
  muted: 'var(--text-muted)',
  green: 'var(--success)',
  red: 'var(--danger)',
  warn: 'var(--warning)',
  info: 'var(--info)',
  warm: 'var(--tone-warm)',
  // Charts (canvas) can't use CSS vars — they use useChartTokens() instead.
};

const blockToTime = (b) => {
  const m = (b - 1) * 15;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};
const fmt = (v, d = 1) =>
  v == null || !Number.isFinite(v)
    ? '--'
    : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });
const sgn = (v) => (v >= 0 ? '+' : '') + fmt(v, 1);

// API URL builders moved to apiConfig.ts
const API_URL = getApiUrl;
const TRAINING_API_URL = getTrainingApiUrl;
const WEATHER_API_BASE = API_BASE;

/* ─── Spearman ─── */
function spearman(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 4) return 0;
  const rank = (arr) => {
    const s = arr
      .slice(0, n)
      .map((v, i) => ({ v, i }))
      .sort((a, b) => a.v - b.v);
    const r = new Array(n);
    let i = 0;
    while (i < n) {
      let j = i;
      while (j < n - 1 && s[j + 1].v === s[j].v) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[s[k].i] = avg;
      i = j + 1;
    }
    return r;
  };
  const rx = rank(x),
    ry = rank(y);
  const mx = rx.reduce((s, v) => s + v, 0) / n,
    my = ry.reduce((s, v) => s + v, 0) / n;
  let num = 0,
    dx = 0,
    dy = 0;
  for (let i = 0; i < n; i++) {
    const a = rx[i] - mx,
      b = ry[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  return dx === 0 || dy === 0 ? 0 : num / Math.sqrt(dx * dy);
}

/* ─── MW Attribution Coefficients (Odisha) ─── */
const MW_COEFS = {
  temperature: {
    coef: 32,
    threshold: 28,
    label: 'Temperature',
    unit: 'MW/°C',
    calc: (t) => 32 * Math.max(0, t - 28),
  },
  humidity: {
    coef: 7,
    threshold: 60,
    label: 'Humidity',
    unit: 'MW/%',
    calc: (h) => 7 * Math.max(0, h - 60),
  },
  cloud: {
    coef: -3,
    threshold: 0,
    label: 'Cloud Cover',
    unit: 'MW/%',
    calc: (c) => -3 * (c / 100) * 100,
  },
  rain: {
    coef: -18,
    threshold: 0,
    label: 'Rain',
    unit: 'MW/mm',
    calc: (r) => -18 * Math.min(Math.max(0, r), 20),
  },
  wind: {
    coef: -1.8,
    threshold: 20,
    label: 'Wind',
    unit: 'MW/(km/h)',
    calc: (w) => -1.8 * Math.max(0, w - 20),
  },
};

/* ─── Regime classifier ─── */
const classifyRegime = (tempAvg, hum, rain, cloud) => {
  if (tempAvg > 42) return { name: 'Extreme Heat', color: C.red, badge: 'bg-danger' };
  if (rain > 10 && cloud > 70) return { name: 'Rainy', color: C.accent2, badge: 'bg-info' };
  if (tempAvg > 38 && hum < 35) return { name: 'Hot & Dry', color: C.warm, badge: 'bg-warn' };
  if (tempAvg > 35 && hum > 65) return { name: 'Hot & Humid', color: C.red, badge: 'bg-danger' };
  if (tempAvg < 28 && cloud > 60)
    return { name: 'Cool & Cloudy', color: C.info, badge: 'bg-info' };
  return { name: 'Normal', color: C.green, badge: 'bg-ok' };
};

/* ─── ECharts base theme ─── */
const ecBase = (tk) => ({
  backgroundColor: 'transparent',
  textStyle: { color: tk.textSecondary, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 },
  grid: { top: 62, right: 28, bottom: 72, left: 56, containLabel: true },
  tooltip: {
    trigger: 'axis',
    backgroundColor: tk.elevated,
    borderColor: tk.outline,
    borderWidth: 1,
    textStyle: { color: tk.text, fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" },
    extraCssText: 'box-shadow: 0 8px 24px rgba(var(--shadow-rgb), 0.25); border-radius: 8px; padding: 8px 12px;',
  },
  legend: {
    textStyle: { color: tk.textSecondary, fontSize: 10 },
    top: 10,
    left: 'center',
    itemWidth: 14,
    itemHeight: 3,
  },
  xAxis: {
    type: 'category',
    axisLine: { lineStyle: { color: tk.outline } },
    axisLabel: { color: tk.textMuted, fontSize: 9, margin: 16 },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    nameGap: 22,
    axisLine: { show: false },
    axisLabel: { color: tk.textMuted, fontSize: 9 },
    splitLine: { lineStyle: { color: tk.outline, type: 'dashed', opacity: 0.3 } },
  },
});

/* ─── Collapsible Section ─── */
function Section({ title, children, defaultOpen = true, accent }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section style={{ marginBottom: 6 }}>
      <div
        onClick={() => setOpen(!open)}
        className="weather-section-header"
        style={{
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '14px 20px',
          background: C.card,
          borderBottom: `1px solid ${open ? 'var(--outline)' : 'transparent'}`,
          userSelect: 'none',
        }}
      >
        <span
          style={{
            fontSize: 10,
            opacity: 0.5,
            transition: 'transform 0.2s',
            transform: open ? 'rotate(90deg)' : 'rotate(0)',
            display: 'inline-block',
          }}
        >
          ▶
        </span>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: 2,
            color: accent || C.accent,
          }}
        >
          {title}
        </span>
      </div>
      {open && <div style={{ padding: '16px 20px', background: C.card }}>{children}</div>}
    </section>
  );
}

/* ─── Metric Card ─── */
function MetricCard({ label, value, unit, sub, delta, color, wide }) {
  const dColor = delta > 0 ? C.red : delta < 0 ? C.green : C.muted;
  return (
    <div
      className="metric-card"
      style={{
        flex: wide ? '1 1 240px' : '1 1 200px',
        minWidth: 180,
        minHeight: 124,
        padding: '16px 18px',
        background: 'linear-gradient(180deg, var(--bg-panel), var(--bg-surface))',
        borderRadius: 14,
        border: `1px solid var(--outline)`,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        gap: 10,
      }}
    >
      <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: C.muted }}>
        {label}
      </div>
      <div>
        <div style={{ fontSize: 28, fontWeight: 700, color: color || C.text, lineHeight: 1.05 }}>
          {value}{' '}
          {unit && <span style={{ fontSize: 12, fontWeight: 500, opacity: 0.6 }}>{unit}</span>}
        </div>
        {delta != null && Number.isFinite(delta) && (
          <div style={{ fontSize: 10, color: dColor, marginTop: 6, fontWeight: 600 }}>
            {delta >= 0 ? '▲' : '▼'} {sgn(delta)} vs yesterday
          </div>
        )}
      </div>
      {sub ? <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.45 }}>{sub}</div> : <div />}
    </div>
  );
}

function DetailOverlay({ open, title, subtitle, onClose, children }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content fp-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            {subtitle && <p className="fp-modal-subtitle">{subtitle}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: C.muted,
              cursor: 'pointer',
              fontSize: 18,
            }}
          >
            ×
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

/* ─── Gauge SVG ─── */
function ArcGauge({ value, min = -1, max = 1, label, sublabel, size = 110 }) {
  const tk = useChartTokens();
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const absVal = Math.abs(value);
  const color = absVal > 0.6 ? tk.success : absVal > 0.3 ? tk.warning : tk.danger;
  const strength =
    absVal > 0.7 ? 'Strong' : absVal > 0.4 ? 'Moderate' : absVal > 0.2 ? 'Weak' : 'Very Weak';
  const dir = value > 0.05 ? 'Positive' : value < -0.05 ? 'Negative' : 'None';
  const cx = size / 2,
    cy = size * 0.62,
    r = size * 0.36;
  const needleAngle = Math.PI - pct * Math.PI;
  const nx = cx + (r - 4) * Math.cos(needleAngle),
    ny = cy - (r - 4) * Math.sin(needleAngle);
  const arc = (a1, a2) => {
    const x1 = cx + r * Math.cos(a1),
      y1 = cy - r * Math.sin(a1),
      x2 = cx + r * Math.cos(a2),
      y2 = cy - r * Math.sin(a2);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  return (
    <div style={{ textAlign: 'center', flex: '1 1 100px', minWidth: 90 }}>
      {sublabel && (
        <div
          style={{
            fontSize: 8,
            color: C.muted,
            letterSpacing: 1.2,
            textTransform: 'uppercase',
            marginBottom: 2,
          }}
        >
          {sublabel}
        </div>
      )}
      <svg width={size} height={size * 0.68} viewBox={`0 0 ${size} ${size * 0.68}`}>
        <path
          d={arc(Math.PI, 0)}
          fill="none"
          stroke={withAlpha(tk.textMuted, 0.15)}
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d={arc(Math.PI, needleAngle)}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          opacity="0.85"
        />
        <circle cx={nx} cy={ny} r="4.5" fill={color} />
        <circle cx={nx} cy={ny} r="2" fill={tk.panel} />
        <text
          x={cx}
          y={cy + 3}
          textAnchor="middle"
          fill={tk.text}
          fontSize="14"
          fontWeight="700"
          fontFamily="'IBM Plex Mono',monospace"
        >
          {value.toFixed(2)}
        </text>
      </svg>
      <div style={{ fontSize: 9, color, fontWeight: 600 }}>
        {strength} {dir}
      </div>
      {label && <div style={{ fontSize: 8, color: C.muted, marginTop: 1 }}>{label}</div>}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function WeatherDeepPage({
  effectiveDate,
  dayAheadData,
  dayAheadSeries,
  liveData,
  selectedRegion,
  horizon = 't1',
  setHorizon,
  t2Date,
  dayAheadT2,
}) {
  const [chartTab, setChartTab] = useState('intraday');
  const [weatherFeature, setWeatherFeature] = useState('temperature');
  const [showNormal, setShowNormal] = useState(true);
  const [showLoad, setShowLoad] = useState(true);
  const [expandedFactor, setExpandedFactor] = useState(null);
  const [activePanel, setActivePanel] = useState(null);
  // Multi-date comparison for DoD
  const [dodDates, setDodDates] = useState([]);
  const [dodDateData, setDodDateData] = useState({}); // { date: { temperature: [...96], humidity: [...96], ... } }
  const [dodLoading, setDodLoading] = useState(false);
  const [fallbackDayAhead, setFallbackDayAhead] = useState(null);
  const [fallbackLoading, setFallbackLoading] = useState(false);
  const [fallbackError, setFallbackError] = useState('');
  const [hasTriedFallback, setHasTriedFallback] = useState(false);
  const tk = useChartTokens();
  // Pipeline DB weather state

  const loadWeatherFallback = useCallback(async () => {
    if (!effectiveDate) return;
    setHasTriedFallback(true);
    setFallbackLoading(true);
    setFallbackError('');
    try {
      const res = await axios.post(
        TRAINING_API_URL('/v2/dayahead'),
        {
          date: effectiveDate,
          baseline_days: 7,
          region: selectedRegion || 'odisha',
        },
        { timeout: 60000 }
      );
      setFallbackDayAhead(res.data || null);
    } catch (e) {
      setFallbackError(
        e?.response?.data?.detail || e?.message || 'Unable to load weather analysis.'
      );
    } finally {
      setFallbackLoading(false);
    }
  }, [effectiveDate, selectedRegion]);

  useEffect(() => {
    setFallbackDayAhead(null);
    setFallbackError('');
    setHasTriedFallback(false);
  }, [effectiveDate, selectedRegion]);

  useEffect(() => {
    // Only skip fallback when weather_analysis is actually present — series alone is not enough
    if (dayAheadData?.weather_analysis) {
      setFallbackDayAhead(null);
      setFallbackError('');
      return;
    }
    if (fallbackDayAhead?.weather_analysis) return;
    if (!effectiveDate || fallbackLoading || hasTriedFallback) return;
    loadWeatherFallback();
  }, [
    dayAheadData,
    effectiveDate,
    loadWeatherFallback,
    fallbackLoading,
    fallbackDayAhead,
    hasTriedFallback,
  ]);

  const addDodDate = useCallback(
    async (date) => {
      if (!date || dodDates.includes(date) || date === effectiveDate) return;
      setDodDates((prev) => [...prev, date]);
      if (dodDateData[date]) return; // already fetched
      setDodLoading(true);
      try {
        const res = await axios.post(
          TRAINING_API_URL('/v2/dayahead'),
          {
            date,
            baseline_days: 7,
            region: selectedRegion || 'odisha',
          },
          { timeout: 60000 }
        );
        const intra = res.data?.weather_analysis?.intraday;
        if (intra) {
          setDodDateData((prev) => ({
            ...prev,
            [date]: {
              temperature: intra.temperature?.actual || [],
              humidity: intra.humidity?.actual || [],
              precipitation: intra.precipitation?.actual || [],
              cloud_cover: intra.cloud_cover?.actual || [],
              solar_radiation: intra.solar_radiation?.actual || [],
              load: res.data?.series?.actual || [],
            },
          }));
        }
      } catch (e) {
        console.warn('Failed to fetch weather for', date, e);
      }
      setDodLoading(false);
    },
    [dodDates, dodDateData, effectiveDate, selectedRegion]
  );

  const removeDodDate = useCallback((date) => {
    setDodDates((prev) => prev.filter((d) => d !== date));
  }, []);

  // ─── Extract real data ───
  // When T+2 is active, prefer dayAheadT2 (fetched for t2_date); fall back to fallbackDayAhead.
  const resolvedDayAhead =
    horizon === 't2' ? dayAheadT2 || fallbackDayAhead : dayAheadData || fallbackDayAhead;
  const wa = resolvedDayAhead?.weather_analysis;
  const intra = wa?.intraday;
  const series = dayAheadSeries || resolvedDayAhead?.series;
  const meta = resolvedDayAhead?.metadata;
  const kpis = resolvedDayAhead?.kpis_full;
  const sensitivity = wa?.sensitivity;
  const peakWindows = wa?.peak_windows;
  const rainMetrics = wa?.rain_metrics;
  const dod = wa?.dod_changes;
  const dodSeries = wa?.dod_series;
  const weatherImpact = series?.weather_impact || [];
  const weatherImpactPct = series?.weather_impact_pct || [];
  const blocks = series?.blocks || Array.from({ length: 96 }, (_, i) => i + 1);
  const timeLabels = blocks.map(blockToTime);

  // ─── Available weather features ───
  const WEATHER_FEATURES = [
    { key: 'temperature', label: 'Temperature', unit: '°C', color: tk.warm },
    { key: 'humidity', label: 'Humidity', unit: '%', color: tk.info },
    { key: 'precipitation', label: 'Precipitation', unit: 'mm', color: tk.accent2 },
    { key: 'cloud_cover', label: 'Cloud Cover', unit: '%', color: tk.textMuted },
    { key: 'solar_radiation', label: 'Solar Radiation', unit: 'W/m²', color: tk.warning },
  ];

  // ─── Intraday arrays ───
  const getIntra = (feature, type = 'actual') => intra?.[feature]?.[type] || [];
  const tempActual = getIntra('temperature');
  const tempNormal = getIntra('temperature', 'normal');
  const tempDelta = getIntra('temperature', 'delta');
  const humActual = getIntra('humidity');
  const precipActual = getIntra('precipitation');
  const cloudActual = getIntra('cloud_cover');
  const solarActual = getIntra('solar_radiation');
  const loadForecast = series?.forecast || [];
  // Use real actuals when available; fall back to forecast for future-date correlation analysis
  const loadActual = (series?.actual?.length ? series.actual : loadForecast);
  const loadBaseline = series?.baseline || [];

  // ─── Computed stats from real data ───
  const computed = useMemo(() => {
    const clean = (arr) => arr.filter((v) => Number.isFinite(v) && v !== 0);
    const mean = (arr) => {
      const c = clean(arr);
      return c.length ? c.reduce((s, v) => s + v, 0) / c.length : 0;
    };
    const max = (arr) => {
      const c = clean(arr);
      return c.length ? Math.max(...c) : 0;
    };
    const min = (arr) => {
      const c = clean(arr);
      return c.length ? Math.min(...c) : 0;
    };

    const avgTemp = mean(tempActual);
    const maxTemp = max(tempActual);
    const minTemp = min(tempActual);
    const avgHum = mean(humActual);
    const totalPrecip = precipActual.reduce((s, v) => s + (v || 0), 0);
    const avgCloud = mean(cloudActual);
    const avgSolar = mean(solarActual);
    const avgWind = 0; // not in intraday — fallback

    const cdd = Math.max(0, avgTemp - 24);
    const hdd = Math.max(0, 18 - avgTemp);
    const regime = classifyRegime(maxTemp, avgHum, totalPrecip, avgCloud);

    // MW attributions
    const mwTemp = MW_COEFS.temperature.calc(avgTemp);
    const mwHum = MW_COEFS.humidity.calc(avgHum);
    const mwCloud = MW_COEFS.cloud.calc(avgCloud, mean(loadBaseline) || 4500);
    const mwRain = MW_COEFS.rain.calc(totalPrecip);
    const mwWind = MW_COEFS.wind.calc(avgWind);
    const totalMW = mwTemp + mwHum + mwCloud + mwRain + mwWind;

    // Compound effects
    const compounds = [];
    if (maxTemp > 35 && avgHum > 65)
      compounds.push({ name: 'Hot + Humid', mult: 1.25, desc: 'AC stress amplified' });
    if (maxTemp > 38 && avgHum < 30 && avgWind < 10)
      compounds.push({ name: 'Hot + Dry + Still', mult: 1.15, desc: 'Desert coolers ineffective' });
    if (totalPrecip > 5 && avgCloud > 70)
      compounds.push({ name: 'Rainy + Cloudy', mult: 0.85, desc: 'Demand suppression' });
    const compoundMult = compounds.reduce((m, c) => m * c.mult, 1.0);
    const netMW = totalMW * compoundMult;

    // Comfort score
    let comfort = 100;
    if (maxTemp > 35) comfort -= (maxTemp - 35) * 6;
    else if (maxTemp > 30) comfort -= (maxTemp - 30) * 3;
    if (avgHum > 70) comfort -= (avgHum - 70) * 0.5;
    if (totalPrecip > 5) comfort -= Math.min(15, totalPrecip * 1.5);
    comfort = Math.max(0, Math.min(100, Math.round(comfort)));

    // Per-block MW attribution (for expandable detail)
    const blockMW = blocks.map((b, i) => {
      const t = tempActual[i] || 0,
        h = humActual[i] || 0,
        c = cloudActual[i] || 0,
        p = precipActual[i] || 0;
      return {
        block: b,
        time: blockToTime(b),
        temp: MW_COEFS.temperature.calc(t),
        hum: MW_COEFS.humidity.calc(h),
        cloud: MW_COEFS.cloud.calc(c),
        rain: MW_COEFS.rain.calc(p),
        rawTemp: t,
        rawHum: h,
        rawCloud: c,
        rawPrecip: p,
      };
    });
    blockMW.forEach((bm) => {
      bm.total = bm.temp + bm.hum + bm.cloud + bm.rain;
    });

    // Net weather impact from series
    const netImpactMW = weatherImpact.reduce((s, v) => s + (Number(v) || 0), 0);
    const peakImpactMW =
      weatherImpact.length > 0
        ? Math.max(...weatherImpact.map((v) => Math.abs(Number(v) || 0)))
        : 0;

    // Spearman correlations
    const corrTemp = spearman(tempActual, loadActual);
    const corrHum = spearman(humActual, loadActual);
    const corrCloud = spearman(cloudActual, loadActual);
    const corrPrecip = spearman(precipActual, loadActual);
    const corrSolar = spearman(solarActual, loadActual);

    // Cross-correlation matrix
    const allFeatures = {
      Temperature: tempActual,
      Humidity: humActual,
      Cloud: cloudActual,
      Precip: precipActual,
      Solar: solarActual,
      Load: loadActual,
    };
    const corrKeys = Object.keys(allFeatures);
    const corrMatrix = {};
    corrKeys.forEach((k1) => {
      corrMatrix[k1] = {};
      corrKeys.forEach((k2) => {
        corrMatrix[k1][k2] = spearman(allFeatures[k1], allFeatures[k2]);
      });
    });

    // Solar Ramp Risk (Rajasthan): if solar drops >30% between blocks 32-48 (8AM-12PM)
    let solarRampRisk = null;
    if (solarActual.length >= 48) {
      const solarWindow = solarActual.slice(31, 48); // blocks 32-48
      const maxSolar = Math.max(...solarWindow);
      const minSolar = Math.min(...solarWindow);
      if (maxSolar > 50) {
        // only flag if there's meaningful solar generation
        const dropPct = ((maxSolar - minSolar) / maxSolar) * 100;
        solarRampRisk = { dropPct: Math.round(dropPct), flag: dropPct > 30 };
      }
    }

    return {
      avgTemp,
      maxTemp,
      minTemp,
      avgHum,
      totalPrecip,
      avgCloud,
      avgSolar,
      cdd,
      hdd,
      regime,
      comfort,
      blockMW,
      mwTemp,
      mwHum,
      mwCloud,
      mwRain,
      mwWind,
      totalMW,
      netMW,
      compoundMult,
      compounds,
      netImpactMW,
      peakImpactMW,
      corrTemp,
      corrHum,
      corrCloud,
      corrPrecip,
      corrSolar,
      corrMatrix,
      corrKeys,
      solarRampRisk,
    };
  }, [
    tempActual,
    humActual,
    precipActual,
    cloudActual,
    solarActual,
    loadActual,
    loadBaseline,
    weatherImpact,
  ]);

  // ─── DoD values ───
  const dodMaxTemp = dod?.max_temp;
  const dodMinTemp = dod?.min_temp;
  const dodHum = dod?.avg_humidity;

  // ─── No data guard ───
  if (!wa && !series) {
    return (
      <div
        style={{
          padding: 40,
          textAlign: 'center',
          color: C.muted,
          fontFamily: "'IBM Plex Mono', monospace",
        }}
      >
        <div style={{ fontSize: 18, marginBottom: 8 }}>
          {fallbackLoading ? 'Preparing weather analysis' : 'Weather analysis is unavailable'}
        </div>
        <div style={{ fontSize: 12, marginBottom: 16 }}>
          {fallbackLoading
            ? 'Fetching day-ahead weather data for this state.'
            : fallbackError || 'Unable to load weather data for the selected date.'}
        </div>
        {!fallbackLoading && (
          <button type="button" className="secondary-btn" onClick={() => loadWeatherFallback()}>
            Retry Weather Data
          </button>
        )}
      </div>
    );
  }

  /* ─── Chart Builders ─── */
  const intradayChartOption = () => {
    const feat = WEATHER_FEATURES.find((f) => f.key === weatherFeature);
    const actual = getIntra(weatherFeature);
    const normal = getIntra(weatherFeature, 'normal');
    const delta = getIntra(weatherFeature, 'delta');
    const opt = {
      ...ecBase(tk),
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
    };
    opt.series = [
      {
        name: `${feat?.label || weatherFeature} (Actual)`,
        type: 'line',
        data: actual,
        smooth: true,
        lineStyle: { width: 2.5, color: feat?.color },
        itemStyle: { color: feat?.color },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(feat?.color, 0.19) },
              { offset: 1, color: 'transparent' },
            ],
          },
        },
        symbol: 'none',
      },
    ];
    if (showNormal && normal.length) {
      opt.series.push({
        name: 'Normal/Baseline',
        type: 'line',
        data: normal,
        smooth: true,
        lineStyle: { width: 1.5, color: tk.textMuted, type: 'dashed' },
        itemStyle: { color: tk.textMuted },
        symbol: 'none',
      });
    }
    if (showLoad && loadActual.length) {
      opt.yAxis = [
        { ...ecBase(tk).yAxis, name: feat?.unit },
        {
          ...ecBase(tk).yAxis,
          name: 'MW',
          position: 'right',
          axisLine: { show: true, lineStyle: { color: tk.success } },
        },
      ];
      opt.series.push({
        name: 'Actual Load',
        type: 'line',
        yAxisIndex: 1,
        data: loadActual,
        smooth: true,
        lineStyle: { width: 1.5, color: tk.success },
        itemStyle: { color: tk.success },
        symbol: 'none',
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(tk.success, 0.08) },
              { offset: 1, color: 'transparent' },
            ],
          },
        },
      });
    }
    opt.tooltip.axisPointer = { type: 'cross', crossStyle: { color: tk.textMuted } };
    return opt;
  };

  const deltaChartOption = () => {
    const delta = getIntra(weatherFeature, 'delta');
    const feat = WEATHER_FEATURES.find((f) => f.key === weatherFeature);
    return {
      ...ecBase(tk),
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase(tk).yAxis, name: `Δ ${feat?.unit || ''}` },
      series: [
        {
          name: `${feat?.label} Δ from Normal`,
          type: 'bar',
          data: delta.map((v) => ({
            value: v,
            itemStyle: {
              color: v >= 0 ? tk.danger : tk.success,
              borderRadius: v >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3],
            },
          })),
        },
      ],
      tooltip: {
        ...ecBase(tk).tooltip,
        formatter: (p) => {
          const v = p[0]?.value;
          return `<b>${p[0]?.axisValue}</b><br/>Δ: <span style="color:${v >= 0 ? tk.danger : tk.success};font-weight:700">${v >= 0 ? '+' : ''}${v?.toFixed(2)} ${feat?.unit}</span>`;
        },
      },
    };
  };

  const weatherImpactChartOption = () => {
    // Use backend weather_impact if available, otherwise compute from blockMW
    const hasBackend = weatherImpact.some((v) => Number.isFinite(v) && v !== 0);
    const impactData = hasBackend ? weatherImpact : computed.blockMW.map((bm) => bm.total);
    const avgBase =
      loadBaseline.reduce((s, v) => s + (Number(v) || 0), 0) /
        Math.max(1, loadBaseline.filter(Number.isFinite).length) || 4500;
    const pctData = hasBackend
      ? weatherImpactPct
      : impactData.map((v) => (avgBase > 0 ? (v / avgBase) * 100 : 0));
    return {
      ...ecBase(tk),
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
      yAxis: [
        { ...ecBase(tk).yAxis, name: 'MW Impact' },
        { ...ecBase(tk).yAxis, name: '% of Base', position: 'right' },
      ],
      series: [
        {
          name: 'Weather Impact (MW)',
          type: 'bar',
          data: impactData.map((v) => ({
            value: Number(v) || 0,
            itemStyle: {
              color: v >= 0 ? withAlpha(tk.danger, 0.67) : withAlpha(tk.success, 0.67),
              borderRadius: v >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3],
            },
          })),
          barMaxWidth: 6,
        },
        {
          name: 'Impact %',
          type: 'line',
          yAxisIndex: 1,
          data: pctData,
          smooth: true,
          lineStyle: { color: tk.warning, width: 1.5 },
          itemStyle: { color: tk.warning },
          symbol: 'none',
        },
      ],
    };
  };

  const correlationChartOption = () => {
    const feat = WEATHER_FEATURES.find((f) => f.key === weatherFeature);
    const weatherData = getIntra(weatherFeature);
    const rho = spearman(weatherData, loadActual);
    return {
      ...ecBase(tk),
      title: {
        text: `ρ = ${rho.toFixed(3)}`,
        right: 16,
        top: 8,
        textStyle: {
          color: Math.abs(rho) > 0.5 ? tk.success : tk.warning,
          fontSize: 14,
          fontWeight: 700,
        },
      },
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
      yAxis: [
        { ...ecBase(tk).yAxis, name: feat?.unit },
        { ...ecBase(tk).yAxis, name: 'MW', position: 'right' },
      ],
      series: [
        {
          name: feat?.label,
          type: 'line',
          data: weatherData,
          smooth: true,
          lineStyle: { width: 2, color: feat?.color },
          itemStyle: { color: feat?.color },
          symbol: 'none',
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: withAlpha(feat?.color, 0.13) },
                { offset: 1, color: 'transparent' },
              ],
            },
          },
        },
        {
          name: 'Load',
          type: 'line',
          yAxisIndex: 1,
          data: loadActual,
          smooth: true,
          lineStyle: { width: 2, color: tk.text },
          itemStyle: { color: tk.text },
          symbol: 'none',
        },
      ],
    };
  };

  const forecastErrorChartOption = () => {
    const feat = WEATHER_FEATURES.find((f) => f.key === weatherFeature);
    const weatherData = getIntra(weatherFeature);
    const error = loadActual.map((a, i) =>
      Number.isFinite(a) && Number.isFinite(loadForecast[i]) ? a - loadForecast[i] : null
    );
    return {
      ...ecBase(tk),
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
      yAxis: [
        { ...ecBase(tk).yAxis, name: 'Error (MW)' },
        { ...ecBase(tk).yAxis, name: feat?.unit, position: 'right' },
      ],
      series: [
        {
          name: 'Forecast Error',
          type: 'bar',
          data: error.map((v) =>
            v == null
              ? null
              : {
                  value: v,
                  itemStyle: {
                    color: Math.abs(v) > 100 ? tk.danger : withAlpha(tk.accent, 0.53),
                    borderRadius: v >= 0 ? [2, 2, 0, 0] : [0, 0, 2, 2],
                  },
                }
          ),
        },
        {
          name: feat?.label,
          type: 'line',
          yAxisIndex: 1,
          data: weatherData,
          smooth: true,
          lineStyle: { width: 2, color: feat?.color },
          itemStyle: { color: feat?.color },
          symbol: 'none',
        },
      ],
      markLine: {
        data: [
          { yAxis: 100, lineStyle: { color: tk.danger, type: 'dashed' } },
          { yAxis: -100, lineStyle: { color: tk.danger, type: 'dashed' } },
        ],
      },
    };
  };

  const dodChartOption = () => {
    const feat = WEATHER_FEATURES.find((f) => f.key === weatherFeature);
    const dateColors = [
      tk.accent,
      tk.success,
      tk.warning,
      tk.accent2,
      tk.danger,
      tk.info,
      tk.warm,
      tk.textSecondary,
    ];
    const seriesList = [];
    // Current date (primary)
    const currentData = getIntra(weatherFeature);
    if (currentData.length) {
      seriesList.push({
        name: effectiveDate || 'Current',
        type: 'line',
        data: currentData,
        smooth: true,
        lineStyle: { width: 2.5, color: feat?.color || tk.accent },
        itemStyle: { color: feat?.color || tk.accent },
        symbol: 'none',
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(feat?.color || tk.accent, 0.13) },
              { offset: 1, color: 'transparent' },
            ],
          },
        },
      });
    }
    // Comparison dates
    dodDates.forEach((date, i) => {
      const dd = dodDateData[date];
      const arr = dd?.[weatherFeature] || [];
      if (arr.length) {
        const color = dateColors[i % dateColors.length];
        seriesList.push({
          name: date,
          type: 'line',
          data: arr,
          smooth: true,
          lineStyle: { width: 1.5, color, type: 'dashed' },
          itemStyle: { color },
          symbol: 'none',
        });
      }
    });
    return {
      ...ecBase(tk),
      xAxis: {
        ...ecBase(tk).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(tk).xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase(tk).yAxis, name: feat?.unit || '' },
      series: seriesList,
    };
  };

  /* ─── Chart Tab Config ─── */
  const CHART_TABS = [
    { id: 'intraday', label: 'Intraday Profile' },
    { id: 'delta', label: 'Δ from Normal' },
    { id: 'impact', label: 'MW Impact' },
    { id: 'correlation', label: 'Weather vs Load' },
    { id: 'error', label: 'Forecast Error' },
    { id: 'dod', label: 'Multi-Day Compare' },
  ];

  const chartOptions = {
    intraday: intradayChartOption,
    delta: deltaChartOption,
    impact: weatherImpactChartOption,
    correlation: correlationChartOption,
    error: forecastErrorChartOption,
    dod: dodChartOption,
  };

  const attributionPanel = (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
      <div
        style={{
          background: C.surface,
          borderRadius: 10,
          border: `1px solid var(--outline)`,
          overflow: 'hidden',
        }}
      >
        {(() => {
          const base =
            loadBaseline.reduce((s, v) => s + (Number(v) || 0), 0) /
              Math.max(1, loadBaseline.filter(Number.isFinite).length) || 4500;
          const factors = [
            {
              id: 'temp',
              name: 'Temperature',
              val: computed.avgTemp,
              unit: '°C',
              thresh: '>28°C',
              mw: computed.mwTemp,
              blockKey: 'temp',
            },
            {
              id: 'hum',
              name: 'Humidity',
              val: computed.avgHum,
              unit: '%',
              thresh: '>60%',
              mw: computed.mwHum,
              blockKey: 'hum',
            },
            {
              id: 'cloud',
              name: 'Cloud Cover',
              val: computed.avgCloud,
              unit: '%',
              thresh: 'Any',
              mw: computed.mwCloud,
              blockKey: 'cloud',
            },
            {
              id: 'rain',
              name: 'Rain',
              val: rainMetrics?.total_mm != null ? rainMetrics.total_mm : computed.totalPrecip,
              unit: 'mm',
              thresh: '>0',
              mw: computed.mwRain,
              blockKey: 'rain',
            },
          ];
          return (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Factor', 'Avg Value', 'Threshold', 'Daily MW Impact', '% of Base', ''].map(
                    (h) => (
                      <th
                        key={h}
                        style={{
                          fontSize: 9,
                          color: C.muted,
                          textTransform: 'uppercase',
                          letterSpacing: 1,
                          padding: '10px 12px',
                          borderBottom: `1px solid var(--outline)`,
                          textAlign: 'left',
                        }}
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {factors.map((r) => (
                  <React.Fragment key={r.id}>
                    <tr
                      onClick={() => setExpandedFactor(expandedFactor === r.id ? null : r.id)}
                      style={{
                        cursor: 'pointer',
                        background: expandedFactor === r.id ? `color-mix(in srgb, ${C.accent} 3%, transparent)` : 'transparent',
                        transition: 'background 0.15s',
                      }}
                    >
                      <td
                        style={{
                          fontSize: 11,
                          fontWeight: 600,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                        }}
                      >
                        {r.name}
                      </td>
                      <td
                        style={{
                          fontSize: 11,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                        }}
                      >
                        {fmt(r.val, r.val > 0 && r.val < 0.01 ? 4 : r.val < 1 ? 3 : 1)} {r.unit}
                      </td>
                      <td
                        style={{
                          fontSize: 10,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                          color: C.muted,
                        }}
                      >
                        {r.thresh}
                      </td>
                      <td
                        style={{
                          fontSize: 12,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                          fontWeight: 700,
                          color: r.mw >= 0 ? C.red : C.green,
                        }}
                      >
                        {sgn(r.mw)} MW
                      </td>
                      <td
                        style={{
                          fontSize: 11,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                        }}
                      >
                        {((r.mw / base) * 100).toFixed(2)}%
                      </td>
                      <td
                        style={{
                          fontSize: 10,
                          padding: '10px 12px',
                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 13%, transparent)`,
                          color: C.accent,
                        }}
                      >
                        {expandedFactor === r.id ? '▼ hide blocks' : '▶ per block'}
                      </td>
                    </tr>
                    {expandedFactor === r.id && (
                      <tr>
                        <td colSpan={6} style={{ padding: 0 }}>
                          <div
                            style={{
                              maxHeight: 300,
                              overflowY: 'auto',
                              background: `color-mix(in srgb, ${C.border} 7%, transparent)`,
                            }}
                          >
                            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                              <thead>
                                <tr>
                                  {['Block', 'Time', `Raw ${r.unit}`, 'MW Impact'].map((h) => (
                                    <th
                                      key={h}
                                      style={{
                                        fontSize: 8,
                                        color: C.muted,
                                        textTransform: 'uppercase',
                                        letterSpacing: 1,
                                        padding: '5px 10px',
                                        borderBottom: `1px solid color-mix(in srgb, ${C.border} 20%, transparent)`,
                                        textAlign: 'left',
                                        position: 'sticky',
                                        top: 0,
                                        background: C.surface,
                                      }}
                                    >
                                      {h}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {computed.blockMW
                                  .filter((_, i) => i % 4 === 0)
                                  .map((bm) => (
                                    <tr key={bm.block}>
                                      <td
                                        style={{
                                          fontSize: 10,
                                          padding: '3px 10px',
                                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 7%, transparent)`,
                                        }}
                                      >
                                        {bm.block}
                                      </td>
                                      <td
                                        style={{
                                          fontSize: 10,
                                          padding: '3px 10px',
                                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 7%, transparent)`,
                                        }}
                                      >
                                        {bm.time}
                                      </td>
                                      <td
                                        style={{
                                          fontSize: 10,
                                          padding: '3px 10px',
                                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 7%, transparent)`,
                                        }}
                                      >
                                        {fmt(
                                          r.blockKey === 'temp'
                                            ? bm.rawTemp
                                            : r.blockKey === 'hum'
                                              ? bm.rawHum
                                              : r.blockKey === 'cloud'
                                                ? bm.rawCloud
                                                : bm.rawPrecip
                                        )}
                                      </td>
                                      <td
                                        style={{
                                          fontSize: 10,
                                          padding: '3px 10px',
                                          borderBottom: `1px solid color-mix(in srgb, ${C.border} 7%, transparent)`,
                                          fontWeight: 600,
                                          color: bm[r.blockKey] >= 0 ? C.red : C.green,
                                        }}
                                      >
                                        {sgn(bm[r.blockKey])} MW
                                      </td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td
                    colSpan={3}
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      padding: '10px 12px',
                      borderTop: `2px solid ${C.accent}`,
                      color: C.accent,
                    }}
                  >
                    NET
                    {computed.compoundMult !== 1
                      ? ` (×${computed.compoundMult.toFixed(2)} compound)`
                      : ''}
                  </td>
                  <td
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      padding: '10px 12px',
                      borderTop: `2px solid ${C.accent}`,
                      color: computed.netMW >= 0 ? C.red : C.green,
                    }}
                  >
                    {sgn(computed.netMW)} MW
                  </td>
                  <td
                    colSpan={2}
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      padding: '10px 12px',
                      borderTop: `2px solid ${C.accent}`,
                      color: C.accent,
                    }}
                  >
                    {((computed.netMW / base) * 100).toFixed(2)}%
                  </td>
                </tr>
              </tfoot>
            </table>
          );
        })()}
      </div>

      <div
        style={{
          background: C.surface,
          borderRadius: 10,
          border: `1px solid var(--outline)`,
          padding: 20,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <div
          style={{
            fontSize: 9,
            textTransform: 'uppercase',
            letterSpacing: 2,
            color: C.accent,
            fontWeight: 700,
          }}
        >
          Recommended Adjustment
        </div>
        <div style={{ textAlign: 'center', padding: '12px 0' }}>
          <div
            style={{
              fontSize: 36,
              fontWeight: 700,
              color: computed.netMW >= 0 ? C.red : C.green,
              lineHeight: 1.1,
            }}
          >
            {sgn(computed.netMW)} MW
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 6 }}>for next forecast cycle</div>
        </div>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
          {[
            { label: 'P10', mult: 0.7, color: C.green },
            { label: 'P50', mult: 1.0, color: C.accent },
            { label: 'P90', mult: 1.4, color: C.red },
          ].map((b, i) => (
            <div
              key={i}
              style={{
                textAlign: 'center',
                flex: 1,
                padding: '8px 4px',
                background: `color-mix(in srgb, ${b.color} 6%, transparent)`,
                borderRadius: 8,
                border: `1px solid color-mix(in srgb, ${b.color} 13%, transparent)`,
              }}
            >
              <div
                style={{
                  fontSize: 8,
                  color: C.muted,
                  textTransform: 'uppercase',
                  letterSpacing: 1,
                }}
              >
                {b.label}
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, color: b.color }}>
                {sgn(computed.netMW * b.mult)} MW
              </div>
            </div>
          ))}
        </div>

        {computed.compounds.length > 0 && (
          <div style={{ marginTop: 4 }}>
            {computed.compounds.map((c, i) => (
              <div
                key={i}
                style={{
                  fontSize: 10,
                  padding: '4px 10px',
                  marginBottom: 4,
                  background: `color-mix(in srgb, ${C.red} 7%, transparent)`,
                  border: `1px solid color-mix(in srgb, ${C.red} 13%, transparent)`,
                  borderRadius: 6,
                  color: C.red,
                }}
              >
                {c.name} ×{c.mult} — {c.desc}
              </div>
            ))}
          </div>
        )}

        {sensitivity && (
          <div style={{ marginTop: 4, fontSize: 10, color: C.muted }}>
            <div>
              Cooling:{' '}
              <span style={{ color: C.text, fontWeight: 600 }}>
                {fmt(sensitivity.cooling_sensitivity)} MW/°C
              </span>
            </div>
            <div>
              Heating:{' '}
              <span style={{ color: C.text, fontWeight: 600 }}>
                {fmt(sensitivity.heating_sensitivity)} MW/°C
              </span>
            </div>
            <div>
              Humidity amp:{' '}
              <span style={{ color: C.text, fontWeight: 600 }}>
                {fmt(sensitivity.humidity_amplification)}×
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const peakStressPanel = peakWindows ? (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
      {Object.entries(peakWindows).map(([name, data]) => {
        const stress = data.stress_status;
        const stressColor =
          stress === 'Heat Stress' ? C.red : stress === 'Cold Stress' ? C.info : C.green;
        return (
          <div
            key={name}
            style={{
              background: C.surface,
              borderRadius: 10,
              border: `1px solid var(--outline)`,
              padding: 16,
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 10,
              }}
            >
              <span style={{ fontSize: 12, fontWeight: 700 }}>{name}</span>
              <span
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  padding: '2px 10px',
                  borderRadius: 20,
                  background: `color-mix(in srgb, ${stressColor} 9%, transparent)`,
                  color: stressColor,
                }}
              >
                {stress}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
              <div>
                <span style={{ color: C.muted }}>Temp</span>{' '}
                <span style={{ fontWeight: 600 }}>{fmt(data.temp_avg)}°C</span>
              </div>
              <div>
                <span style={{ color: C.muted }}>Δ</span>{' '}
                <span style={{ color: data.temp_delta > 0 ? C.red : C.green, fontWeight: 600 }}>
                  {sgn(data.temp_delta)}°C
                </span>
              </div>
              <div>
                <span style={{ color: C.muted }}>Hum</span>{' '}
                <span style={{ fontWeight: 600 }}>{fmt(data.hum_avg, 0)}%</span>
              </div>
              <div>
                <span style={{ color: C.muted }}>Δ</span>{' '}
                <span style={{ color: data.hum_delta > 0 ? C.red : C.green, fontWeight: 600 }}>
                  {sgn(data.hum_delta)}%
                </span>
              </div>
              <div style={{ gridColumn: '1 / -1' }}>
                <span style={{ color: C.muted }}>Precip</span>{' '}
                <span style={{ fontWeight: 600 }}>{fmt(data.precip_total)} mm</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  ) : (
    <div style={{ color: C.muted, fontSize: 12 }}>No peak stress analysis available.</div>
  );

  const rainPanel = rainMetrics ? (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <MetricCard
        label="Total Rainfall"
        value={fmt(rainMetrics.total_mm)}
        unit="mm"
        color={C.accent2}
      />
      <MetricCard
        label="Max Intensity"
        value={fmt(rainMetrics.max_intensity)}
        unit="mm/15m"
        color={C.info}
      />
      <MetricCard
        label="Rain Duration"
        value={fmt(rainMetrics.rain_hours, 0)}
        unit="hours"
        color={C.info}
      />
      <MetricCard
        label="Load Impact"
        value={fmt(rainMetrics.load_impact_mw, 0)}
        unit="MW"
        color={rainMetrics.load_impact_mw < 0 ? C.green : C.red}
      />
    </div>
  ) : (
    <div style={{ color: C.muted, fontSize: 12 }}>No rain and cloud intelligence available.</div>
  );

  /* ═══════════════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════════════ */
  // Both-day summary metrics (shown in the page header meta strip)
  const _avg = (arr) => {
    if (!Array.isArray(arr) || arr.length === 0) return null;
    const xs = arr.map(Number).filter((v) => Number.isFinite(v));
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  };
  const _fmtN = (v, d = 1) => (v == null ? '—' : Number(v).toFixed(d));
  const _t1Intra = dayAheadData?.weather_analysis?.intraday;
  const _t2Intra = dayAheadT2?.weather_analysis?.intraday;

  return (
    <VpPageShell className="weather-page">
      {setHorizon && (
        <div className="vp-horizon-floater">
          <HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date} />
        </div>
      )}

      {/* ═══ TOP METRICS ═══ */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 14,
        }}
      >
        <MetricCard
          label="Avg Temperature"
          value={fmt(computed.avgTemp)}
          unit="°C"
          color={C.warm}
          delta={dodMaxTemp?.change}
          sub={`Range: ${fmt(computed.minTemp)} – ${fmt(computed.maxTemp)}°C`}
        />
        <MetricCard
          label="Avg Humidity"
          value={`${fmt(computed.avgHum, 0)}`}
          unit="%"
          color={C.info}
          delta={dodHum?.change}
        />
        {(() => {
          const rainVal =
            rainMetrics?.total_mm != null ? rainMetrics.total_mm : computed.totalPrecip;
          const decimals = rainVal > 0 && rainVal < 0.01 ? 4 : rainVal < 1 ? 3 : 2;
          return (
            <MetricCard
              label="Total Precipitation"
              value={fmt(rainVal, decimals)}
              unit="mm"
              color={C.accent2}
              sub={
                rainMetrics
                  ? `${rainMetrics.rain_hours || 0}h duration • max ${fmt(rainMetrics.max_intensity, 4)} mm/15m`
                  : undefined
              }
            />
          );
        })()}
        <MetricCard
          label="CDD"
          value={fmt(computed.cdd)}
          unit="deg-day"
          color={C.warm}
          sub={`≈ ${sgn(computed.cdd * 20)} MW cooling load`}
        />
        <MetricCard
          label="HDD"
          value={fmt(computed.hdd)}
          unit="deg-day"
          color={C.info}
          sub={`≈ ${sgn(computed.hdd * 18)} MW heating load`}
        />
      </div>

      {/* ═══ MAIN CHART AREA ═══ */}
      <div
        style={{
          background: C.card,
          borderRadius: 18,
          border: `1px solid var(--outline)`,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          flex: 1,
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            padding: '16px 16px 12px',
            borderBottom: `1px solid var(--outline)`,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 14,
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <span
                style={{
                  fontSize: 9,
                  textTransform: 'uppercase',
                  letterSpacing: 1.5,
                  color: C.muted,
                }}
              >
                Feature
              </span>
              <select
                value={weatherFeature}
                onChange={(e) => setWeatherFeature(e.target.value)}
                style={{
                  background: C.surface,
                  border: `1px solid var(--outline)`,
                  color: C.text,
                  fontSize: 11,
                  padding: '6px 12px',
                  borderRadius: 999,
                  fontFamily: 'inherit',
                  cursor: 'pointer',
                }}
              >
                {WEATHER_FEATURES.map((f) => (
                  <option key={f.key} value={f.key}>
                    {f.label}
                  </option>
                ))}
              </select>

              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                {[
                  { label: 'Normal Overlay', val: showNormal, set: setShowNormal },
                  { label: 'Load Overlay', val: showLoad, set: setShowLoad },
                ].map((t, i) => (
                  <label
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      fontSize: 10,
                      color: C.muted,
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      onClick={() => t.set(!t.val)}
                      style={{
                        width: 34,
                        height: 18,
                        borderRadius: 9,
                        background: t.val ? C.accent : 'var(--outline)',
                        transition: 'background 0.2s',
                        position: 'relative',
                        cursor: 'pointer',
                      }}
                    >
                      <div
                        style={{
                          width: 14,
                          height: 14,
                          borderRadius: '50%',
                          background: 'var(--bg-elevated)',
                          boxShadow: '0 1px 2px rgba(var(--shadow-rgb), 0.35)',
                          position: 'absolute',
                          top: 2,
                          left: t.val ? 18 : 2,
                          transition: 'left 0.2s',
                        }}
                      />
                    </div>
                    {t.label}
                  </label>
                ))}
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                flexWrap: 'wrap',
                justifyContent: 'flex-end',
              }}
            >
              {computed.regime && (
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '5px 12px',
                    borderRadius: 20,
                    background: `color-mix(in srgb, ${computed.regime.color} 9%, transparent)`,
                    color: computed.regime.color,
                    border: `1px solid color-mix(in srgb, ${computed.regime.color} 20%, transparent)`,
                  }}
                >
                  {computed.regime.name}
                </span>
              )}
              <span style={{ fontSize: 10, color: C.muted }}>
                {effectiveDate} • {selectedRegion || 'ODISHA'}
              </span>
            </div>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 14,
              flexWrap: 'wrap',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                flexWrap: 'wrap',
                minWidth: 0,
                flex: '1 1 520px',
              }}
            >
              <div
                style={{
                  display: 'inline-flex',
                  gap: 3,
                  padding: '5px',
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--outline)',
                  borderRadius: 999,
                  flexWrap: 'wrap',
                }}
              >
                {CHART_TABS.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setChartTab(t.id)}
                    style={{
                      flex: '0 0 auto',
                      padding: '7px 14px',
                      border: 'none',
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: 0.5,
                      borderRadius: 999,
                      background: chartTab === t.id ? 'rgba(var(--accent-rgb), 0.09)' : 'transparent',
                      color: chartTab === t.id ? 'var(--accent)' : C.muted,
                      transition: 'all 0.15s',
                    }}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {chartTab === 'dod' && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    flexWrap: 'wrap',
                    minWidth: 0,
                  }}
                >
                  <input
                    type="date"
                    onChange={(e) => {
                      addDodDate(e.target.value);
                      e.target.value = '';
                    }}
                    style={{
                      background: C.surface,
                      border: `1px solid ${C.border}`,
                      color: C.text,
                      fontSize: 11,
                      padding: '8px 12px',
                      borderRadius: 999,
                      fontFamily: 'inherit',
                      cursor: 'pointer',
                    }}
                  />
                  {dodLoading && <span style={{ fontSize: 10, color: C.accent }}>Loading...</span>}
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '5px 10px',
                      borderRadius: 20,
                      background: `color-mix(in srgb, ${C.accent} 13%, transparent)`,
                      color: C.accent,
                      border: `1px solid color-mix(in srgb, ${C.accent} 20%, transparent)`,
                    }}
                  >
                    {effectiveDate || 'Current'} (primary)
                  </span>
                  {dodDates.map((d) => (
                    <span
                      key={d}
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        padding: '5px 10px',
                        borderRadius: 20,
                        background: `color-mix(in srgb, ${C.accent2} 8%, transparent)`,
                        color: C.accent2,
                        border: `1px solid color-mix(in srgb, ${C.accent2} 20%, transparent)`,
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      {d}
                      {!dodDateData[d] && (
                        <span style={{ color: C.warn, fontSize: 9 }}>(loading)</span>
                      )}
                      <button
                        onClick={() => removeDodDate(d)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: C.red,
                          cursor: 'pointer',
                          fontFamily: 'inherit',
                          fontSize: 12,
                          padding: 0,
                          lineHeight: 1,
                        }}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  {dodDates.length > 0 && (
                    <button
                      onClick={() => {
                        setDodDates([]);
                      }}
                      style={{
                        fontSize: 9,
                        background: 'transparent',
                        border: `1px solid ${C.border}`,
                        color: C.muted,
                        padding: '5px 10px',
                        borderRadius: 999,
                        cursor: 'pointer',
                        fontFamily: 'inherit',
                      }}
                    >
                      Clear All
                    </button>
                  )}
                </div>
              )}
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                flexWrap: 'wrap',
                justifyContent: 'flex-end',
              }}
            >
              <span
                style={{
                  fontSize: 9,
                  textTransform: 'uppercase',
                  letterSpacing: 1.5,
                  color: C.muted,
                }}
              >
                Detail Panels
              </span>
              <button
                onClick={() => setActivePanel('coefs')}
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: '7px 12px',
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: C.surface,
                  color: C.text,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                MW Attribution Coefficients
              </button>
              <button
                onClick={() => setActivePanel('attribution')}
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: '7px 12px',
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: `color-mix(in srgb, ${C.accent} 6%, transparent)`,
                  color: C.text,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                Open MW Attribution
              </button>
              <button
                onClick={() => setActivePanel('peak')}
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: '7px 12px',
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: C.surface,
                  color: C.text,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                Open Peak Stress
              </button>
              <button
                onClick={() => setActivePanel('rain')}
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: '7px 12px',
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: C.surface,
                  color: C.text,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                Open Rain & Cloud
              </button>
            </div>
          </div>
        </div>

        <div
          style={{
            padding: '10px 12px 16px',
            background: 'linear-gradient(180deg, var(--bg-panel), var(--bg-surface))',
            flex: 1,
            minHeight: 0,
            display: 'flex',
          }}
        >
          <div
            style={{
              background: 'rgba(var(--panel-rgb), 0.9)',
              border: `1px solid ${C.border}`,
              borderRadius: 14,
              padding: '10px 10px 18px',
              boxShadow: 'inset 0 1px 0 rgba(var(--overlay-rgb), 0.02)',
              display: 'flex',
              flex: 1,
              minHeight: 0,
            }}
          >
            <div style={{ height: '100%', width: '100%', minHeight: 320 }}>
              <ReactECharts
                option={chartOptions[chartTab]?.()}
                style={{ height: '100%', width: '100%' }}
                notMerge
                lazyUpdate
              />
            </div>
          </div>
        </div>
      </div>

      <DetailOverlay
        open={activePanel === 'coefs'}
        title="MW Attribution Coefficients"
        subtitle="These coefficients drive the Total Weather MW figure"
        onClose={() => setActivePanel(null)}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 12,
          }}
        >
          {Object.entries(MW_COEFS).map(([key, def]) => {
            const mwValue =
              key === 'temperature'
                ? computed.mwTemp
                : key === 'humidity'
                  ? computed.mwHum
                  : key === 'cloud'
                    ? computed.mwCloud
                    : key === 'rain'
                      ? computed.mwRain
                      : computed.mwWind;
            return (
              <div
                key={key}
                style={{
                  background: 'var(--bg)',
                  borderRadius: 10,
                  padding: '12px 16px',
                  border: '1px solid var(--outline)',
                }}
              >
                <div
                  style={{
                    fontSize: 9,
                    color: C.muted,
                    textTransform: 'uppercase',
                    letterSpacing: 1,
                  }}
                >
                  {def.label}
                </div>
                <div
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: mwValue >= 0 ? C.red : C.green,
                    marginTop: 4,
                  }}
                >
                  {mwValue >= 0 ? '+' : ''}
                  {fmt(mwValue, 0)} MW
                </div>
                <div style={{ fontSize: 9, color: C.muted, marginTop: 4 }}>
                  {def.coef > 0 ? '+' : ''}
                  {def.coef} {def.unit}
                  {def.threshold ? ` above ${def.threshold}` : ''}
                </div>
              </div>
            );
          })}
          <div
            style={{
              background: 'var(--bg)',
              borderRadius: 10,
              padding: '12px 16px',
              border: `1px solid color-mix(in srgb, ${C.accent} 27%, transparent)`,
            }}
          >
            <div
              style={{ fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: 1 }}
            >
              Total Weather MW
            </div>
            <div
              style={{
                fontSize: 18,
                fontWeight: 800,
                color: computed.totalMW >= 0 ? C.red : C.green,
                marginTop: 4,
              }}
            >
              {computed.totalMW >= 0 ? '+' : ''}
              {fmt(computed.totalMW, 0)} MW
            </div>
            {computed.compoundMult !== 1.0 && (
              <div style={{ fontSize: 9, color: C.accent, marginTop: 4 }}>
                ×{computed.compoundMult.toFixed(2)} compound effect
              </div>
            )}
          </div>
        </div>
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'attribution'}
        title="MW Attribution & Adjustment"
        subtitle="Moved into a detail panel to preserve the single-screen weather layout"
        onClose={() => setActivePanel(null)}
      >
        {attributionPanel}
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'peak'}
        title="Peak Window Stress Analysis"
        subtitle="Peak-period weather stress and deltas"
        onClose={() => setActivePanel(null)}
      >
        {peakStressPanel}
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'rain'}
        title="Rain & Cloud Intelligence"
        subtitle="Rainfall intensity, duration, and load impact"
        onClose={() => setActivePanel(null)}
      >
        {rainPanel}
      </DetailOverlay>
    </VpPageShell>
  );
}
