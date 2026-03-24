import React, { useState, useMemo, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';

/* ─── Spearman ─── */
function computeSpearmanRho(xArr, yArr) {
  const n = Math.min(xArr.length, yArr.length);
  if (n < 3) return 0;
  function rank(arr) {
    const s = arr.slice(0, n).map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const r = new Array(n);
    let i = 0;
    while (i < n) { let j = i; while (j < n - 1 && s[j + 1].v === s[j].v) j++; const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[s[k].i] = avg; i = j + 1; }
    return r;
  }
  const rx = rank(xArr), ry = rank(yArr);
  const mx = rx.reduce((s, v) => s + v, 0) / n, my = ry.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { const a = rx[i] - mx, b = ry[i] - my; num += a * b; dx += a * a; dy += b * b; }
  return dx === 0 || dy === 0 ? 0 : num / Math.sqrt(dx * dy);
}

/* ─── TOD Definitions ─── */
const TOD_DEFS = [
  { name: 'Off-Peak Night', start: 0, end: 23, color: '#7B8CDE' },
  { name: 'Morning Ramp', start: 24, end: 35, color: '#F09844' },
  { name: 'Morning Peak', start: 36, end: 47, color: '#F87171' },
  { name: 'Afternoon Shoulder', start: 48, end: 67, color: '#FBBF24' },
  { name: 'Evening Peak', start: 68, end: 83, color: '#C084FC' },
  { name: 'Late Shoulder', start: 84, end: 95, color: '#34D399' },
];

function getTod(idx) { for (const t of TOD_DEFS) if (idx >= t.start && idx <= t.end) return t; return TOD_DEFS[0]; }
function blockToTime(idx) { const h = Math.floor(idx / 4); const m = (idx % 4) * 15; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`; }

const fmt = (v, d = 0) => v == null || !Number.isFinite(v) ? '--' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });
const CAPACITY = 6500;

/* ─── Inline Styles ─── */
const S = {
  page: { fontFamily: "'IBM Plex Mono', monospace", color: '#ECEEF3', minHeight: '100%', display: 'flex', flexDirection: 'column', gap: 2 },
  kpiRow: { display: 'flex', gap: 8, padding: '10px 16px 6px' },
  kpi: (accent) => ({ flex: '1 1 0', padding: '12px 14px', background: '#1A191E', borderRadius: 8, border: '1px solid #2A292F', position: 'relative', overflow: 'hidden', ...(accent ? { borderLeft: `3px solid ${accent}` } : {}) }),
  kpiLabel: { fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 4 },
  kpiVal: (c) => ({ fontSize: 26, fontWeight: 700, color: c || '#ECEEF3', lineHeight: 1.15 }),
  kpiUnit: { fontSize: 11, fontWeight: 400, opacity: 0.5 },
  kpiSub: { fontSize: 9, color: '#6B7186', marginTop: 3 },
  kpiSpark: { position: 'absolute', bottom: 8, right: 12, opacity: 0.25 },
  mainGrid: { display: 'grid', gridTemplateColumns: '1fr 420px', gap: 8, padding: '0 16px', minHeight: 420 },
  card: { background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F', overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase', color: '#A0A5B8', padding: '10px 14px 0' },
  weatherGrid: { display: 'grid', gridTemplateColumns: '180px 1fr', gap: 8, padding: '0 16px 12px' },
  tabBar: { display: 'inline-flex', gap: 3, padding: '4px 6px', background: '#1A191E', border: '1px solid #2A292F', borderRadius: 999 },
  tab: (active) => ({ flex: '0 0 auto', padding: '6px 14px', fontSize: 9, fontWeight: 600, letterSpacing: 0.5, cursor: 'pointer', border: 'none', fontFamily: 'inherit', borderRadius: 999, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#6B7186', transition: 'all 0.15s' }),
  pill: (active) => ({ fontSize: 9, padding: '3px 10px', borderRadius: 14, cursor: 'pointer', fontFamily: 'inherit', border: `1px solid ${active ? '#F07825' : '#2A292F'}`, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#6B7186', textTransform: 'capitalize' }),
  th: { fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, padding: '8px 10px', borderBottom: '1px solid #2A292F', textAlign: 'left', fontWeight: 600 },
  td: { fontSize: 10, padding: '6px 10px', borderBottom: '1px solid #2A292F22' },
  badge: (color) => ({ fontSize: 8, fontWeight: 700, padding: '2px 8px', borderRadius: 12, background: `${color}18`, color }),
  sectionLabel: (color) => ({ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 2, color: color || '#F07825', padding: '8px 16px 2px' }),
};

/* ─── ECharts base ─── */
const ecBase = () => ({
  backgroundColor: 'transparent',
  textStyle: { color: '#A0A5B8', fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 },
  grid: { top: 36, right: 20, bottom: 32, left: 50, containLabel: false },
  tooltip: { trigger: 'axis', backgroundColor: '#1A191E', borderColor: '#2A292F', borderWidth: 1, textStyle: { color: '#ECEEF3', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }, confine: true },
  legend: { textStyle: { color: '#A0A5B8', fontSize: 9 }, top: 4, right: 8, itemWidth: 12, itemHeight: 3 },
  xAxis: { type: 'category', axisLine: { lineStyle: { color: '#2A292F' } }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { show: false } },
  yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { lineStyle: { color: '#2A292F', type: 'dashed', opacity: 0.3 } } },
});

/* ─── Sparkline SVG ─── */
function Sparkline({ data, color = '#F07825', w = 60, h = 22 }) {
  if (!data || data.length < 2) return null;
  const max = Math.max(...data), min = Math.min(...data), range = max - min || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`).join(' ');
  return <svg width={w} height={h}><polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" /></svg>;
}

/* ─── Load Factor Ring ─── */
function LoadFactorRing({ value, size = 72 }) {
  const pct = Math.min(100, Math.max(0, value));
  const r = (size - 8) / 2, circ = 2 * Math.PI * r;
  const color = pct >= 75 ? '#34D399' : pct >= 60 ? '#FBBF24' : '#F87171';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#2A292F" strokeWidth="5" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="5"
          strokeDasharray={circ} strokeDashoffset={circ - (pct / 100) * circ}
          transform={`rotate(-90 ${size / 2} ${size / 2})`} strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
        <text x={size / 2} y={size / 2 + 5} textAnchor="middle" fill="#fff" fontSize="14" fontWeight="700" fontFamily="'IBM Plex Mono',monospace">{pct.toFixed(1)}%</text>
      </svg>
      <div>
        <div style={{ fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1 }}>Load Factor</div>
      </div>
    </div>
  );
}

/* ─── Weather Icon ─── */
function WeatherIcon({ condition, size = 36 }) {
  const s = size;
  if (condition === 'Heavy Rain') return (
    <svg width={s} height={s} viewBox="0 0 36 36"><circle cx="18" cy="14" r="8" fill="#636e72" opacity="0.7"/><circle cx="12" cy="12" r="6" fill="#636e72" opacity="0.8"/><circle cx="24" cy="13" r="5" fill="#636e72" opacity="0.6"/><line x1="12" y1="24" x2="10" y2="30" stroke="#45b7d1" strokeWidth="1.5" strokeLinecap="round"/><line x1="18" y1="24" x2="16" y2="30" stroke="#45b7d1" strokeWidth="1.5" strokeLinecap="round"/><line x1="24" y1="24" x2="22" y2="30" stroke="#45b7d1" strokeWidth="1.5" strokeLinecap="round"/></svg>
  );
  if (condition === 'Rainy') return (
    <svg width={s} height={s} viewBox="0 0 36 36"><circle cx="18" cy="14" r="8" fill="#636e72" opacity="0.6"/><circle cx="12" cy="12" r="6" fill="#636e72" opacity="0.7"/><circle cx="24" cy="13" r="5" fill="#636e72" opacity="0.5"/><line x1="14" y1="24" x2="13" y2="29" stroke="#45b7d1" strokeWidth="1.5" strokeLinecap="round"/><line x1="22" y1="24" x2="21" y2="29" stroke="#45b7d1" strokeWidth="1.5" strokeLinecap="round"/></svg>
  );
  if (condition === 'Cloudy') return (
    <svg width={s} height={s} viewBox="0 0 36 36"><circle cx="18" cy="16" r="9" fill="#636e72" opacity="0.6"/><circle cx="11" cy="14" r="7" fill="#636e72" opacity="0.7"/><circle cx="25" cy="15" r="6" fill="#636e72" opacity="0.5"/></svg>
  );
  if (condition === 'Partly Cloudy') return (
    <svg width={s} height={s} viewBox="0 0 36 36"><circle cx="14" cy="14" r="8" fill="#FBBF24" opacity="0.8"/><circle cx="22" cy="18" r="7" fill="#636e72" opacity="0.6"/><circle cx="16" cy="17" r="5" fill="#636e72" opacity="0.7"/><circle cx="27" cy="19" r="4" fill="#636e72" opacity="0.5"/></svg>
  );
  // Clear / default — sun
  return (
    <svg width={s} height={s} viewBox="0 0 36 36"><circle cx="18" cy="18" r="7" fill="#FBBF24" opacity="0.9"/>{[0,45,90,135,180,225,270,315].map(a => <line key={a} x1={18+10*Math.cos(a*Math.PI/180)} y1={18+10*Math.sin(a*Math.PI/180)} x2={18+13*Math.cos(a*Math.PI/180)} y2={18+13*Math.sin(a*Math.PI/180)} stroke="#FBBF24" strokeWidth="1.5" strokeLinecap="round"/>)}</svg>
  );
}

/* ─── Spearman Arc ─── */
function ArcGauge({ value, sublabel, size = 90 }) {
  const pct = Math.max(0, Math.min(1, (value + 1) / 2));
  const abs = Math.abs(value);
  const color = abs > 0.6 ? '#34D399' : abs > 0.3 ? '#FBBF24' : '#F87171';
  const strength = abs > 0.7 ? 'Strong' : abs > 0.4 ? 'Moderate' : abs > 0.2 ? 'Weak' : 'Very Weak';
  const dir = value > 0.05 ? 'Positive' : value < -0.05 ? 'Negative' : 'None';
  const cx = size / 2, cy = size * 0.6, r = size * 0.34;
  const needleAngle = Math.PI - pct * Math.PI;
  const nx = cx + (r - 3) * Math.cos(needleAngle), ny = cy - (r - 3) * Math.sin(needleAngle);
  const arc = (a1, a2) => { const x1 = cx + r * Math.cos(a1), y1 = cy - r * Math.sin(a1), x2 = cx + r * Math.cos(a2), y2 = cy - r * Math.sin(a2); return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`; };
  return (
    <div style={{ textAlign: 'center', flex: '1 1 80px', minWidth: 75 }}>
      {sublabel && <div style={{ fontSize: 7, color: '#6B7186', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 1 }}>{sublabel}</div>}
      <svg width={size} height={size * 0.62} viewBox={`0 0 ${size} ${size * 0.62}`}>
        <path d={arc(Math.PI, 0)} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" strokeLinecap="round" />
        <path d={arc(Math.PI, needleAngle)} fill="none" stroke={color} strokeWidth="5" strokeLinecap="round" opacity="0.85" />
        <circle cx={nx} cy={ny} r="4" fill={color} /><circle cx={nx} cy={ny} r="1.5" fill="#fff" />
        <text x={cx} y={cy + 2} textAnchor="middle" fill="#fff" fontSize="12" fontWeight="700" fontFamily="'IBM Plex Mono',monospace">{value.toFixed(2)}</text>
      </svg>
      <div style={{ fontSize: 8, color, fontWeight: 600 }}>{strength} {dir}</div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function AnalysisPage({ effectiveDate: propDate, dayAheadData, liveForecastData }) {
  const dateStr = propDate || new Date().toISOString().slice(0, 10);
  const [clock, setClock] = useState(new Date());
  const [chartTab, setChartTab] = useState('load_curve');
  const [weatherOverlay, setWeatherOverlay] = useState('temperature');

  useEffect(() => { const id = setInterval(() => setClock(new Date()), 1000); return () => clearInterval(id); }, []);

  /* ─── Extract data ─── */
  const { loads, yLoads, temps, hums, precips, clouds, timeLabels } = useMemo(() => {
    const s = dayAheadData?.series || liveForecastData?.series;
    const intra = dayAheadData?.weather_analysis?.intraday || {};
    const blocks = s?.blocks || Array.from({ length: 96 }, (_, i) => i + 1);
    const actual = s?.actual || s?.forecast || [];
    const baseline = s?.baseline || s?.hybrid_baseline || [];
    return {
      loads: blocks.map((_, i) => Number.isFinite(actual[i]) ? Math.round(actual[i]) : 0),
      yLoads: blocks.map((_, i) => Number.isFinite(baseline[i]) ? Math.round(baseline[i]) : 0),
      temps: (intra.temperature?.actual || []).map(v => Number(v) || 0),
      hums: (intra.humidity?.actual || []).map(v => Number(v) || 0),
      precips: (intra.precipitation?.actual || []).map(v => Number(v) || 0),
      clouds: (intra.cloud_cover?.actual || []).map(v => Number(v) || 0),
      timeLabels: blocks.map(blockToTime),
    };
  }, [dayAheadData, liveForecastData]);

  /* ─── Current block ─── */
  const nowHour = clock.getHours(), nowMin = clock.getMinutes();
  const currentBlock = Math.min(95, nowHour * 4 + Math.floor(nowMin / 15));
  const currentTod = getTod(currentBlock);
  const shiftLabel = nowHour < 8 ? 'Night (00–08)' : nowHour < 16 ? 'Day (08–16)' : 'Evening (16–00)';
  const blocksLeft = 96 - currentBlock;
  const timeLeft = `${Math.floor(blocksLeft / 4)}h ${(blocksLeft % 4) * 15}m`;

  /* ─── KPIs ─── */
  const dayEnergy = loads.reduce((s, v) => s + v * 0.25, 0);
  const yEnergy = yLoads.reduce((s, v) => s + v * 0.25, 0);
  const peakLoad = Math.max(...loads.filter(Number.isFinite));
  const peakIdx = loads.indexOf(peakLoad);
  const minLoad = Math.min(...loads.filter(v => v > 0));
  const minIdx = loads.indexOf(minLoad);
  const loadFactor = peakLoad > 0 ? (dayEnergy / (peakLoad * 24)) * 100 : 0;
  const ramps = loads.slice(1).map((v, i) => Math.abs(v - loads[i]));
  const maxRamp = ramps.length ? Math.max(...ramps) : 0;
  const maxRampIdx = ramps.indexOf(maxRamp) + 1;
  const healthPct = (peakLoad / CAPACITY) * 100;
  const healthBadge = healthPct > 95 ? { l: 'RED', c: '#F87171' } : healthPct > 90 ? { l: 'AMBER', c: '#FBBF24' } : { l: 'GREEN', c: '#34D399' };

  /* ─── Correlations ─── */
  const rhoTemp = computeSpearmanRho(temps, loads);
  const rhoHum = computeSpearmanRho(hums, loads);
  const rhoCloud = computeSpearmanRho(clouds, loads);
  const rhoPrecip = computeSpearmanRho(precips, loads);
  const rhoRain = computeSpearmanRho(precips, loads);

  /* ─── TOD groups ─── */
  const todGroups = useMemo(() => TOD_DEFS.map(tod => {
    const indices = [];
    for (let i = tod.start; i <= tod.end; i++) indices.push(i);
    const tLoads = indices.map(i => loads[i]).filter(Number.isFinite);
    const avgLoad = tLoads.length ? tLoads.reduce((s, v) => s + v, 0) / tLoads.length : 0;
    const peakMw = tLoads.length ? Math.max(...tLoads) : 0;
    const energy = tLoads.reduce((s, v) => s + v * 0.25, 0);
    const status = tod.name.includes('Peak') ? (avgLoad > 5000 ? 'On Track' : 'Low') : 'On Track';
    return { ...tod, blocks: indices.length, avgLoad, peakMw, energy, status };
  }), [loads]);

  /* ─── Anomalies ─── */
  const anomalies = useMemo(() => {
    const flags = [];
    const todStats = {};
    TOD_DEFS.forEach(tod => {
      const tLoads = [];
      for (let i = tod.start; i <= tod.end; i++) if (loads[i] > 0) tLoads.push(loads[i]);
      const mean = tLoads.reduce((s, v) => s + v, 0) / (tLoads.length || 1);
      const std = Math.sqrt(tLoads.reduce((s, v) => s + (v - mean) ** 2, 0) / (tLoads.length || 1));
      todStats[tod.name] = { mean, std };
    });
    loads.forEach((load, i) => {
      if (load === 0) return;
      const tod = getTod(i);
      const ts = todStats[tod.name];
      if (ts && ts.std > 0 && Math.abs(load - ts.mean) > 2 * ts.std)
        flags.push({ block: blockToTime(i), load, reason: `Load > 2σ from ${tod.name} mean`, severity: 'high' });
    });
    return flags;
  }, [loads]);

  /* ─── Weather snapshot (use latest block with actual data) ─── */
  const lastDataBlock = useMemo(() => {
    for (let i = loads.length - 1; i >= 0; i--) if (loads[i] > 0) return i;
    return currentBlock;
  }, [loads, currentBlock]);
  const snapBlock = Math.min(lastDataBlock, currentBlock);

  // Compute aggregates from actual weather data
  const weatherAgg = useMemo(() => {
    const validLen = snapBlock + 1;
    const slice = (arr) => arr.slice(0, validLen).filter(v => v !== 0 || arr === precips);
    const avg = (arr) => { const s = slice(arr); return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0; };
    const max = (arr) => { const s = slice(arr); return s.length ? Math.max(...s) : 0; };
    const min = (arr) => { const s = slice(arr); return s.length ? Math.min(...s) : 0; };
    const sum = (arr) => slice(arr).reduce((a, b) => a + b, 0);
    return {
      tempNow: temps[snapBlock] || 0,
      tempAvg: avg(temps),
      tempMax: max(temps),
      tempMin: min(temps),
      humNow: hums[snapBlock] || 0,
      humAvg: avg(hums),
      cloudNow: clouds[snapBlock] || 0,
      cloudAvg: avg(clouds),
      precipNow: precips[snapBlock] || 0,
      precipTotal: sum(precips),
      windNow: 0, // not available in current data
    };
  }, [temps, hums, clouds, precips, snapBlock]);

  const feelsLike = weatherAgg.tempNow + (weatherAgg.humNow > 50 ? (weatherAgg.humNow - 50) * 0.05 : 0);
  const weatherCond = weatherAgg.precipNow > 2 ? 'Heavy Rain' : weatherAgg.precipTotal > 0.5 ? 'Rainy' : weatherAgg.cloudNow > 60 ? 'Cloudy' : weatherAgg.cloudNow > 20 ? 'Partly Cloudy' : weatherAgg.tempMax > 38 ? 'Hot & Clear' : 'Clear';

  const hasData = loads.some(v => v > 0);

  /* ═══ CHARTS ═══ */
  const loadCurveOption = () => {
    const todAreas = TOD_DEFS.map(t => [
      { name: t.name, xAxis: blockToTime(t.start), itemStyle: { color: `${t.color}10` }, label: { show: true, color: '#F07825', fontSize: 10, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", position: 'insideTop' } },
      { xAxis: blockToTime(t.end) },
    ]);
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: ['Today', 'Yesterday'], formatter: (n) => n === 'Today' ? 'Solid: Today' : 'Dashed: Yesterday' },
      xAxis: { ...ecBase().xAxis, data: timeLabels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [
        {
          name: 'Today', type: 'line', data: loads, smooth: true, symbol: 'none',
          lineStyle: { width: 2.5, color: '#F07825' }, itemStyle: { color: '#F07825' },
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#F0782520' }, { offset: 1, color: 'transparent' }] } },
          markPoint: {
            data: [
              { type: 'max', symbolSize: 36, label: { formatter: p => `${fmt(p.value)} MW`, fontSize: 9, fontWeight: 700, color: '#F07825' } },
              { type: 'min', symbolSize: 28, label: { formatter: p => `${fmt(p.value)} MW`, fontSize: 9, color: '#34D399' }, itemStyle: { color: '#34D399' } },
            ],
            itemStyle: { color: '#F07825' },
          },
          markArea: { silent: true, data: todAreas },
          markLine: {
            silent: true, symbol: 'none',
            data: [{ xAxis: blockToTime(currentBlock), lineStyle: { color: '#F87171', type: 'dashed', width: 1.5 }, label: { formatter: 'NOW', fontSize: 8, color: '#F87171' } }],
          },
        },
        {
          name: 'Yesterday', type: 'line', data: yLoads, smooth: true, symbol: 'none',
          lineStyle: { width: 1.5, color: '#ECEEF330', type: 'dashed' }, itemStyle: { color: '#ECEEF330' },
        },
      ],
    };
  };

  const weatherOverlayOption = () => {
    const map = { temperature: { d: temps, c: '#F07825', u: '°C' }, humidity: { d: hums, c: '#45b7d1', u: '%' }, cloud: { d: clouds, c: '#636e72', u: '%' }, precipitation: { d: precips, c: '#C084FC', u: 'mm' } };
    const ov = map[weatherOverlay] || map.temperature;
    const rho = computeSpearmanRho(ov.d, loads);
    return {
      ...ecBase(),
      title: { text: `ρ = ${rho.toFixed(2)}`, right: 12, top: 6, textStyle: { color: Math.abs(rho) > 0.5 ? '#34D399' : '#FBBF24', fontSize: 12, fontWeight: 700 } },
      xAxis: { ...ecBase().xAxis, data: timeLabels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: [{ ...ecBase().yAxis, name: ov.u }, { ...ecBase().yAxis, name: 'MW', position: 'right' }],
      series: [
        { name: weatherOverlay, type: 'line', data: ov.d, smooth: true, symbol: 'none', lineStyle: { width: 2, color: ov.c }, itemStyle: { color: ov.c },
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${ov.c}18` }, { offset: 1, color: 'transparent' }] } } },
        { name: 'Load', type: 'line', yAxisIndex: 1, data: loads, smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#F07825' }, itemStyle: { color: '#F07825' } },
      ],
    };
  };

  const loadDistOption = () => {
    const binSize = 200;
    const valid = loads.filter(v => v > 0);
    if (!valid.length) return ecBase();
    const lo = Math.floor(Math.min(...valid) / binSize) * binSize;
    const hi = Math.ceil(Math.max(...valid) / binSize) * binSize;
    const bins = [];
    for (let b = lo; b < hi; b += binSize) bins.push({ r: `${(b / 1000).toFixed(1)}k`, c: valid.filter(v => v >= b && v < b + binSize).length });
    return { ...ecBase(), xAxis: { ...ecBase().xAxis, data: bins.map(b => b.r) }, yAxis: { ...ecBase().yAxis, name: 'Blocks' },
      series: [{ type: 'bar', data: bins.map(b => ({ value: b.c, itemStyle: { color: '#F07825', borderRadius: [3, 3, 0, 0] } })), barMaxWidth: 24 }] };
  };

  const rampOption = () => ({
    ...ecBase(),
    xAxis: { ...ecBase().xAxis, data: timeLabels.slice(1), axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
    yAxis: { ...ecBase().yAxis, name: 'MW/15m' },
    series: [{ type: 'bar', data: ramps.map(v => ({ value: v, itemStyle: { color: v > 100 ? '#F87171' : v > 50 ? '#FBBF24' : '#F0782588', borderRadius: [2, 2, 0, 0] } })), barMaxWidth: 4 }],
  });

  const chartOpts = { load_curve: loadCurveOption, weather_overlay: weatherOverlayOption, distribution: loadDistOption, ramp: rampOption };
  const TABS = [{ id: 'load_curve', l: '24h Load Curve' }, { id: 'weather_overlay', l: 'Weather vs Load' }, { id: 'distribution', l: 'Distribution' }, { id: 'ramp', l: 'Ramp Analysis' }];

  /* ═══ RENDER ═══ */
  return (
    <div style={S.page}>

      {!hasData ? (
        <div style={{ padding: 60, textAlign: 'center', color: '#6B7186' }}>
          <div style={{ fontSize: 16, marginBottom: 6 }}>No data available</div>
          <div style={{ fontSize: 11 }}>Run a forecast to populate analysis</div>
        </div>
      ) : (
        <>
          {/* ─── KPI CARDS ─── */}
          <div style={S.kpiRow}>
            <div style={S.kpi()}>
              <div style={S.kpiLabel}>Day Energy</div>
              <div style={S.kpiVal('#F07825')}>{fmt(dayEnergy, 0)} <span style={S.kpiUnit}>MWh</span></div>
              <div style={{ fontSize: 9, color: dayEnergy - yEnergy >= 0 ? '#F87171' : '#34D399', marginTop: 3, fontWeight: 600 }}>
                {dayEnergy - yEnergy >= 0 ? '▲' : '▼'} {fmt(Math.abs(dayEnergy - yEnergy), 0)} MWh vs yesterday
              </div>
              <div style={S.kpiSpark}><Sparkline data={loads.filter((_, i) => i % 4 === 0)} /></div>
            </div>
            <div style={S.kpi()}>
              <div style={S.kpiLabel}>Peak Demand</div>
              <div style={S.kpiVal('#F87171')}>{fmt(peakLoad)} <span style={S.kpiUnit}>MW</span></div>
              <div style={S.kpiSub}>at {blockToTime(peakIdx)} • {healthPct.toFixed(1)}% of {CAPACITY} MW</div>
            </div>
            <div style={S.kpi()}>
              <div style={S.kpiLabel}>Minimum Demand</div>
              <div style={S.kpiVal('#34D399')}>{fmt(minLoad)} <span style={S.kpiUnit}>MW</span></div>
              <div style={S.kpiSub}>at {blockToTime(minIdx)} • baseload reference</div>
            </div>
            <div style={S.kpi()}>
              <LoadFactorRing value={loadFactor} />
            </div>
            <div style={S.kpi()}>
              <div style={S.kpiLabel}>Ramp (Max)</div>
              <div style={S.kpiVal('#FBBF24')}>{fmt(maxRamp)} <span style={S.kpiUnit}>MW/15m</span></div>
              <div style={S.kpiSub}>at {blockToTime(maxRampIdx)} • {fmt(maxRamp * 4, 0)} MW/hr</div>
            </div>
          </div>

          {/* ─── CHART TABS (floating) ─── */}
          <div style={{ display: 'flex', alignItems: 'center', padding: '0 16px' }}>
            <div style={S.tabBar}>
              {TABS.map(t => <button key={t.id} onClick={() => setChartTab(t.id)} style={S.tab(chartTab === t.id)}>{t.l}</button>)}
            </div>
            <div style={{ fontSize: 9, color: '#6B7186', marginLeft: 'auto' }}>Solid: Today | Dashed: Yesterday</div>
          </div>

          {/* ─── MAIN: Chart + TOD side by side ─── */}
          <div style={S.mainGrid}>
            {/* Chart card */}
            <div style={S.card}>
              {chartTab === 'weather_overlay' && (
                <div style={{ padding: '6px 12px', borderBottom: '1px solid #2A292F', display: 'flex', gap: 5 }}>
                  {['temperature', 'humidity', 'cloud', 'precipitation'].map(k => (
                    <button key={k} onClick={() => setWeatherOverlay(k)} style={S.pill(weatherOverlay === k)}>{k}</button>
                  ))}
                </div>
              )}
              <div style={{ flex: 1, padding: '4px 6px' }}>
                <ReactECharts option={chartOpts[chartTab]?.()} style={{ height: '100%', minHeight: 350 }} notMerge />
              </div>
            </div>

            {/* TOD Performance */}
            <div style={S.card}>
              <div style={{ fontSize: 10, fontWeight: 700, padding: '10px 12px', borderBottom: '1px solid #2A292F', letterSpacing: 1, color: '#A0A5B8' }}>TOD Performance</div>
              <div style={{ flex: 1, overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      {['PERIOD', 'BLKS', 'AVG MW', 'PEAK', 'ENERGY', 'STATUS'].map(h => (
                        <th key={h} style={S.th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {todGroups.map(t => {
                      const isCur = t.name === currentTod.name;
                      return (
                        <tr key={t.name} style={{ background: isCur ? '#F0782508' : 'transparent' }}>
                          <td style={{ ...S.td, fontWeight: 700, color: t.color, borderLeft: isCur ? `3px solid ${t.color}` : '3px solid transparent', whiteSpace: 'nowrap' }}>{t.name}</td>
                          <td style={S.td}>{t.blocks}</td>
                          <td style={{ ...S.td, fontWeight: 600 }}>{fmt(t.avgLoad)}</td>
                          <td style={S.td}>{fmt(t.peakMw)}</td>
                          <td style={S.td}>{fmt(t.energy)}</td>
                          <td style={S.td}><span style={S.badge(t.status === 'On Track' ? '#34D399' : '#FBBF24')}>{t.status}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {/* TOD Energy Share bar */}
              <div style={{ padding: '8px 12px', borderTop: '1px solid #2A292F' }}>
                <div style={{ fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>TOD Energy Share</div>
                <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}>
                  {todGroups.map(t => {
                    const pct = dayEnergy > 0 ? (t.energy / dayEnergy) * 100 : 0;
                    return <div key={t.name} style={{ width: `${pct}%`, background: t.color, opacity: 0.85 }} title={`${t.name}: ${pct.toFixed(1)}%`} />;
                  })}
                </div>
              </div>
              {/* Anomalies inline */}
              {anomalies.length > 0 && (
                <div style={{ borderTop: '1px solid #2A292F' }}>
                  <div style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 9, fontWeight: 700, color: '#F87171', letterSpacing: 1, textTransform: 'uppercase' }}>Anomalies Detected ({anomalies.length})</span>
                  </div>
                  <div style={{ maxHeight: 120, overflow: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Block', 'Load', 'Reason', ''].map(h => <th key={h} style={{ ...S.th, padding: '4px 10px' }}>{h}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {anomalies.slice(0, 10).map((a, i) => (
                          <tr key={i}>
                            <td style={{ ...S.td, fontWeight: 600, padding: '3px 10px' }}>{a.block}</td>
                            <td style={{ ...S.td, padding: '3px 10px' }}>{fmt(a.load)}</td>
                            <td style={{ ...S.td, color: '#A0A5B8', fontSize: 8, padding: '3px 10px' }}>{a.reason}</td>
                            <td style={{ ...S.td, padding: '3px 10px' }}><span style={S.badge('#F87171')}>{a.severity}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ─── WEATHER INTELLIGENCE ─── */}
          <div style={S.sectionLabel()}>Weather Intelligence</div>
          <div style={S.weatherGrid}>
            {/* Weather Snapshot */}
            <div style={S.card}>
              <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <WeatherIcon condition={weatherCond} size={36} />
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700 }}>{weatherCond}</div>
                    <div style={{ fontSize: 8, color: '#6B7186' }}>Block {snapBlock + 1} • {blockToTime(snapBlock)}</div>
                  </div>
                </div>
                {[
                  { l: 'Temperature', v: `${weatherAgg.tempNow.toFixed(1)}°C`, c: '#F07825' },
                  { l: 'Feels Like', v: `${feelsLike.toFixed(1)}°C`, c: '#F07825' },
                  { l: 'Temp Range', v: `${weatherAgg.tempMin.toFixed(1)}–${weatherAgg.tempMax.toFixed(1)}°C`, c: '#F07825' },
                  { l: 'Humidity', v: `${weatherAgg.humNow.toFixed(1)}%`, c: '#45b7d1' },
                  { l: 'Precipitation', v: (() => { const p = weatherAgg.precipNow; return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2); })() + ' mm', c: '#C084FC' },
                  { l: 'Rain (Total)', v: (() => { const p = weatherAgg.precipTotal; return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2); })() + ' mm', c: '#C084FC' },
                  { l: 'Cloud', v: `${weatherAgg.cloudNow.toFixed(0)}%`, c: '#636e72' },
                ].map((r, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                    <span style={{ color: '#6B7186' }}>{r.l}</span>
                    <span style={{ color: r.c, fontWeight: 600 }}>{r.v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Spearman Correlations */}
            <div style={S.card}>
              <div style={{ fontSize: 10, fontWeight: 700, padding: '10px 14px', borderBottom: '1px solid #2A292F', letterSpacing: 1, color: '#A0A5B8' }}>Spearman Rank Correlations (ρ)</div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-around', padding: '8px 6px', flexWrap: 'wrap' }}>
                <ArcGauge value={rhoTemp} sublabel="Temp → Load" />
                <ArcGauge value={rhoHum} sublabel="Hum → Load" />
                <ArcGauge value={rhoCloud} sublabel="Cloud → Load" />
                <ArcGauge value={rhoPrecip} sublabel="Precip → Load" />
                <ArcGauge value={rhoRain} sublabel="Rain → Load" />
              </div>
              {/* Cross-correlation matrix */}
              <div style={{ padding: '0 14px 10px' }}>
                <div style={{ fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Cross-Correlation Matrix</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 9 }}>
                  <thead>
                    <tr>
                      {['', 'LOAD', 'TEMP', 'HUMIDITY', 'PRECIP', 'RAIN', 'CLOUD'].map(h => (
                        <th key={h} style={{ padding: '3px 6px', color: '#6B7186', fontWeight: 600, textAlign: 'center', borderBottom: '1px solid #2A292F22' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const rTH = computeSpearmanRho(temps, hums);
                      const rTP = computeSpearmanRho(temps, precips);
                      const rTC = computeSpearmanRho(temps, clouds);
                      const rHP = computeSpearmanRho(hums, precips);
                      const rHC = computeSpearmanRho(hums, clouds);
                      const rPC = computeSpearmanRho(precips, clouds);
                      return [
                        { l: 'Load',   vals: [1,       rhoTemp, rhoHum,  rhoPrecip, rhoRain, rhoCloud] },
                        { l: 'Temp',   vals: [rhoTemp, 1,       rTH,     rTP,       rTP,     rTC] },
                        { l: 'Humidity', vals: [rhoHum, rTH,    1,       rHP,       rHP,     rHC] },
                        { l: 'Precip', vals: [rhoPrecip, rTP,   rHP,     1,         1,       rPC] },
                        { l: 'Rain',   vals: [rhoRain, rTP,     rHP,     1,         1,       rPC] },
                        { l: 'Cloud',  vals: [rhoCloud, rTC,    rHC,     rPC,       rPC,     1] },
                      ];
                    })().map((row, ri) => (
                      <tr key={ri}>
                        <td style={{ padding: '3px 6px', fontWeight: 600, color: '#A0A5B8' }}>{row.l}</td>
                        {row.vals.map((v, ci) => {
                          const abs = Math.abs(v);
                          const bg = v === 1 ? '#F0782515' : abs > 0.5 ? '#34D39910' : abs > 0.3 ? '#FBBF2408' : 'transparent';
                          return <td key={ci} style={{ padding: '3px 6px', textAlign: 'center', color: '#ECEEF3', background: bg }}>{v.toFixed(2)}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>

        </>
      )}
    </div>
  );
}
