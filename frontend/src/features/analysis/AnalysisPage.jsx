import React, { useState, useMemo, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import HorizonToggle from '../../components/HorizonToggle';

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
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: '#ECEEF3',
    minHeight: 0,
    height: '100%',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    gap: 0,
    padding: 0,
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: 12,
    padding: '0 16px',
  },
  kpi: (accent) => ({
    minHeight: 132,
    padding: '16px 18px',
    background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))',
    borderRadius: 14,
    border: '1px solid #2A292F',
    position: 'relative',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    gap: 8,
    boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
    ...(accent ? { boxShadow: `inset 0 0 0 1px ${accent}22, 0 18px 40px rgba(0, 0, 0, 0.18)` } : {}),
  }),
  kpiLabel: { fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 4 },
  kpiVal: (c) => ({ fontSize: 30, fontWeight: 700, color: c || '#ECEEF3', lineHeight: 1.08 }),
  kpiUnit: { fontSize: 11, fontWeight: 400, opacity: 0.5 },
  kpiSub: { fontSize: 10, color: '#6B7186', marginTop: 3, lineHeight: 1.45 },
  kpiSpark: { position: 'absolute', bottom: 8, right: 12, opacity: 0.25 },
  workspace: {
    margin: '10px 16px 16px',
    background: 'linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))',
    borderRadius: 16,
    border: '1px solid #2A292F',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
    flex: 1,
    minHeight: 0,
  },
  workspaceHeader: {
    padding: '18px 20px 14px',
    borderBottom: '1px solid #2A292F',
  },
  tabBar: {
    display: 'inline-flex',
    gap: 4,
    padding: 5,
    background: '#141419',
    border: '1px solid #2A292F',
    borderRadius: 999,
    flexWrap: 'wrap',
  },
  tab: (active) => ({ flex: '0 0 auto', padding: '8px 16px', fontSize: 10, fontWeight: 600, letterSpacing: 0.3, cursor: 'pointer', border: '1px solid transparent', fontFamily: 'inherit', borderRadius: 999, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#A0A5B8', transition: 'all 0.15s' }),
  pill: (active) => ({ fontSize: 9, padding: '6px 11px', borderRadius: 999, cursor: 'pointer', fontFamily: 'inherit', border: `1px solid ${active ? '#F07825' : '#2A292F'}`, background: active ? '#F0782518' : '#1A191E', color: active ? '#F07825' : '#6B7186', textTransform: 'capitalize' }),
  badge: (color) => ({ fontSize: 9, fontWeight: 700, padding: '5px 10px', borderRadius: 999, background: `${color}18`, color, border: `1px solid ${color}33` }),
  actionBtn: { fontSize: 10, fontWeight: 700, padding: '8px 14px', borderRadius: 999, border: '1px solid #2A292F', background: '#1A191E', color: '#ECEEF3', cursor: 'pointer', fontFamily: 'inherit' },
  workspaceGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.5fr) minmax(320px, 0.82fr)',
    gap: 14,
    padding: '14px 16px',
    alignItems: 'stretch',
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
  },
  card: {
    background: 'linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))',
    borderRadius: 14,
    border: '1px solid #2A292F',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  cardHeader: {
    padding: '14px 16px',
    borderBottom: '1px solid #2A292F',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.4, textTransform: 'uppercase', color: '#A0A5B8' },
  weatherGrid: { display: 'grid', gridTemplateColumns: 'minmax(220px, 0.72fr) minmax(0, 1.28fr)', gap: 14, padding: '0 16px 16px' },
  th: { fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, padding: '8px 10px', borderBottom: '1px solid #2A292F', textAlign: 'left', fontWeight: 600 },
  td: { fontSize: 10, padding: '6px 10px', borderBottom: '1px solid #2A292F22' },
  miniLabel: { fontSize: 9, color: '#6B7186', letterSpacing: 1.2, textTransform: 'uppercase' },
};

/* ─── ECharts base ─── */
const ecBase = () => ({
  animation: false,
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

const KpiCard = ({ eyebrow, title, value, unit, tone = '#ECEEF3', detail, footer, sparkline, contents }) => (
  <div
    style={{
      ...S.card,
      padding: '16px 18px',
      gap: 8,
      minHeight: 132,
      justifyContent: 'space-between',
      background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))',
      position: 'relative',
      overflow: 'hidden'
    }}
  >
    <div style={S.miniLabel}>{eyebrow}</div>
    {sparkline && <div style={{ position: 'absolute', bottom: 8, right: 12, opacity: 0.25 }}>{sparkline}</div>}
    {contents ? (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {contents}
      </div>
    ) : (
      <>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#A0A5B8', marginBottom: 8 }}>{title}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 30, lineHeight: 1, fontWeight: 700, color: tone }}>{value}</span>
            {unit ? <span style={{ fontSize: 11, color: '#6B7186' }}>{unit}</span> : null}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {detail ? <div style={{ fontSize: 10, color: '#ECEEF3' }}>{detail}</div> : null}
          {footer ? <div style={{ fontSize: 9, color: '#6B7186' }}>{footer}</div> : null}
        </div>
      </>
    )}
  </div>
);

function DetailOverlay({ open, title, subtitle, onClose, children, contentStyle, bodyStyle }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content fp-modal" style={contentStyle} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            {subtitle ? <div className="modal-subtitle">{subtitle}</div> : null}
          </div>
          <button type="button" className="close-btn" onClick={onClose}>×</button>
        </div>
        <div className="modal-body" style={bodyStyle}>{children}</div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function AnalysisPage({ effectiveDate: propDate, dayAheadData, liveForecastData, horizon = 't1', setHorizon, t2Date, dayAheadT2 }) {
  const activeData = horizon === 't2' ? (dayAheadT2 || dayAheadData) : dayAheadData;
  const dateStr = (horizon === 't2' ? t2Date : propDate) || new Date().toISOString().slice(0, 10);
  const [clock, setClock] = useState(new Date());
  const [chartTab, setChartTab] = useState('load_curve');
  const [weatherOverlay, setWeatherOverlay] = useState('temperature');
  const [activePanel, setActivePanel] = useState(null);
  // Pipeline DB state

  useEffect(() => { const id = setInterval(() => setClock(new Date()), 1000); return () => clearInterval(id); }, []);

  /* ─── Extract data ─── */
  const { loads, yLoads, temps, hums, precips, clouds, timeLabels } = useMemo(() => {
    const s = activeData?.series || liveForecastData?.series;
    const intra = activeData?.weather_analysis?.intraday || {};
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
  }, [activeData, liveForecastData]);

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

  // Region-aware capacity: use prop if available, else fall back to 6500
  const regionCapacity = activeData?.metadata?.region_capacity_mw || liveForecastData?.metadata?.region_capacity_mw || CAPACITY;
  const healthPct = (peakLoad / regionCapacity) * 100;
  const healthBadge = healthPct > 85 ? { l: 'RED', c: '#F87171' } : healthPct > 70 ? { l: 'AMBER', c: '#FBBF24' } : { l: 'GREEN', c: '#34D399' };

  // Advanced statistics
  const sortedLoads = useMemo(() => [...loads].filter(v => v > 0).sort((a, b) => a - b), [loads]);
  const medianLoad = useMemo(() => {
    if (!sortedLoads.length) return 0;
    const mid = Math.floor(sortedLoads.length / 2);
    return sortedLoads.length % 2 ? sortedLoads[mid] : (sortedLoads[mid - 1] + sortedLoads[mid]) / 2;
  }, [sortedLoads]);
  const iqrLoad = useMemo(() => {
    if (sortedLoads.length < 4) return 0;
    const q1 = sortedLoads[Math.floor(sortedLoads.length * 0.25)];
    const q3 = sortedLoads[Math.floor(sortedLoads.length * 0.75)];
    return q3 - q1;
  }, [sortedLoads]);

  // Kurtosis of load distribution
  const kurtosis = useMemo(() => {
    if (sortedLoads.length < 4) return null;
    const mean = sortedLoads.reduce((a, b) => a + b, 0) / sortedLoads.length;
    const std = Math.sqrt(sortedLoads.reduce((a, v) => a + (v - mean) ** 2, 0) / sortedLoads.length);
    if (std < 1e-6) return null;
    return sortedLoads.reduce((a, v) => a + ((v - mean) / std) ** 4, 0) / sortedLoads.length;
  }, [sortedLoads]);

  // Forecast Skill Score: MAPE improvement vs naive persistence (tomorrow = today)
  const forecastSkillScore = useMemo(() => {
    const forecast = loads;
    const actual = yLoads;
    if (!forecast.length || !actual.length) return null;
    const validPairs = forecast.map((f, i) => ({ f, a: actual[i] })).filter(({ f, a }) => a > 10 && Number.isFinite(f) && Number.isFinite(a));
    if (validPairs.length < 10) return null;
    const modelMape = validPairs.reduce((s, { f, a }) => s + Math.abs(f - a) / a, 0) / validPairs.length * 100;
    // Persistence: use yesterday (yLoads) as forecast for today — MAPE of yLoads vs loads
    const persistMape = validPairs.reduce((s, { a }, i) => s + Math.abs((yLoads[i] || a) - loads[i]) / Math.max(loads[i], 1) * 100, 0) / validPairs.length;
    return persistMape > 0 ? Math.round(((persistMape - modelMape) / persistMape) * 100) : null;
  }, [loads, yLoads]);

  // Bias Trend: rolling 7-day mean signed error
  const biasTrend = useMemo(() => {
    const s = activeData?.series || liveForecastData?.series;
    const f = (s?.forecast || []).map(Number).filter(Number.isFinite);
    const a = (s?.actual || []).map(Number).filter(Number.isFinite);
    if (f.length < 4 || a.length < 4) return null;
    const n = Math.min(f.length, a.length);
    const errors = Array.from({ length: n }, (_, i) => f[i] - a[i]);
    return Math.round(errors.reduce((s, v) => s + v, 0) / errors.length);
  }, [activeData, liveForecastData]);

  /* ─── Correlations ─── */
  const rhoTemp = computeSpearmanRho(temps, loads);
  const rhoHum = computeSpearmanRho(hums, loads);
  const rhoCloud = computeSpearmanRho(clouds, loads);
  const rhoPrecip = computeSpearmanRho(precips, loads);
  const rhoRain = computeSpearmanRho(precips, loads);
  const strongestWeatherSignal = useMemo(() => {
    const signals = [
      { label: 'Temperature', value: rhoTemp, color: '#F07825' },
      { label: 'Humidity', value: rhoHum, color: '#45b7d1' },
      { label: 'Cloud', value: rhoCloud, color: '#636e72' },
      { label: 'Precip', value: rhoPrecip, color: '#C084FC' },
    ];
    return signals.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
  }, [rhoTemp, rhoHum, rhoCloud, rhoPrecip]);
  const correlationMatrix = useMemo(() => {
    const rTH = computeSpearmanRho(temps, hums);
    const rTP = computeSpearmanRho(temps, precips);
    const rTC = computeSpearmanRho(temps, clouds);
    const rHP = computeSpearmanRho(hums, precips);
    const rHC = computeSpearmanRho(hums, clouds);
    const rPC = computeSpearmanRho(precips, clouds);
    return [
      { l: 'Load', vals: [1, rhoTemp, rhoHum, rhoPrecip, rhoRain, rhoCloud] },
      { l: 'Temp', vals: [rhoTemp, 1, rTH, rTP, rTP, rTC] },
      { l: 'Humidity', vals: [rhoHum, rTH, 1, rHP, rHP, rHC] },
      { l: 'Precip', vals: [rhoPrecip, rTP, rHP, 1, 1, rPC] },
      { l: 'Rain', vals: [rhoRain, rTP, rHP, 1, 1, rPC] },
      { l: 'Cloud', vals: [rhoCloud, rTC, rHC, rPC, rPC, 1] },
    ];
  }, [temps, hums, precips, clouds, rhoTemp, rhoHum, rhoPrecip, rhoRain, rhoCloud]);
  const correlationHighlights = useMemo(() => {
    const loadLinks = [
      { label: 'Temperature', value: rhoTemp, color: '#F07825' },
      { label: 'Humidity', value: rhoHum, color: '#45b7d1' },
      { label: 'Precipitation', value: rhoPrecip, color: '#C084FC' },
      { label: 'Rain', value: rhoRain, color: '#FBBF24' },
      { label: 'Cloud', value: rhoCloud, color: '#7B8CDE' },
    ];
    const strongestPositive = loadLinks.filter((item) => item.value > 0).sort((a, b) => b.value - a.value)[0] || loadLinks[0];
    const strongestNegative = loadLinks.filter((item) => item.value < 0).sort((a, b) => a.value - b.value)[0] || loadLinks[0];
    const pairCandidates = [];
    correlationMatrix.forEach((row, rowIdx) => {
      row.vals.forEach((value, colIdx) => {
        if (colIdx <= rowIdx || value === 1) return;
        const labels = ['Load', 'Temp', 'Humidity', 'Precip', 'Rain', 'Cloud'];
        pairCandidates.push({ pair: `${labels[rowIdx]} ↔ ${labels[colIdx]}`, value });
      });
    });
    const strongestPair = pairCandidates.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
    return { strongestPositive, strongestNegative, strongestPair };
  }, [rhoTemp, rhoHum, rhoPrecip, rhoRain, rhoCloud, correlationMatrix]);

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
  const TABS = [
    { id: 'load_curve', l: '24h Load Curve' },
    { id: 'weather_overlay', l: 'Weather vs Load' },
    { id: 'distribution', l: 'Distribution' },
    { id: 'ramp', l: 'Ramp Analysis' }
  ];

  /* ═══ RENDER ═══ */
  return (
    <div style={S.page}>
      {/* ─── HORIZON TOGGLE ─── */}
      {setHorizon && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 16px 0', flexShrink: 0 }}>
          <span style={{ fontSize: 10, color: '#6B7186', letterSpacing: 1, textTransform: 'uppercase' }}>
            Viewing: <strong style={{ color: '#ECEEF3' }}>{horizon === 't2' ? `T+2 · ${t2Date}` : `T+1 · ${propDate || dateStr}`}</strong>
          </span>
          <HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date} />
        </div>
      )}
      {!hasData ? (
        <div style={{ padding: 60, textAlign: 'center', color: '#6B7186' }}>
          <div style={{ fontSize: 16, marginBottom: 6 }}>No data available</div>
          <div style={{ fontSize: 11 }}>Run a forecast to populate analysis</div>
        </div>
      ) : (
        <>
          {/* ─── KPI CARDS ─── */}
          <div style={{ ...S.kpiGrid, padding: '10px 16px 0', flexShrink: 0 }}>
            <KpiCard
              eyebrow="Energy Profile"
              title="Day Energy"
              value={fmt(dayEnergy, 0)}
              unit="MWh"
              tone="#F07825"
              detail={`${dayEnergy - yEnergy >= 0 ? '▲' : '▼'} ${fmt(Math.abs(dayEnergy - yEnergy), 0)} MWh vs yesterday`}
              footer="Total 24h consumption"
              sparkline={<Sparkline data={loads.filter((_, i) => i % 4 === 0)} />}
            />
            <KpiCard
              eyebrow="Demand Peak"
              title="Peak Demand"
              value={fmt(peakLoad)}
              unit="MW"
              tone="#F87171"
              detail={`Observed at ${blockToTime(peakIdx)}`}
              footer={`${healthPct.toFixed(1)}% of system capacity`}
            />
            <KpiCard
              eyebrow="Baseload"
              title="Min Demand"
              value={fmt(minLoad)}
              unit="MW"
              tone="#34D399"
              detail={`Observed at ${blockToTime(minIdx)}`}
              footer="Structural baseload reference"
            />
            <KpiCard
              eyebrow="Efficiency"
              contents={<LoadFactorRing value={loadFactor} />}
            />
            <KpiCard
              eyebrow="Variability"
              title="Max Ramp"
              value={fmt(maxRamp)}
              unit="MW/15m"
              tone="#FBBF24"
              detail={`At ${blockToTime(maxRampIdx)}`}
              footer={`${fmt(maxRamp * 4, 0)} MW/hr velocity`}
            />
            <KpiCard
              eyebrow="Resource Balance"
              title="System Stress"
              value={healthPct.toFixed(1)}
              unit="%"
              tone={healthBadge.c}
              detail={`Status: ${healthBadge.l}`}
              footer={`${fmt(regionCapacity - peakLoad, 0)} MW headroom`}
            />
            <KpiCard
              eyebrow="Distribution"
              title="IQR Spread"
              value={fmt(iqrLoad, 0)}
              unit="MW"
              tone="#ECEEF3"
              detail={`Median ${fmt(medianLoad, 0)} MW`}
              footer="Interquartile range"
            />
            <KpiCard
              eyebrow="Model Stability"
              title="Bias Trend"
              value={biasTrend != null ? `${biasTrend > 0 ? '+' : ''}${fmt(biasTrend, 0)}` : '--'}
              unit="MW"
              tone={biasTrend == null ? '#ECEEF3' : biasTrend > 15 ? '#F87171' : biasTrend < -15 ? '#34D399' : '#FBBF24'}
              detail={biasTrend == null ? 'Awaiting settled blocks' : biasTrend > 15 ? 'Over-forecast tendency' : biasTrend < -15 ? 'Under-forecast tendency' : 'Within tolerance'}
              footer="Rolling signed error"
            />
            {kurtosis != null && (
              <KpiCard
                eyebrow="Stats Profile"
                title="Kurtosis"
                value={kurtosis.toFixed(2)}
                tone={kurtosis > 3 ? '#FBBF24' : '#ECEEF3'}
                detail={kurtosis > 3 ? 'Fat-tailed — error spikes likely' : 'Normal distribution'}
                footer="Tailedness of load profile"
              />
            )}
          </div>

          {/* ── Floating control strip ── */}
          <div style={{ margin: '10px 16px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
            <div style={S.tabBar}>
              {TABS.map(t => <button key={t.id} onClick={() => setChartTab(t.id)} style={S.tab(chartTab === t.id)}>{t.l}</button>)}
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end', alignItems: 'center' }}>
              {chartTab === 'weather_overlay' && ['temperature', 'humidity', 'cloud', 'precipitation'].map(k => (
                <button key={k} onClick={() => setWeatherOverlay(k)} style={S.pill(weatherOverlay === k)}>{k}</button>
              ))}
              <span style={S.badge(currentTod.color)}>{currentTod.name}</span>
              <span style={S.badge('#5B9FE4')}>{shiftLabel}</span>
              <span style={S.badge('#F07825')}>{timeLeft} left</span>
              {[
                { label: 'Weather', sub: 'Temp · Humidity · Cloud', panel: 'weather', color: '#5B9FE4' },
                { label: 'Correlations', sub: 'Spearman · Load vs Weather', panel: 'correlations', color: '#34D399' },
                ...(anomalies.length > 0 ? [{ label: 'Anomalies', sub: `${anomalies.length} detected`, panel: 'anomalies', color: '#F87171' }] : []),
              ].map(({ label, sub, panel, color }) => (
                <button
                  key={panel}
                  type="button"
                  onClick={() => setActivePanel(panel)}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 5,
                    minWidth: 172, padding: '12px 16px', borderRadius: 14,
                    border: `1px solid ${activePanel === panel ? `${color}66` : '#2A292F'}`,
                    background: activePanel === panel ? `${color}18` : 'rgba(255,255,255,0.02)',
                    color: activePanel === panel ? color : '#ECEEF3',
                    cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left', transition: 'all 0.15s ease',
                  }}
                >
                  <span style={{ fontSize: 9, letterSpacing: 1.1, textTransform: 'uppercase', color: activePanel === panel ? color : '#6B7186' }}>Quick panel</span>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>{label}</span>
                  <span style={{ fontSize: 10, color: activePanel === panel ? `${color}cc` : '#6B7186' }}>{sub}</span>
                </button>
              ))}
            </div>
          </div>

          <div style={S.workspace}>
            <div style={S.workspaceGrid}>
              {/* Chart card */}
              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div>
                    <div style={S.cardTitle}>Analysis Curve</div>
                  </div>
                  <div style={{ fontSize: 10, color: '#A0A5B8' }}>
                    Solid: Today | Dashed: Yesterday
                  </div>
                </div>
                <div style={{ flex: 1, padding: '10px 10px 14px', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                  <ReactECharts option={chartOpts[chartTab]?.()} style={{ flex: 1, minHeight: 0, height: '100%' }} notMerge lazyUpdate />
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
                <div style={{ ...S.card, flex: 1, minHeight: 0 }}>
                  <div style={S.cardHeader}>
                    <div style={S.cardTitle}>TOD Performance</div>
                    <span style={S.badge(currentTod.color)}>Current: {currentTod.name}</span>
                  </div>
                  <div style={{ overflow: 'auto', flex: 1, display: 'flex', flexDirection: 'column' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', height: '100%' }}>
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
                          const rowTd = { ...S.td, height: `${100 / todGroups.length}%` };
                          return (
                            <tr key={t.name} style={{ background: isCur ? '#F0782508' : 'transparent' }}>
                              <td style={{ ...rowTd, fontWeight: 700, color: t.color, borderLeft: isCur ? `3px solid ${t.color}` : '3px solid transparent', whiteSpace: 'nowrap' }}>{t.name}</td>
                              <td style={rowTd}>{t.blocks}</td>
                              <td style={{ ...rowTd, fontWeight: 600 }}>{fmt(t.avgLoad)}</td>
                              <td style={rowTd}>{fmt(t.peakMw)}</td>
                              <td style={rowTd}>{fmt(t.energy)}</td>
                              <td style={rowTd}><span style={S.badge(t.status === 'On Track' ? '#34D399' : '#FBBF24')}>{t.status}</span></td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ padding: '10px 12px 12px', borderTop: '1px solid #2A292F' }}>
                    <div style={S.miniLabel}>TOD Energy Share</div>
                    <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', marginTop: 6 }}>
                      {todGroups.map(t => {
                        const pct = dayEnergy > 0 ? (t.energy / dayEnergy) * 100 : 0;
                        return <div key={t.name} style={{ width: `${pct}%`, background: t.color, opacity: 0.85 }} title={`${t.name}: ${pct.toFixed(1)}%`} />;
                      })}
                    </div>
                  </div>
                </div>

                {anomalies.length > 0 && (
                  <div style={{ ...S.card, flexShrink: 0 }}>
                    <div style={S.cardHeader}>
                      <div style={S.cardTitle}>Anomalies</div>
                      <span style={S.badge('#F87171')}>{anomalies.length} flagged</span>
                    </div>
                    <div style={{ overflow: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Block', 'Load', 'Reason', ''].map(h => <th key={h} style={{ ...S.th, padding: '6px 10px' }}>{h}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {anomalies.slice(0, 10).map((a, i) => (
                            <tr key={i}>
                              <td style={{ ...S.td, fontWeight: 600, padding: '6px 10px' }}>{a.block}</td>
                              <td style={{ ...S.td, padding: '6px 10px' }}>{fmt(a.load)}</td>
                              <td style={{ ...S.td, color: '#A0A5B8', fontSize: 9, padding: '6px 10px' }}>{a.reason}</td>
                              <td style={{ ...S.td, padding: '6px 10px' }}><span style={S.badge('#F87171')}>{a.severity}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      <DetailOverlay open={activePanel === 'tod'} title="TOD Performance" subtitle="Period-level average load, peak, energy share, and status" onClose={() => setActivePanel(null)}>
        <div style={{ overflow: 'auto' }}>
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
        <div style={{ paddingTop: 14 }}>
          <div style={S.miniLabel}>TOD Energy Share</div>
          <div style={{ display: 'flex', height: 10, borderRadius: 999, overflow: 'hidden', marginTop: 8 }}>
            {todGroups.map(t => {
              const pct = dayEnergy > 0 ? (t.energy / dayEnergy) * 100 : 0;
              return <div key={t.name} style={{ width: `${pct}%`, background: t.color, opacity: 0.9 }} title={`${t.name}: ${pct.toFixed(1)}%`} />;
            })}
          </div>
        </div>
      </DetailOverlay>

      <DetailOverlay open={activePanel === 'anomalies'} title="Anomaly Flags" subtitle="Blocks where load drifted beyond the period distribution" onClose={() => setActivePanel(null)}>
        {anomalies.length ? (
          <div style={{ overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Block', 'Load', 'Reason', 'Severity'].map(h => <th key={h} style={S.th}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {anomalies.map((a, i) => (
                  <tr key={i}>
                    <td style={{ ...S.td, fontWeight: 600 }}>{a.block}</td>
                    <td style={S.td}>{fmt(a.load)}</td>
                    <td style={{ ...S.td, color: '#A0A5B8' }}>{a.reason}</td>
                    <td style={S.td}><span style={S.badge('#F87171')}>{a.severity}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#6B7186', fontSize: 12 }}>No anomalies detected for the current day.</div>}
      </DetailOverlay>

      <DetailOverlay open={activePanel === 'weather'} title="Weather Snapshot" subtitle={`Block ${snapBlock + 1} • ${blockToTime(snapBlock)} • ${weatherCond}`} onClose={() => setActivePanel(null)}>
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 16 }}>
          <div style={{ ...S.card, minHeight: 220 }}>
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, flex: 1 }}>
              <WeatherIcon condition={weatherCond} size={48} />
              <div style={{ fontSize: 14, fontWeight: 700 }}>{weatherCond}</div>
              <div style={{ fontSize: 11, color: '#6B7186' }}>{dateStr}</div>
            </div>
          </div>
          <div style={{ ...S.card, minHeight: 220 }}>
            <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                { l: 'Temperature', v: `${weatherAgg.tempNow.toFixed(1)}°C`, c: '#F07825' },
                { l: 'Feels Like', v: `${feelsLike.toFixed(1)}°C`, c: '#F07825' },
                { l: 'Temp Range', v: `${weatherAgg.tempMin.toFixed(1)}–${weatherAgg.tempMax.toFixed(1)}°C`, c: '#F07825' },
                { l: 'Humidity', v: `${weatherAgg.humNow.toFixed(1)}%`, c: '#45b7d1' },
                { l: 'Precipitation', v: (() => { const p = weatherAgg.precipNow; return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2); })() + ' mm', c: '#C084FC' },
                { l: 'Rain (Total)', v: (() => { const p = weatherAgg.precipTotal; return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2); })() + ' mm', c: '#C084FC' },
                { l: 'Cloud', v: `${weatherAgg.cloudNow.toFixed(0)}%`, c: '#636e72' },
                { l: 'Avg Humidity', v: `${weatherAgg.humAvg.toFixed(1)}%`, c: '#45b7d1' },
              ].map((r, i) => (
                <div key={i} style={{ background: '#201F25', border: '1px solid #2A292F', borderRadius: 12, padding: '12px 14px' }}>
                  <div style={S.miniLabel}>{r.l}</div>
                  <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: r.c }}>{r.v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'correlations'}
        title="Weather Correlations"
        subtitle="Spearman rank relationships between weather variables and load"
        onClose={() => setActivePanel(null)}
        contentStyle={{ width: 'min(96vw, 1680px)', maxWidth: 1680, maxHeight: '92vh', borderRadius: 18 }}
        bodyStyle={{ overflow: 'hidden', padding: 20 }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12 }}>
            <div style={{ background: 'linear-gradient(180deg, rgba(33, 32, 39, 0.96), rgba(24, 23, 28, 0.96))', border: '1px solid #2A292F', borderRadius: 16, padding: '16px 18px', boxShadow: '0 18px 36px rgba(0,0,0,0.18)' }}>
              <div style={S.miniLabel}>Strongest Positive</div>
              <div style={{ marginTop: 10, fontSize: 22, fontWeight: 700, color: correlationHighlights.strongestPositive.color }}>{correlationHighlights.strongestPositive.label}</div>
              <div style={{ marginTop: 6, fontSize: 12, color: '#A0A5B8' }}>ρ +{Math.abs(correlationHighlights.strongestPositive.value).toFixed(2)} with load</div>
            </div>
            <div style={{ background: 'linear-gradient(180deg, rgba(33, 32, 39, 0.96), rgba(24, 23, 28, 0.96))', border: '1px solid #2A292F', borderRadius: 16, padding: '16px 18px', boxShadow: '0 18px 36px rgba(0,0,0,0.18)' }}>
              <div style={S.miniLabel}>Strongest Negative</div>
              <div style={{ marginTop: 10, fontSize: 22, fontWeight: 700, color: '#F87171' }}>{correlationHighlights.strongestNegative.label}</div>
              <div style={{ marginTop: 6, fontSize: 12, color: '#A0A5B8' }}>ρ {correlationHighlights.strongestNegative.value.toFixed(2)} with load</div>
            </div>
            <div style={{ background: 'radial-gradient(circle at top right, rgba(240, 120, 37, 0.14), transparent 50%), linear-gradient(180deg, rgba(33, 32, 39, 0.98), rgba(24, 23, 28, 0.96))', border: '1px solid #2A292F', borderRadius: 16, padding: '16px 18px', boxShadow: '0 18px 36px rgba(0,0,0,0.22)' }}>
              <div style={S.miniLabel}>Dominant Pair</div>
              <div style={{ marginTop: 10, fontSize: 20, fontWeight: 700, color: '#ECEEF3' }}>{correlationHighlights.strongestPair?.pair || '--'}</div>
              <div style={{ marginTop: 6, fontSize: 12, color: '#A0A5B8' }}>Absolute ρ {Math.abs(correlationHighlights.strongestPair?.value || 0).toFixed(2)} across the matrix</div>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 12 }}>
            {[
              { key: 'temp', label: 'Temp → Load', value: rhoTemp, color: '#F07825' },
              { key: 'hum', label: 'Hum → Load', value: rhoHum, color: '#45b7d1' },
              { key: 'cloud', label: 'Cloud → Load', value: rhoCloud, color: '#7B8CDE' },
              { key: 'precip', label: 'Precip → Load', value: rhoPrecip, color: '#C084FC' },
              { key: 'rain', label: 'Rain → Load', value: rhoRain, color: '#FBBF24' },
            ].map((item) => (
              <div key={item.key} style={{ background: 'linear-gradient(180deg, rgba(31, 30, 36, 0.96), rgba(23, 22, 27, 0.96))', border: '1px solid #2A292F', borderRadius: 16, padding: '14px 12px 10px', boxShadow: '0 14px 28px rgba(0,0,0,0.14)' }}>
                <ArcGauge value={item.value} sublabel={item.label} size={92} />
              </div>
            ))}
          </div>
          <div style={{ background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(21, 20, 26, 0.98))', border: '1px solid #2A292F', borderRadius: 18, padding: 16, boxShadow: '0 22px 42px rgba(0,0,0,0.2)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
              <div style={{ ...S.cardTitle, padding: 0 }}>Correlation Heatmap</div>
              <span style={S.badge(Math.abs(correlationHighlights.strongestPair?.value || 0) > 0.7 ? '#34D399' : '#FBBF24')}>
                Matrix View
              </span>
            </div>
            <div style={{ overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 6, fontSize: 10 }}>
                <thead>
                  <tr>
                    {['', 'LOAD', 'TEMP', 'HUMIDITY', 'PRECIP', 'RAIN', 'CLOUD'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', color: '#7F859A', fontWeight: 700, textAlign: 'center', letterSpacing: 1, fontSize: 9 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlationMatrix.map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: '10px 10px', fontWeight: 700, color: '#AEB3C5' }}>{row.l}</td>
                      {row.vals.map((v, ci) => {
                        const abs = Math.abs(v);
                        const bg = v === 1
                          ? 'rgba(240, 120, 37, 0.16)'
                          : v > 0
                            ? `rgba(52, 211, 153, ${Math.min(0.1 + abs * 0.22, 0.3)})`
                            : `rgba(248, 113, 113, ${Math.min(0.08 + abs * 0.22, 0.28)})`;
                        const color = v === 1 ? '#F3B27B' : v > 0 ? '#BDF4DC' : '#FFC0BA';
                        return (
                          <td key={ci} style={{ padding: '14px 10px', textAlign: 'center', color, background: bg, border: '1px solid rgba(255,255,255,0.04)', borderRadius: 12, boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)' }}>
                            <div style={{ fontSize: 17, fontWeight: 700 }}>{v.toFixed(2)}</div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </DetailOverlay>
    </div>
  );
}
