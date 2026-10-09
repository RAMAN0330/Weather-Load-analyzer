import React, { useState, useMemo, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  PageShell as VpPageShell,
  EmptyState as VpEmptyState,
} from '../../components/page/PagePrimitives.jsx';
import { useChartTokens, withAlpha } from '../../lib/chartTheme';

/* ─── Spearman ─── */
function computeSpearmanRho(xArr, yArr) {
  const n = Math.min(xArr.length, yArr.length);
  if (n < 3) return 0;
  function rank(arr) {
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
  }
  const rx = rank(xArr),
    ry = rank(yArr);
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

/* ─── TOD Definitions ─── */
const TOD_DEFS = [
  { name: 'Off-Peak Night', start: 0, end: 23, color: 'var(--info)', tk: 'info' },
  { name: 'Morning Ramp', start: 24, end: 35, color: 'var(--tone-warm)', tk: 'warm' },
  { name: 'Morning Peak', start: 36, end: 47, color: 'var(--danger)', tk: 'danger' },
  { name: 'Afternoon Shoulder', start: 48, end: 67, color: 'var(--warning)', tk: 'warning' },
  { name: 'Evening Peak', start: 68, end: 83, color: 'var(--accent2)', tk: 'accent2' },
  { name: 'Late Shoulder', start: 84, end: 95, color: 'var(--success)', tk: 'success' },
];

function getTod(idx) {
  for (const t of TOD_DEFS) if (idx >= t.start && idx <= t.end) return t;
  return TOD_DEFS[0];
}
function blockToTime(idx) {
  const h = Math.floor(idx / 4);
  const m = (idx % 4) * 15;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

const fmt = (v, d = 0) =>
  v == null || !Number.isFinite(v)
    ? '--'
    : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });
// No hardcoded capacity — each region differs. UI shows '—' when not supplied by backend.

/* ─── Inline Styles ─── */
const S = {
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: 'var(--text)',
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
    background: 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.98))',
    borderRadius: 14,
    border: '1px solid var(--outline)',
    position: 'relative',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    gap: 8,
    boxShadow: '0 18px 40px rgba(var(--shadow-rgb), 0.18)',
    ...(accent
      ? { boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${accent} 13%, transparent), 0 18px 40px rgba(var(--shadow-rgb), 0.18)` }
      : {}),
  }),
  kpiLabel: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    color: 'var(--text-muted)',
    marginBottom: 4,
  },
  kpiVal: (c) => ({ fontSize: 30, fontWeight: 700, color: c || 'var(--text)', lineHeight: 1.08 }),
  kpiUnit: { fontSize: 12, fontWeight: 500, color: 'var(--text-muted)' },
  kpiSub: { fontSize: 11, color: 'var(--text-muted)', marginTop: 3, lineHeight: 1.45 },
  kpiSpark: { position: 'absolute', bottom: 8, right: 12, opacity: 0.25 },
  workspace: {
    margin: '10px 16px 16px',
    background: 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.96))',
    borderRadius: 16,
    border: '1px solid var(--outline)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 18px 40px rgba(var(--shadow-rgb), 0.18)',
    flex: 1,
    minHeight: 0,
  },
  workspaceHeader: {
    padding: '18px 20px 14px',
    borderBottom: '1px solid var(--outline)',
  },
  tabBar: {
    display: 'inline-flex',
    gap: 4,
    padding: 5,
    background: 'var(--bg-surface)',
    border: '1px solid var(--outline)',
    borderRadius: 999,
    flexWrap: 'wrap',
  },
  tab: (active) => ({
    flex: '0 0 auto',
    padding: '8px 16px',
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: 0.2,
    cursor: 'pointer',
    border: '1px solid transparent',
    fontFamily: 'inherit',
    borderRadius: 999,
    background: active ? 'rgba(var(--accent-rgb), 0.09)' : 'transparent',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
    transition: 'all 0.15s',
  }),
  pill: (active) => ({
    fontSize: 12,
    padding: '6px 11px',
    borderRadius: 999,
    cursor: 'pointer',
    fontFamily: 'inherit',
    border: `1px solid ${active ? 'var(--accent)' : 'var(--outline)'}`,
    background: active ? 'rgba(var(--accent-rgb), 0.09)' : 'var(--bg-panel)',
    color: active ? 'var(--accent)' : 'var(--text-muted)',
    textTransform: 'capitalize',
  }),
  badge: (color) => ({
    fontSize: 11,
    fontWeight: 700,
    padding: '5px 10px',
    borderRadius: 999,
    background: `color-mix(in srgb, ${color} 9%, transparent)`,
    color,
    border: `1px solid color-mix(in srgb, ${color} 20%, transparent)`,
  }),
  actionBtn: {
    fontSize: 12,
    fontWeight: 700,
    padding: '8px 14px',
    borderRadius: 999,
    border: '1px solid var(--outline)',
    background: 'var(--bg-panel)',
    color: 'var(--text)',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
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
    background: 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.96))',
    borderRadius: 14,
    border: '1px solid var(--outline)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  cardHeader: {
    padding: '14px 16px',
    borderBottom: '1px solid var(--outline)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    color: 'var(--text-secondary)',
  },
  weatherGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(220px, 0.72fr) minmax(0, 1.28fr)',
    gap: 14,
    padding: '0 16px 16px',
  },
  th: {
    fontSize: 11,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    padding: '8px 10px',
    borderBottom: '1px solid var(--outline)',
    textAlign: 'left',
    fontWeight: 600,
  },
  td: { fontSize: 12, padding: '6px 10px', borderBottom: '1px solid color-mix(in srgb, var(--outline) 60%, transparent)', color: 'var(--text)' },
  miniLabel: { fontSize: 11, color: 'var(--text-muted)', letterSpacing: 0.6, textTransform: 'uppercase' },
};

/* ─── ECharts base ─── */
const ecBase = (ct) => ({
  animation: false,
  backgroundColor: 'transparent',
  textStyle: { color: ct.textSecondary, fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 },
  grid: { top: 36, right: 20, bottom: 32, left: 50, containLabel: false },
  tooltip: {
    trigger: 'axis',
    backgroundColor: ct.elevated,
    borderColor: ct.outline,
    borderWidth: 1,
    textStyle: { color: ct.text, fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" },
    extraCssText: 'box-shadow: 0 8px 24px rgba(var(--shadow-rgb), 0.4); border-radius: 8px; padding: 8px 12px;',
    confine: true,
  },
  legend: {
    textStyle: { color: ct.textSecondary, fontSize: 12 },
    top: 4,
    right: 8,
    itemWidth: 12,
    itemHeight: 3,
  },
  xAxis: {
    type: 'category',
    axisLine: { lineStyle: { color: ct.outline } },
    axisLabel: { color: ct.textMuted, fontSize: 11 },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    axisLine: { show: false },
    axisLabel: { color: ct.textMuted, fontSize: 11 },
    splitLine: { lineStyle: { color: ct.outline, type: 'dashed', opacity: 0.6 } },
  },
});

/* ─── Sparkline SVG ─── */
function Sparkline({ data, color, w = 60, h = 22 }) {
  const ct = useChartTokens();
  if (!data || data.length < 2) return null;
  const max = Math.max(...data),
    min = Math.min(...data),
    range = max - min || 1;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(' ');
  return (
    <svg width={w} height={h}>
      <polyline points={pts} fill="none" stroke={color || ct.accent} strokeWidth="1.5" />
    </svg>
  );
}

/* ─── Load Factor Ring ─── */
function LoadFactorRing({ value, size = 72 }) {
  const pct = Math.min(100, Math.max(0, value));
  const r = (size - 8) / 2,
    circ = 2 * Math.PI * r;
  const ct = useChartTokens();
  const color = pct >= 75 ? ct.success : pct >= 60 ? ct.warning : ct.danger;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={ct.outline} strokeWidth="5" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeDasharray={circ}
          strokeDashoffset={circ - (pct / 100) * circ}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
        <text
          x={size / 2}
          y={size / 2 + 5}
          textAnchor="middle"
          fill={ct.text}
          fontSize="14"
          fontWeight="700"
          fontFamily="'IBM Plex Mono',monospace"
        >
          {pct.toFixed(1)}%
        </text>
      </svg>
      <div>
        <div
          style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}
        >
          Load Factor
        </div>
      </div>
    </div>
  );
}

/* ─── Weather Icon ─── */
function WeatherIcon({ condition, size = 36 }) {
  const ct = useChartTokens();
  const s = size;
  if (condition === 'Heavy Rain')
    return (
      <svg width={s} height={s} viewBox="0 0 36 36">
        <circle cx="18" cy="14" r="8" fill={ct.textMuted} opacity="0.7" />
        <circle cx="12" cy="12" r="6" fill={ct.textMuted} opacity="0.8" />
        <circle cx="24" cy="13" r="5" fill={ct.textMuted} opacity="0.6" />
        <line
          x1="12"
          y1="24"
          x2="10"
          y2="30"
          stroke={ct.info}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="18"
          y1="24"
          x2="16"
          y2="30"
          stroke={ct.info}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="24"
          y1="24"
          x2="22"
          y2="30"
          stroke={ct.info}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    );
  if (condition === 'Rainy')
    return (
      <svg width={s} height={s} viewBox="0 0 36 36">
        <circle cx="18" cy="14" r="8" fill={ct.textMuted} opacity="0.6" />
        <circle cx="12" cy="12" r="6" fill={ct.textMuted} opacity="0.7" />
        <circle cx="24" cy="13" r="5" fill={ct.textMuted} opacity="0.5" />
        <line
          x1="14"
          y1="24"
          x2="13"
          y2="29"
          stroke={ct.info}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="22"
          y1="24"
          x2="21"
          y2="29"
          stroke={ct.info}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    );
  if (condition === 'Cloudy')
    return (
      <svg width={s} height={s} viewBox="0 0 36 36">
        <circle cx="18" cy="16" r="9" fill={ct.textMuted} opacity="0.6" />
        <circle cx="11" cy="14" r="7" fill={ct.textMuted} opacity="0.7" />
        <circle cx="25" cy="15" r="6" fill={ct.textMuted} opacity="0.5" />
      </svg>
    );
  if (condition === 'Partly Cloudy')
    return (
      <svg width={s} height={s} viewBox="0 0 36 36">
        <circle cx="14" cy="14" r="8" fill={ct.warning} opacity="0.8" />
        <circle cx="22" cy="18" r="7" fill={ct.textMuted} opacity="0.6" />
        <circle cx="16" cy="17" r="5" fill={ct.textMuted} opacity="0.7" />
        <circle cx="27" cy="19" r="4" fill={ct.textMuted} opacity="0.5" />
      </svg>
    );
  // Clear / default — sun
  return (
    <svg width={s} height={s} viewBox="0 0 36 36">
      <circle cx="18" cy="18" r="7" fill={ct.warning} opacity="0.9" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
        <line
          key={a}
          x1={18 + 10 * Math.cos((a * Math.PI) / 180)}
          y1={18 + 10 * Math.sin((a * Math.PI) / 180)}
          x2={18 + 13 * Math.cos((a * Math.PI) / 180)}
          y2={18 + 13 * Math.sin((a * Math.PI) / 180)}
          stroke={ct.warning}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

/* ─── Spearman Arc ─── */
function ArcGauge({ value, sublabel, size = 90 }) {
  const pct = Math.max(0, Math.min(1, (value + 1) / 2));
  const abs = Math.abs(value);
  const ct = useChartTokens();
  const color = abs > 0.6 ? ct.success : abs > 0.3 ? ct.warning : ct.danger;
  const strength = abs > 0.7 ? 'Strong' : abs > 0.4 ? 'Moderate' : abs > 0.2 ? 'Weak' : 'Very Weak';
  const dir = value > 0.05 ? 'Positive' : value < -0.05 ? 'Negative' : 'None';
  const cx = size / 2,
    cy = size * 0.6,
    r = size * 0.34;
  const needleAngle = Math.PI - pct * Math.PI;
  const nx = cx + (r - 3) * Math.cos(needleAngle),
    ny = cy - (r - 3) * Math.sin(needleAngle);
  const arc = (a1, a2) => {
    const x1 = cx + r * Math.cos(a1),
      y1 = cy - r * Math.sin(a1),
      x2 = cx + r * Math.cos(a2),
      y2 = cy - r * Math.sin(a2);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  return (
    <div style={{ textAlign: 'center', flex: '1 1 80px', minWidth: 75 }}>
      {sublabel && (
        <div
          style={{
            fontSize: 7,
            color: 'var(--text-muted)',
            letterSpacing: 1,
            textTransform: 'uppercase',
            marginBottom: 1,
          }}
        >
          {sublabel}
        </div>
      )}
      <svg width={size} height={size * 0.62} viewBox={`0 0 ${size} ${size * 0.62}`}>
        <path
          d={arc(Math.PI, 0)}
          fill="none"
          stroke={ct.outline}
          strokeWidth="5"
          strokeLinecap="round"
        />
        <path
          d={arc(Math.PI, needleAngle)}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          opacity="0.85"
        />
        <circle cx={nx} cy={ny} r="4" fill={color} />
        <circle cx={nx} cy={ny} r="1.5" fill={ct.panel} />
        <text
          x={cx}
          y={cy + 2}
          textAnchor="middle"
          fill={ct.text}
          fontSize="12"
          fontWeight="700"
          fontFamily="'IBM Plex Mono',monospace"
        >
          {value.toFixed(2)}
        </text>
      </svg>
      <div style={{ fontSize: 12, color, fontWeight: 600 }}>
        {strength} {dir}
      </div>
    </div>
  );
}

const KpiCard = ({
  eyebrow,
  title,
  value,
  unit,
  tone = 'var(--text)',
  detail,
  footer,
  sparkline,
  contents,
}) => (
  <div
    style={{
      ...S.card,
      padding: '16px 18px',
      gap: 8,
      minHeight: 132,
      justifyContent: 'space-between',
      background: 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.98))',
      position: 'relative',
      overflow: 'hidden',
    }}
  >
    <div style={S.miniLabel}>{eyebrow}</div>
    {sparkline && (
      <div style={{ position: 'absolute', bottom: 8, right: 12, opacity: 0.25 }}>{sparkline}</div>
    )}
    {contents ? (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {contents}
      </div>
    ) : (
      <>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
            {title}
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 30, lineHeight: 1, fontWeight: 700, color: tone }}>
              {value}
            </span>
            {unit ? <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{unit}</span> : null}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {detail ? <div style={{ fontSize: 12, color: 'var(--text)' }}>{detail}</div> : null}
          {footer ? <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{footer}</div> : null}
        </div>
      </>
    )}
  </div>
);

function DetailOverlay({ open, title, subtitle, onClose, children, contentStyle, bodyStyle }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content fp-modal"
        style={contentStyle}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h3>{title}</h3>
            {subtitle ? <div className="modal-subtitle">{subtitle}</div> : null}
          </div>
          <button type="button" className="close-btn" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-body" style={bodyStyle}>
          {children}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function AnalysisPage({
  effectiveDate: propDate,
  dayAheadData,
  liveForecastData,
  liveT2,
  horizon = 't1',
  setHorizon,
  t2Date,
  dayAheadT2,
}) {
  // For T+2: use liveT2 for forecast/load series, dayAheadT2 for weather analysis metadata
  const activeData = horizon === 't2' ? dayAheadT2 || dayAheadData : dayAheadData;
  const activeSeries = horizon === 't2' ? liveT2 || dayAheadT2 : liveForecastData || dayAheadData;
  const dateStr = (horizon === 't2' ? t2Date : propDate) || new Date().toISOString().slice(0, 10);
  const [clock, setClock] = useState(new Date());
  const [chartTab, setChartTab] = useState('load_curve');
  const [weatherOverlay, setWeatherOverlay] = useState('temperature');
  const [activePanel, setActivePanel] = useState(null);
  const ct = useChartTokens();
  // Pipeline DB state

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  /* ─── Extract data ─── */
  const { loads, yLoads, temps, hums, precips, clouds, timeLabels } = useMemo(() => {
    // Load/forecast series: prefer liveT2 for T+2, live for T+1
    const s = activeSeries?.series || activeData?.series;
    // Weather intraday: from dayAheadT2/dayAheadData weather_analysis
    const intra = activeData?.weather_analysis?.intraday || {};
    const blocks = s?.blocks || Array.from({ length: 96 }, (_, i) => i + 1);
    // T+2 has no actuals, use forecast; T+1 prefers actual
    const actual = s?.actual?.length ? s.actual : s?.forecast || [];
    const baseline = s?.baseline || s?.hybrid_baseline || [];
    return {
      loads: blocks.map((_, i) => (Number.isFinite(actual[i]) ? Math.round(actual[i]) : 0)),
      yLoads: blocks.map((_, i) => (Number.isFinite(baseline[i]) ? Math.round(baseline[i]) : 0)),
      temps: (intra.temperature?.actual || []).map((v) => Number(v) || 0),
      hums: (intra.humidity?.actual || []).map((v) => Number(v) || 0),
      precips: (intra.precipitation?.actual || []).map((v) => Number(v) || 0),
      clouds: (intra.cloud_cover?.actual || []).map((v) => Number(v) || 0),
      timeLabels: blocks.map(blockToTime),
    };
  }, [activeData, activeSeries]);

  /* ─── Current block ─── */
  const nowHour = clock.getHours(),
    nowMin = clock.getMinutes();
  const currentBlock = Math.min(95, nowHour * 4 + Math.floor(nowMin / 15));
  const currentTod = getTod(currentBlock);
  const shiftLabel =
    nowHour < 8 ? 'Night (00–08)' : nowHour < 16 ? 'Day (08–16)' : 'Evening (16–00)';
  const blocksLeft = 96 - currentBlock;
  const timeLeft = `${Math.floor(blocksLeft / 4)}h ${(blocksLeft % 4) * 15}m`;

  /* ─── KPIs ─── */
  const dayEnergy = loads.reduce((s, v) => s + v * 0.25, 0);
  const yEnergy = yLoads.reduce((s, v) => s + v * 0.25, 0);
  const _finiteLoads = loads.filter(Number.isFinite);
  const _posLoads = loads.filter((v) => v > 0);
  const peakLoad = _finiteLoads.length ? Math.max(..._finiteLoads) : 0;
  const peakIdx = peakLoad > 0 ? loads.indexOf(peakLoad) : -1;
  const minLoad = _posLoads.length ? Math.min(..._posLoads) : 0;
  const minIdx = minLoad > 0 ? loads.indexOf(minLoad) : -1;
  const loadFactor = peakLoad > 0 ? (dayEnergy / (peakLoad * 24)) * 100 : 0;
  const ramps = loads.slice(1).map((v, i) => Math.abs(v - loads[i]));
  const maxRamp = ramps.length ? Math.max(...ramps) : 0;
  const maxRampIdx = ramps.indexOf(maxRamp) + 1;

  // Region-aware capacity: use prop if available, else fall back to 6500
  const regionCapacity =
    activeData?.metadata?.region_capacity_mw ||
    liveForecastData?.metadata?.region_capacity_mw ||
    null;
  const healthPct = regionCapacity && peakLoad > 0 ? (peakLoad / regionCapacity) * 100 : null;
  const healthBadge =
    healthPct == null
      ? { l: 'N/A', c: 'var(--text-muted)' }
      : healthPct > 85
        ? { l: 'RED', c: 'var(--danger)' }
        : healthPct > 70
          ? { l: 'AMBER', c: 'var(--warning)' }
          : { l: 'GREEN', c: 'var(--success)' };

  // Advanced statistics
  const sortedLoads = useMemo(() => [...loads].filter((v) => v > 0).sort((a, b) => a - b), [loads]);
  const medianLoad = useMemo(() => {
    if (!sortedLoads.length) return 0;
    const mid = Math.floor(sortedLoads.length / 2);
    return sortedLoads.length % 2
      ? sortedLoads[mid]
      : (sortedLoads[mid - 1] + sortedLoads[mid]) / 2;
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
    const std = Math.sqrt(
      sortedLoads.reduce((a, v) => a + (v - mean) ** 2, 0) / sortedLoads.length
    );
    if (std < 1e-6) return null;
    return sortedLoads.reduce((a, v) => a + ((v - mean) / std) ** 4, 0) / sortedLoads.length;
  }, [sortedLoads]);

  // Forecast Skill Score: % improvement of model MAPE over naive persistence
  // actuals  = loads  (today's settled actual, or forecast where actuals missing)
  // model Fx = activeSeries?.series?.forecast (what the pipeline predicted)
  // persistence = yLoads (yesterday's baseline used as naive "same as yesterday" forecast)
  const forecastSkillScore = useMemo(() => {
    const s = activeSeries?.series;
    const modelForecast = (s?.forecast || []).map(Number);
    const actuals = loads; // today's actual load (may fall back to forecast for T+2)
    if (!modelForecast.length || !actuals.length) return null;
    const validPairs = actuals
      .map((a, i) => ({ a, f: modelForecast[i], p: yLoads[i] }))
      .filter(({ a, f, p }) => a > 10 && Number.isFinite(f) && Number.isFinite(p) && Number.isFinite(a));
    if (validPairs.length < 10) return null;
    const modelMape =
      (validPairs.reduce((s, { a, f }) => s + Math.abs(f - a) / a, 0) / validPairs.length) * 100;
    const persistMape =
      (validPairs.reduce((s, { a, p }) => s + Math.abs(p - a) / a, 0) / validPairs.length) * 100;
    return persistMape > 0 ? Math.round(((persistMape - modelMape) / persistMape) * 100) : null;
  }, [loads, yLoads, activeSeries]);

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
  const rhoRain = rhoPrecip; // precipitation and rain share the same series
  const strongestWeatherSignal = useMemo(() => {
    const signals = [
      { label: 'Temperature', value: rhoTemp, color: 'var(--tone-warm)' },
      { label: 'Humidity', value: rhoHum, color: 'var(--info)' },
      { label: 'Cloud', value: rhoCloud, color: 'var(--text-dim)' },
      { label: 'Precip', value: rhoPrecip, color: 'var(--accent2)' },
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
      { l: 'Load', vals: [1, rhoTemp, rhoHum, rhoPrecip, rhoCloud] },
      { l: 'Temp', vals: [rhoTemp, 1, rTH, rTP, rTC] },
      { l: 'Humidity', vals: [rhoHum, rTH, 1, rHP, rHC] },
      { l: 'Precip', vals: [rhoPrecip, rTP, rHP, 1, rPC] },
      { l: 'Cloud', vals: [rhoCloud, rTC, rHC, rPC, 1] },
    ];
  }, [temps, hums, precips, clouds, rhoTemp, rhoHum, rhoPrecip, rhoCloud]);
  const correlationHighlights = useMemo(() => {
    const loadLinks = [
      { label: 'Temperature', value: rhoTemp, color: 'var(--tone-warm)' },
      { label: 'Humidity', value: rhoHum, color: 'var(--info)' },
      { label: 'Precipitation', value: rhoPrecip, color: 'var(--accent2)' },
      { label: 'Cloud', value: rhoCloud, color: 'var(--accent)' },
    ];
    const strongestPositive =
      loadLinks.filter((item) => item.value > 0).sort((a, b) => b.value - a.value)[0] ||
      loadLinks[0];
    const strongestNegative =
      loadLinks.filter((item) => item.value < 0).sort((a, b) => a.value - b.value)[0] ||
      loadLinks[0];
    const pairCandidates = [];
    correlationMatrix.forEach((row, rowIdx) => {
      row.vals.forEach((value, colIdx) => {
        if (colIdx <= rowIdx || value === 1) return;
        const labels = ['Load', 'Temp', 'Humidity', 'Precip', 'Cloud'];
        pairCandidates.push({ pair: `${labels[rowIdx]} ↔ ${labels[colIdx]}`, value });
      });
    });
    const strongestPair = pairCandidates.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
    return { strongestPositive, strongestNegative, strongestPair };
  }, [rhoTemp, rhoHum, rhoPrecip, rhoCloud, correlationMatrix]);

  /* ─── TOD groups ─── */
  const todGroups = useMemo(
    () =>
      TOD_DEFS.map((tod) => {
        const indices = [];
        for (let i = tod.start; i <= tod.end; i++) indices.push(i);
        const tLoads = indices.map((i) => loads[i]).filter(Number.isFinite);
        const avgLoad = tLoads.length ? tLoads.reduce((s, v) => s + v, 0) / tLoads.length : 0;
        const peakMw = tLoads.length ? Math.max(...tLoads) : 0;
        const energy = tLoads.reduce((s, v) => s + v * 0.25, 0);
        const status = tod.name.includes('Peak')
          ? avgLoad > 5000
            ? 'On Track'
            : 'Low'
          : 'On Track';
        return { ...tod, blocks: indices.length, avgLoad, peakMw, energy, status };
      }),
    [loads]
  );

  /* ─── Anomalies ─── */
  const anomalies = useMemo(() => {
    const flags = [];
    const todStats = {};
    TOD_DEFS.forEach((tod) => {
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
        flags.push({
          block: blockToTime(i),
          load,
          reason: `Load > 2σ from ${tod.name} mean`,
          severity: 'high',
        });
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
    // allowZero=true for precipitation (0mm is valid data, not missing)
    const slice = (arr, allowZero = false) =>
      arr.slice(0, validLen).filter((v) => allowZero || v !== 0);
    const avg = (arr, allowZero = false) => {
      const s = slice(arr, allowZero);
      return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
    };
    const max = (arr, allowZero = false) => {
      const s = slice(arr, allowZero);
      return s.length ? Math.max(...s) : 0;
    };
    const min = (arr, allowZero = false) => {
      const s = slice(arr, allowZero);
      return s.length ? Math.min(...s) : 0;
    };
    const sum = (arr, allowZero = false) => slice(arr, allowZero).reduce((a, b) => a + b, 0);
    return {
      tempNow: temps[snapBlock] || 0,
      tempAvg: avg(temps),
      tempMax: max(temps),
      tempMin: min(temps),
      humNow: hums[snapBlock] || 0,
      humAvg: avg(hums),
      cloudNow: clouds[snapBlock] || 0,
      cloudAvg: avg(clouds),
      precipNow: precips[snapBlock] ?? 0,
      precipTotal: sum(precips, true), // 0mm is valid; include it
      windNow: 0, // not available in current data
    };
  }, [temps, hums, clouds, precips, snapBlock]);

  // Humidity adjustment is symmetric: above 50% adds heat, below 50% subtracts
  const feelsLike = weatherAgg.tempNow + (weatherAgg.humNow - 50) * 0.05;
  const weatherCond =
    weatherAgg.precipNow > 2
      ? 'Heavy Rain'
      : weatherAgg.precipTotal > 0.5
        ? 'Rainy'
        : weatherAgg.cloudNow > 60
          ? 'Cloudy'
          : weatherAgg.cloudNow > 20
            ? 'Partly Cloudy'
            : weatherAgg.tempMax > 38
              ? 'Hot & Clear'
              : 'Clear';

  const hasData = loads.some((v) => v > 0);

  /* ═══ CHARTS ═══ */
  const loadCurveOption = () => {
    const todAreas = TOD_DEFS.map((t) => [
      {
        name: t.name,
        xAxis: blockToTime(t.start),
        itemStyle: { color: withAlpha(ct[t.tk], 0.06) },
        label: {
          show: true,
          color: ct.accent,
          fontSize: 10,
          fontWeight: 600,
          fontFamily: "'IBM Plex Mono', monospace",
          position: 'insideTop',
        },
      },
      { xAxis: blockToTime(t.end) },
    ]);
    return {
      ...ecBase(ct),
      legend: {
        ...ecBase(ct).legend,
        data: ['Today', 'Yesterday'],
        formatter: (n) => (n === 'Today' ? 'Solid: Today' : 'Dashed: Yesterday'),
      },
      xAxis: {
        ...ecBase(ct).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(ct).xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase(ct).yAxis, name: 'MW' },
      series: [
        {
          name: 'Today',
          type: 'line',
          data: loads,
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 2.5, color: ct.accent },
          itemStyle: { color: ct.accent },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: withAlpha(ct.accent, 0.13) },
                { offset: 1, color: 'transparent' },
              ],
            },
          },
          markPoint: {
            data: [
              {
                type: 'max',
                symbolSize: 36,
                label: {
                  formatter: (p) => `${fmt(p.value)} MW`,
                  fontSize: 12,
                  fontWeight: 700,
                  color: ct.accent,
                },
              },
              {
                type: 'min',
                symbolSize: 28,
                label: { formatter: (p) => `${fmt(p.value)} MW`, fontSize: 12, color: ct.success },
                itemStyle: { color: ct.success },
              },
            ],
            itemStyle: { color: ct.accent },
          },
          markArea: { silent: true, data: todAreas },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              {
                xAxis: blockToTime(currentBlock),
                lineStyle: { color: ct.danger, type: 'dashed', width: 1.5 },
                label: { formatter: 'NOW', fontSize: 12, color: ct.danger },
              },
            ],
          },
        },
        {
          name: 'Yesterday',
          type: 'line',
          data: yLoads,
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 1.5, color: withAlpha(ct.accent2, 0.75), type: 'dashed' },
          itemStyle: { color: withAlpha(ct.accent2, 0.75) },
        },
      ],
    };
  };

  const weatherOverlayOption = () => {
    const map = {
      temperature: { d: temps, c: ct.warm, u: '°C' },
      humidity: { d: hums, c: ct.success, u: '%' },
      cloud: { d: clouds, c: ct.textMuted, u: '%' },
      precipitation: { d: precips, c: ct.accent2, u: 'mm' },
    };
    const ov = map[weatherOverlay] || map.temperature;
    const rho = computeSpearmanRho(ov.d, loads);
    return {
      ...ecBase(ct),
      title: {
        text: `ρ = ${rho.toFixed(2)}`,
        right: 12,
        top: 6,
        textStyle: {
          color: Math.abs(rho) > 0.5 ? ct.success : ct.warning,
          fontSize: 12,
          fontWeight: 700,
        },
      },
      xAxis: {
        ...ecBase(ct).xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase(ct).xAxis.axisLabel, interval: 11 },
      },
      yAxis: [
        { ...ecBase(ct).yAxis, name: ov.u },
        { ...ecBase(ct).yAxis, name: 'MW', position: 'right' },
      ],
      series: [
        {
          name: weatherOverlay,
          type: 'line',
          data: ov.d,
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 2, color: ov.c },
          itemStyle: { color: ov.c },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: withAlpha(ov.c, 0.09) },
                { offset: 1, color: 'transparent' },
              ],
            },
          },
        },
        {
          name: 'Load',
          type: 'line',
          yAxisIndex: 1,
          data: loads,
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 2, color: ct.accent },
          itemStyle: { color: ct.accent },
        },
      ],
    };
  };

  const loadDistOption = () => {
    const binSize = 200;
    const valid = loads.filter((v) => v > 0);
    if (!valid.length) return ecBase(ct);
    const lo = Math.floor(Math.min(...valid) / binSize) * binSize;
    const hi = Math.ceil(Math.max(...valid) / binSize) * binSize;
    const bins = [];
    for (let b = lo; b < hi; b += binSize)
      bins.push({
        r: `${(b / 1000).toFixed(1)}k`,
        c: valid.filter((v) => v >= b && v < b + binSize).length,
      });
    return {
      ...ecBase(ct),
      xAxis: { ...ecBase(ct).xAxis, data: bins.map((b) => b.r) },
      yAxis: { ...ecBase(ct).yAxis, name: 'Blocks' },
      series: [
        {
          type: 'bar',
          data: bins.map((b) => ({
            value: b.c,
            itemStyle: { color: ct.accent, borderRadius: [3, 3, 0, 0] },
          })),
          barMaxWidth: 24,
        },
      ],
    };
  };

  const rampOption = () => ({
    ...ecBase(ct),
    xAxis: {
      ...ecBase(ct).xAxis,
      data: timeLabels.slice(1),
      axisLabel: { ...ecBase(ct).xAxis.axisLabel, interval: 11 },
    },
    yAxis: { ...ecBase(ct).yAxis, name: 'MW/15m' },
    series: [
      {
        type: 'bar',
        data: ramps.map((v) => ({
          value: v,
          itemStyle: {
            color: v > 100 ? ct.danger : v > 50 ? ct.warning : withAlpha(ct.accent, 0.53),
            borderRadius: [2, 2, 0, 0],
          },
        })),
        barMaxWidth: 4,
      },
    ],
  });

  const chartOpts = {
    load_curve: loadCurveOption,
    weather_overlay: weatherOverlayOption,
    distribution: loadDistOption,
    ramp: rampOption,
  };
  const TABS = [
    { id: 'load_curve', l: '24h Load Curve' },
    { id: 'weather_overlay', l: 'Weather vs Load' },
    { id: 'distribution', l: 'Distribution' },
    { id: 'ramp', l: 'Ramp Analysis' },
  ];

  /* ═══ RENDER ═══ */
  return (
    <VpPageShell className="analysis-page">
      {!hasData ? (
        <VpEmptyState title="No data available" description="Run a forecast to populate analysis." />
      ) : (
        <>
          {/* ─── KPI CARDS ─── */}
          <div style={{ ...S.kpiGrid, padding: '10px 16px 0', flexShrink: 0 }}>
            <KpiCard
              eyebrow="Energy Profile"
              title="Day Energy"
              value={fmt(dayEnergy, 0)}
              unit="MWh"
              tone="var(--accent)"
              detail={`${dayEnergy - yEnergy >= 0 ? '▲' : '▼'} ${fmt(Math.abs(dayEnergy - yEnergy), 0)} MWh vs yesterday`}
              footer="Total 24h consumption"
              sparkline={<Sparkline data={loads.filter((_, i) => i % 4 === 0)} />}
            />
            <KpiCard
              eyebrow="Demand Peak"
              title="Peak Demand"
              value={fmt(peakLoad)}
              unit="MW"
              tone="var(--danger)"
              detail={`Observed at ${blockToTime(peakIdx)}`}
              footer={healthPct != null ? `${healthPct.toFixed(1)}% of system capacity` : 'System capacity not configured'}
            />
            <KpiCard
              eyebrow="Baseload"
              title="Min Demand"
              value={fmt(minLoad)}
              unit="MW"
              tone="var(--success)"
              detail={`Observed at ${blockToTime(minIdx)}`}
              footer="Structural baseload reference"
            />
            <KpiCard eyebrow="Efficiency" contents={<LoadFactorRing value={loadFactor} />} />
            <KpiCard
              eyebrow="Distribution"
              title="IQR Spread"
              value={fmt(iqrLoad, 0)}
              unit="MW"
              tone="var(--text)"
              detail={`Median ${fmt(medianLoad, 0)} MW`}
              footer="Interquartile range"
            />
            {kurtosis != null && (
              <KpiCard
                eyebrow="Stats Profile"
                title="Kurtosis"
                value={kurtosis.toFixed(2)}
                tone={kurtosis > 3 ? 'var(--warning)' : 'var(--text)'}
                detail={kurtosis > 3 ? 'Fat-tailed — error spikes likely' : 'Normal distribution'}
                footer="Tailedness of load profile"
              />
            )}
          </div>

          {/* ── Floating control strip ── */}
          <div
            style={{
              margin: '10px 16px 0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 10,
              flexWrap: 'wrap',
            }}
          >
            <div style={S.tabBar}>
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setChartTab(t.id)}
                  style={S.tab(chartTab === t.id)}
                >
                  {t.l}
                </button>
              ))}
            </div>
            <div
              style={{
                display: 'flex',
                gap: 6,
                flexWrap: 'wrap',
                justifyContent: 'flex-end',
                alignItems: 'center',
              }}
            >
              {chartTab === 'weather_overlay' &&
                ['temperature', 'humidity', 'cloud', 'precipitation'].map((k) => (
                  <button
                    key={k}
                    onClick={() => setWeatherOverlay(k)}
                    style={S.pill(weatherOverlay === k)}
                  >
                    {k}
                  </button>
                ))}
              <span style={S.badge(currentTod.color)}>{currentTod.name}</span>
              <span style={S.badge('var(--accent)')}>{shiftLabel}</span>
              <span style={S.badge('var(--tone-warm)')}>{timeLeft} left</span>
              {[
                {
                  label: 'Weather',
                  sub: 'Temp · Humidity · Cloud',
                  panel: 'weather',
                  color: 'var(--accent)',
                },
                {
                  label: 'Correlations',
                  sub: 'Spearman · Load vs Weather',
                  panel: 'correlations',
                  color: 'var(--success)',
                },
                ...(anomalies.length > 0
                  ? [
                      {
                        label: 'Anomalies',
                        sub: `${anomalies.length} detected`,
                        panel: 'anomalies',
                        color: 'var(--danger)',
                      },
                    ]
                  : []),
              ].map(({ label, sub, panel, color }) => (
                <button
                  key={panel}
                  type="button"
                  onClick={() => setActivePanel(panel)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    gap: 5,
                    minWidth: 172,
                    padding: '12px 16px',
                    borderRadius: 14,
                    border: `1px solid ${activePanel === panel ? `color-mix(in srgb, ${color} 40%, transparent)` : 'var(--outline)'}`,
                    background: activePanel === panel ? `color-mix(in srgb, ${color} 9%, transparent)` : 'rgba(var(--overlay-rgb), 0.02)',
                    color: activePanel === panel ? color : 'var(--text)',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    textAlign: 'left',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span
                    style={{
                      fontSize: 11,
                      letterSpacing: 0.5,
                      textTransform: 'uppercase',
                      color: activePanel === panel ? color : 'var(--text-muted)',
                    }}
                  >
                    Quick panel
                  </span>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>{label}</span>
                  <span
                    style={{
                      fontSize: 12,
                      color: activePanel === panel ? `color-mix(in srgb, ${color} 80%, transparent)` : 'var(--text-muted)',
                    }}
                  >
                    {sub}
                  </span>
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
                  <div style={{ fontSize: 10, color: 'var(--text-secondary)' }}>
                    Solid: Today | Dashed: Yesterday
                  </div>
                </div>
                <div
                  style={{
                    flex: 1,
                    padding: '10px 10px 14px',
                    minHeight: 0,
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <ReactECharts
                    option={chartOpts[chartTab]?.()}
                    style={{ flex: 1, minHeight: 0, height: '100%' }}
                    notMerge
                    lazyUpdate
                  />
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minHeight: 0 }}>
                <div style={{ ...S.card, flex: 1, minHeight: 0 }}>
                  <div style={S.cardHeader}>
                    <div style={S.cardTitle}>TOD Performance</div>
                    <span style={S.badge(currentTod.color)}>Current: {currentTod.name}</span>
                  </div>
                  <div
                    style={{ overflow: 'auto', flex: 1, display: 'flex', flexDirection: 'column' }}
                  >
                    <table style={{ width: '100%', borderCollapse: 'collapse', height: '100%' }}>
                      <thead>
                        <tr>
                          {['PERIOD', 'BLKS', 'AVG MW', 'PEAK', 'ENERGY', 'STATUS'].map((h) => (
                            <th key={h} style={S.th}>
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {todGroups.map((t) => {
                          const isCur = t.name === currentTod.name;
                          const rowTd = { ...S.td, height: `${100 / todGroups.length}%` };
                          return (
                            <tr
                              key={t.name}
                              style={{ background: isCur ? 'rgba(var(--accent-rgb), 0.03)' : 'transparent' }}
                            >
                              <td
                                style={{
                                  ...rowTd,
                                  fontWeight: 700,
                                  color: t.color,
                                  borderLeft: isCur
                                    ? `3px solid ${t.color}`
                                    : '3px solid transparent',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                {t.name}
                              </td>
                              <td style={rowTd}>{t.blocks}</td>
                              <td style={{ ...rowTd, fontWeight: 600 }}>{fmt(t.avgLoad)}</td>
                              <td style={rowTd}>{fmt(t.peakMw)}</td>
                              <td style={rowTd}>{fmt(t.energy)}</td>
                              <td style={rowTd}>
                                <span
                                  style={S.badge(t.status === 'On Track' ? 'var(--success)' : 'var(--warning)')}
                                >
                                  {t.status}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ padding: '10px 12px 12px', borderTop: '1px solid var(--outline)' }}>
                    <div style={S.miniLabel}>TOD Energy Share</div>
                    <div
                      style={{
                        display: 'flex',
                        height: 8,
                        borderRadius: 4,
                        overflow: 'hidden',
                        marginTop: 6,
                      }}
                    >
                      {todGroups.map((t) => {
                        const pct = dayEnergy > 0 ? (t.energy / dayEnergy) * 100 : 0;
                        return (
                          <div
                            key={t.name}
                            style={{ width: `${pct}%`, background: t.color, opacity: 0.85 }}
                            title={`${t.name}: ${pct.toFixed(1)}%`}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>

                {anomalies.length > 0 && (
                  <div style={{ ...S.card, flexShrink: 0 }}>
                    <div style={S.cardHeader}>
                      <div style={S.cardTitle}>Anomalies</div>
                      <span style={S.badge('var(--danger)')}>{anomalies.length} flagged</span>
                    </div>
                    <div style={{ overflow: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Block', 'Load', 'Reason', ''].map((h) => (
                              <th key={h} style={{ ...S.th, padding: '6px 10px' }}>
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {anomalies.slice(0, 10).map((a, i) => (
                            <tr key={i}>
                              <td style={{ ...S.td, fontWeight: 600, padding: '6px 10px' }}>
                                {a.block}
                              </td>
                              <td style={{ ...S.td, padding: '6px 10px' }}>{fmt(a.load)}</td>
                              <td style={{ ...S.td, color: 'var(--text-secondary)', padding: '6px 10px' }}>
                                {a.reason}
                              </td>
                              <td style={{ ...S.td, padding: '6px 10px' }}>
                                <span style={S.badge('var(--danger)')}>{a.severity}</span>
                              </td>
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

      <DetailOverlay
        open={activePanel === 'tod'}
        title="TOD Performance"
        subtitle="Period-level average load, peak, energy share, and status"
        onClose={() => setActivePanel(null)}
      >
        <div style={{ overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['PERIOD', 'BLKS', 'AVG MW', 'PEAK', 'ENERGY', 'STATUS'].map((h) => (
                  <th key={h} style={S.th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {todGroups.map((t) => {
                const isCur = t.name === currentTod.name;
                return (
                  <tr key={t.name} style={{ background: isCur ? 'rgba(var(--accent-rgb), 0.03)' : 'transparent' }}>
                    <td
                      style={{
                        ...S.td,
                        fontWeight: 700,
                        color: t.color,
                        borderLeft: isCur ? `3px solid ${t.color}` : '3px solid transparent',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {t.name}
                    </td>
                    <td style={S.td}>{t.blocks}</td>
                    <td style={{ ...S.td, fontWeight: 600 }}>{fmt(t.avgLoad)}</td>
                    <td style={S.td}>{fmt(t.peakMw)}</td>
                    <td style={S.td}>{fmt(t.energy)}</td>
                    <td style={S.td}>
                      <span style={S.badge(t.status === 'On Track' ? 'var(--success)' : 'var(--warning)')}>
                        {t.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ paddingTop: 14 }}>
          <div style={S.miniLabel}>TOD Energy Share</div>
          <div
            style={{
              display: 'flex',
              height: 10,
              borderRadius: 999,
              overflow: 'hidden',
              marginTop: 8,
            }}
          >
            {todGroups.map((t) => {
              const pct = dayEnergy > 0 ? (t.energy / dayEnergy) * 100 : 0;
              return (
                <div
                  key={t.name}
                  style={{ width: `${pct}%`, background: t.color, opacity: 0.9 }}
                  title={`${t.name}: ${pct.toFixed(1)}%`}
                />
              );
            })}
          </div>
        </div>
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'anomalies'}
        title="Anomaly Flags"
        subtitle="Blocks where load drifted beyond the period distribution"
        onClose={() => setActivePanel(null)}
      >
        {anomalies.length ? (
          <div style={{ overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Block', 'Load', 'Reason', 'Severity'].map((h) => (
                    <th key={h} style={S.th}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {anomalies.map((a, i) => (
                  <tr key={i}>
                    <td style={{ ...S.td, fontWeight: 600 }}>{a.block}</td>
                    <td style={S.td}>{fmt(a.load)}</td>
                    <td style={{ ...S.td, color: 'var(--text-secondary)' }}>{a.reason}</td>
                    <td style={S.td}>
                      <span style={S.badge('var(--danger)')}>{a.severity}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            No anomalies detected for the current day.
          </div>
        )}
      </DetailOverlay>

      <DetailOverlay
        open={activePanel === 'weather'}
        title="Weather Snapshot"
        subtitle={`Block ${snapBlock + 1} • ${blockToTime(snapBlock)} • ${weatherCond}`}
        onClose={() => setActivePanel(null)}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 16 }}>
          <div style={{ ...S.card, minHeight: 220 }}>
            <div
              style={{
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
                flex: 1,
              }}
            >
              <WeatherIcon condition={weatherCond} size={48} />
              <div style={{ fontSize: 14, fontWeight: 700 }}>{weatherCond}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{dateStr}</div>
            </div>
          </div>
          <div style={{ ...S.card, minHeight: 220 }}>
            <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                { l: 'Temperature', v: `${weatherAgg.tempNow.toFixed(1)}°C`, c: 'var(--tone-warm)' },
                { l: 'Feels Like', v: `${feelsLike.toFixed(1)}°C`, c: 'var(--tone-warm)' },
                {
                  l: 'Temp Range',
                  v: `${weatherAgg.tempMin.toFixed(1)}–${weatherAgg.tempMax.toFixed(1)}°C`,
                  c: 'var(--tone-warm)',
                },
                { l: 'Humidity', v: `${weatherAgg.humNow.toFixed(1)}%`, c: 'var(--info)' },
                {
                  l: 'Precipitation',
                  v:
                    (() => {
                      const p = weatherAgg.precipNow;
                      return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2);
                    })() + ' mm',
                  c: 'var(--accent2)',
                },
                {
                  l: 'Rain (Total)',
                  v:
                    (() => {
                      const p = weatherAgg.precipTotal;
                      return p > 0 && p < 0.01 ? p.toFixed(4) : p < 1 ? p.toFixed(3) : p.toFixed(2);
                    })() + ' mm',
                  c: 'var(--accent2)',
                },
                { l: 'Cloud', v: `${weatherAgg.cloudNow.toFixed(0)}%`, c: 'var(--text-dim)' },
                { l: 'Avg Humidity', v: `${weatherAgg.humAvg.toFixed(1)}%`, c: 'var(--info)' },
              ].map((r, i) => (
                <div
                  key={i}
                  style={{
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--outline)',
                    borderRadius: 12,
                    padding: '12px 14px',
                  }}
                >
                  <div style={S.miniLabel}>{r.l}</div>
                  <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: r.c }}>
                    {r.v}
                  </div>
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
        contentStyle={{
          width: 'min(96vw, 1680px)',
          maxWidth: 1680,
          maxHeight: '92vh',
          borderRadius: 18,
        }}
        bodyStyle={{ overflow: 'hidden', padding: 20 }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div
            style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 12 }}
          >
            <div
              style={{
                background:
                  'linear-gradient(180deg, rgba(var(--panel-rgb), 0.96), rgba(var(--panel-rgb), 0.96))',
                border: '1px solid var(--outline)',
                borderRadius: 16,
                padding: '16px 18px',
                boxShadow: '0 18px 36px rgba(var(--shadow-rgb), 0.18)',
              }}
            >
              <div style={S.miniLabel}>Strongest Positive</div>
              <div
                style={{
                  marginTop: 10,
                  fontSize: 22,
                  fontWeight: 700,
                  color: correlationHighlights.strongestPositive.color,
                }}
              >
                {correlationHighlights.strongestPositive.label}
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                ρ +{Math.abs(correlationHighlights.strongestPositive.value).toFixed(2)} with load
              </div>
            </div>
            <div
              style={{
                background:
                  'linear-gradient(180deg, rgba(var(--panel-rgb), 0.96), rgba(var(--panel-rgb), 0.96))',
                border: '1px solid var(--outline)',
                borderRadius: 16,
                padding: '16px 18px',
                boxShadow: '0 18px 36px rgba(var(--shadow-rgb), 0.18)',
              }}
            >
              <div style={S.miniLabel}>Strongest Negative</div>
              <div style={{ marginTop: 10, fontSize: 22, fontWeight: 700, color: 'var(--danger)' }}>
                {correlationHighlights.strongestNegative.label}
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                ρ {correlationHighlights.strongestNegative.value.toFixed(2)} with load
              </div>
            </div>
            <div
              style={{
                background:
                  'radial-gradient(circle at top right, rgba(var(--accent-rgb), 0.14), transparent 50%), linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.96))',
                border: '1px solid var(--outline)',
                borderRadius: 16,
                padding: '16px 18px',
                boxShadow: '0 18px 36px rgba(var(--shadow-rgb), 0.22)',
              }}
            >
              <div style={S.miniLabel}>Dominant Pair</div>
              <div style={{ marginTop: 10, fontSize: 20, fontWeight: 700, color: 'var(--text)' }}>
                {correlationHighlights.strongestPair?.pair || '--'}
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                Absolute ρ {Math.abs(correlationHighlights.strongestPair?.value || 0).toFixed(2)}{' '}
                across the matrix
              </div>
            </div>
          </div>
          <div
            style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 12 }}
          >
            {[
              { key: 'temp', label: 'Temp → Load', value: rhoTemp, color: 'var(--tone-warm)' },
              { key: 'hum', label: 'Hum → Load', value: rhoHum, color: 'var(--info)' },
              { key: 'cloud', label: 'Cloud → Load', value: rhoCloud, color: 'var(--accent)' },
              { key: 'precip', label: 'Precip → Load', value: rhoPrecip, color: 'var(--accent2)' },
            ].map((item) => (
              <div
                key={item.key}
                style={{
                  background:
                    'linear-gradient(180deg, rgba(var(--panel-rgb), 0.96), rgba(var(--panel-rgb), 0.96))',
                  border: '1px solid var(--outline)',
                  borderRadius: 16,
                  padding: '14px 12px 10px',
                  boxShadow: '0 14px 28px rgba(var(--shadow-rgb), 0.14)',
                }}
              >
                <ArcGauge value={item.value} sublabel={item.label} size={92} />
              </div>
            ))}
          </div>
          <div
            style={{
              background: 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.98))',
              border: '1px solid var(--outline)',
              borderRadius: 18,
              padding: 16,
              boxShadow: '0 22px 42px rgba(var(--shadow-rgb), 0.2)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                marginBottom: 12,
              }}
            >
              <div style={{ ...S.cardTitle, padding: 0 }}>Correlation Heatmap</div>
              <span
                style={S.badge(
                  Math.abs(correlationHighlights.strongestPair?.value || 0) > 0.7
                    ? 'var(--success)'
                    : 'var(--warning)'
                )}
              >
                Matrix View
              </span>
            </div>
            <div style={{ overflow: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'separate',
                  borderSpacing: 6,
                  fontSize: 10,
                }}
              >
                <thead>
                  <tr>
                    {['', 'LOAD', 'TEMP', 'HUMIDITY', 'PRECIP', 'RAIN', 'CLOUD'].map((h) => (
                      <th
                        key={h}
                        style={{
                          padding: '8px 10px',
                          color: 'var(--text-muted)',
                          fontWeight: 700,
                          textAlign: 'center',
                          letterSpacing: 0.5,
                          fontSize: 12,
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlationMatrix.map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: '10px 10px', fontWeight: 700, color: 'var(--text-secondary)' }}>
                        {row.l}
                      </td>
                      {row.vals.map((v, ci) => {
                        const abs = Math.abs(v);
                        const bg =
                          v === 1
                            ? 'rgba(var(--accent-rgb), 0.16)'
                            : v > 0
                              ? `color-mix(in srgb, var(--success) ${Math.min(0.1 + abs * 0.22, 0.3) * 100}%, transparent)`
                              : `color-mix(in srgb, var(--danger) ${Math.min(0.08 + abs * 0.22, 0.28) * 100}%, transparent)`;
                        const color =
                          v === 1
                            ? 'color-mix(in srgb, var(--accent) 60%, var(--text))'
                            : v > 0
                              ? 'color-mix(in srgb, var(--success) 60%, var(--text))'
                              : 'color-mix(in srgb, var(--danger) 60%, var(--text))';
                        return (
                          <td
                            key={ci}
                            style={{
                              padding: '14px 10px',
                              textAlign: 'center',
                              color,
                              background: bg,
                              border: '1px solid rgba(var(--overlay-rgb), 0.04)',
                              borderRadius: 12,
                              boxShadow: 'inset 0 1px 0 rgba(var(--overlay-rgb), 0.02)',
                            }}
                          >
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
    </VpPageShell>
  );
}
