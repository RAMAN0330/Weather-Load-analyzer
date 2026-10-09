import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, RefreshCw, Activity } from 'lucide-react';
import {
  fetchPipelineSldcRange,
  fetchPipelineSldcForecastRange,
  buildTimeAxis,
} from '../pipeline/pipelineApi';
import { useChartTokens, withAlpha } from '../../lib/chartTheme';

// ── Theme (resolved from CSS tokens; see docs/theming-guide.md) ────────────
function chartTheme(tk) {
  return {
    TOOLTIP: {
      trigger: 'axis',
      backgroundColor: tk.elevated,
      borderColor: tk.outline,
      borderWidth: 1,
      textStyle: { color: tk.text, fontSize: 12 },
    },
    LEGEND: {
      type: 'scroll', bottom: 0, icon: 'roundRect',
      itemWidth: 12, itemHeight: 6,
      textStyle: { fontSize: 11, color: tk.textSecondary },
    },
    AXIS: {
      axisLine: { lineStyle: { color: tk.outline } },
      axisTick: { lineStyle: { color: tk.outline } },
      axisLabel: { color: tk.textSecondary, fontSize: 11 },
      splitLine: { lineStyle: { color: withAlpha(tk.outline, 0.6), type: 'dashed' } },
    },
    ACTUAL_COLOR: tk.success,
    FORECAST_COLOR: tk.accent,
    OVERLAYCOLORS: [tk.warm, tk.accent, tk.success, tk.danger, tk.warning, tk.accent2, tk.textSecondary, tk.info],
  };
}

// Peak windows (block numbers, inclusive)
const MORNING_WIN  = [24, 40];
const SOLAR_WIN    = [40, 72];
const EVENING_WIN  = [73, 92];

function today() { return new Date().toISOString().slice(0, 10); }
function daysAgo(n) {
  const d = new Date(); d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
}
function dateBlockToDatetime(date, block) {
  const dt = new Date(date + 'T00:00:00');
  dt.setMinutes(dt.getMinutes() + (block - 1) * 15);
  return dt.toISOString().slice(0, 16).replace('T', ' ');
}
function safe(v) { return v == null || isNaN(v) ? null : v; }

function Card({ title, children, extra }) {
  return (
    <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--outline)', borderRadius: 18, padding: 22, marginBottom: 18, boxShadow: '0 4px 28px rgba(var(--shadow-rgb),0.18), 0 1px 0 rgba(var(--overlay-rgb),0.05)' }}>
      <div className="flex items-center justify-between mb-4">
        <span className="font-semibold text-sm" style={{ color: 'var(--text)' }}>{title}</span>
        {extra}
      </div>
      {children}
    </div>
  );
}

function StatsModal({ title, children, onClose }) {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(var(--shadow-rgb),0.72)', backdropFilter: 'blur(6px)' }}
      onClick={onClose}>
      <div style={{ background: 'var(--bg-panel)', border: '1px solid rgba(var(--overlay-rgb),0.1)', borderRadius: 20, padding: 24, maxWidth: '92vw', width: 860, maxHeight: '80vh', display: 'flex', flexDirection: 'column', boxShadow: '0 24px 64px rgba(var(--shadow-rgb),0.7)' }}
        onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text)' }}>{title}</span>
          <button onClick={onClose} style={{ background: 'rgba(var(--overlay-rgb),0.07)', border: 'none', borderRadius: 8, color: 'var(--text-muted)', cursor: 'pointer', padding: '4px 10px', fontSize: 13 }}>✕ Close</button>
        </div>
        <div style={{ overflowY: 'auto', overflowX: 'auto', flex: 1 }}>{children}</div>
      </div>
    </div>
  );
}
function Label({ children }) {
  return <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>{children}</label>;
}
function Input({ ...props }) {
  return <input {...props} style={{ background: 'rgba(var(--overlay-rgb),0.06)', border: '1px solid rgba(var(--overlay-rgb),0.12)', borderRadius: 8, color: 'var(--text)', padding: '6px 10px', fontSize: 13, ...props.style }} />;
}

// ── Error metrics helpers ────────────────────────────────────────────────────

function computeMetrics(actualMap, forecastMap) {
  const dates = [...new Set([...Object.keys(actualMap), ...Object.keys(forecastMap)])].sort();
  const results = [];
  for (const d of dates) {
    const aBlocks = actualMap[d] || {};
    const fBlocks = forecastMap[d] || {};
    const keys = Object.keys(aBlocks).filter((b) => fBlocks[b] != null && aBlocks[b] != null);
    if (keys.length < 96) continue;
    const pairs = keys.map((b) => ({ a: aBlocks[b], f: fBlocks[b] }));
    const mape = pairs.reduce((s, { a, f }) => s + Math.abs(a - f) / Math.max(Math.abs(a), 1), 0) / pairs.length * 100;
    const rmse = Math.sqrt(pairs.reduce((s, { a, f }) => s + (a - f) ** 2, 0) / pairs.length);
    const mad  = pairs.reduce((s, { a, f }) => s + Math.abs(a - f), 0) / pairs.length;
    const bigErrors = pairs.filter(({ a, f }) => Math.abs(a - f) > 500).length;
    results.push({ date: d, mape: mape.toFixed(2), rmse: rmse.toFixed(1), mad: mad.toFixed(1), bigErrors });
  }
  return results;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function Dashboard2Load({ state }) {
  const [fromDate, setFromDate] = useState(daysAgo(14));
  const [toDate, setToDate]     = useState(today());
  const [actualRows, setActualRows]   = useState([]);
  const [forecastRows, setForecastRows] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [overlayDates, setOverlayDates] = useState([]);
  const [overlayInput, setOverlayInput] = useState(today());
  const [statsOpen, setStatsOpen] = useState(false);
  const [overlayMode, setOverlayMode]   = useState('both'); // actual | forecast | both
  const tk = useChartTokens();
  const { TOOLTIP, LEGEND, AXIS, ACTUAL_COLOR, FORECAST_COLOR, OVERLAYCOLORS } = useMemo(() => chartTheme(tk), [tk]);

  const load = useCallback(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetchPipelineSldcRange(state, fromDate, toDate),
      fetchPipelineSldcForecastRange(state, fromDate, toDate),
    ])
      .then(([a, f]) => {
        setActualRows(Array.isArray(a) ? a : []);
        setForecastRows(Array.isArray(f) ? f : []);
        setLoading(false);
      })
      .catch((e) => { setError(e.message || 'Fetch error'); setLoading(false); });
  }, [state, fromDate, toDate]);

  useEffect(() => { load(); }, [load]);

  // Index by datetime
  const actualMap  = useMemo(() => {
    const m = {};
    for (const r of actualRows) {
      if (!m[r.date]) m[r.date] = {};
      m[r.date][r.time_block] = safe(r.actual);
    }
    return m;
  }, [actualRows]);

  const forecastMap = useMemo(() => {
    const m = {};
    for (const r of forecastRows) {
      if (!m[r.date]) m[r.date] = {};
      m[r.date][r.time_block] = safe(r.forecast);
    }
    return m;
  }, [forecastRows]);

  // ── Time-series chart ─────────────────────────────────────────────────────

  const tsOption = useMemo(() => {
    const allDates = [...new Set([...actualRows.map((r) => r.date), ...forecastRows.map((r) => r.date)])].sort();
    const times = [];
    const actualSeries = [];
    const forecastSeries = [];

    for (const d of allDates) {
      for (let b = 1; b <= 96; b++) {
        const dt = dateBlockToDatetime(d, b);
        times.push(dt);
        actualSeries.push(actualMap[d]?.[b] ?? null);
        forecastSeries.push(forecastMap[d]?.[b] ?? null);
      }
    }

    // Add 12h interval markLines
    const markLineData = [];
    for (let i = 0; i < times.length; i += 48) markLineData.push({ xAxis: i });

    return {
      backgroundColor: 'transparent',
      tooltip: TOOLTIP,
      legend: LEGEND,
      grid: { top: 40, right: 20, bottom: 50, left: 70 },
      xAxis: {
        type: 'category',
        data: times,
        ...AXIS,
        axisLabel: { ...AXIS.axisLabel, rotate: 30, formatter: (v) => v.slice(5, 16), interval: 47 },
      },
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: tk.textSecondary, fontSize: 11 } },
      series: [
        {
          name: 'Actual',
          type: 'line', data: actualSeries, symbol: 'none',
          color: ACTUAL_COLOR, lineStyle: { width: 1.5 }, connectNulls: false,
          markLine: { silent: true, lineStyle: { color: withAlpha(tk.outline, 0.8), type: 'dashed' }, data: markLineData.map((x) => [{ xAxis: x.xAxis }, { xAxis: x.xAxis }]) },
        },
        {
          name: 'Forecast',
          type: 'line', data: forecastSeries, symbol: 'none',
          color: FORECAST_COLOR, lineStyle: { width: 1.5, type: 'dashed' }, connectNulls: false,
        },
      ],
    };
  }, [actualRows, forecastRows, actualMap, forecastMap, tk]);

  // ── Daily overlay ─────────────────────────────────────────────────────────

  const blockAxis = useMemo(() => buildTimeAxis(), []);

  const overlayOption = useMemo(() => {
    if (!overlayDates.length) return {};
    const series = [];
    for (let i = 0; i < overlayDates.length; i++) {
      const d = overlayDates[i];
      const col = OVERLAYCOLORS[i % OVERLAYCOLORS.length];
      if (overlayMode !== 'forecast') {
        series.push({
          name: `${d} Actual`,
          type: 'line',
          data: Array.from({ length: 96 }, (_, b) => actualMap[d]?.[b + 1] ?? null),
          color: col, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: false,
        });
      }
      if (overlayMode !== 'actual') {
        series.push({
          name: `${d} Forecast`,
          type: 'line',
          data: Array.from({ length: 96 }, (_, b) => forecastMap[d]?.[b + 1] ?? null),
          color: col, symbol: 'none', lineStyle: { width: 1.5, type: 'dashed' }, connectNulls: false,
        });
      }
    }
    return {
      backgroundColor: 'transparent',
      tooltip: TOOLTIP,
      legend: LEGEND,
      grid: { top: 40, right: 20, bottom: 50, left: 70 },
      xAxis: { type: 'category', data: blockAxis, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 }, name: 'Block' },
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: tk.textSecondary, fontSize: 11 } },
      series,
    };
  }, [overlayDates, overlayMode, actualMap, forecastMap, blockAxis, tk]);

  // ── Daily summary ─────────────────────────────────────────────────────────

  const dailySummary = useMemo(() => {
    const dates = Object.keys(actualMap).sort();
    return dates.map((d) => {
      const blocks = actualMap[d] || {};
      const allVals = Object.values(blocks).filter((v) => v != null);
      const winAvg = (lo, hi) => {
        const w = Object.entries(blocks).filter(([b]) => +b >= lo && +b <= hi).map(([, v]) => v).filter((v) => v != null);
        return w.length ? (w.reduce((a, b) => a + b, 0) / w.length).toFixed(0) : '—';
      };
      return {
        date: d,
        morning: winAvg(...MORNING_WIN),
        solar: winAvg(...SOLAR_WIN),
        evening: winAvg(...EVENING_WIN),
        avg: allVals.length ? (allVals.reduce((a, b) => a + b, 0) / allVals.length).toFixed(0) : '—',
        peak: allVals.length ? Math.max(...allVals).toFixed(0) : '—',
      };
    });
  }, [actualMap]);

  // ── Error metrics ─────────────────────────────────────────────────────────

  const errorMetrics = useMemo(() => computeMetrics(actualMap, forecastMap), [actualMap, forecastMap]);

  const hasData = actualRows.length > 0 || forecastRows.length > 0;

  return (
    <div className="p-6" style={{ maxWidth: 1200, margin: '0 auto', width: '100%', overflowX: 'hidden' }}>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <div><Label>From</Label><Input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} /></div>
        <div><Label>To</Label><Input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} /></div>
        <button onClick={load} className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg"
          style={{ background: 'var(--accent)', color: 'var(--accent-fg)', border: 'none', cursor: 'pointer', height: 34 }}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Load
        </button>
        {error && <span className="text-xs" style={{ color: 'var(--danger)' }}>{error}</span>}
        {!loading && <span className="text-xs" style={{ color: 'var(--text-dim)' }}>{actualRows.length} actual, {forecastRows.length} forecast rows</span>}
        {(dailySummary.length > 0 || errorMetrics.length > 0) && (
          <button onClick={() => setStatsOpen(true)}
            className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg"
            style={{ background: 'rgba(var(--overlay-rgb),0.07)', border: '1px solid rgba(var(--overlay-rgb),0.12)', color: 'var(--text)', cursor: 'pointer', height: 34, marginLeft: 'auto' }}>
            <Activity className="h-3.5 w-3.5" /> Stats
          </button>
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      )}

      {!loading && hasData && (
        <>
          {/* Time-series: actual vs forecast */}
          <Card title="Time-Series — Actual vs Forecast (12h tick intervals)">
            <ReactECharts option={tsOption} style={{ height: 320 }} />
          </Card>

          {/* Daily overlay */}
          <Card
            title="Daily Overlay — by 15-min Block"
            extra={
              <div className="flex items-center gap-2">
                {['actual','forecast','both'].map((m) => (
                  <button key={m} onClick={() => setOverlayMode(m)}
                    style={{ padding: '3px 10px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none',
                      background: overlayMode === m ? 'var(--accent)' : 'rgba(var(--overlay-rgb),0.07)',
                      color: overlayMode === m ? 'var(--accent-fg)' : 'var(--text-muted)' }}>
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
                <Input type="date" value={overlayInput} onChange={(e) => setOverlayInput(e.target.value)} style={{ padding: '4px 8px', fontSize: 12 }} />
                <button onClick={() => { if (overlayInput && !overlayDates.includes(overlayInput)) setOverlayDates((p) => [...p, overlayInput].slice(-8)); }}
                  style={{ background: 'var(--accent)', color: 'var(--accent-fg)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>
                  + Add
                </button>
                {overlayDates.length > 0 && (
                  <button onClick={() => setOverlayDates([])}
                    style={{ background: 'rgba(var(--overlay-rgb),0.08)', color: 'var(--text-muted)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>
                    Clear
                  </button>
                )}
              </div>
            }
          >
            {overlayDates.length === 0 ? (
              <p className="text-center py-8 text-sm" style={{ color: 'var(--text-dim)' }}>Add dates to compare days side-by-side</p>
            ) : (
              <>
                <div className="flex flex-wrap gap-2 mb-3">
                  {overlayDates.map((d, i) => (
                    <span key={d} className="text-xs px-2 py-0.5 rounded-full"
                      style={{ background: withAlpha(OVERLAYCOLORS[i % OVERLAYCOLORS.length], 0.19), color: OVERLAYCOLORS[i % OVERLAYCOLORS.length], border: `1px solid ${withAlpha(OVERLAYCOLORS[i % OVERLAYCOLORS.length], 0.31)}` }}>
                      {d}
                      <button onClick={() => setOverlayDates((p) => p.filter((x) => x !== d))} style={{ marginLeft: 4, background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>×</button>
                    </span>
                  ))}
                </div>
                <ReactECharts option={overlayOption} style={{ height: 300 }} />
              </>
            )}
          </Card>

        </>
      )}

      {!loading && !hasData && !error && (
        <div className="flex flex-col items-center justify-center py-24" style={{ color: 'var(--text-dim)' }}>
          <Activity className="h-12 w-12 mb-4" style={{ opacity: 0.3 }} />
          <p className="text-sm">No load data for {state} in selected range</p>
        </div>
      )}

      {statsOpen && (
        <StatsModal title="Load Stats" onClose={() => setStatsOpen(false)}>
          {dailySummary.length > 0 && (
            <>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Daily Summary — Peak Windows (MW avg)</p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: 24 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.1)' }}>
                    {['Date','Morning (B24–40)','Solar (B40–72)','Evening (B73–92)','Day Avg','Peak MW'].map((h) => (
                      <th key={h} style={{ padding: '6px 10px', color: 'var(--text-muted)', fontWeight: 500, textAlign: h === 'Date' ? 'left' : 'right', fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dailySummary.map(({ date, morning, solar, evening, avg, peak }, i) => (
                    <tr key={date} style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.05)', background: i % 2 ? 'rgba(var(--overlay-rgb),0.015)' : 'transparent' }}>
                      <td style={{ padding: '6px 10px', color: 'var(--text)', fontWeight: 500 }}>{date}</td>
                      <td style={{ padding: '6px 10px', color: 'var(--warning)', textAlign: 'right' }}>{morning}</td>
                      <td style={{ padding: '6px 10px', color: 'var(--success)', textAlign: 'right' }}>{solar}</td>
                      <td style={{ padding: '6px 10px', color: 'var(--tone-warm)', textAlign: 'right' }}>{evening}</td>
                      <td style={{ padding: '6px 10px', color: 'var(--text-secondary)', textAlign: 'right' }}>{avg}</td>
                      <td style={{ padding: '6px 10px', color: 'var(--accent)', textAlign: 'right' }}>{peak}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {errorMetrics.length > 0 && (
            <>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Forecast Error</p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.1)' }}>
                    {['Date','MAPE (%)','RMSE (MW)','MAD (MW)','Errors > 500 MW'].map((h) => (
                      <th key={h} style={{ padding: '6px 10px', color: 'var(--text-muted)', fontWeight: 500, textAlign: h === 'Date' ? 'left' : 'right' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {errorMetrics.map(({ date, mape, rmse, mad, bigErrors }, i) => (
                    <tr key={date} style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.05)', background: i % 2 ? 'rgba(var(--overlay-rgb),0.015)' : 'transparent' }}>
                      <td style={{ padding: '6px 10px', color: 'var(--text)', fontWeight: 500 }}>{date}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: +mape > 5 ? 'var(--danger)' : +mape > 3 ? 'var(--warning)' : 'var(--success)' }}>{mape}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{rmse}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--text-secondary)' }}>{mad}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: bigErrors > 0 ? 'var(--danger)' : 'var(--success)' }}>{bigErrors}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </StatsModal>
      )}
    </div>
  );
}
