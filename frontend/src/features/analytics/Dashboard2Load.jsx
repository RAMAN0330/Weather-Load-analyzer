import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, RefreshCw, Activity } from 'lucide-react';
import {
  fetchPipelineSldcRange,
  fetchPipelineSldcForecastRange,
  buildTimeAxis,
} from '../pipeline/pipelineApi';

// ── Theme ───────────────────────────────────────────────────────────────────
const TOOLTIP = {
  trigger: 'axis',
  backgroundColor: 'rgba(20,19,26,0.95)',
  borderColor: 'rgba(255,255,255,0.1)',
  borderWidth: 1,
  textStyle: { color: '#F0F2F8', fontSize: 12 },
};
const LEGEND = {
  type: 'scroll', bottom: 0, icon: 'roundRect',
  itemWidth: 12, itemHeight: 6,
  textStyle: { fontSize: 11, color: '#B8BDCC' },
};
const AXIS = {
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.18)' } },
  axisTick: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
  axisLabel: { color: '#B8BDCC', fontSize: 11 },
  splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
};
const ACTUAL_COLOR   = '#F07825';
const FORECAST_COLOR = '#5B9FE4';

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
    <div style={{ background: 'rgba(255,255,255,0.035)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 18, padding: 22, marginBottom: 18, boxShadow: '0 4px 28px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-4">
        <span className="font-semibold text-sm" style={{ color: '#F0F2F8' }}>{title}</span>
        {extra}
      </div>
      {children}
    </div>
  );
}

function StatsModal({ title, children, onClose }) {
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(6px)' }}
      onClick={onClose}>
      <div style={{ background: '#16141c', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 20, padding: 24, maxWidth: '92vw', width: 860, maxHeight: '80vh', display: 'flex', flexDirection: 'column', boxShadow: '0 24px 64px rgba(0,0,0,0.7)' }}
        onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: 14, color: '#F0F2F8' }}>{title}</span>
          <button onClick={onClose} style={{ background: 'rgba(255,255,255,0.07)', border: 'none', borderRadius: 8, color: 'rgba(255,255,255,0.5)', cursor: 'pointer', padding: '4px 10px', fontSize: 13 }}>✕ Close</button>
        </div>
        <div style={{ overflowY: 'auto', overflowX: 'auto', flex: 1 }}>{children}</div>
      </div>
    </div>
  );
}
function Label({ children }) {
  return <label className="text-xs block mb-1" style={{ color: 'rgba(255,255,255,0.45)' }}>{children}</label>;
}
function Input({ ...props }) {
  return <input {...props} style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '6px 10px', fontSize: 13, ...props.style }} />;
}
const OVERLAYCOLORS = ['#F07825','#5B9FE4','#34D399','#F87171','#FBBF24','#C084FC','#F472B6','#45b7d1'];

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
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
      series: [
        {
          name: 'Actual',
          type: 'line', data: actualSeries, symbol: 'none',
          color: ACTUAL_COLOR, lineStyle: { width: 1.5 }, connectNulls: false,
          markLine: { silent: true, lineStyle: { color: 'rgba(255,255,255,0.08)', type: 'dashed' }, data: markLineData.map((x) => [{ xAxis: x.xAxis }, { xAxis: x.xAxis }]) },
        },
        {
          name: 'Forecast',
          type: 'line', data: forecastSeries, symbol: 'none',
          color: FORECAST_COLOR, lineStyle: { width: 1.5, type: 'dashed' }, connectNulls: false,
        },
      ],
    };
  }, [actualRows, forecastRows, actualMap, forecastMap]);

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
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
      series,
    };
  }, [overlayDates, overlayMode, actualMap, forecastMap, blockAxis]);

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
          style={{ background: '#F07825', color: '#fff', border: 'none', cursor: 'pointer', height: 34 }}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Load
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
        {!loading && <span className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{actualRows.length} actual, {forecastRows.length} forecast rows</span>}
        {(dailySummary.length > 0 || errorMetrics.length > 0) && (
          <button onClick={() => setStatsOpen(true)}
            className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', color: '#F0F2F8', cursor: 'pointer', height: 34, marginLeft: 'auto' }}>
            <Activity className="h-3.5 w-3.5" /> Stats
          </button>
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin" style={{ color: '#F07825' }} />
        </div>
      )}

      {!loading && hasData && (
        <>
          {/* Time-series: actual vs forecast */}
          <Card title="Time-Series — Actual vs Forecast (12h tick intervals)">
            <ReactECharts option={tsOption} style={{ height: 320 }} theme="dark" />
          </Card>

          {/* Daily overlay */}
          <Card
            title="Daily Overlay — by 15-min Block"
            extra={
              <div className="flex items-center gap-2">
                {['actual','forecast','both'].map((m) => (
                  <button key={m} onClick={() => setOverlayMode(m)}
                    style={{ padding: '3px 10px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none',
                      background: overlayMode === m ? '#F07825' : 'rgba(255,255,255,0.07)',
                      color: overlayMode === m ? '#fff' : 'rgba(255,255,255,0.5)' }}>
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
                <Input type="date" value={overlayInput} onChange={(e) => setOverlayInput(e.target.value)} style={{ padding: '4px 8px', fontSize: 12 }} />
                <button onClick={() => { if (overlayInput && !overlayDates.includes(overlayInput)) setOverlayDates((p) => [...p, overlayInput].slice(-8)); }}
                  style={{ background: '#F07825', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>
                  + Add
                </button>
                {overlayDates.length > 0 && (
                  <button onClick={() => setOverlayDates([])}
                    style={{ background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.5)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>
                    Clear
                  </button>
                )}
              </div>
            }
          >
            {overlayDates.length === 0 ? (
              <p className="text-center py-8 text-sm" style={{ color: 'rgba(255,255,255,0.35)' }}>Add dates to compare days side-by-side</p>
            ) : (
              <>
                <div className="flex flex-wrap gap-2 mb-3">
                  {overlayDates.map((d, i) => (
                    <span key={d} className="text-xs px-2 py-0.5 rounded-full"
                      style={{ background: OVERLAYCOLORS[i % OVERLAYCOLORS.length] + '30', color: OVERLAYCOLORS[i % OVERLAYCOLORS.length], border: `1px solid ${OVERLAYCOLORS[i % OVERLAYCOLORS.length]}50` }}>
                      {d}
                      <button onClick={() => setOverlayDates((p) => p.filter((x) => x !== d))} style={{ marginLeft: 4, background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>×</button>
                    </span>
                  ))}
                </div>
                <ReactECharts option={overlayOption} style={{ height: 300 }} theme="dark" />
              </>
            )}
          </Card>

        </>
      )}

      {!loading && !hasData && !error && (
        <div className="flex flex-col items-center justify-center py-24" style={{ color: 'rgba(255,255,255,0.3)' }}>
          <Activity className="h-12 w-12 mb-4" style={{ opacity: 0.3 }} />
          <p className="text-sm">No load data for {state} in selected range</p>
        </div>
      )}

      {statsOpen && (
        <StatsModal title="Load Stats" onClose={() => setStatsOpen(false)}>
          {dailySummary.length > 0 && (
            <>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.5)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Daily Summary — Peak Windows (MW avg)</p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: 24 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    {['Date','Morning (B24–40)','Solar (B40–72)','Evening (B73–92)','Day Avg','Peak MW'].map((h) => (
                      <th key={h} style={{ padding: '6px 10px', color: 'rgba(255,255,255,0.45)', fontWeight: 500, textAlign: h === 'Date' ? 'left' : 'right', fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dailySummary.map(({ date, morning, solar, evening, avg, peak }, i) => (
                    <tr key={date} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                      <td style={{ padding: '6px 10px', color: '#F0F2F8', fontWeight: 500 }}>{date}</td>
                      <td style={{ padding: '6px 10px', color: '#FBBF24', textAlign: 'right' }}>{morning}</td>
                      <td style={{ padding: '6px 10px', color: '#34D399', textAlign: 'right' }}>{solar}</td>
                      <td style={{ padding: '6px 10px', color: '#F07825', textAlign: 'right' }}>{evening}</td>
                      <td style={{ padding: '6px 10px', color: '#B8BDCC', textAlign: 'right' }}>{avg}</td>
                      <td style={{ padding: '6px 10px', color: '#5B9FE4', textAlign: 'right' }}>{peak}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {errorMetrics.length > 0 && (
            <>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'rgba(255,255,255,0.5)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Forecast Error</p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    {['Date','MAPE (%)','RMSE (MW)','MAD (MW)','Errors > 500 MW'].map((h) => (
                      <th key={h} style={{ padding: '6px 10px', color: 'rgba(255,255,255,0.45)', fontWeight: 500, textAlign: h === 'Date' ? 'left' : 'right' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {errorMetrics.map(({ date, mape, rmse, mad, bigErrors }, i) => (
                    <tr key={date} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                      <td style={{ padding: '6px 10px', color: '#F0F2F8', fontWeight: 500 }}>{date}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: +mape > 5 ? '#F87171' : +mape > 3 ? '#FBBF24' : '#34D399' }}>{mape}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: '#B8BDCC' }}>{rmse}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: '#B8BDCC' }}>{mad}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', color: bigErrors > 0 ? '#F87171' : '#34D399' }}>{bigErrors}</td>
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
