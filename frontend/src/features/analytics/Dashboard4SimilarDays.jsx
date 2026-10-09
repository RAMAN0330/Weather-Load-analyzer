import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, Search, Calendar } from 'lucide-react';
import {
  fetchPipelineSimilarity,
  fetchPipelineLoadRange,
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
    COLORS: [tk.warm, tk.accent, tk.success, tk.danger, tk.warning, tk.accent2, tk.textSecondary, tk.info],
  };
}

function today() { return new Date().toISOString().slice(0, 10); }
function daysAgo(n) { const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString().slice(0, 10); }

function Card({ title, children }) {
  return (
    <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--outline)', borderRadius: 12, padding: 20, marginBottom: 20 }}>
      <div className="mb-4">
        <span className="font-semibold text-sm" style={{ color: 'var(--text)' }}>{title}</span>
      </div>
      {children}
    </div>
  );
}
function Inp({ ...p }) {
  return <input {...p} style={{ background: 'rgba(var(--overlay-rgb),0.06)', border: '1px solid rgba(var(--overlay-rgb),0.12)', borderRadius: 8, color: 'var(--text)', padding: '6px 10px', fontSize: 13, ...p.style }} />;
}
function Label({ children }) {
  return <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>{children}</label>;
}
function Tab({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{ padding: '7px 18px', borderRadius: 0, fontSize: 13, cursor: 'pointer', border: 'none', background: 'transparent', color: active ? 'var(--text)' : 'var(--text-muted)', borderBottom: active ? '2px solid var(--accent2)' : '2px solid transparent', fontWeight: active ? 600 : 400 }}>
      {children}
    </button>
  );
}

const WX_LABELS = { temp: 'Temperature (°C)', humidity: 'Humidity (%)', precip: 'Precipitation (mm)' };
const WX_COLORS = { temp: 'var(--tone-warm)', humidity: 'var(--accent)', precip: 'var(--success)' };

export default function Dashboard4SimilarDays({ state }) {
  const [targetDate, setTargetDate] = useState(daysAgo(1));
  const [topK, setTopK]             = useState(5);
  const [result, setResult]         = useState(null);
  const [loadProfiles, setLoadProfiles] = useState({});
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [activeTab, setActiveTab]   = useState('table');
  const [wxVar, setWxVar]           = useState('temp');

  const blockAxis = useMemo(() => buildTimeAxis(), []);
  const tk = useChartTokens();
  const { TOOLTIP, LEGEND, AXIS, COLORS } = useMemo(() => chartTheme(tk), [tk]);

  const findSimilar = useCallback(() => {
    setLoading(true);
    setError(null);
    setResult(null);
    setLoadProfiles({});
    fetchPipelineSimilarity(state, targetDate, 'euclidean', topK)
      .then((data) => {
        setResult(data);
        // Fetch load profiles for target + similar dates
        const dates = [data.target_date, ...(data.similar || []).map((s) => s.date)];
        const minDate = dates.reduce((a, b) => (a < b ? a : b));
        const maxDate = dates.reduce((a, b) => (a > b ? a : b));
        return fetchPipelineLoadRange(state, minDate, maxDate).then((loadRows) => {
          const byDate = {};
          for (const r of loadRows) {
            if (!byDate[r.date]) byDate[r.date] = Array(96).fill(null);
            if (r.time_block >= 1 && r.time_block <= 96)
              byDate[r.date][r.time_block - 1] = r.load ?? null;
          }
          setLoadProfiles(byDate);
          setLoading(false);
        });
      })
      .catch((e) => { setError(e.message || 'Fetch error'); setLoading(false); });
  }, [state, targetDate, topK]);

  // ── Ranked table ─────────────────────────────────────────────────────────

  const RankedTable = useMemo(() => {
    if (!result?.similar?.length) return null;
    return (
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.1)' }}>
              {['Rank','Date','Euclidean Distance','Day of Week','Category'].map((h) => (
                <th key={h} style={{ padding: '6px 12px', color: 'var(--text-muted)', fontWeight: 500, textAlign: h === 'Rank' || h === 'Euclidean Distance' ? 'right' : 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.07)', background: 'color-mix(in srgb, var(--tone-warm) 7%, transparent)' }}>
              <td style={{ padding: '7px 12px', textAlign: 'right', color: 'var(--tone-warm)', fontWeight: 600 }}>Target</td>
              <td style={{ padding: '7px 12px', color: 'var(--tone-warm)', fontWeight: 600 }}>{result.target_date}</td>
              <td colSpan={3} style={{ padding: '7px 12px', color: 'var(--text-dim)', fontSize: 11 }}>Target day</td>
            </tr>
            {result.similar.map((s, i) => {
              const dt = new Date(s.date);
              const dow = dt.toLocaleDateString('en-IN', { weekday: 'short' });
              return (
                <tr key={s.date} style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.05)', background: i % 2 ? 'rgba(var(--overlay-rgb),0.015)' : 'transparent' }}>
                  <td style={{ padding: '7px 12px', textAlign: 'right', color: 'var(--accent2)', fontWeight: 600 }}>#{i + 1}</td>
                  <td style={{ padding: '7px 12px', color: 'var(--text)' }}>{s.date}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'right', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{s.distance?.toFixed(2) ?? '—'}</td>
                  <td style={{ padding: '7px 12px', color: 'var(--text-secondary)' }}>{dow}</td>
                  <td style={{ padding: '7px 12px' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, background: dt.getDay() === 0 ? 'color-mix(in srgb, var(--warning) 15%, transparent)' : 'var(--success-dim)', color: dt.getDay() === 0 ? 'var(--warning)' : 'var(--success)' }}>
                      {dt.getDay() === 0 ? 'Sunday' : 'Working'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }, [result]);

  // ── Weather overlay chart ─────────────────────────────────────────────────

  const wxOverlayOption = useMemo(() => {
    if (!result?.profiles) return {};
    const allDates = [result.target_date, ...(result.similar || []).map((s) => s.date)];
    return {
      backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
      grid: { top: 40, right: 20, bottom: 50, left: 60 },
      xAxis: { type: 'category', data: blockAxis, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 }, name: 'Block' },
      yAxis: { type: 'value', ...AXIS, name: WX_LABELS[wxVar] || wxVar, nameTextStyle: { color: tk.textSecondary, fontSize: 11 } },
      series: allDates.map((d, i) => {
        const profile = result.profiles[d] || {};
        return {
          name: d === result.target_date ? `${d} (target)` : d,
          type: 'line',
          data: profile[wxVar] || Array(96).fill(null),
          color: COLORS[i % COLORS.length],
          symbol: 'none',
          lineStyle: { width: d === result.target_date ? 2.5 : 1.5, type: d === result.target_date ? 'solid' : 'dashed' },
          connectNulls: false,
        };
      }),
    };
  }, [result, wxVar, blockAxis, tk]);

  // ── Load profile chart ────────────────────────────────────────────────────

  const loadOverlayOption = useMemo(() => {
    if (!result) return {};
    const allDates = [result.target_date, ...(result.similar || []).map((s) => s.date)];
    return {
      backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
      grid: { top: 40, right: 20, bottom: 50, left: 70 },
      xAxis: { type: 'category', data: blockAxis, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 }, name: 'Block' },
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: tk.textSecondary, fontSize: 11 } },
      series: allDates.map((d, i) => ({
        name: d === result.target_date ? `${d} (target)` : d,
        type: 'line',
        data: loadProfiles[d] || Array(96).fill(null),
        color: COLORS[i % COLORS.length],
        symbol: 'none',
        lineStyle: { width: d === result.target_date ? 2.5 : 1.5, type: d === result.target_date ? 'solid' : 'dashed' },
        connectNulls: false,
      })),
    };
  }, [result, loadProfiles, blockAxis, tk]);

  return (
    <div className="p-6" style={{ maxWidth: 1200, margin: '0 auto', width: '100%' }}>

      {/* Controls */}
      <Card title="Find Similar Weather Days">
        <div className="flex flex-wrap items-end gap-5">
          <div>
            <Label>Target Date</Label>
            <Inp type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
          </div>
          <div>
            <Label>Top K Similar Days</Label>
            <select value={topK} onChange={(e) => setTopK(+e.target.value)}
              style={{ background: 'rgba(var(--overlay-rgb),0.06)', border: '1px solid rgba(var(--overlay-rgb),0.12)', borderRadius: 8, color: 'var(--text)', padding: '6px 10px', fontSize: 13, cursor: 'pointer' }}>
              {[3,5,7,10].map((n) => <option key={n} value={n} style={{ background: 'var(--bg-panel)' }}>{n}</option>)}
            </select>
          </div>
          <button onClick={findSimilar}
            className="flex items-center gap-1.5 text-sm font-semibold px-5 py-2 rounded-lg"
            style={{ background: 'var(--accent2)', color: 'var(--accent-fg)', border: 'none', cursor: 'pointer', height: 36, boxShadow: '0 4px 14px color-mix(in srgb, var(--accent2) 25%, transparent)' }}
            disabled={loading}
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Find Similar Days
          </button>
          {error && <span className="text-xs" style={{ color: 'var(--danger)' }}>{error}</span>}
        </div>

        {!result && !loading && (
          <div className="flex flex-col items-center py-14" style={{ color: 'var(--text-dim)' }}>
            <Calendar className="h-10 w-10 mb-3" style={{ opacity: 0.4 }} />
            <p className="text-sm">Pick a target date and click "Find Similar Days"</p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-dim)' }}>
              Computes Euclidean distance over 96-block weather vectors (temperature, humidity, precipitation)
            </p>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--accent2)' }} />
          </div>
        )}
      </Card>

      {/* Results */}
      {result && !loading && (
        <>
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Top {result.similar?.length} days similar to <span style={{ color: 'var(--tone-warm)', fontWeight: 600 }}>{result.target_date}</span> — ranked by Euclidean distance over weather profile
            </p>
          </div>

          {/* Sub-tabs */}
          <div style={{ borderBottom: '1px solid rgba(var(--overlay-rgb),0.08)', marginBottom: 20 }}>
            <Tab active={activeTab === 'table'}   onClick={() => setActiveTab('table')}>Ranked Similar Days</Tab>
            <Tab active={activeTab === 'weather'} onClick={() => setActiveTab('weather')}>Weather Overlay</Tab>
            <Tab active={activeTab === 'load'}    onClick={() => setActiveTab('load')}>Load Profile</Tab>
          </div>

          {activeTab === 'table' && (
            <Card title="Similar Days — Ranked by Euclidean Distance">
              {RankedTable || <p className="text-sm text-center py-6" style={{ color: 'var(--text-dim)' }}>No similar days found</p>}
            </Card>
          )}

          {activeTab === 'weather' && (
            <Card title="Weather Profile Comparison — 96-block overlay">
              <div className="flex gap-3 mb-4">
                {Object.entries(WX_LABELS).map(([k, label]) => (
                  <button key={k} onClick={() => setWxVar(k)}
                    style={{ padding: '4px 12px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none',
                      background: wxVar === k ? WX_COLORS[k] : 'rgba(var(--overlay-rgb),0.07)',
                      color: wxVar === k ? 'var(--accent-fg)' : 'var(--text-muted)' }}>
                    {label}
                  </button>
                ))}
              </div>
              <ReactECharts option={wxOverlayOption} style={{ height: 320 }} />
              <p className="text-xs mt-2" style={{ color: 'var(--text-dim)' }}>
                Solid line = target day · Dashed = similar days
              </p>
            </Card>
          )}

          {activeTab === 'load' && (
            <Card title="Load Profile Comparison — 96-block overlay">
              {Object.keys(loadProfiles).length === 0
                ? <p className="text-sm text-center py-8" style={{ color: 'var(--text-dim)' }}>No load data found for matched dates</p>
                : <>
                    <ReactECharts option={loadOverlayOption} style={{ height: 320 }} />
                    <p className="text-xs mt-2" style={{ color: 'var(--text-dim)' }}>
                      Solid line = target day · Dashed = similar days
                    </p>
                  </>}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
