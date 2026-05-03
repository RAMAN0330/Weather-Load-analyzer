import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, RefreshCw, BarChart2 } from 'lucide-react';
import {
  fetchPipelineWeatherRange,
  blockToTime,
  buildTimeAxis,
} from '../pipeline/pipelineApi';

// ── Theme constants ─────────────────────────────────────────────────────────
const TOOLTIP = {
  trigger: 'axis',
  backgroundColor: 'rgba(20,19,26,0.95)',
  borderColor: 'rgba(255,255,255,0.1)',
  borderWidth: 1,
  textStyle: { color: '#F0F2F8', fontSize: 12 },
  axisPointer: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
};
const LEGEND = {
  type: 'scroll',
  bottom: 0,
  icon: 'roundRect',
  itemWidth: 12,
  itemHeight: 6,
  textStyle: { fontSize: 11, color: '#B8BDCC' },
};
const AXIS = {
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.18)' } },
  axisTick: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
  axisLabel: { color: '#B8BDCC', fontSize: 11 },
  splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
};
const COLORS = ['#F07825','#5B9FE4','#34D399','#F87171','#FBBF24','#C084FC','#45b7d1','#F472B6'];

const WX_VARS = [
  { key: 'temperature_2m',       label: 'Temperature (°C)',  unit: '°C' },
  { key: 'apparent_temperature', label: 'Apparent Temp (°C)',unit: '°C' },
  { key: 'humidity',             label: 'Humidity (%)',       unit: '%'  },
  { key: 'relative_humidity_2m', label: 'Rel. Humidity (%)', unit: '%'  },
  { key: 'precipitation',        label: 'Precipitation (mm)',unit: 'mm' },
  { key: 'cloud_cover',          label: 'Cloud Cover (%)',   unit: '%'  },
  { key: 'wind_speed_10m',       label: 'Wind Speed (m/s)', unit: 'm/s'},
  { key: 'sunshine_duration',    label: 'Sunshine (s)',      unit: 's'  },
];

function today() { return new Date().toISOString().slice(0, 10); }
function daysAgo(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
}

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
  return (
    <input
      {...props}
      style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '6px 10px', fontSize: 13, ...props.style }}
    />
  );
}

function Chip({ active, onClick, children, color }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '3px 10px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none',
        background: active ? (color || '#F07825') : 'rgba(255,255,255,0.07)',
        color: active ? '#fff' : 'rgba(255,255,255,0.5)',
        transition: 'all .15s',
      }}
    >{children}</button>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function dateBlockToDatetime(date, block) {
  const dt = new Date(date + 'T00:00:00');
  dt.setMinutes(dt.getMinutes() + (block - 1) * 15);
  return dt.toISOString().slice(0, 16).replace('T', ' ');
}

function detectVars(rows) {
  if (!rows.length) return [];
  const sample = rows[0];
  return WX_VARS.filter((v) => v.key in sample && sample[v.key] !== null && sample[v.key] !== undefined);
}

export default function Dashboard1Weather({ state }) {
  const [fromDate, setFromDate] = useState(daysAgo(14));
  const [toDate, setToDate] = useState(today());
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [statsOpen, setStatsOpen] = useState(false);
  const [selectedVars, setSelectedVars] = useState(['temperature_2m', 'humidity', 'precipitation']);
  const [overlayDates, setOverlayDates] = useState([]);
  const [overlayDateInput, setOverlayDateInput] = useState(today());

  const availVars = useMemo(() => detectVars(rows), [rows]);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchPipelineWeatherRange(state, fromDate, toDate)
      .then((data) => {
        setRows(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((e) => { setError(e.message || 'Fetch error'); setLoading(false); });
  }, [state, fromDate, toDate]);

  useEffect(() => { load(); }, [load]);

  // ── Derived data ──────────────────────────────────────────────────────────

  const timeSeriesOption = useMemo(() => {
    if (!rows.length) return {};
    const grouped = {};
    for (const row of rows) {
      const dt = dateBlockToDatetime(row.date, row.time_block);
      grouped[dt] = row;
    }
    const times = Object.keys(grouped).sort();
    const activeVars = selectedVars.filter((v) => availVars.some((a) => a.key === v));
    return {
      backgroundColor: 'transparent',
      tooltip: { ...TOOLTIP },
      legend: { ...LEGEND },
      grid: { top: 40, right: 20, bottom: 50, left: 60 },
      xAxis: {
        type: 'category',
        data: times,
        ...AXIS,
        axisLabel: { ...AXIS.axisLabel, rotate: 30, formatter: (v) => v.slice(5, 16) },
      },
      yAxis: { type: 'value', ...AXIS },
      series: activeVars.map((vk, i) => {
        const meta = WX_VARS.find((w) => w.key === vk);
        return {
          name: meta?.label || vk,
          type: 'line',
          data: times.map((t) => grouped[t]?.[vk] ?? null),
          lineStyle: { width: 1.5 },
          symbol: 'none',
          color: COLORS[i % COLORS.length],
          connectNulls: false,
        };
      }),
    };
  }, [rows, selectedVars, availVars]);

  const blockOverlayOption = useMemo(() => {
    if (!rows.length || !overlayDates.length) return {};
    const byDate = {};
    for (const row of rows) {
      if (!byDate[row.date]) byDate[row.date] = {};
      byDate[row.date][row.time_block] = row;
    }
    const blocks = buildTimeAxis();
    const activeVar = selectedVars[0] || availVars[0]?.key;
    if (!activeVar) return {};
    return {
      backgroundColor: 'transparent',
      tooltip: { ...TOOLTIP },
      legend: { ...LEGEND },
      grid: { top: 40, right: 20, bottom: 50, left: 60 },
      xAxis: {
        type: 'category',
        data: blocks,
        ...AXIS,
        axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 },
        name: 'Block (15 min)',
        nameTextStyle: { color: '#B8BDCC', fontSize: 11 },
      },
      yAxis: { type: 'value', ...AXIS, name: WX_VARS.find((w) => w.key === activeVar)?.unit || '' },
      series: overlayDates.map((d, i) => ({
        name: d,
        type: 'line',
        data: Array.from({ length: 96 }, (_, idx) => byDate[d]?.[idx + 1]?.[activeVar] ?? null),
        lineStyle: { width: 1.5 },
        symbol: 'none',
        color: COLORS[i % COLORS.length],
        connectNulls: false,
      })),
    };
  }, [rows, overlayDates, selectedVars, availVars]);

  // ── Daily stats ───────────────────────────────────────────────────────────

  const dailyStats = useMemo(() => {
    if (!rows.length) return [];
    const byDate = {};
    for (const row of rows) {
      if (!byDate[row.date]) byDate[row.date] = {};
      for (const { key } of WX_VARS) {
        if (row[key] !== null && row[key] !== undefined) {
          if (!byDate[row.date][key]) byDate[row.date][key] = [];
          byDate[row.date][key].push(row[key]);
        }
      }
    }
    const activeVarsInfo = availVars.filter((v) => selectedVars.includes(v.key));
    return Object.entries(byDate).sort().map(([date, varMap]) => {
      const stats = {};
      for (const { key, unit } of activeVarsInfo) {
        const vals = varMap[key] || [];
        if (vals.length) {
          stats[key] = {
            avg: (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1),
            min: Math.min(...vals).toFixed(1),
            max: Math.max(...vals).toFixed(1),
            unit,
            peakBlock: (() => {
              let maxI = 0, maxV = -Infinity;
              (varMap[key] || []).forEach((v, i) => { if (v > maxV) { maxV = v; maxI = i; } });
              return blockToTime(maxI + 1);
            })(),
          };
        }
      }
      return { date, stats };
    });
  }, [rows, selectedVars, availVars]);

  const activeVarsForTable = availVars.filter((v) => selectedVars.includes(v.key));

  return (
    <div className="p-6" style={{ maxWidth: 1200, margin: '0 auto', width: '100%', overflowX: 'hidden' }}>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <div>
          <Label>From</Label>
          <Input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} />
        </div>
        <div>
          <Label>To</Label>
          <Input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} />
        </div>
        <button
          onClick={load}
          className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg transition-all"
          style={{ background: '#F07825', color: '#fff', border: 'none', cursor: 'pointer', height: 34 }}
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Load
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
        {!loading && <span className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>{rows.length} rows</span>}
        {dailyStats.length > 0 && (
          <button
            onClick={() => setStatsOpen(true)}
            className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg transition-all"
            style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', color: '#F0F2F8', cursor: 'pointer', height: 34, marginLeft: 'auto' }}
          >
            <BarChart2 className="h-3.5 w-3.5" /> Stats
          </button>
        )}
      </div>

      {/* Var selector */}
      {availVars.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-6">
          {availVars.map(({ key, label }, i) => (
            <Chip
              key={key}
              active={selectedVars.includes(key)}
              color={COLORS[i % COLORS.length]}
              onClick={() => setSelectedVars((prev) =>
                prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
              )}
            >{label}</Chip>
          ))}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin" style={{ color: '#F07825' }} />
        </div>
      )}

      {!loading && rows.length > 0 && (
        <>
          {/* Time-series */}
          <Card title={`Time-Series — Multi-Variable (${fromDate} → ${toDate})`}>
            <ReactECharts option={timeSeriesOption} style={{ height: 340 }} theme="dark" />
          </Card>

          {/* Overlay by blocks */}
          <Card
            title={`Block Overlay — Daily Comparison (block 1–96, variable: ${WX_VARS.find((w) => w.key === selectedVars[0])?.label || selectedVars[0] || '—'})`}
            extra={
              <div className="flex items-center gap-2">
                <Input
                  type="date"
                  value={overlayDateInput}
                  onChange={(e) => setOverlayDateInput(e.target.value)}
                  style={{ padding: '4px 8px', fontSize: 12 }}
                />
                <button
                  onClick={() => {
                    if (overlayDateInput && !overlayDates.includes(overlayDateInput))
                      setOverlayDates((p) => [...p, overlayDateInput].slice(-8));
                  }}
                  style={{ background: '#F07825', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}
                >+ Add</button>
                {overlayDates.length > 0 && (
                  <button
                    onClick={() => setOverlayDates([])}
                    style={{ background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.5)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}
                  >Clear</button>
                )}
              </div>
            }
          >
            {overlayDates.length === 0 ? (
              <p className="text-center py-10 text-sm" style={{ color: 'rgba(255,255,255,0.35)' }}>
                Add dates above to compare days side-by-side
              </p>
            ) : (
              <>
                <div className="flex flex-wrap gap-2 mb-3">
                  {overlayDates.map((d, i) => (
                    <span key={d} className="text-xs px-2 py-0.5 rounded-full" style={{ background: COLORS[i % COLORS.length] + '30', color: COLORS[i % COLORS.length], border: `1px solid ${COLORS[i % COLORS.length]}50` }}>
                      {d} <button onClick={() => setOverlayDates((p) => p.filter((x) => x !== d))} style={{ marginLeft: 4, background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>×</button>
                    </span>
                  ))}
                </div>
                <ReactECharts option={blockOverlayOption} style={{ height: 300 }} theme="dark" />
              </>
            )}
          </Card>

        </>
      )}

      {/* Stats modal overlay */}
      {statsOpen && (
        <StatsModal title="Daily Stats — Avg / Min / Max (with peak-time annotation)" onClose={() => setStatsOpen(false)}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <th style={{ textAlign: 'left', padding: '6px 10px', color: 'rgba(255,255,255,0.45)', fontWeight: 500 }}>Date</th>
                {activeVarsForTable.map(({ key, label }) => (
                  <th key={key} colSpan={4} style={{ textAlign: 'center', padding: '6px 10px', color: 'rgba(255,255,255,0.45)', fontWeight: 500 }}>{label}</th>
                ))}
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                <th />
                {activeVarsForTable.map(({ key }) => (
                  <React.Fragment key={key}>
                    <th style={{ padding: '4px 8px', color: 'rgba(255,255,255,0.3)', fontWeight: 400, fontSize: 11 }}>Avg</th>
                    <th style={{ padding: '4px 8px', color: 'rgba(255,255,255,0.3)', fontWeight: 400, fontSize: 11 }}>Min</th>
                    <th style={{ padding: '4px 8px', color: 'rgba(255,255,255,0.3)', fontWeight: 400, fontSize: 11 }}>Max</th>
                    <th style={{ padding: '4px 8px', color: 'rgba(255,255,255,0.3)', fontWeight: 400, fontSize: 11 }}>Peak @</th>
                  </React.Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {dailyStats.map(({ date, stats }, ri) => (
                <tr key={date} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: ri % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                  <td style={{ padding: '6px 10px', color: '#F0F2F8', fontWeight: 500 }}>{date}</td>
                  {activeVarsForTable.map(({ key }) => {
                    const s = stats[key];
                    return s ? (
                      <React.Fragment key={key}>
                        <td style={{ padding: '6px 8px', color: '#F0F2F8', textAlign: 'right' }}>{s.avg}{s.unit}</td>
                        <td style={{ padding: '6px 8px', color: '#5B9FE4', textAlign: 'right' }}>{s.min}</td>
                        <td style={{ padding: '6px 8px', color: '#F07825', textAlign: 'right' }}>{s.max}</td>
                        <td style={{ padding: '6px 8px', color: '#B8BDCC', textAlign: 'right' }}>{s.peakBlock}</td>
                      </React.Fragment>
                    ) : (
                      <React.Fragment key={key}>
                        <td colSpan={4} style={{ padding: '6px 8px', color: 'rgba(255,255,255,0.2)', textAlign: 'center' }}>—</td>
                      </React.Fragment>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </StatsModal>
      )}

      {!loading && rows.length === 0 && !error && (
        <div className="flex flex-col items-center justify-center py-24" style={{ color: 'rgba(255,255,255,0.3)' }}>
          <BarChart2 className="h-12 w-12 mb-4" style={{ opacity: 0.3 }} />
          <p className="text-sm">No weather data for {state} in selected range</p>
        </div>
      )}
    </div>
  );
}
