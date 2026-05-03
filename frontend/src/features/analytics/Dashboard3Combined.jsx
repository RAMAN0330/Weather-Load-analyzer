import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, RefreshCw, Layers } from 'lucide-react';
import {
  fetchPipelineWeatherRange,
  fetchPipelineWeatherLocRange,
  fetchPipelineSldcRange,
  buildTimeAxis,
} from '../pipeline/pipelineApi';

// ── Theme ────────────────────────────────────────────────────────────────────
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
const COLORS = ['#F07825','#5B9FE4','#34D399','#F87171','#FBBF24','#C084FC','#F472B6','#45b7d1'];
const WX_VARS = [
  { key: 'temperature_2m',       label: 'Temperature (°C)' },
  { key: 'apparent_temperature', label: 'Apparent Temp (°C)' },
  { key: 'humidity',             label: 'Humidity (%)' },
  { key: 'relative_humidity_2m', label: 'Rel. Humidity (%)' },
  { key: 'precipitation',        label: 'Precipitation (mm)' },
  { key: 'cloud_cover',          label: 'Cloud Cover (%)' },
  { key: 'wind_speed_10m',       label: 'Wind Speed (m/s)' },
];

function today() { return new Date().toISOString().slice(0, 10); }
function daysAgo(n) { const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString().slice(0, 10); }
function dtFromDateBlock(date, block) {
  const d = new Date(date + 'T00:00:00');
  d.setMinutes(d.getMinutes() + (block - 1) * 15);
  return d.toISOString().slice(0, 16).replace('T', ' ');
}

function Card({ title, children, extra, section }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.035)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 18, padding: 22, marginBottom: 18, boxShadow: '0 4px 28px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {section && <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: 'rgba(240,120,37,0.15)', color: '#F07825' }}>{section}</span>}
          <span className="font-semibold text-sm" style={{ color: '#F0F2F8' }}>{title}</span>
        </div>
        {extra}
      </div>
      {children}
    </div>
  );
}
function Label({ children }) {
  return <label className="text-xs block mb-1" style={{ color: 'rgba(255,255,255,0.45)' }}>{children}</label>;
}
function Inp({ ...p }) {
  return <input {...p} style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '6px 10px', fontSize: 13, ...p.style }} />;
}
function Chip({ active, onClick, color, children }) {
  return (
    <button onClick={onClick} style={{ padding: '3px 10px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none', background: active ? (color || '#F07825') : 'rgba(255,255,255,0.07)', color: active ? '#fff' : 'rgba(255,255,255,0.5)', transition: 'all .15s' }}>
      {children}
    </button>
  );
}

function AddDateBar({ overlayDates, onAdd, onRemove, onClear }) {
  const [input, setInput] = useState(today());
  return (
    <div className="flex flex-wrap items-center gap-2 mb-3">
      <Inp type="date" value={input} onChange={(e) => setInput(e.target.value)} style={{ padding: '4px 8px', fontSize: 12 }} />
      <button onClick={() => { if (input && !overlayDates.includes(input)) onAdd(input); }}
        style={{ background: '#F07825', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>+ Add</button>
      {overlayDates.length > 0 && (
        <button onClick={onClear}
          style={{ background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.5)', border: 'none', borderRadius: 6, padding: '4px 10px', fontSize: 12, cursor: 'pointer' }}>Clear</button>
      )}
      {overlayDates.map((d, i) => (
        <span key={d} className="text-xs px-2 py-0.5 rounded-full"
          style={{ background: COLORS[i % COLORS.length] + '30', color: COLORS[i % COLORS.length], border: `1px solid ${COLORS[i % COLORS.length]}50` }}>
          {d} <button onClick={() => onRemove(d)} style={{ marginLeft: 4, background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>×</button>
        </span>
      ))}
    </div>
  );
}

// ── Charts builders ──────────────────────────────────────────────────────────

function buildTsOption(rows, varKeys, dateRange) {
  const grouped = {};
  for (const r of rows) {
    const dt = dtFromDateBlock(r.date, r.time_block);
    if (!grouped[dt]) grouped[dt] = {};
    for (const k of varKeys) if (r[k] != null) grouped[dt][k] = r[k];
  }
  const times = Object.keys(grouped).sort();
  return {
    backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
    grid: { top: 40, right: 20, bottom: 50, left: 60 },
    xAxis: { type: 'category', data: times, ...AXIS, axisLabel: { ...AXIS.axisLabel, rotate: 30, formatter: (v) => v.slice(5, 16), interval: 47 } },
    yAxis: { type: 'value', ...AXIS },
    series: varKeys.map((vk, i) => ({
      name: WX_VARS.find((w) => w.key === vk)?.label || vk,
      type: 'line', data: times.map((t) => grouped[t]?.[vk] ?? null),
      color: COLORS[i % COLORS.length], symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: false,
    })),
  };
}

function buildLoadTsOption(rows) {
  const grouped = {};
  for (const r of rows) {
    const dt = dtFromDateBlock(r.date, r.time_block);
    grouped[dt] = r.actual ?? null;
  }
  const times = Object.keys(grouped).sort();
  return {
    backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
    grid: { top: 40, right: 20, bottom: 50, left: 70 },
    xAxis: { type: 'category', data: times, ...AXIS, axisLabel: { ...AXIS.axisLabel, rotate: 30, formatter: (v) => v.slice(5, 16), interval: 47 } },
    yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
    series: [{ name: 'Actual Load', type: 'line', data: times.map((t) => grouped[t]), color: '#F07825', symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: false }],
  };
}

function buildBlockOverlayOption(rows, days, valueKey, yName, label) {
  const byDate = {};
  for (const r of rows) {
    if (!byDate[r.date]) byDate[r.date] = {};
    byDate[r.date][r.time_block] = r[valueKey] ?? null;
  }
  const blocks = buildTimeAxis();
  return {
    backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
    grid: { top: 40, right: 20, bottom: 50, left: 70 },
    xAxis: { type: 'category', data: blocks, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 }, name: 'Block' },
    yAxis: { type: 'value', ...AXIS, name: yName || '', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
    series: days.map((d, i) => ({
      name: `${d} ${label || ''}`,
      type: 'line', data: Array.from({ length: 96 }, (_, b) => byDate[d]?.[b + 1] ?? null),
      color: COLORS[i % COLORS.length], symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: false,
    })),
  };
}

function buildDualAxisOption(loadRows, wxRows, wxVarKey, wxLabel, days) {
  const loadByDate = {};
  for (const r of loadRows) {
    if (!loadByDate[r.date]) loadByDate[r.date] = {};
    loadByDate[r.date][r.time_block] = r.actual ?? null;
  }
  const wxByDate = {};
  for (const r of wxRows) {
    if (!wxByDate[r.date]) wxByDate[r.date] = {};
    if (r[wxVarKey] != null) wxByDate[r.date][r.time_block] = r[wxVarKey];
  }
  const blocks = buildTimeAxis();
  const series = [];
  days.forEach((d, i) => {
    const col = COLORS[i % COLORS.length];
    series.push({
      name: `${d} Load`, type: 'line', yAxisIndex: 0,
      data: Array.from({ length: 96 }, (_, b) => loadByDate[d]?.[b + 1] ?? null),
      color: col, symbol: 'none', lineStyle: { width: 1.5 }, connectNulls: false,
    });
    series.push({
      name: `${d} ${wxLabel}`, type: 'line', yAxisIndex: 1,
      data: Array.from({ length: 96 }, (_, b) => wxByDate[d]?.[b + 1] ?? null),
      color: col, symbol: 'none', lineStyle: { width: 1.5, type: 'dashed' }, connectNulls: false,
    });
  });
  return {
    backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
    grid: { top: 40, right: 70, bottom: 50, left: 70 },
    xAxis: { type: 'category', data: blocks, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 } },
    yAxis: [
      { type: 'value', ...AXIS, name: 'Load MW', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
      { type: 'value', ...AXIS, name: wxLabel, nameTextStyle: { color: '#B8BDCC', fontSize: 11 }, position: 'right', splitLine: { show: false } },
    ],
    series,
  };
}

// ── Main component ───────────────────────────────────────────────────────────

export default function Dashboard3Combined({ state }) {
  const [fromDate, setFromDate] = useState(daysAgo(14));
  const [toDate, setToDate]     = useState(today());
  const [wxRows,  setWxRows]    = useState([]);
  const [loadRows, setLoadRows] = useState([]);
  const [wxLocRows, setWxLocRows] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  // Overlay days (sections C-F use a shared accumulator)
  const [overlayDays, setOverlayDays] = useState([]);

  // Variable selectors
  const [wxVars,    setWxVars]    = useState(['temperature_2m']);
  const [wxVarDual, setWxVarDual] = useState('temperature_2m');

  // Location (section F)
  const locations = useMemo(() => {
    const s = new Set(wxLocRows.map((r) => r.location).filter(Boolean));
    return [...s].sort();
  }, [wxLocRows]);
  const [selLoc, setSelLoc]       = useState('');
  const [locWxVar, setLocWxVar]   = useState('temperature_2m');

  const locRows = useMemo(() => wxLocRows.filter((r) => r.location === selLoc), [wxLocRows, selLoc]);
  const locVars = useMemo(() => WX_VARS.filter((v) => locRows.some((r) => r[v.key] != null)), [locRows]);

  const availVars = useMemo(() => WX_VARS.filter((v) => wxRows.some((r) => r[v.key] != null)), [wxRows]);

  const load = useCallback(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetchPipelineWeatherRange(state, fromDate, toDate),
      fetchPipelineSldcRange(state, fromDate, toDate),
      fetchPipelineWeatherLocRange(state, fromDate, toDate),
    ]).then(([wx, ld, wxloc]) => {
      setWxRows(Array.isArray(wx) ? wx : []);
      setLoadRows(Array.isArray(ld) ? ld : []);
      setWxLocRows(Array.isArray(wxloc) ? wxloc : []);
      setLoading(false);
    }).catch((e) => { setError(e.message || 'Fetch error'); setLoading(false); });
  }, [state, fromDate, toDate]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (locations.length && !selLoc) setSelLoc(locations[0]); }, [locations, selLoc]);

  const hasData = wxRows.length > 0 || loadRows.length > 0;

  const addOverlay = (d) => setOverlayDays((p) => [...p, d].slice(-8));
  const removeOverlay = (d) => setOverlayDays((p) => p.filter((x) => x !== d));

  // Memoised ECharts options
  const wxTsOpt   = useMemo(() => buildTsOption(wxRows, wxVars.filter((k) => availVars.some((v) => v.key === k)), `${fromDate} → ${toDate}`), [wxRows, wxVars, availVars, fromDate, toDate]);
  const ldTsOpt   = useMemo(() => buildLoadTsOption(loadRows), [loadRows]);
  const ldOverOpt = useMemo(() => overlayDays.length ? buildBlockOverlayOption(loadRows, overlayDays, 'actual', 'MW', 'Load') : null, [loadRows, overlayDays]);
  const wxOverOpt = useMemo(() => overlayDays.length ? buildBlockOverlayOption(wxRows, overlayDays, wxVarDual, WX_VARS.find((w) => w.key === wxVarDual)?.label, '') : null, [wxRows, overlayDays, wxVarDual]);
  const dualMeanOpt = useMemo(() => overlayDays.length ? buildDualAxisOption(loadRows, wxRows, wxVarDual, WX_VARS.find((w) => w.key === wxVarDual)?.label || wxVarDual, overlayDays) : null, [loadRows, wxRows, wxVarDual, overlayDays]);
  const dualLocOpt  = useMemo(() => overlayDays.length && selLoc ? buildDualAxisOption(loadRows, locRows, locWxVar, `${selLoc} – ${WX_VARS.find((w) => w.key === locWxVar)?.label || locWxVar}`, overlayDays) : null, [loadRows, locRows, locWxVar, selLoc, overlayDays]);

  return (
    <div className="p-6" style={{ maxWidth: 1200, margin: '0 auto', width: '100%', overflowX: 'hidden' }}>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <div><Label>From</Label><Inp type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} /></div>
        <div><Label>To</Label><Inp type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} /></div>
        <button onClick={load} className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-lg"
          style={{ background: '#F07825', color: '#fff', border: 'none', cursor: 'pointer', height: 34 }}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />} Load
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>

      {loading && <div className="flex items-center justify-center py-20"><Loader2 className="h-8 w-8 animate-spin" style={{ color: '#F07825' }} /></div>}

      {!loading && hasData && (
        <>
          {/* Section A — Weather time-series */}
          <Card section="A" title="Weather Time-Series" extra={
            <div className="flex flex-wrap gap-1.5">
              {availVars.map(({ key, label }, i) => (
                <Chip key={key} active={wxVars.includes(key)} color={COLORS[i % COLORS.length]}
                  onClick={() => setWxVars((p) => p.includes(key) ? p.filter((k) => k !== key) : [...p, key])}>
                  {label}
                </Chip>
              ))}
            </div>
          }>
            {wxVars.length ? <ReactECharts option={wxTsOpt} style={{ height: 300 }} theme="dark" /> : <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>Select variables above</p>}
          </Card>

          {/* Section B — Load time-series */}
          <Card section="B" title="Actual Load Time-Series (same date range)">
            <ReactECharts option={ldTsOpt} style={{ height: 280 }} theme="dark" />
          </Card>

          {/* Shared date accumulator */}
          <div style={{ background: 'rgba(91,159,228,0.06)', border: '1px solid rgba(91,159,228,0.15)', borderRadius: 10, padding: '14px 18px', marginBottom: 20 }}>
            <p className="text-xs font-semibold mb-3" style={{ color: '#5B9FE4', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Shared Date Picker for Sections C – F (accumulates up to 8 days)
            </p>
            <AddDateBar overlayDates={overlayDays} onAdd={addOverlay} onRemove={removeOverlay} onClear={() => setOverlayDays([])} />
          </div>

          {/* Section C — Load overlay */}
          <Card section="C" title="Load Overlay — by block 1–96">
            {overlayDays.length === 0
              ? <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>Add dates in the shared picker above</p>
              : <ReactECharts option={ldOverOpt} style={{ height: 280 }} theme="dark" />}
          </Card>

          {/* Section D — Weather overlay */}
          <Card section="D" title="Weather Overlay — by block 1–96" extra={
            <select value={wxVarDual} onChange={(e) => setWxVarDual(e.target.value)}
              style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '4px 8px', fontSize: 12, cursor: 'pointer' }}>
              {availVars.map(({ key, label }) => <option key={key} value={key} style={{ background: '#1a1922' }}>{label}</option>)}
            </select>
          }>
            {overlayDays.length === 0
              ? <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>Add dates in the shared picker above</p>
              : <ReactECharts option={wxOverOpt} style={{ height: 280 }} theme="dark" />}
          </Card>

          {/* Section E — Dual-axis: Load + state-mean weather */}
          <Card section="E" title={`Dual-Axis Overlay — Load + State Mean ${WX_VARS.find((w) => w.key === wxVarDual)?.label || wxVarDual}`}>
            {overlayDays.length === 0
              ? <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>Add dates in the shared picker above</p>
              : dualMeanOpt
                ? <ReactECharts option={dualMeanOpt} style={{ height: 300 }} theme="dark" />
                : <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>No matching data for selected dates</p>}
          </Card>

          {/* Section F — Dual-axis: Load + location weather */}
          <Card section="F" title="Dual-Axis Overlay — Load + Location Weather" extra={
            <div className="flex items-center gap-3">
              {locations.length > 0 && (
                <select value={selLoc} onChange={(e) => setSelLoc(e.target.value)}
                  style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '4px 8px', fontSize: 12, cursor: 'pointer' }}>
                  {locations.map((l) => <option key={l} value={l} style={{ background: '#1a1922' }}>{l}</option>)}
                </select>
              )}
              {locVars.length > 0 && (
                <select value={locWxVar} onChange={(e) => setLocWxVar(e.target.value)}
                  style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '4px 8px', fontSize: 12, cursor: 'pointer' }}>
                  {locVars.map(({ key, label }) => <option key={key} value={key} style={{ background: '#1a1922' }}>{label}</option>)}
                </select>
              )}
            </div>
          }>
            {wxLocRows.length === 0
              ? <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>No location weather data for {state}</p>
              : overlayDays.length === 0
                ? <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>Add dates in the shared picker above</p>
                : dualLocOpt
                  ? <ReactECharts option={dualLocOpt} style={{ height: 300 }} theme="dark" />
                  : <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>No matching data for selected location/dates</p>}
          </Card>
        </>
      )}

      {!loading && !hasData && !error && (
        <div className="flex flex-col items-center justify-center py-24" style={{ color: 'rgba(255,255,255,0.3)' }}>
          <Layers className="h-12 w-12 mb-4" style={{ opacity: 0.3 }} />
          <p className="text-sm">No data for {state} in selected range</p>
        </div>
      )}
    </div>
  );
}
