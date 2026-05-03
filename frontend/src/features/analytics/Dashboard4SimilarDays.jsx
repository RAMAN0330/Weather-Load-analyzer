import React, { useState, useEffect, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2, Search, Calendar } from 'lucide-react';
import {
  fetchPipelineSimilarity,
  fetchPipelineLoadRange,
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

function today() { return new Date().toISOString().slice(0, 10); }
function daysAgo(n) { const d = new Date(); d.setDate(d.getDate() - n); return d.toISOString().slice(0, 10); }

function Card({ title, children }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20, marginBottom: 20 }}>
      <div className="mb-4">
        <span className="font-semibold text-sm" style={{ color: '#F0F2F8' }}>{title}</span>
      </div>
      {children}
    </div>
  );
}
function Inp({ ...p }) {
  return <input {...p} style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '6px 10px', fontSize: 13, ...p.style }} />;
}
function Label({ children }) {
  return <label className="text-xs block mb-1" style={{ color: 'rgba(255,255,255,0.45)' }}>{children}</label>;
}
function Tab({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{ padding: '7px 18px', borderRadius: 0, fontSize: 13, cursor: 'pointer', border: 'none', background: 'transparent', color: active ? '#F0F2F8' : 'rgba(255,255,255,0.4)', borderBottom: active ? '2px solid #C084FC' : '2px solid transparent', fontWeight: active ? 600 : 400 }}>
      {children}
    </button>
  );
}

const WX_LABELS = { temp: 'Temperature (°C)', humidity: 'Humidity (%)', precip: 'Precipitation (mm)' };
const WX_COLORS = { temp: '#F07825', humidity: '#5B9FE4', precip: '#34D399' };

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
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              {['Rank','Date','Euclidean Distance','Day of Week','Category'].map((h) => (
                <th key={h} style={{ padding: '6px 12px', color: 'rgba(255,255,255,0.45)', fontWeight: 500, textAlign: h === 'Rank' || h === 'Euclidean Distance' ? 'right' : 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.07)', background: 'rgba(240,120,37,0.07)' }}>
              <td style={{ padding: '7px 12px', textAlign: 'right', color: '#F07825', fontWeight: 600 }}>Target</td>
              <td style={{ padding: '7px 12px', color: '#F07825', fontWeight: 600 }}>{result.target_date}</td>
              <td colSpan={3} style={{ padding: '7px 12px', color: 'rgba(255,255,255,0.3)', fontSize: 11 }}>Target day</td>
            </tr>
            {result.similar.map((s, i) => {
              const dt = new Date(s.date);
              const dow = dt.toLocaleDateString('en-IN', { weekday: 'short' });
              return (
                <tr key={s.date} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: i % 2 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                  <td style={{ padding: '7px 12px', textAlign: 'right', color: '#C084FC', fontWeight: 600 }}>#{i + 1}</td>
                  <td style={{ padding: '7px 12px', color: '#F0F2F8' }}>{s.date}</td>
                  <td style={{ padding: '7px 12px', textAlign: 'right', color: '#B8BDCC', fontFamily: 'monospace' }}>{s.distance?.toFixed(2) ?? '—'}</td>
                  <td style={{ padding: '7px 12px', color: '#B8BDCC' }}>{dow}</td>
                  <td style={{ padding: '7px 12px' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, background: dt.getDay() === 0 ? 'rgba(251,191,36,0.15)' : 'rgba(52,211,153,0.12)', color: dt.getDay() === 0 ? '#FBBF24' : '#34D399' }}>
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
      yAxis: { type: 'value', ...AXIS, name: WX_LABELS[wxVar] || wxVar, nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
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
  }, [result, wxVar, blockAxis]);

  // ── Load profile chart ────────────────────────────────────────────────────

  const loadOverlayOption = useMemo(() => {
    if (!result) return {};
    const allDates = [result.target_date, ...(result.similar || []).map((s) => s.date)];
    return {
      backgroundColor: 'transparent', tooltip: TOOLTIP, legend: LEGEND,
      grid: { top: 40, right: 20, bottom: 50, left: 70 },
      xAxis: { type: 'category', data: blockAxis, ...AXIS, axisLabel: { ...AXIS.axisLabel, interval: 11, rotate: 30 }, name: 'Block' },
      yAxis: { type: 'value', ...AXIS, name: 'MW', nameTextStyle: { color: '#B8BDCC', fontSize: 11 } },
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
  }, [result, loadProfiles, blockAxis]);

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
              style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#F0F2F8', padding: '6px 10px', fontSize: 13, cursor: 'pointer' }}>
              {[3,5,7,10].map((n) => <option key={n} value={n} style={{ background: '#1a1922' }}>{n}</option>)}
            </select>
          </div>
          <button onClick={findSimilar}
            className="flex items-center gap-1.5 text-sm font-semibold px-5 py-2 rounded-lg"
            style={{ background: '#C084FC', color: '#fff', border: 'none', cursor: 'pointer', height: 36, boxShadow: '0 4px 14px rgba(192,132,252,0.25)' }}
            disabled={loading}
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Find Similar Days
          </button>
          {error && <span className="text-xs text-red-400">{error}</span>}
        </div>

        {!result && !loading && (
          <div className="flex flex-col items-center py-14" style={{ color: 'rgba(255,255,255,0.25)' }}>
            <Calendar className="h-10 w-10 mb-3" style={{ opacity: 0.4 }} />
            <p className="text-sm">Pick a target date and click "Find Similar Days"</p>
            <p className="text-xs mt-1" style={{ color: 'rgba(255,255,255,0.15)' }}>
              Computes Euclidean distance over 96-block weather vectors (temperature, humidity, precipitation)
            </p>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin" style={{ color: '#C084FC' }} />
          </div>
        )}
      </Card>

      {/* Results */}
      {result && !loading && (
        <>
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
              Top {result.similar?.length} days similar to <span style={{ color: '#F07825', fontWeight: 600 }}>{result.target_date}</span> — ranked by Euclidean distance over weather profile
            </p>
          </div>

          {/* Sub-tabs */}
          <div style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', marginBottom: 20 }}>
            <Tab active={activeTab === 'table'}   onClick={() => setActiveTab('table')}>Ranked Similar Days</Tab>
            <Tab active={activeTab === 'weather'} onClick={() => setActiveTab('weather')}>Weather Overlay</Tab>
            <Tab active={activeTab === 'load'}    onClick={() => setActiveTab('load')}>Load Profile</Tab>
          </div>

          {activeTab === 'table' && (
            <Card title="Similar Days — Ranked by Euclidean Distance">
              {RankedTable || <p className="text-sm text-center py-6" style={{ color: 'rgba(255,255,255,0.3)' }}>No similar days found</p>}
            </Card>
          )}

          {activeTab === 'weather' && (
            <Card title="Weather Profile Comparison — 96-block overlay">
              <div className="flex gap-3 mb-4">
                {Object.entries(WX_LABELS).map(([k, label]) => (
                  <button key={k} onClick={() => setWxVar(k)}
                    style={{ padding: '4px 12px', borderRadius: 20, fontSize: 12, cursor: 'pointer', border: 'none',
                      background: wxVar === k ? WX_COLORS[k] : 'rgba(255,255,255,0.07)',
                      color: wxVar === k ? '#fff' : 'rgba(255,255,255,0.5)' }}>
                    {label}
                  </button>
                ))}
              </div>
              <ReactECharts option={wxOverlayOption} style={{ height: 320 }} theme="dark" />
              <p className="text-xs mt-2" style={{ color: 'rgba(255,255,255,0.25)' }}>
                Solid line = target day · Dashed = similar days
              </p>
            </Card>
          )}

          {activeTab === 'load' && (
            <Card title="Load Profile Comparison — 96-block overlay">
              {Object.keys(loadProfiles).length === 0
                ? <p className="text-sm text-center py-8" style={{ color: 'rgba(255,255,255,0.3)' }}>No load data found for matched dates</p>
                : <>
                    <ReactECharts option={loadOverlayOption} style={{ height: 320 }} theme="dark" />
                    <p className="text-xs mt-2" style={{ color: 'rgba(255,255,255,0.25)' }}>
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
