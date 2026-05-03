import React, { useState, useEffect, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2 } from 'lucide-react';
import { DB_STATES, fetchPipelineCircleImpact, fetchPipelineWeatherLoc } from './pipelineApi';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
} from '../../components/page/PagePrimitives.jsx';

const COLORS = [
  '#F07825',
  '#5B9FE4',
  '#34D399',
  '#F87171',
  '#FBBF24',
  '#C084FC',
  '#45b7d1',
  '#F472B6',
];

const VAR_OPTIONS = [
  { value: 'temperature_2m', label: 'Temperature (°C)' },
  { value: 'apparent_temperature', label: 'Apparent Temp (°C)' },
  { value: 'relative_humidity_2m', label: 'Humidity (%)' },
  { value: 'precipitation', label: 'Precipitation (mm)' },
];

const DARK_TOOLTIP = {
  trigger: 'axis',
  backgroundColor: 'rgba(20,19,26,0.95)',
  borderColor: 'rgba(255,255,255,0.1)',
  borderWidth: 1,
  textStyle: { color: '#ECEEF3', fontSize: 12 },
};
const DARK_LEGEND = {
  type: 'scroll',
  bottom: 0,
  icon: 'roundRect',
  itemWidth: 14,
  itemHeight: 8,
  textStyle: { fontSize: 11, color: '#A0A5B8' },
};
const DARK_AXIS = {
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
  axisTick: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
  axisLabel: { color: '#A0A5B8', fontSize: 11 },
  splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
};

const S = {
  page: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
    background: 'var(--bg)',
  },
  topBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 16px',
    flexShrink: 0,
    borderBottom: '1px solid var(--outline)',
    background: 'var(--bg-elevated)',
    flexWrap: 'wrap',
  },
  label: { fontSize: 11, color: 'var(--text-secondary)', whiteSpace: 'nowrap' },
  select: {
    background: 'var(--bg-surface)',
    color: 'var(--text)',
    border: '1px solid var(--outline)',
    borderRadius: 6,
    padding: '4px 8px',
    fontSize: 12,
  },
  body: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    padding: 12,
    gap: 10,
  },
  card: {
    background: 'var(--bg-elevated)',
    borderRadius: 10,
    border: '1px solid var(--outline)',
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    overflow: 'hidden',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '10px 14px',
    borderBottom: '1px solid var(--outline)',
    flexShrink: 0,
  },
  cardTitle: { fontSize: 13, fontWeight: 600, color: 'var(--text)' },
  tabRow: {
    display: 'flex',
    gap: 0,
    borderBottom: '1px solid var(--outline)',
    background: 'var(--bg-elevated)',
    padding: '0 12px',
    flexShrink: 0,
  },
  tab: (active) => ({
    padding: '7px 14px',
    fontSize: 12,
    fontWeight: active ? 700 : 500,
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
  }),
  metricsRow: {
    display: 'flex',
    gap: 8,
    flexShrink: 0,
  },
  metricCard: (color) => ({
    flex: 1,
    background: 'var(--bg-elevated)',
    borderRadius: 8,
    padding: '8px 12px',
    borderTop: `2px solid ${color}`,
    border: '1px solid var(--outline)',
  }),
  metricLabel: { fontSize: 10, color: 'var(--text-secondary)', marginBottom: 2 },
  metricValue: { fontSize: 18, fontWeight: 700, color: 'var(--text)' },
  empty: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'var(--text-secondary)',
    fontSize: 13,
  },
  locList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    padding: '8px 14px',
    flexShrink: 0,
    borderBottom: '1px solid var(--outline)',
  },
  locChip: (active) => ({
    padding: '3px 10px',
    borderRadius: 20,
    fontSize: 11,
    cursor: 'pointer',
    userSelect: 'none',
    background: active ? 'var(--accent)' : 'var(--bg-surface)',
    color: active ? '#fff' : 'var(--text-secondary)',
    border: `1px solid ${active ? 'var(--accent)' : 'var(--outline)'}`,
    fontWeight: active ? 600 : 400,
  }),
};

export default function WeatherLocPage({ initialState = 'HARYANA', hideStateSelector = false }) {
  const [dbState, setDbState] = useState(initialState);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [circleImpact, setCircleImpact] = useState(null);
  const [circleLoading, setCircleLoading] = useState(false);
  const [circleError, setCircleError] = useState(null);

  const [selectedVar, setSelectedVar] = useState('temperature_2m');
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedLocs, setSelectedLocs] = useState([]);
  const [activeTab, setActiveTab] = useState('overlay');

  useEffect(() => {
    if (!initialState) return;
    setDbState(initialState);
  }, [initialState]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData([]);
    setSelectedLocs([]);
    fetchPipelineWeatherLoc(dbState)
      .then((d) => {
        setData(d);
        const locs = [...new Set(d.map((r) => r.location).filter(Boolean))];
        setSelectedLocs(locs.slice(0, 4));
        const dates = [...new Set(d.map((r) => r.date).filter(Boolean))].sort();
        if (dates.length) setSelectedDate(dates[dates.length - 1]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [dbState]);

  useEffect(() => {
    if (String(dbState || '').toUpperCase() !== 'HARYANA') {
      setCircleImpact(null);
      setCircleError(null);
      setCircleLoading(false);
      return;
    }
    setCircleLoading(true);
    setCircleError(null);
    fetchPipelineCircleImpact(dbState, 365)
      .then((d) => setCircleImpact(d))
      .catch((e) => setCircleError(e.message))
      .finally(() => setCircleLoading(false));
  }, [dbState]);

  const locations = useMemo(
    () => [...new Set(data.map((r) => r.location).filter(Boolean))],
    [data]
  );
  const dates = useMemo(() => [...new Set(data.map((r) => r.date).filter(Boolean))].sort(), [data]);

  const toggleLoc = (loc) => {
    setSelectedLocs((prev) =>
      prev.includes(loc) ? prev.filter((l) => l !== loc) : [...prev, loc]
    );
  };

  // Overlay chart: selected locations on same selected date, 96 blocks x-axis
  const overlayOption = useMemo(() => {
    if (!selectedDate || !selectedLocs.length) return null;
    const series = selectedLocs.map((loc, i) => {
      const dd = data
        .filter((r) => r.date === selectedDate && r.location === loc)
        .sort((a, b) => (a.time_block || 0) - (b.time_block || 0));
      return {
        name: loc,
        type: 'line',
        smooth: true,
        symbolSize: 0,
        data: dd.map((r) => (r[selectedVar] !== undefined ? Number(r[selectedVar]) : null)),
        lineStyle: { width: 2, color: COLORS[i % COLORS.length] },
        itemStyle: { color: COLORS[i % COLORS.length] },
      };
    });
    const maxLen = Math.max(
      ...selectedLocs.map(
        (loc) => data.filter((r) => r.date === selectedDate && r.location === loc).length
      ),
      1
    );
    const varLabel = VAR_OPTIONS.find((o) => o.value === selectedVar)?.label || selectedVar;
    return {
      backgroundColor: 'transparent',
      color: COLORS,
      tooltip: DARK_TOOLTIP,
      legend: DARK_LEGEND,
      grid: { top: 46, right: 24, bottom: 64, left: 60, containLabel: true },
      xAxis: {
        type: 'category',
        data: Array.from({ length: maxLen }, (_, i) => i + 1),
        boundaryGap: false,
        name: 'Block',
        ...DARK_AXIS,
      },
      yAxis: { type: 'value', name: varLabel, nameTextStyle: { color: '#A0A5B8' }, ...DARK_AXIS },
      series,
    };
  }, [selectedDate, selectedLocs, selectedVar, data]);

  // Time-series chart: selected locations across all dates
  const tsOption = useMemo(() => {
    if (!selectedLocs.length) return null;
    const series = selectedLocs.map((loc, i) => {
      const dd = data
        .filter((r) => r.location === loc)
        .sort((a, b) => {
          if (a.date !== b.date) return a.date.localeCompare(b.date);
          return (a.time_block || 0) - (b.time_block || 0);
        });
      return {
        name: loc,
        type: 'line',
        smooth: true,
        symbolSize: 0,
        data: dd.map((r) => (r[selectedVar] !== undefined ? Number(r[selectedVar]) : null)),
        lineStyle: { width: 1.5, color: COLORS[i % COLORS.length] },
        itemStyle: { color: COLORS[i % COLORS.length] },
      };
    });
    const allDd = data
      .filter((r) => selectedLocs.includes(r.location))
      .sort((a, b) => {
        if (a.date !== b.date) return a.date.localeCompare(b.date);
        return (a.time_block || 0) - (b.time_block || 0);
      });
    const xData = allDd
      .filter((r) => r.location === selectedLocs[0])
      .map((r) => `${r.date} B${r.time_block || 1}`);
    const varLabel = VAR_OPTIONS.find((o) => o.value === selectedVar)?.label || selectedVar;
    return {
      backgroundColor: 'transparent',
      color: COLORS,
      tooltip: DARK_TOOLTIP,
      legend: DARK_LEGEND,
      grid: { top: 46, right: 24, bottom: 72, left: 60, containLabel: true },
      xAxis: {
        type: 'category',
        data: xData,
        boundaryGap: false,
        axisLabel: {
          rotate: 30,
          fontSize: 9,
          interval: Math.floor(xData.length / 10),
          color: '#A0A5B8',
        },
        axisLine: DARK_AXIS.axisLine,
        axisTick: DARK_AXIS.axisTick,
      },
      yAxis: { type: 'value', name: varLabel, nameTextStyle: { color: '#A0A5B8' }, ...DARK_AXIS },
      dataZoom: [
        { type: 'inside' },
        {
          type: 'slider',
          height: 18,
          bottom: 28,
          borderColor: 'rgba(255,255,255,0.1)',
          fillerColor: 'rgba(240,120,37,0.15)',
        },
      ],
      series,
    };
  }, [selectedLocs, selectedVar, data]);

  const avgByLoc = useMemo(() => {
    if (!data.length || !selectedDate) return [];
    return locations
      .map((loc) => {
        const vals = data
          .filter(
            (r) => r.date === selectedDate && r.location === loc && r[selectedVar] !== undefined
          )
          .map((r) => Number(r[selectedVar]))
          .filter((x) => !isNaN(x));
        const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        return { loc, avg: avg !== null ? avg.toFixed(1) : '—' };
      })
      .filter((r) => r.avg !== '—');
  }, [data, selectedDate, selectedVar, locations]);

  return (
    <VpPageShell className="weather-loc-page">
      {/* Top control bar */}
      <div style={S.topBar}>
        {!hideStateSelector && (
          <>
            <span style={S.label}>State:</span>
            <select value={dbState} onChange={(e) => setDbState(e.target.value)} style={S.select}>
              {DB_STATES.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0) + s.slice(1).toLowerCase()}
                </option>
              ))}
            </select>
          </>
        )}

        <span style={S.label}>Variable:</span>
        <select
          value={selectedVar}
          onChange={(e) => setSelectedVar(e.target.value)}
          style={S.select}
        >
          {VAR_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>

        {activeTab === 'overlay' && (
          <>
            <span style={S.label}>Date:</span>
            <select
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={S.select}
            >
              {dates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </>
        )}

        <span style={{ ...S.label, marginLeft: 8 }}>
          {locations.length} location{locations.length !== 1 ? 's' : ''} · {dates.length} days
        </span>
      </div>

      {/* Sub-tabs */}
      <div style={S.tabRow}>
        {[
          ['overlay', 'Block Overlay (single day)'],
          ['ts', 'Time Series (all days)'],
        ].map(([k, l]) => (
          <button key={k} onClick={() => setActiveTab(k)} style={S.tab(activeTab === k)}>
            {l}
          </button>
        ))}
      </div>

      {loading ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            gap: 8,
            color: 'var(--text-secondary)',
          }}
        >
          <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Loading location
          weather…
        </div>
      ) : error ? (
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#F87171',
            fontSize: 13,
          }}
        >
          {error}
        </div>
      ) : data.length === 0 ? (
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-secondary)',
            fontSize: 13,
          }}
        >
          No location weather data for {dbState}.
        </div>
      ) : (
        <div style={S.body}>
          {/* Metrics row */}
          {activeTab === 'overlay' && avgByLoc.length > 0 && (
            <div style={S.metricsRow}>
              {avgByLoc.slice(0, 6).map((r, i) => (
                <div key={r.loc} style={S.metricCard(COLORS[i % COLORS.length])}>
                  <div style={S.metricLabel}>{r.loc}</div>
                  <div style={S.metricValue}>{r.avg}</div>
                </div>
              ))}
            </div>
          )}

          {/* Chart card */}
          <div style={S.card}>
            <div style={S.cardHeader}>
              <span style={S.cardTitle}>
                {activeTab === 'overlay'
                  ? `Location Overlay — ${VAR_OPTIONS.find((o) => o.value === selectedVar)?.label} on ${selectedDate}`
                  : `Location Time-Series — ${VAR_OPTIONS.find((o) => o.value === selectedVar)?.label}`}
              </span>
            </div>

            {/* Location toggle chips */}
            {locations.length > 0 && (
              <div style={S.locList}>
                {locations.map((loc) => (
                  <span
                    key={loc}
                    onClick={() => toggleLoc(loc)}
                    style={S.locChip(selectedLocs.includes(loc))}
                  >
                    {loc}
                  </span>
                ))}
              </div>
            )}

            {selectedLocs.length === 0 ? (
              <div style={S.empty}>Select at least one location above.</div>
            ) : activeTab === 'overlay' && overlayOption ? (
              <ReactECharts
                option={overlayOption}
                style={{ flex: 1, minHeight: 0 }}
                notMerge
                lazyUpdate
              />
            ) : activeTab === 'ts' && tsOption ? (
              <ReactECharts
                option={tsOption}
                style={{ flex: 1, minHeight: 0 }}
                notMerge
                lazyUpdate
              />
            ) : (
              <div style={S.empty}>No data for selected options.</div>
            )}
          </div>

          {String(dbState || '').toUpperCase() === 'HARYANA' && (
            <div style={{ ...S.card, marginTop: 12 }}>
              <div style={S.cardHeader}>
                <span style={S.cardTitle}>Circle Impact on Total Load</span>
              </div>
              {circleLoading ? (
                <div style={S.empty}>Loading circle impact…</div>
              ) : circleError ? (
                <div style={{ ...S.empty, color: '#F87171' }}>{circleError}</div>
              ) : !circleImpact?.available ? (
                <div style={S.empty}>No circle_load table found for Haryana.</div>
              ) : (circleImpact?.circles || []).length === 0 ? (
                <div style={S.empty}>No circle impact rows in the selected window.</div>
              ) : (
                <div style={{ padding: '10px 12px 14px', overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ color: '#A0A5B8', textAlign: 'left' }}>
                        <th style={{ padding: '8px 6px' }}>Circle</th>
                        <th style={{ padding: '8px 6px' }}>Share %</th>
                        <th style={{ padding: '8px 6px' }}>Corr</th>
                        <th style={{ padding: '8px 6px' }}>R²</th>
                        <th style={{ padding: '8px 6px' }}>β</th>
                        <th style={{ padding: '8px 6px' }}>Days</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(circleImpact.circles || []).slice(0, 12).map((r) => (
                        <tr key={r.circle} style={{ borderTop: '1px solid #2A292F' }}>
                          <td style={{ padding: '8px 6px', color: '#ECEEF3' }}>{r.circle}</td>
                          <td style={{ padding: '8px 6px' }}>{r.avg_share_pct}</td>
                          <td style={{ padding: '8px 6px' }}>{r.corr_daily_mean}</td>
                          <td style={{ padding: '8px 6px' }}>{r.r2}</td>
                          <td style={{ padding: '8px 6px' }}>{r.beta_mw_per_mw}</td>
                          <td style={{ padding: '8px 6px' }}>{r.n_days}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </VpPageShell>
  );
}
