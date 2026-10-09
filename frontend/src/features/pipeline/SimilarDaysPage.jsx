import React, { useState, useEffect, useMemo, useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import { Loader2 } from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
} from '../../components/page/PagePrimitives.jsx';
import {
  DB_STATES,
  fetchPipelineWeather,
  fetchPipelineLoad,
  fetchPipelineSimilarity,
  buildTimeAxis,
} from './pipelineApi';
import HorizonToggle from '../../components/HorizonToggle';
import { useChartTokens, withAlpha } from '../../lib/chartTheme';

// Chart theme — resolved from CSS tokens (see docs/theming-guide.md).
function chartTheme(tk) {
  return {
    COLORS: [tk.warm, tk.accent, tk.success, tk.danger, tk.warning, tk.accent2, tk.info, tk.textSecondary],
    TARGET_COLOR: tk.danger,
    DARK_TOOLTIP: {
      trigger: 'axis',
      backgroundColor: tk.elevated,
      borderColor: tk.outline,
      borderWidth: 1,
      textStyle: { color: tk.text, fontSize: 13 },
    },
    DARK_LEGEND: {
      type: 'scroll',
      bottom: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 8,
      textStyle: { fontSize: 12, color: tk.textSecondary },
    },
    DARK_AXIS: {
      axisLine: { lineStyle: { color: tk.outline } },
      axisTick: { lineStyle: { color: tk.outline } },
      axisLabel: { color: tk.textSecondary, fontSize: 12 },
      splitLine: { lineStyle: { color: withAlpha(tk.outline, 0.6), type: 'dashed' } },
    },
  };
}

const DAY_BADGE_COLOR = { working: 'var(--success)', sunday: 'var(--warning)', holiday: 'var(--danger)' };

function fmt(v, d = 3) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

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
  label: { fontSize: 13, color: 'var(--text-secondary)', whiteSpace: 'nowrap' },
  input: {
    background: 'var(--bg-surface)',
    color: 'var(--text)',
    border: '1px solid var(--outline)',
    borderRadius: 6,
    padding: '4px 8px',
    fontSize: 13,
  },
  btn: {
    background: 'var(--accent)',
    color: 'var(--accent-fg)',
    border: 'none',
    borderRadius: 6,
    padding: '5px 14px',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  },
  btnDisabled: { opacity: 0.5, cursor: 'not-allowed' },
  body: { display: 'flex', flex: 1, overflow: 'hidden', gap: 0 },
  left: {
    width: 320,
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    borderRight: '1px solid var(--outline)',
  },
  right: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  panelTitle: {
    padding: '8px 12px',
    fontSize: 12,
    fontWeight: 700,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    borderBottom: '1px solid var(--outline)',
    background: 'var(--bg-elevated)',
    flexShrink: 0,
  },
  tableWrap: { flex: 1, overflow: 'auto', padding: '8px 12px' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: {
    padding: '6px 8px',
    textAlign: 'left',
    borderBottom: '1px solid var(--outline)',
    color: 'var(--text-secondary)',
    fontWeight: 600,
    fontSize: 12,
    whiteSpace: 'nowrap',
  },
  td: {
    padding: '5px 8px',
    borderBottom: '1px solid rgba(var(--overlay-rgb),0.04)',
    color: 'var(--text)',
    fontSize: 13,
  },
  badge: (cat) => ({
    display: 'inline-block',
    padding: '2px 7px',
    borderRadius: 10,
    fontSize: 11,
    fontWeight: 600,
    background: `color-mix(in srgb, ${DAY_BADGE_COLOR[cat] || 'var(--text-muted)'} 13%, transparent)`,
    color: DAY_BADGE_COLOR[cat] || 'var(--text-muted)',
    border: `1px solid color-mix(in srgb, ${DAY_BADGE_COLOR[cat] || 'var(--text-muted)'} 33%, transparent)`,
  }),
  tabBar: {
    display: 'flex',
    gap: 0,
    borderBottom: '1px solid var(--outline)',
    background: 'var(--bg-elevated)',
    padding: '0 12px',
    flexShrink: 0,
  },
  tab: (active) => ({
    padding: '8px 14px',
    fontSize: 13,
    fontWeight: active ? 700 : 500,
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
    transition: 'all 0.15s',
  }),
  chartArea: { flex: 1, overflow: 'hidden', padding: 12, display: 'flex', flexDirection: 'column' },
  chartCard: {
    flex: 1,
    background: 'var(--bg-elevated)',
    borderRadius: 10,
    border: '1px solid var(--outline)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  chartTitle: {
    padding: '10px 14px',
    fontSize: 13,
    fontWeight: 600,
    color: 'var(--text)',
    borderBottom: '1px solid var(--outline)',
    flexShrink: 0,
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    color: 'var(--text-secondary)',
    fontSize: 13,
  },
  metricsRow: {
    display: 'flex',
    gap: 8,
    padding: '8px 12px',
    flexShrink: 0,
    borderBottom: '1px solid var(--outline)',
    background: 'var(--bg-elevated)',
  },
  metricCard: (color) => ({
    flex: 1,
    background: 'var(--bg-surface)',
    borderRadius: 8,
    padding: '8px 10px',
    borderTop: `2px solid ${color}`,
    minWidth: 0,
  }),
  metricLabel: { fontSize: 11, color: 'var(--text-secondary)', marginBottom: 2 },
  metricValue: {
    fontSize: 16,
    fontWeight: 700,
    color: 'var(--text)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
};

const SIMILAR_ROW_H = 36;
const LOAD_ROW_H = 32;

function VirtualTable({ rows, columns, rowHeight, maxHeight, renderRow, keyFn }) {
  const parentRef = useRef(null);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => rowHeight,
    overscan: 5,
  });
  const totalH = virtualizer.getTotalSize();
  const items = virtualizer.getVirtualItems();

  return (
    <div ref={parentRef} style={{ maxHeight, overflowY: 'auto', position: 'relative' }}>
      <div style={{ height: totalH, width: '100%', position: 'relative' }}>
        {items.map((vi) => (
          <div
            key={keyFn(rows[vi.index], vi.index)}
            data-index={vi.index}
            ref={virtualizer.measureElement}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${vi.start}px)`,
            }}
          >
            {renderRow(rows[vi.index], vi.index)}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SimilarDaysPage({ horizon = 't1', setHorizon, t2Date, t1Forecast = [] }) {
  const [dbState, setDbState] = useState('HARYANA');
  const [targetDate, setTargetDate] = useState('');
  const [method, setMethod] = useState('euclidean');
  const [topK, setTopK] = useState(5);

  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  const [weatherData, setWeatherData] = useState([]);
  const [loadData, setLoadData] = useState([]);
  const [dataLoading, setDataLoading] = useState(false);

  const [activeTab, setActiveTab] = useState('weather');
  const [selectedSimilar, setSelectedSimilar] = useState([]);

  // Auto-update targetDate when horizon switches to T+2
  useEffect(() => {
    if (horizon === 't2' && t2Date) setTargetDate(t2Date);
  }, [horizon, t2Date]);

  useEffect(() => {
    setDataLoading(true);
    setResult(null);
    setWeatherData([]);
    setLoadData([]);
    Promise.all([fetchPipelineWeather(dbState), fetchPipelineLoad(dbState)])
      .then(([wd, ld]) => {
        setWeatherData(wd);
        setLoadData(ld);
        if (wd.length > 0) {
          const dates = [...new Set(wd.map((r) => r.date))].sort();
          setTargetDate(dates[Math.floor(dates.length / 2)]);
        }
      })
      .catch(() => {})
      .finally(() => setDataLoading(false));
  }, [dbState]);

  const run = async () => {
    if (!targetDate) return;
    setRunning(true);
    setRunError(null);
    try {
      const res = await fetchPipelineSimilarity(dbState, targetDate, method, topK);
      setResult(res);
      setSelectedSimilar(res.similar.map((r) => r.date));
    } catch (e) {
      setRunError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const timeAxis = useMemo(() => buildTimeAxis(), []);
  const tk = useChartTokens();
  const { COLORS, TARGET_COLOR, DARK_TOOLTIP, DARK_LEGEND, DARK_AXIS } = useMemo(() => chartTheme(tk), [tk]);

  const weatherSubplotOption = useMemo(() => {
    if (!result || !selectedSimilar.length) return null;
    const allDates = [result.target_date, ...selectedSimilar];
    const profiles = result.profiles;
    const makeSeries = (key, xIdx, yIdx) =>
      allDates
        .filter((d) => profiles[d])
        .map((d, i) => {
          const isTarget = d === result.target_date;
          return {
            name: isTarget ? `${d} (target)` : d,
            type: 'line',
            smooth: true,
            symbolSize: 0,
            xAxisIndex: xIdx,
            yAxisIndex: yIdx,
            data: profiles[d][key] || [],
            lineStyle: {
              width: isTarget ? 3 : 1.5,
              color: isTarget ? TARGET_COLOR : COLORS[i % COLORS.length],
              type: isTarget ? 'solid' : 'dashed',
            },
            itemStyle: { color: isTarget ? TARGET_COLOR : COLORS[i % COLORS.length] },
          };
        });
    const maxLen = Math.max(
      ...allDates
        .filter((d) => profiles[d])
        .map((d) =>
          Math.max(
            (profiles[d].temp || []).length,
            (profiles[d].humidity || []).length,
            (profiles[d].precip || []).length
          )
        ),
      1
    );
    const xData = Array.from({ length: maxLen }, (_, i) => i + 1);
    const axisStyle = { ...DARK_AXIS };
    return {
      backgroundColor: 'transparent',
      color: COLORS,
      tooltip: DARK_TOOLTIP,
      legend: {
        ...DARK_LEGEND,
        data: allDates.map((d) => (d === result.target_date ? `${d} (target)` : d)),
      },
      grid: [
        { top: 24, bottom: '72%', left: 52, right: 16, containLabel: true },
        { top: '34%', bottom: '40%', left: 52, right: 16, containLabel: true },
        { top: '64%', bottom: '12%', left: 52, right: 16, containLabel: true },
      ],
      xAxis: [
        {
          type: 'category',
          data: xData,
          gridIndex: 0,
          boundaryGap: false,
          axisLabel: { show: false, ...axisStyle.axisLabel },
          axisLine: axisStyle.axisLine,
        },
        {
          type: 'category',
          data: xData,
          gridIndex: 1,
          boundaryGap: false,
          axisLabel: { show: false, ...axisStyle.axisLabel },
          axisLine: axisStyle.axisLine,
        },
        {
          type: 'category',
          data: xData,
          gridIndex: 2,
          boundaryGap: false,
          name: 'Block',
          ...axisStyle,
        },
      ],
      yAxis: [
        {
          type: 'value',
          name: 'Temp (°C)',
          gridIndex: 0,
          nameTextStyle: { fontSize: 12, color: tk.textSecondary },
          ...axisStyle,
        },
        {
          type: 'value',
          name: 'Humidity (%)',
          gridIndex: 1,
          nameTextStyle: { fontSize: 12, color: tk.textSecondary },
          ...axisStyle,
        },
        {
          type: 'value',
          name: 'Precip (mm)',
          gridIndex: 2,
          nameTextStyle: { fontSize: 12, color: tk.textSecondary },
          ...axisStyle,
        },
      ],
      series: [
        ...makeSeries('temp', 0, 0),
        ...makeSeries('humidity', 1, 1),
        ...makeSeries('precip', 2, 2),
      ],
    };
  }, [result, selectedSimilar, tk]);

  const loadOverlayOption = useMemo(() => {
    if (!result || !selectedSimilar.length) return null;
    const allDates = [result.target_date, ...selectedSimilar];
    const series = allDates.map((d, i) => {
      const isTarget = d === result.target_date;
      let data = [];

      if (isTarget && horizon === 't2' && t1Forecast?.length) {
        data = t1Forecast;
      } else {
        const dd = loadData
          .filter((r) => r.date === d)
          .sort((a, b) => (a.time_block || 0) - (b.time_block || 0));
        data = dd.map((r) => (r.load !== undefined ? Number(r.load) : null));
      }

      return {
        name: isTarget ? (horizon === 't2' ? `${d} (T+1 Forecast)` : `${d} (target)`) : d,
        type: 'line',
        smooth: true,
        symbolSize: 0,
        data,
        lineStyle: {
          width: isTarget ? 3 : 1.5,
          color: isTarget ? TARGET_COLOR : COLORS[i % COLORS.length],
          type: isTarget ? 'solid' : 'dashed',
        },
        itemStyle: { color: isTarget ? TARGET_COLOR : COLORS[i % COLORS.length] },
      };
    });
    const maxLen = Math.max(
      96,
      ...allDates.map((d) => loadData.filter((r) => r.date === d).length)
    );
    return {
      backgroundColor: 'transparent',
      color: COLORS,
      tooltip: DARK_TOOLTIP,
      legend: DARK_LEGEND,
      grid: { top: 46, right: 24, bottom: 64, left: 56, containLabel: true },
      xAxis: {
        type: 'category',
        data: Array.from({ length: maxLen }, (_, i) => i + 1),
        boundaryGap: false,
        name: 'Block',
        ...DARK_AXIS,
      },
      yAxis: {
        type: 'value',
        name: 'MW',
        nameTextStyle: { color: tk.textSecondary, fontSize: 12 },
        ...DARK_AXIS,
      },
      series,
    };
  }, [result, selectedSimilar, loadData, horizon, t1Forecast, tk]);

  const loadChangeTable = useMemo(() => {
    if (!result) return [];
    const getAvg = (d) => {
      if (d === result.target_date && horizon === 't2' && t1Forecast?.length) {
        return t1Forecast.reduce((a, b) => a + (Number(b) || 0), 0) / t1Forecast.length;
      }
      const vals = loadData
        .filter((r) => r.date === d)
        .map((r) => Number(r.load))
        .filter((x) => !isNaN(x));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const targetAvg = getAvg(result.target_date);
    return result.similar.map((r) => {
      const simAvg = getAvg(r.date);
      const pct =
        targetAvg && simAvg ? (((simAvg - targetAvg) / targetAvg) * 100).toFixed(1) : null;
      return { ...r, avg: simAvg ? simAvg.toFixed(1) : '—', pct };
    });
  }, [result, loadData, horizon, t1Forecast]);

  const uniqueWeatherDates = useMemo(
    () => [...new Set(weatherData.map((r) => r.date).filter(Boolean))],
    [weatherData]
  );
  const bestMatch = result?.similar?.[0];

  return (
    <VpPageShell className="similar-days-page">
      {/* Top control bar */}
      <div style={S.topBar}>
        <span style={S.label}>State:</span>
        <select value={dbState} onChange={(e) => setDbState(e.target.value)} style={S.input}>
          {DB_STATES.map((s) => (
            <option key={s} value={s}>
              {s.charAt(0) + s.slice(1).toLowerCase()}
            </option>
          ))}
        </select>
        <span style={{ ...S.label, marginLeft: 8 }}>Target Date:</span>
        <input
          type="date"
          value={targetDate}
          onChange={(e) => setTargetDate(e.target.value)}
          style={S.input}
        />
        <span style={S.label}>Method:</span>
        <select value={method} onChange={(e) => setMethod(e.target.value)} style={S.input}>
          <option value="euclidean">Euclidean (weather, standardised)</option>
          <option value="weighted">Weighted</option>
          <option value="cosine">Cosine</option>
        </select>
        <span style={S.label}>Top-K: {topK}</span>
        <input
          type="range"
          min={1}
          max={20}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          style={{ width: 80, accentColor: 'var(--accent)' }}
        />
        <button
          onClick={run}
          disabled={running || !targetDate || dataLoading}
          style={{ ...S.btn, ...(running || !targetDate || dataLoading ? S.btnDisabled : {}) }}
        >
          {running ? 'Searching…' : 'Find Similar Days'}
        </button>
        {runError && <span style={{ fontSize: 11, color: 'var(--danger)' }}>{runError}</span>}
        {setHorizon && (
          <HorizonToggle
            horizon={horizon}
            setHorizon={setHorizon}
            t2Date={t2Date}
            style={{ marginLeft: 'auto' }}
          />
        )}
      </div>

      {/* Metrics row */}
      <div style={S.metricsRow}>
        {[
          { label: 'Target Date', value: targetDate || '—', color: 'var(--accent)' },
          {
            label: 'Candidate Days',
            value: uniqueWeatherDates.length.toLocaleString(),
            color: 'var(--success)',
          },
          {
            label: 'Best Match',
            value: bestMatch ? bestMatch.date : 'Run search',
            color: 'var(--warning)',
          },
          {
            label: 'Best Distance',
            value: bestMatch ? fmt(bestMatch.distance) : '—',
            color: 'var(--accent2)',
          },
        ].map((m) => (
          <div key={m.label} style={S.metricCard(m.color)}>
            <div style={S.metricLabel}>{m.label}</div>
            <div style={S.metricValue}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Body: left table + right chart */}
      {dataLoading ? (
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
          <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Loading data…
        </div>
      ) : (
        <div style={S.body}>
          {/* Left — results table */}
          <div style={S.left}>
            <div style={S.panelTitle}>
              {result ? `Similar Days to ${result.target_date}` : 'Results'}
            </div>
            <div style={S.tableWrap}>
              {running && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    color: 'var(--text-secondary)',
                    padding: 16,
                  }}
                >
                  <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> Searching…
                </div>
              )}
              {!running && !result && (
                <div style={{ color: 'var(--text-secondary)', fontSize: 12, padding: 16 }}>
                  Run a search to see results.
                </div>
              )}
              {result && !running && (
                <>
                  <table style={S.table}>
                    <thead>
                      <tr>
                        {['#', 'Date', 'Day', 'Type', 'Dist', '✓'].map((h) => (
                          <th key={h} style={S.th}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.similar.map((r, i) => (
                        <tr key={r.date}>
                          <td style={S.td}>{i + 1}</td>
                          <td style={{ ...S.td, fontWeight: 600 }}>{r.date}</td>
                          <td style={S.td}>{r.day_of_week?.slice(0, 3)}</td>
                          <td style={S.td}>
                            <span style={S.badge(r.category)}>{r.category}</span>
                          </td>
                          <td style={S.td}>{fmt(r.distance)}</td>
                          <td style={S.td}>
                            <input
                              type="checkbox"
                              checked={selectedSimilar.includes(r.date)}
                              onChange={(e) => {
                                if (e.target.checked) setSelectedSimilar((p) => [...p, r.date]);
                                else setSelectedSimilar((p) => p.filter((x) => x !== r.date));
                              }}
                              style={{ accentColor: 'var(--accent)', width: 13, height: 13 }}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {loadChangeTable.length > 0 && (
                    <>
                      <div
                        style={{
                          ...S.panelTitle,
                          marginTop: 12,
                          borderTop: '1px solid var(--outline)',
                          borderRadius: 0,
                        }}
                      >
                        Load % vs Target
                      </div>
                      <table style={S.table}>
                        <thead>
                          <tr>
                            {['Date', 'Avg MW', '% Change'].map((h) => (
                              <th key={h} style={S.th}>
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {loadChangeTable.map((r) => (
                            <tr key={r.date}>
                              <td style={S.td}>{r.date}</td>
                              <td style={S.td}>{r.avg}</td>
                              <td style={S.td}>
                                {r.pct !== null ? (
                                  <span style={S.badge(Number(r.pct) > 0 ? 'working' : 'holiday')}>
                                    {Number(r.pct) > 0 ? '+' : ''}
                                    {r.pct}%
                                  </span>
                                ) : (
                                  '—'
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Right — charts */}
          <div style={S.right}>
            <div style={S.tabBar}>
              {[
                ['weather', 'Weather Profiles'],
                ['load', 'Load Comparison'],
              ].map(([k, l]) => (
                <button key={k} onClick={() => setActiveTab(k)} style={S.tab(activeTab === k)}>
                  {l}
                </button>
              ))}
            </div>
            <div style={S.chartArea}>
              {!result ? (
                <div style={S.empty}>Run a search to see charts.</div>
              ) : (
                <div style={S.chartCard}>
                  <div style={S.chartTitle}>
                    {activeTab === 'weather'
                      ? 'Weather Profiles — Temp / Humidity / Precipitation'
                      : 'Load Profiles — Similar Days vs Target'}
                  </div>
                  {activeTab === 'weather' && weatherSubplotOption && (
                    <ReactECharts
                      option={weatherSubplotOption}
                      style={{ flex: 1, minHeight: 0 }}
                      notMerge
                      lazyUpdate
                    />
                  )}
                  {activeTab === 'load' && loadOverlayOption && (
                    <ReactECharts
                      option={loadOverlayOption}
                      style={{ flex: 1, minHeight: 0 }}
                      notMerge
                      lazyUpdate
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </VpPageShell>
  );
}
