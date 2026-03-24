import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

/* ─── Helpers ─── */
const fmt = (v, d = 1) => v == null || !Number.isFinite(v) ? '--' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });
const blockToTime = (b) => { const h = Math.floor((b - 1) / 4); const m = ((b - 1) % 4) * 15; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`; };
const blocks96 = Array.from({ length: 96 }, (_, i) => i + 1);
const timeLabels = blocks96.map(blockToTime);

/* ─── Styles ─── */
const S = {
  page: { fontFamily: "'IBM Plex Mono', monospace", color: '#ECEEF3', minHeight: '100%', display: 'flex', flexDirection: 'column', gap: 4 },
  card: { background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F', overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1, color: '#A0A5B8', padding: '10px 14px', borderBottom: '1px solid #2A292F' },
  badge: (color) => ({ fontSize: 8, fontWeight: 700, padding: '2px 8px', borderRadius: 12, background: `${color}18`, color }),
  th: { fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, padding: '6px 8px', borderBottom: '1px solid #2A292F', textAlign: 'right', fontWeight: 600 },
  td: { fontSize: 10, padding: '5px 8px', borderBottom: '1px solid #2A292F22', textAlign: 'right' },
  tabBar: { display: 'inline-flex', gap: 3, padding: '4px 6px', background: '#1A191E', border: '1px solid #2A292F', borderRadius: 999 },
  tab: (active) => ({ padding: '5px 13px', fontSize: 9, fontWeight: 600, cursor: 'pointer', border: 'none', fontFamily: 'inherit', borderRadius: 999, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#6B7186', transition: 'all 0.15s' }),
};

const ecBase = () => ({
  backgroundColor: 'transparent',
  textStyle: { color: '#A0A5B8', fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 },
  grid: { top: 36, right: 20, bottom: 32, left: 55 },
  tooltip: { trigger: 'axis', backgroundColor: '#1A191E', borderColor: '#2A292F', borderWidth: 1, textStyle: { color: '#ECEEF3', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }, confine: true },
  legend: { textStyle: { color: '#A0A5B8', fontSize: 9 }, top: 4, right: 8, itemWidth: 12, itemHeight: 3 },
  xAxis: { type: 'category', axisLine: { lineStyle: { color: '#2A292F' } }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { show: false } },
  yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { lineStyle: { color: '#2A292F', type: 'dashed', opacity: 0.3 } } },
});

/* ═══════════════════════════════════════════════════════════════ */
export default function LoadAnalysisPage({
  effectiveDate, benchmarkData, momentumChange,
  multiDaySeries, multiSelectedDates, setMultiSelectedDates,
  detectedSeason, liveData, dayAheadData,
}) {
  const [mainTab, setMainTab] = useState('benchmark');
  const [compareDates, setCompareDates] = useState(multiSelectedDates || []);

  const today = useMemo(() => (benchmarkData?.today || []).map(Number).filter(Number.isFinite), [benchmarkData]);
  const t1 = useMemo(() => (benchmarkData?.t1 || []).map(Number).filter(Number.isFinite), [benchmarkData]);
  const t7 = useMemo(() => (benchmarkData?.t7 || []).map(Number).filter(Number.isFinite), [benchmarkData]);
  const t365 = useMemo(() => (benchmarkData?.t365 || []).map(Number).filter(Number.isFinite), [benchmarkData]);

  /* ─── Derived metrics ─── */
  const peak = today.length ? Math.max(...today) : 0;
  const valley = today.length ? Math.min(...today.filter(v => v > 0)) : 0;
  const yPeak = t1.length ? Math.max(...t1) : 0;
  const yValley = t1.length ? Math.min(...t1.filter(v => v > 0)) : 0;
  const peakIdx = today.indexOf(peak);
  const energy = today.reduce((s, v) => s + v * 0.25, 0);
  const yEnergy = t1.reduce((s, v) => s + v * 0.25, 0);
  const avgLoad = today.length ? today.reduce((s, v) => s + v, 0) / today.length : 0;
  const loadFactor = peak > 0 ? (avgLoad / peak) * 100 : 0;

  /* Block-wise deviation */
  const deviation = useMemo(() => today.map((v, i) => t1[i] ? v - t1[i] : 0), [today, t1]);
  const absDeviation = deviation.map(Math.abs);
  const maxDevBlock = absDeviation.indexOf(Math.max(...absDeviation));
  const avgDev = absDeviation.length ? absDeviation.reduce((s, v) => s + v, 0) / absDeviation.length : 0;

  /* Load Duration Curve (sorted descending) */
  const sortedLoad = useMemo(() => [...today].sort((a, b) => b - a), [today]);

  /* Hourly aggregation for heatmap-style view */
  const hourlyAvg = useMemo(() => {
    const hrs = Array.from({ length: 24 }, () => []);
    today.forEach((v, i) => hrs[Math.floor(i / 4)].push(v));
    return hrs.map(arr => arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0);
  }, [today]);

  const yHourlyAvg = useMemo(() => {
    const hrs = Array.from({ length: 24 }, () => []);
    t1.forEach((v, i) => hrs[Math.floor(i / 4)].push(v));
    return hrs.map(arr => arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0);
  }, [t1]);

  /* Ramps */
  const ramps = useMemo(() => today.length > 1 ? today.slice(1).map((v, i) => v - today[i]) : [], [today]);
  const maxRampUp = ramps.length ? Math.max(...ramps) : 0;
  const maxRampDown = ramps.length ? Math.min(...ramps) : 0;

  /* Regime */
  const dayRegime = useMemo(() => {
    const fc = liveData?.series?.forecast || [];
    const bl = liveData?.series?.hybrid_baseline || [];
    if (!fc.length || !bl.length) { const d = new Date(effectiveDate).getDay(); return d === 0 || d === 6 ? 'Weekend' : 'Weekday'; }
    const r = fc.reduce((a, b) => a + b, 0) / Math.max(bl.reduce((a, b) => a + b, 0), 1);
    if (liveData?.metadata?.holiday_flags?.is_holiday) return 'Holiday';
    if (r > 1.15) return 'High Load';
    if (r < 0.85) return 'Low Demand';
    const d = new Date(effectiveDate).getDay();
    return d === 0 || d === 6 ? 'Weekend' : 'Weekday';
  }, [liveData, effectiveDate]);

  /* Trend */
  const trend = useMemo(() => {
    if (!momentumChange?.data?.length) return { dir: '--', val: 0, data: [] };
    const r = momentumChange.data.slice(-7);
    const avg = r.reduce((s, d) => s + (d.value || 0), 0) / r.length;
    return { dir: avg > 0.5 ? 'Rising' : avg < -0.5 ? 'Falling' : 'Flat', val: avg, data: momentumChange.data };
  }, [momentumChange]);

  const similarDays = liveData?.metadata?.similar_days || [];
  const hasData = today.length > 0;

  /* ═══ CHART OPTIONS ═══ */

  /* 1. Benchmark overlay */
  const benchmarkOpt = useMemo(() => {
    if (!today.length) return null;
    const now = new Date(); const nowBlk = Math.min(95, now.getHours() * 4 + Math.floor(now.getMinutes() / 15));
    const mk = (name, data, color, w = 1.5, dash = false) => data.length ? { name, type: 'line', data, smooth: true, symbol: 'none', lineStyle: { width: w, color, type: dash ? 'dashed' : 'solid' }, itemStyle: { color } } : null;
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: ['Today', 'Yesterday', 'Last Week', 'Last Year'].filter((_, i) => [today, t1, t7, t365][i].length) },
      xAxis: { ...ecBase().xAxis, data: timeLabels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [
        { ...mk('Today', today, '#F07825', 2.5), z: 10,
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#F0782518' }, { offset: 1, color: 'transparent' }] } },
          markPoint: { data: [{ type: 'max', symbolSize: 28, label: { formatter: p => `${fmt(p.value, 0)}`, fontSize: 9, color: '#F07825', fontWeight: 700 } }], itemStyle: { color: '#F07825' } },
          markLine: { silent: true, symbol: 'none', data: [{ xAxis: blockToTime(nowBlk + 1), lineStyle: { color: '#F87171', type: 'dashed', width: 1 }, label: { formatter: 'NOW', fontSize: 8, color: '#F87171' } }] },
        },
        mk('Yesterday', t1, '#ECEEF340', 1.5, true),
        mk('Last Week', t7, '#FBBF2480', 1.2, true),
        mk('Last Year', t365, '#8B5CF680', 1.2, true),
      ].filter(Boolean),
    };
  }, [today, t1, t7, t365]);

  /* 2. Deviation waterfall */
  const deviationOpt = useMemo(() => {
    if (!deviation.length) return null;
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: timeLabels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW Δ vs Yesterday' },
      visualMap: { show: false, min: -300, max: 300, inRange: { color: ['#34D399', '#2A292F', '#F87171'] } },
      series: [{ type: 'bar', data: deviation, barMaxWidth: 5,
        markLine: { silent: true, symbol: 'none', data: [{ yAxis: 0, lineStyle: { color: '#6B718640', width: 1 } }] },
      }],
    };
  }, [deviation]);

  /* 3. Load Duration Curve */
  const durationOpt = useMemo(() => {
    if (!sortedLoad.length) return null;
    const pctLabels = sortedLoad.map((_, i) => `${((i / sortedLoad.length) * 100).toFixed(0)}%`);
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: pctLabels, name: '% of Time Exceeded', axisLabel: { ...ecBase().xAxis.axisLabel, interval: 9 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [{ type: 'line', data: sortedLoad, smooth: true, symbol: 'none',
        lineStyle: { width: 2.5, color: '#F07825' }, itemStyle: { color: '#F07825' },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#F0782515' }, { offset: 1, color: 'transparent' }] } },
        markPoint: { data: [
          { coord: [0, sortedLoad[0]], symbolSize: 20, label: { formatter: `Peak\n${fmt(sortedLoad[0], 0)}`, fontSize: 8, color: '#F87171' }, itemStyle: { color: '#F87171' } },
          { coord: [pctLabels[Math.floor(sortedLoad.length * 0.5)], sortedLoad[Math.floor(sortedLoad.length * 0.5)]], symbolSize: 16, label: { formatter: `P50\n${fmt(sortedLoad[Math.floor(sortedLoad.length * 0.5)], 0)}`, fontSize: 8, color: '#FBBF24' }, itemStyle: { color: '#FBBF24' } },
        ] },
      }],
    };
  }, [sortedLoad]);

  /* 4. Multi-day comparison */
  const comparisonOpt = useMemo(() => {
    const dates = Object.keys(multiDaySeries || {});
    if (!dates.length) return null;
    const colors = ['#F07825', '#34D399', '#F87171', '#FBBF24', '#8B5CF6', '#45b7d1', '#EC4899'];
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: dates },
      xAxis: { ...ecBase().xAxis, data: timeLabels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: dates.map((d, i) => ({ name: d, type: 'line', data: multiDaySeries[d], smooth: true, symbol: 'none', lineStyle: { width: 2, color: colors[i % colors.length] }, itemStyle: { color: colors[i % colors.length] } })),
    };
  }, [multiDaySeries]);

  /* 5. 30-day momentum */
  const momentumOpt = useMemo(() => {
    if (!trend.data.length) return null;
    return {
      ...ecBase(),
      grid: { ...ecBase().grid, bottom: 50 },
      xAxis: { ...ecBase().xAxis, data: trend.data.map(d => d.date), axisLabel: { ...ecBase().xAxis.axisLabel, rotate: 35 } },
      yAxis: { ...ecBase().yAxis, name: 'DoD %' },
      series: [{
        type: 'bar', data: trend.data.map(d => ({ value: d.value, itemStyle: { color: d.value >= 0 ? '#34D399' : '#F87171', borderRadius: d.value >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3] } })),
        barMaxWidth: 14,
        markLine: { silent: true, symbol: 'none', data: [{ yAxis: 0, lineStyle: { color: '#6B718640' } }] },
      }],
    };
  }, [trend]);

  /* 6. Hourly deviation heatbar */
  /* Hourly deviation: only for hours with actual data (today > 0 and yesterday > 0) */
  const hourlyDev = useMemo(() => {
    return Array.from({ length: 24 }, (_, h) => {
      const t = hourlyAvg[h], y = yHourlyAvg[h];
      if (!t || t <= 0 || !y || y <= 0) return null;
      return { hour: `${String(h).padStart(2, '0')}:00`, pct: ((t - y) / y) * 100, today: t, yday: y };
    }).filter(Boolean);
  }, [hourlyAvg, yHourlyAvg]);

  const TABS = [
    { id: 'benchmark', l: 'Benchmark Overlay' },
    { id: 'deviation', l: 'Deviation Map' },
    { id: 'duration', l: 'Load Duration' },
    { id: 'comparison', l: 'Multi-Day' },
    { id: 'momentum', l: '30D Momentum' },
  ];
  const chartMap = { benchmark: benchmarkOpt, deviation: deviationOpt, duration: durationOpt, comparison: comparisonOpt, momentum: momentumOpt };

  /* ═══ RENDER ═══ */
  return (
    <div style={S.page}>
      {!hasData ? (
        <div style={{ padding: 60, textAlign: 'center', color: '#6B7186' }}>
          <div style={{ fontSize: 16, marginBottom: 6 }}>No load data available</div>
          <div style={{ fontSize: 11 }}>Select a date to load benchmark data</div>
        </div>
      ) : (
        <>
          {/* ═══ ROW 1: Scoreboard ═══ */}
          <div style={{ display: 'flex', gap: 6, padding: '6px 16px', flexWrap: 'wrap' }}>
            {/* Today vs Yesterday comparison cards */}
            {[
              { label: 'Peak', today: peak, yday: yPeak, unit: 'MW', color: '#F07825' },
              { label: 'Valley', today: valley, yday: yValley, unit: 'MW', color: '#34D399' },
              { label: 'Energy', today: energy, yday: yEnergy, unit: 'MWh', color: '#ECEEF3', round: true },
              { label: 'Load Factor', today: loadFactor, yday: null, unit: '%', color: loadFactor > 70 ? '#34D399' : '#FBBF24' },
            ].map(m => {
              const delta = m.yday ? ((m.today - m.yday) / Math.max(m.yday, 1)) * 100 : null;
              return (
                <div key={m.label} style={{ flex: '1 1 0', minWidth: 130, padding: '10px 14px', background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F' }}>
                  <div style={{ fontSize: 8, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 3 }}>{m.label}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: m.color, lineHeight: 1.15 }}>
                    {fmt(m.today, m.round ? 0 : 1)}<span style={{ fontSize: 10, fontWeight: 400, opacity: 0.5 }}> {m.unit}</span>
                  </div>
                  {delta != null && (
                    <div style={{ fontSize: 9, color: delta >= 0 ? '#F87171' : '#34D399', fontWeight: 600, marginTop: 2 }}>
                      {delta >= 0 ? '▲' : '▼'} {fmt(Math.abs(delta))}% vs yday
                    </div>
                  )}
                </div>
              );
            })}
            {/* Regime + Season */}
            <div style={{ flex: '1 1 0', minWidth: 130, padding: '10px 14px', background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F' }}>
              <div style={{ fontSize: 8, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 3 }}>Regime</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#F07825' }}>{dayRegime}</div>
              <div style={{ fontSize: 9, color: '#6B7186', marginTop: 2 }}>{detectedSeason} • {effectiveDate}</div>
            </div>
            {/* 30D Trend mini */}
            <div style={{ flex: '1 1 0', minWidth: 130, padding: '10px 14px', background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F' }}>
              <div style={{ fontSize: 8, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 3 }}>30D Trend</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: trend.val > 0.5 ? '#F87171' : trend.val < -0.5 ? '#34D399' : '#ECEEF3' }}>
                {trend.dir} <span style={{ fontSize: 10, fontWeight: 400 }}>({trend.val > 0 ? '+' : ''}{fmt(trend.val)}%/d)</span>
              </div>
              {/* sparkline */}
              <div style={{ display: 'flex', gap: 1, alignItems: 'flex-end', height: 18, marginTop: 4 }}>
                {trend.data.slice(-14).map((d, i) => {
                  const mx = Math.max(...trend.data.slice(-14).map(x => Math.abs(x.value || 0)), 1);
                  return <div key={i} style={{ flex: 1, height: Math.max(2, Math.abs(d.value || 0) / mx * 16), background: (d.value || 0) >= 0 ? '#34D39966' : '#F8717166', borderRadius: 1 }} />;
                })}
              </div>
            </div>
          </div>

          {/* ═══ Hourly Deviation Strip ═══ */}
          {hourlyDev.length > 0 && (
            <div style={{ padding: '0 16px' }}>
              <div style={S.card}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 14px' }}>
                  <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1, color: '#A0A5B8' }}>HOURLY LOAD vs YESTERDAY</span>
                  <div style={{ display: 'flex', gap: 12, fontSize: 8, color: '#6B7186' }}>
                    <span><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#34D399', marginRight: 3 }} />Lower</span>
                    <span><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#F87171', marginRight: 3 }} />Higher</span>
                  </div>
                </div>
                <div style={{ display: 'flex', padding: '4px 14px 10px', gap: 3 }}>
                  {hourlyDev.map(h => {
                    const clamp = Math.max(-10, Math.min(10, h.pct));
                    const intensity = Math.abs(clamp) / 10;
                    const bg = h.pct >= 0
                      ? `rgba(248, 113, 113, ${0.15 + intensity * 0.6})`
                      : `rgba(52, 211, 153, ${0.15 + intensity * 0.6})`;
                    return (
                      <div key={h.hour} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }} title={`${h.hour}: Today ${fmt(h.today, 0)} MW vs Y'day ${fmt(h.yday, 0)} MW`}>
                        <div style={{ width: '100%', height: 28, borderRadius: 4, background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <span style={{ fontSize: 8, fontWeight: 700, color: '#ECEEF3' }}>
                            {h.pct >= 0 ? '+' : ''}{fmt(h.pct, 1)}%
                          </span>
                        </div>
                        <span style={{ fontSize: 7, color: '#6B7186' }}>{h.hour.slice(0, 2)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* ═══ ROW 2: Tabs + Main Chart ═══ */}
          <div style={{ display: 'flex', alignItems: 'center', padding: '2px 16px', gap: 8 }}>
            <div style={S.tabBar}>
              {TABS.map(t => <button key={t.id} onClick={() => setMainTab(t.id)} style={S.tab(mainTab === t.id)}>{t.l}</button>)}
            </div>
            {mainTab === 'comparison' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 8 }}>
                <input type="date" onChange={e => {
                  const d = e.target.value;
                  if (d && !compareDates.includes(d)) { const n = [...compareDates, d]; setCompareDates(n); setMultiSelectedDates(n); }
                }} style={{ background: '#1A191E', border: '1px solid #2A292F', borderRadius: 999, padding: '4px 10px', color: '#ECEEF3', fontSize: 9, fontFamily: 'inherit' }} />
                {compareDates.map(d => (
                  <span key={d} style={{ ...S.badge('#F07825'), display: 'flex', alignItems: 'center', gap: 4 }}>
                    {d} <button onClick={() => { const n = compareDates.filter(x => x !== d); setCompareDates(n); setMultiSelectedDates(n); }} style={{ border: 'none', background: 'none', color: '#F07825', cursor: 'pointer', fontSize: 10, fontWeight: 700 }}>×</button>
                  </span>
                ))}
                {compareDates.length > 0 && <button onClick={() => { setCompareDates([]); setMultiSelectedDates([]); }} style={{ ...S.badge('#F87171'), cursor: 'pointer', border: 'none', fontFamily: 'inherit' }}>Clear</button>}
              </div>
            )}
          </div>

          <div style={{ padding: '0 16px' }}>
            <div style={S.card}>
              <div style={{ flex: 1, padding: 4 }}>
                {chartMap[mainTab] ? (
                  <ReactECharts option={chartMap[mainTab]} style={{ height: 340, width: '100%' }} notMerge />
                ) : (
                  <div style={{ padding: 50, textAlign: 'center', color: '#6B7186', fontSize: 11 }}>
                    {mainTab === 'comparison' ? 'Pick dates above to compare' : 'No data for this view'}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ═══ ROW 3: Bottom panels ═══ */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, padding: '0 16px 12px' }}>

            {/* Block-wise Structural Comparison Table */}
            <div style={S.card}>
              <div style={S.cardTitle}>Structural Comparison</div>
              <div style={{ padding: 0 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ ...S.th, textAlign: 'left' }}>Metric</th>
                      <th style={S.th}>Today</th>
                      <th style={S.th}>Y'day</th>
                      <th style={S.th}>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { m: 'Peak (MW)', t: peak, y: yPeak },
                      { m: 'Valley (MW)', t: valley, y: yValley },
                      { m: 'Energy (MWh)', t: energy, y: yEnergy },
                      { m: 'Peak Hour', t: blockToTime(peakIdx + 1), y: t1.length ? blockToTime(t1.indexOf(yPeak) + 1) : '--', raw: true },
                      { m: 'Load Factor %', t: loadFactor, y: yPeak > 0 && t1.length ? (t1.reduce((s, v) => s + v, 0) / t1.length / yPeak * 100) : 0 },
                      { m: 'Max Ramp ↑', t: maxRampUp, y: null },
                      { m: 'Max Ramp ↓', t: maxRampDown, y: null },
                      { m: 'Avg Dev (MW)', t: avgDev, y: null },
                    ].map((r, i) => {
                      const delta = !r.raw && r.y != null && r.y > 0 ? ((r.t - r.y) / r.y * 100) : null;
                      return (
                        <tr key={i}>
                          <td style={{ ...S.td, textAlign: 'left', fontWeight: 600, color: '#A0A5B8' }}>{r.m}</td>
                          <td style={{ ...S.td, fontWeight: 600, color: '#ECEEF3' }}>{r.raw ? r.t : fmt(r.t, 0)}</td>
                          <td style={{ ...S.td, color: '#6B7186' }}>{r.y == null ? '--' : r.raw ? r.y : fmt(r.y, 0)}</td>
                          <td style={{ ...S.td, color: delta != null ? (delta >= 0 ? '#F87171' : '#34D399') : '#6B7186', fontWeight: 600 }}>
                            {delta != null ? `${delta >= 0 ? '+' : ''}${fmt(delta)}%` : '--'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Similar Days */}
            <div style={S.card}>
              <div style={S.cardTitle}>Similar Historical Days</div>
              <div style={{ padding: 8 }}>
                {similarDays.length > 0 ? similarDays.slice(0, 6).map((d, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '5px 6px', borderBottom: '1px solid #2A292F22' }}>
                    <span style={{ fontSize: 10, fontWeight: 600, color: '#ECEEF3' }}>{d.date}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {d.day_type && <span style={{ fontSize: 8, color: '#6B7186' }}>{d.day_type}</span>}
                      <div style={{ width: 60, height: 6, background: '#2A292F', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${(d.similarity_score * 100)}%`, height: '100%', background: '#F07825', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 9, fontWeight: 700, color: '#F07825', minWidth: 28, textAlign: 'right' }}>{(d.similarity_score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                )) : <div style={{ padding: 20, textAlign: 'center', color: '#6B7186', fontSize: 10 }}>No similar days found</div>}
              </div>
            </div>

            {/* Load Shape Fingerprint */}
            <div style={S.card}>
              <div style={S.cardTitle}>Load Shape Fingerprint</div>
              <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 5 }}>
                {[
                  { l: 'Peak Hour', v: blockToTime(peakIdx + 1), c: '#F07825' },
                  { l: 'Night Base (00–06)', v: `${fmt(hourlyAvg.slice(0, 6).reduce((s, v) => s + v, 0) / 6, 0)} MW`, c: '#6366F1' },
                  { l: 'Morning Ramp (06–10)', v: `+${fmt(hourlyAvg[10] - hourlyAvg[6], 0)} MW`, c: '#FBBF24' },
                  { l: 'Afternoon Plateau', v: `${fmt(hourlyAvg.slice(12, 17).reduce((s, v) => s + v, 0) / 5, 0)} MW`, c: '#F07825' },
                  { l: 'Evening Peak (18–21)', v: `${fmt(hourlyAvg.slice(18, 21).reduce((s, v) => s + v, 0) / 3, 0)} MW`, c: '#EC4899' },
                  { l: 'Demand Range', v: `${fmt(peak - valley, 0)} MW`, c: '#ECEEF3' },
                  { l: 'P/V Ratio', v: valley > 0 ? (peak / valley).toFixed(2) : '--', c: '#ECEEF3' },
                ].map((r, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                    <span style={{ color: '#6B7186' }}>{r.l}</span>
                    <span style={{ color: r.c, fontWeight: 600 }}>{r.v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
