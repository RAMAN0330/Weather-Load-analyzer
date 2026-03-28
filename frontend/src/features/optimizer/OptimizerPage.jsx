import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

/* ─── Helpers ─── */
const fmt = (v, d = 1) => v == null || !Number.isFinite(v) ? '--' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });
function blockToTime(idx) { const h = Math.floor((idx - 1) / 4); const m = ((idx - 1) % 4) * 15; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`; }

const getNumericStats = (values = []) => {
  const nums = values.filter(v => Number.isFinite(v));
  if (!nums.length) return null;
  const mean = nums.reduce((s, v) => s + v, 0) / nums.length;
  const sorted = [...nums].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const variance = nums.reduce((s, v) => s + (v - mean) ** 2, 0) / nums.length;
  const std = Math.sqrt(variance);
  const mae = nums.reduce((s, v) => s + Math.abs(v), 0) / nums.length;
  return { mean, median, std, mae, min: sorted[0], max: sorted[sorted.length - 1], count: nums.length };
};

const buildHistogram = (values = [], binCount = 10) => {
  const nums = values.filter(v => Number.isFinite(v));
  if (!nums.length) return null;
  const min = Math.min(...nums), max = Math.max(...nums);
  const range = max - min || 1;
  const binWidth = range / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    start: min + i * binWidth, end: min + (i + 1) * binWidth,
    mid: min + (i + 0.5) * binWidth, count: 0
  }));
  nums.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
    bins[idx].count++;
  });
  return { bins };
};

/* ─── Styles ─── */
const S = {
  page: { fontFamily: "'IBM Plex Mono', monospace", color: '#ECEEF3', minHeight: 0, height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 2 },
  kpiRow: { display: 'flex', gap: 8, padding: '10px 16px 6px', flexWrap: 'wrap' },
  kpi: { flex: '1 1 0', minWidth: 130, padding: '12px 14px', background: '#1A191E', borderRadius: 8, border: '1px solid #2A292F' },
  kpiLabel: { fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 4 },
  kpiVal: (c) => ({ fontSize: 24, fontWeight: 700, color: c || '#ECEEF3', lineHeight: 1.15 }),
  kpiUnit: { fontSize: 11, fontWeight: 400, opacity: 0.5 },
  kpiSub: { fontSize: 9, color: '#6B7186', marginTop: 3 },
  card: { background: '#1A191E', borderRadius: 10, border: '1px solid #2A292F', overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1, color: '#A0A5B8', padding: '10px 14px', borderBottom: '1px solid #2A292F' },
  tabBar: { display: 'inline-flex', gap: 3, padding: '4px 6px', background: '#1A191E', border: '1px solid #2A292F', borderRadius: 999, margin: '8px 16px 4px' },
  tab: (active) => ({ flex: '0 0 auto', padding: '6px 14px', fontSize: 10, fontWeight: 600, letterSpacing: 0.5, cursor: 'pointer', border: 'none', fontFamily: 'inherit', borderRadius: 999, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#6B7186', transition: 'all 0.15s', display: 'flex', alignItems: 'center', gap: 6 }),
  badge: (color) => ({ fontSize: 8, fontWeight: 700, padding: '2px 8px', borderRadius: 12, background: `${color}18`, color }),
  sectionLabel: (color) => ({ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 2, color: color || '#F07825', padding: '8px 16px 2px' }),
  th: { fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, padding: '8px 10px', borderBottom: '1px solid #2A292F', textAlign: 'left', fontWeight: 600 },
  td: { fontSize: 10, padding: '6px 10px', borderBottom: '1px solid #2A292F22' },
};

/* ─── ECharts base ─── */
const ecBase = () => ({
  backgroundColor: 'transparent',
  textStyle: { color: '#A0A5B8', fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 },
  grid: { top: 36, right: 20, bottom: 32, left: 50, containLabel: false },
  tooltip: { trigger: 'axis', backgroundColor: '#1A191E', borderColor: '#2A292F', borderWidth: 1, textStyle: { color: '#ECEEF3', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }, confine: true },
  legend: { textStyle: { color: '#A0A5B8', fontSize: 9 }, top: 4, right: 8, itemWidth: 12, itemHeight: 3 },
  xAxis: { type: 'category', axisLine: { lineStyle: { color: '#2A292F' } }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { show: false } },
  yAxis: { type: 'value', axisLine: { show: false }, axisLabel: { color: '#6B7186', fontSize: 9 }, splitLine: { lineStyle: { color: '#2A292F', type: 'dashed', opacity: 0.3 } } },
});

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function OptimizerPage({
  dayAheadData,
  dayAheadSeries,
  baselineWindowMapes = [],
  baselineDays,
  setBaselineDays,
  availableActualBlocks = 0,
  effectiveDate,
}) {
  const [activeTab, setActiveTab] = useState('window');
  const dateStr = effectiveDate || dayAheadData?.metadata?.effective_date || '--';

  /* ─── Computed Data ─── */
  const residuals = useMemo(() => {
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!actual.length || !baseline.length) return [];
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    return actual.slice(0, limit).map((val, idx) => {
      const base = baseline[idx];
      if (!Number.isFinite(val) || val === 0 || !Number.isFinite(base)) return null;
      return val - base;
    });
  }, [dayAheadSeries, availableActualBlocks]);

  const residualStats = useMemo(() => getNumericStats(residuals), [residuals]);

  const stability = useMemo(() => {
    const values = baselineWindowMapes.map(r => Number(r.baseline_mape)).filter(Number.isFinite).sort((a, b) => a - b);
    if (!values.length) return null;
    return { min: values[0], max: values[values.length - 1], median: values[Math.floor(values.length / 2)], q1: values[Math.floor((values.length - 1) * 0.25)], q3: values[Math.floor((values.length - 1) * 0.75)] };
  }, [baselineWindowMapes]);

  const peakMarkers = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!blocks.length) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    let aIdx = -1, aPeak = -Infinity, bIdx = -1, bPeak = -Infinity;
    actual.slice(0, limit).forEach((v, i) => { if (Number.isFinite(v) && v > 0 && v > aPeak) { aPeak = v; aIdx = i; } });
    baseline.forEach((v, i) => { if (Number.isFinite(v) && v > bPeak) { bPeak = v; bIdx = i; } });
    if (aIdx < 0 && bIdx < 0) return null;
    return { actual: aIdx >= 0 ? { block: blocks[aIdx], value: aPeak } : null, baseline: bIdx >= 0 ? { block: blocks[bIdx], value: bPeak } : null };
  }, [dayAheadSeries, availableActualBlocks]);

  const peakError = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!blocks.length) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    let aIdx = -1, aPeak = -Infinity;
    actual.slice(0, limit).forEach((v, i) => { if (Number.isFinite(v) && v > 0 && v > aPeak) { aPeak = v; aIdx = i; } });
    if (aIdx < 0) return null;
    const bAtPeak = Number.isFinite(baseline[aIdx]) ? baseline[aIdx] : null;
    return { block: blocks[aIdx], actualPeak: aPeak, baselineAtPeak: bAtPeak, errorMw: Number.isFinite(bAtPeak) ? aPeak - bAtPeak : null };
  }, [dayAheadSeries, availableActualBlocks]);

  const rampSeries = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (blocks.length < 2) return null;
    const limit = availableActualBlocks > 0 ? availableActualBlocks : actual.length;
    const rampBlocks = blocks.slice(1, limit);
    const actualRamp = rampBlocks.map((_, i) => { const c = actual[i + 1], p = actual[i]; return Number.isFinite(c) && c > 0 && Number.isFinite(p) && p > 0 ? c - p : null; });
    const baselineRamp = rampBlocks.map((_, i) => { const c = baseline[i + 1], p = baseline[i]; return Number.isFinite(c) && Number.isFinite(p) ? c - p : null; });
    const rampError = actualRamp.map((v, i) => { const b = baselineRamp[i]; return Number.isFinite(v) && Number.isFinite(b) ? v - b : null; });
    return { rampBlocks, actualRamp, baselineRamp, rampError };
  }, [dayAheadSeries, availableActualBlocks]);

  /* ─── Window Tab Data ─── */
  const bestRow = baselineWindowMapes.length > 0 ? [...baselineWindowMapes].sort((a, b) => Number(a.baseline_mape) - Number(b.baseline_mape))[0] : null;
  const bestWindow = bestRow ? Number(bestRow.window_days) : null;
  const bestMape = bestRow ? Number(bestRow.baseline_mape) : null;
  const currentMape = dayAheadData?.metadata?.selected_window_baseline_mape;
  const mapeGap = currentMape != null && bestMape != null ? Number(currentMape) - bestMape : null;
  const stabilitySpread = stability ? stability.max - stability.min : null;
  const isStable = stabilitySpread != null && stabilitySpread < 1.0;
  const windowAction = mapeGap != null && mapeGap > 0.3 ? `Switch to ${bestWindow}D (−${fmt(mapeGap)}%)` : 'Near-optimal';

  /* ─── Pattern Tab Data ─── */
  const bias = residualStats?.mean ?? 0;
  const biasDir = bias > 10 ? 'Under-forecasting' : bias < -10 ? 'Over-forecasting' : 'Well-centered';
  const windowDefs = [
    { name: 'Morning (5-10h)', s: 20, e: 40 }, { name: 'Midday (10-17h)', s: 40, e: 68 },
    { name: 'Evening (17-23h)', s: 68, e: 92 }, { name: 'Night (23-5h)', s: 92, e: 20 },
  ];
  let worstWindow = '--', worstMAE = 0;
  const validResiduals = residuals.filter(Number.isFinite);
  windowDefs.forEach(w => {
    const blocks = w.s < w.e ? validResiduals.slice(w.s, w.e) : [...validResiduals.slice(w.s), ...validResiduals.slice(0, w.e)];
    const mae = blocks.length ? blocks.reduce((s, v) => s + Math.abs(v), 0) / blocks.length : 0;
    if (mae > worstMAE) { worstMAE = mae; worstWindow = w.name; }
  });
  let rampMAE = 0;
  if (rampSeries?.rampError) { const vr = rampSeries.rampError.filter(Number.isFinite); rampMAE = vr.length ? vr.reduce((s, v) => s + Math.abs(v), 0) / vr.length : 0; }
  const peakTimingOff = peakMarkers?.actual && peakMarkers?.baseline ? Math.abs(peakMarkers.actual.block - peakMarkers.baseline.block) : null;

  /* ─── Error Tab Data ─── */
  const threshold2std = residualStats ? residualStats.std * 2 : Infinity;
  const outlierCount = validResiduals.filter(r => Math.abs(r) > threshold2std).length;
  const outlierPct = validResiduals.length ? (outlierCount / validResiduals.length) * 100 : 0;
  let maxConsec = 0, curConsec = 0, lastSign = 0;
  validResiduals.forEach(r => { const s = r > 0 ? 1 : -1; if (s === lastSign) curConsec++; else { curConsec = 1; lastSign = s; } if (curConsec > maxConsec) maxConsec = curConsec; });
  const hasSystematicBias = maxConsec >= 8;
  const morningErr = validResiduals.slice(20, 40);
  const eveningErr = validResiduals.slice(68, 92);
  const morningMAE = morningErr.length ? morningErr.reduce((s, v) => s + Math.abs(v), 0) / morningErr.length : 0;
  const eveningMAE = eveningErr.length ? eveningErr.reduce((s, v) => s + Math.abs(v), 0) / eveningErr.length : 0;
  const errorConc = morningMAE > eveningMAE ? 'Morning ramp' : 'Evening peak';
  const mape = dayAheadData?.kpis?.mape;
  const grade = mape != null ? (mape < 1.5 ? 'A' : mape < 3 ? 'B' : mape < 5 ? 'C' : 'D') : '--';

  /* ═══ CHART OPTIONS ═══ */
  const accuracyOption = useMemo(() => {
    if (!baselineWindowMapes.length) return null;
    const rows = [...baselineWindowMapes].sort((a, b) => Number(a.window_days) - Number(b.window_days));
    const windows = rows.map(r => r.window_days);
    const mapes = rows.map(r => Number(r.baseline_mape) || 0);
    let bIdx = 0, bVal = Infinity;
    mapes.forEach((v, i) => { if (Number.isFinite(v) && v < bVal) { bVal = v; bIdx = i; } });
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: windows, name: 'Window (Days)' },
      yAxis: { ...ecBase().yAxis, name: 'MAPE %' },
      series: [{
        name: 'MAPE', type: 'line', data: mapes, smooth: true, symbolSize: 6,
        lineStyle: { color: '#F07825', width: 2.5 }, itemStyle: { color: '#F07825' },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#F0782520' }, { offset: 1, color: 'transparent' }] } },
        markPoint: Number.isFinite(bVal) ? { data: [{ coord: [windows[bIdx], bVal], symbolSize: 12, itemStyle: { color: '#34D399' }, label: { formatter: `Best: ${bVal.toFixed(2)}%`, fontSize: 9, color: '#34D399' } }] } : undefined,
        markLine: Number.isFinite(baselineDays) ? { data: [{ xAxis: baselineDays, lineStyle: { color: '#F0782550', type: 'dashed' }, label: { formatter: 'Current', fontSize: 8, color: '#F07825' } }], symbol: 'none' } : undefined,
      }],
    };
  }, [baselineWindowMapes, baselineDays]);

  const patternFitOption = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    const actual = dayAheadSeries?.actual || [];
    const baseline = dayAheadSeries?.baseline || [];
    if (!blocks.length) return null;
    const labels = blocks.map(b => blockToTime(b));
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: ['Actual', 'Baseline'] },
      xAxis: { ...ecBase().xAxis, data: labels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [
        { name: 'Actual', type: 'line', data: actual, smooth: true, symbol: 'none', lineStyle: { width: 2.5, color: '#34D399' }, itemStyle: { color: '#34D399' },
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#34D39918' }, { offset: 1, color: 'transparent' }] } },
          markPoint: { data: [{ type: 'max', symbolSize: 32, label: { formatter: p => `${fmt(p.value, 0)} MW`, fontSize: 9, color: '#34D399' } }], itemStyle: { color: '#34D399' } },
        },
        { name: 'Baseline', type: 'line', data: baseline, smooth: true, symbol: 'none', lineStyle: { width: 1.5, color: '#ECEEF330', type: 'dashed' }, itemStyle: { color: '#ECEEF330' } },
      ],
    };
  }, [dayAheadSeries]);

  const residualBarOption = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    if (!blocks.length || !residuals.length) return null;
    const labels = blocks.slice(0, residuals.length).map(b => blockToTime(b));
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: labels, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 } },
      yAxis: { ...ecBase().yAxis, name: 'MW (Residual)' },
      series: [{ type: 'bar', data: residuals.map(v => ({ value: v, itemStyle: { color: v != null && v > 0 ? '#34D39988' : '#F8717188', borderRadius: v != null && v > 0 ? [3, 3, 0, 0] : [0, 0, 3, 3] } })), barMaxWidth: 4 }],
    };
  }, [dayAheadSeries, residuals]);

  const rampCompareOption = useMemo(() => {
    if (!rampSeries) return null;
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: ['Actual Ramp', 'Baseline Ramp'] },
      xAxis: { ...ecBase().xAxis, data: rampSeries.rampBlocks, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 7 } },
      yAxis: { ...ecBase().yAxis, name: 'MW/15m' },
      series: [
        { name: 'Actual Ramp', type: 'line', data: rampSeries.actualRamp, smooth: true, symbol: 'none', lineStyle: { color: '#34D399', width: 2 }, itemStyle: { color: '#34D399' } },
        { name: 'Baseline Ramp', type: 'line', data: rampSeries.baselineRamp, smooth: true, symbol: 'none', lineStyle: { color: '#ECEEF330', width: 1.5, type: 'dashed' }, itemStyle: { color: '#ECEEF330' } },
      ],
    };
  }, [rampSeries]);

  const histogramOption = useMemo(() => {
    const hist = buildHistogram(residuals, 10);
    if (!hist) return null;
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: hist.bins.map(b => Math.round(b.mid).toString()) },
      yAxis: { ...ecBase().yAxis, name: 'Count' },
      series: [{ type: 'bar', data: hist.bins.map(b => ({ value: b.count, itemStyle: { color: '#F07825', borderRadius: [3, 3, 0, 0] } })), barMaxWidth: 28 }],
    };
  }, [residuals]);

  const heatmapOption = useMemo(() => {
    const blocks = dayAheadSeries?.blocks || [];
    if (!blocks.length || !residuals.length) return null;
    const data = residuals.map((v, i) => Number.isFinite(v) ? [i, 0, Math.abs(v)] : null).filter(Boolean);
    if (!data.length) return null;
    const maxV = Math.max(...data.map(r => r[2]), 1);
    return {
      ...ecBase(),
      grid: { ...ecBase().grid, top: 20, bottom: 50 },
      xAxis: { ...ecBase().xAxis, data: blocks, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 5 } },
      yAxis: { type: 'category', data: ['Error'], axisLabel: { color: '#6B7186', fontSize: 9 }, axisLine: { show: false } },
      visualMap: { min: 0, max: maxV, orient: 'horizontal', left: 'center', bottom: 0, inRange: { color: ['#1A191E', '#F07825', '#F87171'] }, textStyle: { color: '#6B7186', fontSize: 9 } },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: 'rgba(0,0,0,0.4)', borderWidth: 1 } }],
    };
  }, [dayAheadSeries, residuals]);

  const peakBarOption = useMemo(() => {
    if (!peakError || !Number.isFinite(peakError.actualPeak) || !Number.isFinite(peakError.baselineAtPeak)) return null;
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: ['Baseline @ Peak', 'Actual Peak'] },
      yAxis: { ...ecBase().yAxis, scale: true, name: 'MW' },
      series: [{ type: 'bar', data: [
        { value: peakError.baselineAtPeak, itemStyle: { color: '#ECEEF320', borderRadius: [4, 4, 0, 0] } },
        { value: peakError.actualPeak, itemStyle: { color: '#34D399', borderRadius: [4, 4, 0, 0] } },
      ], barWidth: '40%' }],
    };
  }, [peakError]);

  const rampErrorOption = useMemo(() => {
    if (!rampSeries) return null;
    const absErr = rampSeries.rampError.map(v => Number.isFinite(v) ? Math.abs(v) : null);
    if (!absErr.some(Number.isFinite)) return null;
    return {
      ...ecBase(),
      xAxis: { ...ecBase().xAxis, data: rampSeries.rampBlocks, axisLabel: { ...ecBase().xAxis.axisLabel, interval: 7 } },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [{ type: 'bar', data: absErr.map(v => ({ value: v, itemStyle: { color: '#F8717188', borderRadius: [3, 3, 0, 0] } })), barMaxWidth: 4 }],
    };
  }, [rampSeries]);

  const hasData = (dayAheadSeries?.actual?.length > 0) || baselineWindowMapes.length > 0;

  const TABS = [
    { id: 'window', l: 'Window Selection' },
    { id: 'pattern', l: 'Pattern Fit' },
    { id: 'errors', l: 'Error Diagnostics' },
  ];

  /* ═══ RENDER ═══ */
  return (
    <div style={S.page}>

      {!hasData ? (
        <div style={{ padding: 60, textAlign: 'center', color: '#6B7186' }}>
          <div style={{ fontSize: 16, marginBottom: 6 }}>No optimizer data available</div>
          <div style={{ fontSize: 11 }}>Run a forecast to populate the optimizer</div>
        </div>
      ) : (
        <>
          {/* ─── TABS ─── */}
          <div style={S.tabBar}>
            {TABS.map(t => <button key={t.id} onClick={() => setActiveTab(t.id)} style={S.tab(activeTab === t.id)}>{t.l}</button>)}
          </div>

          {/* ═══ WINDOW SELECTION TAB ═══ */}
          {activeTab === 'window' && (
            <>
              {/* KPIs */}
              <div style={S.kpiRow}>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Best Window</div>
                  <div style={S.kpiVal('#F07825')}>{bestWindow ?? '--'}<span style={S.kpiUnit}> D</span></div>
                  <div style={S.kpiSub}>MAPE {bestMape != null ? fmt(bestMape) + '%' : '--'}</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Current Window</div>
                  <div style={S.kpiVal()}>{baselineDays}<span style={S.kpiUnit}> D</span></div>
                  <div style={S.kpiSub}>MAPE {currentMape != null ? fmt(currentMape) + '%' : '--'}</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>MAPE Gap</div>
                  <div style={S.kpiVal(mapeGap != null && mapeGap > 0.3 ? '#F87171' : '#34D399')}>{mapeGap != null ? (mapeGap > 0 ? '+' : '') + fmt(mapeGap) + '%' : '--'}</div>
                  <div style={S.kpiSub}>vs best window</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Stability</div>
                  <div style={S.kpiVal(isStable ? '#34D399' : '#FBBF24')}>{stabilitySpread != null ? fmt(stabilitySpread) + '%' : '--'}</div>
                  <div style={S.kpiSub}>{isStable ? 'Stable across windows' : 'Sensitive to window'}</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Action</div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#ECEEF3', marginTop: 6 }}>{windowAction}</div>
                </div>
              </div>

              {/* Accuracy Curve + Side panels */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 8, padding: '0 16px' }}>
                {/* Accuracy Chart */}
                <div style={S.card}>
                  <div style={S.cardTitle}>Accuracy Curve — MAPE vs Window Length</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {accuracyOption ? <ReactECharts option={accuracyOption} style={{ height: 360, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No accuracy data</div>}
                  </div>
                </div>

                {/* Stability + Slider */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {/* Stability Band */}
                  <div style={S.card}>
                    <div style={S.cardTitle}>Stability Band</div>
                    <div style={{ padding: 14 }}>
                      {stability ? (() => {
                        const span = Math.max(stability.max - stability.min, 1e-6);
                        const left = ((stability.q1 - stability.min) / span) * 100;
                        const width = ((stability.q3 - stability.q1) / span) * 100;
                        const median = ((stability.median - stability.min) / span) * 100;
                        return (
                          <>
                            <div style={{ position: 'relative', height: 20, background: '#2A292F', borderRadius: 10, marginBottom: 8 }}>
                              <div style={{ position: 'absolute', top: 2, height: 16, borderRadius: 8, background: '#F0782540', left: `${left}%`, width: `${width}%` }} />
                              <div style={{ position: 'absolute', top: 0, height: 20, width: 3, borderRadius: 2, background: '#F07825', left: `${median}%` }} />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#6B7186' }}>
                              <span>Min <strong style={{ color: '#ECEEF3' }}>{fmt(stability.min)}%</strong></span>
                              <span>Median <strong style={{ color: '#F07825' }}>{fmt(stability.median)}%</strong></span>
                              <span>Max <strong style={{ color: '#ECEEF3' }}>{fmt(stability.max)}%</strong></span>
                            </div>
                          </>
                        );
                      })() : <div style={{ color: '#6B7186', fontSize: 10 }}>No stability data</div>}
                    </div>
                  </div>

                  {/* Baseline Window Selector */}
                  <div style={{ ...S.card, flex: 1 }}>
                    <div style={S.cardTitle}>Baseline Window (MAPE Optimized)</div>
                    <div style={{ padding: 14, flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <input type="range" min="1" max="30" value={baselineDays} onChange={e => setBaselineDays(Number(e.target.value))}
                          style={{ flex: 1, accentColor: '#F07825' }} />
                        <span style={{ fontSize: 16, fontWeight: 700 }}>{baselineDays} Days</span>
                      </div>
                      <div style={{ fontSize: 9, color: '#6B7186' }}>
                        {currentMape != null ? `Current MAPE: ${fmt(currentMape)}%` : 'Select window to evaluate'}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {baselineWindowMapes.map(row => {
                          const wd = Number(row.window_days);
                          const active = baselineDays === wd;
                          return (
                            <button key={`bw-${row.window_days}`} onClick={() => Number.isFinite(wd) && setBaselineDays(wd)}
                              style={{
                                fontSize: 9, padding: '4px 10px', borderRadius: 14, cursor: 'pointer', fontFamily: 'inherit',
                                border: `1px solid ${active ? '#F07825' : '#2A292F'}`,
                                background: active ? '#F0782518' : 'transparent',
                                color: active ? '#F07825' : '#6B7186',
                              }}>
                              {Number.isFinite(wd) ? wd : row.window_days}D • {fmt(row.baseline_mape)}%
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ═══ PATTERN FIT TAB ═══ */}
          {activeTab === 'pattern' && (
            <>
              {/* KPIs */}
              <div style={S.kpiRow}>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Residual MAE</div>
                  <div style={S.kpiVal('#F07825')}>{residualStats ? fmt(residualStats.mae) : '--'}<span style={S.kpiUnit}> MW</span></div>
                  <div style={S.kpiSub}>Avg |actual − baseline|</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Bias Direction</div>
                  <div style={S.kpiVal(Math.abs(bias) > 10 ? '#FBBF24' : '#34D399')}>{biasDir}</div>
                  <div style={S.kpiSub}>Mean residual {bias > 0 ? '+' : ''}{fmt(bias)} MW</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Worst Window</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#F87171', marginTop: 2 }}>{worstWindow}</div>
                  <div style={S.kpiSub}>MAE {fmt(worstMAE)} MW</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Ramp Alignment</div>
                  <div style={S.kpiVal(rampMAE > 20 ? '#F87171' : '#34D399')}>{fmt(rampMAE)}<span style={S.kpiUnit}> MW</span></div>
                  <div style={S.kpiSub}>Avg ramp mismatch/blk</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Peak Timing</div>
                  <div style={S.kpiVal(peakTimingOff != null && peakTimingOff > 4 ? '#F87171' : '#34D399')}>{peakTimingOff ?? '--'}<span style={S.kpiUnit}> blks</span></div>
                  <div style={S.kpiSub}>{peakError ? `${fmt(peakError.errorMw)} MW gap` : 'Peak offset'}</div>
                </div>
              </div>

              {/* Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, padding: '0 16px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Pattern Fit — Actual vs Baseline</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {patternFitOption ? <ReactECharts option={patternFitOption} style={{ height: 320, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No pattern data</div>}
                  </div>
                </div>
                <div style={S.card}>
                  <div style={S.cardTitle}>Residual Profile (Actual − Baseline)</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {residualBarOption ? <ReactECharts option={residualBarOption} style={{ height: 320, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No residual data</div>}
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8, padding: '6px 16px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Ramp Slope Comparison — MW/15min</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {rampCompareOption ? <ReactECharts option={rampCompareOption} style={{ height: 280, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No ramp data</div>}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ═══ ERROR DIAGNOSTICS TAB ═══ */}
          {activeTab === 'errors' && (
            <>
              {/* KPIs */}
              <div style={S.kpiRow}>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Error Grade</div>
                  <div style={S.kpiVal(grade === 'A' ? '#34D399' : grade === 'B' ? '#FBBF24' : '#F87171')}>{grade}</div>
                  <div style={S.kpiSub}>MAPE {mape != null ? fmt(mape) + '%' : '--'}</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>MAE / Std</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#ECEEF3', marginTop: 2 }}>{residualStats ? fmt(residualStats.mae) : '--'} / {residualStats ? fmt(residualStats.std) : '--'}</div>
                  <div style={S.kpiSub}>MW mean abs / spread</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Outlier Blocks</div>
                  <div style={S.kpiVal(outlierPct > 5 ? '#F87171' : '#34D399')}>{outlierCount} <span style={S.kpiUnit}>({fmt(outlierPct)}%)</span></div>
                  <div style={S.kpiSub}>&gt;2σ threshold</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Systematic Bias</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: hasSystematicBias ? '#F87171' : '#34D399', marginTop: 2 }}>{hasSystematicBias ? `${maxConsec} consecutive` : 'None detected'}</div>
                  <div style={S.kpiSub}>Same-sign run ≥2hrs</div>
                </div>
                <div style={S.kpi}>
                  <div style={S.kpiLabel}>Error Hotspot</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#FBBF24', marginTop: 2 }}>{errorConc}</div>
                  <div style={S.kpiSub}>AM {fmt(morningMAE)} / PM {fmt(eveningMAE)} MW</div>
                </div>
              </div>

              {/* Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, padding: '0 16px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Residual Histogram</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {histogramOption ? <ReactECharts option={histogramOption} style={{ height: 280, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No residuals</div>}
                  </div>
                </div>
                <div style={S.card}>
                  <div style={S.cardTitle}>Block-wise Error Heatmap</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {heatmapOption ? <ReactECharts option={heatmapOption} style={{ height: 280, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No heatmap data</div>}
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, padding: '6px 16px 12px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Peak MW Error Bar</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {peakBarOption ? <ReactECharts option={peakBarOption} style={{ height: 260, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No peak data</div>}
                  </div>
                </div>
                <div style={S.card}>
                  <div style={S.cardTitle}>Ramp Error Index</div>
                  <div style={{ flex: 1, padding: 4 }}>
                    {rampErrorOption ? <ReactECharts option={rampErrorOption} style={{ height: 260, width: '100%' }} notMerge /> : <div style={{ padding: 40, textAlign: 'center', color: '#6B7186' }}>No ramp error data</div>}
                  </div>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
