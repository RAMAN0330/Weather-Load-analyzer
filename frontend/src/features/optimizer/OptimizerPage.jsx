import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import HorizonToggle from '../../components/HorizonToggle';


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
  page: { fontFamily: "'IBM Plex Mono', monospace", color: '#ECEEF3', minHeight: 0, height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: 14, padding: '8px 0 0', position: 'relative' },
  kpiRow: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, padding: '0 16px', flexShrink: 0 },
  kpi: { minWidth: 0, minHeight: 132, padding: '16px 18px', background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))', borderRadius: 14, border: '1px solid #2A292F', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)' },
  kpiLabel: { fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: '#6B7186', marginBottom: 8 },
  kpiVal: (c) => ({ fontSize: 30, fontWeight: 700, color: c || '#ECEEF3', lineHeight: 1.05 }),
  kpiUnit: { fontSize: 11, fontWeight: 400, opacity: 0.5 },
  kpiSub: { fontSize: 10, color: '#6B7186', marginTop: 6, lineHeight: 1.45 },
  card: { background: 'linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))', borderRadius: 16, border: '1px solid #2A292F', overflow: 'hidden', display: 'flex', flexDirection: 'column', boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)' },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.2, color: '#A0A5B8', padding: '14px 16px', borderBottom: '1px solid #2A292F', textTransform: 'uppercase' },
  tabBar: { display: 'inline-flex', gap: 4, padding: '5px', background: '#141419', border: '1px solid #2A292F', borderRadius: 999, flexWrap: 'wrap' },
  tab: (active) => ({ flex: '0 0 auto', padding: '8px 16px', fontSize: 10, fontWeight: 600, letterSpacing: 0.5, cursor: 'pointer', border: '1px solid transparent', fontFamily: 'inherit', borderRadius: 999, background: active ? '#F0782518' : 'transparent', color: active ? '#F07825' : '#A0A5B8', transition: 'all 0.15s ease', display: 'flex', alignItems: 'center', gap: 6 }),
  badge: (color) => ({ fontSize: 8, fontWeight: 700, padding: '2px 8px', borderRadius: 12, background: `${color}18`, color }),
  sectionLabel: (color) => ({ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 2, color: color || '#F07825', padding: '8px 16px 2px' }),
  th: { fontSize: 8, color: '#6B7186', textTransform: 'uppercase', letterSpacing: 1, padding: '10px 12px', borderBottom: '1px solid #2A292F', textAlign: 'left', fontWeight: 600 },
  td: { fontSize: 10, padding: '8px 12px', borderBottom: '1px solid #2A292F22' },
};

/* ─── ECharts base ─── */
const ecBase = () => ({
  backgroundColor: 'transparent',
  textStyle: { color: '#A0A5B8', fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 },
  grid: { top: 44, right: 24, bottom: 38, left: 56, containLabel: false },
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
  horizon = 't1',
  setHorizon,
  t2Date,
}) {
  const [activeTab, setActiveTab] = useState('window');
  const [windowOverlayPanel, setWindowOverlayPanel] = useState(null);
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
  const windowRows = useMemo(() => (
    baselineWindowMapes
      .map((row) => ({
        window: Number(row.window_days),
        mape: Number(row.baseline_mape),
      }))
      .filter((row) => Number.isFinite(row.window) && Number.isFinite(row.mape))
      .sort((a, b) => a.mape - b.mape)
      .map((row, index) => ({ ...row, rank: index + 1 }))
  ), [baselineWindowMapes]);
  const currentWindowRow = useMemo(() => (
    windowRows.find((row) => row.window === Number(baselineDays)) || null
  ), [windowRows, baselineDays]);
  const currentRank = currentWindowRow?.rank ?? null;
  const nearOptimalRows = useMemo(() => (
    windowRows.filter((row) => bestMape != null && (row.mape - bestMape) <= 0.2)
  ), [windowRows, bestMape]);
  const topClusterLabel = nearOptimalRows.length
    ? `${nearOptimalRows[0].window}D-${nearOptimalRows[nearOptimalRows.length - 1].window}D`
    : '--';
  const currentVsMedian = currentMape != null && stability?.median != null ? Number(currentMape) - stability.median : null;
  const selectionConfidence = currentRank === 1
    ? 'Primary winner'
    : mapeGap != null && mapeGap <= 0.2 && isStable
      ? 'Safe alternate'
      : mapeGap != null && mapeGap <= 0.4
        ? 'Usable with caution'
        : 'Revisit selection';
  const selectionSignals = useMemo(() => {
    const signals = [];
    if (bestWindow != null && bestMape != null) {
      signals.push({
        label: 'Lowest MAPE',
        value: `${bestWindow}D at ${fmt(bestMape)}%`,
        tone: '#34D399',
      });
    }
    if (topClusterLabel !== '--') {
      signals.push({
        label: 'Near-optimal band',
        value: `${nearOptimalRows.length} windows in ${topClusterLabel}`,
        tone: '#FBBF24',
      });
    }
    if (currentRank != null) {
      signals.push({
        label: 'Current rank',
        value: `#${currentRank} of ${windowRows.length}`,
        tone: currentRank <= 3 ? '#34D399' : '#F87171',
      });
    }
    if (stabilitySpread != null) {
      signals.push({
        label: 'Spread across windows',
        value: `${fmt(stabilitySpread)}%`,
        tone: isStable ? '#34D399' : '#FBBF24',
      });
    }
    if (currentVsMedian != null) {
      signals.push({
        label: 'Current vs median',
        value: `${currentVsMedian > 0 ? '+' : ''}${fmt(currentVsMedian)}%`,
        tone: currentVsMedian <= 0 ? '#34D399' : '#F87171',
      });
    }
    return signals;
  }, [bestWindow, bestMape, topClusterLabel, nearOptimalRows.length, currentRank, windowRows.length, stabilitySpread, isStable, currentVsMedian]);

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

  const windowGapOption = useMemo(() => {
    if (!windowRows.length || bestMape == null) return null;
    return {
      ...ecBase(),
      grid: { ...ecBase().grid, top: 28, right: 14, bottom: 28, left: 42, containLabel: true },
      xAxis: {
        ...ecBase().xAxis,
        data: windowRows.map((row) => `${row.window}D`),
        axisLabel: { ...ecBase().xAxis.axisLabel, interval: 0, rotate: 0, fontSize: 8 },
      },
      yAxis: { ...ecBase().yAxis, name: 'Gap %' },
      series: [{
        type: 'bar',
        data: windowRows.map((row) => ({
          value: Math.max(0, row.mape - bestMape),
          itemStyle: {
            color: row.window === Number(baselineDays) ? '#F07825' : row.rank <= 3 ? '#34D39988' : '#6B718688',
            borderRadius: [3, 3, 0, 0],
          },
        })),
        barMaxWidth: 18,
      }],
    };
  }, [windowRows, bestMape, baselineDays]);

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
      {/* ─── HORIZON TOGGLE ─── */}
      {setHorizon && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 16px 0', flexShrink: 0, zIndex: 10 }}>
          <span style={{ fontSize: 10, color: '#6B7186', letterSpacing: 1, textTransform: 'uppercase' }}>
            Viewing: <strong style={{ color: '#ECEEF3' }}>{horizon === 't2' ? `T+2 · ${t2Date}` : `T+1 · ${effectiveDate}`}</strong>
          </span>
          <HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date} />
        </div>
      )}

      {!hasData ? (
        <div style={{ padding: 60, textAlign: 'center', color: '#6B7186' }}>
          <div style={{ fontSize: 16, marginBottom: 6 }}>No optimizer data available</div>
          <div style={{ fontSize: 11 }}>Run a forecast to populate the optimizer</div>
        </div>
      ) : (
        <>
          <div style={S.kpiRow}>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Selected Window</div>
              <div style={S.kpiVal()}>{baselineDays}<span style={S.kpiUnit}> D</span></div>
              <div style={S.kpiSub}>
                {currentMape != null ? `MAPE ${fmt(currentMape)}% on active baseline` : 'Active optimizer baseline length'}
              </div>
            </div>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Gap To Best</div>
              <div style={S.kpiVal(mapeGap != null && mapeGap > 0.3 ? '#F87171' : '#34D399')}>
                {mapeGap != null ? (mapeGap > 0 ? '+' : '') + fmt(mapeGap) : '--'}<span style={S.kpiUnit}> %</span>
              </div>
              <div style={S.kpiSub}>
                {currentRank != null ? `Rank #${currentRank} of ${windowRows.length || '--'} across evaluated windows` : 'Distance from the best-ranked window'}
              </div>
            </div>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Residual MAE</div>
              <div style={S.kpiVal((residualStats?.mae ?? Infinity) <= 20 ? '#34D399' : (residualStats?.mae ?? Infinity) <= 35 ? '#FBBF24' : '#F87171')}>
                {residualStats ? fmt(residualStats.mae) : '--'}<span style={S.kpiUnit}> MW</span>
              </div>
              <div style={S.kpiSub}>Avg |actual − baseline| across observed blocks</div>
            </div>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Peak Timing</div>
              <div style={S.kpiVal(peakTimingOff != null && peakTimingOff > 4 ? '#F87171' : '#34D399')}>
                {peakTimingOff ?? '--'}<span style={S.kpiUnit}> blks</span>
              </div>
              <div style={S.kpiSub}>{peakError ? `${fmt(peakError.errorMw)} MW gap at actual peak` : 'Peak alignment vs baseline'}</div>
            </div>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Error Grade</div>
              <div style={S.kpiVal(grade === 'A' ? '#34D399' : grade === 'B' ? '#FBBF24' : '#F87171')}>
                {grade}
              </div>
              <div style={S.kpiSub}>{mape != null ? `Overall MAPE ${fmt(mape)}%` : 'Forecast quality score'}</div>
            </div>
            <div style={S.kpi}>
              <div style={S.kpiLabel}>Outlier Blocks</div>
              <div style={S.kpiVal(outlierPct > 5 ? '#F87171' : '#34D399')}>
                {outlierCount}<span style={S.kpiUnit}> ({fmt(outlierPct)}%)</span>
              </div>
              <div style={S.kpiSub}>{hasSystematicBias ? `${maxConsec} consecutive same-sign blocks` : 'No major outlier concentration'}</div>
            </div>
          </div>

          {/* ─── TABS ─── */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '0 16px', flexWrap: 'wrap', flexShrink: 0 }}>
            <div style={S.tabBar}>
              {TABS.map(t => <button key={t.id} onClick={() => setActiveTab(t.id)} style={S.tab(activeTab === t.id)}>{t.l}</button>)}
            </div>
            {activeTab === 'window' && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end', alignItems: 'center' }}>
                {[
                  { panel: 'rationale', label: 'Selection Rationale', sub: 'Why this window was chosen', color: '#F07825' },
                  { panel: 'signals', label: 'Selection Signals', sub: 'Load · Weather · Price signals', color: '#5B9FE4' },
                  { panel: 'leaderboard', label: 'Window Leaderboard', sub: 'Ranked window candidates', color: '#34D399' },
                  { panel: 'gap', label: 'Gap Profile', sub: 'Supply-demand gap view', color: '#C084FC' },
                ].map(({ panel, label, sub, color }) => (
                  <button
                    key={panel}
                    type="button"
                    onClick={() => setWindowOverlayPanel(panel)}
                    style={{
                      display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 5,
                      minWidth: 130, padding: '10px 14px', borderRadius: 14,
                      border: `1px solid ${windowOverlayPanel === panel ? `${color}66` : `${color}33`}`,
                      background: windowOverlayPanel === panel ? `${color}18` : `${color}10`,
                      color: '#ECEEF3', cursor: 'pointer', fontFamily: 'inherit',
                      textAlign: 'left', transition: 'all 0.15s ease',
                    }}
                  >
                    <span style={{ fontSize: 8, letterSpacing: 1.1, textTransform: 'uppercase', color: '#6B7186' }}>Quick panel</span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: windowOverlayPanel === panel ? color : color }}>{label}</span>
                    <span style={{ fontSize: 9, color: '#6B7186' }}>{sub}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* ═══ WINDOW SELECTION TAB ═══ */}
          {activeTab === 'window' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1, minHeight: 0, overflow: 'hidden' }}>
              {/* Accuracy Curve + Side panels */}
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.32fr) minmax(360px, 0.92fr)', gap: 12, padding: '0 16px 16px', flex: 1, minHeight: 0, alignItems: 'stretch', overflow: 'hidden' }}>
                {/* Accuracy Chart */}
                <div style={{ ...S.card, minHeight: 0 }}>
                  <div style={S.cardTitle}>Accuracy Curve — MAPE vs Window Length</div>
                  <div style={{ flex: 1, padding: 10, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    {accuracyOption ? <ReactECharts option={accuracyOption} style={{ flex: 1, minHeight: 0, height: '100%', width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No accuracy data</div>}
                  </div>
                </div>

                {/* Stability + Slider */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0, overflow: 'auto', flex: 1 }}>
                  {/* Stability Band */}
                  <div style={S.card}>
                    <div style={S.cardTitle}>Stability Band</div>
                    <div style={{ padding: 18 }}>
                      {stability ? (() => {
                        const span = Math.max(stability.max - stability.min, 1e-6);
                        const left = ((stability.q1 - stability.min) / span) * 100;
                        const width = ((stability.q3 - stability.q1) / span) * 100;
                        const median = ((stability.median - stability.min) / span) * 100;
                        return (
                          <>
                            <div style={{ position: 'relative', height: 22, background: '#2A292F', borderRadius: 999, marginBottom: 10 }}>
                              <div style={{ position: 'absolute', top: 3, height: 16, borderRadius: 999, background: '#F0782540', left: `${left}%`, width: `${width}%` }} />
                              <div style={{ position: 'absolute', top: 0, height: 22, width: 4, borderRadius: 999, background: '#F07825', left: `${median}%` }} />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#6B7186' }}>
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
                  <div style={{ ...S.card, flex: 1, minHeight: 0 }}>
                    <div style={S.cardTitle}>Baseline Window (MAPE Optimized)</div>
                    <div style={{ padding: '12px 16px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 10, justifyContent: 'space-between' }}>
                      {/* Slider row */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <input type="range" min="1" max="30" value={baselineDays} onChange={e => setBaselineDays(Number(e.target.value))}
                          style={{ flex: 1, accentColor: '#F07825' }} />
                        <span style={{ fontSize: 22, fontWeight: 700, minWidth: 70, textAlign: 'right' }}>{baselineDays} Days</span>
                      </div>
                      {/* MAPE label */}
                      <div style={{ fontSize: 10, color: '#6B7186', marginTop: -4 }}>
                        {currentMape != null ? `Current MAPE: ${fmt(currentMape)}%` : 'Select window to evaluate'}
                      </div>
                      {/* Info cards */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, flex: 1, minHeight: 0 }}>
                        <div style={{ background: '#201F25', border: '1px solid #2A292F', borderRadius: 12, padding: '10px 12px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                          <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: 'uppercase', color: '#6B7186', marginBottom: 6 }}>Why it matters</div>
                          <div style={{ fontSize: 10, color: '#ECEEF3', lineHeight: 1.5 }}>
                            Selection balances lowest error, cluster stability, and how far the current window sits from the best performer.
                          </div>
                        </div>
                        <div style={{ background: '#201F25', border: '1px solid #2A292F', borderRadius: 12, padding: '10px 12px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                          <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: 'uppercase', color: '#6B7186', marginBottom: 6 }}>Current posture</div>
                          <div style={{ fontSize: 14, fontWeight: 700, color: currentRank != null && currentRank <= 3 ? '#34D399' : '#FBBF24' }}>
                            {selectionConfidence}
                          </div>
                          <div style={{ fontSize: 9, color: '#A0A5B8', marginTop: 4 }}>{windowAction}</div>
                        </div>
                      </div>
                      {/* Day buttons */}
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, flex: 1, alignContent: 'flex-start' }}>
                        {baselineWindowMapes.map(row => {
                          const wd = Number(row.window_days);
                          const active = baselineDays === wd;
                          return (
                            <button key={`bw-${row.window_days}`} onClick={() => Number.isFinite(wd) && setBaselineDays(wd)}
                              style={{
                                fontSize: 11, fontWeight: 600, padding: '9px 14px', borderRadius: 999, cursor: 'pointer', fontFamily: 'inherit',
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
            </div>
          )}

          {/* ═══ PATTERN FIT TAB ═══ */}
          {activeTab === 'pattern' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1, minHeight: 0, overflow: 'hidden' }}>
              {/* Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: '0 16px 16px', flex: 1, minHeight: 0, alignItems: 'stretch', overflow: 'hidden' }}>
                <div style={{ ...S.card, minHeight: 0 }}>
                  <div style={S.cardTitle}>Pattern Fit — Actual vs Baseline</div>
                  <div style={{ flex: 1, padding: 10, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    {patternFitOption ? <ReactECharts option={patternFitOption} style={{ flex: 1, minHeight: 0, height: '100%', width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No pattern data</div>}
                  </div>
                </div>
                <div style={{ ...S.card, minHeight: 0 }}>
                  <div style={S.cardTitle}>Residual Profile (Actual − Baseline)</div>
                  <div style={{ flex: 1, padding: 10, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    {residualBarOption ? <ReactECharts option={residualBarOption} style={{ flex: 1, minHeight: 0, height: '100%', width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No residual data</div>}
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12, padding: '0 16px 6px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Ramp Slope Comparison — MW/15min</div>
                  <div style={{ flex: 1, padding: 10 }}>
                    {rampCompareOption ? <ReactECharts option={rampCompareOption} style={{ height: 300, width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No ramp data</div>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ═══ ERROR DIAGNOSTICS TAB ═══ */}
          {activeTab === 'errors' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {/* Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: '0 16px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Residual Histogram</div>
                  <div style={{ flex: 1, padding: 10 }}>
                    {histogramOption ? <ReactECharts option={histogramOption} style={{ height: 300, width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No residuals</div>}
                  </div>
                </div>
                <div style={S.card}>
                  <div style={S.cardTitle}>Block-wise Error Heatmap</div>
                  <div style={{ flex: 1, padding: 10 }}>
                    {heatmapOption ? <ReactECharts option={heatmapOption} style={{ height: 300, width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No heatmap data</div>}
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: '0 16px 8px' }}>
                <div style={S.card}>
                  <div style={S.cardTitle}>Peak MW Error Bar</div>
                  <div style={{ flex: 1, padding: 10 }}>
                    {peakBarOption ? <ReactECharts option={peakBarOption} style={{ height: 280, width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No peak data</div>}
                  </div>
                </div>
                <div style={S.card}>
                  <div style={S.cardTitle}>Ramp Error Index</div>
                  <div style={{ flex: 1, padding: 10 }}>
                    {rampErrorOption ? <ReactECharts option={rampErrorOption} style={{ height: 280, width: '100%' }} notMerge /> : <div style={{ padding: 48, textAlign: 'center', color: '#6B7186' }}>No ramp error data</div>}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {windowOverlayPanel && (
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(7, 7, 10, 0.56)', backdropFilter: 'blur(4px)', display: 'flex', justifyContent: 'center', alignItems: 'center', padding: 20 }}>
          <div style={{ width: 'min(920px, 100%)', maxHeight: '85vh', background: '#16161B', border: '1px solid #2A292F', borderRadius: 16, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, padding: '16px 18px', borderBottom: '1px solid #2A292F' }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase', color: '#A0A5B8' }}>
                  {windowOverlayPanel === 'signals' && 'Selection Signals'}
                  {windowOverlayPanel === 'leaderboard' && 'Window Leaderboard'}
                  {windowOverlayPanel === 'gap' && 'Gap Profile'}
                  {windowOverlayPanel === 'rationale' && 'Selection Rationale'}
                </div>
                <div style={{ fontSize: 11, color: '#6B7186', marginTop: 4 }}>
                  {windowOverlayPanel === 'signals' && `Signals used to explain the ${baselineDays}D choice on ${dateStr}.`}
                  {windowOverlayPanel === 'leaderboard' && `Full ranking of evaluated windows for ${dateStr}, ordered by baseline MAPE.`}
                  {windowOverlayPanel === 'gap' && `Window-to-window distance from the best performer on ${dateStr}.`}
                  {windowOverlayPanel === 'rationale' && `Why the optimizer prefers this window on ${dateStr}: error rank, near-optimal band, and stability across candidate windows.`}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setWindowOverlayPanel(null)}
                style={{ width: 30, height: 30, borderRadius: 999, border: '1px solid #2A292F', background: '#1A191E', color: '#A0A5B8', cursor: 'pointer', fontFamily: 'inherit' }}
              >
                x
              </button>
            </div>

            <div style={{ padding: 18, overflowY: 'auto' }}>
              {windowOverlayPanel === 'signals' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div style={S.card}>
                    <div style={S.cardTitle}>Selection Signals</div>
                    <div style={{ padding: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      {selectionSignals.map((signal) => (
                        <div key={`signal-${signal.label}`} style={{ background: '#201F25', border: '1px solid #2A292F', borderRadius: 8, padding: '10px 12px' }}>
                          <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: 'uppercase', color: '#6B7186', marginBottom: 6 }}>{signal.label}</div>
                          <div style={{ fontSize: 12, fontWeight: 700, color: signal.tone }}>{signal.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div style={S.card}>
                      <div style={S.cardTitle}>Decision Summary</div>
                      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10, fontSize: 10, color: '#A0A5B8' }}>
                        <div><strong style={{ color: '#ECEEF3' }}>Current selection:</strong> {baselineDays}D</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Confidence:</strong> {selectionConfidence}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Near-optimal band:</strong> {topClusterLabel}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Spread across windows:</strong> {stabilitySpread != null ? `${fmt(stabilitySpread)}%` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Current vs median:</strong> {currentVsMedian != null ? `${currentVsMedian > 0 ? '+' : ''}${fmt(currentVsMedian)}%` : '--'}</div>
                      </div>
                    </div>
                    <div style={S.card}>
                      <div style={S.cardTitle}>Gap to Best</div>
                      <div style={{ flex: 1, padding: 4 }}>
                        {windowGapOption ? <ReactECharts option={windowGapOption} style={{ height: 220, width: '100%' }} notMerge /> : <div style={{ padding: 30, textAlign: 'center', color: '#6B7186' }}>No ranking data</div>}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {(windowOverlayPanel === 'leaderboard' || windowOverlayPanel === 'rationale') && (
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 12 }}>
                  <div style={S.card}>
                    <div style={S.cardTitle}>Full Window Ranking</div>
                    <div style={{ padding: 0 }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            <th style={S.th}>Rank</th>
                            <th style={S.th}>Window</th>
                            <th style={S.th}>MAPE</th>
                            <th style={S.th}>Gap to Best</th>
                            <th style={S.th}>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {windowRows.map((row) => {
                            const gapToBest = bestMape != null ? row.mape - bestMape : null;
                            const status = row.rank === 1 ? 'Best' : gapToBest != null && gapToBest <= 0.2 ? 'Near-optimal' : row.window === Number(baselineDays) ? 'Selected' : 'Fallback';
                            return (
                              <tr key={`overlay-${row.window}`}>
                                <td style={S.td}>#{row.rank}</td>
                                <td style={{ ...S.td, fontWeight: 700, color: row.window === Number(baselineDays) ? '#F07825' : '#ECEEF3' }}>{row.window}D</td>
                                <td style={S.td}>{fmt(row.mape)}%</td>
                                <td style={{ ...S.td, color: row.rank === 1 || (gapToBest != null && gapToBest <= 0.2) ? '#34D399' : '#6B7186' }}>{gapToBest != null ? `+${fmt(gapToBest)}%` : '--'}</td>
                                <td style={{ ...S.td, color: status === 'Best' ? '#34D399' : status === 'Selected' ? '#F07825' : '#A0A5B8', fontWeight: 600 }}>{status}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div style={S.card}>
                      <div style={S.cardTitle}>Why This Window</div>
                      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10, fontSize: 10, color: '#A0A5B8' }}>
                        <div><strong style={{ color: '#ECEEF3' }}>Current selection:</strong> {baselineDays}D</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Current rank:</strong> {currentRank != null ? `#${currentRank}` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>MAPE gap to best:</strong> {mapeGap != null ? `${mapeGap > 0 ? '+' : ''}${fmt(mapeGap)}%` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Near-optimal band:</strong> {topClusterLabel}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Stability spread:</strong> {stabilitySpread != null ? `${fmt(stabilitySpread)}%` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Decision:</strong> {windowAction}</div>
                      </div>
                    </div>
                    {windowOverlayPanel === 'rationale' && (
                      <div style={S.card}>
                        <div style={S.cardTitle}>Gap to Best</div>
                        <div style={{ flex: 1, padding: 4 }}>
                          {windowGapOption ? <ReactECharts option={windowGapOption} style={{ height: 220, width: '100%' }} notMerge /> : <div style={{ padding: 30, textAlign: 'center', color: '#6B7186' }}>No ranking data</div>}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {windowOverlayPanel === 'gap' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 300px', gap: 12 }}>
                  <div style={S.card}>
                    <div style={S.cardTitle}>Gap to Best by Window</div>
                    <div style={{ flex: 1, padding: 4 }}>
                      {windowGapOption ? <ReactECharts option={windowGapOption} style={{ height: 320, width: '100%' }} notMerge /> : <div style={{ padding: 30, textAlign: 'center', color: '#6B7186' }}>No ranking data</div>}
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div style={S.card}>
                      <div style={S.cardTitle}>Gap Readout</div>
                      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10, fontSize: 10, color: '#A0A5B8' }}>
                        <div><strong style={{ color: '#ECEEF3' }}>Best window:</strong> {bestWindow != null ? `${bestWindow}D` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Selected window:</strong> {baselineDays}D</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Gap to best:</strong> {mapeGap != null ? `${mapeGap > 0 ? '+' : ''}${fmt(mapeGap)}%` : '--'}</div>
                        <div><strong style={{ color: '#ECEEF3' }}>Cluster band:</strong> {topClusterLabel}</div>
                      </div>
                    </div>
                    <div style={S.card}>
                      <div style={S.cardTitle}>Top Windows</div>
                      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {windowRows.slice(0, 5).map((row) => (
                          <div key={`gap-top-${row.window}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, fontSize: 10 }}>
                            <span style={{ color: '#ECEEF3', fontWeight: 700 }}>{row.window}D</span>
                            <span style={{ color: '#A0A5B8' }}>{fmt(row.mape)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
