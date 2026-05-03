import axios from 'axios';
import React, { useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
  EmptyState as VpEmptyState,
} from '../../components/page/PagePrimitives.jsx';

const fmt = (v, d = 1) =>
  v == null || !Number.isFinite(v)
    ? '--'
    : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d });

const blockToTime = (b) => {
  const h = Math.floor((b - 1) / 4);
  const m = ((b - 1) % 4) * 15;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
};

const blocks96 = Array.from({ length: 96 }, (_, i) => i + 1);
const timeLabels = blocks96.map(blockToTime);

const DETAIL_PANELS = [
  { id: 'structural', label: 'Structural Comparison' },
  { id: 'similar', label: 'Similar Historical Days' },
  { id: 'fingerprint', label: 'Load Shape Fingerprint' },
];

const S = {
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: '#ECEEF3',
    minHeight: 0,
    height: '100%',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    gap: 0,
    padding: '10px 12px 0',
  },
  card: {
    background: 'linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))',
    borderRadius: 16,
    border: '1px solid #2A292F',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 0.6,
    color: '#B8BDCC',
    textTransform: 'uppercase',
  },
  badge: (color) => ({
    fontSize: 11,
    fontWeight: 700,
    padding: '4px 10px',
    borderRadius: 999,
    background: `${color}18`,
    color,
    border: `1px solid ${color}33`,
  }),
  th: {
    fontSize: 11,
    color: '#8A90A6',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    padding: '10px 12px',
    borderBottom: '1px solid #2A292F',
    textAlign: 'right',
    fontWeight: 600,
  },
  td: {
    fontSize: 13,
    padding: '10px 12px',
    borderBottom: '1px solid rgba(42, 41, 47, 0.7)',
    textAlign: 'right',
    color: '#F0F2F8',
  },
  tabBar: {
    display: 'inline-flex',
    gap: 4,
    padding: '5px',
    background: '#141419',
    border: '1px solid #2A292F',
    borderRadius: 999,
    flexWrap: 'wrap',
  },
  tab: (active) => ({
    padding: '7px 14px',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    border: '1px solid transparent',
    fontFamily: 'inherit',
    borderRadius: 999,
    background: active ? 'rgba(240, 120, 37, 0.14)' : 'transparent',
    color: active ? '#F07825' : '#B8BDCC',
    transition: 'all 0.15s ease',
  }),
  actionBtn: (active) => ({
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 4,
    minWidth: 180,
    padding: '10px 12px',
    borderRadius: 14,
    border: `1px solid ${active ? '#F0782566' : '#2A292F'}`,
    background: active ? 'rgba(240, 120, 37, 0.12)' : 'rgba(255, 255, 255, 0.02)',
    color: active ? '#F07825' : '#ECEEF3',
    cursor: 'pointer',
    fontFamily: 'inherit',
    textAlign: 'left',
    transition: 'all 0.15s ease',
  }),
  miniLabel: {
    fontSize: 11,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    color: '#8A90A6',
  },
};

const ecBase = () => ({
  backgroundColor: 'transparent',
  textStyle: { color: '#8a90a6', fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 },
  grid: { top: 36, right: 20, bottom: 36, left: 55 },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(22,22,27,0.96)',
    borderColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1,
    textStyle: { color: '#f0f2f8', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" },
    extraCssText: 'box-shadow: 0 8px 24px rgba(0,0,0,0.4); border-radius: 8px; padding: 8px 12px;',
    confine: true,
  },
  legend: {
    textStyle: { color: '#8a90a6', fontSize: 11 },
    top: 4,
    right: 8,
    itemWidth: 12,
    itemHeight: 3,
  },
  xAxis: {
    type: 'category',
    axisLine: { lineStyle: { color: '#2a292f' } },
    axisTick: { show: false },
    axisLabel: { color: '#555a6e', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#555a6e', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" },
    splitLine: { lineStyle: { color: '#1e1d23', type: 'dashed', width: 1 } },
  },
});

const safeAverage = (values = []) => {
  const clean = values.filter((value) => Number.isFinite(value));
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
};

const KpiCard = ({ eyebrow, title, value, unit, tone = '#ECEEF3', detail, footer }) => (
  <div
    className="kpi-card"
    style={{
      ...S.card,
      padding: '10px 14px',
      gap: 4,
      minHeight: 88,
      justifyContent: 'space-between',
      background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))',
    }}
  >
    <div style={S.miniLabel}>{eyebrow}</div>
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#A0A5B8', marginBottom: 8 }}>
        {title}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 22, lineHeight: 1, fontWeight: 700, color: tone, fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.4px', fontFamily: "'IBM Plex Mono', monospace" }}>{value}</span>
        {unit ? <span style={{ fontSize: 11, color: '#6B7186' }}>{unit}</span> : null}
      </div>
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {detail ? <div style={{ fontSize: 12, color: '#F0F2F8' }}>{detail}</div> : null}
      {footer ? <div style={{ fontSize: 11, color: '#8A90A6' }}>{footer}</div> : null}
    </div>
  </div>
);

export default function LoadAnalysisPage({
  effectiveDate,
  benchmarkData,
  momentumChange,
  multiDaySeries,
  multiSelectedDates,
  setMultiSelectedDates,
  detectedSeason,
  liveData,
  dayAheadData,
  apiUrl,
  horizon = 't1',
  setHorizon,
  t2Date,
  liveT2,
}) {
  // When T+2 is active, use T+2 live series for forecast/regime analysis
  const activeLiveData = horizon === 't2' && liveT2 ? liveT2 : liveData;
  const [mainTab, setMainTab] = useState('benchmark');
  const [compareDates, setCompareDates] = useState(multiSelectedDates || []);
  const [overlayPanel, setOverlayPanel] = useState(null);
  const [localMomentum, setLocalMomentum] = useState(null);
  const [localMultiSeries, setLocalMultiSeries] = useState({});
  const [selfBenchmark, setSelfBenchmark] = useState(null);
  const [selfFetching, setSelfFetching] = useState(false);

  // Self-fetch benchmark data when parent hasn't provided it yet
  useEffect(() => {
    if (benchmarkData || selfBenchmark || selfFetching || !effectiveDate) return;
    setSelfFetching(true);
    axios
      .post(apiUrl('/v2/load_benchmarks'), { date: effectiveDate })
      .then((res) => setSelfBenchmark(res.data))
      .catch((e) => console.warn('load_benchmarks self-fetch failed:', e?.message))
      .finally(() => setSelfFetching(false));
  }, [effectiveDate, benchmarkData, selfBenchmark, selfFetching, apiUrl]);

  const resolvedBenchmark = benchmarkData || selfBenchmark;

  useEffect(() => {
    setCompareDates(multiSelectedDates || []);
  }, [multiSelectedDates]);

  useEffect(() => {
    setLocalMultiSeries(multiDaySeries || {});
  }, [multiDaySeries]);

  const today = useMemo(
    () => (resolvedBenchmark?.today || []).map(Number).filter(Number.isFinite),
    [resolvedBenchmark]
  );
  const t1 = useMemo(
    () => (resolvedBenchmark?.t1 || []).map(Number).filter(Number.isFinite),
    [resolvedBenchmark]
  );
  const t7 = useMemo(
    () => (resolvedBenchmark?.t7 || []).map(Number).filter(Number.isFinite),
    [resolvedBenchmark]
  );
  const t365 = useMemo(
    () => (resolvedBenchmark?.t365 || []).map(Number).filter(Number.isFinite),
    [resolvedBenchmark]
  );

  const peak = today.length ? Math.max(...today) : 0;
  const valley = today.length ? Math.min(...today.filter((v) => v > 0)) : 0;
  const yPeak = t1.length ? Math.max(...t1) : 0;
  const yValley = t1.length ? Math.min(...t1.filter((v) => v > 0)) : 0;
  const peakIdx = today.indexOf(peak);
  const yPeakIdx = t1.indexOf(yPeak);
  const energy = today.reduce((s, v) => s + v * 0.25, 0);
  const yEnergy = t1.reduce((s, v) => s + v * 0.25, 0);
  const avgLoad = today.length ? today.reduce((s, v) => s + v, 0) / today.length : 0;
  const yAvgLoad = t1.length ? t1.reduce((s, v) => s + v, 0) / t1.length : 0;
  const loadFactor = peak > 0 ? (avgLoad / peak) * 100 : 0;
  const yLoadFactor = yPeak > 0 ? (yAvgLoad / yPeak) * 100 : 0;

  const deviation = useMemo(() => today.map((v, i) => (t1[i] ? v - t1[i] : 0)), [today, t1]);
  const absDeviation = deviation.map(Math.abs);
  const maxDevBlock = absDeviation.indexOf(Math.max(...absDeviation));
  const avgDev = absDeviation.length
    ? absDeviation.reduce((s, v) => s + v, 0) / absDeviation.length
    : 0;
  const sortedLoad = useMemo(() => [...today].sort((a, b) => b - a), [today]);

  const hourlyAvg = useMemo(() => {
    const hrs = Array.from({ length: 24 }, () => []);
    today.forEach((v, i) => hrs[Math.floor(i / 4)].push(v));
    return hrs.map((arr) => safeAverage(arr));
  }, [today]);

  const yHourlyAvg = useMemo(() => {
    const hrs = Array.from({ length: 24 }, () => []);
    t1.forEach((v, i) => hrs[Math.floor(i / 4)].push(v));
    return hrs.map((arr) => safeAverage(arr));
  }, [t1]);

  const ramps = useMemo(
    () => (today.length > 1 ? today.slice(1).map((v, i) => v - today[i]) : []),
    [today]
  );
  const yRamps = useMemo(() => (t1.length > 1 ? t1.slice(1).map((v, i) => v - t1[i]) : []), [t1]);
  const maxRampUp = ramps.length ? Math.max(...ramps) : 0;
  const maxRampDown = ramps.length ? Math.min(...ramps) : 0;
  const yMaxRampUp = yRamps.length ? Math.max(...yRamps) : 0;
  const yMaxRampDown = yRamps.length ? Math.min(...yRamps) : 0;

  const dayRegime = useMemo(() => {
    const fc = activeLiveData?.series?.forecast || [];
    const bl = activeLiveData?.series?.hybrid_baseline || [];
    if (!fc.length || !bl.length) {
      const d = new Date(horizon === 't2' && t2Date ? t2Date : effectiveDate).getDay();
      return d === 0 || d === 6 ? 'Weekend' : 'Weekday';
    }
    const r =
      fc.reduce((a, b) => a + b, 0) /
      Math.max(
        bl.reduce((a, b) => a + b, 0),
        1
      );
    if (activeLiveData?.metadata?.holiday_flags?.is_holiday) return 'Holiday';
    if (r > 1.15) return 'High Load';
    if (r < 0.85) return 'Low Demand';
    const d = new Date(horizon === 't2' && t2Date ? t2Date : effectiveDate).getDay();
    return d === 0 || d === 6 ? 'Weekend' : 'Weekday';
  }, [activeLiveData, effectiveDate, horizon, t2Date]);

  const normalizedMomentum = useMemo(() => {
    const source = localMomentum || momentumChange;
    if (Array.isArray(source)) return { type: 'daily', data: source };
    if (source?.type === 'daily' && Array.isArray(source.data)) return source;
    if (Array.isArray(source?.data)) return { type: source.type || 'daily', data: source.data };
    return { type: 'daily', data: [] };
  }, [localMomentum, momentumChange]);

  const trend = useMemo(() => {
    if (!normalizedMomentum.data.length) return { dir: '--', val: 0, data: [] };
    const recent = normalizedMomentum.data.slice(-7);
    const avg = recent.reduce((s, d) => s + (d.value || 0), 0) / recent.length;
    return {
      dir: avg > 0.5 ? 'Rising' : avg < -0.5 ? 'Falling' : 'Flat',
      val: avg,
      data: normalizedMomentum.data,
    };
  }, [normalizedMomentum]);

  const similarDays = activeLiveData?.metadata?.similar_days || [];
  const hasData = today.length > 0;

  const peakShiftBlocks = peakIdx >= 0 && yPeakIdx >= 0 ? peakIdx - yPeakIdx : null;
  const nightBase = safeAverage(hourlyAvg.slice(0, 6));
  const morningRampWindow = (hourlyAvg[10] || 0) - (hourlyAvg[6] || 0);
  const afternoonPlateau = safeAverage(hourlyAvg.slice(12, 17));
  const eveningPeakWindow = safeAverage(hourlyAvg.slice(18, 21));
  const demandRange = peak - valley;
  const pvRatio = valley > 0 ? peak / valley : null;
  const reserveMargin = dayAheadData?.kpis_full?.reserve_margin_adequacy_pct;

  // Duck Curve Severity: midday min (blocks 40-56) / morning peak (blocks 44-55)
  const duckCurveSeverity = useMemo(() => {
    if (today.length < 56) return null;
    const middayMin = Math.min(...today.slice(39, 56).filter((v) => v > 0));
    const morningPeak = Math.max(...today.slice(43, 56));
    if (morningPeak <= 0) return null;
    return middayMin / morningPeak;
  }, [today]);
  const duckCurveFlag = duckCurveSeverity != null && duckCurveSeverity < 0.65;

  const structuralRows = useMemo(
    () => [
      { m: 'Peak (MW)', t: peak, y: yPeak },
      { m: 'Valley (MW)', t: valley, y: yValley },
      { m: 'Energy (MWh)', t: energy, y: yEnergy },
      {
        m: 'Peak Hour',
        t: peakIdx >= 0 ? blockToTime(peakIdx + 1) : '--',
        y: yPeakIdx >= 0 ? blockToTime(yPeakIdx + 1) : '--',
        raw: true,
      },
      { m: 'Load Factor %', t: loadFactor, y: yLoadFactor },
      { m: 'Max Ramp Up', t: maxRampUp, y: yMaxRampUp },
      { m: 'Max Ramp Down', t: maxRampDown, y: yMaxRampDown },
      { m: 'Avg Dev (MW)', t: avgDev, y: null },
    ],
    [
      peak,
      yPeak,
      valley,
      yValley,
      energy,
      yEnergy,
      peakIdx,
      yPeakIdx,
      loadFactor,
      yLoadFactor,
      maxRampUp,
      yMaxRampUp,
      maxRampDown,
      yMaxRampDown,
      avgDev,
    ]
  );

  const fingerprintRows = useMemo(
    () => [
      { l: 'Peak Hour', v: peakIdx >= 0 ? blockToTime(peakIdx + 1) : '--', c: '#F07825' },
      { l: 'Night Base (00-06)', v: `${fmt(nightBase, 0)} MW`, c: '#5B9FE4' },
      {
        l: 'Morning Ramp (06-10)',
        v: `${morningRampWindow >= 0 ? '+' : ''}${fmt(morningRampWindow, 0)} MW`,
        c: '#FBBF24',
      },
      { l: 'Afternoon Plateau', v: `${fmt(afternoonPlateau, 0)} MW`, c: '#F07825' },
      { l: 'Evening Peak (18-21)', v: `${fmt(eveningPeakWindow, 0)} MW`, c: '#EC4899' },
      { l: 'Demand Range', v: `${fmt(demandRange, 0)} MW`, c: '#ECEEF3' },
      { l: 'P/V Ratio', v: pvRatio != null ? pvRatio.toFixed(2) : '--', c: '#ECEEF3' },
    ],
    [
      peakIdx,
      nightBase,
      morningRampWindow,
      afternoonPlateau,
      eveningPeakWindow,
      demandRange,
      pvRatio,
    ]
  );

  const quickStats = useMemo(
    () => [
      {
        eyebrow: 'Peak posture',
        title: 'Peak Load',
        value: fmt(peak, 0),
        unit: 'MW',
        tone: '#F07825',
        detail:
          peakShiftBlocks == null
            ? 'Peak timing unavailable'
            : `${peakShiftBlocks > 0 ? '+' : ''}${peakShiftBlocks} blocks vs yesterday`,
        footer: peakIdx >= 0 ? `Observed at ${blockToTime(peakIdx + 1)}` : 'Awaiting signal',
      },
      {
        eyebrow: 'Energy balance',
        title: 'Daily Energy',
        value: fmt(energy, 0),
        unit: 'MWh',
        tone: '#ECEEF3',
        detail:
          yEnergy > 0
            ? `${(energy - yEnergy) / yEnergy >= 0 ? '+' : ''}${fmt(((energy - yEnergy) / yEnergy) * 100)}% vs yesterday`
            : 'No comparison baseline',
        footer: `Average load ${fmt(avgLoad, 0)} MW`,
      },
      {
        eyebrow: 'Shape efficiency',
        title: 'Load Factor',
        value: fmt(loadFactor, 1),
        unit: '%',
        tone: loadFactor >= 70 ? '#34D399' : '#FBBF24',
        detail: yLoadFactor
          ? `${loadFactor - yLoadFactor >= 0 ? '+' : ''}${fmt(loadFactor - yLoadFactor, 1)} pts vs yesterday`
          : 'No yesterday factor',
        footer: `Range spread ${fmt(demandRange, 0)} MW`,
      },
      {
        eyebrow: 'Day context',
        title: dayRegime,
        value: detectedSeason ? detectedSeason.toUpperCase() : '--',
        tone: '#5B9FE4',
        detail: effectiveDate || 'No active date',
        footer:
          reserveMargin != null
            ? `Reserve margin ${fmt(reserveMargin, 1)}%`
            : 'Reserve margin unavailable',
      },
      {
        eyebrow: 'Momentum',
        title: trend.dir,
        value: `${trend.val > 0 ? '+' : ''}${fmt(trend.val, 1)}`,
        unit: '%/day',
        tone: trend.val > 0.5 ? '#F87171' : trend.val < -0.5 ? '#34D399' : '#ECEEF3',
        detail: trend.data.length
          ? `${trend.data.length} daily points tracked`
          : 'Momentum unavailable',
        footer: similarDays[0]
          ? `Best historical match ${similarDays[0].date}`
          : 'No similar-day anchor',
      },
      {
        eyebrow: 'Storage dispatch',
        title: 'Peak-to-Valley',
        value: pvRatio != null ? pvRatio.toFixed(2) : '--',
        unit: 'ratio',
        tone: pvRatio != null && pvRatio > 1.5 ? '#F87171' : '#ECEEF3',
        detail:
          pvRatio != null
            ? `Peak ${fmt(peak, 0)} MW / Valley ${fmt(valley, 0)} MW`
            : 'Awaiting load signal',
        footer: 'Higher ratio = more storage/hydro opportunity',
      },
      {
        eyebrow: 'Solar integration',
        title: duckCurveFlag ? 'Duck Curve Risk' : 'Duck Curve',
        value: duckCurveSeverity != null ? duckCurveSeverity.toFixed(2) : '--',
        unit: 'ratio',
        tone: duckCurveFlag
          ? '#F87171'
          : duckCurveSeverity != null && duckCurveSeverity < 0.75
            ? '#FBBF24'
            : '#34D399',
        detail: duckCurveFlag
          ? 'Midday dip severe — solar ramp risk'
          : duckCurveSeverity != null
            ? 'Midday min / Morning peak'
            : 'Awaiting load signal',
        footer: 'Flag raised if ratio < 0.65 (Rajasthan critical)',
      },
    ],
    [
      peak,
      peakShiftBlocks,
      peakIdx,
      energy,
      yEnergy,
      avgLoad,
      loadFactor,
      yLoadFactor,
      demandRange,
      dayRegime,
      detectedSeason,
      effectiveDate,
      reserveMargin,
      avgDev,
      maxDevBlock,
      maxRampUp,
      maxRampDown,
      trend,
      similarDays,
      pvRatio,
      duckCurveSeverity,
      duckCurveFlag,
    ]
  );

  const benchmarkOpt = useMemo(() => {
    if (!today.length) return null;
    const now = new Date();
    const nowBlk = Math.min(95, now.getHours() * 4 + Math.floor(now.getMinutes() / 15));
    const mk = (name, data, color, width = 1.5, dash = false) =>
      data.length
        ? {
            name,
            type: 'line',
            data,
            smooth: true,
            symbol: 'none',
            lineStyle: { width, color, type: dash ? 'dashed' : 'solid' },
            itemStyle: { color },
          }
        : null;

    return {
      ...ecBase(),
      legend: {
        ...ecBase().legend,
        data: ['Today', 'Yesterday', 'Last Week', 'Last Year'].filter(
          (_, i) => [today, t1, t7, t365][i].length
        ),
      },
      xAxis: {
        ...ecBase().xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [
        {
          ...mk('Today', today, '#F07825', 2.6),
          z: 10,
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#F078251A' },
                { offset: 1, color: 'transparent' },
              ],
            },
          },
          markPoint: {
            data: [
              {
                type: 'max',
                symbolSize: 30,
                label: {
                  formatter: (p) => `${fmt(p.value, 0)}`,
                  fontSize: 12,
                  color: '#F07825',
                  fontWeight: 700,
                },
              },
            ],
            itemStyle: { color: '#F07825' },
          },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              {
                xAxis: blockToTime(nowBlk + 1),
                lineStyle: { color: '#F87171', type: 'dashed', width: 1 },
                label: { formatter: 'NOW', fontSize: 12, color: '#F87171' },
              },
            ],
          },
        },
        mk('Yesterday', t1, '#ECEEF360', 1.5, true),
        mk('Last Week', t7, '#FBBF2490', 1.25, true),
        mk('Last Year', t365, '#8B5CF690', 1.25, true),
      ].filter(Boolean),
    };
  }, [today, t1, t7, t365]);

  const deviationOpt = useMemo(() => {
    if (!deviation.length) return null;
    return {
      ...ecBase(),
      xAxis: {
        ...ecBase().xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase().yAxis, name: 'MW delta vs yesterday' },
      visualMap: {
        show: false,
        min: -300,
        max: 300,
        inRange: { color: ['#34D399', '#2A292F', '#F87171'] },
      },
      series: [
        {
          type: 'bar',
          data: deviation,
          barMaxWidth: 6,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ yAxis: 0, lineStyle: { color: '#6B718640', width: 1 } }],
          },
        },
      ],
    };
  }, [deviation]);

  const durationOpt = useMemo(() => {
    if (!sortedLoad.length) return null;
    const pctLabels = sortedLoad.map((_, i) => `${((i / sortedLoad.length) * 100).toFixed(0)}%`);
    return {
      ...ecBase(),
      xAxis: {
        ...ecBase().xAxis,
        data: pctLabels,
        name: '% of time exceeded',
        axisLabel: { ...ecBase().xAxis.axisLabel, interval: 9 },
      },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: [
        {
          type: 'line',
          data: sortedLoad,
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 2.5, color: '#F07825' },
          itemStyle: { color: '#F07825' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#F0782515' },
                { offset: 1, color: 'transparent' },
              ],
            },
          },
          markPoint: {
            data: [
              {
                coord: [0, sortedLoad[0]],
                symbolSize: 20,
                label: {
                  formatter: `Peak\n${fmt(sortedLoad[0], 0)}`,
                  fontSize: 12,
                  color: '#F87171',
                },
                itemStyle: { color: '#F87171' },
              },
              {
                coord: [
                  pctLabels[Math.floor(sortedLoad.length * 0.5)],
                  sortedLoad[Math.floor(sortedLoad.length * 0.5)],
                ],
                symbolSize: 16,
                label: {
                  formatter: `P50\n${fmt(sortedLoad[Math.floor(sortedLoad.length * 0.5)], 0)}`,
                  fontSize: 12,
                  color: '#FBBF24',
                },
                itemStyle: { color: '#FBBF24' },
              },
            ],
          },
        },
      ],
    };
  }, [sortedLoad]);

  const comparisonSeries = useMemo(() => {
    const merged = { ...(localMultiSeries || {}) };
    if (today.length && effectiveDate) merged[effectiveDate] = today;
    return merged;
  }, [localMultiSeries, today, effectiveDate]);

  const comparisonOpt = useMemo(() => {
    const dates = Object.keys(comparisonSeries || {});
    if (!dates.length) return null;
    const colors = ['#F07825', '#34D399', '#F87171', '#FBBF24', '#8B5CF6', '#45B7D1', '#EC4899'];
    return {
      ...ecBase(),
      legend: { ...ecBase().legend, data: dates },
      xAxis: {
        ...ecBase().xAxis,
        data: timeLabels,
        axisLabel: { ...ecBase().xAxis.axisLabel, interval: 11 },
      },
      yAxis: { ...ecBase().yAxis, name: 'MW' },
      series: dates.map((date, i) => ({
        name: date,
        type: 'line',
        data: comparisonSeries[date],
        smooth: true,
        symbol: 'none',
        lineStyle: { width: date === effectiveDate ? 2.8 : 2, color: colors[i % colors.length] },
        itemStyle: { color: colors[i % colors.length] },
      })),
    };
  }, [comparisonSeries, effectiveDate]);

  const momentumOpt = useMemo(() => {
    if (!trend.data.length) return null;
    return {
      ...ecBase(),
      grid: { ...ecBase().grid, bottom: 56 },
      xAxis: {
        ...ecBase().xAxis,
        data: trend.data.map((d) => d.date),
        axisLabel: { ...ecBase().xAxis.axisLabel, rotate: 35 },
      },
      yAxis: { ...ecBase().yAxis, name: 'DoD %' },
      series: [
        {
          type: 'bar',
          data: trend.data.map((d) => ({
            value: d.value,
            itemStyle: {
              color: d.value >= 0 ? '#34D399' : '#F87171',
              borderRadius: d.value >= 0 ? [3, 3, 0, 0] : [0, 0, 3, 3],
            },
          })),
          barMaxWidth: 14,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ yAxis: 0, lineStyle: { color: '#6B718640' } }],
          },
        },
      ],
    };
  }, [trend]);

  const hourlyDev = useMemo(
    () =>
      Array.from({ length: 24 }, (_, h) => {
        const t = hourlyAvg[h];
        const y = yHourlyAvg[h];
        if (!t || t <= 0 || !y || y <= 0) return null;
        return {
          hour: `${String(h).padStart(2, '0')}:00`,
          pct: ((t - y) / y) * 100,
          today: t,
          yday: y,
        };
      }).filter(Boolean),
    [hourlyAvg, yHourlyAvg]
  );

  const TABS = [
    { id: 'benchmark', label: 'Benchmark Overlay' },
    { id: 'deviation', label: 'Deviation Map' },
    { id: 'duration', label: 'Load Duration' },
    { id: 'comparison', label: 'Multi-Day' },
    { id: 'momentum', label: '30D Momentum' },
  ];

  const chartMap = {
    benchmark: benchmarkOpt,
    deviation: deviationOpt,
    duration: durationOpt,
    comparison: comparisonOpt,
    momentum: momentumOpt,
  };

  useEffect(() => {
    if (mainTab !== 'momentum' || !effectiveDate || !apiUrl) return;

    let cancelled = false;

    const fetchMomentum = async () => {
      try {
        const dates30 = [];
        for (let i = 30; i >= 0; i -= 1) {
          const d = new Date(effectiveDate);
          d.setDate(d.getDate() - i);
          dates30.push(d.toISOString().split('T')[0]);
        }

        const res = await axios.post(apiUrl('/v2/load_change'), {
          date1: effectiveDate,
          date2: dates30[dates30.length - 2],
          dates: dates30,
        });

        if (!cancelled) setLocalMomentum(res.data);
      } catch (error) {
        console.error('Failed to fetch load momentum:', error);
      }
    };

    fetchMomentum();

    return () => {
      cancelled = true;
    };
  }, [mainTab, effectiveDate, apiUrl]);

  useEffect(() => {
    if (mainTab !== 'comparison' || !apiUrl) return;

    const requestDates = Array.from(new Set(compareDates.filter(Boolean)));
    if (!requestDates.length) {
      setLocalMultiSeries({});
      return;
    }

    let cancelled = false;

    const fetchSeries = async () => {
      try {
        const res = await axios.post(apiUrl('/v2/load_series'), { dates: requestDates });
        if (!cancelled) setLocalMultiSeries(res.data?.series || {});
      } catch (error) {
        console.error('Failed to fetch multi-day load series:', error);
      }
    };

    fetchSeries();

    return () => {
      cancelled = true;
    };
  }, [mainTab, compareDates, apiUrl]);

  const upsertCompareDate = (date) => {
    if (!date) return;
    const next = compareDates.includes(date) ? compareDates : [...compareDates, date];
    setCompareDates(next);
    setMultiSelectedDates(next);
    setMainTab('comparison');
  };

  const removeCompareDate = (date) => {
    const next = compareDates.filter((item) => item !== date);
    setCompareDates(next);
    setMultiSelectedDates(next);
  };

  const overlayContent = useMemo(() => {
    if (overlayPanel === 'structural') {
      return (
        <>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: 10,
              marginBottom: 16,
            }}
          >
            <div style={{ ...S.card, padding: '12px 14px', gap: 4 }}>
              <span style={S.miniLabel}>Peak shift</span>
              <strong style={{ fontSize: 22, color: '#F07825' }}>
                {peakShiftBlocks == null
                  ? '--'
                  : `${peakShiftBlocks > 0 ? '+' : ''}${peakShiftBlocks}`}
              </strong>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>15-min blocks vs yesterday</span>
            </div>
            <div style={{ ...S.card, padding: '12px 14px', gap: 4 }}>
              <span style={S.miniLabel}>Largest block delta</span>
              <strong style={{ fontSize: 22, color: '#F87171' }}>
                {fmt(absDeviation[maxDevBlock], 0)} MW
              </strong>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>
                {maxDevBlock >= 0 ? `Around ${blockToTime(maxDevBlock + 1)}` : 'Not available'}
              </span>
            </div>
            <div style={{ ...S.card, padding: '12px 14px', gap: 4 }}>
              <span style={S.miniLabel}>Ramp spread</span>
              <strong style={{ fontSize: 22, color: '#34D399' }}>
                {fmt(maxRampUp - maxRampDown, 0)} MW
              </strong>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>Intraday movement envelope</span>
            </div>
          </div>
          <div style={{ ...S.card, borderRadius: 14 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ ...S.th, textAlign: 'left' }}>Metric</th>
                  <th style={S.th}>Today</th>
                  <th style={S.th}>Yesterday</th>
                  <th style={S.th}>Delta</th>
                </tr>
              </thead>
              <tbody>
                {structuralRows.map((row) => {
                  const delta =
                    !row.raw && row.y != null && row.y !== 0
                      ? ((row.t - row.y) / row.y) * 100
                      : null;
                  return (
                    <tr key={row.m}>
                      <td style={{ ...S.td, textAlign: 'left', fontWeight: 600, color: '#A0A5B8' }}>
                        {row.m}
                      </td>
                      <td style={{ ...S.td, fontWeight: 600 }}>
                        {row.raw ? row.t : fmt(row.t, 0)}
                      </td>
                      <td style={{ ...S.td, color: '#6B7186' }}>
                        {row.y == null ? '--' : row.raw ? row.y : fmt(row.y, 0)}
                      </td>
                      <td
                        style={{
                          ...S.td,
                          color: delta != null ? (delta >= 0 ? '#F87171' : '#34D399') : '#6B7186',
                          fontWeight: 600,
                        }}
                      >
                        {delta != null ? `${delta >= 0 ? '+' : ''}${fmt(delta, 1)}%` : '--'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      );
    }

    if (overlayPanel === 'similar') {
      return similarDays.length ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ ...S.card, padding: '14px 16px', gap: 6 }}>
            <span style={S.miniLabel}>Top match</span>
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div>
                <div style={{ fontSize: 19, fontWeight: 700, color: '#ECEEF3' }}>
                  {similarDays[0].date}
                </div>
                <div style={{ fontSize: 10, color: '#6B7186', marginTop: 4 }}>
                  {similarDays[0].day_type || 'Historical analog'} with{' '}
                  {(similarDays[0].similarity_score * 100).toFixed(0)}% similarity
                </div>
              </div>
              <button
                type="button"
                onClick={() => upsertCompareDate(similarDays[0].date)}
                style={{
                  ...S.badge('#F07825'),
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  border: 'none',
                }}
              >
                Overlay on multi-day
              </button>
            </div>
          </div>
          {similarDays.slice(0, 8).map((item) => (
            <div
              key={item.date}
              style={{
                ...S.card,
                padding: '12px 14px',
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                borderRadius: 14,
              }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#ECEEF3' }}>{item.date}</div>
                <div style={{ fontSize: 12, color: '#8A90A6', marginTop: 4 }}>
                  {item.day_type || 'Historical day pattern'}
                </div>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  flexWrap: 'wrap',
                  justifyContent: 'flex-end',
                }}
              >
                <div
                  style={{
                    width: 92,
                    height: 8,
                    background: '#2A292F',
                    borderRadius: 999,
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${Math.max(0, Math.min(100, item.similarity_score * 100))}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #FBBF24, #F07825)',
                      borderRadius: 999,
                    }}
                  />
                </div>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    color: '#F07825',
                    minWidth: 38,
                    textAlign: 'right',
                  }}
                >
                  {(item.similarity_score * 100).toFixed(0)}%
                </span>
                <button
                  type="button"
                  onClick={() => upsertCompareDate(item.date)}
                  style={{
                    ...S.badge('#5B9FE4'),
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    border: 'none',
                  }}
                >
                  Add
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ padding: 20, textAlign: 'center', color: '#6B7186', fontSize: 11 }}>
          No similar historical days are available for this date.
        </div>
      );
    }

    if (overlayPanel === 'fingerprint') {
      return (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
            gap: 10,
          }}
        >
          {fingerprintRows.map((row) => (
            <div key={row.l} style={{ ...S.card, padding: '14px 16px', gap: 6, borderRadius: 14 }}>
              <span style={S.miniLabel}>{row.l}</span>
              <strong style={{ fontSize: 21, color: row.c, lineHeight: 1.15 }}>{row.v}</strong>
              <span style={{ fontSize: 12, color: '#8A90A6' }}>
                {row.l === 'Peak Hour'
                  ? 'Shape anchor for intraday operations'
                  : 'Fingerprint trait for today'}
              </span>
            </div>
          ))}
        </div>
      );
    }

    return null;
  }, [
    overlayPanel,
    peakShiftBlocks,
    absDeviation,
    maxDevBlock,
    maxRampUp,
    maxRampDown,
    structuralRows,
    similarDays,
    fingerprintRows,
    compareDates,
  ]);

  return (
    <VpPageShell className="load-analysis-page">
      {!hasData ? (
        <VpEmptyState
          title={selfFetching ? 'Loading benchmark data…' : 'No load data available'}
          description={selfFetching ? `Fetching data for ${effectiveDate}` : 'Select a date to load benchmark data'}
        />
      ) : (
        <>
          <div style={{ padding: '0 0 8px' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(155px, 1fr))',
                gap: 10,
              }}
            >
              {quickStats.map((card) => (
                <KpiCard
                  key={card.title}
                  eyebrow={card.eyebrow}
                  title={card.title}
                  value={card.value}
                  unit={card.unit}
                  tone={card.tone}
                  detail={card.detail}
                  footer={card.footer}
                />
              ))}
            </div>
          </div>

          {hourlyDev.length > 0 && (
            <div style={{ padding: '0 0 8px' }}>
              <div style={{ ...S.card, padding: '10px 14px', gap: 8 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 12,
                    flexWrap: 'wrap',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      flexWrap: 'wrap',
                      minWidth: 0,
                    }}
                  >
                    <div style={S.cardTitle}>Hourly Load vs Yesterday</div>
                    <div style={{ fontSize: 10, color: '#6B7186', whiteSpace: 'nowrap' }}>
                      Fast hourly heatbar to spot where the shape drifted.
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#8A90A6' }}>
                    <span>
                      <span
                        style={{
                          display: 'inline-block',
                          width: 8,
                          height: 8,
                          borderRadius: 2,
                          background: '#34D399',
                          marginRight: 4,
                        }}
                      />
                      Lower
                    </span>
                    <span>
                      <span
                        style={{
                          display: 'inline-block',
                          width: 8,
                          height: 8,
                          borderRadius: 2,
                          background: '#F87171',
                          marginRight: 4,
                        }}
                      />
                      Higher
                    </span>
                  </div>
                </div>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(24, minmax(0, 1fr))',
                    gap: 4,
                  }}
                >
                  {hourlyDev.map((h) => {
                    const clamp = Math.max(-10, Math.min(10, h.pct));
                    const intensity = Math.abs(clamp) / 10;
                    const bg =
                      h.pct >= 0
                        ? `rgba(248, 113, 113, ${0.14 + intensity * 0.62})`
                        : `rgba(52, 211, 153, ${0.14 + intensity * 0.62})`;

                    return (
                      <div
                        key={h.hour}
                        title={`${h.hour}: Today ${fmt(h.today, 0)} MW vs Y'day ${fmt(h.yday, 0)} MW`}
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: 4,
                        }}
                      >
                        <div
                          style={{
                            width: '100%',
                            minHeight: 38,
                            borderRadius: 8,
                            background: bg,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '0 2px',
                          }}
                        >
                          <span style={{ fontSize: 8, fontWeight: 700, color: '#ECEEF3' }}>
                            {h.pct >= 0 ? '+' : ''}
                            {fmt(h.pct, 1)}%
                          </span>
                        </div>
                        <span style={{ fontSize: 8, color: '#6B7186' }}>{h.hour.slice(0, 2)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          <div style={{ padding: '0 0 0', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ ...S.card, position: 'relative', overflow: 'hidden', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                  padding: '16px 16px 10px',
                  borderBottom: '1px solid rgba(42, 41, 47, 0.9)',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 12,
                    flexWrap: 'wrap',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      flexWrap: 'wrap',
                      minWidth: 0,
                      flex: '1 1 420px',
                    }}
                  >
                    <div style={S.tabBar}>
                      {TABS.map((tab) => (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => setMainTab(tab.id)}
                          style={S.tab(mainTab === tab.id)}
                        >
                          {tab.label}
                        </button>
                      ))}
                    </div>
                    {mainTab === 'comparison' && (
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          flexWrap: 'wrap',
                          minWidth: 0,
                        }}
                      >
                        <input
                          type="date"
                          onChange={(e) => upsertCompareDate(e.target.value)}
                          style={{
                            background: '#141419',
                            border: '1px solid #2A292F',
                            borderRadius: 999,
                            padding: '8px 12px',
                            color: '#ECEEF3',
                            fontSize: 10,
                            fontFamily: 'inherit',
                          }}
                        />
                        {compareDates.map((date) => (
                          <span
                            key={date}
                            style={{
                              ...S.badge('#F07825'),
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                            }}
                          >
                            {date}
                            <button
                              type="button"
                              onClick={() => removeCompareDate(date)}
                              style={{
                                border: 'none',
                                background: 'none',
                                color: '#F07825',
                                cursor: 'pointer',
                                fontSize: 11,
                                fontWeight: 700,
                              }}
                            >
                              x
                            </button>
                          </span>
                        ))}
                        {compareDates.length > 0 ? (
                          <button
                            type="button"
                            onClick={() => {
                              setCompareDates([]);
                              setMultiSelectedDates([]);
                            }}
                            style={{
                              ...S.badge('#F87171'),
                              cursor: 'pointer',
                              border: 'none',
                              fontFamily: 'inherit',
                            }}
                          >
                            Clear
                          </button>
                        ) : null}
                      </div>
                    )}
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      gap: 8,
                      flexWrap: 'wrap',
                      justifyContent: 'flex-end',
                    }}
                  >
                    {DETAIL_PANELS.map((panel) => {
                      const meta =
                        panel.id === 'structural'
                          ? `${fmt(avgDev, 0)} MW avg drift`
                          : panel.id === 'similar'
                            ? `${similarDays.length} matches`
                            : peakIdx >= 0
                              ? `Peak ${blockToTime(peakIdx + 1)}`
                              : 'View details';
                      const active = overlayPanel === panel.id;

                      return (
                        <button
                          key={panel.id}
                          type="button"
                          onClick={() => setOverlayPanel(active ? null : panel.id)}
                          style={S.actionBtn(active)}
                        >
                          <span style={S.miniLabel}>
                            {active ? 'Close overlay' : 'Quick panel'}
                          </span>
                          <span style={{ fontSize: 11, fontWeight: 700 }}>{panel.label}</span>
                          <span style={{ fontSize: 12, color: active ? '#F5B17D' : '#8A90A6' }}>
                            {meta}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div style={{ position: 'relative', padding: '6px 8px 8px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                {chartMap[mainTab] ? (
                  <ReactECharts
                    option={chartMap[mainTab]}
                    style={{ height: '100%', width: '100%', minHeight: 220 }}
                    notMerge
                  />
                ) : (
                  <div style={{ padding: 70, textAlign: 'center', color: '#6B7186', fontSize: 11 }}>
                    {mainTab === 'comparison'
                      ? 'Pick dates above or from similar-day overlays to compare.'
                      : selfFetching
                        ? 'Loading data…'
                        : 'No data for this view.'}
                  </div>
                )}

                {overlayPanel ? (
                  <div
                    onClick={() => setOverlayPanel(null)}
                    style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(7, 7, 10, 0.52)',
                      backdropFilter: 'blur(4px)',
                      display: 'flex',
                      justifyContent: 'flex-end',
                      padding: 12,
                    }}
                  >
                    <div
                      style={{
                        width: 'min(460px, 100%)',
                        height: '100%',
                        ...S.card,
                        borderRadius: 18,
                        background:
                          'linear-gradient(180deg, rgba(24, 24, 29, 0.98), rgba(18, 18, 22, 0.98))',
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          justifyContent: 'space-between',
                          gap: 12,
                          padding: '16px 16px 14px',
                          borderBottom: '1px solid rgba(42, 41, 47, 0.9)',
                        }}
                      >
                        <div>
                          <div style={S.cardTitle}>
                            {DETAIL_PANELS.find((panel) => panel.id === overlayPanel)?.label}
                          </div>
                          <div style={{ fontSize: 10, color: '#6B7186', marginTop: 4 }}>
                            {overlayPanel === 'structural' &&
                              'Shape deltas, peak timing, and ramp comparison against yesterday.'}
                            {overlayPanel === 'similar' &&
                              'Historical analogs you can push directly into multi-day overlays.'}
                            {overlayPanel === 'fingerprint' &&
                              'Compact load-shape descriptors for operator-side pattern recognition.'}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setOverlayPanel(null)}
                          style={{
                            width: 30,
                            height: 30,
                            borderRadius: 999,
                            border: '1px solid #2A292F',
                            background: '#141419',
                            color: '#A0A5B8',
                            cursor: 'pointer',
                            fontFamily: 'inherit',
                          }}
                        >
                          x
                        </button>
                      </div>
                      <div style={{ padding: 16, overflowY: 'auto' }}>{overlayContent}</div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </>
      )}
    </VpPageShell>
  );
}
