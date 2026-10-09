import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { ForecastResponse } from '../../types/api';
import { blockToTime } from '../../lib/utils';
import { useChartTokens, withAlpha } from '../../lib/chartTheme';

interface Props {
  data: ForecastResponse | undefined;
  loading?: boolean;
  height?: number | string;
  /** Compare-against series — e.g. T+1 plotted while viewing T+2. */
  compare?: ForecastResponse | undefined;
  compareLabel?: string;
  onBlockHover?: (block: number | null) => void;
  showActual?: boolean;
}

export function ForecastChart({
  data,
  compare,
  compareLabel = 'Compare',
  loading,
  height = 380,
  onBlockHover,
  showActual = true,
}: Props) {
  const t = useChartTokens();
  const option = useMemo(() => {
    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);
    const xLabels = blocks.map(blockToTime);

    const forecast = data?.series?.forecast || [];
    const baseline = data?.series?.hybrid_baseline || [];
    const actual = data?.series?.actual || [];

    const compareSeries = compare?.series?.forecast;

    const uncertainty = data?.forecast_uncertainty;
    const p10 = uncertainty?.length === 96 ? uncertainty.map((u) => u.p10 ?? null) : null;
    const p90 = uncertainty?.length === 96 ? uncertainty.map((u) => u.p90 ?? null) : null;

    const series: any[] = [];

    // Confidence band (p10..p90) drawn as two stacked area series
    if (p10 && p90) {
      const lower = p10.map((v) => (v == null ? null : v));
      const widths = p90.map((v, i) => {
        const lo = p10[i];
        return v == null || lo == null ? null : v - lo;
      });
      series.push(
        {
          name: 'p10',
          type: 'line',
          data: lower,
          stack: 'confidence',
          showSymbol: false,
          lineStyle: { opacity: 0 },
          areaStyle: { color: 'transparent' },
          tooltip: { show: false },
          z: 1,
        },
        {
          name: 'Confidence',
          type: 'line',
          data: widths,
          stack: 'confidence',
          showSymbol: false,
          lineStyle: { opacity: 0 },
          areaStyle: { color: withAlpha(t.accent, 0.12) },
          tooltip: { show: false },
          z: 1,
        }
      );
    }

    if (baseline.length === 96) {
      series.push({
        name: 'Baseline',
        type: 'line',
        data: baseline,
        showSymbol: false,
        smooth: true,
        lineStyle: { color: t.accent2, width: 1.2, type: 'dashed' },
        emphasis: { disabled: true },
        z: 2,
      });
    }

    if (compareSeries && compareSeries.length === 96) {
      series.push({
        name: compareLabel,
        type: 'line',
        data: compareSeries,
        showSymbol: false,
        smooth: true,
        lineStyle: { color: t.info, width: 1.6, opacity: 0.85 },
        z: 3,
      });
    }

    series.push({
      name: 'Forecast',
      type: 'line',
      data: forecast,
      showSymbol: false,
      smooth: true,
      lineStyle: { color: t.accent, width: 2.6, shadowColor: withAlpha(t.accent, 0.5), shadowBlur: 8 },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: withAlpha(t.accent, 0.18) },
            { offset: 1, color: withAlpha(t.accent, 0) },
          ],
        },
      },
      z: 4,
    });

    if (showActual && actual.length === 96 && actual.some((v) => v != null)) {
      series.push({
        name: 'Actual',
        type: 'line',
        data: actual.map((v) => (v == null ? null : v)),
        showSymbol: false,
        smooth: false,
        connectNulls: false,
        lineStyle: { color: t.success, width: 1.6 },
        z: 5,
      });
    }

    return {
      grid: { top: 24, right: 16, bottom: 36, left: 56 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: t.elevated,
        borderColor: t.outline,
        borderWidth: 1,
        textStyle: { color: t.text, fontSize: 12 },
        axisPointer: { lineStyle: { color: withAlpha(t.accent, 0.7), type: 'solid' } },
        formatter: (items: any[]) => {
          if (!items?.length) return '';
          const blockIdx = items[0].dataIndex;
          const time = blockToTime(blockIdx + 1);
          const rows = items
            .filter((it) => it.seriesName !== 'p10' && it.value != null)
            .map((it) => {
              const valStr = typeof it.value === 'number' ? it.value.toLocaleString('en-IN', { maximumFractionDigits: 0 }) : '—';
              return `<div style="display:flex;justify-content:space-between;gap:18px"><span style="color:${it.color}">●</span><span>${it.seriesName}</span><span style="font-family:'IBM Plex Mono',monospace;font-weight:600">${valStr}</span></div>`;
            })
            .join('');
          return `<div style="font-size:11px"><div style="margin-bottom:4px;color:${t.textMuted};text-transform:uppercase;letter-spacing:0.08em;font-size:10px">Block ${blockIdx + 1} · ${time}</div>${rows}</div>`;
        },
      },
      legend: {
        bottom: 4,
        textStyle: { color: t.textMuted, fontSize: 11 },
        itemWidth: 10,
        itemHeight: 10,
        icon: 'roundRect',
        data: [
          { name: 'Forecast', icon: 'roundRect' },
          ...(showActual && actual.some((v) => v != null) ? [{ name: 'Actual' }] : []),
          ...(baseline.length === 96 ? [{ name: 'Baseline' }] : []),
          ...(compareSeries && compareSeries.length === 96 ? [{ name: compareLabel }] : []),
          ...(p10 && p90 ? [{ name: 'Confidence' }] : []),
        ],
      },
      xAxis: {
        type: 'category',
        data: xLabels,
        axisLabel: {
          color: t.textMuted,
          fontSize: 10,
          interval: 11,
          formatter: (v: string) => v,
        },
        axisLine: { lineStyle: { color: t.outline } },
        axisTick: { show: false },
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          color: t.textMuted,
          fontSize: 10,
          formatter: (v: number) =>
            Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${v.toFixed(0)}`,
        },
        splitLine: { lineStyle: { color: withAlpha(t.outline, 0.6), type: 'dashed' } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series,
      animation: true,
      animationDuration: 320,
      animationEasing: 'cubicOut',
    };
  }, [data, compare, compareLabel, showActual, t]);

  const onEvents = useMemo(
    () =>
      onBlockHover
        ? {
            updateAxisPointer: (e: any) => {
              const idx = e?.dataIndex;
              onBlockHover(typeof idx === 'number' ? idx + 1 : null);
            },
          }
        : undefined,
    [onBlockHover]
  );

  return (
    <div className="relative h-full w-full" aria-busy={loading}>
      <ReactECharts
        option={option}
        notMerge={false}
        lazyUpdate
        style={{ width: '100%', height: typeof height === 'number' ? `${height}px` : height }}
        onEvents={onEvents as any}
      />
      {loading && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center bg-base/40 backdrop-blur-[1px]">
          <div className="rounded-md border border-border bg-elev2/80 px-3 py-1 text-2xs uppercase tracking-wider text-muted">
            Updating…
          </div>
        </div>
      )}
    </div>
  );
}
