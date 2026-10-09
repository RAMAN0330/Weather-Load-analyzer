import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useReducedMotion } from 'framer-motion';
import { themedBase, themedLegend, useChartPalette } from '../chartPalette';
import type { ForecastResponse } from '../api';
import {
  PEAK_WINDOW,
  TIME_LABELS,
  blockError,
  blockRange,
  fmtInt,
  fmtSigned,
  isNum,
} from '../utils';

const BAND = 'P10–P90 band';

const nz = (v: number | null | undefined) => (isNum(v) ? v : null);

/** Fills 100% of its parent; echarts-for-react's size-sensor tracks container resizes. */
export function ForecastChart({ forecast }: { forecast: ForecastResponse }) {
  const reduceMotion = useReducedMotion();
  const C = useChartPalette();

  const option = useMemo(() => {
    const blocks = forecast.blocks ?? [];
    const byIdx = TIME_LABELS.map((_, i) => blocks.find((b) => b.block === i + 1) ?? null);
    const p10 = byIdx.map((b) => nz(b?.p10));
    const p90 = byIdx.map((b) => nz(b?.p90));
    const bandDelta = byIdx.map((b) => (b && isNum(b.p10) && isNum(b.p90) ? b.p90 - b.p10 : null));
    const p50 = byIdx.map((b) => nz(b?.p50));
    const baseline = byIdx.map((b) => nz(b?.baseline));
    const actual = byIdx.map((b) => nz(b?.actual));
    const hasActual = actual.some((v) => v != null);
    const hasBand = bandDelta.some((v) => v != null);
    const s = forecast.summary;

    const tooltipRow = (color: string, label: string, val: string, dashed = false) =>
      `<div style="display:flex;justify-content:space-between;gap:16px;align-items:center">` +
      `<span style="display:inline-flex;align-items:center;gap:6px;color:${C.tooltip.muted}">` +
      `<span style="width:10px;height:0;border-top:2px ${dashed ? 'dashed' : 'solid'} ${color}"></span>${label}</span>` +
      `<span style="font-variant-numeric:tabular-nums">${val}</span></div>`;

    const legend = [
      ...(hasBand ? [BAND] : []),
      'P50',
      'Baseline',
      ...(hasActual ? ['Actual'] : []),
    ];

    const base = themedBase(C);
    return {
      ...base,
      animation: !reduceMotion,
      animationDuration: 400,
      grid: { left: 60, right: 20, top: 36, bottom: 56 },
      legend: {
        top: 4,
        right: 8,
        data: legend,
        ...themedLegend(C),
      },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { color: C.axisPointer, type: 'dashed' } },
        formatter: (params: Array<{ dataIndex: number }>) => {
          const i = params?.[0]?.dataIndex ?? 0;
          const b = byIdx[i];
          if (!b) return '';
          const e = blockError(b);
          const rows = [
            `<div style="margin-bottom:6px;color:${C.tooltip.text};font-weight:600">Block ${b.block} · ${blockRange(b.block)}</div>`,
            tooltipRow(C.bandStrong, 'P90', `${fmtInt(b.p90)} MW`),
            tooltipRow(C.p50, 'P50', `${fmtInt(b.p50)} MW`),
            tooltipRow(C.bandStrong, 'P10', `${fmtInt(b.p10)} MW`),
            tooltipRow(C.baseline, 'Baseline', `${fmtInt(b.baseline)} MW`, true),
          ];
          if (hasActual) {
            rows.push(tooltipRow(C.actual, 'Actual', `${fmtInt(b.actual)} MW`));
            if (e.mw != null)
              rows.push(
                `<div style="margin-top:4px;padding-top:4px;border-top:1px solid ${C.tooltip.divider};display:flex;justify-content:space-between;gap:16px;color:${C.tooltip.muted}"><span>Error (P50−A)</span><span>${fmtSigned(e.mw)} MW · ${fmtSigned(e.pct, 2)}%</span></div>`
              );
          }
          return rows.join('');
        },
      },
      xAxis: {
        ...base.xAxis,
        type: 'category',
        boundaryGap: false,
        data: TIME_LABELS,
        axisLabel: { ...base.xAxis.axisLabel, interval: 7 },
      },
      yAxis: {
        ...base.yAxis,
        type: 'value',
        scale: true,
        name: 'MW',
        nameTextStyle: { color: C.mutedText, fontSize: 10, padding: [0, 40, 0, 0] },
        axisLabel: {
          ...base.yAxis.axisLabel,
          formatter: (v: number) => v.toLocaleString('en-IN'),
        },
        splitLine: { lineStyle: { color: C.grid, type: 'dashed' } },
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        {
          type: 'slider',
          xAxisIndex: 0,
          height: 16,
          bottom: 8,
          borderColor: C.zoom.border,
          backgroundColor: C.zoom.bg,
          fillerColor: C.zoom.filler,
          dataBackground: {
            lineStyle: { color: C.zoom.dataLine },
            areaStyle: { color: C.zoom.dataArea },
          },
          selectedDataBackground: {
            lineStyle: { color: C.zoom.handle },
            areaStyle: { color: C.zoom.filler },
          },
          handleStyle: { color: C.zoom.handle, borderColor: C.zoom.handle },
          moveHandleStyle: { color: C.zoom.moveHandle },
          textStyle: { color: C.mutedText, fontSize: 10 },
          labelFormatter: (_: number, v: string) => v,
        },
      ],
      series: [
        // Band = invisible P10 base + stacked (P90 − P10) area. Both share the legend name.
        ...(hasBand
          ? [
              {
                name: BAND,
                type: 'line',
                data: p10,
                stack: 'band',
                symbol: 'none',
                lineStyle: { opacity: 0 },
                silent: true,
                z: 1,
              },
              {
                name: BAND,
                type: 'line',
                data: bandDelta,
                stack: 'band',
                symbol: 'none',
                lineStyle: { opacity: 0 },
                areaStyle: { color: C.band },
                itemStyle: { color: C.bandStrong },
                silent: true,
                z: 1,
              },
            ]
          : []),
        {
          name: 'Baseline',
          type: 'line',
          data: baseline,
          symbol: 'none',
          lineStyle: { color: C.baseline, width: 1.5, type: 'dashed' },
          itemStyle: { color: C.baseline },
          z: 2,
        },
        {
          name: 'P50',
          type: 'line',
          data: p50,
          symbol: 'none',
          lineStyle: { color: C.p50, width: 2.25 },
          itemStyle: { color: C.p50 },
          z: 4,
          markArea: {
            silent: true,
            itemStyle: {
              color: C.peakArea,
              borderColor: C.peakAreaBorder,
              borderWidth: 0,
            },
            label: {
              color: C.peakLabel,
              fontSize: 10,
              position: 'insideTop',
              fontFamily: "'IBM Plex Mono', monospace",
            },
            data: [[{ name: 'Evening peak', xAxis: PEAK_WINDOW.from }, { xAxis: PEAK_WINDOW.to }]],
          },
          markPoint:
            s && isNum(s.peak_block) && isNum(s.peak_mw)
              ? {
                  symbol: 'circle',
                  symbolSize: 8,
                  itemStyle: { color: C.p50, borderColor: C.markBorder, borderWidth: 2 },
                  label: {
                    show: true,
                    position: 'top',
                    distance: 8,
                    color: C.text,
                    fontSize: 10,
                    fontFamily: "'IBM Plex Mono', monospace",
                    formatter: () => `peak ${fmtInt(s.peak_mw)}`,
                  },
                  data: [{ coord: [TIME_LABELS[s.peak_block - 1], s.peak_mw] }],
                }
              : undefined,
        },
        ...(hasActual
          ? [
              {
                name: 'Actual',
                type: 'line',
                data: actual,
                symbol: 'none',
                connectNulls: false,
                lineStyle: { color: C.actual, width: 2 },
                itemStyle: { color: C.actual },
                z: 5,
              },
            ]
          : []),
      ],
    };
  }, [forecast, reduceMotion, C]);

  return (
    <figure
      className="data-surface flex h-full min-h-0 min-w-0 flex-col rounded-xl px-3 pb-1 pt-2.5"
      aria-label="Forecast load curve"
    >
      <figcaption className="flex shrink-0 items-baseline justify-between gap-3">
        <span className="shrink-0 text-xs font-semibold uppercase tracking-wider text-muted">
          Load curve · {forecast.target_date}
        </span>
        <span
          className="min-w-0 truncate font-mono text-2xs text-faint"
          title="Scroll or drag to zoom"
        >
          shaded P10–P90 · dashed baseline · amber {PEAK_WINDOW.from}–{PEAK_WINDOW.to} peak window
        </span>
      </figcaption>
      {/* Bounded flex child; legacy CSS forces .echarts-for-react to 100% of this box. */}
      <div className="relative min-h-0 flex-1">
        <div className="absolute inset-0">
          <ReactECharts
            option={option}
            notMerge
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        </div>
      </div>
    </figure>
  );
}
