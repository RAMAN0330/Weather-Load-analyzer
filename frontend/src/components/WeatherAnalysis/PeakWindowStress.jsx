import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useChartTokens, themedEcBase } from '../../lib/chartTheme';

export default function PeakWindowStress({ data, intraday }) {
  const t = useChartTokens();
  if (!data) return null;

  const windows = ['Morning', 'Midday', 'Evening', 'Night'];
  const blockTime = (block) => {
    const minutes = (block - 1) * 15;
    const h = String(Math.floor(minutes / 60)).padStart(2, '0');
    const m = String(minutes % 60).padStart(2, '0');
    return `${h}:${m}`;
  };

  const barOption = useMemo(() => {
    const tempDeltas = windows.map((w) => data[w]?.temp_delta || 0);
    const humDeltas = windows.map((w) => data[w]?.hum_delta || 0);

    return {
      title: {
        text: 'Window Comparison',
        left: 'center',
        textStyle: { color: t.textSecondary, fontSize: 14 },
      },
      tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 20, textStyle: { color: t.textMuted } },
      grid: { left: 40, right: 20, bottom: 20 },
      xAxis: {
        type: 'category',
        data: windows,
        axisLabel: { color: t.textMuted },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: t.textMuted },
        splitLine: { lineStyle: { color: t.outline } },
      },
      series: [
        {
          name: 'Temp Delta',
          type: 'bar',
          data: tempDeltas,
          itemStyle: { color: t.warm },
        },
        {
          name: 'Hum Delta',
          type: 'bar',
          data: humDeltas,
          itemStyle: { color: t.info },
        },
      ],
    };
  }, [data, t]);

  const gaugeOption = useMemo(() => {
    const maxDelta = Math.max(...windows.map((w) => Math.abs(data[w]?.temp_delta || 0)));

    return {
      title: {
        text: 'Peak Heat Stress',
        left: 'center',
        top: 10,
        textStyle: { color: t.textSecondary, fontSize: 14 },
      },
      series: [
        {
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 10,
          splitNumber: 5,
          itemStyle: { color: t.danger },
          progress: { show: true, width: 10 },
          pointer: { show: false },
          axisLine: {
            lineStyle: {
              width: 10,
              color: [
                [0.3, t.success],
                [0.7, t.warning],
                [1, t.danger],
              ],
            },
          },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          detail: {
            valueAnimation: true,
            offsetCenter: [0, '20%'],
            fontSize: 20,
            color: 'inherit',
            formatter: '{value} C Delta',
          },
          data: [{ value: maxDelta, name: 'Max Deviation' }],
        },
      ],
    };
  }, [data, t]);

  const peakShift = useMemo(() => {
    if (!intraday) return null;
    const actual = intraday.temperature.actual || [];
    const normal = intraday.temperature.normal || [];
    if (!actual.length || !normal.length) return null;
    const maxAct = Math.max(...actual);
    const maxNorm = Math.max(...normal);
    const idxAct = actual.indexOf(maxAct) + 1;
    const idxNorm = normal.indexOf(maxNorm) + 1;
    if (!idxAct || !idxNorm) return null;
    const shiftBlocks = idxAct - idxNorm;
    const minutes = Math.abs(shiftBlocks) * 15;
    const direction = shiftBlocks === 0 ? 'No Shift' : shiftBlocks > 0 ? 'Later' : 'Earlier';
    const significant = Math.abs(shiftBlocks) >= 4;
    const normPct = ((idxNorm - 1) / 95) * 100;
    const actPct = ((idxAct - 1) / 95) * 100;
    const left = Math.min(normPct, actPct);
    const width = Math.max(1, Math.abs(actPct - normPct));
    return {
      idxAct,
      idxNorm,
      shiftBlocks,
      minutes,
      direction,
      significant,
      normPct,
      actPct,
      left,
      width,
      normalTime: blockTime(idxNorm),
      actualTime: blockTime(idxAct),
    };
  }, [intraday]);

  const peakImpactOption = useMemo(() => {
    const tempDeltas = windows.map((w) => Number(data[w]?.temp_delta || 0));
    const humDeltas = windows.map((w) => Number(data[w]?.hum_delta || 0));
    const precip = windows.map((w) => Number(data[w]?.precip_total || 0));
    const impactIndex = windows.map(
      (_, i) => tempDeltas[i] * 1.2 + humDeltas[i] * 0.4 - precip[i] * 0.15
    );
    return {
      title: {
        text: 'Peak Window Impact',
        left: 'center',
        textStyle: { color: t.textSecondary, fontSize: 13 },
      },
      tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: '{b}: {c} MW' },
      grid: { left: 45, right: 20, top: 35, bottom: 20 },
      xAxis: {
        type: 'category',
        data: windows,
        axisLabel: { color: t.textMuted },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: t.textMuted },
        splitLine: { lineStyle: { color: t.outline } },
      },
      series: [
        {
          type: 'bar',
          data: impactIndex,
          itemStyle: { color: t.warning },
        },
      ],
    };
  }, [data, t]);

  const rampOption = useMemo(() => {
    const deltaSeries = intraday?.temperature?.delta || [];
    if (!deltaSeries.length) return null;
    const ramp = deltaSeries.map((val, idx) => {
      if (idx === 0) return 0;
      return (Number(val) || 0) - (Number(deltaSeries[idx - 1]) || 0);
    });
    return {
      title: {
        text: 'Ramp Acceleration',
        left: 'center',
        textStyle: { color: t.textSecondary, fontSize: 13 },
      },
      tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis', formatter: '{b}: {c} MW' },
      grid: { left: 45, right: 20, top: 35, bottom: 20 },
      xAxis: {
        type: 'category',
        data: ramp.map((_, i) => i + 1),
        axisLabel: { color: t.textMuted, interval: 7 },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: t.textMuted },
        splitLine: { lineStyle: { color: t.outline } },
      },
      series: [
        {
          type: 'line',
          data: ramp,
          smooth: true,
          lineStyle: { color: t.info, width: 2 },
          itemStyle: { color: t.info },
        },
      ],
    };
  }, [intraday, t]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full">
      <div className="lg:col-span-2 flex flex-col gap-4">
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[260px]">
          <ReactECharts option={barOption} style={{ height: '320px', width: '100%' }} />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[220px]">
            <ReactECharts option={peakImpactOption} style={{ height: '280px', width: '100%' }} />
          </div>
          <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[220px]">
            {rampOption ? (
              <ReactECharts option={rampOption} style={{ height: '280px', width: '100%' }} />
            ) : (
              <div className="h-full flex items-center justify-center text-[color:var(--text-dim)] text-sm">
                No ramp data.
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-4">
        <div className="flex-1 bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[140px]">
          <ReactECharts option={gaugeOption} style={{ height: '100%', width: '100%' }} />
        </div>
        {peakShift && (
          <div className="flex-1 bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-3 min-h-[140px] flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-[color:var(--text)]">Peak Shift Timeline</span>
              <span
                className={`text-[11px] px-2 py-1 rounded-full border ${peakShift.significant ? 'border-[color:color-mix(in_srgb,var(--danger)_60%,transparent)] text-[color:var(--danger)]' : 'border-[color:color-mix(in_srgb,var(--success)_50%,transparent)] text-[color:var(--success)]'}`}
              >
                {peakShift.significant ? 'Significant' : 'Minor'}
              </span>
            </div>
            <div className="text-xs text-[color:var(--text-muted)]">
              {peakShift.direction} by{' '}
              <span className="text-[color:var(--text)] font-semibold">
                {Math.abs(peakShift.shiftBlocks)} blocks
              </span>{' '}
              ({peakShift.minutes} min)
            </div>
            <div className="relative h-3 rounded-full bg-[var(--bg-surface)] border border-[color:var(--outline)]">
              <div
                className="absolute top-0 h-full rounded-full bg-[color:color-mix(in_srgb,var(--info)_60%,transparent)]"
                style={{ left: `${peakShift.left}%`, width: `${peakShift.width}%` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[var(--text-muted)] border border-[color:var(--text)]"
                style={{ left: `calc(${peakShift.normPct}% - 6px)` }}
                title={`Normal Peak: Block ${peakShift.idxNorm}`}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[var(--danger)] border border-[color:var(--bg-panel)]"
                style={{ left: `calc(${peakShift.actPct}% - 6px)` }}
                title={`Actual Peak: Block ${peakShift.idxAct}`}
              />
            </div>
            <div className="flex items-center justify-between text-xs text-[color:var(--text-muted)]">
              <span>
                Normal: Block {peakShift.idxNorm} • {peakShift.normalTime}
              </span>
              <span>
                Actual: Block {peakShift.idxAct} • {peakShift.actualTime}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
