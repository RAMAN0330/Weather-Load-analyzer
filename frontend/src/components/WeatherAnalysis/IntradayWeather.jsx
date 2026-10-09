import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useChartTokens, themedEcBase } from '../../lib/chartTheme';

export default function IntradayWeather({ data }) {
  const t = useChartTokens();
  if (!data) return null;

  const blocks = Array.from({ length: 96 }, (_, i) => i + 1);

  const getOption = (title, actualData, normalData, unit, color) => ({
    title: {
      text: title,
      left: 'center',
      textStyle: { color: t.textSecondary, fontSize: 14 },
    },
    tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis' },
    legend: { top: 20, textStyle: { color: t.textMuted } },
    grid: { left: 40, right: 20, top: 50, bottom: 20 },
    xAxis: {
      type: 'category',
      data: blocks,
      axisLabel: { color: t.textMuted, interval: 7 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: t.textMuted, formatter: `{value} ${unit}` },
      splitLine: { lineStyle: { color: t.outline } },
    },
    series: [
      {
        name: 'Actual',
        type: 'line',
        data: actualData,
        itemStyle: { color: color },
        areaStyle: { opacity: 0.2, color: color },
        showSymbol: false,
        smooth: true,
        z: 10,
      },
      {
        name: 'Normal',
        type: 'line',
        data: normalData,
        itemStyle: { color: t.textMuted }, // muted = normal/baseline
        lineStyle: { type: 'dashed', width: 2 },
        areaStyle: { opacity: 0.1, color: t.textMuted },
        showSymbol: false,
        smooth: true,
        z: 1,
      },
    ],
  });

  const tempOption = useMemo(
    () =>
      getOption('Temperature', data.temperature.actual, data.temperature.normal, '°C', t.warm),
    [data, t]
  );
  const humOption = useMemo(
    () => getOption('Humidity', data.humidity.actual, data.humidity.normal, '%', t.info),
    [data, t]
  );
  const precipOption = useMemo(
    () => ({
      title: { text: 'Precipitation', left: 'center', textStyle: { color: t.textSecondary, fontSize: 14 } },
      tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis' },
      grid: { left: 40, right: 20, top: 40, bottom: 20 },
      xAxis: { type: 'category', data: blocks, axisLabel: { color: t.textMuted, interval: 7 } },
      yAxis: {
        type: 'value',
        axisLabel: { color: t.textMuted },
        splitLine: { lineStyle: { color: t.outline } },
      },
      series: [
        {
          name: 'Precip',
          type: 'bar',
          data: data.precipitation.actual,
          itemStyle: { color: t.accent2 },
          barWidth: 4, // Thin bars for timeline effect
          animationDelay: (idx) => idx * 10,
        },
      ],
    }),
    [data, t]
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="h-64 bg-[rgba(var(--panel-rgb),0.5)] rounded-lg p-2 border border-[color:var(--outline)]">
        <ReactECharts option={tempOption} style={{ height: '100%', width: '100%' }} />
      </div>
      <div className="h-64 bg-[rgba(var(--panel-rgb),0.5)] rounded-lg p-2 border border-[color:var(--outline)]">
        <ReactECharts option={humOption} style={{ height: '100%', width: '100%' }} />
      </div>
      <div className="h-48 bg-[rgba(var(--panel-rgb),0.5)] rounded-lg p-2 border border-[color:var(--outline)] md:col-span-2">
        <ReactECharts option={precipOption} style={{ height: '100%', width: '100%' }} />
      </div>
    </div>
  );
}
