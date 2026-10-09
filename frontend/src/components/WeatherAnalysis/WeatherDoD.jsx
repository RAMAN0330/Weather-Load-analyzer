import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useChartTokens, themedEcBase } from '../../lib/chartTheme';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

export default function WeatherDoD({ data, series }) {
  const t = useChartTokens();
  if (!data)
    return (
      <div className="p-4 text-[color:var(--text-muted)] italic">No comparison data available for this date.</div>
    );

  const metrics = [
    { key: 'max_temp', label: 'Max Temp', unit: '°C' },
    { key: 'min_temp', label: 'Min Temp', unit: '°C' },
    { key: 'avg_humidity', label: 'Avg Hum', unit: '%' },
  ];

  const blocks = Array.from({ length: 96 }, (_, i) => i + 1);

  const getOption = (title, values, unit, isHum = false) => ({
    title: { text: title, left: 'center', textStyle: { color: t.textSecondary, fontSize: 14 } },
    tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis' },
    grid: { left: 40, right: 20, top: 40, bottom: 20 },
    xAxis: {
      type: 'category',
      data: blocks,
      axisLabel: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: t.textMuted },
      splitLine: { lineStyle: { color: t.outline } },
    },
    series: [
      {
        name: 'Change',
        type: 'bar',
        data: values,
        barWidth: 4,
        itemStyle: {
          color: (params) => {
            const val = params.value;
            if (isHum) return val > 0 ? t.info : t.warm; // Wet vs Dry
            return val > 0 ? t.danger : t.info; // Hot vs Cold
          },
        },
      },
    ],
  });

  const tempOption = useMemo(() => {
    if (!series?.temperature) return null;
    return getOption('Temperature Change vs Yesterday', series.temperature, '°C');
  }, [series, t]);

  const humOption = useMemo(() => {
    if (!series?.humidity) return null;
    return getOption('Humidity Change vs Yesterday', series.humidity, '%', true);
  }, [series, t]);

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="grid grid-cols-3 gap-4 shrink-0">
        {metrics.map((m) => {
          const d = data[m.key];
          const change = parseFloat(d?.change || 0);
          const color =
            change > 0 ? 'text-[color:var(--danger)]' : change < 0 ? 'text-[color:var(--info)]' : 'text-[color:var(--text-muted)]';
          const Icon = change > 0 ? ArrowUp : change < 0 ? ArrowDown : Minus;

          return (
            <div
              key={m.key}
              className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-3 flex justify-between items-center"
            >
              <div>
                <h4 className="text-xs font-semibold text-[color:var(--text-muted)] uppercase">{m.label}</h4>
                <div className="text-xl font-bold text-[color:var(--text)] mt-1">
                  {d?.today}
                  {m.unit}
                </div>
              </div>
              <div className={`flex flex-col items-end ${color}`}>
                <Icon size={20} />
                <span className="text-sm font-bold">{Math.abs(change).toFixed(1)}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex-1 grid grid-cols-1 gap-4 min-h-0">
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[320px]">
          {tempOption && (
            <ReactECharts option={tempOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[320px]">
          {humOption && (
            <ReactECharts option={humOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
      </div>
    </div>
  );
}
