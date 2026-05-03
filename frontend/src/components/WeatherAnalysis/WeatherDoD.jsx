import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

export default function WeatherDoD({ data, series }) {
  if (!data)
    return (
      <div className="p-4 text-slate-400 italic">No comparison data available for this date.</div>
    );

  const metrics = [
    { key: 'max_temp', label: 'Max Temp', unit: '°C' },
    { key: 'min_temp', label: 'Min Temp', unit: '°C' },
    { key: 'avg_humidity', label: 'Avg Hum', unit: '%' },
  ];

  const blocks = Array.from({ length: 96 }, (_, i) => i + 1);

  const getOption = (title, values, unit, isHum = false) => ({
    title: { text: title, left: 'center', textStyle: { color: '#ccc', fontSize: 14 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 40, right: 20, top: 40, bottom: 20 },
    xAxis: {
      type: 'category',
      data: blocks,
      axisLabel: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#aaa' },
      splitLine: { lineStyle: { color: '#333' } },
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
            if (isHum) return val > 0 ? '#3b82f6' : '#f59e0b'; // Blue (Wet) vs Orange (Dry)
            return val > 0 ? '#ef4444' : '#3b82f6'; // Red (Hot) vs Blue (Cold)
          },
        },
      },
    ],
  });

  const tempOption = useMemo(() => {
    if (!series?.temperature) return null;
    return getOption('Temperature Change vs Yesterday', series.temperature, '°C');
  }, [series]);

  const humOption = useMemo(() => {
    if (!series?.humidity) return null;
    return getOption('Humidity Change vs Yesterday', series.humidity, '%', true);
  }, [series]);

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="grid grid-cols-3 gap-4 shrink-0">
        {metrics.map((m) => {
          const d = data[m.key];
          const change = parseFloat(d?.change || 0);
          const color =
            change > 0 ? 'text-red-400' : change < 0 ? 'text-blue-400' : 'text-slate-400';
          const Icon = change > 0 ? ArrowUp : change < 0 ? ArrowDown : Minus;

          return (
            <div
              key={m.key}
              className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 flex justify-between items-center"
            >
              <div>
                <h4 className="text-xs font-semibold text-slate-400 uppercase">{m.label}</h4>
                <div className="text-xl font-bold text-slate-200 mt-1">
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
        <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[320px]">
          {tempOption && (
            <ReactECharts option={tempOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
        <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[320px]">
          {humOption && (
            <ReactECharts option={humOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
      </div>
    </div>
  );
}
