import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useChartTokens, themedEcBase } from '../../lib/chartTheme';
import { CloudRain, CloudLightning, Activity } from 'lucide-react';

export default function RainCloud({ data, intraday }) {
  const t = useChartTokens();
  if (!data) return null;

  const combinedOption = useMemo(() => {
    if (!intraday) return null;
    const rain = intraday.precipitation?.actual || [];
    const cloud = intraday.cloud_cover?.actual || [];
    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);

    return {
      title: {
        text: 'Rain & Cloud Cover',
        left: 'center',
        textStyle: { color: t.textSecondary, fontSize: 14 },
      },
      tooltip: { ...themedEcBase(t).tooltip, trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { top: 20, textStyle: { color: t.textMuted } },
      grid: { left: 40, right: 40, bottom: 20 },
      xAxis: {
        type: 'category',
        data: blocks,
        axisLabel: { color: t.textMuted, interval: 7 },
        axisPointer: { type: 'shadow' },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Rain (mm)',
          position: 'left',
          axisLabel: { color: t.accent2 },
          splitLine: { lineStyle: { color: t.outline } },
          nameTextStyle: { color: t.accent2 },
        },
        {
          type: 'value',
          name: 'Cloud (%)',
          min: 0,
          max: 100,
          position: 'right',
          axisLabel: { color: t.textMuted },
          splitLine: { show: false },
          nameTextStyle: { color: t.textMuted },
        },
      ],
      series: [
        {
          name: 'Rain',
          type: 'bar',
          data: rain,
          itemStyle: { color: t.accent2 },
          barWidth: 4,
          yAxisIndex: 0,
        },
        {
          name: 'Cloud Cover',
          type: 'line',
          data: cloud,
          itemStyle: { color: t.textMuted },
          areaStyle: { opacity: 0.3, color: t.textMuted },
          showSymbol: false,
          smooth: true,
          yAxisIndex: 1,
        },
      ],
    };
  }, [intraday, t]);

  const solarGaugeOption = useMemo(() => {
    if (!intraday) return null;
    const cloud = intraday.cloud_cover?.actual || [];
    // Avg cloud cover (simple mean)
    const avgCloud = cloud.length > 0 ? cloud.reduce((a, b) => a + b, 0) / cloud.length : 0;
    const solarAvail = Math.round(100 - avgCloud);

    return {
      title: {
        text: 'Solar Availability',
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
          max: 100,
          splitNumber: 5,
          itemStyle: { color: t.warning },
          progress: { show: true, width: 10 },
          pointer: { show: false },
          axisLine: {
            lineStyle: {
              width: 10,
              color: [
                [0.3, t.danger],
                [0.7, t.warning],
                [1, t.success],
              ],
            },
          }, // Red (Low) to Green (High)
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          detail: {
            valueAnimation: true,
            offsetCenter: [0, '20%'],
            fontSize: 20,
            color: 'inherit',
            formatter: '{value}%',
          },
          data: [{ value: solarAvail, name: 'Potential' }],
        },
      ],
    };
  }, [intraday, t]);

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="grid grid-cols-3 gap-4 shrink-0">
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-4 flex flex-col items-center justify-center">
          <CloudRain className="text-[color:var(--accent2)] mb-2" size={24} />
          <span className="text-xl font-bold text-[color:var(--text)]">{data.total_mm} mm</span>
          <span className="text-[10px] text-[color:var(--text-muted)] uppercase tracking-wider mt-1">
            Total Scale
          </span>
        </div>
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-4 flex flex-col items-center justify-center">
          <Activity className="text-[color:var(--warning)] mb-2" size={24} />
          <span className="text-xl font-bold text-[color:var(--text)]">{data.max_intensity}</span>
          <span className="text-[10px] text-[color:var(--text-muted)] uppercase tracking-wider mt-1">
            Max Int/Blk
          </span>
        </div>
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-4 flex flex-col items-center justify-center">
          <CloudLightning className="text-[color:var(--info)] mb-2" size={24} />
          <span className="text-xl font-bold text-[color:var(--text)]">{data.rain_hours} h</span>
          <span className="text-[10px] text-[color:var(--text-muted)] uppercase tracking-wider mt-1">Duration</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="md:col-span-2 bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[350px]">
          {combinedOption && (
            <ReactECharts option={combinedOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
        <div className="bg-[rgba(var(--panel-rgb),0.5)] border border-[color:var(--outline)] rounded-lg p-2 min-h-[350px]">
          {solarGaugeOption && (
            <ReactECharts option={solarGaugeOption} style={{ height: '100%', width: '100%' }} />
          )}
        </div>
      </div>
    </div>
  );
}
