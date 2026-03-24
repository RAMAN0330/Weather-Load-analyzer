import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

export default function IntradayWeather({ data }) {
    if (!data) return null;

    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);

    const getOption = (title, actualData, normalData, unit, color) => ({
        title: {
            text: title,
            left: 'center',
            textStyle: { color: '#ccc', fontSize: 14 }
        },
        tooltip: { trigger: 'axis' },
        legend: { top: 20, textStyle: { color: '#aaa' } },
        grid: { left: 40, right: 20, top: 50, bottom: 20 },
        xAxis: {
            type: 'category',
            data: blocks,
            axisLabel: { color: '#aaa', interval: 7 }
        },
        yAxis: {
            type: 'value',
            axisLabel: { color: '#aaa', formatter: `{value} ${unit}` },
            splitLine: { lineStyle: { color: '#333' } }
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
                z: 10
            },
            {
                name: 'Normal',
                type: 'line',
                data: normalData,
                itemStyle: { color: '#94a3b8' }, // Slate-400
                lineStyle: { type: 'dashed', width: 2 },
                areaStyle: { opacity: 0.1, color: '#94a3b8' },
                showSymbol: false,
                smooth: true,
                z: 1
            }
        ]
    });

    const tempOption = useMemo(() => getOption('Temperature', data.temperature.actual, data.temperature.normal, '°C', '#ef4444'), [data]);
    const humOption = useMemo(() => getOption('Humidity', data.humidity.actual, data.humidity.normal, '%', '#3b82f6'), [data]);
    const precipOption = useMemo(() => ({
        title: { text: 'Precipitation', left: 'center', textStyle: { color: '#ccc', fontSize: 14 } },
        tooltip: { trigger: 'axis' },
        grid: { left: 40, right: 20, top: 40, bottom: 20 },
        xAxis: { type: 'category', data: blocks, axisLabel: { color: '#aaa', interval: 7 } },
        yAxis: { type: 'value', axisLabel: { color: '#aaa' }, splitLine: { lineStyle: { color: '#333' } } },
        series: [{
            name: 'Precip',
            type: 'bar',
            data: data.precipitation.actual,
            itemStyle: { color: '#60a5fa' },
            barWidth: 4, // Thin bars for timeline effect
            animationDelay: (idx) => idx * 10
        }]
    }), [data]);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="h-64 bg-slate-900/50 rounded-lg p-2 border border-slate-700">
                <ReactECharts option={tempOption} style={{ height: '100%', width: '100%' }} />
            </div>
            <div className="h-64 bg-slate-900/50 rounded-lg p-2 border border-slate-700">
                <ReactECharts option={humOption} style={{ height: '100%', width: '100%' }} />
            </div>
            <div className="h-48 bg-slate-900/50 rounded-lg p-2 border border-slate-700 md:col-span-2">
                <ReactECharts option={precipOption} style={{ height: '100%', width: '100%' }} />
            </div>
        </div>
    );
}
