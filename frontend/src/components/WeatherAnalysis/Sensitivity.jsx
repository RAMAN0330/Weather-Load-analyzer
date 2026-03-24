import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

export default function Sensitivity({ data, series }) {
    if (!data || !series) return null;

    // Use the backend's explicit driver_contributions array
    const drivers = series.driver_contributions || [];

    // Filter out zero-impact drivers to keep the UI clean (ML focus on > 98% accuracy metrics relevance!)
    const activeDrivers = drivers.filter(d => Math.abs(d.mw) > 0.5);

    const waterfallOption = useMemo(() => {
        if (!activeDrivers.length) return {};

        let cumulative = 0;
        const categories = [];
        const offsets = [];
        const impactData = [];

        activeDrivers.forEach((d) => {
            const mw = d.mw;
            categories.push(d.factor);
            if (mw >= 0) {
                offsets.push(cumulative);
            } else {
                offsets.push(cumulative + mw);
            }
            impactData.push({
                value: mw,
                itemStyle: { color: d.color || (mw >= 0 ? '#ef4444' : '#3b82f6') }
            });
            cumulative += mw;
        });

        // Add Net result
        categories.push('Net Exogenous');
        offsets.push(0);
        impactData.push({
            value: cumulative,
            itemStyle: { color: '#38bdf8' } // Cyan for net
        });

        return {
            title: { text: 'Exogenous Impact Waterfall', left: 'center', textStyle: { color: '#ccc', fontSize: 14 } },
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                formatter: (params) => {
                    const idx = params?.[1]?.dataIndex ?? params?.[0]?.dataIndex ?? 0;
                    const label = categories[idx];
                    const val = impactData[idx]?.value ?? 0;
                    return `${label}: ${val >= 0 ? '+' : ''}${val.toFixed(1)} MW`;
                }
            },
            grid: { left: 80, right: 30, top: 50, bottom: 40 },
            xAxis: {
                type: 'category',
                data: categories,
                axisLabel: { color: '#aaa', rotate: 30, interval: 0 },
                axisLine: { lineStyle: { color: '#333' } }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#aaa', formatter: '{value} MW' },
                splitLine: { lineStyle: { color: '#1f2937' } }
            },
            series: [
                {
                    type: 'bar',
                    stack: 'total',
                    itemStyle: { color: 'transparent' },
                    data: offsets
                },
                {
                    type: 'bar',
                    stack: 'total',
                    label: {
                        show: true,
                        position: 'top',
                        formatter: (p) => `${p.value >= 0 ? '+' : ''}${p.value.toFixed(0)}`
                    },
                    data: impactData
                }
            ]
        };
    }, [activeDrivers]);

    const pieOption = useMemo(() => {
        if (!activeDrivers.length) return {};

        const pieData = activeDrivers.map(d => ({
            name: d.factor,
            value: Math.abs(d.mw),
            itemStyle: { color: d.color || '#9ca3af' }
        }));

        return {
            title: { text: 'Absolute Impact Distribution', left: 'center', textStyle: { color: '#ccc', fontSize: 14 } },
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c} MW ({d}%)'
            },
            series: [
                {
                    type: 'pie',
                    radius: ['40%', '70%'],
                    avoidLabelOverlap: true,
                    itemStyle: {
                        borderRadius: 5,
                        borderColor: '#0f172a',
                        borderWidth: 2
                    },
                    label: {
                        show: true,
                        formatter: '{b}\n{d}%',
                        color: '#cbd5e1'
                    },
                    data: pieData
                }
            ]
        };
    }, [activeDrivers]);

    if (!activeDrivers.length) {
        return <div className="muted p-4">No significant exogenous drivers active for this forecast.</div>;
    }

    return (
        <div className="flex flex-col gap-6 p-2 min-h-0">
            <div className="bg-slate-900/20 border border-slate-800 rounded-lg p-4">
                <div className="mb-2">
                    <h4 className="text-sm font-semibold text-slate-200">Exogenous Accuracy Attributions (ML Context)</h4>
                    <p className="text-xs text-slate-400 mt-1">Breaks down the distinct systemic components shifting the forecast day-over-day.</p>
                </div>
                <div className="min-h-[350px] h-[420px] overflow-hidden">
                    <ReactECharts option={waterfallOption} style={{ height: '100%', width: '100%' }} />
                </div>
            </div>

            <div className="bg-slate-900/20 border border-slate-800 rounded-lg p-4">
                <div className="min-h-[300px] h-[380px] overflow-hidden">
                    <ReactECharts option={pieOption} style={{ height: '100%', width: '100%' }} />
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {activeDrivers.slice(0, 4).map((d, i) => (
                    <div key={i} className="bg-slate-900/50 border border-slate-700 hover:border-slate-500 transition-colors rounded-lg p-4 flex flex-col justify-center items-center">
                        <span className="text-xs font-medium text-slate-400 mb-1 text-center truncate w-full">{d.factor}</span>
                        <span className="text-xl font-bold" style={{ color: d.color || '#fff' }}>
                            {d.mw > 0 ? '+' : ''}{Math.round(d.mw)} MW
                        </span>
                        <span className="text-[10px] text-slate-500 mt-1">{d.pct.toFixed(1)}% of total shift</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
