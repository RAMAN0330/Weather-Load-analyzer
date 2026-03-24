import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';

export default function PeakWindowStress({ data, intraday }) {
    if (!data) return null;

    const windows = ['Morning', 'Midday', 'Evening', 'Night'];
    const blockTime = (block) => {
        const minutes = (block - 1) * 15;
        const h = String(Math.floor(minutes / 60)).padStart(2, '0');
        const m = String(minutes % 60).padStart(2, '0');
        return `${h}:${m}`;
    };

    const barOption = useMemo(() => {
        const tempDeltas = windows.map(w => data[w]?.temp_delta || 0);
        const humDeltas = windows.map(w => data[w]?.hum_delta || 0);

        return {
            title: { text: 'Window Comparison', left: 'center', textStyle: { color: '#ccc', fontSize: 14 } },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            legend: { top: 20, textStyle: { color: '#aaa' } },
            grid: { left: 40, right: 20, bottom: 20 },
            xAxis: {
                type: 'category',
                data: windows,
                axisLabel: { color: '#aaa' },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#aaa' },
                splitLine: { lineStyle: { color: '#333' } }
            },
            series: [
                {
                    name: 'Temp Delta',
                    type: 'bar',
                    data: tempDeltas,
                    itemStyle: { color: '#ef4444' }
                },
                {
                    name: 'Hum Delta',
                    type: 'bar',
                    data: humDeltas,
                    itemStyle: { color: '#3b82f6' }
                }
            ]
        };
    }, [data]);

    const gaugeOption = useMemo(() => {
        const maxDelta = Math.max(...windows.map(w => Math.abs(data[w]?.temp_delta || 0)));

        return {
            title: { text: 'Peak Heat Stress', left: 'center', top: 10, textStyle: { color: '#ccc', fontSize: 14 } },
            series: [
                {
                    type: 'gauge',
                    startAngle: 180,
                    endAngle: 0,
                    min: 0,
                    max: 10,
                    splitNumber: 5,
                    itemStyle: { color: '#ef4444' },
                    progress: { show: true, width: 10 },
                    pointer: { show: false },
                    axisLine: { lineStyle: { width: 10, color: [[0.3, '#10b981'], [0.7, '#f59e0b'], [1, '#ef4444']] } },
                    axisTick: { show: false },
                    splitLine: { show: false },
                    axisLabel: { show: false },
                    detail: {
                        valueAnimation: true,
                        offsetCenter: [0, '20%'],
                        fontSize: 20,
                        color: 'inherit',
                        formatter: '{value} C Delta'
                    },
                    data: [{ value: maxDelta, name: 'Max Deviation' }]
                }
            ]
        };
    }, [data]);

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
        const direction = shiftBlocks === 0 ? 'No Shift' : (shiftBlocks > 0 ? 'Later' : 'Earlier');
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
            actualTime: blockTime(idxAct)
        };
    }, [intraday]);

    const peakImpactOption = useMemo(() => {
        const tempDeltas = windows.map(w => Number(data[w]?.temp_delta || 0));
        const humDeltas = windows.map(w => Number(data[w]?.hum_delta || 0));
        const precip = windows.map(w => Number(data[w]?.precip_total || 0));
        const impactIndex = windows.map((_, i) => (tempDeltas[i] * 1.2) + (humDeltas[i] * 0.4) - (precip[i] * 0.15));
        return {
            title: { text: 'Peak Window Impact', left: 'center', textStyle: { color: '#ccc', fontSize: 13 } },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: '{b}: {c} MW' },
            grid: { left: 45, right: 20, top: 35, bottom: 20 },
            xAxis: {
                type: 'category',
                data: windows,
                axisLabel: { color: '#aaa' },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#aaa' },
                splitLine: { lineStyle: { color: '#333' } }
            },
            series: [{
                type: 'bar',
                data: impactIndex,
                itemStyle: { color: '#f59e0b' }
            }]
        };
    }, [data]);

    const rampOption = useMemo(() => {
        const deltaSeries = intraday?.temperature?.delta || [];
        if (!deltaSeries.length) return null;
        const ramp = deltaSeries.map((val, idx) => {
            if (idx === 0) return 0;
            return (Number(val) || 0) - (Number(deltaSeries[idx - 1]) || 0);
        });
        return {
            title: { text: 'Ramp Acceleration', left: 'center', textStyle: { color: '#ccc', fontSize: 13 } },
            tooltip: { trigger: 'axis', formatter: '{b}: {c} MW' },
            grid: { left: 45, right: 20, top: 35, bottom: 20 },
            xAxis: {
                type: 'category',
                data: ramp.map((_, i) => i + 1),
                axisLabel: { color: '#aaa', interval: 7 },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                axisLabel: { color: '#aaa' },
                splitLine: { lineStyle: { color: '#333' } }
            },
            series: [{
                type: 'line',
                data: ramp,
                smooth: true,
                lineStyle: { color: '#38bdf8', width: 2 },
                itemStyle: { color: '#38bdf8' }
            }]
        };
    }, [intraday]);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full">
            <div className="lg:col-span-2 flex flex-col gap-4">
                <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[260px]">
                    <ReactECharts option={barOption} style={{ height: '320px', width: '100%' }} />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[220px]">
                        <ReactECharts option={peakImpactOption} style={{ height: '280px', width: '100%' }} />
                    </div>
                    <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[220px]">
                        {rampOption
                             ? <ReactECharts option={rampOption} style={{ height: '280px', width: '100%' }} />
                            : <div className="h-full flex items-center justify-center text-slate-500 text-sm">No ramp data.</div>}
                    </div>
                </div>
            </div>
            <div className="flex flex-col gap-4">
                <div className="flex-1 bg-slate-900/50 border border-slate-700 rounded-lg p-2 min-h-[140px]">
                    <ReactECharts option={gaugeOption} style={{ height: '100%', width: '100%' }} />
                </div>
                {peakShift && (
                    <div className="flex-1 bg-slate-900/50 border border-slate-700 rounded-lg p-3 min-h-[140px] flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold text-slate-200">Peak Shift Timeline</span>
                            <span className={`text-[11px] px-2 py-1 rounded-full border ${peakShift.significant ? 'border-red-400/60 text-red-300' : 'border-emerald-400/50 text-emerald-300'}`}>
                                {peakShift.significant ? 'Significant' : 'Minor'}
                            </span>
                        </div>
                        <div className="text-xs text-slate-400">
                            {peakShift.direction} by <span className="text-slate-100 font-semibold">{Math.abs(peakShift.shiftBlocks)} blocks</span>
                            {' '}({peakShift.minutes} min)
                        </div>
                        <div className="relative h-3 rounded-full bg-slate-800/80 border border-slate-700">
                            <div
                                className="absolute top-0 h-full rounded-full bg-cyan-400/60"
                                style={{ left: `${peakShift.left}%`, width: `${peakShift.width}%` }}
                            />
                            <div
                                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-slate-300 border border-slate-100"
                                style={{ left: `calc(${peakShift.normPct}% - 6px)` }}
                                title={`Normal Peak: Block ${peakShift.idxNorm}`}
                            />
                            <div
                                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-red-400 border border-red-200"
                                style={{ left: `calc(${peakShift.actPct}% - 6px)` }}
                                title={`Actual Peak: Block ${peakShift.idxAct}`}
                            />
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-400">
                            <span>Normal: Block {peakShift.idxNorm} • {peakShift.normalTime}</span>
                            <span>Actual: Block {peakShift.idxAct} • {peakShift.actualTime}</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
