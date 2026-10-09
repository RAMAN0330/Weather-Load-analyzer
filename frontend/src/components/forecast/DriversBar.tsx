import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import type { ForecastResponse } from '../../types/api';
import { blockToTime } from '../../lib/utils';
import { useChartTokens, withAlpha, type ChartTokens } from '../../lib/chartTheme';

interface Props {
  data: ForecastResponse | undefined;
  loading?: boolean;
}

const FEATURE_COLORS: Record<string, keyof ChartTokens> = {
  temperature: 'warm',
  humidity: 'info',
  precipitation: 'accent2',
  rain: 'accent',
  showers: 'info',
  snowfall: 'textSecondary',
  apparent_temperature: 'danger',
  cloud_cover: 'textMuted',
  wind: 'success',
  daytype: 'accent2',
  holiday: 'danger',
  manual: 'warning',
};

export function DriversBar({ data, loading }: Props) {
  const t = useChartTokens();
  const option = useMemo(() => {
    const series = data?.series as any;
    if (!series) return null;

    const blocks = Array.from({ length: 96 }, (_, i) => i + 1);
    const xLabels = blocks.map((b) => blockToTime(b));

    // Pull *_contrib_mw arrays from the series payload — these exist on the
    // backend response.
    const FEATURES = [
      'temperature',
      'humidity',
      'precipitation',
      'wind',
      'apparent',
      'cloud',
      'sunshine',
      'radiation',
    ];
    const seriesArr: any[] = [];
    for (const f of FEATURES) {
      const key = `${f}_contrib_mw`;
      const arr = series[key];
      if (!Array.isArray(arr) || arr.length !== 96) continue;
      const allZero = arr.every((v: number) => Math.abs(v) < 0.5);
      if (allZero) continue;
      seriesArr.push({
        name: f,
        type: 'bar',
        stack: 'drivers',
        data: arr,
        itemStyle: { color: t[FEATURE_COLORS[f]] || t.textMuted },
        emphasis: { focus: 'series' },
        barCategoryGap: '15%',
      });
    }

    return {
      grid: { top: 16, right: 16, bottom: 32, left: 56 },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: t.elevated,
        borderColor: t.outline,
        textStyle: { color: t.text, fontSize: 12 },
      },
      legend: {
        top: 0,
        right: 0,
        textStyle: { color: t.textMuted, fontSize: 11 },
        itemWidth: 10,
        itemHeight: 10,
      },
      xAxis: {
        type: 'category',
        data: xLabels,
        axisLabel: { color: t.textMuted, fontSize: 10, interval: 11 },
        axisLine: { lineStyle: { color: t.outline } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: { color: t.textMuted, fontSize: 10 },
        splitLine: { lineStyle: { color: withAlpha(t.outline, 0.6), type: 'dashed' } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: seriesArr,
      animationDuration: 320,
    };
  }, [data, t]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Per-block drivers</CardTitle>
        <CardSubtitle>Stacked MW contribution by feature</CardSubtitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-64 w-full" />
        ) : option && (option.series as any[]).length > 0 ? (
          <ReactECharts option={option} style={{ height: 280, width: '100%' }} notMerge lazyUpdate />
        ) : (
          <div className="grid h-64 place-items-center text-sm text-muted">No driver decomposition available.</div>
        )}
      </CardContent>
    </Card>
  );
}
