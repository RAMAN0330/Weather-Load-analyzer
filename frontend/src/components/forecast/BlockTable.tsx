import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from '../ui/card';
import { ScrollArea } from '../ui/scroll-area';
import { Skeleton } from '../ui/skeleton';
import type { ForecastResponse } from '../../types/api';
import { blockToTime, fmtMw } from '../../lib/utils';

interface Props {
  data: ForecastResponse | undefined;
  loading?: boolean;
}

export function BlockTable({ data, loading }: Props) {
  const rows = useMemo(() => {
    const series = data?.series;
    if (!series?.forecast) return [];
    const f = series.forecast;
    const a = series.actual || [];
    const b = series.hybrid_baseline || [];
    return Array.from({ length: 96 }, (_, i) => ({
      block: i + 1,
      time: blockToTime(i + 1),
      forecast: f[i],
      actual: a[i] ?? null,
      baseline: b[i] ?? null,
    }));
  }, [data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>96-block table</CardTitle>
        <CardSubtitle>15-min granularity, full day</CardSubtitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-72 w-full" />
        ) : (
          <ScrollArea className="h-72 rounded-md border border-border">
            <table className="w-full text-sm tabular">
              <thead className="sticky top-0 bg-elev2 text-2xs uppercase tracking-wider text-faint">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">Block</th>
                  <th className="px-3 py-2 text-left font-semibold">Time</th>
                  <th className="px-3 py-2 text-right font-semibold">Forecast (MW)</th>
                  <th className="px-3 py-2 text-right font-semibold">Baseline</th>
                  <th className="px-3 py-2 text-right font-semibold">Actual</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.block} className="border-t border-border/50 hover:bg-elev2/50">
                    <td className="px-3 py-1.5 font-mono text-faint">{r.block}</td>
                    <td className="px-3 py-1.5 font-mono text-muted">{r.time}</td>
                    <td className="px-3 py-1.5 text-right font-mono font-semibold text-fg">{fmtMw(r.forecast, 0)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">{fmtMw(r.baseline, 0)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-fg">{fmtMw(r.actual, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
