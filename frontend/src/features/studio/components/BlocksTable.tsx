import { memo, useMemo } from 'react';
import { cn } from '../../../lib/utils';
import type { ForecastBlock, ForecastResponse } from '../api';
import {
  blockError,
  blockRange,
  blockTime,
  errorTint,
  fmtInt,
  fmtSigned,
  isNum,
  isPeakWindow,
} from '../utils';

const COLS = [
  { key: 'time', label: 'Time', align: 'left', title: 'Block start (IST)' },
  { key: 'p10', label: 'P10', align: 'right', title: 'P10 MW' },
  { key: 'p50', label: 'P50', align: 'right', title: 'P50 MW' },
  { key: 'p90', label: 'P90', align: 'right', title: 'P90 MW' },
  { key: 'baseline', label: 'Base', align: 'right', title: 'Baseline MW' },
  { key: 'actual', label: 'Actual', align: 'right', title: 'Actual MW' },
  { key: 'err', label: 'Err %', align: 'right', title: 'Error (P50 − actual) as % of actual' },
] as const;

const cell = 'px-1.5 py-1';
const fmtGwh = (mwh: number) => (mwh / 1000).toFixed(2);

const Row = memo(function Row({ b, peak }: { b: ForecastBlock; peak: boolean }) {
  const e = blockError(b);
  const tint = errorTint(e.pct);
  const time = b.time || blockTime(b.block);
  return (
    <tr
      className={cn(
        'border-b border-border/50 transition-colors hover:bg-elev2/70',
        isPeakWindow(time) && 'bg-warning/[0.04]',
        peak && 'bg-accent/10 hover:bg-accent/15'
      )}
      aria-current={peak ? 'true' : undefined}
      title={`Block ${b.block} · ${blockRange(b.block)}`}
    >
      <td className={cn(cell, 'whitespace-nowrap text-fg/90')}>
        {time}
        {peak && (
          <span className="ml-1.5 rounded bg-accent/20 px-1 text-[10px] font-semibold uppercase text-accent">
            pk
          </span>
        )}
      </td>
      <td className={cn(cell, 'text-right text-muted')}>{fmtInt(b.p10)}</td>
      <td className={cn(cell, 'text-right font-semibold text-fg')}>{fmtInt(b.p50)}</td>
      <td className={cn(cell, 'text-right text-muted')}>{fmtInt(b.p90)}</td>
      <td className={cn(cell, 'text-right text-muted')}>{fmtInt(b.baseline)}</td>
      <td className={cn(cell, 'text-right text-fg/90')}>{fmtInt(b.actual)}</td>
      <td
        className={cn(cell, 'text-right')}
        style={{ backgroundColor: tint }}
        title={isNum(e.mw) ? `${fmtSigned(e.mw)} MW` : undefined}
      >
        {isNum(e.pct) ? fmtSigned(e.pct, 2) : '—'}
      </td>
    </tr>
  );
});

/** Fills its parent; rows scroll inside with sticky header and footer. */
export function BlocksTable({ forecast }: { forecast: ForecastResponse }) {
  const blocks = useMemo(
    () => [...(forecast.blocks ?? [])].sort((a, b) => a.block - b.block),
    [forecast.blocks]
  );
  const peakBlock = forecast.summary?.peak_block;
  const nActual = blocks.filter((b) => isNum(b.actual)).length;
  const totals = useMemo(() => {
    const sum = (k: 'p10' | 'p50' | 'p90' | 'baseline' | 'actual') =>
      blocks.reduce((s, b) => (isNum(b[k]) ? s + (b[k] as number) / 4 : s), 0);
    return {
      p10: sum('p10'),
      p50: sum('p50'),
      p90: sum('p90'),
      baseline: sum('baseline'),
      actual: sum('actual'),
    };
  }, [blocks]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        className="min-h-0 flex-1 overflow-auto"
        tabIndex={0}
        aria-label="Forecast blocks table, scrollable"
      >
        <table className="w-full border-collapse font-mono text-2xs tabular-nums">
          <caption className="sr-only">
            96 fifteen-minute blocks in MW. Error = P50 minus actual; positive = over-forecast.
          </caption>
          <thead className="sticky top-0 z-10 bg-elev2">
            <tr className="border-b border-border">
              {COLS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  title={c.title}
                  className={cn(
                    'h-8 whitespace-nowrap px-1.5 font-sans text-2xs font-semibold uppercase tracking-wider text-muted',
                    c.align === 'right' ? 'text-right' : 'text-left'
                  )}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {blocks.map((b) => (
              <Row key={b.block} b={b} peak={b.block === peakBlock} />
            ))}
          </tbody>
          <tfoot className="sticky bottom-0 bg-elev2">
            <tr className="border-t border-border text-fg" title="Day energy per column, GWh">
              <td className="px-1.5 py-1.5 font-sans text-2xs font-semibold uppercase tracking-wider text-muted">
                GWh
              </td>
              <td className="px-1.5 py-1.5 text-right text-muted">{fmtGwh(totals.p10)}</td>
              <td className="px-1.5 py-1.5 text-right font-semibold">{fmtGwh(totals.p50)}</td>
              <td className="px-1.5 py-1.5 text-right text-muted">{fmtGwh(totals.p90)}</td>
              <td className="px-1.5 py-1.5 text-right text-muted">{fmtGwh(totals.baseline)}</td>
              <td className="px-1.5 py-1.5 text-right">{nActual ? fmtGwh(totals.actual) : '—'}</td>
              <td className="px-1.5 py-1.5 text-right text-2xs text-muted">{nActual}/96</td>
            </tr>
          </tfoot>
        </table>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-t border-border px-3 py-1.5 text-2xs text-muted">
        <span className="inline-flex items-center gap-1">
          <span className="size-2 rounded-sm bg-danger/40" aria-hidden /> under
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="size-2 rounded-sm bg-info/40" aria-hidden /> over
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="size-2 rounded-sm bg-warning/25" aria-hidden /> 17–23 h
        </span>
        <span className="ml-auto text-faint">Err = P50 − actual · “—” = missing</span>
      </div>
    </div>
  );
}
