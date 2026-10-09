import { useState } from 'react';
import { CloudSun, Database, Info } from 'lucide-react';
import type { ReactNode } from 'react';
import { Alert, AlertDescription, AlertTitle } from '../../../components/ui/alert';
import { Skeleton } from '../../../components/ui/skeleton';
import { ToggleGroup, ToggleGroupItem } from '../../../components/ui/toggle-group';
import { cn } from '../../../lib/utils';
import { useDataQuality, type ForecastResponse } from '../api';
import { fmtDateLong, fmtInt, fmtPctVal, isNum } from '../utils';
import { ErrorState } from './States';
import { StatusBadge } from './StatusBadge';

function Meter({ value, label }: { value: number | null; label: string }) {
  const v = isNum(value) ? Math.max(0, Math.min(100, value)) : 0;
  const tone = !isNum(value)
    ? 'bg-faint'
    : v >= 98
      ? 'bg-success'
      : v >= 90
        ? 'bg-warning'
        : 'bg-danger';
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-2xs uppercase tracking-wider text-muted">{label}</span>
        <span className="font-mono text-lg font-semibold tabular-nums text-fg">
          {fmtPctVal(value, 1)}
        </span>
      </div>
      <div
        className="h-1.5 overflow-hidden rounded-full bg-elev3"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={isNum(value) ? Number(value.toFixed(1)) : undefined}
      >
        <div className={cn('h-full rounded-full', tone)} style={{ width: `${v}%` }} />
      </div>
    </div>
  );
}

function Stat({ label, value, warn }: { label: string; value: ReactNode; warn?: boolean }) {
  return (
    <div className="flex items-center justify-between border-b border-border/60 py-1.5 text-xs last:border-0">
      <span className="text-muted">{label}</span>
      <span className={cn('font-mono tabular-nums', warn ? 'text-warning' : 'text-fg')}>
        {value}
      </span>
    </div>
  );
}

export function DataQualityPanel({
  region,
  forecast,
}: {
  region: string;
  forecast?: ForecastResponse;
}) {
  const [days, setDays] = useState(30);
  const q = useDataQuality(region, days);
  const d = q.data;
  const loadPct =
    d && d.load.expected_blocks > 0 ? (d.load.present_blocks / d.load.expected_blocks) * 100 : null;
  const fq = forecast?.quality;

  const card = 'data-surface flex min-h-0 flex-col gap-3 overflow-auto rounded-xl p-3';
  const warnTone =
    d?.status === 'failed' ? 'destructive' : d?.status === 'degraded' ? 'warning' : 'info';

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Input data quality
          </h3>
          {d && <StatusBadge status={d.status} />}
          {d && (
            <span className="truncate font-mono text-2xs text-faint">
              {fmtDateLong(d.from)} → {fmtDateLong(d.to)}
            </span>
          )}
        </div>
        <ToggleGroup
          type="single"
          value={String(days)}
          onValueChange={(v) => v && setDays(Number(v))}
          aria-label="Look-back window"
          className="h-8 p-0.5"
        >
          {[7, 30, 90].map((n) => (
            <ToggleGroupItem
              key={n}
              value={String(n)}
              aria-label={`Last ${n} days`}
              className="h-6 px-2.5"
            >
              {n}d
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {q.isLoading ? (
        <div
          className="grid min-h-0 flex-1 auto-rows-[minmax(0,1fr)] gap-2 md:grid-cols-2 xl:grid-cols-3"
          aria-busy
        >
          <Skeleton className="h-full rounded-xl" />
          <Skeleton className="h-full rounded-xl" />
          <Skeleton className="h-full rounded-xl" />
        </div>
      ) : q.isError ? (
        <ErrorState
          className="min-h-0 flex-1"
          title="Could not load data-quality report"
          message={q.error.message}
          onRetry={() => q.refetch()}
          retrying={q.isFetching}
        />
      ) : d ? (
        <div className="grid min-h-0 flex-1 auto-rows-[minmax(0,1fr)] gap-2 md:grid-cols-2 xl:grid-cols-3">
          <section className={card} aria-label="Load data">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted">
              <Database className="size-3.5 text-accent" aria-hidden /> Load (SCADA drawal)
            </div>
            <Meter value={loadPct} label="Block completeness" />
            <div>
              <Stat
                label="Present / expected blocks"
                value={`${fmtInt(d.load.present_blocks)} / ${fmtInt(d.load.expected_blocks)}`}
              />
              <Stat
                label="Duplicate blocks"
                value={fmtInt(d.load.duplicate_blocks)}
                warn={d.load.duplicate_blocks > 0}
              />
              <Stat
                label="Non-positive blocks"
                value={fmtInt(d.load.nonpositive_blocks)}
                warn={d.load.nonpositive_blocks > 0}
              />
              <Stat
                label="Missing dates"
                value={d.load.missing_dates.length}
                warn={d.load.missing_dates.length > 0}
              />
            </div>
            {d.load.missing_dates.length > 0 && (
              <div className="flex flex-wrap gap-1.5" aria-label="Missing dates">
                {d.load.missing_dates.map((m) => (
                  <span
                    key={m}
                    className="rounded border border-warning/40 bg-warning/10 px-1.5 py-0.5 font-mono text-2xs text-warning"
                  >
                    {m}
                  </span>
                ))}
              </div>
            )}
          </section>

          <section className={card} aria-label="Weather data">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted">
              <CloudSun className="size-3.5 text-accent2" aria-hidden /> Weather (district grid)
            </div>
            <Meter value={d.weather.coverage_pct} label="District coverage" />
            <div>
              <Stat
                label="Districts seen / expected"
                value={`${d.weather.districts_seen} / ${d.weather.districts_expected}`}
                warn={d.weather.districts_seen < d.weather.districts_expected}
              />
              <Stat
                label="Missing districts"
                value={d.weather.missing_districts.length}
                warn={d.weather.missing_districts.length > 0}
              />
            </div>
            {d.weather.missing_districts.length > 0 ? (
              <div className="flex flex-wrap gap-1.5" aria-label="Missing districts">
                {d.weather.missing_districts.map((m) => (
                  <span
                    key={m}
                    className="rounded border border-danger/40 bg-danger/10 px-1.5 py-0.5 font-mono text-2xs text-danger"
                  >
                    {m}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-2xs text-success">All districts reporting.</p>
            )}
          </section>

          <section
            className={cn(card, 'md:col-span-2 xl:col-span-1')}
            aria-label="Warnings and run inputs"
          >
            {d.warnings?.length > 0 ? (
              <Alert variant={warnTone} className="px-3 py-2">
                <Info aria-hidden />
                <AlertTitle className="text-xs">
                  {d.warnings.length} data warning{d.warnings.length > 1 ? 's' : ''}
                </AlertTitle>
                <AlertDescription className="text-xs">
                  <ul className="mt-1 list-disc space-y-0.5 pl-4">
                    {d.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            ) : (
              <p className="text-xs text-success">No data warnings in this window.</p>
            )}
            {fq ? (
              <div>
                <div className="mb-1 flex items-center gap-2">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Inputs used by current run
                  </h4>
                  <StatusBadge status={fq.status} />
                </div>
                <Stat
                  label="Weather coverage"
                  value={fmtPctVal(fq.weather_coverage_pct)}
                  warn={(fq.weather_coverage_pct ?? 100) < 95}
                />
                <Stat
                  label="Load-share coverage"
                  value={fmtPctVal(fq.load_share_coverage_pct)}
                  warn={(fq.load_share_coverage_pct ?? 100) < 95}
                />
                <Stat
                  label="History"
                  value={isNum(fq.history_days) ? `${fq.history_days} days` : '—'}
                  warn={(fq.history_days ?? 999) < 28}
                />
                <Stat
                  label="Missing load blocks"
                  value={fmtInt(fq.missing_load_blocks)}
                  warn={(fq.missing_load_blocks ?? 0) > 0}
                />
              </div>
            ) : (
              <p className="text-2xs text-muted">Run a forecast to see the inputs it used.</p>
            )}
          </section>
        </div>
      ) : null}
    </div>
  );
}
