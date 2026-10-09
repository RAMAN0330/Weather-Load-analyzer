import { AlertTriangle, ChevronDown, Info, XOctagon } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../../../components/ui/popover';
import { cn } from '../../../lib/utils';
import type { ForecastResponse } from '../api';
import { useThemeScopeClass } from '../theme';
import { fmtPctVal } from '../utils';

/**
 * Compact single-line quality strip. Shown whenever the run is not `ok` or any
 * warning exists. The first warning is visible inline (truncated); "Details"
 * opens a popover with every warning — warnings are never hidden.
 */
export function QualityBanner({ forecast }: { forecast: ForecastResponse }) {
  const scope = useThemeScopeClass();
  const q = forecast.quality;
  const status =
    forecast.status === 'failed' || q?.status === 'failed'
      ? 'failed'
      : forecast.status === 'degraded' || q?.status === 'degraded'
        ? 'degraded'
        : 'ok';
  const warnings = q?.warnings ?? [];
  if (status === 'ok' && warnings.length === 0) return null;

  const failed = status === 'failed';
  const Icon = failed ? XOctagon : status === 'degraded' ? AlertTriangle : Info;
  const title = failed
    ? 'Forecast failed — do not schedule'
    : status === 'degraded'
      ? 'Degraded inputs'
      : 'Data notes';
  const tone = failed
    ? 'border-danger/50 bg-danger/10 [&_.tone]:text-danger'
    : status === 'degraded'
      ? 'border-warning/45 bg-warning/10 [&_.tone]:text-warning'
      : 'border-info/40 bg-info/10 [&_.tone]:text-info';

  return (
    <div
      role="alert"
      className={cn('flex h-8 shrink-0 items-center gap-2 rounded-lg border px-3 text-xs', tone)}
    >
      <Icon className="tone size-3.5 shrink-0" aria-hidden />
      <span className="shrink-0 font-semibold text-fg">
        {title}
        <span className="font-normal text-muted">
          {' '}
          · {warnings.length} warning{warnings.length === 1 ? '' : 's'}
        </span>
      </span>
      {warnings[0] && (
        <span className="min-w-0 truncate text-fg/85" title={warnings[0]}>
          — {warnings[0]}
        </span>
      )}
      <Popover>
        <PopoverTrigger
          className="ml-auto inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-0.5 font-semibold text-fg hover:bg-elev3/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Show all ${warnings.length} quality warnings`}
        >
          Details
          <ChevronDown className="size-3" aria-hidden />
        </PopoverTrigger>
        <PopoverContent align="end" className={cn(scope, 'glass-strong w-[30rem] max-w-[90vw]')}>
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
            <Icon className="tone size-4" aria-hidden />
            {title}
          </div>
          {warnings.length ? (
            <ul className="max-h-60 list-disc space-y-1 overflow-auto pl-5 text-sm">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted">No specific warnings were returned by the engine.</p>
          )}
          <p className="mt-3 border-t border-border pt-2 font-mono text-2xs text-muted">
            weather coverage {fmtPctVal(q?.weather_coverage_pct)} · load-share{' '}
            {fmtPctVal(q?.load_share_coverage_pct)} · history {q?.history_days ?? '—'} d · missing
            load blocks {q?.missing_load_blocks ?? '—'}
          </p>
        </PopoverContent>
      </Popover>
    </div>
  );
}
