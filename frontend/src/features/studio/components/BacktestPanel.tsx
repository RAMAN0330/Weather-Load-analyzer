import { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { useReducedMotion } from 'framer-motion';
import { ChevronDown, FlaskConical, Info, Loader2, Play } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { Skeleton } from '../../../components/ui/skeleton';
import { ToggleGroup, ToggleGroupItem } from '../../../components/ui/toggle-group';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../../../components/ui/dropdown-menu';
import { themedBase, themedLegend, useChartPalette } from '../chartPalette';
import { cn } from '../../../lib/utils';
import { useBacktestV3, type BacktestResponse, type Horizon, type ModelInfo } from '../api';
import {
  daysBetweenInclusive,
  fmtFixed,
  fmtInt,
  fmtSigned,
  isNum,
  shiftDate,
  todayIST,
} from '../utils';
import { useThemeScopeClass } from '../theme';
import { EmptyState, ErrorState } from './States';

const MAX_DAYS = 62;
function seriesName(model: string, horizon: number, models: ModelInfo[]) {
  return `${models.find((m) => m.key === model)?.label ?? model} · T+${horizon}`;
}

function ResultsTable({ data, models }: { data: BacktestResponse; models: ModelInfo[] }) {
  const rows = useMemo(
    () =>
      [...data.results].sort(
        (a, b) => a.horizon - b.horizon || (a.metrics.wape ?? 1e9) - (b.metrics.wape ?? 1e9)
      ),
    [data.results]
  );
  const bestByHorizon = useMemo(() => {
    const m = new Map<number, number>();
    for (const r of data.results) {
      const w = r.metrics.wape;
      if (isNum(w) && (!m.has(r.horizon) || w < (m.get(r.horizon) as number))) m.set(r.horizon, w);
    }
    return m;
  }, [data.results]);

  const head = [
    ['Model', 'left', 'Model'],
    ['H', 'left', 'Horizon'],
    ['WAPE %', 'right', 'Weighted absolute percentage error'],
    ['WAPE-derived acc. %', 'right', 'WAPE-derived accuracy = 100 − WAPE'],
    ['MAE', 'right', 'Mean absolute error, MW'],
    ['RMSE', 'right', 'Root mean squared error, MW'],
    ['Bias', 'right', 'Mean(forecast − actual), MW; positive = over-forecast'],
    ['Peak WAPE %', 'right', 'WAPE over peak blocks'],
    ['P95 |err|', 'right', '95th percentile absolute error, MW'],
    ['Cov. %', 'right', 'P10–P90 interval coverage (nominal 80%)'],
    ['Days', 'right', 'Target dates evaluated'],
  ] as const;

  return (
    <div className="data-surface flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-xl">
      <div
        className="min-h-0 flex-1 overflow-auto"
        tabIndex={0}
        aria-label="Backtest metrics, scrollable"
      >
        <table className="w-full border-collapse font-mono text-xs tabular-nums">
          <caption className="sr-only">Backtest metrics by model and horizon</caption>
          <thead className="sticky top-0 z-10 bg-elev2">
            <tr className="border-b border-border">
              {head.map(([h, a, t]) => (
                <th
                  key={h}
                  scope="col"
                  title={t}
                  className={cn(
                    'h-9 px-1.5 py-1 font-sans text-2xs font-semibold uppercase leading-tight tracking-wide text-muted first:pl-3',
                    a === 'right' ? 'text-right' : 'text-left'
                  )}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = r.metrics;
              const best = isNum(m.wape) && bestByHorizon.get(r.horizon) === m.wape;
              const cov = m.interval_coverage_pct;
              const covWarn = isNum(cov) && Math.abs(cov - 80) > 10;
              return (
                <tr
                  key={`${r.model}-${r.horizon}`}
                  className={cn(
                    'border-b border-border/50 hover:bg-elev2/70',
                    best && 'bg-success/[0.06]'
                  )}
                >
                  <td className="whitespace-nowrap py-1.5 pl-3 pr-1.5 font-sans text-xs text-fg">
                    {models.find((x) => x.key === r.model)?.label ?? r.model}
                    {best && (
                      <span className="ml-1.5 rounded bg-success/15 px-1 text-[10px] font-semibold uppercase text-success">
                        best
                      </span>
                    )}
                  </td>
                  <td className="px-1.5 py-1.5 text-muted">T+{r.horizon}</td>
                  <td className="px-1.5 py-1.5 text-right font-semibold text-fg">
                    {fmtFixed(m.wape, 2)}
                  </td>
                  <td className="px-1.5 py-1.5 text-right text-fg">
                    {fmtFixed(m.accuracy_wape, 2)}
                  </td>
                  <td className="px-1.5 py-1.5 text-right">{fmtInt(m.mae)}</td>
                  <td className="px-1.5 py-1.5 text-right">{fmtInt(m.rmse)}</td>
                  <td
                    className={cn(
                      'px-1.5 py-1.5 text-right',
                      isNum(m.bias_mw) && Math.abs(m.bias_mw) > 100 && 'text-warning'
                    )}
                  >
                    {fmtSigned(m.bias_mw)}
                  </td>
                  <td className="px-1.5 py-1.5 text-right">{fmtFixed(m.peak_wape, 2)}</td>
                  <td className="px-1.5 py-1.5 text-right">{fmtInt(m.p95_abs_error_mw)}</td>
                  <td
                    className={cn('px-1.5 py-1.5 text-right', covWarn ? 'text-warning' : 'text-fg')}
                  >
                    {isNum(cov) ? fmtFixed(cov, 1) : <span className="text-faint">n/a</span>}
                  </td>
                  <td className="px-1.5 py-1.5 pr-3 text-right text-muted">{r.n_days}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p
        className="shrink-0 truncate border-t border-border px-3 py-1.5 text-2xs text-faint"
        title="WAPE-derived accuracy = 100 − WAPE. Bias = mean(forecast − actual); positive = over-forecast. Nominal P10–P90 coverage is 80%; deviations beyond ±10 pts are flagged."
      >
        Acc. = 100 − WAPE · Bias + = over-forecast · coverage nominal 80%, ±10 pts flagged
      </p>
    </div>
  );
}

function DailyChart({ data, models }: { data: BacktestResponse; models: ModelInfo[] }) {
  const reduceMotion = useReducedMotion();
  const C = useChartPalette();
  const option = useMemo(() => {
    const dates = [...new Set(data.daily.map((d) => d.date))].sort();
    const groups = new Map<string, Map<string, number | null>>();
    for (const d of data.daily) {
      const k = seriesName(d.model, d.horizon, models);
      if (!groups.has(k)) groups.set(k, new Map());
      groups.get(k)!.set(d.date, d.wape);
    }
    const names = [...groups.keys()];
    const base = themedBase(C);
    return {
      ...base,
      animation: !reduceMotion,
      color: C.series,
      grid: { left: 36, right: 12, top: 32, bottom: 24 },
      legend: {
        top: 0,
        type: 'scroll',
        data: names,
        ...themedLegend(C),
      },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: (v: number | null) => (isNum(v) ? `${v.toFixed(2)}%` : '—'),
      },
      xAxis: {
        ...base.xAxis,
        type: 'category',
        data: dates,
        axisLabel: {
          ...base.xAxis.axisLabel,
          formatter: (v: string) => v.slice(5),
        },
      },
      yAxis: {
        ...base.yAxis,
        type: 'value',
      },
      series: names.map((n) => ({
        name: n,
        type: 'line',
        symbol: 'circle',
        symbolSize: 4,
        showSymbol: dates.length <= 31,
        lineStyle: {
          width: n.endsWith('T+2') ? 1.5 : 2,
          type: n.endsWith('T+2') ? 'dashed' : 'solid',
        },
        data: dates.map((dt) => groups.get(n)!.get(dt) ?? null),
      })),
    };
  }, [data.daily, models, reduceMotion, C]);

  if (!data.daily.length)
    return (
      <div className="data-surface grid h-full place-items-center rounded-xl text-xs text-muted">
        No daily breakdown returned.
      </div>
    );
  return (
    <figure
      className="data-surface flex h-full min-h-0 min-w-0 flex-col rounded-xl px-3 pb-1 pt-2.5"
      aria-label="Daily WAPE by model"
    >
      <figcaption className="shrink-0 truncate text-xs font-semibold uppercase tracking-wider text-muted">
        Daily WAPE % by model · solid T+1, dashed T+2
      </figcaption>
      <div className="relative min-h-0 flex-1">
        <div className="absolute inset-0">
          <ReactECharts option={option} notMerge style={{ height: '100%', width: '100%' }} />
        </div>
      </div>
    </figure>
  );
}

export function BacktestPanel({ region, models }: { region: string; models: ModelInfo[] }) {
  const yesterday = shiftDate(todayIST(), -1);
  const [dateFrom, setDateFrom] = useState(shiftDate(yesterday, -29));
  const [dateTo, setDateTo] = useState(yesterday);
  const [horizons, setHorizons] = useState<string[]>(['1']);
  const [selected, setSelected] = useState<string[] | null>(null);
  const bt = useBacktestV3();
  const scope = useThemeScopeClass();

  const chosen = selected ?? models.map((m) => m.key);
  const nDays = daysBetweenInclusive(dateFrom, dateTo);
  const error =
    !dateFrom || !dateTo
      ? 'Choose both dates.'
      : nDays === 0
        ? '“From” must be on or before “To”.'
        : nDays > MAX_DAYS
          ? `Range is ${nDays} days; the engine caps a run at ${MAX_DAYS} target dates.`
          : !horizons.length
            ? 'Select at least one horizon.'
            : !chosen.length
              ? 'Select at least one model.'
              : null;

  const toggleModel = (k: string, on: boolean) => {
    const base = selected ?? models.map((m) => m.key);
    setSelected(on ? [...new Set([...base, k])] : base.filter((x) => x !== k));
  };

  const run = () => {
    if (error) return;
    bt.mutate(
      {
        region,
        date_from: dateFrom,
        date_to: dateTo,
        horizons: horizons.map(Number).sort() as Horizon[],
        models: chosen,
      },
      {
        onSuccess: (d) =>
          toast.success('Backtest complete', {
            description: `${d.results.length} model × horizon results`,
          }),
        onError: (e) => toast.error('Backtest failed', { description: e.message }),
      }
    );
  };

  const modelLabel =
    chosen.length === models.length
      ? `All models (${models.length})`
      : chosen.length === 0
        ? 'No models'
        : chosen.length === 1
          ? (models.find((m) => m.key === chosen[0])?.label ?? chosen[0])
          : `${chosen.length} models`;

  const field = 'flex items-center gap-1.5';
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <form
        className="glass-chrome flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 rounded-xl px-3 py-2"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
        aria-label="Backtest parameters"
      >
        <div className={field}>
          <Label htmlFor="bt-from">From</Label>
          <Input
            id="bt-from"
            type="date"
            value={dateFrom}
            max={dateTo || undefined}
            onChange={(e) => setDateFrom(e.target.value)}
            className="h-8 w-[9.5rem] font-mono"
          />
        </div>
        <div className={field}>
          <Label htmlFor="bt-to">To</Label>
          <Input
            id="bt-to"
            type="date"
            value={dateTo}
            min={dateFrom || undefined}
            onChange={(e) => setDateTo(e.target.value)}
            className="h-8 w-[9.5rem] font-mono"
          />
        </div>
        <ToggleGroup
          type="multiple"
          value={horizons}
          onValueChange={setHorizons}
          aria-label="Backtest horizons"
          className="h-8 p-0.5"
        >
          <ToggleGroupItem value="1" className="h-6 px-2.5">
            T+1
          </ToggleGroupItem>
          <ToggleGroupItem value="2" className="h-6 px-2.5">
            T+2
          </ToggleGroupItem>
        </ToggleGroup>
        <DropdownMenu>
          <DropdownMenuTrigger
            className="inline-flex h-8 min-w-[10rem] items-center justify-between gap-2 rounded-md border border-border bg-elev2 px-2.5 text-sm text-fg hover:bg-elev3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`Models: ${modelLabel}`}
          >
            {modelLabel}
            <ChevronDown className="size-3.5 text-muted" aria-hidden />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className={cn(scope, 'glass-strong min-w-[14rem]')}>
            <DropdownMenuLabel>Compare models</DropdownMenuLabel>
            {models.map((m) => (
              <DropdownMenuCheckboxItem
                key={m.key}
                checked={chosen.includes(m.key)}
                onCheckedChange={(on) => toggleModel(m.key, Boolean(on))}
                onSelect={(e) => e.preventDefault()}
              >
                {m.label}
              </DropdownMenuCheckboxItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem
              checked={chosen.length === models.length}
              onCheckedChange={(on) => setSelected(on ? models.map((m) => m.key) : [])}
              onSelect={(e) => e.preventDefault()}
            >
              Select all
            </DropdownMenuCheckboxItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <span
          className={cn('font-mono text-2xs', nDays > MAX_DAYS ? 'text-danger' : 'text-muted')}
          title="Target dates in range / engine cap"
        >
          {nDays}/{MAX_DAYS} days
        </span>
        <div className="ml-auto flex min-w-0 items-center gap-2">
          {error && (
            <span className="truncate text-2xs text-warning" role="status" title={error}>
              {error}
            </span>
          )}
          <Button type="submit" size="sm" className="h-8" disabled={Boolean(error) || bt.isPending}>
            {bt.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
            {bt.isPending ? 'Backtesting…' : 'Run backtest'}
          </Button>
        </div>
      </form>

      {bt.data?.weather_note && !bt.isPending && (
        <div
          role="note"
          className="flex shrink-0 items-start gap-2 rounded-lg border border-info/40 bg-info/10 px-3 py-1.5 text-xs"
        >
          <Info className="mt-px size-3.5 shrink-0 text-info" aria-hidden />
          <p className="line-clamp-2 min-w-0" title={bt.data.weather_note}>
            <span className="font-semibold">How to read these numbers: </span>
            {bt.data.weather_note}
          </p>
        </div>
      )}

      <div className="min-h-0 flex-1">
        {bt.isPending ? (
          <div
            className="grid h-full gap-2 grid-rows-[auto_minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] xl:grid-rows-[auto_minmax(0,1fr)]"
            aria-busy
          >
            <p className="text-xs text-muted xl:col-span-2">
              Rolling-origin backtest over {nDays} target dates — each day is re-forecast using only
              data up to its origin. This can take a few minutes.
            </p>
            <Skeleton className="h-full rounded-xl" />
            <Skeleton className="h-full rounded-xl" />
          </div>
        ) : bt.isError ? (
          <ErrorState
            className="h-full"
            title="Backtest failed"
            message={bt.error.message}
            onRetry={run}
          />
        ) : bt.data ? (
          <div className="grid h-full gap-2 grid-rows-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] xl:grid-rows-1">
            <ResultsTable data={bt.data} models={models} />
            <DailyChart data={bt.data} models={models} />
          </div>
        ) : (
          <EmptyState
            className="h-full"
            icon={<FlaskConical />}
            title="Compare models on history"
            description="Pick a date range (max 62 days), horizons and models, then run a rolling-origin backtest. Results include WAPE, MAE, RMSE, bias, peak error and interval coverage."
          />
        )}
      </div>
    </div>
  );
}
