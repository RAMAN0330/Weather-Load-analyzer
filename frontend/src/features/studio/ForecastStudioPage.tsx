import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MotionConfig } from 'framer-motion';
import { BarChart3, Database, FlaskConical, Sparkles, Table2, Thermometer } from 'lucide-react';
import { Toaster, toast } from 'sonner';
import '../../styles/ds-scope.css';
import { Button } from '../../components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { TooltipProvider } from '../../components/ui/tooltip';
import {
  FALLBACK_MODELS,
  useForecastV3,
  useModels,
  type ForecastRequest,
  type Horizon,
} from './api';
import { BacktestPanel } from './components/BacktestPanel';
import { BlocksTable } from './components/BlocksTable';
import { CommandBar } from './components/CommandBar';
import { DataQualityPanel } from './components/DataQualityPanel';
import { DriversPanel } from './components/DriversPanel';
import { ForecastChart } from './components/ForecastChart';
import { KpiRow } from './components/KpiRow';
import { QualityBanner } from './components/QualityBanner';
import { ChartSkeleton, EmptyState, ErrorState, KpiSkeletonRow } from './components/States';
import { cn } from '../../lib/utils';
import { useResolvedTheme, useThemeScopeClass } from './theme';
import { buildBlocksCsv, downloadText, fmtDateLong, shiftDate, titleCase, todayIST } from './utils';

/**
 * Module-level client: the legacy forecast shell (App.jsx) has no
 * QueryClientProvider, so the studio brings its own instead of touching
 * global wiring.
 */
const studioQueryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 60_000 },
  },
});

const REGIONS = ['haryana', 'odisha', 'rajasthan', 'chhattisgarh'];

export interface ForecastStudioPageProps {
  region?: string | null;
  /** Alias accepted for parity with other App.jsx pages. */
  selectedRegion?: string | null;
}

const sameReq = (a: ForecastRequest | null, b: ForecastRequest | null) =>
  !!a &&
  !!b &&
  a.region === b.region &&
  a.target_date === b.target_date &&
  a.horizon === b.horizon &&
  a.model === b.model;

function Studio({ initialRegion }: { initialRegion: string }) {
  const [region, setRegion] = useState(initialRegion);
  const [targetDate, setTargetDate] = useState(() => shiftDate(todayIST(), 1));
  const [horizon, setHorizon] = useState<Horizon>(1);
  const [model, setModel] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState<ForecastRequest | null>(null);
  const [view, setView] = useState('forecast');
  const [sideTab, setSideTab] = useState('blocks');

  useEffect(() => setRegion(initialRegion), [initialRegion]);

  const modelsQ = useModels();
  const models = modelsQ.data?.models?.length ? modelsQ.data.models : FALLBACK_MODELS;
  const modelsFallback = !modelsQ.data?.models?.length && !modelsQ.isLoading;
  const effectiveModel =
    model ?? models.find((m) => m.default)?.key ?? models[0]?.key ?? 'lgbm_residual';

  const regions = useMemo(
    () => (REGIONS.includes(region) ? REGIONS : [region, ...REGIONS]),
    [region]
  );

  const fq = useForecastV3(submitted);
  const forecast = fq.data;

  const current: ForecastRequest = {
    region,
    target_date: targetDate,
    horizon,
    model: effectiveModel,
  };
  const dirty = submitted != null && !sameReq(current, submitted);

  // Toasts on run outcome (keyed on update timestamps so re-runs toast again).
  const lastOk = useRef(0);
  const lastErr = useRef(0);
  useEffect(() => {
    if (fq.data && fq.dataUpdatedAt && fq.dataUpdatedAt !== lastOk.current) {
      lastOk.current = fq.dataUpdatedAt;
      const d = fq.data;
      const desc = `${titleCase(d.region)} · ${fmtDateLong(d.target_date)} · T+${d.horizon} · ${d.model}`;
      if (d.status === 'ok') toast.success('Forecast ready', { description: desc });
      else if (d.status === 'degraded')
        toast.warning('Forecast ready — degraded inputs', { description: desc });
      else toast.error('Forecast failed quality checks', { description: desc });
    }
  }, [fq.data, fq.dataUpdatedAt]);
  useEffect(() => {
    if (fq.error && fq.errorUpdatedAt && fq.errorUpdatedAt !== lastErr.current) {
      lastErr.current = fq.errorUpdatedAt;
      toast.error(fq.error.status === 422 ? 'Insufficient history' : 'Forecast request failed', {
        description: fq.error.message,
      });
    }
  }, [fq.error, fq.errorUpdatedAt]);

  const run = useCallback(() => {
    if (!targetDate) return;
    if (sameReq(current, submitted)) fq.refetch();
    else setSubmitted(current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region, targetDate, horizon, effectiveModel, submitted, fq.refetch]);

  const onExport = useCallback(() => {
    if (!forecast?.blocks?.length) return;
    downloadText(`${forecast.forecast_id || 'forecast'}.csv`, buildBlocksCsv(forecast));
    toast.success('CSV exported', { description: `${forecast.blocks.length} blocks` });
  }, [forecast]);

  const running = fq.isFetching;
  const showSkeleton = submitted != null && !forecast && fq.isFetching;
  const showError = !forecast && fq.isError;

  const forecastBody = forecast ? (
    <div
      className={cn(
        'grid h-full min-h-0 gap-2 transition-opacity',
        'grid-rows-[minmax(0,1.2fr)_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_400px] lg:grid-rows-1 2xl:grid-cols-[minmax(0,1fr)_440px]',
        running && 'opacity-60'
      )}
      aria-busy={running}
    >
      <ForecastChart forecast={forecast} />
      <Tabs
        value={sideTab}
        onValueChange={setSideTab}
        className="data-surface flex min-h-0 min-w-0 flex-col overflow-hidden rounded-xl"
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-2 py-1.5">
          <TabsList aria-label="Forecast detail" className="h-8 p-0.5">
            <TabsTrigger value="blocks" className="gap-1.5 px-2.5 py-1">
              <Table2 className="size-3.5" aria-hidden /> Blocks
            </TabsTrigger>
            <TabsTrigger value="drivers" className="gap-1.5 px-2.5 py-1">
              <Thermometer className="size-3.5" aria-hidden /> Drivers
            </TabsTrigger>
          </TabsList>
          <span className="truncate font-mono text-2xs text-faint">MW · 15-min IST</span>
        </div>
        <TabsContent value="blocks" className="mt-0 min-h-0 flex-1 animate-none">
          <BlocksTable forecast={forecast} />
        </TabsContent>
        <TabsContent value="drivers" className="mt-0 min-h-0 flex-1 animate-none">
          <DriversPanel forecast={forecast} />
        </TabsContent>
      </Tabs>
    </div>
  ) : showSkeleton ? (
    <ChartSkeleton />
  ) : showError ? (
    <ErrorState
      className="h-full"
      title={fq.error?.status === 422 ? 'Insufficient history for this origin' : 'Forecast failed'}
      message={fq.error?.message ?? 'Unknown error'}
      onRetry={() => fq.refetch()}
      retrying={fq.isFetching}
    />
  ) : (
    <EmptyState
      className="h-full"
      icon={<Sparkles />}
      title="No forecast yet"
      description={
        <>
          Choose a target date, horizon and model, then run. T+{horizon} for{' '}
          <span className="font-mono text-fg/90">{targetDate || '—'}</span> uses data up to{' '}
          <span className="font-mono text-fg/90">
            {targetDate ? shiftDate(targetDate, -horizon) : '—'}
          </span>
          .
        </>
      }
      action={
        <Button onClick={run} disabled={!targetDate}>
          <Sparkles aria-hidden />
          Run forecast
        </Button>
      }
    />
  );

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {/* Header zone — fixed height */}
      <CommandBar
        region={region}
        regions={regions}
        onRegion={setRegion}
        targetDate={targetDate}
        onTargetDate={setTargetDate}
        horizon={horizon}
        onHorizon={setHorizon}
        model={effectiveModel}
        models={models}
        modelsFallback={modelsFallback}
        onModel={setModel}
        onRun={run}
        running={running}
        dirty={dirty}
        forecast={forecast}
        onExport={onExport}
      />
      <p className="sr-only" aria-live="polite">
        {running ? 'Forecast running' : forecast ? `Forecast ${forecast.status}` : ''}
      </p>
      {forecast && <QualityBanner forecast={forecast} />}
      {forecast ? <KpiRow forecast={forecast} /> : showSkeleton ? <KpiSkeletonRow /> : null}

      {/* Work area — fills the remaining height */}
      <Tabs value={view} onValueChange={setView} className="flex min-h-0 flex-1 flex-col">
        <TabsList aria-label="Studio views" className="h-8 shrink-0 self-start p-0.5">
          <TabsTrigger value="forecast" className="gap-1.5 px-2.5 py-1">
            <BarChart3 className="size-3.5" aria-hidden /> Forecast
          </TabsTrigger>
          <TabsTrigger value="backtest" className="gap-1.5 px-2.5 py-1">
            <FlaskConical className="size-3.5" aria-hidden /> Backtest
          </TabsTrigger>
          <TabsTrigger value="quality" className="gap-1.5 px-2.5 py-1">
            <Database className="size-3.5" aria-hidden /> Data Quality
          </TabsTrigger>
        </TabsList>
        <TabsContent value="forecast" className="mt-2 min-h-0 flex-1">
          {forecastBody}
        </TabsContent>
        <TabsContent value="backtest" className="mt-2 min-h-0 flex-1">
          <BacktestPanel region={region} models={models} />
        </TabsContent>
        <TabsContent value="quality" className="mt-2 min-h-0 flex-1">
          <DataQualityPanel region={region} forecast={forecast} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function ForecastStudioPage({ region, selectedRegion }: ForecastStudioPageProps) {
  const initialRegion = (region || selectedRegion || 'haryana').toLowerCase();
  // Resolved synchronously (persisted mode + matchMedia) on first render → no flash.
  const resolved = useResolvedTheme();
  const scope = useThemeScopeClass();
  return (
    <QueryClientProvider client={studioQueryClient}>
      <TooltipProvider delayDuration={150}>
        <MotionConfig reducedMotion="user">
          {/*
            Viewport-fit app layout: the root never scrolls; dense regions scroll
            inside their own panels. Below 680px viewport height the inner layout
            gets a 620px min-height and only this root scrolls (graceful degrade).
          */}
          <div
            className={cn(
              scope,
              'flex h-full min-h-0 w-full flex-1 flex-col overflow-hidden bg-base transition-colors',
              '[@media(max-height:679px)]:overflow-y-auto'
            )}
            data-testid="forecast-studio"
            data-theme={resolved}
          >
            <div className="flex min-h-0 w-full flex-1 flex-col p-2 [@media(max-height:679px)]:min-h-[620px]">
              <Studio initialRegion={initialRegion} />
            </div>
            <Toaster
              theme={resolved}
              position="bottom-right"
              richColors
              closeButton
              toastOptions={{ style: { fontFamily: "'Manrope', system-ui, sans-serif" } }}
            />
          </div>
        </MotionConfig>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
