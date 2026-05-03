import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, Gauge, AlertTriangle, Zap } from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import type { ForecastResponse } from '../../types/api';
import { fmtMw, fmtSignedPct, fmtPct, blockToTime } from '../../lib/utils';

interface KpiCardProps {
  label: string;
  value: string;
  unit?: string;
  delta?: { value: number; positiveIsGood?: boolean };
  sub?: string;
  icon: React.ComponentType<{ className?: string }>;
  index: number;
  loading?: boolean;
  accent?: string;
}

function KpiCard({ label, value, unit, delta, sub, icon: Icon, index, loading, accent = 'accent' }: KpiCardProps) {
  const showDelta = delta != null && Number.isFinite(delta.value);
  const positive = (delta?.value ?? 0) > 0;
  const good = positive ? delta?.positiveIsGood !== false : delta?.positiveIsGood === false;
  const DeltaIcon = positive ? TrendingUp : TrendingDown;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, delay: index * 0.06, ease: [0.2, 0.7, 0.2, 1] }}
    >
      <Card className="overflow-hidden transition-all hover:border-accent/40">
        <CardContent className="p-5">
          <div className="flex items-start justify-between">
            <div className="text-2xs font-semibold uppercase tracking-wider text-muted">{label}</div>
            <div className={`grid h-7 w-7 place-items-center rounded-md bg-${accent}/15 text-${accent}`}>
              <Icon className="h-3.5 w-3.5" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline gap-1.5">
            {loading ? (
              <Skeleton className="h-8 w-28" />
            ) : (
              <>
                <span className="font-display text-3xl font-bold tabular-nums tracking-tight text-fg">{value}</span>
                {unit && <span className="text-sm text-muted">{unit}</span>}
              </>
            )}
          </div>
          <div className="mt-1.5 flex items-center justify-between text-xs">
            <span className="text-faint">{sub || ' '}</span>
            {showDelta && !loading && (
              <span
                className={`inline-flex items-center gap-1 font-mono font-semibold ${
                  good ? 'text-success' : 'text-danger'
                }`}
              >
                <DeltaIcon className="h-3 w-3" />
                {fmtSignedPct(delta!.value)}
              </span>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

interface Props {
  forecast: ForecastResponse | undefined;
  prior: ForecastResponse | undefined;
  loading?: boolean;
}

export function KpiStrip({ forecast, prior, loading }: Props) {
  const series = forecast?.series?.forecast || [];
  const peak = series.length ? Math.max(...series) : NaN;
  const peakBlock = series.length ? series.indexOf(peak) + 1 : 0;
  const trough = series.length ? Math.min(...series) : NaN;
  const avg = series.length ? series.reduce((a, b) => a + b, 0) / series.length : NaN;

  const priorSeries = prior?.series?.forecast || [];
  const priorAvg = priorSeries.length ? priorSeries.reduce((a, b) => a + b, 0) / priorSeries.length : NaN;
  const trendPct = Number.isFinite(avg) && Number.isFinite(priorAvg) && priorAvg > 0
    ? ((avg - priorAvg) / priorAvg) * 100
    : NaN;

  // Average forecast confidence (height of p10..p90 band relative to forecast)
  const uncertainty = forecast?.forecast_uncertainty;
  let avgConfidence = NaN;
  if (uncertainty?.length === 96 && Number.isFinite(avg) && avg > 0) {
    const widths = uncertainty
      .map((u) => (u.p90 != null && u.p10 != null ? u.p90 - u.p10 : NaN))
      .filter((v) => Number.isFinite(v)) as number[];
    if (widths.length) {
      const mean = widths.reduce((a, b) => a + b, 0) / widths.length;
      avgConfidence = Math.max(0, Math.min(100, 100 - (mean / avg) * 100));
    }
  }

  // Risk: count of blocks tagged high in decision_signals
  const decisionSignals = forecast?.decision_signals || [];
  const highRisk = decisionSignals.filter((s: any) => (s.severity || '').toLowerCase() === 'high').length;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <KpiCard
        index={0}
        loading={loading}
        label="Peak Load"
        value={fmtMw(peak, 0)}
        unit="MW"
        sub={Number.isFinite(peak) ? `at block ${peakBlock} · ${blockToTime(peakBlock)}` : '—'}
        icon={Zap}
        delta={Number.isFinite(trendPct) ? { value: trendPct, positiveIsGood: false } : undefined}
        accent="accent"
      />
      <KpiCard
        index={1}
        loading={loading}
        label="Average Load"
        value={fmtMw(avg, 0)}
        unit="MW"
        sub={Number.isFinite(trough) ? `Low ${fmtMw(trough, 0)} MW` : '—'}
        icon={Activity}
        accent="accent2"
      />
      <KpiCard
        index={2}
        loading={loading}
        label="Confidence"
        value={Number.isFinite(avgConfidence) ? avgConfidence.toFixed(0) : '—'}
        unit="%"
        sub={Number.isFinite(avgConfidence) ? 'p10–p90 band' : 'no uncertainty data'}
        icon={Gauge}
        accent="success"
      />
      <KpiCard
        index={3}
        loading={loading}
        label="Risk Signals"
        value={String(highRisk)}
        sub={highRisk === 0 ? 'No high-risk blocks' : `${highRisk} high-risk block${highRisk === 1 ? '' : 's'}`}
        icon={AlertTriangle}
        accent={highRisk > 0 ? 'danger' : 'muted'}
      />
    </div>
  );
}
