import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Activity, ArrowDownToLine, Gauge, MoveVertical, Scale, Target, Zap } from 'lucide-react';
import { cn } from '../../../lib/utils';
import type { ForecastResponse } from '../api';
import {
  blockTime,
  computeAccuracy,
  fmtFixed,
  fmtInt,
  fmtSigned,
  isNum,
  maxRamp,
  meanBandWidth,
} from '../utils';

interface KpiProps {
  label: string;
  value: string;
  unit?: string;
  sub?: ReactNode;
  icon: ReactNode;
  tone?: 'default' | 'accent' | 'success' | 'warning' | 'danger';
  index: number;
}

const toneCls: Record<NonNullable<KpiProps['tone']>, string> = {
  default: 'text-muted',
  accent: 'text-accent',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
};

function Kpi({ label, value, unit, sub, icon, tone = 'default', index }: KpiProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, delay: index * 0.03, ease: 'easeOut' }}
      className="glass-chrome flex h-[68px] min-w-0 flex-col justify-between rounded-lg px-3 py-2"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-2xs font-semibold uppercase tracking-wider text-muted">
          {label}
        </span>
        <span className={cn('[&_svg]:size-3.5', toneCls[tone])} aria-hidden>
          {icon}
        </span>
      </div>
      <div className="flex items-baseline gap-1 leading-none">
        <span className="font-mono text-lg font-semibold leading-none tabular-nums text-fg">
          {value}
        </span>
        {unit && <span className="font-mono text-2xs text-muted">{unit}</span>}
      </div>
      {sub && (
        <div
          className="truncate font-mono text-2xs leading-tight text-muted"
          title={typeof sub === 'string' ? sub : undefined}
        >
          {sub}
        </div>
      )}
    </motion.div>
  );
}

export function KpiRow({ forecast }: { forecast: ForecastResponse }) {
  const s = forecast.summary ?? ({} as ForecastResponse['summary']);
  const blocks = forecast.blocks ?? [];
  const band = isNum(s.mean_band_width_mw) ? s.mean_band_width_mw : meanBandWidth(blocks);
  const rampFallback = maxRamp(blocks);
  const ramp = isNum(s.max_ramp_mw) ? s.max_ramp_mw : rampFallback.mw;
  const rampBlock = isNum(s.max_ramp_block) ? s.max_ramp_block : rampFallback.block;
  const acc = computeAccuracy(blocks);
  const bandPct = isNum(band) && isNum(s.peak_mw) && s.peak_mw ? (band / s.peak_mw) * 100 : null;

  const accTone =
    acc.accuracy == null
      ? 'default'
      : acc.accuracy >= 97
        ? 'success'
        : acc.accuracy >= 95
          ? 'warning'
          : 'danger';

  return (
    <section
      aria-label="Forecast key figures"
      className="grid shrink-0 grid-cols-[repeat(auto-fit,minmax(128px,1fr))] gap-2"
    >
      <Kpi
        index={0}
        label="Peak"
        value={fmtInt(s.peak_mw)}
        unit="MW"
        tone="accent"
        icon={<Zap />}
        sub={isNum(s.peak_block) ? `block ${s.peak_block} · ${blockTime(s.peak_block)}` : '—'}
      />
      <Kpi
        index={1}
        label="Energy"
        value={fmtInt(s.energy_mwh)}
        unit="MWh"
        icon={<Activity />}
        sub="P50 day total"
      />
      <Kpi
        index={2}
        label="Minimum"
        value={fmtInt(s.min_mw)}
        unit="MW"
        icon={<ArrowDownToLine />}
        sub={isNum(s.min_block) ? `block ${s.min_block} · ${blockTime(s.min_block)}` : '—'}
      />
      <Kpi
        index={3}
        label="Uncertainty"
        value={isNum(band) ? `±${fmtInt(band / 2)}` : '—'}
        unit="MW"
        icon={<MoveVertical />}
        sub={
          isNum(band)
            ? `band ${fmtInt(band)}${bandPct != null ? ` · ${bandPct.toFixed(1)}%` : ''}`
            : 'point model'
        }
      />
      <Kpi
        index={4}
        label="Max ramp"
        value={fmtSigned(ramp)}
        unit="MW/15m"
        icon={<Gauge />}
        sub={isNum(rampBlock) ? `block ${rampBlock} · ${blockTime(rampBlock)}` : '—'}
      />
      {acc.n > 0 && (
        <>
          <Kpi
            index={5}
            label="Accuracy"
            value={fmtFixed(acc.accuracy, 2)}
            unit="%"
            tone={accTone}
            icon={<Target />}
            sub={`WAPE ${fmtFixed(acc.wape, 2)}% (${acc.n}/${blocks.length})`}
          />
          <Kpi
            index={6}
            label="Bias"
            value={fmtSigned(acc.bias_mw)}
            unit="MW"
            tone={isNum(acc.bias_mw) && Math.abs(acc.bias_mw) > 100 ? 'warning' : 'default'}
            icon={<Scale />}
            sub={`MAE ${fmtInt(acc.mae)}${
              acc.coverage_pct != null ? ` · cov ${acc.coverage_pct.toFixed(0)}%` : ''
            }`}
          />
        </>
      )}
    </section>
  );
}
