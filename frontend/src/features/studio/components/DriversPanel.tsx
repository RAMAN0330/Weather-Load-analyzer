import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useReducedMotion } from 'framer-motion';
import { CloudRain, Droplets, Flame, Moon, Sun, Thermometer, Wind } from 'lucide-react';
import type { ReactNode } from 'react';
import { themedBase, useChartPalette } from '../chartPalette';
import type { ForecastResponse } from '../api';
import { isNum, titleCase } from '../utils';

interface DriverMeta {
  label: string;
  unit: string;
  hint: string;
  icon: ReactNode;
  /** Convert raw value to display value. */
  map?: (v: number) => number;
  digits?: number;
}

const DRIVER_META: Record<string, DriverMeta> = {
  temp_weighted_max: {
    label: 'Load-weighted max temp',
    unit: '°C',
    hint: 'District max temperature weighted by load share',
    icon: <Thermometer />,
  },
  temp_p90_max: {
    label: 'P90 district max temp',
    unit: '°C',
    hint: '90th percentile of district max temperatures',
    icon: <Sun />,
  },
  wet_bulb_weighted_max: {
    label: 'Weighted max wet-bulb',
    unit: '°C',
    hint: 'Humid-heat stress driving cooling load',
    icon: <Droplets />,
  },
  wet_bulb_p90_max: {
    label: 'P90 wet-bulb',
    unit: '°C',
    hint: '90th percentile district wet-bulb',
    icon: <Wind />,
  },
  cdh_24h_end: {
    label: 'Cooling degree-hours',
    unit: '°C·h',
    hint: 'Accumulated cooling degree-hours over trailing 24 h',
    icon: <Flame />,
    digits: 0,
  },
  night_min_temp: {
    label: 'Night minimum temp',
    unit: '°C',
    hint: 'Warm nights sustain overnight AC load',
    icon: <Moon />,
  },
  hot_share_peak: {
    label: 'Hot share at peak',
    unit: '%',
    hint: 'Share of load in districts above heat threshold at peak',
    icon: <Flame />,
    map: (v) => (v <= 1 ? v * 100 : v),
    digits: 0,
  },
  precip_total_mm: {
    label: 'Total precipitation',
    unit: 'mm',
    hint: 'Load-weighted daily rainfall',
    icon: <CloudRain />,
  },
};

function DriverCard({ k, value }: { k: string; value: number | null | undefined }) {
  const meta: DriverMeta = DRIVER_META[k] ?? {
    label: titleCase(k),
    unit: '',
    hint: k,
    icon: <Thermometer />,
  };
  const v = isNum(value) ? (meta.map ? meta.map(value) : value) : null;
  return (
    <div
      className="min-w-0 rounded-lg border border-border bg-elev2/50 px-2.5 py-2"
      title={meta.hint}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-2xs font-semibold uppercase tracking-wider text-muted">
          {meta.label}
        </span>
        <span className="shrink-0 text-accent2 [&_svg]:size-3" aria-hidden>
          {meta.icon}
        </span>
      </div>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="font-mono text-base font-semibold tabular-nums text-fg">
          {v == null ? '—' : v.toFixed(meta.digits ?? 1)}
        </span>
        <span className="font-mono text-2xs text-muted">{meta.unit}</span>
      </div>
    </div>
  );
}

export function DriversPanel({ forecast }: { forecast: ForecastResponse }) {
  const reduceMotion = useReducedMotion();
  const C = useChartPalette();
  const drivers = forecast.drivers ?? {};
  const known = Object.keys(DRIVER_META).filter((k) => k in drivers);
  const extra = Object.keys(drivers).filter((k) => !(k in DRIVER_META));
  const keys = [...known, ...extra];

  const fi = useMemo(
    () =>
      [...(forecast.feature_importance ?? [])]
        .filter((f) => isNum(f.gain_pct))
        .sort((a, b) => b.gain_pct - a.gain_pct)
        .slice(0, 15),
    [forecast.feature_importance]
  );

  const option = useMemo(() => {
    const rows = [...fi].reverse();
    const base = themedBase(C);
    return {
      ...base,
      animation: !reduceMotion,
      grid: { left: 4, right: 44, top: 4, bottom: 4, containLabel: true },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        axisPointer: { type: 'shadow', shadowStyle: { color: C.zoom.dataArea } },
        valueFormatter: (v: number) => `${v.toFixed(1)}% of gain`,
      },
      xAxis: {
        ...base.yAxis,
        type: 'value',
        axisLabel: { ...base.yAxis.axisLabel, formatter: '{value}%' },
      },
      yAxis: {
        ...base.xAxis,
        type: 'category',
        data: rows.map((r) => r.feature),
        axisLabel: { ...base.xAxis.axisLabel, color: C.text, fontSize: 10 },
      },
      series: [
        {
          type: 'bar',
          name: 'Gain',
          data: rows.map((r) => r.gain_pct),
          barMaxWidth: 12,
          itemStyle: { color: C.bar, borderRadius: [0, 3, 3, 0] },
          label: {
            show: true,
            position: 'right',
            color: C.text,
            fontSize: 10,
            fontFamily: "'IBM Plex Mono', monospace",
            formatter: ({ value }: { value: number }) => `${value.toFixed(1)}%`,
          },
        },
      ],
    };
  }, [fi, reduceMotion, C]);

  return (
    <div
      className="h-full min-h-0 space-y-3 overflow-auto p-3"
      tabIndex={0}
      aria-label="Drivers, scrollable"
    >
      <section aria-label="Weather drivers" className="space-y-2">
        <h3 className="text-2xs font-semibold uppercase tracking-wider text-muted">
          Weather drivers · target day
        </h3>
        {keys.length ? (
          <div className="grid grid-cols-2 gap-2">
            {keys.map((k) => (
              <DriverCard key={k} k={k} value={drivers[k]} />
            ))}
          </div>
        ) : (
          <p className="py-4 text-center text-xs text-muted">
            The engine returned no weather drivers for this run.
          </p>
        )}
      </section>

      <section aria-label="Feature importance" className="space-y-1">
        <h3 className="text-2xs font-semibold uppercase tracking-wider text-muted">
          Feature importance · share of gain
        </h3>
        {fi.length ? (
          <div style={{ height: fi.length * 20 + 28 }}>
            <ReactECharts option={option} notMerge style={{ height: '100%', width: '100%' }} />
          </div>
        ) : (
          <p className="py-4 text-center text-xs text-muted">
            Feature importance is only available for gradient-boosted models.
          </p>
        )}
      </section>
    </div>
  );
}
