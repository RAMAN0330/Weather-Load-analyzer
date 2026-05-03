import type { BlockDriver, DriverType } from './types';

const CAPS = {
  weather_pct: 15,
  daytype_pct: 20,
  holiday_pct: 30,
  manual_pct: 40,
  net_pct_min: -40,
  net_pct_max: 50,
} as const;

export const blockTimeLabel = (block: number): string => {
  const minutes = (block - 1) * 15;
  const h = String(Math.floor(minutes / 60)).padStart(2, '0');
  const m = String(minutes % 60).padStart(2, '0');
  return `${h}:${m}`;
};

export const clampDriver = (driver: DriverType, value: number): number => {
  const cap = CAPS[driver];
  return Math.max(-cap, Math.min(cap, Number(value) || 0));
};

export const clampNet = (value: number): number =>
  Math.max(CAPS.net_pct_min, Math.min(CAPS.net_pct_max, value));

export const recalcBlock = (block: BlockDriver): BlockDriver => {
  const impactRaw =
    Number(block.drivers.weather_pct || 0) / 100 +
    Number(block.drivers.daytype_pct || 0) / 100 +
    Number(block.drivers.holiday_pct || 0) / 100 +
    Number(block.drivers.manual_pct || 0) / 100;
  const impactClamped = Math.max(-0.4, Math.min(0.5, impactRaw));
  const mask = Math.max(0, Math.min(1, Number(block.selection_mask ?? 1)));
  const net_pct = clampNet(impactClamped * mask * 100);
  const final_mw = block.baseline_mw * (1 + net_pct / 100);

  return {
    ...block,
    net_pct,
    final_mw,
  };
};
