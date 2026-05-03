import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** 1-based block (1..96) → "HH:MM" string */
export function blockToTime(block: number): string {
  const b = Math.max(1, Math.min(96, Math.round(block)));
  const minutes = (b - 1) * 15;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

/** Format a number as MW with optional decimals; "—" for null/NaN. */
export function fmtMw(v: number | null | undefined, d = 0): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return Number(v).toLocaleString('en-IN', {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

export function fmtNum(v: number | null | undefined, d = 1): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return Number(v).toFixed(d);
}

export function fmtPct(v: number | null | undefined, d = 1): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${Number(v).toFixed(d)}%`;
}

export function fmtSignedPct(v: number | null | undefined, d = 1): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${Number(v).toFixed(d)}%`;
}

const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** "Sun" / "Mon" / ... from a YYYY-MM-DD date string. */
export function dayType(dateStr?: string | null): string {
  if (!dateStr) return '';
  const d = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(d.valueOf())) return '';
  return DOW[d.getDay()];
}

/** Rough season label for India (used as fallback when API omits one) */
export function detectSeason(dateStr?: string | null): string {
  if (!dateStr) return '';
  const m = Number(dateStr.slice(5, 7));
  if ([12, 1, 2].includes(m)) return 'Winter';
  if ([3, 4].includes(m)) return 'Spring';
  if ([5, 6].includes(m)) return 'Summer';
  if ([7, 8, 9].includes(m)) return 'Monsoon';
  return 'Autumn';
}

/** Today's local date as YYYY-MM-DD */
export function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Add days to a YYYY-MM-DD; returns new YYYY-MM-DD. */
export function addDays(dateStr: string, days: number): string {
  const d = new Date(`${dateStr}T00:00:00`);
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}
