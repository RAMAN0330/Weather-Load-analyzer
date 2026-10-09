/** Pure helpers for Forecast Studio — no React, no I/O (except downloadText/copyText). */
import type { ForecastBlock, ForecastResponse } from './api';

export const BLOCKS_PER_DAY = 96;

/** Evening peak window (IST) highlighted on charts and tables. */
export const PEAK_WINDOW = { from: '17:00', to: '23:00' } as const;

// ── Time / blocks ────────────────────────────────────────────────────────────

/** 1-based block (1..96) → start time "HH:MM". Block 1 = 00:00. */
export function blockTime(block: number): string {
  const b = Math.max(1, Math.min(BLOCKS_PER_DAY, Math.round(block)));
  const mins = (b - 1) * 15;
  return `${String(Math.floor(mins / 60)).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}`;
}

/** Block → "HH:MM–HH:MM" (end exclusive; block 96 ends at 24:00). */
export function blockRange(block: number): string {
  const b = Math.max(1, Math.min(BLOCKS_PER_DAY, Math.round(block)));
  const end = b * 15;
  const endLabel = `${String(Math.floor(end / 60)).padStart(2, '0')}:${String(end % 60).padStart(2, '0')}`;
  return `${blockTime(b)}–${endLabel}`;
}

/** 96 x-axis labels: 00:00 … 23:45. */
export const TIME_LABELS: string[] = Array.from({ length: BLOCKS_PER_DAY }, (_, i) =>
  blockTime(i + 1)
);

export function isPeakWindow(time: string): boolean {
  return time >= PEAK_WINDOW.from && time < PEAK_WINDOW.to;
}

// ── Dates (UTC arithmetic on YYYY-MM-DD so local TZ never shifts the day) ───

/** Today's date in IST as YYYY-MM-DD. */
export function todayIST(now: Date = new Date()): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Kolkata',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(now);
}

export function shiftDate(date: string, days: number): string {
  const [y, m, d] = date.split('-').map(Number);
  const t = Date.UTC(y, m - 1, d) + days * 86_400_000;
  return new Date(t).toISOString().slice(0, 10);
}

/** Inclusive day count between two YYYY-MM-DD dates (0 if invalid/reversed). */
export function daysBetweenInclusive(from: string, to: string): number {
  if (!from || !to) return 0;
  const a = Date.parse(`${from}T00:00:00Z`);
  const b = Date.parse(`${to}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b) || b < a) return 0;
  return Math.round((b - a) / 86_400_000) + 1;
}

export function fmtDateLong(date?: string | null): string {
  if (!date) return '—';
  const t = Date.parse(`${date}T00:00:00Z`);
  if (!Number.isFinite(t)) return date;
  return new Date(t).toLocaleDateString('en-IN', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

export function fmtDateTimeIST(iso?: string | null): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  return (
    new Date(t).toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'Asia/Kolkata',
    }) + ' IST'
  );
}

// ── Number formatting ────────────────────────────────────────────────────────

export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export function fmtInt(v: number | null | undefined): string {
  return isNum(v) ? Math.round(v).toLocaleString('en-IN') : '—';
}

export function fmtFixed(v: number | null | undefined, d = 1): string {
  return isNum(v)
    ? v.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })
    : '—';
}

export function fmtSigned(v: number | null | undefined, d = 0): string {
  if (!isNum(v)) return '—';
  const s = v > 0 ? '+' : v < 0 ? '−' : '';
  return `${s}${Math.abs(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })}`;
}

export function fmtPctVal(v: number | null | undefined, d = 1): string {
  return isNum(v) ? `${v.toFixed(d)}%` : '—';
}

// ── Accuracy (client-side, over blocks that have actuals) ───────────────────

export interface LiveAccuracy {
  n: number;
  wape: number | null;
  /** 100 − WAPE ("WAPE-derived accuracy"). */
  accuracy: number | null;
  /** Mean(P50 − actual) in MW; positive = over-forecast. */
  bias_mw: number | null;
  mae: number | null;
  /** Share of actuals inside [P10, P90], in %. */
  coverage_pct: number | null;
}

export function computeAccuracy(blocks: ForecastBlock[]): LiveAccuracy {
  let n = 0;
  let absErr = 0;
  let absAct = 0;
  let signed = 0;
  let inBand = 0;
  let bandN = 0;
  for (const b of blocks) {
    if (!isNum(b.actual) || !isNum(b.p50)) continue;
    const e = b.p50 - b.actual;
    n += 1;
    absErr += Math.abs(e);
    absAct += Math.abs(b.actual);
    signed += e;
    if (isNum(b.p10) && isNum(b.p90)) {
      bandN += 1;
      if (b.actual >= b.p10 && b.actual <= b.p90) inBand += 1;
    }
  }
  if (n === 0) {
    return { n: 0, wape: null, accuracy: null, bias_mw: null, mae: null, coverage_pct: null };
  }
  const wape = absAct > 0 ? (absErr / absAct) * 100 : null;
  return {
    n,
    wape,
    accuracy: wape == null ? null : 100 - wape,
    bias_mw: signed / n,
    mae: absErr / n,
    coverage_pct: bandN ? (inBand / bandN) * 100 : null,
  };
}

/** Per-block error (P50 − actual) in MW and as % of actual. */
export function blockError(b: ForecastBlock): { mw: number | null; pct: number | null } {
  if (!isNum(b.actual) || !isNum(b.p50)) return { mw: null, pct: null };
  const mw = b.p50 - b.actual;
  return { mw, pct: b.actual !== 0 ? (mw / Math.abs(b.actual)) * 100 : null };
}

/**
 * Heat tint for an error cell. Diverging: under-forecast (negative, the
 * operationally riskier side) → red; over-forecast → blue. Alpha scales with
 * |error %| up to `maxPct`; below 0.5 % no tint.
 */
export function errorTint(pct: number | null | undefined, maxPct = 6): string | undefined {
  if (!isNum(pct)) return undefined;
  const mag = Math.abs(pct);
  if (mag < 0.5) return undefined;
  const alpha = 0.08 + Math.min(mag / maxPct, 1) * 0.32;
  return pct < 0
    ? `hsl(var(--danger) / ${alpha.toFixed(3)})`
    : `hsl(var(--info) / ${alpha.toFixed(3)})`;
}

// ── Summary fallbacks (if the backend omits a field) ────────────────────────

export function meanBandWidth(blocks: ForecastBlock[]): number | null {
  let s = 0;
  let n = 0;
  for (const b of blocks) {
    if (isNum(b.p10) && isNum(b.p90)) {
      s += b.p90 - b.p10;
      n += 1;
    }
  }
  return n ? s / n : null;
}

export function maxRamp(blocks: ForecastBlock[]): { mw: number | null; block: number | null } {
  let best: number | null = null;
  let at: number | null = null;
  for (let i = 1; i < blocks.length; i++) {
    const a = blocks[i - 1].p50;
    const b = blocks[i].p50;
    if (!isNum(a) || !isNum(b)) continue;
    const r = b - a;
    if (best == null || Math.abs(r) > Math.abs(best)) {
      best = r;
      at = blocks[i].block;
    }
  }
  return { mw: best, block: at };
}

// ── CSV export ──────────────────────────────────────────────────────────────

function csvCell(v: unknown): string {
  if (v == null) return '';
  const s =
    typeof v === 'number'
      ? Number.isFinite(v)
        ? String(Math.round(v * 100) / 100)
        : ''
      : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** 96-row CSV; missing values are empty cells (never 0). */
export function buildBlocksCsv(f: ForecastResponse): string {
  const header = [
    'forecast_id',
    'region',
    'target_date',
    'origin_date',
    'horizon',
    'model',
    'model_version',
    'feature_version',
    'block',
    'time',
    'p10_mw',
    'p50_mw',
    'p90_mw',
    'baseline_mw',
    'actual_mw',
    'error_mw',
    'error_pct',
  ];
  const rows = f.blocks.map((b) => {
    const e = blockError(b);
    return [
      f.forecast_id,
      f.region,
      f.target_date,
      f.origin_date,
      f.horizon,
      f.model,
      f.model_version,
      f.feature_version,
      b.block,
      b.time || blockTime(b.block),
      b.p10,
      b.p50,
      b.p90,
      b.baseline,
      b.actual,
      e.mw,
      e.pct,
    ]
      .map(csvCell)
      .join(',');
  });
  return [header.join(','), ...rows].join('\r\n') + '\r\n';
}

export function downloadText(filename: string, text: string, mime = 'text/csv;charset=utf-8') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to legacy path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export function titleCase(s: string): string {
  return s.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** "9 Oct" from YYYY-MM-DD (UTC-safe). */
export function fmtDateShort(date?: string | null): string {
  if (!date) return '—';
  const t = Date.parse(`${date}T00:00:00Z`);
  if (!Number.isFinite(t)) return date;
  return new Date(t).toLocaleDateString('en-IN', {
    day: 'numeric',
    month: 'short',
    timeZone: 'UTC',
  });
}
