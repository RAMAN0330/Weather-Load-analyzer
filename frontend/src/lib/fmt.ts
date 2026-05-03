export const fmtMW = (n: number | null | undefined): string =>
  n == null ? '—' : n.toLocaleString('en-IN', { maximumFractionDigits: 0 }) + ' MW'

export const fmtMWh = (n: number | null | undefined): string =>
  n == null ? '—' : n.toLocaleString('en-IN', { maximumFractionDigits: 0 }) + ' MWh'

export const fmtPct = (n: number | null | undefined, d = 1): string =>
  n == null ? '—' : n.toFixed(d) + '%'

export const fmtNum = (n: number | null | undefined, d = 0): string =>
  n == null ? '—' : n.toLocaleString('en-IN', { maximumFractionDigits: d })

export const fmtMWShort = (n: number | null | undefined): string => {
  if (n == null) return '—'
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + ' GW'
  return fmtNum(n, 0) + ' MW'
}
