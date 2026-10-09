# Theming guide — Light / Dark (One Dark Pro)

Every page must follow the app theme. The resolved theme is written to `<html>`
as `data-theme="light|dark"` by `src/features/studio/theme.ts`; CSS variables in
`src/index.css` switch with it. **Never hard-code a colour in a page.**

## 1. Tokens (CSS variables)

| Purpose | Variable | Dark (One Dark Pro) | Light |
|---|---|---|---|
| Page background | `--bg` | #21252B | #F3F4F7 |
| Card / panel | `--bg-panel`, `--bg-elevated` | #282C34 | #FFFFFF |
| Inset / input / table head | `--bg-surface` | #2C313A | #F5F6F8 |
| Hover / active row | `--bg-hover`, `--bg-active` | #333842 / #3E4451 | #ECEEF2 / #E2E5EB |
| Borders | `--outline`, `--outline-light` | #3E4451 / #4B5263 | #D9DCE3 / #C9CDD6 |
| Text | `--text` | #D7DAE0 | #1B2130 |
| Secondary text | `--text-secondary` | #ABB2BF | #3B4252 |
| Muted / labels | `--text-muted` | #9DA5B4 | #545B6A |
| Faint | `--text-dim` | #969CA6 | #5F6675 |
| Accent (UI: active tab, primary button, focus, links) | `--accent` (+ `--accent-dim`) | #61AFEF | #1453DB |
| Text on an accent fill | `--accent-fg` | #282C34 | #FFFFFF |
| Secondary accent (purple) | `--accent2` | #C678DD | #8B30A6 |
| Warm / "uplift" / orange data | `--tone-warm` | #D19A66 | #B45309 |
| Success / Warning / Danger / Info | `--success` `--warning` `--danger` `--info` (+ `-dim`) | One Dark | AA-darkened |
| Channels for alpha | `--accent-rgb`, `--overlay-rgb`, `--shadow-rgb`, `--panel-rgb` | | |

Alpha tints: `rgba(var(--accent-rgb), 0.12)`, hover/hairline overlays
`rgba(var(--overlay-rgb), 0.05)` (white in dark, slate in light), dark glass
panels `rgba(var(--panel-rgb), 0.9)`, shadows `rgba(var(--shadow-rgb), 0.2)`.

## 2. Legacy hex → token map (inline `style`, CSS-in-JS strings)

| Legacy | Use |
|---|---|
| `#ECEEF3` `#F0F2F8` `#E2E4E9` `#F8FAFC` (and `#fff` used as *text* on panels) | `var(--text)` |
| `#B8BDCC` `#A0A5B8` `#C8CCD8` `#94A3B8` `#CCC` | `var(--text-secondary)` |
| `#8A90A6` `#6B7186` `#64748B` `#7A8090` `#AAA` | `var(--text-muted)` |
| `#555A6E` `#565B6B` `#636E72` | `var(--text-dim)` |
| `#2A292F` `#262A35` `#333` | `var(--outline)`; `#3A3940` → `var(--outline-light)` |
| `#1A191E` `#1A1922` `rgba(26,25,30,a)` `rgba(30,29,35,a)` `rgba(22,22,27,a)` `rgba(20,20,24,a)` `rgba(18,20,28,a)` … (dark neutrals) | `var(--bg-panel)` or `rgba(var(--panel-rgb), a)` |
| `#201F25` `#141419` | `var(--bg-surface)` |
| `#121214` `#0E0D12` | `var(--bg)` |
| `#F07825` | UI accent → `var(--accent)`; a *data series / uplift* → `var(--tone-warm)` |
| `#5B9FE4` `#4A90D9` `#3B82F6` `#60A5FA` | `var(--accent)` (blue) |
| `#34D399` `#22C55E` `#238636` | `var(--success)` |
| `#F87171` `#EF4444` `#F43F5E` | `var(--danger)` |
| `#FBBF24` `#F59E0B` | `var(--warning)` |
| `#C084FC` `#F472B6` | `var(--accent2)` |
| `#45B7D1` `#38BDF8` | `var(--info)` |
| `rgba(255,255,255,a)` | `rgba(var(--overlay-rgb), a)` |
| `rgba(240,120,37,a)` | `rgba(var(--accent-rgb), a)` |
| `rgba(0,0,0,a)` shadows | `rgba(var(--shadow-rgb), a)` |
| `#fff` text on an accent/coloured fill | `var(--accent-fg)` (or keep `#fff` only on a fixed dark/brand fill) |

Hex-alpha string tricks (`` `${color}22` ``) break with variables — use
`` `color-mix(in srgb, ${color} 13%, transparent)` `` instead.

## 3. Charts (ECharts canvas / recharts SVG) — `var()` does NOT work there

```js
import { useChartTokens, themedEcBase, withAlpha } from '../../lib/chartTheme';
const t = useChartTokens();                    // resolved hex strings, updates on theme change
const option = useMemo(() => ({
  ...themedEcBase(t),
  series: [{ lineStyle: { color: t.accent }, areaStyle: { color: withAlpha(t.accent, 0.15) } }],
}), [data, t]);                                // include `t` in deps!
```
Recharts: `stroke={t.accent}`, `tick={{ fill: t.textMuted }}`, `<CartesianGrid stroke={t.outline} />`.
Series meaning is fixed across the app: **forecast / adjusted = `accent` (blue), actual = `success` (green),
baseline = `accent2` (purple, dashed)**, peak window / warnings = `warning`, error = `danger`.
Other categorical series may use `t.warm`, `t.info`, `t.accent2`, `t.warning`.

## 4. Rules
- Never remove behaviour; colour/contrast edits only (plus layout only where asked).
- Text must stay ≥ 4.5:1 against its background in **both** themes.
- Gradients that are white→grey "metal" text: use `var(--title-top)` → `var(--text-secondary)`.
- Keep the Forecast Studio brand mark (orange tile) as is.
- Fixed dark overlays (modal backdrops) may stay dark in both themes.
