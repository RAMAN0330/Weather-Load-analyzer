/**
 * Theme tokens for chart libraries (ECharts canvas, recharts SVG) that can't
 * resolve CSS `var(...)`. Values come from the active theme's CSS variables and
 * update when the theme changes, so include the returned object in chart
 * option memo dependencies.
 */
import { ecBase } from './echartsTheme';
import { useThemeColors } from './useThemeColors';

export const CHART_TOKEN_VARS = {
  text: '--text',
  textSecondary: '--text-secondary',
  textMuted: '--text-muted',
  outline: '--outline',
  bg: '--bg',
  panel: '--bg-panel',
  elevated: '--bg-elevated',
  surface: '--bg-surface',
  accent: '--accent',
  accent2: '--accent2',
  success: '--success',
  warning: '--warning',
  danger: '--danger',
  info: '--info',
  warm: '--tone-warm',
} as const;

export type ChartTokens = Record<keyof typeof CHART_TOKEN_VARS, string>;

export function useChartTokens(): ChartTokens {
  return useThemeColors(CHART_TOKEN_VARS);
}

/** `#RRGGBB` (or `#RGB`) → `rgba(r, g, b, a)`; non-hex input is returned unchanged. */
export function withAlpha(color: string, a: number): string {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(color).trim());
  if (!m) return color;
  let h = m[1];
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** Shared ECharts base (`ecBase`) with every colour taken from the theme. */
export function themedEcBase(t: ChartTokens) {
  return {
    ...ecBase,
    textStyle: { ...ecBase.textStyle, color: t.textMuted },
    tooltip: {
      ...ecBase.tooltip,
      backgroundColor: t.elevated,
      borderColor: t.outline,
      textStyle: { ...ecBase.tooltip.textStyle, color: t.text },
      extraCssText: 'box-shadow: 0 8px 24px rgba(0,0,0,0.18); border-radius: 8px; padding: 8px 12px;',
    },
    xAxis: {
      ...ecBase.xAxis,
      axisLine: { lineStyle: { color: t.outline } },
      axisLabel: { ...ecBase.xAxis.axisLabel, color: t.textMuted },
    },
    yAxis: {
      ...ecBase.yAxis,
      axisLabel: { ...ecBase.yAxis.axisLabel, color: t.textMuted },
      splitLine: { lineStyle: { color: withAlpha(t.outline, 0.7), type: 'dashed' as const, width: 1 } },
    },
  };
}
