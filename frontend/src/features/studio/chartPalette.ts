/**
 * Theme-aware ECharts palette for Forecast Studio.
 * Spreads the shared ecBase (src/lib/echartsTheme.ts, used by legacy pages and
 * left untouched) and overrides every colour per resolved theme.
 */
import { useMemo } from 'react';
import { ecBase } from '../../lib/echartsTheme';
import { useResolvedTheme, type ResolvedTheme } from './theme';

export interface ChartPalette {
  theme: ResolvedTheme;
  text: string;
  mutedText: string;
  axis: string;
  grid: string;
  axisPointer: string;
  tooltip: {
    bg: string;
    border: string;
    text: string;
    muted: string;
    divider: string;
    shadow: string;
  };
  legendInactive: string;
  zoom: {
    bg: string;
    border: string;
    filler: string;
    dataLine: string;
    dataArea: string;
    handle: string;
    moveHandle: string;
  };
  p50: string;
  band: string;
  bandStrong: string;
  baseline: string;
  actual: string;
  peakArea: string;
  peakAreaBorder: string;
  peakLabel: string;
  markBorder: string;
  bar: string;
  series: string[];
}

/** One Dark Pro: blue forecast, green actual, purple baseline, yellow peak window. */
const DARK: ChartPalette = {
  theme: 'dark',
  text: '#abb2bf',
  mutedText: '#9da5b4',
  axis: '#3e4451',
  grid: '#2f343e',
  axisPointer: '#7f848e',
  tooltip: {
    bg: 'rgba(33,37,43,0.97)',
    border: '#3e4451',
    text: '#d7dae0',
    muted: '#abb2bf',
    divider: 'rgba(171,178,191,0.14)',
    shadow: '0 8px 24px rgba(0,0,0,0.45)',
  },
  legendInactive: '#5c6370',
  zoom: {
    bg: '#21252b',
    border: '#3e4451',
    filler: 'rgba(97,175,239,0.12)',
    dataLine: '#7f848e',
    dataArea: 'rgba(127,132,142,0.18)',
    handle: '#61afef',
    moveHandle: '#7f848e',
  },
  p50: '#61AFEF',
  band: 'rgba(97,175,239,0.18)',
  bandStrong: 'rgba(97,175,239,0.85)',
  baseline: '#C678DD',
  actual: '#98C379',
  peakArea: 'rgba(229,192,123,0.06)',
  peakAreaBorder: 'rgba(229,192,123,0.3)',
  peakLabel: '#e5c07b',
  markBorder: '#282c34',
  bar: '#61AFEF',
  series: ['#61AFEF', '#C678DD', '#98C379', '#D19A66', '#E5C07B', '#E06C75', '#56B6C2', '#BE5046'],
};

/** One Light semantics (same meaning per colour as dark); darker hues so thin lines hold ≥3:1 on white. */
const LIGHT: ChartPalette = {
  theme: 'light',
  text: '#2b3240',
  mutedText: '#545b6a',
  axis: '#c9ced8',
  grid: '#e7eaf0',
  axisPointer: '#8a90a0',
  tooltip: {
    bg: 'rgba(255,255,255,0.98)',
    border: 'rgba(15,23,42,0.12)',
    text: '#141a26',
    muted: '#4a5160',
    divider: 'rgba(15,23,42,0.08)',
    shadow: '0 8px 24px rgba(30,41,59,0.14)',
  },
  legendInactive: '#b4b9c4',
  zoom: {
    bg: '#f4f6f9',
    border: '#d5d9e0',
    filler: 'rgba(20,83,219,0.10)',
    dataLine: '#9aa0ad',
    dataArea: 'rgba(154,160,173,0.18)',
    handle: '#1453db',
    moveHandle: '#9aa0ad',
  },
  p50: '#1453DB',
  band: 'rgba(20,83,219,0.13)',
  bandStrong: 'rgba(20,83,219,0.85)',
  baseline: '#8B30A6',
  actual: '#0F7A55',
  peakArea: 'rgba(217,119,6,0.07)',
  peakAreaBorder: 'rgba(217,119,6,0.3)',
  peakLabel: '#8a5207',
  markBorder: '#ffffff',
  bar: '#1453DB',
  series: ['#1453DB', '#8B30A6', '#0F7A55', '#B45309', '#A16207', '#C62828', '#0E7490', '#9A3412'],
};

export function getChartPalette(theme: ResolvedTheme): ChartPalette {
  return theme === 'light' ? LIGHT : DARK;
}

export function useChartPalette(): ChartPalette {
  const theme = useResolvedTheme();
  return useMemo(() => getChartPalette(theme), [theme]);
}

const MONO = "'IBM Plex Mono', monospace";

/** ecBase with all colours replaced from the palette. Spread into chart options. */
export function themedBase(p: ChartPalette) {
  return {
    ...ecBase,
    textStyle: { ...ecBase.textStyle, color: p.mutedText },
    tooltip: {
      ...ecBase.tooltip,
      backgroundColor: p.tooltip.bg,
      borderColor: p.tooltip.border,
      textStyle: { ...ecBase.tooltip.textStyle, color: p.tooltip.text },
      extraCssText: `box-shadow: ${p.tooltip.shadow}; border-radius: 8px; padding: 8px 12px;`,
    },
    xAxis: {
      ...ecBase.xAxis,
      axisLine: { lineStyle: { color: p.axis } },
      axisLabel: { ...ecBase.xAxis.axisLabel, color: p.mutedText },
    },
    yAxis: {
      ...ecBase.yAxis,
      axisLabel: { ...ecBase.yAxis.axisLabel, color: p.mutedText },
      splitLine: { lineStyle: { color: p.grid, type: 'dashed' as const, width: 1 } },
    },
  };
}

export function themedLegend(p: ChartPalette) {
  return {
    itemWidth: 16,
    itemHeight: 8,
    textStyle: { color: p.text, fontSize: 11, fontFamily: MONO },
    inactiveColor: p.legendInactive,
    pageTextStyle: { color: p.mutedText },
    pageIconColor: p.text,
    pageIconInactiveColor: p.legendInactive,
  };
}
