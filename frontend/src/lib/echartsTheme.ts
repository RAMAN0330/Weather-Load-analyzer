// Shared ECharts base config — spread into every chart option object.
// Override individual keys per-chart as needed.
export const ecBase = {
  backgroundColor: 'transparent',
  textStyle: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: '#8a90a6',
    fontSize: 11,
  },
  tooltip: {
    backgroundColor: 'rgba(22,22,27,0.96)',
    borderColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1,
    textStyle: {
      color: '#f0f2f8',
      fontSize: 12,
      fontFamily: "'IBM Plex Mono', monospace",
    },
    extraCssText:
      'box-shadow: 0 8px 24px rgba(0,0,0,0.4); border-radius: 8px; padding: 8px 12px;',
  },
  grid: { left: 52, right: 16, top: 20, bottom: 32, containLabel: false },
  xAxis: {
    axisLine: { lineStyle: { color: '#2a292f' } },
    axisTick: { show: false },
    axisLabel: { color: '#555a6e', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" },
    splitLine: { show: false },
  },
  yAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: '#555a6e', fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" },
    splitLine: { lineStyle: { color: '#1e1d23', type: 'dashed' as const, width: 1 } },
  },
}

// Accent colors for series — use these for consistency
export const ecColors = {
  forecast: '#F07825',
  actual:   '#34D399',
  baseline: '#5B9FE4',
  p10:      'rgba(240,120,37,0.25)',
  p90:      'rgba(240,120,37,0.25)',
  warn:     '#FBBF24',
  danger:   '#F87171',
}
