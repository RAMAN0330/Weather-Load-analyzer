/** API response shapes used across the redesign.  Loose-typed where the
    backend payload is sprawling — we declare only the fields we read. */

export type Region = string;

export interface ConfigResponse {
  default_date?: string;
  latest_date?: string;
  partial_latest_date?: string;
  best_baseline_window?: number;
  available_regions?: Region[];
  default_region?: Region;
  dates?: string[];
  all_dates?: string[];
}

export interface ForecastSeries {
  blocks: number[];
  forecast: number[];
  hybrid_baseline?: number[];
  weather_impact?: number[];
  actual?: Array<number | null>;
  weather_impact_pct?: number[];
}

export interface ForecastMetadata {
  effective_date?: string;
  forecast_date?: string;
  region?: string;
  partial_latest_date?: string;
  total_drawal_today?: number;
  peak_load?: number;
  peak_block?: number;
}

export interface SimilarDayPick {
  date: string;
  similarity_score?: number;
  temp_diff?: number;
  hum_diff?: number;
  rain_match?: boolean;
  scale?: number;
}

export interface DriverContribution {
  feature: string;
  value_mw?: number;
  value_pct?: number;
}

export interface ForecastResponse {
  series?: ForecastSeries;
  metadata?: ForecastMetadata;
  kpis_full?: Record<string, number | string | null>;
  similar_days?: SimilarDayPick[];
  driver_contributions?: DriverContribution[];
  decision_signals?: Array<{ block?: number; severity?: string; message?: string }>;
  block_contributors?: Array<{ block: number; primary?: string; secondary?: string }>;
  forecast_uncertainty?: Array<{ block: number; p10?: number; p90?: number }>;
  bias_factor?: number;
  explanation?: string;
  weather_analysis?: WeatherAnalysis;
  base_date?: string;
  t2_date?: string;
  horizon?: 't1' | 't2' | string;
}

export interface WeatherIntradaySeries {
  actual?: number[];
  normal?: number[];
  delta?: number[];
}

export interface WeatherAnalysis {
  intraday?: {
    temperature?: WeatherIntradaySeries;
    humidity?: WeatherIntradaySeries;
    precipitation?: WeatherIntradaySeries;
    rain?: WeatherIntradaySeries;
    showers?: WeatherIntradaySeries;
    snowfall?: WeatherIntradaySeries;
    cloud_cover?: WeatherIntradaySeries;
    solar_radiation?: WeatherIntradaySeries;
    direct_radiation?: WeatherIntradaySeries;
  };
  peak_windows?: unknown;
  rain_metrics?: { total_mm?: number };
  sensitivity?: unknown;
  dod_changes?: unknown;
  dod_series?: unknown;
}

export type Horizon = 't1' | 't2';
