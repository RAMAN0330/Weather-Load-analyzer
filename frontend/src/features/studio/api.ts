/**
 * Forecast Engine v3 client — mirrors docs/forecast-v3-api.md.
 *
 * Uses the default axios instance so the global Bearer header set by
 * features/auth/authStore.js is attached automatically.
 */
import axios from 'axios';
import { useMutation, useQuery } from '@tanstack/react-query';
import { getTrainingApiUrl } from '../../apiConfig';

// ── Contract types ───────────────────────────────────────────────────────────

export type QualityStatus = 'ok' | 'degraded' | 'failed';
export type Horizon = 1 | 2;
export type ModelKey = 'lgbm_residual' | 'lgbm_direct' | 'recent_day' | 'seasonal_naive';

export interface ForecastRequest {
  region: string;
  target_date: string;
  horizon: Horizon;
  model: string;
}

export interface ForecastBlock {
  block: number;
  time: string;
  p10: number | null;
  p50: number | null;
  p90: number | null;
  baseline: number | null;
  actual: number | null;
}

export interface ForecastSummary {
  energy_mwh: number | null;
  peak_mw: number | null;
  peak_block: number | null;
  min_mw: number | null;
  min_block: number | null;
  mean_band_width_mw: number | null;
  max_ramp_mw: number | null;
  max_ramp_block: number | null;
}

/** Known driver keys; the backend may add more, so the record stays open. */
export interface ForecastDrivers {
  temp_weighted_max?: number | null;
  temp_p90_max?: number | null;
  wet_bulb_weighted_max?: number | null;
  wet_bulb_p90_max?: number | null;
  cdh_24h_end?: number | null;
  night_min_temp?: number | null;
  hot_share_peak?: number | null;
  precip_total_mm?: number | null;
  [key: string]: number | null | undefined;
}

export interface ForecastQuality {
  status: QualityStatus;
  weather_coverage_pct: number | null;
  load_share_coverage_pct: number | null;
  history_days: number | null;
  missing_load_blocks: number | null;
  warnings: string[];
}

export interface FeatureImportance {
  feature: string;
  gain_pct: number;
}

export interface ForecastResponse {
  forecast_id: string;
  region: string;
  target_date: string;
  origin_date: string;
  horizon: Horizon;
  model: string;
  model_version: string;
  feature_version: string;
  created_at: string;
  status: QualityStatus;
  blocks: ForecastBlock[];
  summary: ForecastSummary;
  drivers: ForecastDrivers;
  quality: ForecastQuality;
  feature_importance: FeatureImportance[];
}

export interface BacktestRequest {
  region: string;
  date_from: string;
  date_to: string;
  horizons: Horizon[];
  models: string[];
}

export interface BacktestMetrics {
  wape: number | null;
  accuracy_wape: number | null;
  mae: number | null;
  rmse: number | null;
  bias_mw: number | null;
  peak_wape: number | null;
  daily_peak_error_mw: number | null;
  p90_abs_error_mw: number | null;
  p95_abs_error_mw: number | null;
  pinball_loss: number | null;
  interval_coverage_pct: number | null;
}

export interface BacktestResult {
  model: string;
  horizon: Horizon;
  n_days: number;
  metrics: BacktestMetrics;
}

export interface BacktestDaily {
  model: string;
  horizon: Horizon;
  date: string;
  wape: number | null;
  mae: number | null;
  peak_error_mw: number | null;
}

export interface BacktestResponse {
  region: string;
  date_from: string;
  date_to: string;
  results: BacktestResult[];
  daily: BacktestDaily[];
  weather_note: string;
}

export interface DataQualityResponse {
  region: string;
  from: string;
  to: string;
  load: {
    expected_blocks: number;
    present_blocks: number;
    duplicate_blocks: number;
    nonpositive_blocks: number;
    missing_dates: string[];
  };
  weather: {
    districts_expected: number;
    districts_seen: number;
    coverage_pct: number | null;
    missing_districts: string[];
  };
  status: QualityStatus;
  warnings: string[];
}

export interface ModelInfo {
  key: string;
  label: string;
  quantiles: boolean;
  default?: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  model_version: string;
  feature_version: string;
}

/** Used when /v3/models is unreachable so the selector still works. */
export const FALLBACK_MODELS: ModelInfo[] = [
  { key: 'lgbm_residual', label: 'LightGBM residual', quantiles: true, default: true },
  { key: 'lgbm_direct', label: 'LightGBM direct', quantiles: true },
  { key: 'recent_day', label: 'Recent day', quantiles: false },
  { key: 'seasonal_naive', label: 'Seasonal naive', quantiles: false },
];

// ── Errors ───────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Normalise any thrown value into an ApiError whose message is the server `detail`. */
export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === 'string' && detail.trim()) return new ApiError(detail, status);
    // FastAPI request-validation errors: detail is a list of { loc, msg }.
    if (Array.isArray(detail) && detail.length) {
      const msg = detail
        .map((d) => {
          if (d && typeof d === 'object' && 'msg' in d) {
            const loc = Array.isArray((d as { loc?: unknown[] }).loc)
              ? (d as { loc: unknown[] }).loc.slice(1).join('.')
              : '';
            return loc ? `${loc}: ${(d as { msg: string }).msg}` : (d as { msg: string }).msg;
          }
          return String(d);
        })
        .join('; ');
      return new ApiError(msg, status);
    }
    if (status) return new ApiError(`Request failed (HTTP ${status})`, status);
    if (err.code === 'ECONNABORTED') return new ApiError('Request timed out', status);
    return new ApiError(err.message || 'Network error', status);
  }
  if (err instanceof Error) return new ApiError(err.message);
  return new ApiError('Unknown error');
}

async function call<T>(fn: () => Promise<{ data: T }>): Promise<T> {
  try {
    const { data } = await fn();
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

// ── Raw calls ────────────────────────────────────────────────────────────────

export function postForecast(req: ForecastRequest, signal?: AbortSignal) {
  return call<ForecastResponse>(() =>
    axios.post(getTrainingApiUrl('/v3/forecast'), req, { timeout: 180_000, signal })
  );
}

export function postBacktest(req: BacktestRequest) {
  return call<BacktestResponse>(() =>
    axios.post(getTrainingApiUrl('/v3/backtest'), req, { timeout: 600_000 })
  );
}

export function getDataQuality(region: string, days: number, signal?: AbortSignal) {
  return call<DataQualityResponse>(() =>
    axios.get(getTrainingApiUrl('/v3/data-quality'), {
      params: { region, days },
      timeout: 60_000,
      signal,
    })
  );
}

export function getModels(signal?: AbortSignal) {
  return call<ModelsResponse>(() =>
    axios.get(getTrainingApiUrl('/v3/models'), { timeout: 20_000, signal })
  );
}

// ── Query hooks ──────────────────────────────────────────────────────────────

export const studioKeys = {
  forecast: (req: ForecastRequest | null) => ['v3', 'forecast', req] as const,
  dataQuality: (region: string, days: number) => ['v3', 'data-quality', region, days] as const,
  models: () => ['v3', 'models'] as const,
};

/** Forecast for a submitted request. Pass `null` until the user hits Run. */
export function useForecastV3(req: ForecastRequest | null) {
  return useQuery<ForecastResponse, ApiError>({
    queryKey: studioKeys.forecast(req),
    queryFn: ({ signal }) => postForecast(req as ForecastRequest, signal),
    enabled: req != null,
    retry: false,
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });
}

export function useBacktestV3() {
  return useMutation<BacktestResponse, ApiError, BacktestRequest>({
    mutationFn: postBacktest,
  });
}

export function useDataQuality(region: string, days = 30) {
  return useQuery<DataQualityResponse, ApiError>({
    queryKey: studioKeys.dataQuality(region, days),
    queryFn: ({ signal }) => getDataQuality(region, days, signal),
    enabled: Boolean(region),
    retry: 1,
    staleTime: 5 * 60_000,
    refetchOnWindowFocus: false,
  });
}

export function useModels() {
  return useQuery<ModelsResponse, ApiError>({
    queryKey: studioKeys.models(),
    queryFn: ({ signal }) => getModels(signal),
    retry: 1,
    staleTime: 30 * 60_000,
    refetchOnWindowFocus: false,
  });
}
