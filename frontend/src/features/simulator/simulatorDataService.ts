import axios from "axios";
import { getApiUrl } from "../../apiConfig";


export type SimulatorBlockApi = {
  block_number: number;
  baseline_mw: number;
  final_mw?: number;
  selection_mask?: number;
  impact?: number;
  applied_impact?: number;
  temperature_impact_pct?: number;
  humidity_impact_pct?: number;
  precipitation_impact_pct?: number;
  wind_impact_pct?: number;
  temperature_increase_delta_pct?: number | null;
  temperature_reduction_delta_pct?: number | null;
  humidity_increase_delta_pct?: number | null;
  humidity_reduction_delta_pct?: number | null;
  precipitation_increase_delta_pct?: number | null;
  precipitation_reduction_delta_pct?: number | null;
  weather_total_impact_pct?: number;
  weather_increase_delta_pct?: number | null;
  weather_reduction_delta_pct?: number | null;
  weather_direction?: "increase" | "decrease" | "neutral";
  actual_mw?: number | null;
  actual_delta_pct?: number | null;
  actual_increase_delta_pct?: number | null;
  actual_reduction_delta_pct?: number | null;
  actual_direction?: "increase" | "decrease" | "neutral";
  residual_delta_pct?: number | null;
  residual_direction?: "increase" | "decrease" | "neutral";
  residual_contributor?: string;
  actual_vs_weather_alignment?: "aligned" | "opposite" | "weak_signal" | "na";
  actual_vs_weather_explanation?: string;
  season?: string;
  calendar_day_type?: string;
  temperature?: number | null;
  humidity?: number | null;
  precipitation?: number | null;
  wind_mps?: number | null;
  temp_delta_c?: number | null;
  humidity_delta_pct?: number | null;
  precip_delta_mm?: number | null;
  wind_delta_mps?: number | null;
  temperature_effective_coeff?: number | null;
  temperature_increase_coeff?: number | null;
  temperature_reduction_coeff?: number | null;
  weight_confidence?: number | null;
  regime_confidence?: number | null;
};

export type BlockDriverMatrixRowApi = {
  date: string;
  block: number;
  temp_weight?: number;
  humidity_weight?: number;
  rain_weight?: number;
  wind_weight?: number;
  daytype_weight?: number;
  holiday_weight?: number;
  temp_delta?: number;
  humidity_delta?: number;
  rain_delta?: number;
  wind_delta?: number;
  daytype_flag?: number;
  holiday_flag?: number;
  weather_base: number;
  temperature_base?: number;
  humidity_base?: number;
  precipitation_base?: number;
  wind_base?: number;
  cloud_cover_base?: number;
  solar_irradiance_base?: number;
  pressure_base?: number;
  daytype_base: number;
  holiday_base: number;
  manual_base: number;
  weather_scaled: number;
  temperature_scaled?: number;
  humidity_scaled?: number;
  precipitation_scaled?: number;
  wind_scaled?: number;
  cloud_cover_scaled?: number;
  solar_irradiance_scaled?: number;
  pressure_scaled?: number;
  daytype_scaled: number;
  holiday_scaled: number;
  manual_scaled: number;
  final_impact: number;
};

export type DriverWeightMatrixRowApi = {
  date: string;
  block: number;
  temp_weight: number;
  humidity_weight: number;
  rain_weight: number;
  wind_weight: number;
  daytype_weight: number;
  holiday_weight: number;
  weight_confidence?: number;
  sample_confidence?: number;
  fit_confidence?: number;
  samples?: number;
  model_r2?: number;
  alpha_opt?: number | null;
  l1_ratio_opt?: number | null;
};

export type SimulatorBlocksRequestPayload = {
  date?: string;
  baseline_days?: number;
  sliders?: {
    temperature?: number;
    humidity?: number;
    precipitation?: number;
    wind?: number;
    cloud_cover?: number;
    solar_irradiance?: number;
    pressure?: number;
    weather: number;
    daytype: number;
    holiday: number;
    manual: number;
  };
  selection?: {
    scope: "all" | "range" | "single";
    start_block?: number;
    end_block?: number;
    block?: number;
  };
  smooth_selection?: boolean;
  temp_delta?: number[];
  humidity_delta?: number[];
  rain_delta?: number[];
  wind_delta?: number[];
  daytype_flag?: number[];
  holiday_flag?: number[];
  weather_base?: number[];
  temperature_base?: number[];
  humidity_base?: number[];
  precipitation_base?: number[];
  wind_base?: number[];
  cloud_cover_base?: number[];
  solar_irradiance_base?: number[];
  pressure_base?: number[];
  daytype_base?: number[];
  holiday_base?: number[];
  manual_base?: number[];
};

export type SimulatorBlocksResponse = {
  date: string;
  baseline_days: number;
  available_dates?: string[];
  temperature_delta_profile?: {
    source?: string;
    prior_coeff?: number;
    global_increase_coeff?: number;
    global_reduction_coeff?: number;
    global_increase_samples?: number;
    global_reduction_samples?: number;
    diagnostics?: Record<string, unknown>;
  };
  sliders?: {
    temperature?: number;
    humidity?: number;
    precipitation?: number;
    wind?: number;
    cloud_cover?: number;
    solar_irradiance?: number;
    pressure?: number;
    weather: number;
    daytype: number;
    holiday: number;
    manual: number;
  };
  selection?: {
    scope: "all" | "range" | "single";
    start_block?: number;
    end_block?: number;
    block?: number;
  };
  weights_source?: string;
  partial_day_bias_correction?: {
    enabled?: boolean;
    applied?: boolean;
    available_blocks?: number;
    reason?: string;
    fit_points?: number;
    intercept_pct?: number;
    slope_pct_per_block?: number;
    confidence_gain?: number;
    avg_correction_pct?: number;
    max_correction_pct?: number;
  };
  driver_weight_matrix?: DriverWeightMatrixRowApi[];
  block_driver_matrix?: BlockDriverMatrixRowApi[];
  blocks: SimulatorBlockApi[];
};

export type SimulatorDateOptionsResponse = {
  dates: string[];
  all_dates?: string[];
  latest_date?: string | null;
  partial_latest_date?: string | null;
  default_date?: string | null;
};

export const loadSimulatorBlocksApi = async (date?: string, baselineDays = 7, payloadOverride?: Partial<SimulatorBlocksRequestPayload>) => {
  const payload: SimulatorBlocksRequestPayload = { date, baseline_days: baselineDays, ...(payloadOverride || {}) };
  const url = getApiUrl("/simulator/blocks");
  const res = await axios.post(url, payload);
  return res.data as SimulatorBlocksResponse;
};

export const loadSimulatorDateOptionsApi = async (): Promise<SimulatorDateOptionsResponse> => {
  const url = getApiUrl("/v2/config");
  const res = await axios.get(url);
  return res.data as SimulatorDateOptionsResponse;
};
