import axios from "axios";
import { getApiUrl } from "../../apiConfig";

export type TemperatureSensitivityInput = {
  baseline_temp: number;
  adjusted_temp: number;
  block_number: number;
  season: "summer" | "winter" | "monsoon";
  sector_mix: Record<string, number>;
};

export type TemperatureSensitivityResult = {
  block_number: number;
  season: "summer" | "winter" | "monsoon";
  baseline_temp: number;
  adjusted_temp: number;
  sector_mix: Record<string, number>;
  steps: {
    temp_delta: number;
    seasonal_elasticity_coeff: number;
    after_season: number;
    time_of_day_multiplier: number;
    after_time_of_day: number;
    comfort_band_multiplier: number;
    after_comfort_band: number;
    sector_weight: number;
    raw_impact_pct: number;
  };
  temperature_impact_pct: number;
  clamp_bounds_pct: [number, number];
};

export const calculateTemperatureSensitivity = async (
  payload: TemperatureSensitivityInput
): Promise<TemperatureSensitivityResult> => {
  const url = getApiUrl("/temperature-sensitivity/calculate");
  const res = await axios.post(url, payload);
  return res.data;
};
