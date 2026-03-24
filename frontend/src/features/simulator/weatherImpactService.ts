import axios from "axios";

export type WeatherImpactSeason = "summer" | "winter" | "monsoon";

export type WeatherDeltaBlock = {
  temp_delta_c: number;
  humidity_delta_pct: number;
  precip_delta_mm: number;
  wind_delta_mps: number;
};

export type WeatherImpactResponse = {
  season: WeatherImpactSeason;
  coefficients: {
    temp_coeff: number;
    humidity_coeff: number;
    rain_coeff: number;
    wind_coeff: number;
  };
  impacts: Array<{
    block_number: number;
    time_multiplier: number;
    raw_weather_pct: number;
    weather_pct: number;
  }>;
};

export const calculateWeatherImpact = async (
  season: WeatherImpactSeason,
  blocks: WeatherDeltaBlock[]
): Promise<WeatherImpactResponse> => {
  const payload = { season, blocks };
  const primary = "/api/weather-impact/calculate";
  const fallback = "http://localhost:8000/api/weather-impact/calculate";
  try {
    const res = await axios.post(primary, payload);
    return res.data;
  } catch {
    const res = await axios.post(fallback, payload);
    return res.data;
  }
};

