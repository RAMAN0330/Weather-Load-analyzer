const BASE = '/api/pipeline';

export const DB_STATES = ['HARYANA', 'ODISHA', 'RAJASTHAN', 'CHHATTISGARH'];

const safeFetch = (url) =>
  fetch(url).then((r) => {
    if (!r.ok)
      return r.json().then((e) => {
        throw new Error(e.detail || 'Request failed');
      });
    return r.json();
  });

export const fetchPipelineStates = () => safeFetch(`${BASE}/states`);

// Use days=62 so the query window always includes T+1 and T+2 (today + tomorrow)
export const fetchPipelineWeather = (state, days = 62) =>
  safeFetch(`${BASE}/weather/${encodeURIComponent(state)}?days=${days}`);

export const fetchPipelineWeatherLoc = (state, days = 60) =>
  safeFetch(`${BASE}/weather-loc/${encodeURIComponent(state)}?days=${days}`);

export const fetchPipelineCircleImpact = (state, days = 365) =>
  safeFetch(`${BASE}/circle-impact/${encodeURIComponent(state)}?days=${days}`);

export const fetchPipelineLoad = (state, days = 60) =>
  safeFetch(`${BASE}/load/${encodeURIComponent(state)}?days=${days}`);

export const fetchPipelineForecast = (state, days = 60) =>
  safeFetch(`${BASE}/forecast/${encodeURIComponent(state)}?days=${days}`);

/**
 * Similar-day search.
 *
 * Default method: `euclidean` over the joint (load profile + weather exogs)
 * vector — temperature, humidity, precipitation, plus the 96-block load
 * curve.  Other methods are accepted (weighted, cosine) but euclidean is
 * the canonical baseline.  `include_weather=true` is forwarded so the
 * backend treats weather features as part of the distance, not just load.
 */
export const fetchPipelineSimilarity = (state, targetDate, method = 'euclidean', topK = 5) =>
  safeFetch(
    `${BASE}/similarity/${encodeURIComponent(state)}?target_date=${targetDate}` +
      `&method=${method}&top_k=${topK}&include_weather=true`
  );

// ── Analytics date-range variants ─────────────────────────────────────────

export const fetchPipelineWeatherRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/weather/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

export const fetchPipelineWeatherLocRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/weather-loc/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

export const fetchPipelineLoadRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/load/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

export const fetchPipelineForecastRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/forecast/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

export const fetchPipelineSldcRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/sldc/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

export const fetchPipelineSldcForecastRange = (state, from_date, to_date) =>
  safeFetch(`${BASE}/sldc-forecast/${encodeURIComponent(state)}?from_date=${from_date}&to_date=${to_date}`);

/** blockToTime: block 1-96 → "HH:MM" label */
export const blockToTime = (block) => {
  const m = (block - 1) * 15;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};

/** Build time-axis labels for 96 blocks */
export const buildTimeAxis = () => Array.from({ length: 96 }, (_, i) => blockToTime(i + 1));
