const BASE = '/api/pipeline'

export const DB_STATES = ['HARYANA', 'ODISHA', 'RAJASTHAN', 'CHHATTISGARH']

const safeFetch = (url) =>
  fetch(url).then((r) => {
    if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || 'Request failed') })
    return r.json()
  })

export const fetchPipelineStates = () => safeFetch(`${BASE}/states`)

// Use days=62 so the query window always includes T+1 and T+2 (today + tomorrow)
export const fetchPipelineWeather = (state, days = 62) =>
  safeFetch(`${BASE}/weather/${encodeURIComponent(state)}?days=${days}`)

export const fetchPipelineWeatherLoc = (state, days = 60) =>
  safeFetch(`${BASE}/weather-loc/${encodeURIComponent(state)}?days=${days}`)

export const fetchPipelineLoad = (state, days = 60) =>
  safeFetch(`${BASE}/load/${encodeURIComponent(state)}?days=${days}`)

export const fetchPipelineForecast = (state, days = 60) =>
  safeFetch(`${BASE}/forecast/${encodeURIComponent(state)}?days=${days}`)

export const fetchPipelineSimilarity = (state, targetDate, method = 'euclidean', topK = 5) =>
  safeFetch(
    `${BASE}/similarity/${encodeURIComponent(state)}?target_date=${targetDate}&method=${method}&top_k=${topK}`
  )

/** blockToTime: block 1-96 → "HH:MM" label */
export const blockToTime = (block) => {
  const m = (block - 1) * 15
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`
}

/** Build time-axis labels for 96 blocks */
export const buildTimeAxis = () => Array.from({ length: 96 }, (_, i) => blockToTime(i + 1))
