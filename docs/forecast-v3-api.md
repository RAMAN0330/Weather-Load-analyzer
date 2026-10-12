# Forecast Engine v3 — API Contract

Served by FastAPI (`backend/forecast_v3_router.py`), reached from the browser as
`/ml-api/v3/...` (nginx/Vite rewrite `/ml-api/*` → FastAPI `/api/*`).
All routes require the Bearer session token (see `backend/api_auth.py`).

Conventions: dates are `YYYY-MM-DD`; blocks are 1..96 (block 1 = 00:00–00:15 IST);
all load values are MW of **net grid drawal** (`total_drawal`). `null` = missing, never 0.

## Daily issue (T+1 and T+2 only)

Forecasts are issued **once per operating day (IST)** for exactly two targets:
T+1 = tomorrow and T+2 = the day after. A background job (every
`DAILY_JOB_INTERVAL_S`, default 900 s) tops up the dummy data feed and issues
both forecasts with the default model; results are cached for the day.
Past or other target dates are rejected (422) — evaluate history with
`/api/v3/backtest`. Only region `haryana` is available.

## GET /api/v3/forecast/daily?model=lgbm_residual&refresh=false

Returns today's issued pair (served from cache; `refresh=true` recomputes):
```json
{ "ops_date": "2026-10-12", "region": "haryana", "model": "lgbm_residual",
  "targets": { "1": "2026-10-13", "2": "2026-10-14" },
  "t1": { "...": "forecast response (below)", "issued_for": "2026-10-12", "issued_at": "2026-10-12T06:43:01+05:30" },
  "t2": { "...": "forecast response (below)" } }
```

## GET /api/v3/status

Daily-job state: `ops_date`, `targets`, cached forecasts, last run / error,
dummy-feed flag and DB query-cache stats (`hits`, `misses`, `entries`, `ttl_s`).

## POST /api/v3/forecast

Request:
```json
{ "horizon": 1, "model": "lgbm_residual", "refresh": false }
```
- `target_date` is optional; if given it must equal today + `horizon` (else 422).
- `refresh: true` recomputes with the latest data instead of returning today's issued forecast.
- `horizon`: 1 (T+1) or 2 (T+2). Origin = `target_date - horizon` days; only data
  dated **≤ origin** is used.
- `model`: `lgbm_residual` (default) | `lgbm_direct` | `recent_day` | `seasonal_naive`.

Response:
```json
{
  "forecast_id": "hry-20261010-h1-3f9c2a1b",
  "region": "haryana",
  "target_date": "2026-10-10",
  "origin_date": "2026-10-09",
  "horizon": 1,
  "model": "lgbm_residual",
  "model_version": "v3.0.0",
  "feature_version": "wx-f1",
  "created_at": "2026-10-09T10:12:03Z",
  "status": "ok",
  "blocks": [
    { "block": 1, "time": "00:00", "p10": 5810.2, "p50": 6020.4, "p90": 6230.9,
      "baseline": 5990.0, "actual": null }
  ],
  "summary": {
    "energy_mwh": 145230.5, "peak_mw": 9120.3, "peak_block": 60,
    "min_mw": 5402.1, "min_block": 18, "mean_band_width_mw": 410.2,
    "max_ramp_mw": 380.4, "max_ramp_block": 72
  },
  "drivers": {
    "temp_weighted_max": 34.1, "temp_p90_max": 36.0, "wet_bulb_weighted_max": 27.2,
    "wet_bulb_p90_max": 28.4, "cdh_24h_end": 112.0, "night_min_temp": 24.3,
    "hot_share_peak": 0.62, "precip_total_mm": 0.4
  },
  "quality": {
    "status": "ok",
    "weather_coverage_pct": 95.2,
    "load_share_coverage_pct": 98.1,
    "history_days": 728,
    "missing_load_blocks": 0,
    "warnings": ["Weather for 1 of 21 districts missing; weights renormalised."]
  },
  "feature_importance": [ { "feature": "lag_7d_load", "gain_pct": 31.2 } ]
}
```
- `status` / `quality.status`: `ok` | `degraded` | `failed`. `degraded` means a
  forecast was produced but inputs were below standard; the UI must show the warnings.
- `actual` is filled only for blocks where an actual exists (past target dates).
- `blocks` always has exactly 96 entries; p10 ≤ p50 ≤ p90 holds for every block.

## POST /api/v3/backtest

Request:
```json
{ "region": "haryana", "date_from": "2026-08-01", "date_to": "2026-08-31",
  "horizons": [1, 2], "models": ["seasonal_naive", "recent_day", "lgbm_residual"] }
```
Rolling origin: every target date is forecast using only data ≤ its origin.
Capped at 62 target dates per call.

Response:
```json
{
  "region": "haryana", "date_from": "2026-08-01", "date_to": "2026-08-31",
  "results": [
    { "model": "lgbm_residual", "horizon": 1, "n_days": 31,
      "metrics": { "wape": 2.41, "accuracy_wape": 97.59, "mae": 152.3, "rmse": 201.8,
                   "bias_mw": -12.4, "peak_wape": 2.9, "daily_peak_error_mw": 140.2,
                   "p90_abs_error_mw": 310.0, "p95_abs_error_mw": 402.1,
                   "pinball_loss": 61.2, "interval_coverage_pct": 78.4 } }
  ],
  "daily": [
    { "model": "lgbm_residual", "horizon": 1, "date": "2026-08-01",
      "wape": 2.1, "mae": 130.2, "peak_error_mw": 95.1 }
  ],
  "weather_note": "Historical weather is the stored (possibly observed) value, not the forecast issued at origin; accuracy may be optimistic."
}
```
`accuracy_wape` = 100 − WAPE ("WAPE-derived accuracy").

## GET /api/v3/data-quality?region=haryana&days=30

```json
{
  "region": "haryana", "from": "2026-09-09", "to": "2026-10-08",
  "load": { "expected_blocks": 2880, "present_blocks": 2876, "duplicate_blocks": 0,
            "nonpositive_blocks": 0, "missing_dates": [] },
  "weather": { "districts_expected": 21, "districts_seen": 20,
               "coverage_pct": 95.1, "missing_districts": ["FATEHABAD"] },
  "status": "degraded",
  "warnings": ["..."]
}
```

## GET /api/v3/models

```json
{ "models": [ { "key": "lgbm_residual", "label": "LightGBM residual", "quantiles": true,
                "default": true } ],
  "model_version": "v3.0.0", "feature_version": "wx-f1" }
```

## Errors
`4xx/5xx` bodies are `{ "detail": "human readable message" }`. A `422` is returned when
history is insufficient (< 28 days before origin).
