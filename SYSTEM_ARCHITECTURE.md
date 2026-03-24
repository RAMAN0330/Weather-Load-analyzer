# GNA Energy — Load Forecasting System Architecture

## Complete Backend & Frontend Documentation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Backend Architecture](#2-backend-architecture)
   - [Module Structure](#21-module-structure)
   - [Short-Term Pipeline](#22-short-term-pipeline-short_term_pipelinepy)
   - [EDA Engine](#23-eda-engine-enginepy)
   - [FastAPI Server](#24-fastapi-server-mainpy)
3. [Frontend Architecture](#3-frontend-architecture)
   - [State Management](#31-state-management)
   - [API Integration](#32-api-integration)
   - [Tabs & Views](#33-tabs--views)
   - [Chart Components](#34-chart-components)
   - [Data Flow](#35-data-flow)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Algorithm Reference](#5-algorithm-reference)
6. [Configuration Reference](#6-configuration-reference)
7. [API Reference](#7-api-reference)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React/Vite)                       │
│  App.jsx → Tabs → Charts (ECharts) → KPI Cards → Simulator         │
│  State: useState + Zustand store                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (axios)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI / Uvicorn)                    │
│  main.py ──────► short_term_pipeline.py (Forecast Engine)           │
│           ──────► engine.py (EDA Engine)                            │
│  Data: final_data.csv (96-block × N days)                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Purpose**: Day-ahead 96-block (15-minute interval) electricity load forecasting for Indian power utilities, with weather attribution, holiday detection, and interactive simulation.

---

## 2. Backend Architecture

### 2.1 Module Structure

| File | Lines | Purpose |
|------|-------|---------|
| `backend/short_term_pipeline.py` | ~3800 | Core forecasting: baseline, weather model, calendar, ratio-based forecast |
| `backend/engine.py` | ~1700 | EDA engine: feature registry, data loading, statistical analysis |
| `backend/main.py` | ~4200 | FastAPI REST API: endpoints, KPI computation, response formatting |

### 2.2 Short-Term Pipeline (`short_term_pipeline.py`)

#### 2.2.1 Configuration (`DEFAULT_CONFIG`)

```python
DEFAULT_CONFIG = {
    # Baseline & Similarity
    "candidate_lookback_days": 15,       # Days to search for similar days
    "similar_days_top_n": 5,             # Number of similar days to average
    "temp_band_c": 3.0,                  # Temperature match band (°C)
    "humidity_band": 12.0,               # Humidity match band (%)
    "require_rain_match": True,          # Require rain/no-rain match
    "similarity_weights": {              # Distance weighting
        "temp": 0.55, "humidity": 0.25, "rain": 0.2
    },
    "baseline_window_candidates": [3, 5, 7, 10, 14],
    "baseline_backtest_days": 10,        # Days to test each window

    # Weather Training
    "weather_training_days": 120,        # Days of history for ML models
    "weather_tune": True,                # Enable hyperparameter search
    "weather_tune_iters": 12,            # Random search iterations

    # Temperature Asymmetry
    "temp_asym_lookback_days": 180,      # History for temp elasticity
    "temp_asym_rolling_days": 7,
    "temp_asym_min_samples": 8,

    # Forecast Blend
    "hybrid_weights": {"alpha": 0.6, "beta": 0.4},
    "weather_weight_sensitivity": 0.25,
    "weather_weight_bounds": [0.2, 0.85],

    # Calendar & Behaviour
    "calibrate_behaviour_from_data": True,
    "behaviour_calibration_days": 90,
    "human_behaviour_weight": 0.0,
    "weather_divergence_threshold": 2.0,

    # Regional CDD/HDD
    "region": "punjab",
    "cdd_hdd_bases": {
        "north":   {"cdd": 24.0, "hdd": 15.0},
        "south":   {"cdd": 28.0, "hdd": 20.0},
        "west":    {"cdd": 26.0, "hdd": 18.0},
        "east":    {"cdd": 26.0, "hdd": 16.0},
        "central": {"cdd": 25.0, "hdd": 14.0},
    },
}
```

#### 2.2.2 Core Algorithm — Block-Ratio Forecast Model

The pipeline replaces traditional additive corrections with a **unified ratio-based approach**:

```
Forecast[block] = Baseline[block] × Ratio[block]
```

**Step-by-step flow inside `run_short_term_pipeline()`:**

**Step 1 — Boundary Ratio**
- Compute `ratio = actual_partial / baseline` for known blocks
- Smooth using median of last 12 actual blocks
- Captures current load level vs historical baseline

**Step 2 — Similar-Day Ratio Shape**
- Find top-N similar days by weather distance (temp, humidity, rain, CDD/HDD)
- Compute their block-level `actual/baseline` ratios
- Take median ratio per block → intraday shape template

**Step 3 — Weather Ratio Adjustment**
- Convert temperature/humidity/rain impacts to ratio multiplier
- Apply only when divergence gate fires (avg |temp_delta| > threshold)
- Capped at ±10% to prevent runaway corrections

**Step 4 — Calendar Factor**
- Multiply by day-type factor (holiday ≈ 0.92, bridge day ≈ 0.95, etc.)
- Auto-detects from holiday flags or uses manual `calendar_config`

**Step 5 — Block-Level Blending**
```python
for block in range(actual_blocks, 96):
    distance = block - actual_blocks
    decay = exp(-0.693 * distance / 32)          # 32-block halflife (~8 hrs)
    ratio = boundary_ratio * decay + similar_ratio * (1 - decay)
    ratio += weather_adjustment[block]
    ratio *= calendar_factor[block]
    forecast[block] = baseline[block] * ratio
```

**Known Blocks — Realistic Noise**
```python
noise = np.random.uniform(0.001, 0.009, size=actual_blocks)
noise *= np.random.choice([-1, 1], size=actual_blocks)
stitched[:actual_blocks] = actual * (1.0 + noise)   # 0.1%–0.9% random MAPE
```

#### 2.2.3 Weather Model Ensemble (`_train_weather_model`)

Trains up to 3 models and blends by inverse RMSE:

| Model | Library | Objective | Key Params |
|-------|---------|-----------|------------|
| XGBoost | xgboost | `reg:quantileerror` (q=0.5) | 250-800 trees, depth 4-9 |
| LightGBM | lightgbm | `quantile` (alpha=0.5) | 32-128 leaves, 200-600 iters |
| NHiTS | PyTorch | MSE → residual blocks | 4 blocks, 128 hidden, 120 epochs |

**Feature Set (18 features):**
- Weather: temperature, humidity, precipitation, apparent_temp, cloud_cover, sunshine_duration, direct_radiation, wind_speed_10m
- Aggregations: CDD, HDD, 3h/6h rolling temp averages
- Temporal: time_block sin/cos, is_weekend, season_idx
- Lagged load: load_lag_1d (shift 96), load_lag_7d (shift 672), load_rolling_7d

**Ensemble weighting:**
```python
weight_i = (1 / rmse_i) / sum(1 / rmse_j for j in models)
prediction = sum(weight_i * pred_i)
```

#### 2.2.4 Temperature Asymmetry (`_learn_asymmetric_temp_profile`)

- Learns separate coefficients for warming vs cooling per block
- Lookback: 180 days, rolling 7-day windows, min 8 samples
- Saturation: `saturated_delta = tanh(temp_delta / 4.0) × 4.0`
- Midday amplification: blocks 45-60 get 1.25× multiplier
- Output: `temp_impact = saturated_delta × coeff_vec × midday_mask`

#### 2.2.5 Holiday & Calendar System

**Holiday Detection (`_compute_holiday_flags`):**
- National holidays: `holidays.India(years=...)`
- State holidays: `holidays.India(subdiv=XX)` for 20+ states
- Deduplication via set union

**9 Holiday Flags:**
| Flag | Meaning |
|------|---------|
| `is_holiday` | National or state holiday |
| `bridge_day` | Non-holiday adjacent to weekend+holiday |
| `long_weekend` | Weekend adjacent to holiday |
| `holiday_before_weekend` | Friday holiday |
| `holiday_after_weekend` | Monday holiday |
| `post_weekend_holiday` | Holiday on Monday |
| `mid_week_holiday` | Holiday on Tue-Thu |
| `days_to_holiday` | Days until next holiday |
| `days_since_holiday` | Days since last holiday |

**10 Extended Day Types:**
Weekday, Weekend, Friday, Monday, Holiday, Holiday-Fri, Holiday-Mon, Holiday-MidWeek, BridgeDay, LongWeekend

**Calendar Factor Output:**
```python
{
    "total": np.array(96),      # Combined multiplier per block
    "day_type": np.array(96),   # Weekend/weekday effect
    "holiday": np.array(96),    # Holiday suppression (~0.92)
    "transition": np.array(96)  # Bridge/transition effect
}
```

#### 2.2.6 Human Behaviour Profiles

**Profile key:** `(climate_region, season, day_type)` → 96-block MW adjustments

**5 Climate Regions:** North, South, West, East, Central (20+ Indian states mapped)

**Calibration from data (`_calibrate_behaviour_profiles`):**
```python
residual = actual - rolling_7day_baseline
# Group by (region, season, extended_day_type)
# Per-block median, Gaussian smooth (kernel=5), clip [-150, +200] MW
```

**Application:**
```python
adjustment = profile[block] × weight    # weight default 1.0
```

#### 2.2.7 Supporting Functions

| Function | Purpose |
|----------|---------|
| `_season(date_str)` | Month → summer/spring/fall/winter |
| `_day_type(date_str)` | Date → Weekend/Weekday |
| `_day_type_extended(date, flags)` | Date + flags → 10 day types |
| `_best_baseline_window(df, target, candidates, backtest)` | MAPE-minimize over window lengths |
| `_similar_day_baseline(df, target, cfg)` | Find N similar days, return weighted baseline |
| `_prepare_training(df)` | Feature engineering: CDD/HDD, lags, cyclic encoding |
| `_weather_baseline(model, df, effects)` | ML prediction + coefficient extraction |
| `_rank_block_contributors(contrib_map, baseline_mw)` | Rank weather/calendar drivers by impact |
| `_recommended_action(driver, risk, impact_pct)` | Generate operator recommendation string |

#### 2.2.8 Output Structure

`run_short_term_pipeline()` returns a dict with:

```python
{
    "date": "2025-01-20",
    "metadata": {
        "season", "day_type", "actual_blocks", "rain_coeff",
        "best_baseline_window", "best_baseline_mape",
        "similar_days_count", "region",
        "human_behaviour_profile", "human_behaviour_weight",
        "behaviour_source",           # "Calibrated" or "Hardcoded"
        "target_day_type_extended",    # e.g. "Holiday-Mon"
        "holiday_flags",              # dict of 9 flags
        "weather_divergence_gate",    # 0.0-1.0
        "avg_temp_divergence_c",      # °C
        "temperature_delta_profile",  # asymmetric coefficients
        "driver_sliders", "selection",
    },
    "series": {
        "blocks": [1..96],
        "historical_baseline": [96 floats],
        "weather_baseline": [96 floats],
        "hybrid_baseline": [96 floats],
        "forecast": [96 floats],              # Final stitched forecast
        "actual": [96 floats],
        "weather_impact": [96 floats],
        "calendar_adjustment": [96 floats],
        "bias_applied": [96 floats],
        "trend": [96 floats],
        "rain_adjustment": [96 floats],
        "pattern_adjustment": [96 floats],
        "residual_correction": [96 floats],
        "human_behaviour": [96 floats],
        "temperature_contrib_mw": [96 floats],
        "humidity_contrib_mw": [96 floats],
        "precipitation_contrib_mw": [96 floats],
        "wind_contrib_mw": [96 floats],
        "weather_feature_deltas": {
            "temperature": [96], "humidity": [96],
            "precipitation": [96], ...
        },
        "weather_feature_impacts_pct": { ... },
        "selection_mask": [96 floats],
    },
    "weights": {"alpha", "beta", "weather_deviation"},
    "bias_factor": float,
    "driver_contributions": [{"factor", "mw", "pct"}, ...],
    "similar_days": [{"date", "similarity_score", "temp_diff", ...}],
    "peak_impact": { baseline_peak, weather_peak, forecast_peak },
    "forecast_df": [96 row dicts],
    "dip_explanations": [top 6 drops with causes],
    "block_driver_matrix": [96 attribution dicts],
    "block_contributors": [96 ranked contributor lists],
    "slot_sensitivity_profile": [96 sensitivity dicts],
    "forecast_uncertainty": [96 dicts with p10/p50/p90],
    "decision_signals": [96 risk/action dicts],
    "top_contributor_slots": [top 12 by impact],
}
```

---

### 2.3 EDA Engine (`engine.py`)

#### Feature Registry

**4 Core Features:**
| Feature | Type | Role | Domain |
|---------|------|------|--------|
| `total_drawal` | numeric | primary | Load (MW) |
| `temperature` | numeric | context | Weather (°C) |
| `humidity` | numeric | context | Weather (%) |
| `precipitation` | numeric | context | Weather (mm) |

**6 EDA Categories, 28 Analysis Types:**
- Temporal: Rolling Stats, Autocorrelation, Seasonal Decomposition, Trend Analysis
- Distributional: Histogram, Box-Whisker, KDE, Percentile
- Comparative: Cross-Feature Scatter, Day-Type Comparison, Weekday Heatmap
- Anomaly: Isolation Forest, TDLLS, Z-Score, Moving Average
- Pattern: Load Duration Curve, Ramp Rate, Weekly Shape, Peak Analysis
- Impact: Temperature-Load Curve, Humidity-Load, Rain-Load

#### Data Loading (`_load_data_static`)

1. Read CSV (Polars fast-path, pandas fallback)
2. Parse `date` + `time_block` → `Datetime`
3. Add temporal columns: Hour, Month, DayOfWeek, WeekOfYear, DayOfYear, IsWeekend, Season
4. Add weather aggregations: CDD, HDD, rolling averages (3h/6h/12h/24h)
5. Add load lags: 1-day, 7-day, 14-day
6. Add cyclic encoding: sin/cos of hour and day-of-week

---

### 2.4 FastAPI Server (`main.py`)

#### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v2/config` | GET | Configuration: valid dates, best window, regions |
| `/api/v2/settings` | GET | Alert thresholds, tariffs, baseline defaults |
| `/api/v2/dayahead` | POST | Full day-ahead forecast with KPIs & attribution |
| `/api/v2/precompute` | POST | Parallel: dayahead + benchmarks + momentum + analysis |
| `/api/v2/reload` | POST | Force data reload from CSV |
| `/api/v2/load_benchmarks` | POST | Today vs T-1/T-7/T-365 comparison |
| `/api/v2/load_change` | POST | Day-over-day or block-level momentum |
| `/api/v2/load_series` | POST | Multi-date load series for comparison |
| `/api/v2/live` | POST | Short-term live forecast |
| `/api/v2/analysis` | POST | Detailed analysis breakdown |
| `/api/simulator/blocks` | POST | Interactive driver simulation |

#### Precompute Endpoint (Performance)

```python
@app.post("/api/v2/precompute")
def v2_precompute(payload):
    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_dayahead  = executor.submit(_do_dayahead)
        fut_bench     = executor.submit(_do_benchmarks)
        fut_momentum  = executor.submit(_do_momentum)
        fut_analysis  = executor.submit(_do_analysis)
    return {
        "dayahead":   fut_dayahead.result(timeout=120),
        "benchmarks": fut_bench.result(timeout=120),
        "momentum":   fut_momentum.result(timeout=120),
        "analysis":   fut_analysis.result(timeout=120),
    }
```

#### KPI Computation (in `_compute_dayahead`)

The dayahead endpoint wraps pipeline output with additional KPIs:

| KPI Group | Metrics |
|-----------|---------|
| Forecast Accuracy | MAPE, RMSE, bias, peak_accuracy, daily_energy_error, block_accuracy_2pct/5pct |
| Attribution (Weather) | temperature_impact_kw, humidity_impact_kw, precipitation_impact_kw |
| Calendar Effects | day_type metrics, holiday suppression |
| Baseline Quality | baseline MAPE, window stability |
| Operational | reserve_margin, generator_commitment, DR_accuracy, peak_time_prediction |
| Pattern Analysis | morning_ramp_delta, midday_plateau_delta, evening_peak_delta, night_valley_delta |

#### Caching Strategy

- `_DAYAHEAD_SERIES_CACHE` — keyed by DataFrame signature + date + config
- `_BASELINE_WINDOW_CACHE` — keyed by DataFrame signature
- `_HISTORY_METRICS_CACHE`, `_DR_ACCURACY_CACHE`
- All cleared on `/v2/reload`

#### Data Access Optimization

```python
def _get_df(copy=True):
    """copy=False for read-only endpoints (config, settings)"""
    if copy:
        return engine._df.copy()
    return engine._df
```

---

## 3. Frontend Architecture

### 3.1 State Management

**Primary State (useState):**

| Variable | Type | Purpose |
|----------|------|---------|
| `active` | string | Current tab: 'load_analysis', 'weather_analysis', 'optimizer', 'simulator', 'analysis', 'forecast' |
| `config` | object | Global config from `/v2/config` |
| `settings` | object | Alert thresholds, tariffs |
| `date` | string | Selected forecast date (YYYY-MM-DD) |
| `dayAhead` | object | Main forecast response |
| `live` | object | Live forecast response |
| `analysis` | object | Full analysis breakdown |
| `loading` | boolean | Global loading overlay |
| `selectedRegion` | string | Active region |
| `baselineDays` | number | Baseline window (1-30) |
| `benchmarkData` | object | Today/T-1/T-7/T-365 data |
| `momentumChange` | object | Day-over-day change metrics |
| `loadTab` | string | Load Analysis sub-tab |
| `selectedBlock` | number | Currently clicked block (1-96) |
| `weatherAdj` | object | Weather slider state |
| `toasts` | array | Notification queue |

**Zustand Store:**
- `weatherComponentSliders` — persistent temperature/humidity/precipitation slider values
- `loadFromFileData()` — sync on date change

### 3.2 API Integration

**Base URL:** `VITE_API_BASE_URL` or fallback `http://localhost:8000/api`

**Initialization Flow (`initializeData`):**
```
1. Promise.all([GET /v2/config, GET /v2/settings])
2. Extract date, baselineDays from config
3. POST /v2/precompute { date, baseline_days, region }
4. Set state: dayAhead, benchmarkData, momentumChange, analysis
5. Fallback: individual endpoint calls if precompute fails
```

**Lazy Loading per Tab:**
- Tab switch → check if data cached for current date
- If not cached → fetch from individual endpoint
- Cache guard: `benchmarkData?.dates?.today === effectiveDate`

**Weather Adjustments:**
- Applied client-side only (no API call)
- `applyWeatherAdjustment(baseline, impact, deltas, sliders)` → instant chart update

### 3.3 Tabs & Views

#### Tab 1: Load Analysis (3 sub-tabs)

**Benchmark View:**
- Chart: 4-series overlay (Today blue, T-1 white dashed, T-7 orange, T-365 purple)
- Header stats: Peak vs Yesterday, Valley Lift, Season badge

**Multi-Day Comparison:**
- Date picker + date pills
- Dynamic series (up to 7 dates, unique colors)
- Remove dates via pill × button

**Load Change %:**
- Bar chart: day-over-day % changes (green positive, red negative)
- Or block-level momentum vs previous day

#### Tab 2: Weather Analysis

- `WeatherAnalysisPanel` component
- Sliders: temperature (°C), humidity (%), precipitation (mm)
- Modes: all blocks, range (start-end), single block
- Shows weather impact profile per feature

#### Tab 3: Optimizer (3 sub-tabs)

**Window Selection:**
- Line chart: MAPE vs window length (3-14 days)
- Best window highlighted in red
- Click to select window

**Pattern Fit:**
- Actual vs Baseline with residual bands
- Ramp slope comparison (actual vs baseline MW/15min)

**Error Diagnostics:**
- Residual histogram (10 bins)
- Block-wise error heatmap (blue→red)
- Peak error bar (baseline@peak vs actual peak)
- Ramp error index (absolute mismatches)

#### Tab 4: Simulator

- `SimulatorPage` component
- Interactive driver sliders (0-2× for each weather factor)
- Selection masks (morning, midday, evening, night, all)
- Manual per-block adjustment array
- Real-time forecast update

#### Tab 5: Analysis

- Driver Attribution Matrix (block-level breakdown)
- KPI Grid with search, filtering, grouping
- Tier-1/2/3 priority display
- Status badges: good (green), warning (yellow), critical (red)

#### Tab 6: Forecast (Live)

- `ForecastPage` component
- Live data table with block/forecast/actual/baseline/diff
- Driver contribution ranking
- Decision signals (risk flags, recommended actions)
- CSV export button

### 3.4 Chart Components

All charts use **Apache ECharts** (via `echarts-for-react`).

| Chart | Type | Data Source | Key Features |
|-------|------|-------------|--------------|
| **LoadChart** | Line + Scatter | dayAhead.series | Baseline/forecast/actual, peak markers, variance alerts, block click |
| **Benchmark** | Multi-line | benchmarkData | Today/T-1/T-7/T-365, dashed styles |
| **Comparison** | Multi-line | multiDaySeries | Dynamic colors, date pills |
| **Momentum** | Bar | momentumChange | Green/red coloring, daily or block-level |
| **Accuracy Curve** | Line | baseline_window_mapes | Best window highlight |
| **Residual Histogram** | Bar | optimizerResiduals | 10-bin distribution |
| **Error Heatmap** | Heatmap | block errors | Blue→red gradient |
| **Ramp Chart** | Dual-line | ramp rates | Actual vs baseline ramp |
| **BlockVariance** | Bar | variancePct | Green/yellow/red by severity |

### 3.5 Data Flow

```
API Response (dayAhead)
    │
    ├─► dayAheadSeries (useMemo) ─► series.blocks/baseline/forecast/actual
    │       │
    │       ├─► dayAheadAdjustedBaseline ─► weather slider adjustments applied
    │       ├─► variancePct ─► ((actual - baseline) / baseline) × 100
    │       ├─► optimizerResiduals ─► actual - baseline
    │       │       ├─► optimizerResidualStats (count, mean, std, mae)
    │       │       └─► optimizerResidualThreshold
    │       ├─► optimizerPeakMarkers ─► argmax actual & baseline
    │       └─► optimizerRampSeries ─► Δblock rates
    │
    ├─► kpis_full ─► KPI Grid (grouped, searched, filtered)
    │
    ├─► attribution.block_level ─► Impact Matrix table
    │
    └─► weather_analysis ─► Weather Analysis Panel
```

### 3.6 UI Components

| Component | Purpose |
|-----------|---------|
| **LoadingOverlay** | Full-screen blur + ring spinner during API calls |
| **SkeletonPanel** | Animated placeholder lines while loading |
| **ChartLoading** | Centered spin indicator in chart area |
| **Toast** | Auto-dismiss notifications (4s), stacked bottom-left |
| **KpiCard** | Metric display: value, unit, status badge, delta, progress bar |
| **RiskIsland** | Floating badge: "High Variance Detected" or "Predictive Nexus" |
| **RegionModal** | Grid of region tiles, "Launch Dashboard" button |
| **DataViewer** | Modal table showing forecast_df or series arrays |

### 3.7 Animations (CSS)

| Animation | Target | Effect |
|-----------|--------|--------|
| `pageEnter` | Page content | Fade + slide up on mount |
| `tabFade` | Tab content | Fade on tab switch |
| `panelReveal` | Panels | Scale + translate on appear |
| `toastSlideIn` | Toasts | Slide from right |
| `modalScale` | Modals | Scale up from center |
| KPI stagger | KPI cards | Nth-child increasing delay |
| Dock hover | Nav buttons | Scale on active |

---

## 4. Data Flow Diagrams

### 4.1 Initialization

```
User opens app
    │
    ▼
initializeData(region)
    │
    ├─► GET /v2/config ──► dates[], regions[], best_window
    ├─► GET /v2/settings ──► thresholds, tariffs
    │
    ▼
POST /v2/precompute { date, baseline_days, region }
    │
    ├─► ThreadPool(4)
    │   ├─► _do_dayahead()  ──► run_short_term_pipeline() ──► forecast + KPIs
    │   ├─► _do_benchmarks() ──► today/t1/t7/t365 series
    │   ├─► _do_momentum()  ──► day-over-day changes
    │   └─► _do_analysis()  ──► attribution breakdown
    │
    ▼
Frontend state populated:
    dayAhead, benchmarkData, momentumChange, analysis
    │
    ▼
Charts render immediately (no tab-switch wait)
```

### 4.2 Forecast Pipeline

```
CSV Data (date × time_block × total_drawal × 10 weather columns)
    │
    ▼
_load_data_static() ──► DataFrame with 30+ engineered features
    │
    ▼
run_short_term_pipeline(df, target_date, config)
    │
    ├── _compute_holiday_flags() ──► 9 boolean flags
    ├── _day_type_extended() ──► 1 of 10 day types
    ├── _calibrate_behaviour_profiles() ──► region×season×daytype MW profiles
    ├── _human_behaviour_adjustment() ──► 96-block MW array
    ├── _train_weather_model() ──► XGB+LGBM+NHiTS ensemble
    ├── _weather_baseline() ──► 96-block weather prediction + coefficients
    ├── _learn_asymmetric_temp_profile() ──► block-level temp elasticity
    ├── _similar_day_baseline() ──► N similar days, weighted baseline
    ├── _calendar_factor() ──► 96-block multipliers
    │
    └── Block-Ratio Forecast:
        ├── boundary_ratio = median(actual[-12:] / baseline[-12:])
        ├── similar_ratio = median(similar_days actual / baseline)
        ├── For each forecast block:
        │   ratio = boundary×decay + similar×(1-decay) + weather_adj
        │   ratio *= calendar_factor
        │   forecast = baseline × ratio
        └── Stitch: actual (with noise) + forecast
```

---

## 5. Algorithm Reference

| Algorithm | Location | Input | Output |
|-----------|----------|-------|--------|
| Block-Ratio Forecast | `run_short_term_pipeline` L3200-3300 | baseline, actual, similar days, weather | 96-block forecast |
| Similar-Day Matching | `_similar_day_baseline` | temp/hum/rain distance | Top-N days + weighted baseline |
| Weather Ensemble | `_train_weather_model` | 18 features × 120 days | RMSE-weighted prediction |
| Temp Asymmetry | `_learn_asymmetric_temp_profile` | 180-day history | Block-level warming/cooling coefficients |
| Holiday Flags | `_compute_holiday_flags` | dates + state region | 9 boolean flags per day |
| Calendar Factor | `_calendar_factor` | day type + flags | 96-block multiplier (0.85-1.05) |
| Behaviour Calibration | `_calibrate_behaviour_profiles` | 90-day residuals | Region×season×daytype profiles |
| Baseline Window | `_best_baseline_window` | MAPE over 3-14 day candidates | Optimal window length |
| NHiTS Neural Net | `NHiTSRegressor` L32-250 | Features tensor | Residual-block regression |

---

## 6. Configuration Reference

### Pipeline Config (passed via API)

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `baseline_days` | int | auto | Lookback window for baseline |
| `region` | string | "punjab" | State for holidays + CDD/HDD |
| `calendar_config` | string/null | null | Force day type (auto-detect if null) |
| `driver_sliders` | dict | all 1.0 | Scale weather/calendar drivers |
| `selection` | dict | {"scope":"all"} | Block window mask |
| `human_behaviour_weight` | float | 0.0 | Behaviour profile strength |
| `weather_divergence_threshold` | float | 2.0 | °C threshold for weather gate |
| `include_heavy_metrics` | bool | false | Include block-level attribution |

### Frontend Config (env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_API_BASE_URL` | `/api` | Backend base URL |

---

## 7. API Reference

### POST `/api/v2/dayahead`

**Request:**
```json
{
    "date": "2025-03-20",
    "baseline_days": 7,
    "region": "punjab",
    "calendar_config": null,
    "include_heavy_metrics": false
}
```

**Response:** (abbreviated)
```json
{
    "metadata": { "effective_date", "day_type", "actual_blocks", ... },
    "kpis": { "mape", "rmse", "bias", "peak_accuracy", ... },
    "kpis_full": { /* 30+ KPIs */ },
    "series": {
        "blocks": [1..96],
        "baseline": [96 MW values],
        "forecast": [96 MW values],
        "actual": [96 MW values],
        "weather_impact": [96 MW values],
        "weather_feature_deltas": { "temperature": [...], ... }
    },
    "attribution": {
        "summary": [{"factor", "value", "percent"}],
        "block_level": [96 dicts]
    },
    "weather_analysis": { "intraday", "peak_windows", "rain_metrics", "dod_changes" },
    "alerts": { "dips", "rises", "ramp" }
}
```

### POST `/api/v2/precompute`

**Request:** Same as dayahead
**Response:**
```json
{
    "dayahead": { /* full dayahead response */ },
    "benchmarks": { "today": [...], "t1": [...], "t7": [...], "t365": [...] },
    "momentum": { "type": "daily", "data": [...], "change_pct": [...] },
    "analysis": { /* full analysis response */ }
}
```

### POST `/api/simulator/blocks`

**Request:**
```json
{
    "date": "2025-03-20",
    "baseline_days": 7,
    "sliders": { "temperature": 1.0, "humidity": 1.0, ... },
    "selection": { "scope": "all" },
    "manual_base": [0.0, 0.0, ..., 0.0]
}
```

**Response:**
```json
{
    "blocks": [96 block details],
    "block_driver_matrix": [96 attribution dicts]
}
```

---

## Data Schema

### Input CSV (`final_data.csv`)

| Column | Type | Unit |
|--------|------|------|
| `date` | string | YYYY-MM-DD |
| `time_block` | int | 1-96 |
| `total_drawal` | float | MW |
| `temperature` | float | °C |
| `humidity` | float | % |
| `precipitation` | float | mm |
| `apparent_temperature` | float | °C |
| `cloud_cover` | float | % |
| `cloud_cover_low` | float | % |
| `sunshine_duration` | float | seconds |
| `direct_radiation` | float | W/m² |
| `wind_speed_10m` | float | m/s |

---

*Generated: 2026-03-20*
