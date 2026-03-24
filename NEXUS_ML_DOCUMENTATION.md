# NEXUS — Machine Learning & Forecasting Documentation

> **System:** NEXUS Load Forecasting Engine
> **Resolution:** 96 blocks × 15 min = 24-hour horizon
> **Architecture:** Hybrid Additive Model (fully explainable)

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Models](#2-models)
   - [XGBoost (Quantile)](#a-xgboost-quantile-regression)
   - [LightGBM (Quantile)](#b-lightgbm-quantile-regression)
   - [NHiTS Neural Network](#c-nhits-neural-network)
   - [Weather Ensemble](#d-weather-ensemble-combination)
3. [Feature Engineering](#3-feature-engineering)
4. [Similar-Day Baseline](#4-similar-day-baseline)
5. [Weather Impact Modeling](#5-weather-impact-modeling)
6. [Calendar & Human Behaviour](#6-calendar--human-behaviour)
7. [Residual Correction & Trend](#7-residual-correction--trend)
8. [Rain Impact](#8-rain-impact)
9. [Final Forecast Assembly](#9-final-forecast-assembly)
10. [Uncertainty Quantification](#10-uncertainty-quantification)
11. [Explainability Layer](#11-explainability-layer)
12. [Supporting Analysis Models](#12-supporting-analysis-models)
13. [Configuration Reference](#13-configuration-reference)
14. [File Map](#14-file-map)

---

## 1. Pipeline Overview

```
INPUT: Historical data (180+ days) + Target date + Region + Actual blocks
│
├─ 1. SIMILAR-DAY BASELINE
│     Top 5 weather-matched historical days → 96-block MW vector
│
├─ 2. ML WEATHER MODEL (XGB + LGBM + NHiTS ensemble)
│     Trained on 180 days, 28 features → 96-block prediction
│
├─ 3. ASYMMETRIC WEATHER DELTA
│     Per-block temperature elasticity (up vs down) + saturation
│     Gated by divergence threshold (suppressed if baseline already captures weather)
│
├─ 4. CALENDAR + HUMAN BEHAVIOUR
│     Holiday factors, day-type transitions, region×season MW profiles
│
├─ 5. BOUNDARY CORRECTION
│     Anchors forecast to last observed actuals with exponential decay
│
├─ 6. TREND + SIMILAR-DAY SHAPE
│     Linear trend projection + intraday residual pattern from similar days
│
├─ 7. RAIN IMPACT (log-saturated)
│
└─ FINAL FORECAST (additive, per-block):
    forecast[b] = baseline[b]
               + weather_residual[b]
               + calendar_offset[b]
               + trend_correction[b]
               + boundary_correction[b]
               + sim_shape[b]
               + human_behaviour[b]
               - rain_impact[b]
    → clamp ≥ 0, stitch with actuals
```

**Key Principle:** Every component is additive (in MW) — no multiplicative compounding. This makes every MW of deviation fully traceable to a named driver.

---

## 2. Models

### A. XGBoost (Quantile Regression)

| Property | Value |
|----------|-------|
| **File** | `Backend/short_term_pipeline.py` → `_tune_xgb()` |
| **Purpose** | Predict load from weather + temporal features |
| **Objective** | `reg:quantileerror` (P10, P50, P90) |
| **Training data** | Last 180 days of 15-min load + weather |

**Hyperparameter search (random, 20 iterations):**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 250 – 800 |
| `max_depth` | 4 – 9 |
| `learning_rate` | 0.02 – 0.12 |
| `subsample` | 0.7 – 0.95 |
| `colsample_bytree` | 0.6 – 0.95 |
| `min_child_weight` | 1.0 – 6.0 |
| `reg_alpha` (L1) | 0.0 – 0.5 |
| `reg_lambda` (L2) | 0.8 – 2.0 |

**Validation:** Time-based split (last 7 days). Scored by RMSE.

---

### B. LightGBM (Quantile Regression)

| Property | Value |
|----------|-------|
| **File** | `Backend/short_term_pipeline.py` → `_tune_lgbm()` |
| **Purpose** | Same as XGB — complementary tree learner |
| **Objective** | `quantile` with `alpha` = 0.1 / 0.5 / 0.9 |

**Hyperparameter search (random, 12 iterations):**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 250 – 800 |
| `learning_rate` | 0.02 – 0.12 |
| `num_leaves` | 32 – 128 |
| `max_depth` | -1 to 9 |
| `subsample` | 0.7 – 0.95 |
| `colsample_bytree` | 0.6 – 0.95 |
| `min_child_samples` | 10 – 40 |
| `reg_alpha` | 0.0 – 0.5 |
| `reg_lambda` | 0.8 – 2.0 |

---

### C. NHiTS Neural Network

| Property | Value |
|----------|-------|
| **File** | `Backend/short_term_pipeline.py` → `NHiTSRegressor` (line 40) |
| **Purpose** | Capture non-linear temporal patterns trees miss |
| **Framework** | PyTorch (falls back to LinearRegression if unavailable) |

**Architecture:**

```
Input → [Block₁ → Block₂ → Block₃ → Block₄] → Output
         ↓         ↓         ↓         ↓
     128-dim    128-dim   128-dim   128-dim
      2 layers   2 layers  2 layers  2 layers
      dropout    dropout   dropout   dropout
      0.10       0.10      0.10      0.10

Each block: Linear → ReLU → Dropout → Linear → residual connection
```

| Hyperparameter | Value |
|----------------|-------|
| `hidden_dim` | 128 |
| `n_layers` | 2 per block |
| `n_blocks` | 4 |
| `dropout` | 0.10 |
| `epochs` | 120 |
| `batch_size` | 512 |
| `learning_rate` | 3e-3 |
| `weight_decay` | 1e-4 |
| `patience` (early stop) | 18 |
| `min_epochs` | 25 |
| Loss | MSELoss |
| Optimizer | AdamW |

**Scaling:** StandardScaler on X; y normalized by (y - mean) / std.
**Fallback:** If < 64 training samples or no PyTorch → `LinearRegression`.

---

### D. Weather Ensemble (Combination)

| Property | Value |
|----------|-------|
| **File** | `Backend/short_term_pipeline.py` → `WeatherEnsemble` |
| **Method** | Inverse-RMSE weighted average |

```
weight_i = (1 / RMSE_i) / Σ(1 / RMSE_j)
prediction = Σ(weight_i × model_i.predict(X))
```

Better-performing models on the validation set get more weight. This naturally selects the best model per dataset without manual tuning.

---

## 3. Feature Engineering

**Total: 28 features** prepared in `_prepare_training()` (v3.0, was 20)

### Weather (raw)
| Feature | Source |
|---------|--------|
| `temperature` | °C |
| `humidity` | % |
| `precipitation` | mm |
| `apparent_temperature` | °C |
| `cloud_cover` | % |
| `sunshine_duration` | hours |
| `direct_radiation` | W/m² |
| `wind_speed_10m` | m/s |

### Engineered Weather
| Feature | Formula |
|---------|---------|
| `CDD` | max(0, temp − base) — base is regional (24°C north, 28°C south) |
| `HDD` | max(0, base − temp) — base is regional (15°C north, 20°C south) |
| `temp_roll_3h` | 12-block rolling mean |
| `temp_roll_6h` | 24-block rolling mean |
| `WBGT` | Wet-Bulb Globe Temperature (Stull 2011 approximation) |

### Load Lags
| Feature | Description |
|---------|-------------|
| `load_lag_1d` | Same block yesterday (shift 96) |
| `load_lag_2d` | Same block 2 days ago (shift 192) — **v3.0** |
| `load_lag_3d` | Same block 3 days ago (shift 288) — **v3.0** |
| `load_lag_7d` | Same block last week (shift 672) |
| `load_rolling_7d` | 7-day rolling mean at same time block |
| `load_trend_3d` | 3-day load momentum (today avg − 3d ago avg) — **v3.0** |

### Multiplicative Weather × Load Interactions (v3.0)
| Feature | Formula | Why |
|---------|---------|-----|
| `load_x_cdd` | load_lag_1d × CDD | AC impact scales with base load level |
| `load_x_humidity` | load_lag_1d × humidity / 100 | Humid-heat compound effect |
| `cdd_squared` | CDD² | Exponential AC response above comfort threshold |
| `wbgt_x_load` | WBGT × load_rolling_7d | Heat stress × load level interaction |
| `temp_momentum_3d` | temp[today] − temp[3d ago] | Multi-day heat accumulation (thermal mass) |

### Temporal
| Feature | Encoding |
|---------|----------|
| `tb_sin` | sin(2π(block-1)/96) |
| `tb_cos` | cos(2π(block-1)/96) |
| `season_idx` | 0=winter, 1=spring, 2=summer, 3=fall |
| `is_weekend` | Binary |

---

## 4. Similar-Day Baseline

**Function:** `_similar_day_baseline()`
**Output:** 96-block MW vector (weighted average of top 5 similar days)

### Similarity Score

```
score = w_temp × norm(ΔTemp)
      + w_dd  × norm(ΔDegreeDay)
      + w_hum × norm(ΔHumidity)
      + w_rain × penalty(rain_mismatch)
```

| Weight | Value | What it captures |
|--------|-------|------------------|
| Temperature | 0.385 | Dominant cooling/heating driver |
| Degree Days | 0.165 | Non-linear comfort zone departure |
| Humidity | 0.250 | AC load amplifier |
| Rain pattern | 0.200 | Suppresses outdoor activity load |

### Constraints
- Same **season** + same **day type** (weekday/weekend)
- Temperature within band: ±2.5°C (winter) to ±3.5°C (summer)
- Rain pattern match (auto-detected)
- Excludes abnormal days (< 100 MW average)
- **Candidate lookback: 45 days** (v3.0, was 15)
- **Progressive fallback** (v3.0): If < 5 candidates, relax temp band by +1°C; if < 3, relax by +2°C

### Load Growth Normalization
Historical days are scaled to the recent 7-day load level:
```
growth_ratio = clamp(recent_7d_avg / similar_day_avg, 0.85, 1.15)
scaled_baseline = raw_baseline × growth_ratio
```

### Optimal Window Selection
`_best_baseline_window()` tests windows of [3, 5, 7, 10, 14] days, backtested on the last 10 days, scored by MAPE + weather divergence penalty.

---

## 5. Weather Impact Modeling

### Asymmetric Temperature Elasticity

**Function:** `_learn_asymmetric_temp_profile()`
**Training:** 180 days of same season + day type

For each of the 96 blocks, two separate coefficients are learned:

| Coefficient | Meaning |
|-------------|---------|
| `block_increase_coeff[b]` | MW/°C when temperature **rises** above rolling baseline |
| `block_reduction_coeff[b]` | MW/°C when temperature **falls** below rolling baseline |

**Why asymmetric?** A 3°C rise in summer adds AC load exponentially, but a 3°C drop doesn't remove the same amount — cooling inertia, building thermal mass, and occupant behavior differ.

**Saturation:** `impact = tanh(ΔTemp / 6) × 6 × coeff` — caps effect beyond ±6°C (v3.0, was ±4°C — Indian summers routinely exceed 4°C delta)
**Midday amplification:** Blocks 45–60 (12:00–15:00) × 1.25 — peak cooling demand

### Multiplicative Weather Delta (v3.0)

Weather impact is now **multiplicative** — scaled by baseline load level:
```python
reference_load = median(baseline[baseline > 100])
pct_coeff = temp_coeff / reference_load       # MW/°C → %/°C
impact[b] = baseline[b] × pct_coeff × saturated_delta[b]
```

A 3°C rise adds more MW when base load is 5000 MW vs 2000 MW.

### Per-Period Weather Divergence Gate (v3.0)

Gate computed per period (was single daily average):

```
Periods: Night(0-24) | Morning(24-48) | Afternoon(48-72) | Evening(72-96)

For each period:
  period_div = mean(|target_temp[period] − baseline_temp[period]|)
  gate[period] = clip((period_div − 1.5) / 1.875, 0, 1)

weather_impact *= gate   # per-block gating
```

**Key improvement:** Afternoon blocks with +3°C get the delta even if the daily mean gap is only 1.5°C.

**Result:** Weather delta exposure is precision-gated per time period.

---

## 6. Calendar & Human Behaviour

### Holiday System

Uses `holidays.India()` (national) + `holidays.India(subdiv=XX)` (state-specific), merged and deduplicated.

**Supported states:** 22 Indian states mapped via `INDIAN_STATE_HOLIDAY_SUBDIV`.

### Day-Type Classification

| Type | Detection |
|------|-----------|
| Weekday | Mon–Fri, not holiday |
| Friday | Fri, not holiday |
| Monday | Mon, not holiday |
| Weekend | Sat–Sun |
| Holiday | In holiday set |
| Holiday-Fri | Holiday on Friday |
| Holiday-Mon | Holiday on Monday |
| Holiday-MidWeek | Holiday on Tue/Wed/Thu |
| BridgeDay | Weekday adjacent to holiday |
| LongWeekend | Weekend adjacent to holiday |

### Calendar Factors

| Pattern | Factor |
|---------|--------|
| Holiday | 0.92 × weekday average |
| Bridge day | 50% holiday + 50% normal |
| Long weekend | weekend × 0.97 |
| Holiday before weekend | Extra evening dip (blocks 68–88, −5%) |
| Holiday after weekend | Slow morning ramp (blocks 20–40) |
| Pre-holiday (day before) | Evening ease −3% (blocks 72–96) |
| Post-holiday (day after) | Morning recovery ramp (blocks 20–44) |

### Human Behaviour Profiles

**Three-tier priority** (v3.0):
1. **Calibrated** — learned from historical residuals (preferred if 90+ days exist)
2. **State-specific** — hardcoded per state (e.g., Delhi, Maharashtra, Tamil Nadu)
3. **Regional** — indexed by (climate_region, season, day_type) — fallback

#### State-Specific Profiles (v3.0)

| State | Summer Peak MW | Key Driver |
|-------|---------------|------------|
| Delhi (DL) | +400–500 (blocks 37–72) | Dense AC, metro rail, urban heat island |
| Uttar Pradesh (UP) | +250–400 (blocks 40–68) | Agricultural pump 6–9 AM, evening lighting |
| Rajasthan (RJ) | +300–450 (blocks 38–70) | Desert cooling, solar ramp compensation |
| Maharashtra (MH) | +400–600 (blocks 38–70) | Mumbai industrial + Pune IT corridor |
| Tamil Nadu (TN) | +300–500 (blocks 40–72) | Chennai AC, Coimbatore/Tirupur textile |
| Karnataka (KA) | +250–400 (blocks 42–70) | Bengaluru IT sustained 9AM–9PM |
| Telangana (TS) | +280–420 (blocks 40–68) | Hyderabad pharma/IT 24h, high AC |
| West Bengal (WB) | +200–350 (blocks 42–70) | Kolkata urban heat island, jute |
| Odisha (OR) | +180–300 (blocks 44–68) | Steel/mining, cyclone events |
| Chhattisgarh (CG) | +180–300 + 100 base (24h) | Steel/aluminium smelter continuous |

#### Regional Fallback Example — North / Summer / Weekday:
```
Blocks  1–20:  −80 MW   (late-night cool-down, minimal AC)
Blocks 37–72: +200 MW   (AC + industrial peak)
Blocks 73–84: +150 MW   (evening cooking & lighting)
```

#### Calibration from Data (v3.0 upgrades)
- **Recency weighting:** 21-day half-life exponential decay (recent patterns dominate)
- **Proportional clip:** ±8% of state 95th-percentile peak load (was fixed ±150 MW)
- Smoothed with Gaussian kernel (width=5)
- Recalibrates on each run from last 90 days

---

## 7. Residual Correction & Trend

### Boundary Correction (Actual Anchoring)

Uses the last 12 observed blocks to anchor forecast:

```python
boundary_bias = median(recent_actuals − recent_baseline)

# Exponential decay into forecast horizon
for block in range(actual_blocks, 96):
    decay = exp(−0.693 × distance / 24)  # half-life = 24 blocks (6 hours)
    correction[block] = boundary_bias × decay
```

### Piecewise Trend Projection (v3.0)

```python
# Fit separate slopes for last 14 days vs last 60 days
slope_short = polyfit(days[-14:], residuals[-14:], deg=1)[0]
slope_long  = polyfit(days[-60:], residuals[-60:], deg=1)[0]

# Regime change: if short-term diverges >2× from long-term, trust it
if abs(slope_short) > 2 × abs(slope_long):
    slope = slope_short
else:
    slope = 0.7 × slope_short + 0.3 × slope_long

# Proportional clip: ±5% of recent average load (was fixed ±200 MW)
max_trend = 0.05 × recent_avg_load
trend_mw = clip(slope, −max_trend, +max_trend)
```

**Trend weight: 0.12** (v3.0, was 0.05) — captures multi-day regime changes

### Similar-Day Intraday Shape

```python
residual_stack = [actual[day] − baseline for day in similar_days]
shape = median(residual_stack, axis=0)

# Influence grows with distance from actuals
shape_weight = 1 − exp(−0.693 × distance / 24)
contribution[b] = shape[b] × shape_weight[b]
```

---

## 8. Rain Impact

Non-linear log-saturation prevents runaway at heavy rainfall:

```python
precip_saturated = log(1 + precip_mm) × (5 / log(6))
rain_impact = seasonal_coeff × precip_saturated
```

| Season | Coefficient (MW/mm) | Evening boost (blocks 70–96) |
|--------|---------------------|------------------------------|
| Winter | 20.0 | ×1.8 |
| Spring | 18.0 | ×1.8 |
| Summer | 15.0 | ×1.8 |
| Fall | 18.0 | ×1.8 |

---

## 9. Final Forecast Assembly

```python
for block in range(96):
    forecast[block] = baseline[block]
                    + weather_residual[block]
                    + calendar_offset[block]
                    + trend_correction[block]
                    + boundary_correction[block]
                    + sim_shape_contribution[block]
                    + human_behaviour_adj[block]
                    - rain_impact[block]

    forecast[block] = max(forecast[block], 0)
```

**Stitching with actuals:**
- Blocks 0 → actual_blocks: Exact observed values
- Next 4 blocks: Linear taper (actual → forecast)
- Remaining: Pure forecast

---

## 10. Uncertainty Quantification

Per-block standard deviation:

```python
σ = sqrt(similar_residual_std² + recent_residual_std²)
  + |pattern_adjustment| × 0.15
  + max(|net_contribution| × 0.08, 1.0)   # floor

z = 1.2816   # 80% confidence interval

P10 = max(forecast − z × σ, 0)
P50 = forecast
P90 = forecast + z × σ

confidence = clip(1 − (P90 − P10) / (100 × P50), 0.05, 0.99)
```

---

## 11. Explainability Layer

### Block Driver Matrix

For **every block**, the pipeline decomposes the forecast into named MW contributions:

| Driver | How computed |
|--------|-------------|
| Temperature | Asymmetric elasticity × ΔTemp |
| Humidity | Feature effect × ΔHumidity |
| Precipitation | Rain impact (log-saturated) |
| Cloud cover | Feature effect × ΔCloud |
| Wind | Feature effect × ΔWind |
| Radiation | Feature effect × ΔRadiation |
| Day type | Calendar factor offset |
| Holiday | Holiday factor offset |
| Human behaviour | Region×season×day-type profile |
| Trend | Linear projection |
| Boundary | Actual anchoring decay |
| Pattern | Similar-day shape |

### Decision Signals

Risk-ranked recommendations per block:

```json
{
  "block": 73,
  "time": "18:00",
  "risk_flag": "high",
  "primary_driver": "Temperature +4.2°C",
  "recommended_action": "Consider pre-positioning 200 MW spinning reserve"
}
```

---

## 12. Supporting Analysis Models

### EDAEngine (`engine.py`)

| Model | Purpose | Configuration |
|-------|---------|---------------|
| **Isolation Forest** | Anomaly detection | contamination=0.01, features=[load, hour, dow] |
| **K-Means** | Load archetype clustering | 4 clusters, normalized daily profiles |
| **Linear Regression** | Seasonal decomposition | Weather component on residuals |
| **Peak Detection** | `scipy.signal.find_peaks` | Prominence-based |
| **TDLLS** | Level-shift anomalies | Multi-week baselines, 300–500 MW thresholds |

### Correlation Method

All correlations across the system use **Spearman's Rank Correlation** (ρ):
- Robust to non-linear monotonic relationships (e.g., exponential AC load above 35°C)
- Resistant to outliers
- Implemented via `scipy.stats.spearmanr` (backend) and custom rank-based function (frontend)

---

## 13. Configuration Reference

```python
DEFAULT_CONFIG = {
    # Baseline & Similarity
    "candidate_lookback_days": 45,       # v3.0 (was 15)
    "similar_days_top_n": 5,
    "baseline_window_candidates": [3, 5, 7, 10, 14],
    "baseline_backtest_days": 10,

    # Weather Training
    "weather_training_days": 180,        # v3.0 (was 120)
    "weather_tune": True,
    "weather_tune_iters": 20,            # v3.0 (was 12)

    # Similarity Weights
    "similarity_weights": {"temp": 0.55, "humidity": 0.25, "rain": 0.20},
    "auto_similarity_from_data": True,
    "similarity_calibration_days": 120,

    # Temperature Elasticity
    "temp_asym_lookback_days": 180,
    "temp_asym_rolling_days": 7,
    "temp_asym_min_samples": 8,
    "weather_divergence_threshold": 1.5, # v3.0 (was 2.0, now per-period)

    # Component Weights
    "pattern_weight": 0.12,              # v3.0 (was 0.15)
    "trend_weight": 0.12,               # v3.0 (was 0.05)
    "human_behaviour_weight": 0.35,      # v3.0 (was 0.30)

    # Bounds
    "min_valid_load_mw": 100.0,
    "min_actual_blocks": 24,
    "max_actual_blocks": 60,

    # Regional CDD/HDD Bases
    "cdd_hdd_bases": {
        "north":   {"cdd": 24.0, "hdd": 15.0},
        "south":   {"cdd": 28.0, "hdd": 20.0},
        "west":    {"cdd": 26.0, "hdd": 18.0},
        "east":    {"cdd": 26.0, "hdd": 16.0},
        "central": {"cdd": 25.0, "hdd": 14.0},
    },

    # Rain Coefficients (MW per mm)
    "rain_coeffs": {
        "winter": 20.0, "spring": 18.0,
        "summer": 15.0, "fall": 18.0,
    },
}
```

---

## 14. File Map

| File | Lines | Role |
|------|-------|------|
| `Backend/short_term_pipeline.py` | ~3800 | All forecasting: models, features, baseline, weather, assembly |
| `Backend/engine.py` | ~2600 | EDAEngine: anomaly detection, clustering, decomposition |
| `Backend/gridintel_engine.py` | ~500 | GridIntelControlDesk: orchestration, KPI computation |
| `Backend/main.py` | ~4500 | FastAPI endpoints, API config, request routing |

### Key Functions in `short_term_pipeline.py`

| Function | What it does |
|----------|-------------|
| `run_short_term_pipeline()` | Main entry point — orchestrates entire forecast |
| `_prepare_training()` | 20-feature engineering from raw data |
| `_similar_day_baseline()` | Find top-5 similar days, build baseline |
| `_best_baseline_window()` | Cross-validate optimal lookback window |
| `_train_weather_model()` | Train XGB + LGBM + NHiTS ensemble |
| `_tune_xgb()` | XGBoost hyperparameter search |
| `_tune_lgbm()` | LightGBM hyperparameter search |
| `NHiTSRegressor` | PyTorch neural network class |
| `WeatherEnsemble` | RMSE-weighted model combiner |
| `_learn_asymmetric_temp_profile()` | Per-block up/down temperature elasticity |
| `_calendar_factor()` | Holiday/transition day adjustment factors |
| `_compute_holiday_flags()` | State-wise Indian holiday detection |
| `_human_behaviour_adjustment()` | Region×season×day-type MW profiles |
| `_calibrate_behaviour_profiles()` | Learn profiles from historical residuals |
| `_estimate_feature_effects()` | Extract MW/unit sensitivities from models |
| `_safe_corr()` | Spearman rank correlation (used everywhere) |

---

*Generated from codebase analysis — NEXUS Forecast Engine v3.0.0*

---

## 15. v3.0 Changelog

| Change | Old | New | Impact |
|--------|-----|-----|--------|
| Weather delta | Flat MW/°C | Multiplicative (% of baseline) | Biggest single accuracy gain |
| Divergence gate | Single daily average | Per-period (4 periods) | Preserves afternoon signals |
| Tanh saturation | ±4°C | ±6°C | Captures Indian summer extremes |
| Trend weight | 0.05 | 0.12 (piecewise) | Catches regime changes |
| Behaviour clip | Fixed ±150 MW | ±8% of state peak | Proportional to state size |
| Behaviour recency | Flat weighting | 21-day half-life decay | Recent patterns dominate |
| Candidate lookback | 15 days | 45 days + fallback | Wider pool, especially winter |
| Features | 20 | 28 | +5 interaction, +3 lag features |
| State profiles | 5 regions only | 22 states (15+ overrides) | State-level granularity |
| Training data | 120 days | 180 days | More data for state-level |
| Tune iterations | 12 | 20 | Better hyperparameter search |
