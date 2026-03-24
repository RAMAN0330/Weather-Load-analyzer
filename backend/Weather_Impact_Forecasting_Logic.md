# Backend Working: Load vs. Weather Impact & Forecasting (Upgraded V2)

This document details the advanced, block-aware technical implementation of weather impact calculations and the short-term forecasting pipeline within the GridIntel backend.

---

## 1. Weather Impact Methodology (Localized & Regression Based)

The system isolates weather-driven load components using a multi-stage regression approach, respecting regional and temporal variances.

### A. Base Load Calculation
*File: `backend/engine.py` -> `_compute_base_load`*

The system identifies "Fair Weather" days (typically 15°C to 22°C, low rainfall, moderate humidity) to isolate the weather-insensitive load component.
- **Median Baseline**: Computes a stable median load per (Region, Season, DayType, Time Block).

### B. Block-Level Sensitivity Modeling
*File: `backend/engine.py` -> `_compute_block_sensitivity`*

Instead of global multipliers, the system perform linear regression **per 15-minute block**.
- **Localized Coefficients**: Derive specific MW impact constants ($\beta$ for Temp, $\gamma$ for Humidity) that vary naturally throughout the day (e.g., higher cooling sensitivity in the afternoon).

### C. Nonlinear Modeling (HDD/CDD)
*File: `backend/engine.py` -> `run_temperature_load_curve`*

Load response to temperature follows a U-curve, modeled via:
- **Deadband Zone (15-22°C)**: Minimal weather impact.
- **Cooling Zone (> 22°C)**: `CDD = max(0, Temp - 22)`.
- **Heating Zone (< 15°C)**: `HDD = max(0, 15 - Temp)`.
- **Elasticity**: Piecewise linear slope calculations for each zone.

### D. Weather Sensitivity Index (WSI)
*File: `backend/main.py` -> `_compute_weather_index`*

Quantifies the percentage of load attributed to weather per block:
$$WSI_{block} = \frac{Actual - Base Load}{Base Load}$$

---

## 2. Enhanced Forecasting Pipeline

The forecasting engine utilizes a lag-aware, hybrid-blended approach.

### A. Lag & Thermal Inertia
*File: `backend/engine.py` -> `_compute_lag_features`*

To account for building thermal mass, the models include:
- **Instantaneous Weather**: Current block metrics.
- **Short Lag (3h)**: Rolling average of the last 12 blocks.
- **Long Lag (6h)**: Rolling average of the last 24 blocks.

### B. Similar Day Selection (V2)
*File: `backend/short_term_pipeline.py` -> `_similar_day_baseline`*

Selection logic now incorporates:
- **Degree Day Profile**: Matching CDD/HDD shapes.
- **Lag Profiles**: Ensuring thermal memory matches.
- **Growth Scaling**: Adjusting similar days by recent 7-day load levels.

### C. Hybrid Machine Learning Blend
*File: `backend/short_term_pipeline.py` -> `run_short_term_pipeline`*

The final forecast is a structural blend:
1. **ML Model (60%)**: XGBoost/LightGBM trained on localized features and interactions.
2. **Bias-Corrected Baseline (30%)**: Recent trend adjustments.
3. **Block Regression Model (10%)**: Direct physics-based weather baseline.

---

## 3. Key Backend Components

| File | Primary Responsibility |
| :--- | :--- |
| `main.py` | Localized WSI, block-aware contributions, and API summaries. |
| `engine.py` | Piecewise HDD/CDD analysis, base-load isolation, and block sensitivity. |
| `short_term_pipeline.py` | Feature engineering (Lags, CDD), ML training, and hybrid blending. |

> [!IMPORTANT]
> The system now avoids fixed global multipliers (e.g., the old 8 MW/°C) in favor of block-specific regression coefficients derived from historical data.
