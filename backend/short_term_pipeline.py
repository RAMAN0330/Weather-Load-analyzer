import hashlib
import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from .helper_model import run_helper_forecast
except ImportError:
    try:
        from helper_model import run_helper_forecast
    except ImportError:
        run_helper_forecast = None

try:
    from .weather_anomaly import detect_weather_anomalies, WeatherAnomalyResult
except ImportError:
    try:
        from weather_anomaly import detect_weather_anomalies, WeatherAnomalyResult
    except ImportError:
        detect_weather_anomalies = None
        WeatherAnomalyResult = None

try:
    from .accuracy import (
        apply_bias_correction,
        apply_block_type_calibration,
        apply_t2_night_correction,
        build_afternoon_ramp_adjustment,
        build_t2_bias_correction,
        compute_bias_table,
    )
except ImportError:
    try:
        from accuracy import (
            apply_bias_correction,
            apply_block_type_calibration,
            apply_t2_night_correction,
            build_afternoon_ramp_adjustment,
            build_t2_bias_correction,
            compute_bias_table,
        )
    except ImportError:
        apply_bias_correction = None
        apply_block_type_calibration = None
        apply_t2_night_correction = None
        build_afternoon_ramp_adjustment = None
        build_t2_bias_correction = None
        compute_bias_table = None



SEASON_ORDER = ["winter", "spring", "summer", "fall"]

SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

SEASONAL_WEATHER_INTERVALS = {
    "winter": {
        "temp_band": 2.5,
        "hum_band": 10.0,
        "cdd_base": 24.0, 
        "hdd_base": 18.0, 
    },
    "spring": {
        "temp_band": 3.0,
        "hum_band": 12.0,
        "cdd_base": 22.0,
        "hdd_base": 15.0,
    },
    "summer": {
        "temp_band": 3.5,
        "hum_band": 15.0,
        "cdd_base": 20.0, 
        "hdd_base": 12.0, 
    },
    "fall": {
        "temp_band": 3.0,
        "hum_band": 12.0,
        "cdd_base": 22.0,
        "hdd_base": 15.0,
    },
}

DEFAULT_CONFIG = {
    "candidate_lookback_days": 60,
    "similar_days_top_n": 10,
    # Same-month-last-year seasonal pool: +/- N days around the target's month-day,
    # taken from every prior year present in df.  0 disables the seasonal pool.
    "seasonal_window_days": 21,
    # Seasonal-anchor blend: a tighter (+/- 7 day, prior years) curve mixed
    # into the similar-day baseline at this weight.  0 disables.
    "seasonal_anchor_blend": 0.30,
    "seasonal_anchor_window_days": 7,
    # Recent-residual bias correction: subtract per-block (actual - baseline)
    # mean over the last N days from the baseline.  Catches systematic
    # under/over-prediction in the current load regime.  0 disables.
    "recent_bias_correction_days": 7,
    "recent_bias_correction_weight": 0.75,
    "bias_cap_pct": 0.10,           # max additive bias correction as fraction of mean load
    # Base-day -> T+1 seam smoother: linear taper of the first N blocks toward
    # a continuation of the previous day's last block, so the forecast doesn't
    # jump at the day boundary.  0 disables.
    "base_seam_bridge": {"window_blocks": 6},
    # Z-score scaling stats are computed on the candidate pool; min-max fallback
    # is used when std == 0.  Affects similarity scoring only.
    "similarity_normaliser": "zscore",
    "temp_band_c": 3.0,
    "humidity_band": 12.0,
    "require_rain_match": True,
    "wind_gust_threshold_kmh": 25.0,   # ↓ from 30 — earlier gust-dip activation
    "sldc_wind_threshold_kmh": 65.0,   # 80m wind above this triggers SLDC feeder-cut correction
    "similarity_weights": {"temp": 0.55, "humidity": 0.25, "rain": 0.2},
    "auto_similarity_from_data": True,
    "auto_require_rain_match": True,
    "similarity_calibration_days": 120,
    "similarity_weight_prior_blend": 0.35,
    "rain_match_min_days": 3,
    "rain_effect_threshold": 0.01,
    "baseline_window_candidates": [3, 5, 7, 10, 14],
    "baseline_backtest_days": 10,
    "weather_training_days": 180,
    "temp_asym_lookback_days": 180,
    "temp_asym_rolling_days": 7,
    "temp_asym_min_samples": 8,
    "hybrid_weights": {"alpha": 0.6, "beta": 0.4},
    "weather_weight_sensitivity": 0.25,
    "weather_weight_bounds": [0.2, 0.85],
    "baseline_weather_weight": 0.35,
    "trend_degree": 1,
    "trend_sign": 1.0,
    "pattern_weight": 0.12,
    "trend_weight": 0.12,
    "forecast_blend": {"hybrid": 0.20, "bias_applied": 0.80, "weather_baseline": 0.0},
    "forecast_shrinkage": {
        "offpeak_weight": 0.50,
        "peak_weight": 0.25,
        "known_blocks_bonus": 0.20,
        "peak_start_block": 48,
        "peak_end_block": 72,
    },
    "hybrid_ai": {
        "enabled": True,
        # variant:
        # - "distilled_residual": teacher-learned ResNet-LSTM residual corrector (no manual edits)
        # - "legacy": existing ResNet-LSTM curve+scale + MoE + GPR hybrid engine
        "variant": "distilled_residual",
        "lookback_days": 7,
        "min_history_days": 21,
        "training_days": 60,
        "blend_weight": 0.35,
        "min_blend_weight": 0.10,
        "max_blend_weight": 0.45,
        "sequence_epochs": 30,
        "random_state": 42,
        # Distilled residual engine (teacher learning; training-only teacher)
        "distilled_lookback_days": 14,
        "distilled_teacher_epochs": 8,
        "distilled_student_epochs": 35,
        "distilled_lr": 0.002,
    },
    "rain_coeffs": {"winter": 20.0, "spring": 18.0, "summer": 15.0, "fall": 18.0},
    "wind_coeffs": {"winter": 8.0,  "spring": 10.0, "summer": 12.0, "fall": 10.0},
    "min_actual_blocks": 24,
    "max_actual_blocks": 96,
    "weather_tune": True,
    "weather_tune_iters": 10,
    "min_valid_load_mw": 100.0, # Minimum average load to consider a day valid for baseline
    "driver_sliders": {
        "temperature": 1.0,
        "humidity": 1.0,
        "precipitation": 1.0,
        "apparent_temperature": 1.0,
        "cloud_cover": 1.0,
        "sunshine_duration": 1.0,
        "direct_radiation": 1.0,
        "wind_speed_10m": 1.0,
        "weather": 1.0,
        "daytype": 1.0,
        "holiday": 1.0,
        "manual": 1.0
    },
    "selection": {"scope": "all"},
    "selection_smoothing": False,
    "manual_base": [0.0] * 96,
    "region": "haryana",
    "human_behaviour_weight": 0.35,
    "weather_divergence_threshold": 1.5,
    "calibrate_behaviour_from_data": True,
    "behaviour_calibration_days": 90,
    "behaviour_cyclic_harmonics": 4,
    "behaviour_cyclic_blend": 0.55,
    "behaviour_cyclic_smoothing": 5,
    "cdd_hdd_bases": {
        "north":   {"cdd": 24.0, "hdd": 15.0},
        "south":   {"cdd": 28.0, "hdd": 20.0},
        "west":    {"cdd": 26.0, "hdd": 18.0},
        "east":    {"cdd": 26.0, "hdd": 16.0},
        "central": {"cdd": 25.0, "hdd": 14.0},
    },
    # ════════════════════════════════════════════════════════════════════════
    # OPTIMIZATION ENGINE CONFIG (97%+ Accuracy Mode)
    # ════════════════════════════════════════════════════════════════════════
    "optimization_enabled": True,
    "use_advanced_baseline": True,
    "use_horizon_specific_models": True,
    "use_bias_correction": False,
    "adaptive_blend_per_horizon": True,
    "advanced_baseline_lookback_days": 90,
    "advanced_baseline_validation_days": 7,
    "advanced_baseline_blend_method": "rmse_weighted",
    # Horizon-specific model hyperparameters
    "t1_model_n_estimators": 200,
    "t1_model_max_depth": 6,
    "t1_model_learning_rate": 0.05,
    "t2_model_n_estimators": 250,
    "t2_model_max_depth": 6,
    "t2_model_learning_rate": 0.05,
    # Adaptive blend weights (overrides static hybrid_weights when enabled)
    "blend_weights_t1": {"ml_model": 0.70, "baseline": 0.20, "block_regression": 0.10},
    "blend_weights_t2": {"ml_model": 0.60, "baseline": 0.25, "block_regression": 0.15},
    # Generic external feature hook. Enable via config or env vars:
    # - SHORT_TERM_EXTERNAL_FEATURES_ENABLED=1
    # - SHORT_TERM_EXTERNAL_FEATURES_PATH=...\\features.py
    "external_features_enabled": False,
    "external_features_path": None,
    # When enabled: include any *extra numeric* columns produced by the external module
    # (excluding known identifiers/targets) as model features.
    "external_features_include_extra_numeric": True,
    # External final forecast override (bypasses the internal 7-step pipeline).
    # Enable via config or env vars:
    # - SHORT_TERM_EXTERNAL_FINAL_FORECAST_ENABLED=1
    # - SHORT_TERM_EXTERNAL_FINAL_FORECAST_PATH=...\\forecaster.py
    "external_final_forecast_enabled": False,
    "external_final_forecast_path": None,
    "accuracy_corrections": {
        "block_bias_enabled": True,
        "block_bias_window_days": 30,
        "block_bias_smooth_window": 4,
        "afternoon_ramp_enabled": True,
        "direct_wind_adjustment_enabled": False,
        "direct_rain_adjustment_enabled": False,
        "spline_residual_enabled": False,
        "dynamic_shrinkage_enabled": True,
        "calibration_enabled": True,
        "calibration_window_days": 21,
        "calibration_min_confidence": 0.80,
        "ramp_limit_enabled": True,
        "max_ramp_mw": 300.0,
        "momentum_enabled": True,
        "momentum_decay_blocks": 5.0,
        "t2_night_bias_enabled": True,
        "t2_fallback_night_mw": 350.0,
        # Ramp block sample weighting in ML training (blocks 48-68)
        "ramp_block_start": 48,
        "ramp_block_end": 68,
        "ramp_sample_weight": 1.5,
    },
    # Anomaly day exclusion: filter similar-day candidates where daily load
    # deviated >±N% from trailing-7-day mean (baseline contamination guard).
    "anomaly_exclusion_pct": 0.06,
}

_IMPACT_MIN = -0.40
_IMPACT_MAX = 0.50
_BLOCK_COUNT = 96
logger = logging.getLogger(__name__)
_EXTERNAL_FEATURE_MODULE_CACHE: Dict[str, Any] = {}
_EXTERNAL_FEATURE_FAILURES: set = set()


def _parse_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _external_features_settings(cfg: Optional[dict] = None) -> tuple[bool, Optional[str], bool]:
    cfg = cfg or DEFAULT_CONFIG
    env_enabled = os.getenv("SHORT_TERM_EXTERNAL_FEATURES_ENABLED")
    enabled = _parse_bool(env_enabled) if env_enabled is not None else bool(cfg.get("external_features_enabled", False))

    path = os.getenv("SHORT_TERM_EXTERNAL_FEATURES_PATH") or cfg.get("external_features_path")
    path = str(path).strip() if path else None
    if path and ((not path.lower().endswith(".py")) or (not os.path.isfile(path))):
        path = None

    include_extra_numeric = cfg.get("external_features_include_extra_numeric", True)
    return bool(enabled), path, bool(include_extra_numeric)


def _load_external_module(path: str) -> Any:
    cached = _EXTERNAL_FEATURE_MODULE_CACHE.get(path)
    if cached is not None:
        return cached

    if not str(path).lower().endswith(".py"):
        raise ImportError(f"External features module must be a .py file: {path}")
    if not os.path.isfile(path):
        # Don't cache "missing file" as a permanent failure; it might appear later.
        raise ImportError(f"External features module not found: {path}")

    # Cache failures to avoid repeated imports in hot paths.
    if path in _EXTERNAL_FEATURE_FAILURES:
        raise ImportError(f"External features module previously failed: {path}")

    abspath = os.path.abspath(path)
    # Use a stable unique module name per path to avoid cross-module global collisions.
    digest = hashlib.sha1(abspath.lower().encode("utf-8")).hexdigest()[:12]
    mod_name = f"short_term_external_features_{digest}"

    spec = importlib.util.spec_from_file_location(mod_name, abspath)
    if spec is None or spec.loader is None:
        _EXTERNAL_FEATURE_FAILURES.add(path)
        raise ImportError(f"Cannot load external features module: {path}")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        _EXTERNAL_FEATURE_FAILURES.add(path)
        raise

    _EXTERNAL_FEATURE_MODULE_CACHE[path] = mod
    return mod


def _apply_external_features(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    enabled, path, _include_extra_numeric = _external_features_settings(cfg)
    if not enabled or not path:
        return df

    try:
        mod = _load_external_module(path)
    except Exception as e:
        logger.warning("External features disabled (load failed): %s", e)
        return df

    fn = None
    for name in ("add_features", "apply_features", "build_features", "prepare_features"):
        cand = getattr(mod, name, None)
        if callable(cand):
            fn = cand
            break

    if fn is None:
        logger.warning("External features module has no callable feature function: %s", path)
        return df

    try:
        out = fn(df.copy())
    except TypeError:
        # Some modules may accept (df, cfg) or (df, **kwargs); best-effort.
        try:
            out = fn(df.copy(), cfg=cfg or DEFAULT_CONFIG)
        except Exception as e:
            logger.warning("External feature function failed: %s", e)
            return df
    except Exception as e:
        logger.warning("External feature function failed: %s", e)
        return df

    if isinstance(out, pd.DataFrame) and not out.empty:
        return out
    return df


def _infer_extra_numeric_feature_columns(data: pd.DataFrame) -> List[str]:
    # Exclude identifiers, targets, and columns that are not safe model features by default.
    exclude = {
        "date",
        "time_block",
        "total_drawal",
        "load_feature_source",
        "_is_target_row",
    }
    base = set(SHORT_TERM_MODEL_FEATURES)
    extras: List[str] = []
    for col in data.columns:
        if col in exclude or col in base:
            continue
        s = data[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            extras.append(col)
    return sorted(set(extras))


def _external_final_forecast_settings(cfg: Optional[dict] = None) -> tuple[bool, Optional[str]]:
    cfg = cfg or DEFAULT_CONFIG
    env_enabled = os.getenv("SHORT_TERM_EXTERNAL_FINAL_FORECAST_ENABLED")
    enabled = _parse_bool(env_enabled) if env_enabled is not None else bool(cfg.get("external_final_forecast_enabled", False))

    path = os.getenv("SHORT_TERM_EXTERNAL_FINAL_FORECAST_PATH") or cfg.get("external_final_forecast_path")
    path = str(path).strip() if path else None
    if path and ((not path.lower().endswith(".py")) or (not os.path.isfile(path))):
        path = None
    return bool(enabled), path


def _extract_forecast_vector(external_result: Any) -> Optional[np.ndarray]:
    if external_result is None:
        return None

    if isinstance(external_result, (list, tuple, np.ndarray)):
        return np.asarray(external_result, dtype=float).reshape(-1)

    if isinstance(external_result, pd.DataFrame):
        df = external_result.copy()
        if "time_block" in df.columns:
            df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce")
            df = df.dropna(subset=["time_block"]).sort_values("time_block")
        for col in ("final_load", "forecast_mw", "forecast", "yhat"):
            if col in df.columns:
                vec = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                return vec.reshape(-1)
        return None

    if isinstance(external_result, dict):
        series = external_result.get("series") if isinstance(external_result.get("series"), dict) else None
        if series and "forecast" in series:
            return np.asarray(series.get("forecast"), dtype=float).reshape(-1)
        if "forecast" in external_result:
            return np.asarray(external_result.get("forecast"), dtype=float).reshape(-1)
        if "final_load" in external_result:
            return np.asarray(external_result.get("final_load"), dtype=float).reshape(-1)
        return None

    return None


def _build_external_forecast_response(
    *,
    df: pd.DataFrame,
    target_date: str,
    actual_blocks: int,
    forecast_vec: np.ndarray,
    cfg: dict,
    path: str,
    fn_name: str,
) -> Dict[str, Any]:
    forecast_vec = _normalize_block_vector(forecast_vec, length=_BLOCK_COUNT, fill_value=0.0)
    rows = [
        {
            "time_block": int(i + 1),
            "forecast": float(forecast_vec[i]),
            "forecast_mw": float(forecast_vec[i]),
            "final_load": float(forecast_vec[i]),
        }
        for i in range(_BLOCK_COUNT)
    ]

    actual_full = np.full(_BLOCK_COUNT, np.nan, dtype=float)
    try:
        work = df.copy()
        work["date"] = work.get("date").astype(str)
        day = work[work["date"] == str(target_date)].copy()
        if not day.empty and "time_block" in day.columns:
            day["time_block"] = pd.to_numeric(day["time_block"], errors="coerce")
            day = day.dropna(subset=["time_block"]).sort_values("time_block")
            y = pd.to_numeric(day.get("total_drawal"), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if y.size:
                actual_full[: min(_BLOCK_COUNT, y.size)] = y[:_BLOCK_COUNT]
    except Exception:
        pass

    season = None
    day_type = None
    try:
        season = _season(target_date)
    except Exception:
        season = None
    try:
        day_type = _day_type(target_date)
    except Exception:
        day_type = None

    response: Dict[str, Any] = {
        "date": str(target_date),
        "metadata": {
            "season": season,
            "day_type": day_type,
            "actual_blocks": int(np.clip(actual_blocks, 0, _BLOCK_COUNT)),
            "region": str(cfg.get("region", "")),
            "model_scope": "external",
            "external_final_forecast": {
                "enabled": True,
                "path": str(path),
                "function": str(fn_name),
            },
            "feature_count": None,
            "hybrid_ai_engine": {"enabled": False},
            "forecast_shrinkage": None,
        },
        "weights": cfg.get("hybrid_weights"),
        "calendar_factor": None,
        "series": {
            "blocks": list(range(1, _BLOCK_COUNT + 1)),
            "forecast_raw_before_shrinkage": forecast_vec.tolist(),
            "forecast_after_shrinkage_before_hybrid_ai": forecast_vec.tolist(),
            "hybrid_ai_forecast": forecast_vec.tolist(),
            "forecast": forecast_vec.tolist(),
            "final_load": forecast_vec.tolist(),
            "actual": [float(actual_full[i]) if i < int(actual_blocks) and np.isfinite(actual_full[i]) else None for i in range(_BLOCK_COUNT)],
        },
        "explanation": f"External final forecast module ({fn_name}) used instead of internal pipeline.",
        "bias_factor": None,
        "trend_curve": None,
        "rain_adjustment": None,
        "driver_contributions": [],
        "similar_days": [],
        "peak_impact": None,
        "forecast_df": rows,
        "dip_explanations": [],
        "block_driver_matrix": [],
        "block_contributors": [],
        "slot_sensitivity_profile": [],
        "forecast_uncertainty": [],
        "decision_signals": [],
        "top_contributor_slots": [],
    }
    return response


def _maybe_run_external_final_forecast(
    *,
    df: pd.DataFrame,
    target_date: str,
    actual_blocks: int,
    cfg: dict,
) -> Optional[Dict[str, Any]]:
    enabled, path = _external_final_forecast_settings(cfg)
    if not enabled or not path:
        return None

    try:
        mod = _load_external_module(path)
    except Exception as e:
        logger.warning("External final forecast disabled (load failed): %s", e)
        return None

    fn = None
    fn_name = None
    for name in ("final_forecast", "run_final_forecast", "forecast", "run_forecast", "predict_forecast"):
        cand = getattr(mod, name, None)
        if callable(cand):
            fn = cand
            fn_name = name
            break
    if fn is None:
        logger.warning("External final forecast module has no callable forecast function: %s", path)
        return None

    result = None
    try:
        # Preferred signature: (df, target_date, actual_blocks, config)
        result = fn(df.copy(), target_date=str(target_date), actual_blocks=int(actual_blocks), config=cfg)
    except TypeError:
        try:
            result = fn(df.copy(), str(target_date), int(actual_blocks), cfg)
        except Exception as e:
            logger.warning("External final forecast function failed: %s", e)
            return None
    except Exception as e:
        logger.warning("External final forecast function failed: %s", e)
        return None

    vec = _extract_forecast_vector(result)
    if vec is None:
        logger.warning("External final forecast returned no usable forecast vector.")
        return None

    return _build_external_forecast_response(
        df=df,
        target_date=str(target_date),
        actual_blocks=int(actual_blocks),
        forecast_vec=vec,
        cfg=cfg,
        path=str(path),
        fn_name=str(fn_name or "unknown"),
    )
SHORT_TERM_MODEL_FEATURES = [
    "temperature", "humidity", "precipitation",
    "CDD", "HDD", "temp_roll_3h", "temp_roll_6h", "wbgt",
    "cdh_24h", "cdh_48h", "cdh_72h",
    "apparent_temperature", "cloud_cover", "sunshine_duration",
    "direct_radiation", "wind_speed_10m",
    "solar_index", "solar_proxy_mw",
    "tb_sin", "tb_cos", "is_weekend", "season_idx",
    "hour_cos", "dow_sin", "dow_cos", "block_sin", "block_cos",
    "is_peak_hour", "is_night", "is_business_hour",
    "temp_squared", "temp_cubed", "temp_quartic", "temperature_sin",
    "ramp_heat_interaction",
    "lag_1", "lag_7", "lag_block_1", "lag_block_4",
    "rolling_4", "rolling_12",
    "load_lag_1d", "load_lag_2d", "load_lag_3d",
    "load_lag_7d", "load_rolling_7d", "load_trend_3d",
    "load_x_cdd", "load_x_humidity", "cdd_squared",
    "wbgt_x_load", "temp_momentum_3d",
    "prev_night_dip_flag",
]

# ── Indian State → Climate Region Mapping ──────────────────────────
INDIAN_STATE_REGIONS = {
    # North India (cold winters, hot summers, AC + heating heavy)
    "punjab": "north", "haryana": "north", "delhi": "north",
    "uttar pradesh": "north", "uttarakhand": "north",
    "himachal pradesh": "north", "jammu & kashmir": "north",
    # West India (hot & dry, heavy AC summers, moderate winters)
    "rajasthan": "west", "gujarat": "west", "maharashtra": "west", "goa": "west",
    # South India (tropical, year-round AC, monsoon patterns)
    "tamil nadu": "south", "kerala": "south", "karnataka": "south",
    "andhra pradesh": "south", "telangana": "south",
    # East India (humid, moderate temps, monsoon-sensitive)
    "west bengal": "east", "odisha": "east", "bihar": "east",
    "jharkhand": "east", "assam": "east",
    # Central India (agricultural load, extreme summers)
    "madhya pradesh": "central", "chhattisgarh": "central",
}

# ── Indian State → holidays library subdivision code ─────────────
INDIAN_STATE_HOLIDAY_SUBDIV = {
    "punjab": "PB", "haryana": "HR", "delhi": "DL",
    "uttar pradesh": "UP", "uttarakhand": "UK",
    "himachal pradesh": "HP", "jammu & kashmir": "JK",
    "rajasthan": "RJ", "gujarat": "GJ", "maharashtra": "MH", "goa": "GA",
    "tamil nadu": "TN", "kerala": "KL", "karnataka": "KA",
    "andhra pradesh": "AP", "telangana": "TS",
    "west bengal": "WB", "odisha": "OD", "bihar": "BR",
    "jharkhand": "JH", "assam": "AS",
    "madhya pradesh": "MP", "chhattisgarh": "CG",
}

# ── Human Behaviour Profiles ───────────────────────────────────────
# Keys: (climate_region, season, day_type)
# Values: list of (label, block_start, block_end, mw_boost) tuples
# Block ranges: 1-96 (15-min slots), e.g. block 1=00:00, 25=06:00,
# 37=09:00, 49=12:00, 61=15:00, 73=18:00, 85=21:00
HUMAN_BEHAVIOUR_PROFILES = {
    # ── NORTH ──
    ("north", "summer", "Weekday"): [
        ("AC + Industrial peak",       37, 72,  200),
        ("Evening cooking & lighting",  73, 84,  150),
        ("Late-night cool-down",        1, 20,  -80),
    ],
    ("north", "summer", "Weekend"): [
        ("Residential AC (late start)", 41, 76,  250),
        ("Evening leisure",             77, 88,  120),
        ("Night rest",                   1, 24,  -60),
    ],
    ("north", "winter", "Weekday"): [
        ("Morning heating demand",      21, 36,  200),
        ("Evening heating + lighting",  69, 84,  280),
        ("Midday mild",                 41, 60,  -40),
    ],
    ("north", "winter", "Weekend"): [
        ("Late morning heating",        25, 40,  220),
        ("Evening heating + cooking",   73, 88,  250),
        ("Night base",                   1, 20,  -30),
    ],
    ("north", "spring", "Weekday"): [
        ("Morning ramp",               29, 44,   80),
        ("Afternoon moderate",          49, 64,   60),
        ("Evening cooking",             73, 84,   90),
    ],
    ("north", "spring", "Weekend"): [
        ("Late morning activity",       33, 48,   70),
        ("Evening leisure",             73, 84,   80),
    ],
    ("north", "fall", "Weekday"): [
        ("Morning activity",            25, 40,  100),
        ("Evening early heating",       69, 84,  120),
    ],
    ("north", "fall", "Weekend"): [
        ("Morning leisure",             29, 44,   80),
        ("Evening gathering",           73, 88,  100),
    ],
    # ── WEST ──
    ("west", "summer", "Weekday"): [
        ("Extreme AC demand",           33, 76,  280),
        ("Evening industry wind-down",  77, 84,  100),
        ("Night rest",                   1, 20, -100),
    ],
    ("west", "summer", "Weekend"): [
        ("Residential AC all day",      37, 80,  300),
        ("Night cool-off",               1, 24,  -80),
    ],
    ("west", "winter", "Weekday"): [
        ("Mild morning ramp",           25, 40,   80),
        ("Afternoon steady",            45, 64,   50),
        ("Evening social + cooking",    73, 84,  100),
    ],
    ("west", "winter", "Weekend"): [
        ("Morning leisure",             29, 44,   70),
        ("Evening cooking",             73, 84,   90),
    ],
    ("west", "spring", "Weekday"): [
        ("Pre-summer AC start",         37, 68,  120),
        ("Evening moderate",            73, 84,   80),
    ],
    ("west", "spring", "Weekend"): [
        ("Daytime AC",                  41, 72,  100),
        ("Evening activity",            73, 84,   70),
    ],
    ("west", "fall", "Weekday"): [
        ("Post-monsoon moderate",       33, 64,   80),
        ("Evening demand",              73, 84,   90),
    ],
    ("west", "fall", "Weekend"): [
        ("Daytime moderate",            37, 68,   70),
        ("Evening activity",            73, 84,   80),
    ],
    # ── SOUTH ──
    ("south", "summer", "Weekday"): [
        ("Year-round AC",              33, 72,  180),
        ("Evening tropical load",       73, 84,  130),
        ("Night AC base",                1, 20,   40),
    ],
    ("south", "summer", "Weekend"): [
        ("Residential cooling",         37, 76,  200),
        ("Evening social",              77, 88,  110),
    ],
    ("south", "winter", "Weekday"): [
        ("Mild winter – no heating",    33, 68,   60),
        ("Evening cooking + lighting",  73, 84,  100),
    ],
    ("south", "winter", "Weekend"): [
        ("Moderate daytime",            37, 72,   50),
        ("Evening activity",            73, 84,   80),
    ],
    ("south", "spring", "Weekday"): [
        ("Pre-monsoon heat",            33, 72,  140),
        ("Evening demand",              73, 84,  100),
    ],
    ("south", "spring", "Weekend"): [
        ("Daytime cooling",             37, 76,  120),
        ("Evening leisure",             77, 84,   80),
    ],
    ("south", "fall", "Weekday"): [
        ("Monsoon retreat moderate",    33, 68,   80),
        ("Evening demand",              73, 84,   90),
    ],
    ("south", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   70),
        ("Evening activity",            73, 84,   75),
    ],
    # ── EAST ──
    ("east", "summer", "Weekday"): [
        ("Humid heat AC",              33, 72,  160),
        ("Evening humidity surge",      73, 84,  120),
        ("Night base",                   1, 20,  -50),
    ],
    ("east", "summer", "Weekend"): [
        ("Residential cooling",         37, 76,  180),
        ("Evening social",              77, 88,  100),
    ],
    ("east", "winter", "Weekday"): [
        ("Moderate morning",            25, 40,   70),
        ("Evening cooking + warmth",    73, 84,  100),
    ],
    ("east", "winter", "Weekend"): [
        ("Morning leisure",             29, 44,   60),
        ("Evening gathering",           73, 84,   80),
    ],
    ("east", "spring", "Weekday"): [
        ("Pre-monsoon humid",           33, 68,  100),
        ("Evening demand",              73, 84,   80),
    ],
    ("east", "spring", "Weekend"): [
        ("Daytime moderate",            37, 72,   80),
        ("Evening activity",            73, 84,   70),
    ],
    ("east", "fall", "Weekday"): [
        ("Post-monsoon humid",          33, 68,   90),
        ("Evening Diwali-season",       73, 84,  100),
    ],
    ("east", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   70),
        ("Evening social",              73, 84,   80),
    ],
    # ── CENTRAL ──
    ("central", "summer", "Weekday"): [
        ("Extreme heat + agri pumps",   29, 72,  240),
        ("Evening cooking & fans",      73, 84,  140),
        ("Night relief",                 1, 20,  -70),
    ],
    ("central", "summer", "Weekend"): [
        ("All-day heat demand",         33, 80,  260),
        ("Night cool-down",              1, 24,  -60),
    ],
    ("central", "winter", "Weekday"): [
        ("Morning chill",               21, 36,  100),
        ("Evening heating + cooking",   73, 84,  130),
    ],
    ("central", "winter", "Weekend"): [
        ("Morning warmth",              25, 40,   90),
        ("Evening gathering",           73, 84,  110),
    ],
    ("central", "spring", "Weekday"): [
        ("Pre-summer ramp",             33, 68,  100),
        ("Evening demand",              73, 84,   80),
    ],
    ("central", "spring", "Weekend"): [
        ("Daytime moderate",            37, 72,   80),
        ("Evening activity",            73, 84,   70),
    ],
    ("central", "fall", "Weekday"): [
        ("Post-monsoon agricultural",   29, 68,  110),
        ("Evening demand",              73, 84,   90),
    ],
    ("central", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   80),
        ("Evening social",              73, 84,   80),
    ],
}

# ── Extended day-type profiles (Holiday, BridgeDay, Friday, Monday, etc.) ──
# Generated per (region, season) — holidays suppress load, bridge days lighter.
for _region in ("north", "west", "south", "east", "central"):
    for _season in ("summer", "winter", "spring", "fall"):
        # Get weekday profile as reference for scaling
        _wk_key = (_region, _season, "Weekday")
        _wk = HUMAN_BEHAVIOUR_PROFILES.get(_wk_key, [])
        _peak_mw = max((abs(e[3]) for e in _wk), default=100)

        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday")] = [
            ("Holiday suppression",         33, 72, int(-_peak_mw * 0.5)),
            ("Evening leisure bump",        73, 88, int(_peak_mw * 0.3)),
            ("Night rest",                   1, 20, int(-_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-Fri")] = [
            ("Friday holiday + weekend start", 33, 72, int(-_peak_mw * 0.5)),
            ("Extended evening",            73, 92, int(_peak_mw * 0.35)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-Mon")] = [
            ("Slow Monday ramp",            20, 44, int(-_peak_mw * 0.3)),
            ("Holiday suppression",         45, 72, int(-_peak_mw * 0.4)),
            ("Evening recovery",            73, 84, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-MidWeek")] = [
            ("Mid-week holiday dip",        29, 72, int(-_peak_mw * 0.45)),
            ("Evening bump",                73, 84, int(_peak_mw * 0.25)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "BridgeDay")] = [
            ("Bridge day light suppression", 33, 72, int(-_peak_mw * 0.25)),
            ("Evening normal",              73, 84, int(_peak_mw * 0.15)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "LongWeekend")] = [
            ("Deep weekend rest",           33, 76, int(-_peak_mw * 0.35)),
            ("Evening leisure",             77, 88, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Friday")] = [
            ("Pre-weekend wind-down",       65, 76, int(-_peak_mw * 0.1)),
            ("Evening ramp-up",             77, 92, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Monday")] = [
            ("Monday morning recovery",     20, 36, int(-_peak_mw * 0.15)),
            ("Full activity ramp",          37, 72, int(_peak_mw * 0.1)),
        ]
del _region, _season, _wk_key, _wk, _peak_mw

# ── State-Wise Human Behaviour Profiles (v3.0) ────────────────────
# Override regional profiles when state-level granularity matters.
# Keys: (state_name, season, day_type) — only states that deviate
# significantly from their climate region average need entries here.
# The _human_behaviour_adjustment function checks these FIRST.
STATE_BEHAVIOUR_PROFILES: Dict[tuple, list] = {
    # ── DELHI (DL): Dense AC, metro rail, urban heat island ──
    ("delhi", "summer", "Weekday"): [
        ("Metro + AC peak",           37, 72,  400),
        ("Evening cooking + AC",      73, 84,  250),
        ("Night AC base (urban heat)",  1, 20,  -50),
    ],
    ("delhi", "summer", "Weekend"): [
        ("Residential AC all-day",    37, 76,  450),
        ("Evening leisure",           77, 88,  200),
        ("Night urban heat",           1, 24,  -30),
    ],
    ("delhi", "winter", "Weekday"): [
        ("Morning heating demand",    20, 36,  180),
        ("Evening heating + lighting", 69, 84,  280),
        ("Midday mild",               41, 60,  -40),
    ],
    # ── UTTAR PRADESH (UP): Agricultural pump 6-9AM, evening lighting ──
    ("uttar pradesh", "summer", "Weekday"): [
        ("Agri pump dawn load",       24, 36,  300),
        ("AC + industrial peak",      40, 68,  350),
        ("Evening lighting surge",    73, 84,  200),
        ("Night rest",                 1, 20,  -100),
    ],
    ("uttar pradesh", "summer", "Weekend"): [
        ("Agri pump (continues wknd)", 24, 36, 280),
        ("Residential AC",            40, 72,  300),
        ("Evening social",            73, 88,  180),
    ],
    # ── RAJASTHAN (RJ): Desert cooling, solar ramp compensation ──
    ("rajasthan", "summer", "Weekday"): [
        ("Extreme desert cooling",    38, 70,  400),
        ("Solar sunset compensation", 65, 76,  150),
        ("Evening demand",            73, 84,  200),
        ("Night cool desert",          1, 20,  -120),
    ],
    ("rajasthan", "summer", "Weekend"): [
        ("All-day desert AC",         36, 76,  420),
        ("Evening social",            77, 88,  180),
    ],
    # ── PUNJAB (PB): Paddy pump load Jun-Sep ──
    ("punjab", "summer", "Weekday"): [
        ("Paddy pump load",           24, 40,  250),
        ("AC + industrial",           42, 68,  280),
        ("Evening cooking",           73, 84,  160),
    ],
    ("punjab", "winter", "Weekday"): [
        ("Morning heating (Rabi)",    20, 36,  150),
        ("Evening heating + cooking", 69, 84,  250),
    ],
    # ── MAHARASHTRA (MH): Mumbai industrial + Pune IT ──
    ("maharashtra", "summer", "Weekday"): [
        ("Mumbai industrial base",    25, 36,  300),
        ("AC + IT corridor peak",     38, 70,  500),
        ("Evening residential",       73, 84,  250),
        ("Night industrial base",      1, 20,  -80),
    ],
    ("maharashtra", "summer", "Weekend"): [
        ("Residential AC dominant",   37, 76,  450),
        ("Evening social",            77, 88,  200),
    ],
    ("maharashtra", "fall", "Weekday"): [
        ("Post-monsoon humid",        33, 68,  250),
        ("Rain-day suppression",      73, 84, -150),
        ("Evening demand",            77, 88,  180),
    ],
    # ── GUJARAT (GJ): Textile mills, Jamnagar refinery ──
    ("gujarat", "summer", "Weekday"): [
        ("Textile + refinery base",   25, 36,  200),
        ("AC + industrial peak",      38, 68,  420),
        ("Evening demand",            73, 84,  180),
    ],
    # ── TAMIL NADU (TN): Chennai AC, Coimbatore textile ──
    ("tamil nadu", "summer", "Weekday"): [
        ("Textile continuous",        25, 36,  200),
        ("Chennai AC peak",           40, 72,  400),
        ("Evening tropical load",     73, 88,  250),
        ("Night AC base (tropical)",   1, 20,   40),
    ],
    ("tamil nadu", "summer", "Weekend"): [
        ("Residential cooling",       37, 76,  350),
        ("Evening social",            77, 88,  220),
    ],
    # ── KARNATAKA (KA): Bengaluru IT sustained 9AM-9PM ──
    ("karnataka", "summer", "Weekday"): [
        ("IT corridor sustained",     36, 84,  350),
        ("Morning ramp (IT start)",   33, 40,  200),
        ("Evening continued IT",      73, 84,  200),
    ],
    ("karnataka", "summer", "Weekend"): [
        ("Residential + IT weekend",  40, 76,  280),
        ("Evening Bengaluru social",  77, 84,  180),
    ],
    # ── TELANGANA (TS): Hyderabad pharma/IT 24h, high AC ──
    ("telangana", "summer", "Weekday"): [
        ("Pharma/IT 24h base",        1, 96,   80),
        ("AC peak overlay",           40, 68,  350),
        ("Evening residential",       73, 86,  220),
    ],
    ("telangana", "summer", "Weekend"): [
        ("Pharma continuous",          1, 96,   60),
        ("Residential AC",            40, 76,  300),
        ("Evening social",            77, 88,  180),
    ],
    # ── WEST BENGAL (WB): Kolkata urban heat island, jute ──
    ("west bengal", "summer", "Weekday"): [
        ("Kolkata heat island AC",    40, 72,  300),
        ("Jute industry dawn",        24, 36,  150),
        ("Humid evening surge",       73, 84,  200),
    ],
    ("west bengal", "summer", "Weekend"): [
        ("Residential humid AC",      40, 76,  280),
        ("Evening social",            77, 88,  180),
    ],
    # ── ODISHA (OR): Mining/steel, cyclone events ──
    ("odisha", "summer", "Weekday"): [
        ("Steel/mining base",         25, 72,  250),
        ("Agri pump morning",         24, 36,  180),
        ("Evening demand",            73, 84,  150),
    ],
    ("odisha", "fall", "Weekday"): [
        ("Cyclone suppression risk",  33, 72, -200),
        ("Post-event recovery",       73, 84,  100),
    ],
    # ── BIHAR (BR): Agricultural, low industrial, evening lighting ──
    ("bihar", "summer", "Weekday"): [
        ("Agri pump dawn",            24, 36,  200),
        ("Daytime moderate",          40, 66,  200),
        ("Evening lighting peak",     73, 84,  250),
        ("Night low base",             1, 20,  -100),
    ],
    # ── HIMACHAL PRADESH (HP): Hill station, heating dominant ──
    ("himachal pradesh", "winter", "Weekday"): [
        ("Heavy heating morning",     16, 36,  200),
        ("Heating sustained day",     37, 68,  100),
        ("Evening peak heating",      69, 84,  250),
    ],
    ("himachal pradesh", "summer", "Weekday"): [
        ("Tourism mild cooling",      44, 64,   80),
        ("Evening tourism activity",  73, 84,   60),
    ],
    # ── JAMMU & KASHMIR (JK): Electric heating extreme winter ──
    ("jammu & kashmir", "winter", "Weekday"): [
        ("Extreme electric heating",  14, 36,  220),
        ("Sustained daytime heating", 37, 68,  120),
        ("Evening peak heating",      69, 84,  280),
    ],
    # ── ASSAM (AS): Tea processing, oil refinery ──
    ("assam", "summer", "Weekday"): [
        ("Tea processing ramp",       28, 44,  100),
        ("Moderate humid AC",         46, 64,  120),
        ("Evening demand",            73, 84,   80),
    ],
    # ── CHHATTISGARH (CG): Steel/aluminium smelter, Korba ──
    ("chhattisgarh", "summer", "Weekday"): [
        ("Smelter continuous base",    1, 96,  100),
        ("Peak industrial + AC",      38, 68,  250),
        ("Evening demand",            73, 84,  150),
    ],
    ("chhattisgarh", "fall", "Weekday"): [
        ("Monsoon onset sharp drop",  33, 72, -220),
        ("Post-rain recovery",        73, 84,  100),
    ],
    # ── KERALA (KL): High humidity AC, fishing dawn ──
    ("kerala", "summer", "Weekday"): [
        ("Fishing industry dawn",     20, 32,   80),
        ("Humid AC demand",           44, 68,  160),
        ("Evening tropical",          73, 88,  180),
    ],
    # ── ANDHRA PRADESH (AP): Vizag industrial, aquaculture ──
    ("andhra pradesh", "summer", "Weekday"): [
        ("Aquaculture pump dawn",     22, 36,  200),
        ("Industrial + AC peak",      40, 70,  320),
        ("Evening demand",            73, 86,  150),
    ],
}


# ── Calibrated behaviour cache ───────────────────────────────────
_CALIBRATED_BEHAVIOUR: Dict[tuple, np.ndarray] = {}
_CALIBRATED_BEHAVIOUR_META: Dict[tuple, Dict[str, Any]] = {}


def _cyclic_block_distance(block: float, center: float, block_count: int = _BLOCK_COUNT) -> float:
    raw = abs(float(block) - float(center))
    return float(min(raw, float(block_count) - raw))


def _circular_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    size = int(max(1, window))
    if size <= 1 or arr.size == 1:
        return arr.copy()
    if size % 2 == 0:
        size += 1
    radius = size // 2
    smoothed = np.zeros_like(arr, dtype=float)
    for shift in range(-radius, radius + 1):
        smoothed += np.roll(arr, shift)
    return smoothed / float((2 * radius) + 1)


def _build_cyclic_time_basis(blocks: np.ndarray, harmonics: int = 4) -> np.ndarray:
    blk = np.asarray(blocks, dtype=float).reshape(-1)
    angle = (2.0 * np.pi * (blk - 1.0)) / float(_BLOCK_COUNT)
    cols = [np.ones_like(angle)]
    for harmonic in range(1, int(max(1, harmonics)) + 1):
        cols.append(np.sin(harmonic * angle))
        cols.append(np.cos(harmonic * angle))
    return np.column_stack(cols)


def _fit_cyclic_behaviour_curve(
    blocks: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    harmonics: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    blk = np.asarray(blocks, dtype=float).reshape(-1)
    y = np.asarray(residuals, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(blk) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(np.sum(mask)) < max(12, (2 * int(max(1, harmonics))) + 1):
        return np.zeros(_BLOCK_COUNT, dtype=float), {
            "method": "cyclic_fourier",
            "harmonics": int(max(1, harmonics)),
            "samples": int(np.sum(mask)),
            "fitted": False,
        }

    X = _build_cyclic_time_basis(blk[mask], harmonics=harmonics)
    sqrt_w = np.sqrt(np.clip(w[mask], 1e-9, None))
    Xw = X * sqrt_w[:, None]
    yw = y[mask] * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = _build_cyclic_time_basis(np.arange(1, _BLOCK_COUNT + 1), harmonics=harmonics) @ beta
    return np.asarray(pred, dtype=float), {
        "method": "cyclic_fourier",
        "harmonics": int(max(1, harmonics)),
        "samples": int(np.sum(mask)),
        "fitted": True,
    }


def _calibrate_behaviour_profiles(df: pd.DataFrame, region: str, cfg: dict) -> Dict[tuple, np.ndarray]:
    """Learn cyclic 96-block MW profiles from historical residuals per day-type."""
    cal_days = int(cfg.get("behaviour_calibration_days", 90))
    harmonics = int(max(1, cfg.get("behaviour_cyclic_harmonics", 4) or 4))
    cyclic_blend = float(np.clip(float(cfg.get("behaviour_cyclic_blend", 0.55) or 0.55), 0.0, 1.0))
    smooth_window = int(max(1, cfg.get("behaviour_cyclic_smoothing", 5) or 5))
    climate_region = INDIAN_STATE_REGIONS.get(region.lower().strip(), region.lower().strip())
    result: Dict[tuple, np.ndarray] = {}
    global _CALIBRATED_BEHAVIOUR_META
    _CALIBRATED_BEHAVIOUR_META = {}

    if df.empty or "total_drawal" not in df.columns or "date" not in df.columns:
        return result

    work = df.copy()
    work["date"] = work["date"].astype(str)
    dates_sorted = sorted(work["date"].unique())
    if cal_days > 0 and len(dates_sorted) > cal_days:
        keep = set(dates_sorted[-cal_days:])
        work = work[work["date"].isin(keep)]

    if work.empty or "time_block" not in work.columns:
        return result

    # Compute 7-day rolling baseline per block
    work = work.sort_values(["time_block", "date"])
    work["rolling_baseline"] = work.groupby("time_block")["total_drawal"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean()
    )
    work["residual"] = work["total_drawal"] - work["rolling_baseline"]
    work = work.dropna(subset=["residual"])

    # Recency weighting: 21-day half-life exponential decay (v3.0)
    max_date = pd.to_datetime(work["date"]).max()
    work["_age_days"] = (max_date - pd.to_datetime(work["date"])).dt.days
    work["_recency_weight"] = np.exp(-0.693 * work["_age_days"] / 21.0)

    if work.empty:
        return result

    # Compute holiday flags for grouping
    hf = _compute_holiday_flags(work, region=region)
    work["season"] = work["date"].apply(_season)

    # Build extended day type per row
    flag_cols = ["is_holiday", "bridge_day", "long_weekend", "holiday_before_weekend",
                 "holiday_after_weekend", "mid_week_holiday"]
    for col in flag_cols:
        if col in hf.columns:
            work[col] = hf[col].values

    def _row_day_type(row):
        flags = {c: int(row.get(c, 0)) for c in flag_cols}
        return _day_type_extended(row["date"], flags)

    work["ext_day_type"] = work.apply(_row_day_type, axis=1)

    # Group by (season, ext_day_type) and learn a cyclic profile from data.
    for (ssn, dt), grp in work.groupby(["season", "ext_day_type"]):
        if len(grp) < 3:
            continue

        def _weighted_mean(sub):
            w = sub["_recency_weight"].values
            r = sub["residual"].values
            return float(np.average(r, weights=w)) if len(w) > 0 and w.sum() > 0 else float(np.mean(r))

        profile = (
            grp.groupby("time_block")
            .apply(_weighted_mean, include_groups=False)
            .reindex(range(1, _BLOCK_COUNT + 1))
        )
        if profile.isna().all():
            continue

        empirical = profile.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
        cyclic_curve, cyclic_diag = _fit_cyclic_behaviour_curve(
            blocks=grp["time_block"].to_numpy(dtype=float),
            residuals=grp["residual"].to_numpy(dtype=float),
            weights=grp["_recency_weight"].to_numpy(dtype=float),
            harmonics=harmonics,
        )
        if not bool(cyclic_diag.get("fitted")):
            cyclic_curve = empirical.copy()

        day_count = int(grp["date"].astype(str).nunique()) if "date" in grp.columns else 0
        data_blend = float(np.clip(cyclic_blend + (0.20 if day_count < 8 else 0.0) - (0.10 if day_count >= 20 else 0.0), 0.25, 0.85))
        arr = (data_blend * cyclic_curve) + ((1.0 - data_blend) * empirical)
        arr = _circular_smooth(arr, window=smooth_window)

        state_peak = float(grp["total_drawal"].quantile(0.95)) if "total_drawal" in grp.columns else 5000.0
        clip_bound = max(0.08 * state_peak, 150.0)
        arr = np.clip(arr, -clip_bound, clip_bound)
        key = (climate_region, ssn, dt)
        result[key] = arr
        _CALIBRATED_BEHAVIOUR_META[key] = {
            "source": "calibrated_cyclic_data",
            "harmonics": harmonics,
            "day_count": day_count,
            "blend_weight_cyclic": round(float(data_blend), 3),
            "smoothing_window": smooth_window,
            "clip_bound_mw": round(float(clip_bound), 2),
            "fit": cyclic_diag,
        }

    return result


def _human_behaviour_adjustment(
    season: str,
    region: str,
    day_type: str,
    weight: float = 1.0,
) -> Tuple[np.ndarray, str]:
    """
    Compute a 96-block MW adjustment vector based on human behaviour
    patterns for the given season × region × day_type.

    Returns (adjustment_vector, profile_label).
    """
    state_name = region.lower().strip()
    climate_region = INDIAN_STATE_REGIONS.get(state_name, state_name)

    # Priority 1: Calibrated profiles from historical data
    cal_key = (climate_region, season, day_type)
    if _CALIBRATED_BEHAVIOUR and cal_key in _CALIBRATED_BEHAVIOUR:
        meta = _CALIBRATED_BEHAVIOUR_META.get(cal_key, {})
        harmonics = meta.get("harmonics")
        label = f"CalibratedCyclic({climate_region}/{season}/{day_type}"
        if harmonics is not None:
            label += f"/h{int(harmonics)}"
        label += ")"
        return _CALIBRATED_BEHAVIOUR[cal_key] * weight, label

    # Priority 2: State-specific hardcoded profiles (v3.0)
    state_key = (state_name, season, day_type)
    entries = STATE_BEHAVIOUR_PROFILES.get(state_key)
    if not entries:
        # Try state fallback to Weekday/Weekend
        for dt in ("Weekday", "Weekend"):
            state_fb = (state_name, season, dt)
            if state_fb in STATE_BEHAVIOUR_PROFILES:
                entries = STATE_BEHAVIOUR_PROFILES[state_fb]
                break

    # Priority 3: Climate region profiles (original)
    if not entries:
        key = (climate_region, season, day_type)
        entries = HUMAN_BEHAVIOUR_PROFILES.get(key)
    if not entries:
        for dt in ("Weekday", "Weekend"):
            fallback_key = (climate_region, season, dt)
            if fallback_key in HUMAN_BEHAVIOUR_PROFILES:
                entries = HUMAN_BEHAVIOUR_PROFILES[fallback_key]
                break
    if not entries:
        return np.zeros(96, dtype=float), "No profile"

    adjustment = np.zeros(96, dtype=float)
    labels = []

    for label, b_start, b_end, mw_boost in entries:
        labels.append(label)
        # Build a Gaussian-smoothed window to avoid step edges
        center = (b_start + b_end) / 2.0
        width = (b_end - b_start) / 2.0
        sigma = max(width / 2.5, 2.0)  # smooth falloff
        for b in range(96):
            block = b + 1  # 1-indexed
            dist = _cyclic_block_distance(block, center, block_count=_BLOCK_COUNT)
            gauss = np.exp(-0.5 * (dist / sigma) ** 2)
            adjustment[b] += mw_boost * gauss

    adjustment *= weight
    adjustment = _circular_smooth(adjustment, window=5)
    profile_label = " + ".join(labels)
    return adjustment, profile_label

def _safe_corr(a, b) -> float:
    """Spearman rank correlation (robust to outliers and non-linear relationships)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) <= 1e-6 or np.std(b) <= 1e-6:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _normalize_similarity_weights(weights: Dict[str, float]) -> Dict[str, float]:
    vec = np.array([
        max(0.0, float(weights.get("temp", 0.0))),
        max(0.0, float(weights.get("humidity", 0.0))),
        max(0.0, float(weights.get("rain", 0.0))),
    ], dtype=float)
    if float(np.sum(vec)) <= 1e-9:
        vec = np.array([0.55, 0.25, 0.20], dtype=float)
    vec = vec / max(float(np.sum(vec)), 1e-9)
    return {
        "temp": float(vec[0]),
        "humidity": float(vec[1]),
        "rain": float(vec[2]),
    }


def _compute_similarity_profile_from_data(df: pd.DataFrame, target_date: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prior = _normalize_similarity_weights(cfg.get("similarity_weights", {}))
    out = {
        "weights": prior,
        "require_rain_match": bool(cfg.get("require_rain_match", True)),
        "source": "default",
        "diagnostics": {},
    }

    required_cols = {"date", "temperature", "humidity", "precipitation", "total_drawal"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        out["diagnostics"] = {"reason": "insufficient_columns_or_empty"}
        return out

    hist = df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_history_before_target"}
        return out

    calibration_days = int(cfg.get("similarity_calibration_days", cfg.get("weather_training_days", 120)) or 120)
    if calibration_days > 0:
        hist_dates = sorted(hist["date"].dropna().unique().tolist())
        keep_dates = set(hist_dates[-calibration_days:])
        hist = hist[hist["date"].isin(keep_dates)]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_history_after_calibration_window"}
        return out

    for col in ("temperature", "humidity", "precipitation", "total_drawal"):
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.dropna(subset=["total_drawal"])
    if hist.empty:
        out["diagnostics"] = {"reason": "no_numeric_history"}
        return out

    daily = (
        hist.groupby("date")
        .agg(
            temp_mean=("temperature", "mean"),
            hum_mean=("humidity", "mean"),
            rain_any=("precipitation", lambda x: float((x.fillna(0) > 0).any())),
            load_sum=("total_drawal", "sum"),
        )
        .reset_index()
    )
    daily = daily.dropna(subset=["load_sum"])
    if len(daily) < 10:
        out["diagnostics"] = {"reason": "too_few_days", "days": int(len(daily))}
        return out

    # ── Full-day correlations ─────────────────────────────────────────────────
    temp_signal_all = abs(_safe_corr(daily["temp_mean"].to_numpy(dtype=float), daily["load_sum"].to_numpy(dtype=float)))
    hum_signal_all  = abs(_safe_corr(daily["hum_mean"].to_numpy(dtype=float),  daily["load_sum"].to_numpy(dtype=float)))
    rain_signal_all = abs(_safe_corr(daily["rain_any"].to_numpy(dtype=float),  daily["load_sum"].to_numpy(dtype=float)))

    # ── Peak-block correlations (blocks 33–72 by default, updated dynamically) ─
    # Weather drives peak-hour load more than off-peak; compute correlations
    # on peak-block load only to get a sharper signal for similarity matching.
    _peak_s = int(cfg.get("forecast_shrinkage", {}).get("peak_start_block", 33) if isinstance(cfg.get("forecast_shrinkage"), dict) else 33)
    _peak_e = int(cfg.get("forecast_shrinkage", {}).get("peak_end_block", 72)   if isinstance(cfg.get("forecast_shrinkage"), dict) else 72)
    try:
        peak_hist = hist[hist["time_block"].between(_peak_s, _peak_e)] if "time_block" in hist.columns else hist
        peak_daily = (
            peak_hist.groupby("date")
            .agg(
                temp_mean=("temperature", "mean"),
                hum_mean=("humidity", "mean"),
                rain_any=("precipitation", lambda x: float((x.fillna(0) > 0).any())),
                load_sum=("total_drawal", "sum"),
            )
            .reset_index()
            .dropna(subset=["load_sum"])
        )
        if len(peak_daily) >= 10:
            temp_signal_pk = abs(_safe_corr(peak_daily["temp_mean"].to_numpy(dtype=float), peak_daily["load_sum"].to_numpy(dtype=float)))
            hum_signal_pk  = abs(_safe_corr(peak_daily["hum_mean"].to_numpy(dtype=float),  peak_daily["load_sum"].to_numpy(dtype=float)))
            rain_signal_pk = abs(_safe_corr(peak_daily["rain_any"].to_numpy(dtype=float),  peak_daily["load_sum"].to_numpy(dtype=float)))
            # Blend full-day and peak-block signals: 40% full-day + 60% peak
            temp_signal = 0.40 * temp_signal_all + 0.60 * temp_signal_pk
            hum_signal  = 0.40 * hum_signal_all  + 0.60 * hum_signal_pk
            rain_signal = 0.40 * rain_signal_all  + 0.60 * rain_signal_pk
        else:
            temp_signal, hum_signal, rain_signal = temp_signal_all, hum_signal_all, rain_signal_all
    except Exception:
        temp_signal, hum_signal, rain_signal = temp_signal_all, hum_signal_all, rain_signal_all

    data_vec = np.array([temp_signal, hum_signal, rain_signal], dtype=float)

    prior_blend = float(np.clip(float(cfg.get("similarity_weight_prior_blend", 0.35)), 0.0, 1.0))
    prior_vec = np.array([prior["temp"], prior["humidity"], prior["rain"]], dtype=float)
    if float(np.sum(data_vec)) > 1e-9:
        data_vec = data_vec / float(np.sum(data_vec))
        final_vec = ((1.0 - prior_blend) * data_vec) + (prior_blend * prior_vec)
        final_vec = np.clip(final_vec, 1e-6, None)
        final_vec = final_vec / float(np.sum(final_vec))
        out["weights"] = {
            "temp": float(final_vec[0]),
            "humidity": float(final_vec[1]),
            "rain": float(final_vec[2]),
        }
        out["source"] = "data_calibrated"
    else:
        out["weights"] = prior
        out["source"] = "default_no_signal"

    rain_any = (daily["rain_any"].to_numpy(dtype=float) > 0.5)
    rainy_days = int(np.sum(rain_any))
    dry_days = int(len(daily) - rainy_days)
    rain_rate = float(np.mean(rain_any)) if len(daily) else 0.0
    mean_load = float(np.mean(daily["load_sum"])) if len(daily) else 0.0
    mean_rain_load = float(np.mean(daily.loc[rain_any, "load_sum"])) if rainy_days > 0 else np.nan
    mean_dry_load = float(np.mean(daily.loc[~rain_any, "load_sum"])) if dry_days > 0 else np.nan
    if np.isfinite(mean_rain_load) and np.isfinite(mean_dry_load) and mean_load > 1e-6:
        rain_effect = float(abs(mean_rain_load - mean_dry_load) / mean_load)
    else:
        rain_effect = 0.0

    min_rain_days = int(max(1, int(cfg.get("rain_match_min_days", 3))))
    rain_effect_threshold = float(max(0.0, float(cfg.get("rain_effect_threshold", 0.01))))
    auto_rain = (
        rainy_days >= min_rain_days
        and dry_days >= min_rain_days
        and rain_effect >= rain_effect_threshold
        and (0.03 <= rain_rate <= 0.97)
    )

    if bool(cfg.get("auto_require_rain_match", True)):
        out["require_rain_match"] = bool(auto_rain)
    else:
        out["require_rain_match"] = bool(cfg.get("require_rain_match", True))

    out["diagnostics"] = {
        "days_used": int(len(daily)),
        "rainy_days": rainy_days,
        "dry_days": dry_days,
        "rain_rate": float(rain_rate),
        "rain_effect": float(rain_effect),
        "rain_effect_threshold": float(rain_effect_threshold),
        "signals": {
            "temp": float(temp_signal),
            "humidity": float(hum_signal),
            "rain": float(rain_signal),
        },
        "prior_blend": float(prior_blend),
    }
    return out


def _season(date_str: str) -> str:
    try:
        month = int(date_str.split("-")[1])
    except Exception:
        return "unknown"
    return SEASON_BY_MONTH.get(month, "unknown")


def _day_type(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        return "unknown"
    return "Weekend" if dt.weekday() >= 5 else "Weekday"


def _day_type_extended(date_str: str, holiday_flags: dict = None) -> str:
    """Classify into one of 10 day types for behaviour lookup."""
    dt = pd.to_datetime(date_str)
    dow = dt.weekday()
    if holiday_flags:
        if holiday_flags.get("is_holiday"):
            if holiday_flags.get("holiday_before_weekend"):
                return "Holiday-Fri"
            if holiday_flags.get("holiday_after_weekend"):
                return "Holiday-Mon"
            if holiday_flags.get("mid_week_holiday"):
                return "Holiday-MidWeek"
            return "Holiday"
        if holiday_flags.get("bridge_day"):
            return "BridgeDay"
        if holiday_flags.get("long_weekend"):
            return "LongWeekend"
    if dow >= 5:
        return "Weekend"
    if dow == 4:
        return "Friday"
    if dow == 0:
        return "Monday"
    return "Weekday"


def _compute_holiday_flags(df: pd.DataFrame, region: str = "haryana") -> pd.DataFrame:
    """Add is_holiday and all transition flags using national + state-specific holidays."""
    out = df.copy()
    out["is_holiday"] = 0
    try:
        import holidays as _hol
        dates = pd.to_datetime(out["date"])
        years = dates.dt.year.unique().tolist()
        national = _hol.India(years=years)
        subdiv = INDIAN_STATE_HOLIDAY_SUBDIV.get(region.lower().strip(), "HR")
        state = _hol.India(subdiv=subdiv, years=years)
        combined_dates = set(national.keys()) | set(state.keys())
        out["is_holiday"] = dates.map(lambda d: int(d in combined_dates))
    except Exception:
        dates = pd.to_datetime(out["date"])
        combined_dates = set()

    holiday_set = set(dates[out["is_holiday"] == 1].unique()) if out["is_holiday"].any() else set()

    if holiday_set:
        out["holiday_on_weekday"] = dates.map(lambda d: int(d in holiday_set and d.weekday() < 5))
        out["holiday_before_weekend"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 4))
        out["holiday_after_weekend"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 0))
        out["bridge_day"] = dates.map(lambda d: int(
            d not in holiday_set and d.weekday() < 5 and (
                (d + pd.Timedelta(days=1)) in holiday_set or
                (d - pd.Timedelta(days=1)) in holiday_set
            )
        ))
        out["long_weekend"] = dates.map(lambda d: int(
            d.weekday() in (5, 6) and (
                (d - pd.Timedelta(days=1)) in holiday_set or
                (d + pd.Timedelta(days=1)) in holiday_set
            )
        ))
        out["post_weekend_holiday"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 0))
        out["mid_week_holiday"] = dates.map(lambda d: int(
            d in holiday_set and d.weekday() in (1, 2, 3)
        ))
        holidays_sorted = np.array(sorted(holiday_set))
        out["days_to_holiday"] = dates.map(
            lambda d: int((holidays_sorted[holidays_sorted >= d][0] - d).days)
            if (holidays_sorted >= d).any() else 999
        )
        out["days_since_holiday"] = dates.map(
            lambda d: int((d - holidays_sorted[holidays_sorted <= d][-1]).days)
            if (holidays_sorted <= d).any() else 999
        )
    else:
        for col in ["holiday_on_weekday", "holiday_before_weekend", "holiday_after_weekend",
                     "bridge_day", "long_weekend", "post_weekend_holiday", "mid_week_holiday"]:
            out[col] = 0
        out["days_to_holiday"] = 999
        out["days_since_holiday"] = 999

    return out


def _calendar_factor(df: pd.DataFrame, calendar_config: Optional[str],
                     holiday_flags: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Returns discrete calendar impacts.
    Keys: "total", "day_type", "holiday", "transition"
    Supports both legacy string calendar_config AND auto-detect via holiday_flags.
    """
    base = {"total": np.ones(96, dtype=float), "day_type": np.ones(96, dtype=float),
            "holiday": np.ones(96, dtype=float), "transition": np.ones(96, dtype=float)}
    if df.empty:
        return base

    df = df.copy()
    df["day_type"] = df["date"].apply(_day_type)
    daily = df.groupby(["date", "day_type"])["total_drawal"].sum().reset_index()
    weekday_avg = daily[daily["day_type"] == "Weekday"]["total_drawal"].mean()
    weekend_avg = daily[daily["day_type"] == "Weekend"]["total_drawal"].mean()

    if not weekday_avg or weekday_avg < 1e-6:
        return base

    weekend_factor = float((weekend_avg / weekday_avg) if weekend_avg else 0.92)
    holiday_ratio = min(weekend_factor, 0.92)

    # ── Legacy string-based calendar_config (backwards compat) ──
    if calendar_config:
        cfg_str = calendar_config.strip().lower()
        if cfg_str == "weekend":
            base["total"] = np.full(96, weekend_factor, dtype=float)
            base["day_type"] = np.full(96, weekend_factor, dtype=float)
        elif cfg_str in ("holiday", "holiday_to_weekend", "weekend_to_holiday"):
            base["total"] = np.full(96, holiday_ratio, dtype=float)
            base["holiday"] = np.full(96, holiday_ratio, dtype=float)
        elif cfg_str == "weekend_to_weekday":
            trans = np.linspace(weekend_factor, 1.0, 96)
            base["total"] = trans
            base["transition"] = trans
        elif cfg_str == "weekday_to_weekend":
            trans = np.linspace(1.0, weekend_factor, 96)
            base["total"] = trans
            base["transition"] = trans
        return base

    # ── Auto-detect from holiday_flags ──
    if holiday_flags is None:
        return base

    hf = holiday_flags
    is_holiday = hf.get("is_holiday", 0)
    is_bridge = hf.get("bridge_day", 0)
    is_long_wknd = hf.get("long_weekend", 0)
    holiday_before_wknd = hf.get("holiday_before_weekend", 0)
    holiday_after_wknd = hf.get("holiday_after_weekend", 0)
    mid_week_hol = hf.get("mid_week_holiday", 0)
    days_to = hf.get("days_to_holiday", 999)
    days_since = hf.get("days_since_holiday", 999)

    if is_holiday:
        base["total"] = np.full(96, holiday_ratio)
        base["holiday"] = np.full(96, holiday_ratio)
        if holiday_before_wknd:
            evening_dip = np.ones(96)
            evening_dip[68:88] *= 0.95
            base["total"] = base["total"] * evening_dip
        if holiday_after_wknd:
            morning_slow = np.ones(96)
            morning_slow[20:40] = np.linspace(holiday_ratio, 1.0, 20)
            base["transition"] = morning_slow
    elif is_bridge:
        bridge_ratio = 0.5 * holiday_ratio + 0.5 * 1.0
        base["total"] = np.full(96, bridge_ratio)
        base["day_type"] = np.full(96, bridge_ratio)
    elif is_long_wknd:
        deep_weekend = weekend_factor * 0.97
        base["total"] = np.full(96, deep_weekend)
        base["day_type"] = np.full(96, deep_weekend)
    elif mid_week_hol:
        base["total"] = np.full(96, holiday_ratio)
        base["holiday"] = np.full(96, holiday_ratio)
    else:
        if days_to == 1:
            pre_hol = np.ones(96)
            pre_hol[72:96] *= 0.97
            base["total"] = pre_hol
            base["transition"] = pre_hol
        elif days_since == 1:
            post_hol = np.ones(96)
            post_hol[20:44] = np.linspace(0.96, 1.0, 24)
            base["total"] = post_hol
            base["transition"] = post_hol

    return base


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = out["date"].apply(_season)
    out["day_type"] = out["date"].apply(_day_type)
    out["is_weekend"] = out["day_type"].eq("Weekend").astype(int)
    return out


def _timeblock_features(series: pd.Series) -> pd.DataFrame:
    tb = series.astype(float)
    angle = 2 * np.pi * (tb - 1) / 96.0
    return pd.DataFrame({
        "tb_sin": np.sin(angle),
        "tb_cos": np.cos(angle),
    })


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical, categorical & weather-interaction temporal features."""
    if "time_block" not in df.columns:
        return df
    tb = df["time_block"].astype(float)
    hour = ((tb - 1) * 15 / 60).astype(int)
    dow = pd.to_datetime(df["date"]).dt.dayofweek if "date" in df.columns else pd.Series(0, index=df.index)

    # ── cyclical hour / day-of-week ──
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
    df["block_sin"] = np.sin(2 * np.pi * (tb - 1.0) / _BLOCK_COUNT)
    df["block_cos"] = np.cos(2 * np.pi * (tb - 1.0) / _BLOCK_COUNT)

    # ── period flags ──
    df["is_peak_hour"]     = hour.isin([7, 8, 9, 18, 19, 20]).astype(int)
    df["is_night"]         = hour.isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
    df["is_business_hour"] = ((hour >= 9) & (hour <= 17)).astype(int)

    # ── temperature × diurnal interaction ──
    if "temperature" in df.columns:
        t = df["temperature"]
        df["temp_squared"]     = t ** 2
        df["temp_cubed"]       = t ** 3
        df["temperature_sin"]  = t * np.sin(2 * np.pi * hour / 24)

    return df


def _get_short_term_feature_columns(data: pd.DataFrame, cfg: Optional[dict] = None) -> List[str]:
    base = [feature for feature in SHORT_TERM_MODEL_FEATURES if feature in data.columns]
    enabled, _path, include_extra_numeric = _external_features_settings(cfg)
    if enabled and include_extra_numeric:
        extras = _infer_extra_numeric_feature_columns(data)
        return base + [c for c in extras if c not in base]
    return base


def _build_load_anchor(
    baseline_hist: np.ndarray,
    actual: Optional[np.ndarray] = None,
    actual_blocks: int = 0,
) -> np.ndarray:
    anchor = np.asarray(baseline_hist, dtype=float).reshape(-1).copy()
    if anchor.size < _BLOCK_COUNT:
        anchor = np.pad(anchor, (0, _BLOCK_COUNT - anchor.size), mode="edge")
    elif anchor.size > _BLOCK_COUNT:
        anchor = anchor[:_BLOCK_COUNT]

    actual_vec = np.asarray(actual if actual is not None else [], dtype=float).reshape(-1)
    observed = int(np.clip(actual_blocks, 0, min(anchor.size, actual_vec.size)))
    if observed > 0:
        anchor[:observed] = actual_vec[:observed]
    return anchor


def _prepare_inference_frame(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    load_anchor: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    history = history_df.copy() if history_df is not None else pd.DataFrame()
    target = target_df.copy()

    if history.empty:
        history = pd.DataFrame()
    else:
        history["load_feature_source"] = pd.to_numeric(history.get("total_drawal"), errors="coerce")

    target["load_feature_source"] = pd.to_numeric(target.get("total_drawal"), errors="coerce")
    if load_anchor is not None:
        anchor = np.asarray(load_anchor, dtype=float).reshape(-1)
        if anchor.size < len(target):
            anchor = np.pad(anchor, (0, len(target) - anchor.size), mode="edge")
        elif anchor.size > len(target):
            anchor = anchor[:len(target)]
        target["load_feature_source"] = anchor

    target["_is_target_row"] = 1
    if history.empty:
        combined = target.copy()
    else:
        history["_is_target_row"] = 0
        union_cols = sorted(set(history.columns).union(target.columns))
        history = history.reindex(columns=union_cols)
        target = target.reindex(columns=union_cols)
        usable_cols = [
            col for col in union_cols
            if not (history[col].isna().all() and target[col].isna().all())
        ]
        hist_part = history[usable_cols].dropna(axis=1, how="all")
        tgt_part = target[usable_cols].dropna(axis=1, how="all")
        combined = pd.concat(
            [hist_part, tgt_part],
            ignore_index=True,
            sort=False,
        )
        combined = combined.reindex(columns=sorted(combined.columns))
    prepared = _prepare_training(combined, load_source_col="load_feature_source")
    return prepared[prepared["_is_target_row"] == 1].copy()


def _normalize_block_vector(values: Any, length: int = _BLOCK_COUNT, fill_value: float = 0.0) -> np.ndarray:
    vec = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    if vec.size == 0:
        return np.full(length, float(fill_value), dtype=float)
    if vec.size < length:
        pad_value = float(vec[-1]) if vec.size else float(fill_value)
        vec = np.pad(vec, (0, length - vec.size), mode="constant", constant_values=pad_value)
    elif vec.size > length:
        vec = vec[:length]
    return _fill_nonfinite_vector(vec, fill_value=fill_value)


def _fill_nonfinite_vector(values: Any, fill_value: float = 0.0) -> np.ndarray:
    """Return a finite vector, interpolating isolated NaN/inf gaps when possible."""
    vec = np.asarray(values if values is not None else [], dtype=float).reshape(-1).copy()
    if vec.size == 0:
        return vec.astype(float, copy=False)
    finite = np.isfinite(vec)
    if finite.all():
        return vec.astype(float, copy=False)
    if finite.any():
        idx = np.arange(vec.size, dtype=float)
        vec[~finite] = np.interp(idx[~finite], idx[finite], vec[finite])
    else:
        vec[:] = float(fill_value)
    return vec.astype(float, copy=False)


def _build_synthetic_forecast_day(
    template_df: pd.DataFrame,
    target_date: str,
    forecast: Any,
    zero_actuals: bool = False,
) -> pd.DataFrame:
    day = _coerce_day_to_96_blocks(template_df, fill_load=False)
    if day.empty:
        raise ValueError(f"Cannot synthesise day {target_date}: template day is empty.")

    synthetic = day.copy()
    synthetic["date"] = str(target_date)
    synthetic["time_block"] = np.arange(1, _BLOCK_COUNT + 1, dtype=int)
    synthetic["total_drawal"] = _normalize_block_vector(
        np.zeros(_BLOCK_COUNT, dtype=float) if zero_actuals else forecast,
        length=_BLOCK_COUNT,
        fill_value=0.0,
    )
    return synthetic


def _apply_seam_continuity(
    forecast: Any,
    previous_terminal_mw: float,
    window: int = 8,
) -> Tuple[np.ndarray, float, float]:
    adjusted = _normalize_block_vector(forecast, length=_BLOCK_COUNT, fill_value=0.0)
    gap_before = float(adjusted[0] - previous_terminal_mw)
    if adjusted.size == 0 or window <= 0:
        return adjusted, gap_before, gap_before

    taper = min(int(window), adjusted.size)
    if taper <= 1:
        adjusted[0] = float(previous_terminal_mw)
        return adjusted, gap_before, float(adjusted[0] - previous_terminal_mw)

    delta = float(previous_terminal_mw - adjusted[0])
    if abs(delta) < 1e-9:
        return adjusted, gap_before, gap_before

    taper_weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(taper, dtype=float) / float(taper - 1)))
    adjusted[:taper] = adjusted[:taper] + (delta * taper_weights)
    gap_after = float(adjusted[0] - previous_terminal_mw)
    return adjusted, gap_before, gap_after


def _coerce_day_to_96_blocks(day_df: pd.DataFrame, fill_load: bool = False) -> pd.DataFrame:
    if day_df is None or day_df.empty:
        return day_df.copy() if isinstance(day_df, pd.DataFrame) else pd.DataFrame()

    work = day_df.copy()
    date_value = str(work["date"].astype(str).iloc[0]) if "date" in work.columns and not work.empty else None
    work["time_block"] = pd.to_numeric(work.get("time_block"), errors="coerce")
    work = work.dropna(subset=["time_block"])
    work["time_block"] = work["time_block"].astype(int)
    work = work[work["time_block"].between(1, _BLOCK_COUNT)]
    if work.empty:
        return work

    numeric_cols = [col for col in work.columns if col != "date"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("time_block", as_index=False)[numeric_cols].mean()
    grouped = grouped.set_index("time_block").reindex(range(1, _BLOCK_COUNT + 1))
    grouped.index.name = "time_block"
    grouped = grouped.reset_index()
    if date_value is not None:
        grouped["date"] = date_value

    for col in grouped.columns:
        if col in {"date", "time_block", "total_drawal"}:
            continue
        grouped[col] = grouped[col].interpolate(limit_direction="both").ffill().bfill()

    if "total_drawal" in grouped.columns and fill_load:
        grouped["total_drawal"] = grouped["total_drawal"].interpolate(limit_direction="both").ffill().bfill()

    ordered_cols = [col for col in day_df.columns if col in grouped.columns]
    remaining_cols = [col for col in grouped.columns if col not in ordered_cols]
    return grouped[ordered_cols + remaining_cols]


def _ensure_time_block_1_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `time_block` uses 1–96 indexing.

    Our historical CSVs often store blocks as 0–95, but most of the pipeline
    assumes 1–96 (and drops blocks outside that range). Shifting early prevents
    silent 1-block loss and misaligned features/curves.
    """
    if df is None or df.empty or "time_block" not in df.columns:
        return df

    work = df.copy()
    tb = pd.to_numeric(work["time_block"], errors="coerce")
    if not tb.notna().any():
        return work

    has_zero = bool((tb == 0).any())
    has_96 = bool((tb == 96).any())
    try:
        max_tb = int(tb.max())
    except Exception:
        max_tb = None

    if has_zero and max_tb == 95 and not has_96:
        work["time_block"] = (tb + 1).astype(int)
    else:
        work["time_block"] = tb.astype(int)
    return work


def _prepare_training(df: pd.DataFrame, load_source_col: str = "total_drawal") -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    cols_to_clean = [
        "temperature", "humidity", "precipitation",
        "apparent_temperature", "cloud_cover", "sunshine_duration",
        "direct_radiation", "wind_speed_10m", "time_block",
    ]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill().fillna(0.0)

    if "total_drawal" in df.columns:
        df["total_drawal"] = pd.to_numeric(df["total_drawal"], errors="coerce")

    if load_source_col in df.columns:
        load_source = pd.to_numeric(df[load_source_col], errors="coerce")
    elif "total_drawal" in df.columns:
        load_source = pd.to_numeric(df["total_drawal"], errors="coerce")
    else:
        load_source = pd.Series(0.0, index=df.index, dtype=float)

    if load_source_col == "total_drawal":
        load_source = load_source.interpolate(limit_direction="both").ffill().bfill()
    else:
        fallback = pd.to_numeric(df.get("total_drawal"), errors="coerce") if "total_drawal" in df.columns else pd.Series(np.nan, index=df.index)
        load_source = load_source.where(load_source.notna(), fallback)
        load_source = load_source.ffill().bfill()

    df["load_feature_source"] = load_source.fillna(0.0)
    df = df.sort_values(["date", "time_block"]).reset_index(drop=True)

    # --- V2 UPGRADE: Nonlinear & Lag Features ---
    if "temperature" in df.columns:
        # Regional CDD/HDD bases (default north)
        _cdd_base = 24.0
        _hdd_base = 15.0
        try:
            _bases = DEFAULT_CONFIG.get("cdd_hdd_bases", {})
            # Try to detect climate region from data context
            for _cr in ("north", "south", "west", "east", "central"):
                if _cr in _bases:
                    _cdd_base = _bases.get(_cr, {}).get("cdd", _cdd_base)
                    _hdd_base = _bases.get(_cr, {}).get("hdd", _hdd_base)
                    break  # Use first available; caller can override via config
        except Exception:
            pass
        df["CDD"] = (df["temperature"] - _cdd_base).clip(lower=0)
        df["HDD"] = (_hdd_base - df["temperature"]).clip(lower=0)
        # 3h and 6h rolling averages
        df["temp_roll_3h"] = df["temperature"].rolling(window=12, min_periods=1).mean()
        df["temp_roll_6h"] = df["temperature"].rolling(window=24, min_periods=1).mean()

        # ── Thermal inertia: accumulated cooling-degree-hours (CDH) ──
        # After 3 consecutive hot days buildings retain heat; evening load stays elevated.
        # 24h (1 day), 48h (2 day), 72h (3 day) rolling sum of CDD.
        df["cdh_24h"] = df["CDD"].rolling(window=96, min_periods=1).sum()
        df["cdh_48h"] = df["CDD"].rolling(window=192, min_periods=1).sum()
        df["cdh_72h"] = df["CDD"].rolling(window=288, min_periods=1).sum()

        # Wet-bulb globe temperature (WBGT) — better predictor of cooling load
        # in humid Indian summers than dry-bulb or apparent temperature.
        # Simplified Stull (2011) approximation: WBGT ≈ f(T, RH)
        if "humidity" in df.columns:
            T = df["temperature"].values
            RH = df["humidity"].clip(0, 100).values
            # Stull wet-bulb approximation
            wb = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
                 np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
                 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
            # WBGT ≈ 0.7 * wet_bulb + 0.2 * globe + 0.1 * dry_bulb
            # Simplified (no globe temp): WBGT ≈ 0.7 * wb + 0.3 * T
            df["wbgt"] = 0.7 * wb + 0.3 * T
        else:
            df["wbgt"] = df["temperature"]

    # ── Solar generation proxy ──
    # Odisha has growing rooftop solar. Grid drawal = demand - solar.
    # Cloud cover and radiation directly affect the residual the model predicts.
    # Estimated solar index: high radiation + low cloud → high solar output → lower grid load.
    if "direct_radiation" in df.columns and "cloud_cover" in df.columns:
        rad = pd.to_numeric(df["direct_radiation"], errors="coerce").fillna(0).clip(lower=0)
        cloud = pd.to_numeric(df["cloud_cover"], errors="coerce").fillna(50).clip(0, 100)
        # Normalised solar capacity factor (0-1 scale, ~peak at clear-sky noon)
        df["solar_index"] = (rad / max(rad.max(), 1)) * (1 - cloud / 100)
        # Estimated solar MW offset (rough proxy, scales with radiation)
        df["solar_proxy_mw"] = rad * (1 - cloud / 100) * 0.001  # arbitrary scaling
    elif "direct_radiation" in df.columns:
        rad = pd.to_numeric(df["direct_radiation"], errors="coerce").fillna(0).clip(lower=0)
        df["solar_index"] = rad / max(rad.max(), 1)
        df["solar_proxy_mw"] = rad * 0.001

    # Lagged load features (strongly predictive for short-term autocorrelation)
    if "load_feature_source" in df.columns:
        load_series = df["load_feature_source"]
        df["lag_1"] = load_series.shift(_BLOCK_COUNT)
        df["lag_7"] = load_series.shift(_BLOCK_COUNT * 7)
        df["lag_block_1"] = load_series.shift(1)
        df["lag_block_4"] = load_series.shift(4)
        df["rolling_4"] = load_series.shift(1).rolling(window=4, min_periods=1).mean()
        df["rolling_12"] = load_series.shift(1).rolling(window=12, min_periods=1).mean()

        df["load_lag_1d"] = df["lag_1"]
        df["load_lag_2d"] = load_series.shift(_BLOCK_COUNT * 2)
        df["load_lag_3d"] = load_series.shift(_BLOCK_COUNT * 3)
        df["load_lag_7d"] = df["lag_7"]
        df["load_rolling_7d"] = df.groupby("time_block")["load_feature_source"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        # 3-day load trend: slope of last 3 days' daily average (v3.0)
        daily_avg = df.groupby("date")["load_feature_source"].transform("mean")
        df["load_trend_3d"] = (
            daily_avg - daily_avg.shift(_BLOCK_COUNT * 3)
        )

    # ── Multiplicative weather × load interaction features (v3.0) ────
    # Weather impact on load is fundamentally multiplicative — a 3°C rise
    # adds more MW when base load is 5000 vs 2000. These features let
    # tree models learn percentage-based weather sensitivity.
    if "total_drawal" in df.columns and "temperature" in df.columns:
        lag_1d = df.get("load_lag_1d", df["total_drawal"])
        roll_7d = df.get("load_rolling_7d", df["total_drawal"])
        cdd = df.get("CDD", pd.Series(0, index=df.index))
        hum = df.get("humidity", pd.Series(50, index=df.index))
        wbgt = df.get("wbgt", df["temperature"])

        df["load_x_cdd"] = lag_1d * cdd                    # AC impact scales with load
        df["load_x_humidity"] = lag_1d * (hum / 100.0)     # humid-heat compound
        df["cdd_squared"] = cdd ** 2                        # exponential AC response
        df["wbgt_x_load"] = wbgt * roll_7d                  # heat stress × load level
        # 3-day temperature momentum (thermal mass buildup)
        if "temperature" in df.columns:
            df["temp_momentum_3d"] = df["temperature"] - df["temperature"].shift(288).fillna(df["temperature"])

        # Previous-night anomaly flag: set to 1 when yesterday's night blocks (1-32)
        # averaged >15% below their own 7-day per-block rolling mean. This lets the
        # ML model discount lag_1 and load_rolling_7d features on days after a night dip,
        # preventing propagation of the dip into today's forecast.
        _night_mask = df["time_block"].between(1, 32)
        _night_mean_yesterday = (
            df[_night_mask]
            .groupby("date")["load_feature_source"]
            .transform("mean")
            .where(_night_mask, np.nan)
            .shift(_BLOCK_COUNT)
        )
        _night_rolling = (
            df.groupby("time_block")["load_feature_source"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=2).mean())
        )
        _night_rolling_mean_prev = (
            df[_night_mask]
            .groupby("date")["load_feature_source"]
            .transform("mean")
            .where(_night_mask, np.nan)
            .rolling(7, min_periods=2).mean()
            .shift(_BLOCK_COUNT)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            _night_dev = (_night_mean_yesterday - _night_rolling_mean_prev).abs() / _night_rolling_mean_prev.replace(0, np.nan)
        df["prev_night_dip_flag"] = (_night_dev > 0.15).astype(float).fillna(0.0)

        # temp_quartic: captures supralinear AC load response above 31°C where
        # overforecasting is documented (32-34°C band). Scaled by 1e-6 to keep
        # range comparable to temp_squared (~900 range).
        df["temp_quartic"] = (df["temperature"] ** 4) / 1e6

        # ramp_heat_interaction: temperature signal active only during the afternoon
        # ramp window (blocks 48-68, 12:00-17:00 IST). Allows the model to learn a
        # segment-specific heat discount rather than a global temperature coefficient,
        # directly countering the systematic ramp-period overforecast.
        if "time_block" in df.columns:
            _is_ramp = df["time_block"].between(48, 68).astype(float)
        else:
            _is_ramp = pd.Series(0.0, index=df.index)
        df["ramp_heat_interaction"] = df["temperature"] * _is_ramp

    df = _add_calendar(df)
    df = df.reset_index(drop=True)
    tb_feats = _timeblock_features(df["time_block"]).reset_index(drop=True)
    df = pd.concat([df, tb_feats], axis=1)
    df["season_idx"] = df["season"].apply(lambda s: SEASON_ORDER.index(s) if s in SEASON_ORDER else 0)
    df = _add_time_features(df)

    # Optional external feature module hook (generic).
    df = _apply_external_features(df)
    # Ensure extra numeric features are finite if we plan to use them.
    enabled, _path, include_extra_numeric = _external_features_settings(DEFAULT_CONFIG)
    if enabled and include_extra_numeric:
        for col in _infer_extra_numeric_feature_columns(df):
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            df[col] = s.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)

    return df


def _best_baseline_window(df: pd.DataFrame, target_date: str, candidates: List[int], backtest_days: int, weather_weight: float = 0.0) -> Tuple[int, Optional[float]]:
    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        return candidates[0], None
    idx = dates.index(target_date)
    history = dates[:idx]
    if len(history) < 3:
        return candidates[0], None

    recent = history[-backtest_days:]
    best_w = candidates[0]
    best_mape = None
    best_score = None

    target_df = df[df["date"] == target_date]
    if not target_df.empty:
        tgt_weather = target_df[["temperature", "humidity", "precipitation"]].mean()
        target_weather_delta = float(tgt_weather.abs().mean())
    else:
        tgt_weather = None
        target_weather_delta = 0.0

    # Identify synthetic rows (forecast stand-ins, not real actuals) — exclude
    # them as validation targets for MAPE since their "actuals" are the forecast.
    _synthetic_dates: set = set()
    if "_is_synthetic" in df.columns:
        _synthetic_dates = set(df[df["_is_synthetic"] == 1]["date"].astype(str).unique())

    for w in candidates:
        errors = []
        for d in recent:
            if d in _synthetic_dates:
                continue  # skip — "actual" is a forecast value, not real
            di = history.index(d)
            start = max(0, di - w)
            window_days = history[start:di]
            if not window_days:
                continue
            baseline = df[df["date"].isin(window_days)].groupby("time_block")["total_drawal"].mean().reindex(range(1, 97))
            actual = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            base_vals = baseline.to_numpy()
            if len(actual) != 96:
                continue
            mape = np.mean(np.abs(actual - base_vals) / np.maximum(actual, 1e-6)) * 100
            errors.append(mape)
        if not errors:
            continue
        avg_mape = float(np.mean(errors))
        weather_diff = 0.0
        if tgt_weather is not None:
            window_weather = df[df["date"].isin(history[max(0, idx - w):idx])][["temperature", "humidity", "precipitation"]].mean()
            weather_diff = float((tgt_weather - window_weather).abs().mean())
        delta_scale = min(1.0, target_weather_delta / 10.0) if target_weather_delta else 0.0
        weight = float(weather_weight) * (1.0 + delta_scale)
        score = avg_mape + (weight * weather_diff)
        if best_score is None or score < best_score:
            best_score = score
            best_mape = avg_mape
            best_w = w
    return best_w, best_mape


def _zscore_or_minmax(series: pd.Series) -> pd.Series:
    """Standardise a feature for similarity scoring.

    Uses z-score by default; if std == 0 (degenerate column), falls back to
    min-max in [0, 1]; if max == min as well, returns zeros.  Output is
    unit-free so different features (degC, %, mm, MW) can be linearly combined.
    """
    s = pd.to_numeric(series, errors="coerce")
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mu = float(finite.mean())
    sd = float(finite.std(ddof=0))
    if sd > 1e-9:
        out = (s - mu) / sd
    else:
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo > 1e-9:
            out = (s - lo) / (hi - lo)
        else:
            out = pd.Series(0.0, index=s.index)
    return out.fillna(0.0)


def _compute_residual_bias(
    df: pd.DataFrame,
    dates: List[str],
    idx: int,
    fast_days: int = 5,
    slow_days: int = 60,
) -> np.ndarray:
    """Unified fast+slow residual bias correction (FIX 4).

    Combines a fast recent component (last 5 days) and a slow seasonal
    component (last 60 days) with MAD-based clipping to prevent noise
    from a single bad day corrupting the correction.

    Returns a 96-block additive correction array.
    """
    all_days_needed = max(slow_days, fast_days)
    window_dates = dates[max(0, idx - all_days_needed):idx]
    residuals_list: List[np.ndarray] = []

    for d in window_dates:
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue
        actual = (
            day_df.groupby("time_block")["total_drawal"]
            .mean()
            .reindex(range(1, 97))
        )
        d_idx = dates.index(d)
        prior = dates[max(0, d_idx - 7):d_idx]
        if not prior:
            continue
        prior_df = df[df["date"].isin(prior)]
        if prior_df.empty:
            continue
        rolling = (
            prior_df.groupby("time_block")["total_drawal"]
            .mean()
            .reindex(range(1, 97))
        )
        resid = (actual - rolling).to_numpy(dtype=float)

        # Exclude load-shedding / grid-outage days from bias pool.
        # A >30% drop within any 1-hour window (4 blocks) is not a model error — it's
        # an external event that would corrupt the residual estimate if included.
        _act_arr = actual.to_numpy(dtype=float)
        _valid_act = np.isfinite(_act_arr) & (_act_arr > 0)
        _anomalous = False
        if _valid_act.sum() >= 8:
            for _bi in range(4, len(_act_arr)):
                if _valid_act[_bi] and _valid_act[_bi - 4] and _act_arr[_bi - 4] > 500:
                    if _act_arr[_bi] < 0.70 * _act_arr[_bi - 4]:
                        _anomalous = True
                        break
        if _anomalous:
            continue

        # Also exclude days where hub-height wind exceeded SLDC protection threshold —
        # those are supply-cut events (feeder trips), not demand errors.
        _day_wind80 = float(day_df["wind_speed_80m"].max()) if "wind_speed_80m" in day_df.columns else 0.0
        if _day_wind80 >= 65.0:
            continue

        if np.isfinite(resid).any():
            residuals_list.append(resid)

    if not residuals_list:
        return np.zeros(96, dtype=float)

    residuals = np.vstack(residuals_list)  # (n_days, 96)

    fast_slice = residuals[-fast_days:] if len(residuals) >= fast_days else residuals
    slow_slice = residuals[-slow_days:] if len(residuals) >= slow_days else residuals

    with np.errstate(invalid="ignore"):
        fast = np.where(
            np.isfinite(fast_slice).any(axis=0),
            np.nanmean(fast_slice, axis=0),
            0.0,
        )
        slow = np.where(
            np.isfinite(slow_slice).any(axis=0),
            np.nanmean(slow_slice, axis=0),
            0.0,
        )

    # MAD clipping: prevents a noisy day from dominating the fast component
    flat = residuals.flatten()
    flat = flat[np.isfinite(flat)]
    if len(flat) > 0:
        mad = float(np.median(np.abs(flat - np.median(flat))))
        if mad > 0:
            fast = np.clip(fast, -3.0 * mad, 3.0 * mad)

    return 0.4 * fast + 0.1 * slow


def _valid_actual_day_vector(df: pd.DataFrame, date_value: str) -> Optional[np.ndarray]:
    """Return a complete 96-block actual-load vector for a non-synthetic day."""
    if df is None or df.empty or "date" not in df.columns or "time_block" not in df.columns:
        return None
    day = df[df["date"].astype(str) == str(date_value)].copy()
    if day.empty:
        return None
    if "_is_synthetic" in day.columns and pd.to_numeric(day["_is_synthetic"], errors="coerce").fillna(0).max() >= 1:
        return None
    if "total_drawal" not in day.columns:
        return None
    vec = (
        day.groupby("time_block")["total_drawal"]
        .mean()
        .reindex(range(1, _BLOCK_COUNT + 1))
        .to_numpy(dtype=float)
    )
    valid = np.isfinite(vec) & (vec > 50.0)
    if int(valid.sum()) < _BLOCK_COUNT:
        return None
    return vec


def _recent_horizon_calibration(
    df: pd.DataFrame,
    target_date: str,
    forecast: np.ndarray,
    actual_blocks: int,
    actual_partial: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Calibrate final forecast level/shape from pre-target actual history.

    This is intentionally a post-model layer: it does not learn from target-day
    future actuals, but it corrects common operational errors seen in backtests:
    wrong daily level, weak afternoon blocks, and T+2 regime drift.
    """
    cal_cfg = cfg.get("horizon_calibration", {})
    if not isinstance(cal_cfg, dict):
        cal_cfg = {}
    if not bool(cal_cfg.get("enabled", True)):
        return forecast, np.zeros(_BLOCK_COUNT, dtype=float), {"enabled": False}

    fc = _as_96_vector(forecast).astype(float)
    correction = np.zeros(_BLOCK_COUNT, dtype=float)
    horizon = str(cfg.get("forecast_horizon", "t1")).lower().strip()
    accuracy_cfg = cfg.get("accuracy_corrections", {}) if isinstance(cfg.get("accuracy_corrections", {}), dict) else {}

    try:
        work = df.copy()
        work["date"] = work["date"].astype(str)
        all_dates = sorted(work["date"].dropna().unique().tolist())
        hist_dates = [d for d in all_dates if d < str(target_date)]
        hist_vectors: List[Tuple[str, np.ndarray]] = []
        for d in hist_dates[-45:]:
            vec = _valid_actual_day_vector(work, d)
            if vec is not None:
                hist_vectors.append((d, vec))
        if len(hist_vectors) < 3:
            return fc, correction, {"enabled": True, "applied": False, "reason": "insufficient_history"}

        last_vec = hist_vectors[-1][1]
        last3 = np.mean(np.vstack([v for _, v in hist_vectors[-3:]]), axis=0)
        last7 = np.mean(np.vstack([v for _, v in hist_vectors[-min(7, len(hist_vectors)):]]), axis=0)

        target_ts = pd.Timestamp(str(target_date)[:10])
        same_wd = [v for d, v in hist_vectors if pd.Timestamp(d).dayofweek == target_ts.dayofweek]
        same_wd_recent = np.mean(np.vstack(same_wd[-4:]), axis=0) if same_wd else last7

        if horizon == "t2":
            # T+2 has no live actual anchor and historically drifts in daily level.
            # Sunday (6) and Monday (0) have higher same-weekday anchor noise — use 0.30 weight
            # to avoid cascading errors when the prior same-weekday was anomalous.
            _t2_same_wd_w = 0.30 if target_ts.dayofweek in (0, 6) else 0.40
            anchor = (0.30 * last_vec) + (0.30 * last3) + (_t2_same_wd_w * same_wd_recent) + ((0.40 - _t2_same_wd_w) * last7)
            ratio_cap = float(cal_cfg.get("t2_ratio_cap", 0.08))
            blend = float(cal_cfg.get("t2_blend", 0.68))
            segment_blend = float(cal_cfg.get("t2_segment_blend", 0.40))
        else:
            anchor = (0.40 * last_vec) + (0.30 * last3) + (0.30 * same_wd_recent)
            ratio_cap = float(cal_cfg.get("t1_ratio_cap", 0.05))
            blend = float(cal_cfg.get("t1_blend", 0.48))
            segment_blend = float(cal_cfg.get("t1_segment_blend", 0.28))

        # If live actuals exist, they are the strongest non-leaky signal.
        live_ratio = 1.0
        live_bias = 0.0
        if actual_blocks >= 4 and actual_partial.size >= actual_blocks:
            recent_n = min(int(cal_cfg.get("live_anchor_blocks", 24)), actual_blocks)
            act_tail = np.asarray(actual_partial[-recent_n:], dtype=float)
            fc_tail = fc[actual_blocks - recent_n:actual_blocks]
            valid = np.isfinite(act_tail) & np.isfinite(fc_tail) & (fc_tail > 100.0) & (act_tail > 50.0)
            if int(valid.sum()) >= 3:
                ratios = act_tail[valid] / fc_tail[valid]
                ema_alpha = float(cal_cfg.get("live_ratio_ema_alpha", 0.50))
                ema_ratio = float(ratios[0])
                for ratio in ratios[1:]:
                    ema_ratio = (ema_alpha * float(ratio)) + ((1.0 - ema_alpha) * ema_ratio)
                live_low, live_high = (
                    (0.92, 1.08) if horizon == "t2" else (0.95, 1.05)
                )
                live_ratio = float(np.clip(ema_ratio, live_low, live_high))
                live_bias = float(np.median(act_tail[valid] - fc_tail[valid]))
                ratio_cap = max(ratio_cap, live_high - 1.0)
                blend = max(blend, float(cal_cfg.get("live_ratio_weight", 0.55)))
                segment_blend = min(max(segment_blend, 0.20), float(cal_cfg.get("live_segment_blend_cap", 0.30)))

        fc_energy = float(np.sum(fc) * 0.25)
        anchor_energy = float(np.sum(anchor) * 0.25)
        raw_ratio = (anchor_energy / fc_energy) if fc_energy > 100.0 else 1.0
        anchor_ratio = float(np.clip(raw_ratio, 1.0 - ratio_cap, 1.0 + ratio_cap))
        # Surgical hierarchy: horizon calibration is only the EMA live ratio.
        # Historical anchor ratios remain diagnostic, not an extra correction.
        level_ratio = float(live_ratio if actual_blocks >= 4 else 1.0)

        # Block-window correction, smoothed to avoid jagged copied history.
        segment_delta = anchor - fc
        kernel = np.ones(7, dtype=float) / 7.0
        segment_delta = np.convolve(np.pad(segment_delta, (3, 3), mode="edge"), kernel, mode="valid")
        cap = np.maximum(np.abs(fc) * float(cal_cfg.get("mw_cap_pct", 0.10)), float(cal_cfg.get("mw_cap_floor", 250.0)))
        segment_delta = np.clip(segment_delta, -cap, cap)

        future_start = int(np.clip(actual_blocks, 0, _BLOCK_COUNT))
        for i in range(future_start, _BLOCK_COUNT):
            live_decay = 1.0
            level_corr = fc[i] * (level_ratio - 1.0)
            shape_corr = 0.0
            if actual_blocks >= 4 and bool(cal_cfg.get("additive_live_bias_enabled", False)):
                bias_weight = float(cal_cfg.get("live_bias_weight", 0.30))
                bias_corr = live_bias * bias_weight * live_decay
            else:
                bias_corr = 0.0
            correction[i] = level_corr + shape_corr + bias_corr

        calibrated = np.maximum(fc + correction, 0.0)
        calibrated[:future_start] = fc[:future_start]
        return calibrated, correction, {
            "enabled": True,
            "applied": bool(np.any(np.abs(correction[future_start:]) > 1e-6)),
            "horizon": horizon,
            "level_ratio_raw": round(float(raw_ratio), 5),
            "anchor_ratio_diagnostic": round(float(anchor_ratio), 5),
            "level_ratio_applied": round(float(level_ratio), 5),
            "blend": 1.0 if actual_blocks >= 4 else 0.0,
            "segment_blend": round(float(segment_blend), 3),
            "shape_correction_enabled": False,
            "live_ratio": round(float(live_ratio), 5),
            "live_ratio_method": "ema_3_block",
            "live_ratio_clamp": [0.92, 1.08] if horizon == "t2" else [0.95, 1.05],
            "decay_profile": {
                "enabled": False,
                "reason": "live EMA ratio is applied without extra decay",
            },
            "accuracy_layer_enabled": bool(accuracy_cfg),
            "live_bias_mw": round(float(live_bias), 2),
            "additive_live_bias_enabled": bool(cal_cfg.get("additive_live_bias_enabled", False)),
            "additive_live_bias_weight": float(cal_cfg.get("live_bias_weight", 0.30)),
            "mean_correction_mw": round(float(np.mean(correction[future_start:])), 2) if future_start < _BLOCK_COUNT else 0.0,
            "max_abs_correction_mw": round(float(np.max(np.abs(correction[future_start:]))), 2) if future_start < _BLOCK_COUNT else 0.0,
        }
    except Exception as exc:
        logger.debug("Horizon calibration skipped: %s", exc)
        return fc, correction, {"enabled": True, "applied": False, "reason": str(exc)}


def _apply_ramp_limit(
    forecast: np.ndarray,
    *,
    actual_blocks: int = 0,
    max_ramp_mw: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Constrain future block-to-block ramp inside the forecast, not only at export."""
    out = _as_96_vector(forecast).astype(float)
    applied = np.zeros(_BLOCK_COUNT, dtype=float)
    start = int(np.clip(int(actual_blocks), 0, _BLOCK_COUNT))
    if start >= _BLOCK_COUNT:
        return out, applied, {"enabled": True, "applied": False, "max_ramp_mw": float(max_ramp_mw)}
    for i in range(max(1, start), _BLOCK_COUNT):
        prev = out[i - 1]
        before = out[i]
        out[i] = float(np.clip(before, prev - float(max_ramp_mw), prev + float(max_ramp_mw)))
        applied[i] = out[i] - before
    return out, applied, {
        "enabled": True,
        "applied": bool(np.any(np.abs(applied[start:]) > 1e-6)),
        "max_ramp_mw": float(max_ramp_mw),
        "max_abs_applied_mw": round(float(np.max(np.abs(applied[start:]))), 3) if start < _BLOCK_COUNT else 0.0,
        "applied_mw": applied.tolist(),
    }


def _similar_day_baseline(df: pd.DataFrame, target_date: str, cfg: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    df = _add_calendar(df)
    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        # Find latest robust date
        latest_robust = dates[-1] if dates else target_date
        for d in reversed(dates):
            day_df = df[df["date"] == d]
            if day_df["time_block"].nunique() >= 96 and day_df["total_drawal"].mean() > cfg.get("min_valid_load_mw", 100):
                latest_robust = d
                break
        # Keep target_date as-is for season/day_type derivation if it's a future
        # date; only swap when it's missing AND we have no calendar info yet.
        target_for_idx = latest_robust
    else:
        target_for_idx = target_date
    idx = dates.index(target_for_idx)

    # ── Build THREE pools ─────────────────────────────────────────────────────
    # 1) recent_7    : last 7 calendar days before target  → baseline
    # 2) current_month: same calendar month, all prior years, ±seasonal_window
    #                   around the same month-day             → baseline
    # 3) last_year_obs: ±7 days around same date exactly 1 year ago
    #                   → observation / metadata only, NOT blended into baseline
    seasonal_window = int(cfg.get("seasonal_window_days", 21))

    try:
        target_ts = pd.Timestamp(target_date)
    except Exception:
        target_ts = pd.Timestamp(target_for_idx)

    recent_7_pool: set = set(dates[max(0, idx - 7):idx])

    current_month_pool: set = set()
    last_year_obs_pool: set = set()
    if pd.notna(target_ts):
        target_year  = int(target_ts.year)
        target_month = int(target_ts.month)
        target_md    = (target_month, int(target_ts.day))
        last_year    = target_year - 1

        for d in dates:
            try:
                dt = pd.Timestamp(d)
            except Exception:
                continue
            if pd.isna(dt) or int(dt.year) >= target_year:
                continue
            # Last-year observation window: ±7 days around same date in prior year
            if int(dt.year) == last_year:
                try:
                    anchor_ly = pd.Timestamp(year=last_year, month=target_md[0],
                                             day=min(target_md[1], 28))
                except Exception:
                    anchor_ly = None
                if anchor_ly is not None and abs((dt - anchor_ly).days) <= 7:
                    last_year_obs_pool.add(d)

            # Current-month pool: same month, prior years, ±seasonal_window days
            if int(dt.month) == target_month:
                try:
                    anchor = pd.Timestamp(year=int(dt.year), month=target_md[0],
                                          day=min(target_md[1], 28))
                except Exception:
                    continue
                if abs((dt - anchor).days) <= seasonal_window:
                    current_month_pool.add(d)

    # Baseline candidates = recent 7 days + current-month prior-year days
    candidates = sorted(
        (recent_7_pool | current_month_pool)
        - {target_date, target_for_idx}
    )
    if not candidates:
        candidates = dates[:idx]

    # Weekend / holiday separate baseline pools (A6):
    # Weekend and holiday days have distinctly different load profiles.
    # Filter candidates to same day-type if target is a weekend/holiday
    # and sufficient same-type candidates exist.
    _target_is_weekend = pd.notna(target_ts) and target_ts.weekday() >= 5
    _target_day_type_str = (
        df[df["date"] == target_date]["day_type"].iloc[0]
        if not df[df["date"] == target_date].empty and "day_type" in df.columns
        else "weekday"
    )
    _target_is_holiday = "holiday" in str(_target_day_type_str).lower()
    _weekend_baseline_days = int(cfg.get("weekend_baseline_days", 14))

    if _target_is_holiday:
        # Holiday pool: only same-type holiday candidates
        _holiday_candidates = [
            d for d in candidates
            if "holiday" in str(
                df[df["date"] == d]["day_type"].iloc[0]
                if not df[df["date"] == d].empty and "day_type" in df.columns else ""
            ).lower()
        ]
        if len(_holiday_candidates) >= 3:
            candidates = _holiday_candidates
            logger.debug("Holiday baseline pool: %d days", len(candidates))
    elif _target_is_weekend:
        # Weekend pool: filter to same weekday (Sat or Sun) from recent N weeks
        _target_weekday = target_ts.weekday()
        _all_dates_df = pd.to_datetime(pd.Series(dates), errors="coerce")
        _weekend_dates = set(
            str(d.date()) for d in _all_dates_df
            if pd.notna(d) and d.weekday() == _target_weekday
            and pd.Timestamp(d) < target_ts
        )
        # Limit to last weekend_baseline_days same-weekday occurrences
        _sorted_wd = sorted(_weekend_dates, reverse=True)[:_weekend_baseline_days]
        _weekend_candidates = [d for d in candidates if d in set(_sorted_wd)]
        if len(_weekend_candidates) >= 4:
            candidates = _weekend_candidates
            logger.debug(
                "Weekend baseline pool (%s): %d days",
                ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][_target_weekday], len(candidates)
            )

    # Exclude synthetic days
    if "_is_synthetic" in df.columns:
        _synth_set = set(df[df["_is_synthetic"] == 1]["date"].astype(str).unique())
        candidates      = [d for d in candidates if d not in _synth_set]
        last_year_obs_pool -= _synth_set

    # Exclude anomaly days: days where daily load deviated > ±anomaly_exclusion_pct
    # from their own trailing-7-day mean corrupt the baseline pool (r=-0.72 with error).
    _anomaly_exclusion_pct = float(cfg.get("anomaly_exclusion_pct", 0.06))
    if _anomaly_exclusion_pct > 0:
        try:
            _pool_dates = set(candidates) | last_year_obs_pool
            _day_means = (
                df[df["date"].isin(_pool_dates)]
                .groupby("date")["total_drawal"].mean()
                .sort_index()
            )
            _trailing7 = _day_means.rolling(7, min_periods=2).mean().shift(1)
            _dev_pct = ((_day_means - _trailing7) / _trailing7.replace(0, np.nan)).abs()
            _anomaly_dates = set(_dev_pct[_dev_pct > _anomaly_exclusion_pct].index.astype(str))
            # Guard: don't exclude everything — require at least 5 clean candidates
            # Also exclude the day after each anomaly (night-dip propagation)
            _also_exclude = set()
            for _ad in _anomaly_dates:
                try:
                    _also_exclude.add((pd.Timestamp(_ad) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                except Exception:
                    pass
            _all_exclude = _anomaly_dates | _also_exclude
            _clean_candidates = [d for d in candidates if d not in _all_exclude]
            if len(_clean_candidates) >= 5:
                candidates = _clean_candidates
                last_year_obs_pool -= _all_exclude
                logger.debug(
                    "Anomaly day exclusion: removed %d days (>%.0f%% dev from trailing-7) + %d post-anomaly days",
                    len(_anomaly_dates), _anomaly_exclusion_pct * 100, len(_also_exclude),
                )
        except Exception as _ae:
            logger.debug("Anomaly exclusion skipped: %s", _ae)

    target_df = df[df["date"] == target_date]
    target_season = target_df["season"].iloc[0] if not target_df.empty else "unknown"

    s_config = SEASONAL_WEATHER_INTERVALS.get(target_season, SEASONAL_WEATHER_INTERVALS["spring"])
    cdd_base = s_config["cdd_base"]
    hdd_base = s_config["hdd_base"]

    target_temp = target_df["temperature"].mean()
    target_hum  = target_df["humidity"].mean()
    target_cdd  = (target_df["temperature"] - cdd_base).clip(lower=0).mean()
    target_hdd  = (hdd_base - target_df["temperature"]).clip(lower=0).mean()
    target_rain = (target_df["precipitation"] > 0).any() if "precipitation" in target_df.columns else False
    _wind_col   = "wind_speed_10m" if "wind_speed_10m" in target_df.columns else None
    WIND_GUST_THRESHOLD = float(cfg.get("wind_gust_threshold_kmh", 30.0))
    target_windy = bool(_wind_col and (target_df[_wind_col] > WIND_GUST_THRESHOLD).any())
    target_day_type = target_df["day_type"].iloc[0] if not target_df.empty else "unknown"

    df_prep = df.copy()
    df_prep["CDD"] = (df_prep["temperature"] - cdd_base).clip(lower=0)
    df_prep["HDD"] = (hdd_base - df_prep["temperature"]).clip(lower=0)

    all_pool_dates = set(candidates) | last_year_obs_pool
    _agg_spec = dict(
        temp_mean=("temperature", "mean"),
        hum_mean=("humidity", "mean"),
        cdd_mean=("CDD", "mean"),
        hdd_mean=("HDD", "mean"),
        season=("season", "first"),
        day_type=("day_type", "first"),
        avg_load=("total_drawal", "mean"),
    )
    if "precipitation" in df_prep.columns:
        _agg_spec["rain_any"] = ("precipitation", lambda x: float((x > 0).any()))
    else:
        _agg_spec["rain_any"] = ("temperature", lambda x: 0.0)  # fallback zero
    if _wind_col:
        _agg_spec["wind_max"] = (_wind_col, "max")

    summary = (
        df_prep[df_prep["date"].isin(all_pool_dates)]
        .groupby("date")
        .agg(**_agg_spec)
        .reset_index()
    )
    if "wind_max" not in summary.columns:
        summary["wind_max"] = 0.0

    summary = summary[summary["avg_load"] > cfg.get("min_valid_load_mw", 100)].copy()
    if summary.empty:
        baseline = target_df.sort_values("time_block")["total_drawal"].to_numpy()
        return baseline, summary

    # ── Mark last-year observation rows (not used in baseline) ────────────────
    summary["is_last_year_obs"] = summary["date"].isin(last_year_obs_pool)

    # ── Hard weather filter for baseline candidates ────────────────────────────
    # If today is clear (no rain, no wind gust) → skip any candidate day that
    # had rain or wind gusts — their load shape reflects weather suppression
    # that isn't present today.
    baseline_candidate_set = set(candidates)
    if not target_rain and not target_windy:
        _weather_ok = (
            (~summary["rain_any"].astype(bool)) &
            (summary["wind_max"] <= WIND_GUST_THRESHOLD)
        )
        _clear_dates = set(summary.loc[_weather_ok, "date"].tolist())
        baseline_candidate_set = baseline_candidate_set & _clear_dates
        if not baseline_candidate_set:
            # All candidates were rainy/windy — relax and use all (prefer clear still)
            baseline_candidate_set = set(candidates)

    weather_signal_available = True
    if not np.isfinite(float(target_temp)) or not np.isfinite(float(target_hum)):
        weather_signal_available = False
    for _wc in ["temp_mean", "hum_mean", "cdd_mean", "hdd_mean", "rain_any"]:
        if _wc in summary.columns:
            summary[_wc] = pd.to_numeric(summary[_wc], errors="coerce")
    if not summary["temp_mean"].notna().any():
        weather_signal_available = False

    # Work only on baseline candidates for scoring/selection
    score_df = summary[summary["date"].isin(baseline_candidate_set)].copy()
    if score_df.empty:
        score_df = summary[~summary["is_last_year_obs"]].copy()

    target_dow = int(pd.Timestamp(target_date).dayofweek) if pd.notna(target_ts) else -1
    for _sdf in (summary, score_df):
        _sdf["same_season"]   = _sdf["season"].eq(target_season)
        _sdf["same_day_type"] = _sdf["day_type"].eq(target_day_type)
        _sdf["dow"]           = pd.to_datetime(_sdf["date"], errors="coerce").dt.dayofweek
        target_is_weekend = str(target_day_type).lower() in ("weekend", "saturday", "sunday")
        cand_is_weekend   = _sdf["day_type"].astype(str).str.lower().isin(["weekend", "saturday", "sunday"])
        _sdf["day_tier"] = np.where(
            _sdf["dow"].eq(target_dow), 0,
            np.where(_sdf["same_day_type"], 1,
                     np.where(cand_is_weekend == target_is_weekend, 2, 3)),
        )

    # Pool source tag: recent_7 days get a recency boost in weighting
    summary["pool"] = np.where(
        summary["date"].isin(recent_7_pool), "recent_7",
        np.where(summary["is_last_year_obs"], "last_year_obs", "current_month")
    )
    score_df["pool"] = np.where(score_df["date"].isin(recent_7_pool), "recent_7", "current_month")

    if weather_signal_available:
        score_df["temp_diff"]       = (score_df["temp_mean"] - target_temp).abs()
        score_df["hum_diff"]        = (score_df["hum_mean"]  - target_hum).abs()
        score_df["degree_day_diff"] = (
            (score_df["cdd_mean"] - target_cdd).abs() +
            (score_df["hdd_mean"] - target_hdd).abs()
        )
    else:
        score_df["temp_diff"] = score_df["hum_diff"] = score_df["degree_day_diff"] = 0.0

    score_df["temp_norm"] = _zscore_or_minmax(score_df["temp_diff"])
    score_df["hum_norm"]  = _zscore_or_minmax(score_df["hum_diff"])
    score_df["dd_norm"]   = _zscore_or_minmax(score_df["degree_day_diff"])

    sw     = cfg.get("similarity_weights", {"temp": 0.55, "humidity": 0.25, "rain": 0.20})
    w_temp = sw.get("temp", 0.55) * 0.7
    w_dd   = sw.get("temp", 0.55) * 0.3
    w_hum  = sw.get("humidity", 0.25)

    score_df["similarity_score"] = (
        w_temp * score_df["temp_norm"].abs()
        + w_dd  * score_df["dd_norm"].abs()
        + w_hum * score_df["hum_norm"].abs()
    )
    score_df["similarity_score"] = (
        pd.to_numeric(score_df["similarity_score"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1e6 if weather_signal_available else 0.0)
    )

    # ── Tiered selection: tier 0 (exact DOW) first, then widen ───────────────
    top_n = int(cfg.get("similar_days_top_n", 8))
    picks: List[str] = []
    for tier in (0, 1, 2):
        if len(picks) >= top_n:
            break
        slice_ = score_df[score_df["day_tier"] == tier].sort_values(
            ["similarity_score", "date"], ascending=[True, False]
        )
        picks.extend(slice_["date"].head(top_n - len(picks)).tolist())

    if not picks:
        return np.zeros(96, dtype=float), summary

    filtered = score_df[score_df["date"].isin(picks)].copy()
    filtered = filtered.sort_values(["day_tier", "similarity_score", "date"],
                                    ascending=[True, True, False])
    selected_dates = filtered["date"].tolist()

    # ── Level-correction scaling (anchor to recent 7-day mean load) ──────────
    # Exclude anomalous recent days so a single drastic day doesn't shift the
    # level anchor down (or up) and bias every similar-day curve.
    recent_load_level = 1.0
    if recent_7_pool:
        recent_df = df[df["date"].isin(recent_7_pool)]
        day_avgs  = recent_df.groupby("date")["total_drawal"].mean()
        if len(day_avgs) >= 3:
            _mu, _sd = day_avgs.mean(), day_avgs.std()
            if _sd > 0:
                clean_days = day_avgs[(day_avgs - _mu).abs() <= 1.8 * _sd]
                day_avgs = clean_days if len(clean_days) >= 2 else day_avgs
        recent_avg = float(day_avgs.mean()) if len(day_avgs) else 0.0
        if recent_avg > 100:
            recent_load_level = recent_avg
        # Track which recent days are clean — anomalous ones lose their recency boost
        _clean_recent = set(day_avgs.index.tolist())
    else:
        _clean_recent = set()

    similar_days_df = df[df["date"].isin(selected_dates)]

    # Tier-aware weighting + recency boost only for clean recent days
    tier_weights   = {0: 1.00, 1: 0.60, 2: 0.30, 3: 0.10}
    tier_lookup    = filtered.set_index("date")["day_tier"].to_dict() if "day_tier" in filtered.columns else {}
    pool_lookup    = filtered.set_index("date")["pool"].to_dict()   if "pool"     in filtered.columns else {}
    RECENCY_BOOST  = 1.30   # recent_7 days contribute 30% more weight (only if not anomalous)

    scaled_stack: List[np.ndarray] = []
    weight_stack: List[float]      = []
    scale_log: List[float]         = []
    for d in selected_dates:
        day_df = similar_days_df[similar_days_df["date"] == d]
        if day_df.empty:
            continue
        day_avg = day_df["total_drawal"].mean()
        scale = 1.0
        if day_avg > 100 and recent_load_level > 100:
            scale = float(np.clip(recent_load_level / day_avg, 0.70, 1.50))
        scale_log.append(scale)

        vals = (day_df.groupby("time_block")["total_drawal"]
                .mean().reindex(range(1, 97)).ffill().bfill().to_numpy())
        scaled_stack.append(vals * scale)

        tw = tier_weights.get(int(tier_lookup.get(d, 3)), 0.10)
        if pool_lookup.get(d) == "recent_7" and d in _clean_recent:
            tw *= RECENCY_BOOST   # boost only non-anomalous recent days
        weight_stack.append(tw)

    if scale_log:
        try:
            filtered = filtered.assign(scale=pd.Series(scale_log,
                                                        index=filtered.index[:len(scale_log)]))
        except Exception:
            pass

    if not scaled_stack:
        return np.zeros(96), filtered

    weights_arr = np.asarray(weight_stack, dtype=float)
    if weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr)
    weights_arr /= weights_arr.sum()
    baseline = np.tensordot(weights_arr, np.vstack(scaled_stack), axes=(0, 0))

    # Append last-year observation rows to summary for UI reference
    # (they are already in summary with is_last_year_obs=True; merge any
    #  scoring cols back so callers get a consistent schema)
    for _col in ("day_tier", "similarity_score", "pool", "same_season",
                 "same_day_type", "dow"):
        if _col not in summary.columns:
            summary[_col] = np.nan

    # Bias correction applied in run_short_term_pipeline (Stage 4)
    return baseline, summary


def _build_weather_model() -> Any:
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=4,
            verbosity=1,
            random_state=42,
        )
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=1,
            random_state=42,
        )
    return LinearRegression()

def _time_based_split(data: pd.DataFrame, min_val_days: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in data.columns:
        return data, data
    dates = sorted(data["date"].dropna().unique().tolist())
    if len(dates) < min_val_days:
        return data, data
    val_days = min(7, max(1, len(dates) // 5))
    val_dates = set(dates[-val_days:])
    train = data[~data["date"].isin(val_dates)]
    val = data[data["date"].isin(val_dates)]
    if train.empty or val.empty:
        return data, data
    return train, val


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _tune_xgb(X_train, y_train, X_val, y_val, n_iter: int, quantile: float = 0.5) -> Any:
    best_model = None
    best_score = None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        params = {
            "n_estimators": int(rng.integers(250, 800)),
            "max_depth": int(rng.integers(4, 9)),
            "learning_rate": float(rng.uniform(0.02, 0.12)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),
            "min_child_weight": float(rng.uniform(1.0, 6.0)),
            "reg_alpha": float(rng.uniform(0.0, 0.5)),
            "reg_lambda": float(rng.uniform(0.8, 2.0)),
            "objective": "reg:quantileerror",
            "quantile_alpha": quantile,
            "n_jobs": 4,
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        print("\n[ML Pipeline] XGBoost tuning iteration... Training epochs:")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
        preds = model.predict(X_val)
        score = _rmse(y_val, preds)
        if best_score is None or score < best_score:
            best_score = score
            best_model = model
    return best_model


def _tune_lgbm(X_train, y_train, X_val, y_val, n_iter: int, quantile: float = 0.5) -> Any:
    best_model = None
    best_score = None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        params = {
            "n_estimators": int(rng.integers(250, 800)),
            "learning_rate": float(rng.uniform(0.02, 0.12)),
            "num_leaves": int(rng.integers(32, 128)),
            "max_depth": int(rng.integers(-1, 9)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),
            "min_child_samples": int(rng.integers(10, 40)),
            "reg_alpha": float(rng.uniform(0.0, 0.5)),
            "reg_lambda": float(rng.uniform(0.8, 2.0)),
            "objective": "quantile",
            "alpha": quantile,
            "verbosity": -1,
            "random_state": 42,
        }
        model = LGBMRegressor(**params)
        print("\n[ML Pipeline] LightGBM tuning iteration... Training epochs:")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        preds = model.predict(X_val)
        score = _rmse(y_val, preds)
        if best_score is None or score < best_score:
            best_score = score
            best_model = model
    return best_model


class WeatherEnsemble:
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
        
    def predict(self, X):
        if not self.models:
            return np.zeros(X.shape[0])
        preds = np.zeros(X.shape[0])
        total_weight = sum(self.weights)
        for model, w in zip(self.models, self.weights):
            preds += model.predict(X) * w
        return preds / total_weight

def _estimate_feature_effects(model: Any, data: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    y = data["total_drawal"].to_numpy()
    
    # Handle Ensemble
    if isinstance(model, WeatherEnsemble):
        # Average effects across models
        agg_effects = {f: 0.0 for f in features}
        for sub_model, w in zip(model.models, model.weights):
            sub_effects = _estimate_feature_effects(sub_model, data, features)
            for f, val in sub_effects.items():
                agg_effects[f] += val * w
        # Normalize by total weight
        total_w = sum(model.weights)
        return {k: v / total_w for k, v in agg_effects.items()}

    effects = {}
    if hasattr(model, "coef_"):
        # Linear model: coef_ is in feature-units, scale to MW-per-unit
        for f, c in zip(features, model.coef_):
            effects[f] = float(c)
        return effects
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    if importances is None or not np.isfinite(importances).any():
        return {f: 0.0 for f in features}
    importances = importances / max(importances.sum(), 1e-6)
    # For tree models, feature_importances_ is already a relative measure.
    # Use sign from correlation with load, but do NOT scale by y_std/x_std
    # (sparse features like rain have tiny x_std which inflates the ratio).
    for idx, f in enumerate(features):
        x = data[f].to_numpy()
        corr = _safe_corr(pd.Series(y), pd.Series(x))
        sign = 1.0 if corr >= 0 else -1.0
        effects[f] = float(sign * importances[idx])
    return effects


def _train_weather_model(df: pd.DataFrame, tune: bool = True, tune_iters: int = 12) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
    data = _prepare_training(df)
    features = _get_short_term_feature_columns(data)
    data = data.dropna(subset=features + ["total_drawal"])
    X = data[features]
    y = data["total_drawal"].to_numpy()
    model = None
    if tune and (XGBRegressor is not None or LGBMRegressor is not None):
        train_df, val_df = _time_based_split(data)
        X_train = train_df[features]
        y_train = train_df["total_drawal"].to_numpy()
        X_val = val_df[features]
        y_val = val_df["total_drawal"].to_numpy()
        
        models_with_rmse = []
        if XGBRegressor is not None:
            xgb_model = _tune_xgb(X_train, y_train, X_val, y_val, max(4, int(tune_iters)), quantile=0.5)
            xgb_rmse = _rmse(y_val, xgb_model.predict(X_val))
            models_with_rmse.append((xgb_model, xgb_rmse))
        if LGBMRegressor is not None:
            lgbm_model = _tune_lgbm(X_train, y_train, X_val, y_val, max(4, int(tune_iters)), quantile=0.5)
            lgbm_rmse = _rmse(y_val, lgbm_model.predict(X_val))
            models_with_rmse.append((lgbm_model, lgbm_rmse))

        # NHiTS removed from voting ensemble per ensemble_model_strategy.md:
        # it has no weather/calendar conditioning and drags T+2 accuracy.
        # XGBoost + LightGBM are the core ensemble; TiDE/PatchTST are planned replacements.

        if models_with_rmse:
            # RMSE-weighted ensemble (better models get more weight)
            models_list = [m for m, _ in models_with_rmse]
            total_inv_rmse = sum(1.0 / max(r, 1e-6) for _, r in models_with_rmse)
            weights = [(1.0 / max(r, 1e-6)) / total_inv_rmse for _, r in models_with_rmse]
            model = WeatherEnsemble(models_list, weights=weights)

    if model is None:
        model = _build_weather_model()
        print("\n[ML Pipeline] Training base weather model... Training epochs:")
        model.fit(X, y) # Default fallback without eval_set avoids shape issues
    effects = _estimate_feature_effects(model, data, features)
    return model, data, effects


def _weather_baseline(
    model: Any,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    effects: Dict[str, float],
    load_anchor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    data = _prepare_inference_frame(history_df=history_df, target_df=target_df, load_anchor=load_anchor)
    features = _get_short_term_feature_columns(data)
    X = data.reindex(columns=features).copy()

    history_features = _prepare_training(history_df) if history_df is not None and not history_df.empty else pd.DataFrame()
    fill_values = {}
    if not history_features.empty:
        for feature in features:
            if feature in history_features.columns and history_features[feature].notna().any():
                fill_values[feature] = float(history_features[feature].median(skipna=True))

    load_cols = [
        "lag_1", "lag_7", "lag_block_1", "lag_block_4", "rolling_4", "rolling_12",
        "load_lag_1d", "load_lag_2d", "load_lag_3d", "load_lag_7d",
        "load_rolling_7d", "load_trend_3d", "load_x_cdd",
        "load_x_humidity", "wbgt_x_load",
    ]
    anchor_mean = float(np.nanmean(load_anchor)) if load_anchor is not None and len(load_anchor) else 0.0
    for col in load_cols:
        if col in X.columns:
            X[col] = X[col].fillna(fill_values.get(col, anchor_mean))

    for feature in features:
        if feature in X.columns:
            X[feature] = X[feature].fillna(fill_values.get(feature, 0.0))

    preds = model.predict(X)
    contributions = {
        "temperature": float(effects.get("temperature", 0.0)),
        "humidity": float(effects.get("humidity", 0.0)),
        "rain": float(effects.get("precipitation", 0.0)),
        "apparent_temp": float(effects.get("apparent_temperature", 0.0)),
        "cloud": float(effects.get("cloud_cover", 0.0)),
        "sunshine": float(effects.get("sunshine_duration", 0.0)),
        "radiation": float(effects.get("direct_radiation", 0.0)),
        "wind": float(effects.get("wind_speed_10m", 0.0)),
    }
    return preds, contributions


def _compute_weather_deviation(target_df: pd.DataFrame, baseline_df: pd.DataFrame) -> float:
    if target_df.empty or baseline_df.empty:
        return 0.0
    if "temperature" not in target_df.columns or "temperature" not in baseline_df.columns:
        return 0.0
    tgt = float(pd.to_numeric(target_df["temperature"], errors="coerce").mean())
    base = float(pd.to_numeric(baseline_df["temperature"], errors="coerce").mean())
    if not np.isfinite(tgt) or not np.isfinite(base):
        return 0.0
    return float(abs(tgt - base))


def _seasonal_temp_prior(season: str) -> float:
    return {
        "summer": 0.0100,
        "spring": 0.0085,
        "fall": 0.0085,
        "winter": -0.0100,
    }.get(str(season), 0.0090)


def _clip_temp_coeff(value: float, season: str, prior: Optional[float] = None) -> float:
    season_prior = float(_seasonal_temp_prior(season))
    if abs(season_prior) < 1e-6:
        p = float(prior) if prior is not None else 0.0090
    else:
        p = season_prior
    v = float(np.clip(float(value), -0.03, 0.03))
    if p >= 0:
        return float(max(0.0005, v))
    return float(min(-0.0005, v))


def _robust_temp_slope(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy) & (np.abs(xx) > 1e-6)
    xx = xx[valid]
    yy = yy[valid]
    n = int(xx.size)
    if n < 4:
        return None, n
    ratio = yy / xx
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 4:
        return None, int(ratio.size)
    q1, q3 = np.percentile(ratio, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    lo = float(q1 - (2.0 * iqr))
    hi = float(q3 + (2.0 * iqr))
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    use = trimmed if trimmed.size >= 4 else ratio
    if use.size < 4:
        return None, int(use.size)
    return float(np.median(use)), int(use.size)


def _blend_temp_coeff(prior: float, estimate: Optional[float], sample_count: int, ridge: float) -> float:
    if estimate is None:
        return float(prior)
    w = float(np.clip(sample_count / max(sample_count + ridge, 1e-6), 0.0, 1.0))
    return float(((1.0 - w) * prior) + (w * float(estimate)))


def _learn_asymmetric_temp_profile(
    df: pd.DataFrame,
    target_date: str,
    season: str,
    day_type: str,
    lookback_days: int = 180,
    rolling_days: int = 7,
    min_block_samples: int = 8,
    prior_coeff: Optional[float] = None,
) -> Dict[str, Any]:
    prior = float(_seasonal_temp_prior(season) if prior_coeff is None else prior_coeff)
    default_up = np.full(96, prior, dtype=float)
    default_down = np.full(96, prior, dtype=float)
    default_samples = np.zeros(96, dtype=int)
    out = {
        "source": "seasonal_prior",
        "prior_coeff": float(prior),
        "global_increase_coeff": float(prior),
        "global_reduction_coeff": float(prior),
        "global_increase_samples": 0,
        "global_reduction_samples": 0,
        "block_increase_coeffs": default_up,
        "block_reduction_coeffs": default_down,
        "block_increase_samples": default_samples.copy(),
        "block_reduction_samples": default_samples.copy(),
        "diagnostics": {"reason": "fallback_prior"},
    }

    required_cols = {"date", "time_block", "temperature", "total_drawal"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        out["diagnostics"] = {"reason": "missing_required_columns"}
        return out

    dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    if target_date not in dates:
        out["diagnostics"] = {"reason": "target_not_in_dates"}
        return out
    idx = dates.index(target_date)
    hist_dates = dates[max(0, idx - max(lookback_days, 30)):idx]
    if not hist_dates:
        out["diagnostics"] = {"reason": "no_history_dates"}
        return out

    same_context = [d for d in hist_dates if (_season(d) == season and _day_type(d) == day_type)]
    if len(same_context) >= 28:
        selected_dates = same_context[-lookback_days:]
        context_mode = "season_daytype"
    else:
        same_daytype = [d for d in hist_dates if _day_type(d) == day_type]
        if len(same_daytype) >= 28:
            selected_dates = same_daytype[-lookback_days:]
            context_mode = "daytype"
        else:
            selected_dates = hist_dates[-lookback_days:]
            context_mode = "recent_all"

    hist = df[df["date"].astype(str).isin(set(selected_dates))][["date", "time_block", "temperature", "total_drawal"]].copy()
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_history_slice"}
        return out

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist = hist.dropna(subset=["time_block", "temperature", "total_drawal"])
    hist["time_block"] = hist["time_block"].astype(int)
    hist = hist[(hist["time_block"] >= 1) & (hist["time_block"] <= 96)]
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_numeric_clean"}
        return out

    date_order = {d: i for i, d in enumerate(selected_dates)}
    hist["date_idx"] = hist["date"].astype(str).map(date_order)
    hist = hist.dropna(subset=["date_idx"]).sort_values(["time_block", "date_idx"])
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_date_order"}
        return out

    min_periods = max(3, int(min(rolling_days, 7)))
    hist["base_temp"] = (
        hist.groupby("time_block")["temperature"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["base_load"] = (
        hist.groupby("time_block")["total_drawal"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["temp_delta"] = hist["temperature"] - hist["base_temp"]
    hist["load_ratio"] = (hist["total_drawal"] / np.maximum(hist["base_load"], 1e-6)) - 1.0

    valid = (
        np.isfinite(hist["temp_delta"])
        & np.isfinite(hist["load_ratio"])
        & np.isfinite(hist["base_load"])
        & (hist["base_load"] > 1e-6)
        & (np.abs(hist["temp_delta"]) >= 0.15)
    )
    hist = hist[valid]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_valid_delta_rows", "context_mode": context_mode}
        return out

    pos = hist[hist["temp_delta"] > 0]
    neg = hist[hist["temp_delta"] < 0]
    g_up_est, g_up_n = _robust_temp_slope(pos["temp_delta"].to_numpy(dtype=float), pos["load_ratio"].to_numpy(dtype=float))
    g_dn_est, g_dn_n = _robust_temp_slope(neg["temp_delta"].to_numpy(dtype=float), neg["load_ratio"].to_numpy(dtype=float))
    global_up = _clip_temp_coeff(_blend_temp_coeff(prior, g_up_est, g_up_n, ridge=60.0), season, prior=prior)
    global_down = _clip_temp_coeff(_blend_temp_coeff(prior, g_dn_est, g_dn_n, ridge=60.0), season, prior=prior)

    block_up = np.full(96, global_up, dtype=float)
    block_down = np.full(96, global_down, dtype=float)
    block_up_n = np.zeros(96, dtype=int)
    block_down_n = np.zeros(96, dtype=int)

    for b in range(1, 97):
        blk = hist[hist["time_block"] == b]
        if blk.empty:
            continue
        blk_pos = blk[blk["temp_delta"] > 0]
        blk_neg = blk[blk["temp_delta"] < 0]
        up_est, up_n = _robust_temp_slope(blk_pos["temp_delta"].to_numpy(dtype=float), blk_pos["load_ratio"].to_numpy(dtype=float))
        dn_est, dn_n = _robust_temp_slope(blk_neg["temp_delta"].to_numpy(dtype=float), blk_neg["load_ratio"].to_numpy(dtype=float))
        block_up_n[b - 1] = int(up_n)
        block_down_n[b - 1] = int(dn_n)

        if up_n >= int(min_block_samples):
            block_up[b - 1] = _clip_temp_coeff(_blend_temp_coeff(global_up, up_est, up_n, ridge=18.0), season, prior=prior)
        if dn_n >= int(min_block_samples):
            block_down[b - 1] = _clip_temp_coeff(_blend_temp_coeff(global_down, dn_est, dn_n, ridge=18.0), season, prior=prior)

    out.update({
        "source": "learned_history",
        "global_increase_coeff": float(global_up),
        "global_reduction_coeff": float(global_down),
        "global_increase_samples": int(g_up_n),
        "global_reduction_samples": int(g_dn_n),
        "block_increase_coeffs": block_up,
        "block_reduction_coeffs": block_down,
        "block_increase_samples": block_up_n,
        "block_reduction_samples": block_down_n,
        "diagnostics": {
            "context_mode": context_mode,
            "history_dates": int(len(selected_dates)),
            "valid_rows": int(len(hist)),
        },
    })
    return out


def _fit_trend(blocks: np.ndarray, residuals: np.ndarray, degree: int, eval_blocks: Optional[np.ndarray] = None) -> np.ndarray:
    target = eval_blocks if eval_blocks is not None else blocks
    if len(blocks) < 2:
        return np.zeros_like(target, dtype=float)
    deg = max(1, min(int(degree), 3))
    coeffs = np.polyfit(blocks, residuals, deg)
    return np.polyval(coeffs, target)


def _as_96_vector(values: Any, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    default = np.zeros(_BLOCK_COUNT, dtype=float) if fallback is None else np.asarray(fallback, dtype=float)
    default = _fill_nonfinite_vector(default, fill_value=0.0)
    arr = np.asarray(values if values is not None else default, dtype=float).reshape(-1)
    if arr.size == _BLOCK_COUNT:
        return _fill_nonfinite_vector(arr, fill_value=float(default[-1]) if default.size else 0.0)
    if arr.size == 0:
        return default.copy()
    if arr.size == 1:
        scalar = float(arr[0]) if np.isfinite(float(arr[0])) else float(default[-1] if default.size else 0.0)
        return np.full(_BLOCK_COUNT, scalar, dtype=float)
    out = np.zeros(_BLOCK_COUNT, dtype=float)
    n = min(_BLOCK_COUNT, arr.size)
    out[:n] = arr[:n]
    if n < _BLOCK_COUNT:
        out[n:] = out[n - 1]
    return _fill_nonfinite_vector(out, fill_value=float(default[-1]) if default.size else 0.0)


def _selection_mask(selection: Optional[Dict[str, Any]]) -> np.ndarray:
    sel = selection or {}
    scope = str(sel.get("scope", "all")).strip().lower()
    mask = np.zeros(_BLOCK_COUNT, dtype=float)
    if scope in ("all", "overall"):
        mask[:] = 1.0
        return mask
    if scope == "range":
        start = int(sel.get("start_block", sel.get("start", 1)) or 1)
        end = int(sel.get("end_block", sel.get("end", _BLOCK_COUNT)) or _BLOCK_COUNT)
        lo = max(1, min(start, end))
        hi = min(_BLOCK_COUNT, max(start, end))
        mask[lo - 1:hi] = 1.0
        return mask
    if scope in ("single", "block"):
        block = int(sel.get("block", sel.get("block_number", 1)) or 1)
        block = int(np.clip(block, 1, _BLOCK_COUNT))
        mask[block - 1] = 1.0
        return mask
    mask[:] = 1.0
    return mask


def _smoothed_selection_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size != _BLOCK_COUNT:
        return mask
    smooth = mask.astype(float).copy()
    for i in range(_BLOCK_COUNT):
        left = mask[i - 1] if i > 0 else mask[i]
        right = mask[i + 1] if i < (_BLOCK_COUNT - 1) else mask[i]
        if (mask[i] == 0.0 and (left == 1.0 or right == 1.0)) or (mask[i] == 1.0 and (left == 0.0 or right == 0.0)):
            smooth[i] = 0.5
    return smooth


def build_block_driver_matrix(
    target_date: str,
    baseline_load: np.ndarray,
    weather_base: np.ndarray,
    daytype_base: np.ndarray,
    holiday_base: np.ndarray,
    manual_base: np.ndarray,
    temperature_base: Optional[np.ndarray] = None,
    humidity_base: Optional[np.ndarray] = None,
    precipitation_base: Optional[np.ndarray] = None,
    apparent_base: Optional[np.ndarray] = None,
    cloud_base: Optional[np.ndarray] = None,
    sun_base: Optional[np.ndarray] = None,
    radiation_base: Optional[np.ndarray] = None,
    wind_base: Optional[np.ndarray] = None,
    sliders: Optional[Dict[str, float]] = None,
    selection: Optional[Dict[str, Any]] = None,
    smooth_edges: bool = False,
) -> Dict[str, Any]:
    baseline_vec = _as_96_vector(baseline_load)
    w_base = _as_96_vector(weather_base)
    d_base = _as_96_vector(daytype_base)
    h_base = _as_96_vector(holiday_base)
    m_base = _as_96_vector(manual_base)
    t_base = _as_96_vector(temperature_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    hu_base = _as_96_vector(humidity_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    p_base = _as_96_vector(precipitation_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    a_base = _as_96_vector(apparent_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    c_base = _as_96_vector(cloud_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    s_base = _as_96_vector(sun_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    r_base = _as_96_vector(radiation_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    wi_base = _as_96_vector(wind_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))

    if (
        np.allclose(t_base, 0.0)
        and np.allclose(hu_base, 0.0)
        and np.allclose(p_base, 0.0)
        and np.allclose(a_base, 0.0)
        and np.allclose(c_base, 0.0)
        and np.allclose(s_base, 0.0)
        and np.allclose(r_base, 0.0)
        and np.allclose(wi_base, 0.0)
    ):
        # Backward compatibility: when weather is provided as a single base vector.
        t_base = w_base.copy()

    slider_cfg = sliders or {}
    s_w = float(slider_cfg.get("weather", slider_cfg.get("S_w", 1.0)))
    s_d = float(slider_cfg.get("daytype", slider_cfg.get("S_d", 1.0)))
    s_h = float(slider_cfg.get("holiday", slider_cfg.get("S_h", 1.0)))
    s_m = float(slider_cfg.get("manual", slider_cfg.get("S_m", 1.0)))
    s_temp = float(slider_cfg.get("temperature", slider_cfg.get("temp", 1.0)))
    s_hum = float(slider_cfg.get("humidity", 1.0))
    s_prec = float(slider_cfg.get("precipitation", slider_cfg.get("precip", 1.0)))
    s_apparent = float(slider_cfg.get("apparent_temperature", slider_cfg.get("apparent", 1.0)))
    s_cloud = float(slider_cfg.get("cloud_cover", slider_cfg.get("cloud", 1.0)))
    s_sun = float(slider_cfg.get("sunshine_duration", slider_cfg.get("sunshine", slider_cfg.get("sun", 1.0))))
    s_rad = float(slider_cfg.get("direct_radiation", slider_cfg.get("radiation", slider_cfg.get("rad", 1.0))))
    s_wind = float(slider_cfg.get("wind", slider_cfg.get("wind_speed_10m", slider_cfg.get("wind_speed", 1.0))))

    temperature_scaled = t_base * s_w * s_temp
    humidity_scaled = hu_base * s_w * s_hum
    precipitation_scaled = p_base * s_w * s_prec
    apparent_scaled = a_base * s_w * s_apparent
    cloud_scaled = c_base * s_w * s_cloud
    sun_scaled = s_base * s_w * s_sun
    rad_scaled = r_base * s_w * s_rad
    wind_scaled = wi_base * s_w * s_wind
    weather_scaled = (
        temperature_scaled
        + humidity_scaled
        + precipitation_scaled
        + apparent_scaled
        + cloud_scaled
        + sun_scaled
        + rad_scaled
        + wind_scaled
    )
    daytype_scaled = d_base * s_d
    holiday_scaled = h_base * s_h
    manual_scaled = m_base * s_m
    final_impact = np.clip(weather_scaled + daytype_scaled + holiday_scaled + manual_scaled, _IMPACT_MIN, _IMPACT_MAX)

    mask = _selection_mask(selection)
    if smooth_edges:
        mask = _smoothed_selection_mask(mask)

    final_load = baseline_vec * (1.0 + (final_impact * mask))

    matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "weather_base": w_base,
        "temperature_base": t_base,
        "humidity_base": hu_base,
        "precipitation_base": p_base,
        "apparent_base": a_base,
        "cloud_base": c_base,
        "sun_base": s_base,
        "radiation_base": r_base,
        "wind_base": wi_base,
        "daytype_base": d_base,
        "holiday_base": h_base,
        "manual_base": m_base,
        "weather_scaled": weather_scaled,
        "temperature_scaled": temperature_scaled,
        "humidity_scaled": humidity_scaled,
        "precipitation_scaled": precipitation_scaled,
        "apparent_scaled": apparent_scaled,
        "cloud_scaled": cloud_scaled,
        "sun_scaled": sun_scaled,
        "rad_scaled": rad_scaled,
        "wind_scaled": wind_scaled,
        "daytype_scaled": daytype_scaled,
        "holiday_scaled": holiday_scaled,
        "manual_scaled": manual_scaled,
        "final_impact": final_impact,
    })

    return {
        "matrix_df": matrix,
        "selection_mask": mask,
        "final_load": final_load,
    }


def _resolve_wind_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("wind_speed_10m", "wind_speed", "wind", "windspeed", "wind_mps"):
        if col in df.columns:
            return col
    return None


def _resolve_holiday_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("is_holiday", "holiday", "holiday_flag", "holiday_ind"):
        if col in df.columns:
            return col
    return None


def _safe_ratio_delta(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) <= 1e-9:
        return 0.0
    return float((numer - denom) / denom)


def compute_block_driver_weights(history_df: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        hist = history_df.copy()
        hist["date"] = hist["date"].astype(str)

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    if hist.empty:
        raise ValueError("No valid block rows in history_df")

    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["humidity"] = pd.to_numeric(hist["humidity"], errors="coerce")
    hist["precipitation"] = pd.to_numeric(hist["precipitation"], errors="coerce")

    wind_col = _resolve_wind_column(hist)
    if wind_col is None:
        import logging
        logging.getLogger(__name__).warning(
            "No wind column found; using zero proxy – wind driver weights will be zero."
        )
        hist["wind_proxy"] = 0.0
        wind_col = "wind_proxy"
    hist[wind_col] = pd.to_numeric(hist[wind_col], errors="coerce")

    dt = pd.to_datetime(hist["date"], errors="coerce")
    hist["is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    holiday_col = _resolve_holiday_column(hist)
    if holiday_col is not None:
        holiday_series = pd.to_numeric(hist[holiday_col], errors="coerce")
        hist["is_holiday_flag"] = (holiday_series.fillna(0.0) > 0.0).astype(float)
    elif "day_type" in hist.columns:
        day_type = hist["day_type"].astype(str).str.lower()
        hist["is_holiday_flag"] = day_type.str.contains("holiday").astype(float)
    else:
        hist["is_holiday_flag"] = 0.0

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        wind_col: float(hist[wind_col].median(skipna=True)) if hist[wind_col].notna().any() else 0.0,
    }

    grouped = {int(k): v for k, v in hist.groupby(hist["time_block"].astype(int))}
    rows: List[Dict[str, Any]] = []
    feature_cols = ["temperature", "humidity", "precipitation", wind_col]

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=hist.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)
        beta_temp = 0.0
        beta_hum = 0.0
        beta_rain = 0.0
        beta_wind = 0.0

        if len(blk) >= 8 and float(np.std(blk["total_drawal"].to_numpy(dtype=float))) > 1e-9:
            x_df = blk[feature_cols].copy()
            for col in feature_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
                med = float(x_df[col].median(skipna=True)) if x_df[col].notna().any() else global_medians.get(col, 0.0)
                if not np.isfinite(med):
                    med = global_medians.get(col, 0.0)
                x_df[col] = x_df[col].fillna(med)

            x_mat = x_df.to_numpy(dtype=float)
            y_vec = blk["total_drawal"].to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)) and x_mat.shape[0] >= 8:
                try:
                    model = LinearRegression()
                    model.fit(x_mat, y_vec)
                    coefs = model.coef_.reshape(-1)
                    if coefs.size == 4:
                        beta_temp = float(coefs[0])
                        beta_hum = float(coefs[1])
                        beta_rain = float(coefs[2])
                        beta_wind = float(coefs[3])
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).debug(
                        "Block %d LinearRegression failed: %s", block, exc
                    )

        weekday_vals = blk.loc[blk["is_weekend"] < 0.5, "total_drawal"].to_numpy(dtype=float)
        weekend_vals = blk.loc[blk["is_weekend"] >= 0.5, "total_drawal"].to_numpy(dtype=float)
        weekday_avg = float(np.mean(weekday_vals)) if weekday_vals.size else np.nan
        weekend_avg = float(np.mean(weekend_vals)) if weekend_vals.size else np.nan
        if np.isfinite(weekday_avg) and abs(weekday_avg) > 1e-9 and np.isfinite(weekend_avg):
            daytype_weight = _safe_ratio_delta(weekend_avg, weekday_avg)
        else:
            daytype_weight = 0.0

        holiday_vals = blk.loc[blk["is_holiday_flag"] >= 0.5, "total_drawal"].to_numpy(dtype=float)
        normal_vals = blk.loc[blk["is_holiday_flag"] < 0.5, "total_drawal"].to_numpy(dtype=float)
        holiday_avg = float(np.mean(holiday_vals)) if holiday_vals.size else np.nan
        normal_avg = float(np.mean(normal_vals)) if normal_vals.size else np.nan
        if np.isfinite(holiday_avg) and np.isfinite(normal_avg) and abs(normal_avg) > 1e-9:
            holiday_weight = _safe_ratio_delta(holiday_avg, normal_avg)
        else:
            holiday_weight = 0.0

        rows.append({
            "block": block,
            "temp_weight": float(beta_temp / denom),
            "humidity_weight": float(beta_hum / denom),
            "rain_weight": float(beta_rain / denom),
            "wind_weight": float(beta_wind / denom),
            "daytype_weight": float(daytype_weight),
            "holiday_weight": float(holiday_weight),
            "avg_load": float(avg_load),
            "samples": int(len(blk)),
        })

    return pd.DataFrame(rows)


def compute_block_driver_weights_elastic_net(
    history_df: pd.DataFrame,
    target_date: Optional[str] = None,
    l1_ratio_grid: Optional[List[float]] = None,
    alpha_grid: Optional[np.ndarray] = None,
    min_samples: int = 14,
    max_iter: int = 10000,
    random_state: int = 42,
    add_temp_squared: bool = False,
    add_temp_humidity_interaction: bool = False,
    season_segment: bool = False,
    smooth_window: int = 0,
    apply_confidence_shrinkage: bool = True,
    shrink_power: float = 1.2,
) -> pd.DataFrame:
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        hist = history_df.copy()
        hist["date"] = hist["date"].astype(str)

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    if hist.empty:
        raise ValueError("No valid block rows in history_df")

    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["humidity"] = pd.to_numeric(hist["humidity"], errors="coerce")
    hist["precipitation"] = pd.to_numeric(hist["precipitation"], errors="coerce")

    wind_col = _resolve_wind_column(hist)
    if wind_col is None:
        import logging
        logging.getLogger(__name__).warning(
            "No wind column found; using zero proxy – wind driver weights will be zero."
        )
        hist["wind_proxy"] = 0.0
        wind_col = "wind_proxy"
    hist[wind_col] = pd.to_numeric(hist[wind_col], errors="coerce")

    dt = pd.to_datetime(hist["date"], errors="coerce")
    hist["is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    holiday_col = _resolve_holiday_column(hist)
    if holiday_col is not None:
        holiday_series = pd.to_numeric(hist[holiday_col], errors="coerce")
        hist["is_holiday_flag"] = (holiday_series.fillna(0.0) > 0.0).astype(float)
    elif "day_type" in hist.columns:
        day_type = hist["day_type"].astype(str).str.lower()
        hist["is_holiday_flag"] = day_type.str.contains("holiday").astype(float)
    else:
        hist["is_holiday_flag"] = 0.0

    if season_segment:
        season_col = pd.to_datetime(hist["date"], errors="coerce").dt.month.map(SEASON_BY_MONTH)
        hist["season_seg"] = season_col.fillna("unknown")
        if target_date:
            target_ts = pd.to_datetime(target_date, errors="coerce")
            if not pd.isna(target_ts):
                target_season = SEASON_BY_MONTH.get(int(target_ts.month), None)
                if target_season is not None:
                    seg_hist = hist[hist["season_seg"] == target_season].copy()
                    if len(seg_hist) >= max(96, int(min_samples * 8)):
                        hist = seg_hist

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        wind_col: float(hist[wind_col].median(skipna=True)) if hist[wind_col].notna().any() else 0.0,
    }

    ratio_grid = l1_ratio_grid if l1_ratio_grid else [0.15, 0.35, 0.5, 0.7, 0.9]
    alphas = np.asarray(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-4, 0.8, 36)
    alphas = np.sort(np.unique(alphas[(alphas > 0) & np.isfinite(alphas)]))
    if alphas.size == 0:
        alphas = np.logspace(-4, 0.8, 36)

    grouped = {int(k): v for k, v in hist.groupby(hist["time_block"].astype(int))}
    rows: List[Dict[str, Any]] = []

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=hist.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)

        feature_df = pd.DataFrame({
            "temp": pd.to_numeric(blk.get("temperature"), errors="coerce"),
            "humidity": pd.to_numeric(blk.get("humidity"), errors="coerce"),
            "rain": pd.to_numeric(blk.get("precipitation"), errors="coerce"),
            "wind": pd.to_numeric(blk.get(wind_col), errors="coerce"),
            "daytype": pd.to_numeric(blk.get("is_weekend"), errors="coerce"),
            "holiday": pd.to_numeric(blk.get("is_holiday_flag"), errors="coerce"),
        })
        for col in ("temp", "humidity", "rain", "wind", "daytype", "holiday"):
            med = float(feature_df[col].median(skipna=True)) if feature_df[col].notna().any() else 0.0
            if col == "temp" and not np.isfinite(med):
                med = global_medians["temperature"]
            elif col == "humidity" and not np.isfinite(med):
                med = global_medians["humidity"]
            elif col == "rain" and not np.isfinite(med):
                med = global_medians["precipitation"]
            elif col == "wind" and not np.isfinite(med):
                med = global_medians[wind_col]
            if not np.isfinite(med):
                med = 0.0
            feature_df[col] = feature_df[col].fillna(med)

        if add_temp_squared:
            feature_df["temp_sq"] = np.square(feature_df["temp"].to_numpy(dtype=float))
        if add_temp_humidity_interaction:
            feature_df["temp_x_humidity"] = feature_df["temp"].to_numpy(dtype=float) * feature_df["humidity"].to_numpy(dtype=float)

        y_vec = blk["total_drawal"].to_numpy(dtype=float)

        coeffs = {
            "temp": 0.0,
            "humidity": 0.0,
            "rain": 0.0,
            "wind": 0.0,
            "daytype": 0.0,
            "holiday": 0.0,
        }
        model_r2 = 0.0
        alpha_opt = np.nan
        l1_ratio_opt = np.nan

        if len(feature_df) >= int(min_samples) and float(np.std(y_vec)) > 1e-9:
            x_mat = feature_df.to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)):
                cv_folds = int(np.clip(len(feature_df) // 6, 3, 5))
                if len(feature_df) <= cv_folds:
                    cv_folds = max(2, len(feature_df) - 1)
                if cv_folds >= 2:
                    try:
                        model = Pipeline(steps=[
                            ("scale", StandardScaler(with_mean=True, with_std=True)),
                            (
                                "enet",
                                ElasticNetCV(
                                    l1_ratio=ratio_grid,
                                    alphas=alphas,
                                    fit_intercept=True,
                                    cv=cv_folds,
                                    random_state=int(random_state),
                                    max_iter=int(max_iter),
                                    selection="cyclic",
                                ),
                            ),
                        ])
                        model.fit(x_mat, y_vec)

                        scaler = model.named_steps["scale"]
                        enet = model.named_steps["enet"]
                        alpha_opt = float(getattr(enet, "alpha_", np.nan))
                        l1_ratio_opt = float(getattr(enet, "l1_ratio_", np.nan))
                        scaled_coef = np.asarray(enet.coef_, dtype=float).reshape(-1)
                        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
                        if scaled_coef.size == scale.size and scaled_coef.size == feature_df.shape[1]:
                            raw_coef = np.divide(
                                scaled_coef,
                                np.where(np.abs(scale) > 1e-12, scale, 1.0),
                                out=np.zeros_like(scaled_coef, dtype=float),
                                where=np.abs(scale) > 1e-12,
                            )
                            for i, col in enumerate(feature_df.columns.tolist()):
                                if col in coeffs:
                                    coeffs[col] = float(raw_coef[i])

                        y_hat = model.predict(x_mat)
                        ss_res = float(np.sum(np.square(y_vec - y_hat)))
                        ss_tot = float(np.sum(np.square(y_vec - np.mean(y_vec))))
                        if ss_tot > 1e-9:
                            model_r2 = float(1.0 - (ss_res / ss_tot))
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).debug(
                            "Block %d ElasticNetCV failed: %s", block, exc
                        )

        sample_conf = float(np.clip(len(feature_df) / max(float(min_samples) * 3.0, 1.0), 0.0, 1.0))
        fit_conf = float(np.clip(model_r2, 0.0, 1.0))
        weight_conf = float(np.clip((0.6 * sample_conf) + (0.4 * fit_conf), 0.0, 1.0))
        shrink = float(weight_conf ** max(float(shrink_power), 0.0)) if apply_confidence_shrinkage else 1.0

        temp_w_raw = float(coeffs["temp"] / denom)
        hum_w_raw = float(coeffs["humidity"] / denom)
        rain_w_raw = float(coeffs["rain"] / denom)
        wind_w_raw = float(coeffs["wind"] / denom)
        daytype_w_raw = float(coeffs["daytype"] / denom)
        holiday_w_raw = float(coeffs["holiday"] / denom)

        rows.append({
            "block": block,
            "temp_weight_raw": temp_w_raw,
            "humidity_weight_raw": hum_w_raw,
            "rain_weight_raw": rain_w_raw,
            "wind_weight_raw": wind_w_raw,
            "daytype_weight_raw": daytype_w_raw,
            "holiday_weight_raw": holiday_w_raw,
            "temp_weight": float(temp_w_raw * shrink),
            "humidity_weight": float(hum_w_raw * shrink),
            "rain_weight": float(rain_w_raw * shrink),
            "wind_weight": float(wind_w_raw * shrink),
            "daytype_weight": float(daytype_w_raw * shrink),
            "holiday_weight": float(holiday_w_raw * shrink),
            "weight_confidence": weight_conf,
            "sample_confidence": sample_conf,
            "fit_confidence": fit_conf,
            "samples": int(len(feature_df)),
            "model_r2": float(model_r2),
            "alpha_opt": float(alpha_opt) if np.isfinite(alpha_opt) else np.nan,
            "l1_ratio_opt": float(l1_ratio_opt) if np.isfinite(l1_ratio_opt) else np.nan,
        })

    out = pd.DataFrame(rows)
    if int(smooth_window) and int(smooth_window) > 1:
        w = int(max(1, smooth_window))
        for col in ("temp_weight", "humidity_weight", "rain_weight", "wind_weight", "daytype_weight", "holiday_weight"):
            out[col] = out[col].rolling(window=w, center=True, min_periods=1).mean()
        if "weight_confidence" in out.columns:
            out["weight_confidence"] = out["weight_confidence"].rolling(window=w, center=True, min_periods=1).mean()

    return out


def run_weather_impact_elastic_net_engine(
    history_df: pd.DataFrame,
    target_date: str,
    base_load: Optional[np.ndarray] = None,
    momentum_lambda: float = 0.45,
    min_samples: int = 14,
    max_iter: int = 10000,
    random_state: int = 42,
    l1_ratio_grid: Optional[List[float]] = None,
    alpha_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Blockwise weather impact engine:
    1) Train Elastic Net with features temp/humidity/rain/lag_1_load/lag_96_load
    2) Convert weather coefficients to elasticity weights
    3) Compute weather deltas (today - yesterday)
    4) Weather impact = sum(weight * delta)
    5) DoD load delta = Load_t-1 - Load_t-2
    6) Momentum impact = lambda * DoD
    7) Simulated load = Base + Weather Impact + Momentum Impact
    """
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    optional_weather = ["apparent_temperature", "cloud_cover", "sunshine_duration", "direct_radiation", "wind_speed_10m"]
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    # Create missing optional weather columns as 0 so downstream code doesn't break
    for col in optional_weather:
        if col not in hist.columns:
            hist[col] = 0.0
    hist["date"] = hist["date"].astype(str)
    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")

    for col in ["temperature", "humidity", "precipitation"] + optional_weather:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    hist = hist.dropna(subset=["date", "time_block", "total_drawal"])
    if hist.empty:
        raise ValueError("No valid rows in history_df after cleaning")

    hist["time_block"] = hist["time_block"].astype(int)
    hist = hist.sort_values(["date", "time_block"]).reset_index(drop=True)

    dates = sorted(hist["date"].dropna().astype(str).unique().tolist())
    if not dates:
        raise ValueError("No dates available for weather impact engine")
    resolved_date = str(target_date) if str(target_date) in dates else dates[-1]
    target_idx = dates.index(resolved_date)
    prev_date = dates[target_idx - 1] if target_idx >= 1 else None
    prev2_date = dates[target_idx - 2] if target_idx >= 2 else None

    # Sequence lags on full 15-min chronology.
    hist["lag_1_load"] = pd.to_numeric(hist["total_drawal"], errors="coerce").shift(1)
    hist["lag_96_load"] = pd.to_numeric(hist["total_drawal"], errors="coerce").shift(_BLOCK_COUNT)
    train = hist[hist["date"] < resolved_date].copy()

    ratio_grid = l1_ratio_grid if l1_ratio_grid else [0.15, 0.35, 0.5, 0.7, 0.9]
    alphas = np.asarray(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-4, 0.8, 36)
    alphas = np.sort(np.unique(alphas[(alphas > 0) & np.isfinite(alphas)]))
    if alphas.size == 0:
        alphas = np.logspace(-4, 0.8, 36)

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        "apparent_temperature": float(hist["apparent_temperature"].median(skipna=True)) if hist["apparent_temperature"].notna().any() else 0.0,
        "cloud_cover": float(hist["cloud_cover"].median(skipna=True)) if hist["cloud_cover"].notna().any() else 0.0,
        "sunshine_duration": float(hist["sunshine_duration"].median(skipna=True)) if hist["sunshine_duration"].notna().any() else 0.0,
        "direct_radiation": float(hist["direct_radiation"].median(skipna=True)) if hist["direct_radiation"].notna().any() else 0.0,
        "wind_speed_10m": float(hist["wind_speed_10m"].median(skipna=True)) if hist["wind_speed_10m"].notna().any() else 0.0,
        "lag_1_load": float(hist["lag_1_load"].median(skipna=True)) if hist["lag_1_load"].notna().any() else 0.0,
        "lag_96_load": float(hist["lag_96_load"].median(skipna=True)) if hist["lag_96_load"].notna().any() else 0.0,
    }

    rows: List[Dict[str, Any]] = []
    grouped = {int(k): v for k, v in train.groupby(train["time_block"].astype(int))} if not train.empty else {}
    feature_cols = [
        "temperature", "humidity", "precipitation", 
        "apparent_temperature", "cloud_cover", "sunshine_duration", 
        "direct_radiation", "wind_speed_10m",
        "lag_1_load", "lag_96_load"
    ]

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=train.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)

        coeffs = {col: 0.0 for col in feature_cols}
        model_r2 = 0.0
        alpha_opt = np.nan
        l1_ratio_opt = np.nan

        if len(blk) >= int(min_samples) and float(np.std(blk["total_drawal"].to_numpy(dtype=float))) > 1e-9:
            x_df = blk[feature_cols].copy()
            for col in feature_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
                med = float(x_df[col].median(skipna=True)) if x_df[col].notna().any() else global_medians.get(col, 0.0)
                if not np.isfinite(med):
                    med = global_medians.get(col, 0.0)
                x_df[col] = x_df[col].fillna(med)

            x_mat = x_df.to_numpy(dtype=float)
            y_vec = blk["total_drawal"].to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)):
                cv_folds = int(np.clip(len(x_df) // 6, 3, 5))
                if len(x_df) <= cv_folds:
                    cv_folds = max(2, len(x_df) - 1)
                if cv_folds >= 2:
                    try:
                        model = Pipeline(steps=[
                            ("scale", StandardScaler(with_mean=True, with_std=True)),
                            (
                                "enet",
                                ElasticNetCV(
                                    l1_ratio=ratio_grid,
                                    alphas=alphas,
                                    fit_intercept=True,
                                    cv=cv_folds,
                                    random_state=int(random_state),
                                    max_iter=int(max_iter),
                                    selection="cyclic",
                                ),
                            ),
                        ])
                        model.fit(x_mat, y_vec)

                        scaler = model.named_steps["scale"]
                        enet = model.named_steps["enet"]
                        alpha_opt = float(getattr(enet, "alpha_", np.nan))
                        l1_ratio_opt = float(getattr(enet, "l1_ratio_", np.nan))
                        scaled_coef = np.asarray(enet.coef_, dtype=float).reshape(-1)
                        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
                        if scaled_coef.size == scale.size and scaled_coef.size == len(feature_cols):
                            raw_coef = np.divide(
                                scaled_coef,
                                np.where(np.abs(scale) > 1e-12, scale, 1.0),
                                out=np.zeros_like(scaled_coef, dtype=float),
                                where=np.abs(scale) > 1e-12,
                            )
                            for i, col in enumerate(feature_cols):
                                coeffs[col] = float(raw_coef[i])

                        y_hat = model.predict(x_mat)
                        ss_res = float(np.sum(np.square(y_vec - y_hat)))
                        ss_tot = float(np.sum(np.square(y_vec - np.mean(y_vec))))
                        if ss_tot > 1e-9:
                            model_r2 = float(1.0 - (ss_res / ss_tot))
                    except Exception:
                        pass

        sample_conf = float(np.clip(len(blk) / max(float(min_samples) * 3.0, 1.0), 0.0, 1.0))
        fit_conf = float(np.clip(model_r2, 0.0, 1.0))
        weight_conf = float(np.clip((0.6 * sample_conf) + (0.4 * fit_conf), 0.0, 1.0))

        res = {
            "block": block,
            "samples": int(len(blk)),
            "model_r2": float(model_r2),
            "alpha_opt": float(alpha_opt) if np.isfinite(alpha_opt) else np.nan,
            "l1_ratio_opt": float(l1_ratio_opt) if np.isfinite(l1_ratio_opt) else np.nan,
            "weight_confidence": float(weight_conf),
        }
        for col in feature_cols:
            res[f"{col}_weight"] = float(coeffs[col] / denom)
        rows.append(res)

    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        weight_df = pd.DataFrame({
            "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
            "temp_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "humidity_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "rain_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "lag_1_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "lag_96_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "samples": np.zeros(_BLOCK_COUNT, dtype=int),
            "model_r2": np.zeros(_BLOCK_COUNT, dtype=float),
            "alpha_opt": np.full(_BLOCK_COUNT, np.nan, dtype=float),
            "l1_ratio_opt": np.full(_BLOCK_COUNT, np.nan, dtype=float),
            "weight_confidence": np.full(_BLOCK_COUNT, 0.0, dtype=float),
        })

    def _day_vector(date_value: Optional[str], col: str, fill: float = 0.0) -> np.ndarray:
        if not date_value:
            return np.full(_BLOCK_COUNT, float(fill), dtype=float)
        day = hist[hist["date"].astype(str) == str(date_value)][["time_block", col]].copy()
        if day.empty:
            return np.full(_BLOCK_COUNT, float(fill), dtype=float)
        day["time_block"] = pd.to_numeric(day["time_block"], errors="coerce")
        day[col] = pd.to_numeric(day[col], errors="coerce")
        day = day.dropna(subset=["time_block"]).sort_values("time_block")
        ser = day.set_index(day["time_block"].astype(int))[col].reindex(range(1, _BLOCK_COUNT + 1))
        if ser.notna().any():
            ser = ser.ffill().bfill().fillna(float(fill))
        else:
            ser = pd.Series(np.full(_BLOCK_COUNT, float(fill), dtype=float), index=range(1, _BLOCK_COUNT + 1))
        return ser.to_numpy(dtype=float)

    today_temp = _day_vector(resolved_date, "temperature", fill=global_medians["temperature"])
    yday_temp = _day_vector(prev_date, "temperature", fill=global_medians["temperature"])
    today_hum = _day_vector(resolved_date, "humidity", fill=global_medians["humidity"])
    yday_hum = _day_vector(prev_date, "humidity", fill=global_medians["humidity"])
    today_rain = _day_vector(resolved_date, "precipitation", fill=global_medians["precipitation"])
    yday_rain = _day_vector(prev_date, "precipitation", fill=global_medians["precipitation"])
    
    today_apparent = _day_vector(resolved_date, "apparent_temperature", fill=global_medians["apparent_temperature"])
    yday_apparent = _day_vector(prev_date, "apparent_temperature", fill=global_medians["apparent_temperature"])
    today_cloud = _day_vector(resolved_date, "cloud_cover", fill=global_medians["cloud_cover"])
    yday_cloud = _day_vector(prev_date, "cloud_cover", fill=global_medians["cloud_cover"])
    today_sun = _day_vector(resolved_date, "sunshine_duration", fill=global_medians["sunshine_duration"])
    yday_sun = _day_vector(prev_date, "sunshine_duration", fill=global_medians["sunshine_duration"])
    today_rad = _day_vector(resolved_date, "direct_radiation", fill=global_medians["direct_radiation"])
    yday_rad = _day_vector(prev_date, "direct_radiation", fill=global_medians["direct_radiation"])
    today_wind = _day_vector(resolved_date, "wind_speed_10m", fill=global_medians["wind_speed_10m"])
    yday_wind = _day_vector(prev_date, "wind_speed_10m", fill=global_medians["wind_speed_10m"])

    yday_load = _day_vector(prev_date, "total_drawal", fill=0.0)
    yday2_load = _day_vector(prev2_date, "total_drawal", fill=0.0)

    # Baseline anomaly guard: if yesterday's total load is an outlier vs its
    # same-weekday rolling window, fall back to the most-recent clean same-weekday
    # day to avoid corrupting both the baseline and the momentum term.
    if prev_date:
        try:
            prev_dt = pd.to_datetime(prev_date)
            prev_weekday = prev_dt.weekday()
            # Collect daily totals for same weekday in the training window (up to 8 weeks back)
            _same_wd_totals: list[float] = []
            for _d in reversed(dates[:target_idx - 1]):
                _dt = pd.to_datetime(_d)
                if _dt.weekday() == prev_weekday:
                    _vec = _day_vector(_d, "total_drawal", fill=0.0)
                    _tot = float(_vec.sum())
                    if _tot > 0:
                        _same_wd_totals.append(_tot)
                if len(_same_wd_totals) >= 8:
                    break
            if len(_same_wd_totals) >= 3:
                _sw_arr = np.asarray(_same_wd_totals, dtype=float)
                _sw_mean = float(_sw_arr.mean())
                _sw_std = float(_sw_arr.std())
                _yday_total = float(yday_load.sum())
                if _sw_std > 0 and abs(_yday_total - _sw_mean) > 2.0 * _sw_std:
                    # Yesterday is anomalous — find the most recent clean same-weekday fallback
                    logger.warning(
                        "Yesterday (%s) total load %.0f MW is outside 2σ band "
                        "[%.0f ± %.0f MW] — looking for clean same-weekday fallback.",
                        prev_date, _yday_total, _sw_mean, _sw_std,
                    )
                    for _d in reversed(dates[:target_idx - 1]):
                        _dt = pd.to_datetime(_d)
                        if _dt.weekday() == prev_weekday:
                            _vec = _day_vector(_d, "total_drawal", fill=0.0)
                            _tot = float(_vec.sum())
                            if _tot > 0 and abs(_tot - _sw_mean) <= 2.0 * _sw_std:
                                logger.warning("Using %s as clean baseline day instead of %s.", _d, prev_date)
                                yday_load = _vec
                                break
        except Exception as _e:
            logger.debug("Baseline anomaly guard skipped: %s", _e)

    temp_delta = today_temp - yday_temp
    hum_delta = today_hum - yday_hum
    # rain_delta: today minus yesterday precipitation.
    # First-rain-event detection: if today has significant rain (>5mm daily total) but
    # yesterday was dry (<1mm), this is the season opener. The elastic net coefficient
    # was fitted on mid-season rain events where load was already modulated — applying
    # it full-strength on day-1 produces catastrophic over-suppression (May 30 failure).
    # On a first-rain day, cap rain_delta at +3mm/block (modest signal only).
    _today_total_mm = float(np.nansum(today_rain))
    _yday_total_mm  = float(np.nansum(yday_rain))
    _is_first_rain = _today_total_mm > 5.0 and _yday_total_mm < 1.0
    if _is_first_rain:
        # Suppress rain_delta on the season's first rain day: use flat small positive
        # delta so the model registers "rain" but doesn't over-correct.
        rain_delta = np.clip(today_rain - yday_rain, -3.0, 3.0)
        logger.info(
            "[rain_delta] First-rain-event: today=%.1fmm, yday=%.1fmm → "
            "rain_delta clamped to ±3mm (was ±10mm default)",
            _today_total_mm, _yday_total_mm,
        )
    else:
        # Standard clamp: elastic net trained on ≤3mm typical deltas;
        # 20mm monsoon day would extrapolate to −2000MW without this guard.
        rain_delta = np.clip(today_rain - yday_rain, -10.0, 10.0)
    apparent_delta = today_apparent - yday_apparent
    cloud_delta = today_cloud - yday_cloud
    sun_delta = today_sun - yday_sun
    rad_delta = today_rad - yday_rad
    wind_delta = today_wind - yday_wind
    
    dod_load_delta = yday_load - yday2_load if prev2_date else np.zeros(_BLOCK_COUNT, dtype=float)

    ordered = weight_df.sort_values("block")
    temp_w = _as_96_vector(ordered["temperature_weight"].to_numpy(dtype=float))
    hum_w = _as_96_vector(ordered["humidity_weight"].to_numpy(dtype=float))
    rain_w = _as_96_vector(ordered["precipitation_weight"].to_numpy(dtype=float))
    apparent_w = _as_96_vector(ordered["apparent_temperature_weight"].to_numpy(dtype=float))
    cloud_w = _as_96_vector(ordered["cloud_cover_weight"].to_numpy(dtype=float))
    sun_w = _as_96_vector(ordered["sunshine_duration_weight"].to_numpy(dtype=float))
    rad_w = _as_96_vector(ordered["direct_radiation_weight"].to_numpy(dtype=float))
    wind_w = _as_96_vector(ordered["wind_speed_10m_weight"].to_numpy(dtype=float))
    
    conf_w = _as_96_vector(ordered["weight_confidence"].to_numpy(dtype=float))

    base_vec = _as_96_vector(base_load if base_load is not None else yday_load)

    weather_impact_pct = (
        (temp_delta * temp_w) + 
        (hum_delta * hum_w) + 
        (rain_delta * rain_w) +
        (apparent_delta * apparent_w) +
        (cloud_delta * cloud_w) +
        (sun_delta * sun_w) +
        (rad_delta * rad_w) +
        (wind_delta * wind_w)
    )
    # Cap weather_impact_pct — regime-aware:
    #   Hot-Dry: AC penetration means temperature effect can exceed +12%. Raise upper cap to +18%.
    #   Active Monsoon: rain suppression is real but bounded. Keep lower cap at -20%.
    #   First-rain day (monsoon_day_number == 1): tighten suppression cap to -10% to prevent
    #     the elastic net from over-suppressing on the season opener.
    _avg_today_temp  = float(np.nanmean(today_temp))
    _avg_today_hum   = float(np.nanmean(today_hum))
    _total_today_rain = float(np.nansum(today_rain))
    _avg_today_wind  = float(np.nanmean(today_wind))
    # Regime detection (mirrors agent_analysis K-means labels, rule-based proxy):
    #   Hot-Dry:            temp > 38°C AND humidity < 40%
    #   Active Monsoon:     total daily rain > 5mm OR (humidity > 65% AND cloud cover avg > 60%)
    #   Pre-Monsoon Humid:  everything else in summer
    _avg_today_cloud = float(np.nanmean(today_cloud)) if today_cloud is not None else 0.0
    if _avg_today_temp > 38.0 and _avg_today_hum < 40.0:
        _regime = "hot_dry"
    elif _total_today_rain > 5.0 or (_avg_today_hum > 65.0 and _avg_today_cloud > 60.0):
        _regime = "active_monsoon"
    else:
        _regime = "pre_monsoon_humid"
    # monsoon_day_number: days since first rain >5mm this calendar month
    # Approximate via counting rain days in the current month up to today in df.
    _monsoon_day_number = 0
    try:
        _target_dt = pd.to_datetime(str(resolved_date)[:10])
        _month_start = _target_dt.replace(day=1)
        _df_month = df[
            (pd.to_datetime(df["date"].astype(str)) >= _month_start) &
            (pd.to_datetime(df["date"].astype(str)) < _target_dt)
        ]
        if not _df_month.empty:
            _daily_rain = _df_month.groupby("date")["precipitation"].sum()
            _rainy_in_month = (_daily_rain > 5.0).sum()
            _monsoon_day_number = int(_rainy_in_month)
    except Exception:
        pass
    # First rain event: this is the first rain day (>5mm) after ≥7 dry days
    _yday_total_rain = float(np.nansum(yday_rain))
    _is_first_rain_event = (
        _total_today_rain > 5.0 and _yday_total_rain <= 1.0 and _monsoon_day_number == 0
    )
    # Regime-specific caps
    if _regime == "hot_dry":
        # AC penetration growing — upper cap raised to +18% for blocks 25-72 (peak periods)
        _peak_mask = (np.arange(1, _BLOCK_COUNT + 1) >= 25) & (np.arange(1, _BLOCK_COUNT + 1) <= 72)
        _wp_cap_lo_vec = np.full(_BLOCK_COUNT, -0.18)
        _wp_cap_hi_vec = np.full(_BLOCK_COUNT, 0.12)
        _wp_cap_hi_vec[_peak_mask] = 0.18  # raised cap for peak blocks in Hot-Dry
    elif _regime == "active_monsoon" and _is_first_rain_event:
        # First rain day: tighten suppression to -10% to prevent over-suppression
        _wp_cap_lo_vec = np.full(_BLOCK_COUNT, -0.10)
        _wp_cap_hi_vec = np.full(_BLOCK_COUNT, 0.08)
        logger.info(
            "[monsoon] First rain event detected (today=%.1fmm, yday=%.1fmm) — "
            "suppression cap tightened to -10%%",
            _total_today_rain, _yday_total_rain,
        )
    elif _regime == "active_monsoon":
        _wp_cap_lo_vec = np.full(_BLOCK_COUNT, -0.15)  # tighter than bare -0.20
        _wp_cap_hi_vec = np.full(_BLOCK_COUNT, 0.10)
    else:  # pre_monsoon_humid — default seasonal caps
        _wp_cap_lo_vec = np.full(_BLOCK_COUNT, {"winter": -0.18, "spring": -0.15, "summer": -0.20, "fall": -0.18}.get(season, -0.18))
        _wp_cap_hi_vec = np.full(_BLOCK_COUNT, 0.12)
    weather_impact_pct = np.clip(weather_impact_pct, _wp_cap_lo_vec, _wp_cap_hi_vec)
    weather_impact_mw = base_vec * weather_impact_pct

    # Adaptive momentum lambda: on high-volatility days (large yesterday deviation)
    # reduce lambda so we don't amplify an anomalous baseline.
    _HIGH_VOLATILITY_MW = 150.0
    deviation = float(np.abs(dod_load_delta).mean())
    if deviation > _HIGH_VOLATILITY_MW:
        lam = float(np.clip(float(momentum_lambda) * 0.4, 0.15, 0.25))
        logger.debug(
            "High dod_load_delta deviation %.1f MW > %.1f threshold — reducing lambda to %.3f",
            deviation, _HIGH_VOLATILITY_MW, lam,
        )
    else:
        lam = float(np.clip(float(momentum_lambda), 0.3, 0.6))
    momentum_impact_mw = lam * dod_load_delta
    momentum_base = np.divide(
        momentum_impact_mw,
        np.maximum(np.abs(base_vec), 1e-6),
        out=np.zeros_like(momentum_impact_mw, dtype=float),
        where=np.abs(base_vec) > 1e-6,
    )
    simulated_load = base_vec + weather_impact_mw + momentum_impact_mw

    block_results_df = pd.DataFrame({
        "date": [resolved_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "base_load": base_vec,
        "temp_delta": temp_delta,
        "humidity_delta": hum_delta,
        "rain_delta": rain_delta,
        "apparent_delta": apparent_delta,
        "cloud_delta": cloud_delta,
        "sun_delta": sun_delta,
        "rad_delta": rad_delta,
        "wind_delta": wind_delta,
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weather_impact_pct": weather_impact_pct,
        "weather_impact_mw": weather_impact_mw,
        "dod_load_delta_mw": dod_load_delta,
        "momentum_lambda": np.full(_BLOCK_COUNT, lam, dtype=float),
        "momentum_impact_mw": momentum_impact_mw,
        "simulated_load_mw": simulated_load,
        "weight_confidence": conf_w,
    })

    return {
        "target_date": resolved_date,
        "previous_date": prev_date,
        "previous_previous_date": prev2_date,
        "momentum_lambda": lam,
        "driver_weight_matrix": weight_df.sort_values("block").reset_index(drop=True),
        "weather_delta": {
            "temperature": temp_delta.tolist(),
            "humidity": hum_delta.tolist(),
            "rain": rain_delta.tolist(),
            "apparent": apparent_delta.tolist(),
            "cloud": cloud_delta.tolist(),
            "sun": sun_delta.tolist(),
            "radiation": rad_delta.tolist(),
            "wind": wind_delta.tolist(),
        },
        "component_base": {
            "temperature": (temp_delta * temp_w).tolist(),
            "humidity": (hum_delta * hum_w).tolist(),
            "precipitation": (rain_delta * rain_w).tolist(),
            "apparent": (apparent_delta * apparent_w).tolist(),
            "cloud": (cloud_delta * cloud_w).tolist(),
            "sun": (sun_delta * sun_w).tolist(),
            "radiation": (rad_delta * rad_w).tolist(),
            "wind": (wind_delta * wind_w).tolist(),
        },
        "weather_impact_pct": weather_impact_pct.tolist(),
        "weather_impact_mw": weather_impact_mw.tolist(),
        "dod_load_delta_mw": dod_load_delta.tolist(),
        "momentum_impact_mw": momentum_impact_mw.tolist(),
        "momentum_base": momentum_base.tolist(),
        "simulated_load_mw": simulated_load.tolist(),
        "weight_confidence": conf_w.tolist(),
        "block_results_df": block_results_df,
        "metadata": {
            "features": feature_cols,
            "train_dates": int(len(sorted(train["date"].astype(str).unique().tolist()))) if not train.empty else 0,
            "min_samples": int(min_samples),
        },
    }


def _coerce_weight_vector(weight_matrix: Any, column: str) -> np.ndarray:
    aliases = {
        "temperature_weight": ("temp_weight", "temp_weight_raw", "temp_weight_v2", "temp"),
        "humidity_weight": ("humidity_weight_raw", "hum_weight", "hum_weight_raw"),
        "precipitation_weight": ("rain_weight", "rain_weight_raw", "precip_weight", "precipitation"),
        "apparent_temperature_weight": ("apparent_weight", "apparent_weight_raw", "apparent_temp_weight"),
        "cloud_cover_weight": ("cloud_weight", "cloud_weight_raw"),
        "sunshine_duration_weight": ("sun_weight", "sun_weight_raw", "sunshine_weight"),
        "direct_radiation_weight": ("rad_weight", "rad_weight_raw", "radiation_weight"),
        "wind_speed_10m_weight": ("wind_weight", "wind_weight_raw", "wind_speed_weight", "wind_speed_10m", "wind"),
        "daytype_weight": ("day_type_weight", "weekend_weight", "daytype"),
        "holiday_weight": ("holiday",),
        "weight_confidence": ("fit_confidence", "sample_confidence", "confidence"),
    }

    if isinstance(weight_matrix, pd.DataFrame):
        use_col = column
        if use_col not in weight_matrix.columns:
            for alt in aliases.get(column, ()):
                if alt in weight_matrix.columns:
                    use_col = alt
                    break
        if use_col not in weight_matrix.columns:
            return np.zeros(_BLOCK_COUNT, dtype=float)
        if "block" in weight_matrix.columns:
            ordered = (
                weight_matrix[["block", use_col]]
                .copy()
                .assign(block=lambda x: pd.to_numeric(x["block"], errors="coerce"))
                .dropna(subset=["block"])
                .sort_values("block")
            )
            vec = ordered[use_col].to_numpy(dtype=float)
            return _as_96_vector(vec)
        return _as_96_vector(weight_matrix[use_col].to_numpy(dtype=float))
    if isinstance(weight_matrix, dict):
        if column in weight_matrix:
            return _as_96_vector(weight_matrix.get(column))
        for alt in aliases.get(column, ()):
            if alt in weight_matrix:
                return _as_96_vector(weight_matrix.get(alt))
        return np.zeros(_BLOCK_COUNT, dtype=float)
    return _as_96_vector(weight_matrix)


def run_block_driver_weight_delta_engine(
    target_date: str,
    baseline_load: np.ndarray,
    weight_matrix: Any,
    delta_inputs: Optional[Dict[str, Any]] = None,
    sliders: Optional[Dict[str, float]] = None,
    selection: Optional[Dict[str, Any]] = None,
    smooth_edges: bool = False,
    base_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    delta_cfg = delta_inputs or {}
    override_cfg = base_overrides or {}

    baseline_vec = _as_96_vector(baseline_load)

    temp_w = _coerce_weight_vector(weight_matrix, "temperature_weight")
    hum_w = _coerce_weight_vector(weight_matrix, "humidity_weight")
    rain_w = _coerce_weight_vector(weight_matrix, "precipitation_weight")
    apparent_w = _coerce_weight_vector(weight_matrix, "apparent_temperature_weight")
    cloud_w = _coerce_weight_vector(weight_matrix, "cloud_cover_weight")
    sun_w = _coerce_weight_vector(weight_matrix, "sunshine_duration_weight")
    rad_w = _coerce_weight_vector(weight_matrix, "direct_radiation_weight")
    wind_w = _coerce_weight_vector(weight_matrix, "wind_speed_10m_weight")
    
    daytype_w = _coerce_weight_vector(weight_matrix, "daytype_weight")
    holiday_w = _coerce_weight_vector(weight_matrix, "holiday_weight")
    weight_conf = _coerce_weight_vector(weight_matrix, "weight_confidence")

    temp_delta = _as_96_vector(delta_cfg.get("temp_delta", delta_cfg.get("temperature_delta", 0.0)))
    hum_delta = _as_96_vector(delta_cfg.get("humidity_delta", delta_cfg.get("hum_delta", 0.0)))
    rain_delta = _as_96_vector(delta_cfg.get("rain_delta", delta_cfg.get("precip_delta", 0.0)))
    apparent_delta = _as_96_vector(delta_cfg.get("apparent_delta", 0.0))
    cloud_delta = _as_96_vector(delta_cfg.get("cloud_delta", 0.0))
    sun_delta = _as_96_vector(delta_cfg.get("sun_delta", 0.0))
    rad_delta = _as_96_vector(delta_cfg.get("rad_delta", 0.0))
    wind_delta = _as_96_vector(delta_cfg.get("wind_delta", 0.0))
    
    daytype_flag = _as_96_vector(delta_cfg.get("daytype_flag", 0.0))
    holiday_flag = _as_96_vector(delta_cfg.get("holiday_flag", 0.0))

    temperature_base = temp_delta * temp_w
    humidity_base = hum_delta * hum_w
    precipitation_base = rain_delta * rain_w
    apparent_base = apparent_delta * apparent_w
    cloud_base = cloud_delta * cloud_w
    sun_base = sun_delta * sun_w
    radiation_base = rad_delta * rad_w
    wind_base = wind_delta * wind_w
    
    daytype_base = daytype_flag * daytype_w
    holiday_base = holiday_flag * holiday_w
    manual_base = _as_96_vector(delta_cfg.get("manual_base", 0.0))

    if override_cfg.get("temperature_base") is not None:
        temperature_base = _as_96_vector(override_cfg.get("temperature_base"))
    if override_cfg.get("humidity_base") is not None:
        humidity_base = _as_96_vector(override_cfg.get("humidity_base"))
    if override_cfg.get("precipitation_base") is not None:
        precipitation_base = _as_96_vector(override_cfg.get("precipitation_base"))
    if override_cfg.get("wind_base") is not None:
        wind_base = _as_96_vector(override_cfg.get("wind_base"))
    if override_cfg.get("daytype_base") is not None:
        daytype_base = _as_96_vector(override_cfg.get("daytype_base"))
    if override_cfg.get("holiday_base") is not None:
        holiday_base = _as_96_vector(override_cfg.get("holiday_base"))
    if override_cfg.get("manual_base") is not None:
        manual_base = _as_96_vector(override_cfg.get("manual_base"))

    weather_base_override = override_cfg.get("weather_base")
    if weather_base_override is not None:
        weather_override_vec = _as_96_vector(weather_base_override)
        has_component_override = any(
            override_cfg.get(key) is not None
            for key in ("temperature_base", "humidity_base", "precipitation_base", "wind_base")
        )
        if not has_component_override:
            temperature_base = weather_override_vec.copy()
            humidity_base = np.zeros(_BLOCK_COUNT, dtype=float)
            precipitation_base = np.zeros(_BLOCK_COUNT, dtype=float)
            wind_base = np.zeros(_BLOCK_COUNT, dtype=float)

    weather_base = temperature_base + humidity_base + precipitation_base + apparent_base + cloud_base + sun_base + radiation_base + wind_base

    slider_cfg = sliders or {}
    s_w = float(slider_cfg.get("weather", slider_cfg.get("S_w", 1.0)))
    s_d = float(slider_cfg.get("daytype", slider_cfg.get("S_d", 1.0)))
    s_h = float(slider_cfg.get("holiday", slider_cfg.get("S_h", 1.0)))
    s_m = float(slider_cfg.get("manual", slider_cfg.get("S_m", 1.0)))
    s_temp = float(slider_cfg.get("temperature", 1.0))
    s_hum = float(slider_cfg.get("humidity", 1.0))
    s_prec = float(slider_cfg.get("precipitation", 1.0))
    s_apparent = float(slider_cfg.get("apparent_temperature", 1.0))
    s_cloud = float(slider_cfg.get("cloud_cover", 1.0))
    s_sun = float(slider_cfg.get("sunshine_duration", 1.0))
    s_rad = float(slider_cfg.get("direct_radiation", 1.0))
    s_wind = float(slider_cfg.get("wind", 1.0))

    temperature_scaled = temperature_base * s_w * s_temp
    humidity_scaled = humidity_base * s_w * s_hum
    precipitation_scaled = precipitation_base * s_w * s_prec
    apparent_scaled = apparent_base * s_w * s_apparent
    cloud_scaled = cloud_base * s_w * s_cloud
    sun_scaled = sun_base * s_w * s_sun
    rad_scaled = radiation_base * s_w * s_rad
    wind_scaled = wind_base * s_w * s_wind
    
    weather_scaled = (
        temperature_scaled + humidity_scaled + precipitation_scaled + 
        apparent_scaled + cloud_scaled + sun_scaled + rad_scaled + wind_scaled
    )
    daytype_scaled = daytype_base * s_d
    holiday_scaled = holiday_base * s_h
    manual_scaled = manual_base * s_m

    final_impact = np.clip(
        weather_scaled + daytype_scaled + holiday_scaled + manual_scaled,
        _IMPACT_MIN,
        _IMPACT_MAX,
    )

    mask = _selection_mask(selection)
    if smooth_edges:
        mask = _smoothed_selection_mask(mask)
    final_load = baseline_vec * (1.0 + (final_impact * mask))

    matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weight_confidence": weight_conf,
        "temp_delta": temp_delta,
        "humidity_delta": hum_delta,
        "rain_delta": rain_delta,
        "apparent_delta": apparent_delta,
        "cloud_delta": cloud_delta,
        "sun_delta": sun_delta,
        "rad_delta": rad_delta,
        "wind_delta": wind_delta,
        "weather_base": weather_base,
        "temperature_base": temperature_base,
        "humidity_base": humidity_base,
        "precipitation_base": precipitation_base,
        "apparent_base": apparent_base,
        "cloud_base": cloud_base,
        "sun_base": sun_base,
        "radiation_base": radiation_base,
        "wind_base": wind_base,
        "weather_scaled": weather_scaled,
        "temperature_scaled": temperature_scaled,
        "humidity_scaled": humidity_scaled,
        "precipitation_scaled": precipitation_scaled,
        "apparent_scaled": apparent_scaled,
        "cloud_scaled": cloud_scaled,
        "sun_scaled": sun_scaled,
        "rad_scaled": rad_scaled,
        "wind_scaled": wind_scaled,
        "final_impact": final_impact,
    })

    driver_weight_matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weight_confidence": weight_conf,
    })

    return {
        "matrix_df": matrix,
        "driver_weight_matrix": driver_weight_matrix,
        "selection_mask": mask,
        "impact_vector": final_impact,
        "final_load": final_load,
    }


def _rank_block_contributors(contrib_map: Dict[str, float], baseline_mw: Optional[float] = None) -> List[Dict[str, Any]]:
    total_abs = sum(abs(v) for v in contrib_map.values()) or 1.0
    denom_mw = float(baseline_mw) if baseline_mw is not None and np.isfinite(float(baseline_mw)) else None
    ranked = sorted(contrib_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    rows = []
    for feature, val in ranked:
        load_pct = 0.0
        if denom_mw is not None and abs(denom_mw) > 1e-6:
            load_pct = float((float(val) / denom_mw) * 100.0)
        rows.append({
            "feature": feature,
            "contribution_mw": float(val),
            "contribution_pct": float((val / total_abs) * 100.0),
            "contribution_load_pct": load_pct,
            "direction": "up" if val >= 0 else "down",
        })
    return rows


def _recommended_action(primary_feature: str, risk_flag: str, net_impact_pct: float) -> str:
    if risk_flag == "high":
        if primary_feature == "weather":
            return "Increase reserve and monitor weather-driven slots."
        if primary_feature == "calendar":
            return "Review day-type/holiday assumptions before commitment."
        if primary_feature == "manual":
            return "Reduce manual override and apply staged correction."
        return "Trigger operator review and prepare corrective dispatch."
    if risk_flag == "medium":
        if primary_feature in ("weather", "calendar"):
            return "Monitor slot and pre-position balancing support."
        return "Track slot and apply guarded adjustment if deviation grows."
    if abs(net_impact_pct) >= 8:
        return "Keep slot under watch; no immediate intervention."
    return "No action required."


def _detect_peak_blocks(
    df: pd.DataFrame,
    target_date: str,
    lookback_days: int = 30,
    percentile_threshold: float = 75.0,
    min_peak_width: int = 8,
) -> Tuple[int, int]:
    """
    Dynamically detect the peak-load window from historical data.

    Steps:
    1. Take up to `lookback_days` of same-season history before target_date.
    2. Compute median load per time_block across those days.
    3. Mark blocks above the `percentile_threshold`-th percentile of the
       median profile as "peak".
    4. Find the longest contiguous run of peak blocks; return its
       1-based (start, end) inclusive of at least `min_peak_width` blocks.
    5. Falls back to (33, 72) if data is insufficient.

    Returns (peak_start_block, peak_end_block) — 1-based, inclusive end.
    """
    _fallback = (33, 72)
    try:
        work = df.copy()
        work["date"] = work["date"].astype(str)
        if target_date:
            work = work[work["date"] < str(target_date)]
        if "total_drawal" not in work.columns or "time_block" not in work.columns:
            return _fallback

        # Same-season filter using _season helper
        try:
            tgt_season = _season(target_date)
            work["_season"] = work["date"].apply(_season)
            work = work[work["_season"] == tgt_season]
        except Exception:
            pass

        # Limit to lookback window
        available_dates = sorted(work["date"].dropna().unique().tolist())
        if not available_dates:
            return _fallback
        keep = set(available_dates[-int(lookback_days):])
        work = work[work["date"].isin(keep)]

        work["time_block"] = pd.to_numeric(work["time_block"], errors="coerce")
        work = work.dropna(subset=["time_block", "total_drawal"])
        work = work[work["time_block"].between(1, _BLOCK_COUNT)]
        if len(work["date"].unique()) < 3:
            return _fallback

        # Median load profile across history days
        median_profile = (
            work.groupby("time_block")["total_drawal"]
            .median()
            .reindex(range(1, _BLOCK_COUNT + 1))
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )

        threshold = float(np.percentile(median_profile, percentile_threshold))
        is_peak = median_profile >= threshold  # boolean mask, 0-indexed

        # Find longest contiguous True run
        best_start, best_end, cur_start = 0, 0, None
        for i in range(len(is_peak)):
            if is_peak[i]:
                if cur_start is None:
                    cur_start = i
                if (i - cur_start + 1) > (best_end - best_start):
                    best_start, best_end = cur_start, i + 1
            else:
                cur_start = None

        if (best_end - best_start) < min_peak_width:
            return _fallback

        # Convert 0-indexed → 1-based block numbers
        return int(best_start + 1), int(best_end)
    except Exception:
        return _fallback


# ── Accuracy Enhancement Functions ───────────────────────────────────────────

REGION_ACCURACY_CONFIG: Dict[str, Dict] = {
    "haryana": {
        "ramp_weight_multiplier": 1.3,
        "heat_onset_c": 30,
        "anomaly_threshold_pct": 6,
        "weekend_baseline_days": 14,
        "t2_night_fallback_mw": 350.0,
        "bias_window_days": {"apr-jun": 14, "default": 30},
    },
    "odisha": {
        "ramp_weight_multiplier": 1.2,
        "heat_onset_c": 28,
        "anomaly_threshold_pct": 8,
        "weekend_baseline_days": 10,
        "t2_night_fallback_mw": 280.0,
        "bias_window_days": {"jun-sep": 10, "default": 30},
    },
    "rajasthan": {
        "ramp_weight_multiplier": 1.3,
        "heat_onset_c": 32,
        "anomaly_threshold_pct": 6,
        "weekend_baseline_days": 14,
        "t2_night_fallback_mw": 400.0,
        "bias_window_days": {"apr-jun": 14, "default": 30},
    },
    "chhattisgarh": {
        "ramp_weight_multiplier": 1.25,
        "heat_onset_c": 29,
        "anomaly_threshold_pct": 7,
        "weekend_baseline_days": 12,
        "t2_night_fallback_mw": 320.0,
        "bias_window_days": {"default": 30},
    },
}


def intraday_recalibrate(
    forecast_96: np.ndarray,
    settled_actuals: np.ndarray,
    settled_count: int,
) -> np.ndarray:
    """Apply morning-actuals-based correction to unsettled forecast blocks.

    Uses settled morning actuals (blocks 1..settled_count) as a correction
    signal.  Applies a decaying additive bias to remaining blocks so the
    forecast converges toward observed morning behaviour while still using
    the pipeline's shape for the rest of the day.

    Returns the corrected 96-block forecast (or unchanged if correction
    conditions are not met).
    """
    forecast_96 = np.asarray(forecast_96, dtype=float)
    if forecast_96.size != 96:
        return forecast_96
    if settled_count < 8:
        return forecast_96
    if settled_count >= 96:
        return forecast_96

    sa = np.asarray(settled_actuals[:settled_count], dtype=float)
    sf = forecast_96[:settled_count]
    valid = (sa > 10) & (sf > 10)
    if valid.sum() < 4:
        return forecast_96

    morning_bias = float(np.mean(sa[valid] - sf[valid]))
    morning_mape = float(
        np.mean(np.abs(sa[valid] - sf[valid]) / np.maximum(sa[valid], 1.0)) * 100
    )
    if morning_mape < 2.0:
        return forecast_96

    remaining = 96 - settled_count
    taper = np.linspace(1.0, 0.0, remaining)
    corrected = forecast_96.copy()
    corrected[settled_count:] += morning_bias * taper
    return np.maximum(corrected, 0.0)


def t2_night_anchor(
    t2_forecast: np.ndarray,
    t1_settled_tail: Optional[np.ndarray],
    t1_forecast_tail: Optional[np.ndarray],
) -> np.ndarray:
    """Carry T+1 end-of-day bias forward into T+2 opening blocks.

    If T+1's last 4 settled blocks deviate from their forecast, that bias
    is propagated into T+2 blocks 1–32 with exponential decay so the
    T+2 curve doesn't cold-start from an incorrect level.
    """
    t2_forecast = np.asarray(t2_forecast, dtype=float)
    if t1_settled_tail is None or t1_forecast_tail is None:
        return t2_forecast
    tail_a = np.asarray(t1_settled_tail, dtype=float)
    tail_f = np.asarray(t1_forecast_tail, dtype=float)
    if tail_a.size < 4 or tail_f.size < 4:
        return t2_forecast
    valid = (tail_a > 10) & (tail_f > 10)
    if valid.sum() < 2:
        return t2_forecast
    t1_tail_bias = float(np.mean(tail_a[valid] - tail_f[valid]))
    if abs(t1_tail_bias) < 5.0:
        return t2_forecast
    decay = np.exp(-np.arange(96) / 16.0)
    corrected = t2_forecast + t1_tail_bias * decay
    return np.maximum(corrected, 0.0)


def run_short_term_pipeline(df: pd.DataFrame, target_date: str, actual_blocks: int = 40, config: Optional[Dict] = None) -> Dict:
    import time as _time
    _t0 = _time.monotonic()

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Apply region-specific seasonal overrides (REGION_ACCURACY_CONFIG was defined but never merged)
    try:
        _region_key = str(cfg.get("region", "haryana")).lower()
        _r_cfg = REGION_ACCURACY_CONFIG.get(_region_key, {})
        if _r_cfg:
            _tgt_month = pd.Timestamp(str(target_date)[:10]).month
            _bwd = _r_cfg.get("bias_window_days", {})
            if isinstance(_bwd, dict):
                if _tgt_month in (4, 5, 6) and "apr-jun" in _bwd:
                    _seasonal_window = int(_bwd["apr-jun"])
                elif _tgt_month in (6, 7, 8, 9) and "jun-sep" in _bwd:
                    _seasonal_window = int(_bwd["jun-sep"])
                else:
                    _seasonal_window = int(_bwd.get("default", 30))
                if "accuracy_corrections" not in cfg or not isinstance(cfg.get("accuracy_corrections"), dict):
                    cfg["accuracy_corrections"] = {}
                cfg["accuracy_corrections"].setdefault("block_bias_window_days", _seasonal_window)
    except Exception:
        pass

    accuracy_cfg = cfg.get("accuracy_corrections", {}) if isinstance(cfg.get("accuracy_corrections", {}), dict) else {}
    ai_cfg = cfg.get("hybrid_ai", {}) if isinstance(cfg.get("hybrid_ai", {}), dict) else {}
    # Extract callback (not a real config param, remove before use)
    _progress_cb = cfg.pop("progress_callback", None)
    _diag = bool(cfg.get("debug_stages", False))

    def _step(msg: str):
        elapsed = _time.monotonic() - _t0
        logger.info("[pipeline:%s] %s  (%.1fs elapsed)", target_date, msg, elapsed)
        if _progress_cb:
            try:
                _progress_cb({"type": "progress", "message": msg, "elapsed": round(elapsed, 1)})
            except Exception:
                pass

    def _diag_vec(stage: str, vec: np.ndarray) -> None:
        """Log a compact shape summary for a 96-block vector when debug_stages=True."""
        if not _diag:
            return
        try:
            v = np.asarray(vec, dtype=float)
            if v.size == 0:
                logger.info("[diag:%s][%s] empty vector", target_date, stage)
                return
            peak_blk = int(np.argmax(v)) + 1
            valley_blk = int(np.argmin(v)) + 1
            logger.info(
                "[diag:%s][%s] mean=%.1f  peak=%.1f@blk%d  valley=%.1f@blk%d  std=%.1f",
                target_date, stage,
                float(np.nanmean(v)),
                float(np.nanmax(v)), peak_blk,
                float(np.nanmin(v)), valley_blk,
                float(np.nanstd(v)),
            )
        except Exception:
            pass

    _step("▶ START")
    cfg["similarity_weights"] = _normalize_similarity_weights(cfg.get("similarity_weights", {}))
    actual_blocks = int(np.clip(actual_blocks, 0, 96))  # cap at full day, no static config limit
    if actual_blocks < 2:
        actual_blocks = 0

    df = _ensure_time_block_1_based(df)
    if "date" not in df.columns:
        raise ValueError("date column missing")
    df["date"] = df["date"].astype(str)
    # Ensure optional weather columns exist (may be absent in older datasets)
    for _oc in ["apparent_temperature", "cloud_cover", "sunshine_duration",
                 "direct_radiation", "wind_speed_10m", "cloud_cover_low",
                 "wind_speed_80m"]:
        if _oc not in df.columns:
            df[_oc] = 0.0

    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        # Find latest robust date
        target_date = dates[-1]
        for d in reversed(dates):
            day_df = df[df["date"] == d]
            if day_df["time_block"].nunique() >= 96 and day_df["total_drawal"].mean() > cfg.get("min_valid_load_mw", 100):
                target_date = d
                break

    external_response = _maybe_run_external_final_forecast(
        df=df,
        target_date=str(target_date),
        actual_blocks=int(actual_blocks),
        cfg=cfg,
    )
    if external_response is not None:
        _step("✓ DONE — external final forecast complete")
        return external_response


    similar_days = pd.DataFrame()
    best_window = None
    best_mape = None
    similarity_profile = {"source": "lstm_primary", "diagnostics": {}}

    target_df = _coerce_day_to_96_blocks(df[df["date"] == target_date], fill_load=False)
    if target_df.empty:
        raise ValueError("Target date not available")
    if int(target_df["time_block"].nunique()) != _BLOCK_COUNT:
        raise ValueError(f"Target date {target_date} does not contain {_BLOCK_COUNT} aligned blocks")

    season = _season(target_date)
    day_type = _day_type(target_date)

    # Compute holiday flags for target date (used by both behaviour and calendar)
    _hf_df_early = _compute_holiday_flags(
        df[df["date"] == str(target_date)], region=cfg.get("region", "haryana")
    )
    target_holiday_flags = {}
    if not _hf_df_early.empty:
        _hf_row_early = _hf_df_early.iloc[0]
        for _hfc in ["is_holiday", "bridge_day", "long_weekend", "holiday_before_weekend",
                      "holiday_after_weekend", "post_weekend_holiday", "mid_week_holiday",
                      "days_to_holiday", "days_since_holiday"]:
            target_holiday_flags[_hfc] = int(_hf_row_early.get(_hfc, 0))

    # Human behaviour adjustment based on region, season, day type
    hb_region = str(cfg.get("region", "haryana"))
    hb_weight = float(cfg.get("human_behaviour_weight", 1.0))

    # Calibrate behaviour from historical data if enabled
    if cfg.get("calibrate_behaviour_from_data", True):
        global _CALIBRATED_BEHAVIOUR
        _CALIBRATED_BEHAVIOUR = _calibrate_behaviour_profiles(df, hb_region, cfg)

    # Use extended day type if holiday flags available
    hb_day_type = _day_type_extended(target_date, target_holiday_flags) if target_holiday_flags else day_type
    human_behaviour_adj, hb_profile_label = _human_behaviour_adjustment(
        season, hb_region, hb_day_type, weight=hb_weight
    )

    actual = target_df["total_drawal"].to_numpy(dtype=float)
    actual_partial_early = actual[:actual_blocks]

    _step("1/3 running LSTM primary forecaster...")
    import os as _os
    _artifacts_dir = _os.path.join(_os.path.dirname(__file__), "artifacts")

    # Slice df to the UI-selected date range so LSTM trains only on chosen data
    _train_from = cfg.get("train_from_date")
    _train_to   = cfg.get("train_to_date")
    _df_for_lstm = df.copy()
    if _train_from:
        _df_for_lstm = _df_for_lstm[_df_for_lstm["date"].astype(str) >= str(_train_from)[:10]]
    if _train_to:
        _df_for_lstm = _df_for_lstm[_df_for_lstm["date"].astype(str) <= str(_train_to)[:10]]
    if _df_for_lstm.empty:
        logger.warning("train_range filter left 0 rows — falling back to full df for LSTM")
        _df_for_lstm = df.copy()

    # ── Weather anomaly detection (before model call) ─────────────────────────
    _weather_result = None
    if detect_weather_anomalies is not None:
        try:
            _weather_result = detect_weather_anomalies(
                target_df=target_df,
                config=cfg.get("weather_anomaly", {}),
            )
            if _weather_result.severity not in ("NONE", "LOW"):
                logger.info(
                    "[pipeline:%s] weather anomaly detected — severity=%s: %s",
                    target_date, _weather_result.severity, _weather_result.description,
                )
        except Exception as _wa_err:
            logger.debug("Weather anomaly detection failed: %s", _wa_err)
            _weather_result = None

    # Build exempt_blocks: union of all weather-flagged block lists
    _exempt_blocks: set = set()
    if _weather_result is not None:
        for _blk_list in _weather_result.block_flags.values():
            _exempt_blocks.update(_blk_list)

    _lstm_vec = None
    _lstm_t2_vec = None
    if run_helper_forecast is not None:
        _lstm_result = run_helper_forecast(
            df=_df_for_lstm,
            target_date=target_date,
            region=cfg.get("region", "unknown"),
            actual_blocks=actual_blocks,
            actual_partial=actual_partial_early,
            artifacts_dir=_artifacts_dir,
            retrain_every_hours=float(cfg.get("helper_ai", {}).get("retrain_every_hours", 24)),
            blend_weight=1.0,
            return_t2=True,
            clamp_sigma_override=_weather_result.clamp_sigma_override if _weather_result else None,
            correction_decay_override=_weather_result.correction_decay_override if _weather_result else None,
            exempt_blocks=_exempt_blocks if _exempt_blocks else None,
        )
        if isinstance(_lstm_result, tuple):
            _lstm_vec, _lstm_t2_vec = _lstm_result
        else:
            _lstm_vec = _lstm_result   # fallback: old single-return path

    _target_month = pd.Timestamp(str(target_date)[:10]).month
    if _target_month in (4, 5, 6):
        # May pre-monsoon is highly volatile; 3 days is too narrow when all 3 are anomalous.
        # Check if recent days had elevated load variance — if so, widen to 5 to dilute anomalies.
        try:
            _pre = df[df["date"].astype(str) < str(target_date)].copy()
            _pre_dates = sorted(_pre["date"].dropna().astype(str).unique())[-5:]
            if len(_pre_dates) >= 3:
                _daily_means = [_pre[_pre["date"].astype(str) == _d]["total_drawal"].mean() for _d in _pre_dates]
                _cv = float(np.std(_daily_means) / (np.mean(_daily_means) + 1e-6))
                _adaptive_window = 5 if _cv > 0.05 else 3  # extend if day-to-day load varies >5%
            else:
                _adaptive_window = 3
        except Exception:
            _adaptive_window = 3
    elif _target_month in (7, 8, 9):
        # Check for monsoon onset: if recent days show rainfall, patterns are changing fast →
        # use a shorter window so the baseline follows the new regime rather than dry-season history.
        try:
            _pre_m = df[df["date"].astype(str) < str(target_date)].copy()
            _pre_dates_m = sorted(_pre_m["date"].dropna().astype(str).unique())[-7:]
            _rain_col = "precipitation" if "precipitation" in _pre_m.columns else ("rain" if "rain" in _pre_m.columns else None)
            if _rain_col:
                _daily_rain = _pre_m[_pre_m["date"].astype(str).isin(_pre_dates_m)].groupby("date")[_rain_col].sum()
                _rainy_days = int((_daily_rain > 3.0).sum())
                _adaptive_window = 3 if _rainy_days >= 3 else 5  # monsoon active → shorter window
            else:
                _adaptive_window = 5
        except Exception:
            _adaptive_window = 5
    else:
        _adaptive_window = 7
    _hist_for_base = df[df["date"].astype(str) < str(target_date)].copy()
    _valid_dates = []
    for _d in sorted(_hist_for_base["date"].dropna().astype(str).unique().tolist()):
        _dd = _hist_for_base[_hist_for_base["date"].astype(str) == _d]
        if int(_dd["time_block"].nunique()) >= 96 and float(_dd["total_drawal"].mean()) > cfg.get("min_valid_load_mw", 100):
            _valid_dates.append(_d)
    _hist_for_base = _hist_for_base[_hist_for_base["date"].astype(str).isin(set(_valid_dates[-_adaptive_window:]))]
    best_window = _adaptive_window
    _recent_avg = _hist_for_base.groupby("time_block")["total_drawal"].mean()
    statistical_baseline = _recent_avg.reindex(range(1, 97)).ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    if statistical_baseline.sum() == 0:
        logger.error(
            "[pipeline:%s] Similar-day baseline is all-zeros (no valid dates found in window=%d). "
            "Falling back to full-history 7-day mean.",
            target_date, _adaptive_window,
        )
        _fb_hist = df[df["date"].astype(str) < str(target_date)].copy()
        _fb_avg = _fb_hist.groupby("time_block")["total_drawal"].mean()
        statistical_baseline = _fb_avg.reindex(range(1, 97)).ffill().bfill().fillna(3000.0).to_numpy(dtype=float)

    if _lstm_vec is not None and len(_lstm_vec) == 96 and np.any(statistical_baseline > 0):
        lstm_full = _as_96_vector(_lstm_vec)
        lstm_residual = lstm_full - statistical_baseline
        residual_cap = np.maximum(np.abs(statistical_baseline) * 0.12, 250.0)
        lstm_residual = np.clip(lstm_residual, -residual_cap, residual_cap)
        residual_weight = float(np.clip(float(ai_cfg.get("residual_weight", ai_cfg.get("blend_weight", 0.35))), 0.0, 0.60))
        baseline_hist = statistical_baseline + (residual_weight * lstm_residual)
        hybrid_ai_forecast_vec = baseline_hist.copy()
    else:
        logger.warning("LSTM primary forecaster unavailable — falling back to adaptive recent-day average")
        baseline_hist = statistical_baseline.copy()

    weather_coefs = {}
    weather_pred = baseline_hist.copy()
    weather_training_frame = pd.DataFrame()

    # ── Detect drastic weather on target day ─────────────────────────────────
    # Gusty winds (>25 km/h), heavy rain/showers, snowfall all qualify.
    # Used to widen bias-cap and adjust downstream corrections.
    _tgt_wx = target_df.sort_values("time_block")
    _wx_rain   = float(_tgt_wx["rain"].mean())    if "rain"          in _tgt_wx.columns else 0.0
    _wx_shower = float(_tgt_wx["showers"].mean()) if "showers"       in _tgt_wx.columns else 0.0
    _wx_snow   = float(_tgt_wx["snowfall"].mean()) if "snowfall"      in _tgt_wx.columns else 0.0
    if "wind_speed_10m" in _tgt_wx.columns:
        _wx_wind = float(_tgt_wx["wind_speed_10m"].max())
    else:
        _wx_wind = 12.0  # Haryana climatological average ~12 km/h; 0.0 would suppress drastic-weather detection
        logger.warning("[pipeline:%s] weather column 'wind_speed_10m' missing — using climatological default 12.0 km/h", target_date)
    # Hub-height wind (80m) — proxy for SLDC feeder-protection threshold.
    # 80m wind is what transmission towers / conductors experience; surface (10m)
    # is a demand-side comfort signal and is kept separate.
    if "wind_speed_80m" in _tgt_wx.columns and float(_tgt_wx["wind_speed_80m"].max()) > 0:
        _wx_wind_80m = float(_tgt_wx["wind_speed_80m"].max())
    else:
        _wx_wind_80m = _wx_wind * 1.4   # empirical: 80m ≈ 1.4× surface over open terrain
        logger.debug("[pipeline:%s] wind_speed_80m absent — estimated from 10m (×1.4)", target_date)
    _sldc_threshold  = float(cfg.get("sldc_wind_threshold_kmh", 65.0))
    _high_wind_risk  = _wx_wind_80m >= _sldc_threshold
    if _high_wind_risk:
        logger.warning(
            "[pipeline:%s] HIGH WIND RISK: wind_80m=%.1f km/h >= SLDC threshold %.1f — "
            "feeder-cut load shedding likely; shedding correction will be applied",
            target_date, _wx_wind_80m, _sldc_threshold,
        )

    _wx_precip_severity = min(1.0, (_wx_rain + _wx_shower * 1.5 + _wx_snow * 5.0) / 5.0)
    _wx_wind_severity   = max(0.0, (_wx_wind - float(cfg.get("wind_gust_threshold_kmh", 25.0))) / 30.0)
    _drastic_weather    = (_wx_precip_severity > 0.2 or _wx_wind_severity > 0.3 or _wx_snow > 0.5)
    if _drastic_weather:
        logger.info("[pipeline:%s] drastic weather detected — precip_sev=%.2f wind_sev=%.2f snow=%.2f",
                    target_date, _wx_precip_severity, _wx_wind_severity, _wx_snow)

    # ── Residual bias correction on LSTM output ────────────────────────────────
    _step("2/3 applying residual bias correction...")
    if bool(cfg.get("use_bias_correction", True)) and np.any(baseline_hist > 0):
        try:
            _bc_dates = sorted(df["date"].dropna().unique().tolist())
            _bc_target = target_date if target_date in _bc_dates else (_bc_dates[-1] if _bc_dates else target_date)
            _bc_idx = _bc_dates.index(_bc_target) if _bc_target in _bc_dates else len(_bc_dates)
            if _bc_idx >= 12:
                _bc_load_level = max(float(baseline_hist[baseline_hist > 0].mean()), 100.0)

                # Weather-segmented bias pool: on rain/wind days, restrict the
                # residual pool to same-weather-type days so dry-day bias
                # doesn't over-correct a suppressed load day.
                _bc_dates_pool = _bc_dates
                _wa_rain  = _weather_result.day_flags.get("rain_alert", False) if _weather_result else False
                _wa_wind  = _weather_result.day_flags.get("wind_alert", False) if _weather_result else False
                if (_wa_rain or _wa_wind) and "rain" in df.columns:
                    def _is_weather_day(d):
                        _ddf = df[df["date"].astype(str).str[:10] == str(d)[:10]]
                        if _ddf.empty:
                            return False
                        _r = float(_ddf["rain"].mean()) if "rain" in _ddf.columns else 0.0
                        _s = float(_ddf.get("showers", pd.Series([0])).mean())
                        _w = float(_ddf["wind_speed_10m"].max()) if "wind_speed_10m" in _ddf.columns else 0.0
                        return (_r + _s) > 1.0 or _w > 25.0
                    _filtered = [d for d in _bc_dates if _is_weather_day(d)]
                    if len(_filtered) >= 3:
                        _bc_dates_pool = _filtered

                # Recompute _bc_idx relative to the (possibly filtered) pool
                _bc_pool_target = _bc_target if _bc_target in _bc_dates_pool else (
                    _bc_dates_pool[-1] if _bc_dates_pool else _bc_target
                )
                _bc_pool_idx = (_bc_dates_pool.index(_bc_pool_target)
                                if _bc_pool_target in _bc_dates_pool
                                else len(_bc_dates_pool))
                _residual_bias = _compute_residual_bias(df, _bc_dates_pool, _bc_pool_idx)
                # HIGH/CRITICAL anomaly days get a wider cap (15%)
                _base_cap_pct = float(cfg.get("bias_cap_pct", 0.10))
                _wa_sev = (_weather_result.severity if _weather_result else "NONE")
                if _wa_sev in ("HIGH", "CRITICAL"):
                    _bc_cap_mult = 1.50
                elif _drastic_weather:
                    _bc_cap_mult = 1.20
                else:
                    _bc_cap_mult = 1.0
                _bc_cap = _base_cap_pct * _bc_cap_mult * _bc_load_level
                _residual_bias = np.clip(_residual_bias, -_bc_cap, _bc_cap)
                baseline_hist = baseline_hist + _residual_bias
                _diag_vec("baseline_post_bias", baseline_hist)
        except Exception as _bc_err:
            logger.error("Residual bias correction failed: %s", _bc_err, exc_info=True)
            # baseline_hist remains uncorrected — pipeline continues with unbiased baseline

    weather_impact_block = np.zeros(96, dtype=float)
    weather_baseline = baseline_hist.copy()
    weather_dev = 0.0
    # This LSTM-primary path does not train the legacy weather residual model.
    # Keep the legacy KPI/reporting vectors defined so final assembly can still
    # expose component diagnostics without failing late in the run.
    temp_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    hum_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    rain_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    apparent_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    cloud_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    sun_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    rad_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    wind_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    coef_temp = 0.0
    coef_hum = 0.0
    coef_rain = 0.0
    temp_coeff_vec = np.zeros(_BLOCK_COUNT, dtype=float)
    temp_up_coeff = np.zeros(_BLOCK_COUNT, dtype=float)
    temp_down_coeff = np.zeros(_BLOCK_COUNT, dtype=float)
    temp_profile = {
        "source": "lstm_primary_no_legacy_weather_model",
        "prior_coeff": coef_temp,
        "global_increase_coeff": coef_temp,
        "global_reduction_coeff": coef_temp,
        "global_increase_samples": 0,
        "global_reduction_samples": 0,
        "diagnostics": {},
    }
    temperature_impact_block = temp_delta * coef_temp

    # ══════════════════════════════════════════════════════════════════
    # HYBRID ADDITIVE FORECAST MODEL
    # ──────────────────────────────────────────────────────────────────
    # Forecast = Baseline + Weather_Residual + Calendar_Offset
    #          + Trend_Correction + Boundary_Correction
    #
    # Each component is additive MW — transparent, auditable, no
    # multiplicative compounding. The boundary correction anchors the
    # forecast to recent actual levels with exponential decay.
    # ══════════════════════════════════════════════════════════════════

    # Calendar factor (holiday/weekend/transition multiplier → converted to additive MW)
    calendar_factors = _calendar_factor(df, cfg.get("calendar_config"), holiday_flags=target_holiday_flags)
    calendar_factor_total = calendar_factors.get("total", np.ones(96))
    cal_daytype_factor = calendar_factors.get("day_type", np.ones(96))
    cal_holiday_factor = calendar_factors.get("holiday", np.ones(96))
    cal_transition_factor = calendar_factors.get("transition", np.ones(96))

    actual_full = actual.copy()
    actual_partial = actual[:actual_blocks]
    safe_baseline = np.maximum(baseline_hist, 1.0)

    # ── Component 1: Weather Residual (direct ML additive correction) ──
    # weather_impact_block is already in MW, gated by divergence
    weather_residual = weather_impact_block.copy()

    # ── Component 2: Calendar Offset (additive MW from factor) ─────────
    # Convert multiplicative calendar factor to additive MW offset
    calendar_offset = baseline_hist * (calendar_factor_total - 1.0)

    # ── Component 3: Piecewise Trend Correction (v3.0) ─────────────────
    # Fit separate slopes for short-term (14d) vs long-term (60d) residuals.
    # If short-term diverges >2× from long-term, trust short-term (regime change).
    # Clip is proportional to recent load (±5%) instead of fixed ±200 MW.
    similar_dates = []
    trend_correction = np.zeros(96, dtype=float)

    # ── Component 4: Boundary Correction (anchor to actual level) ──────
    # Ratio-based correction over a wide window captures proportional level deviation.
    # Additive correction captures residual absolute offset. Both decay over 48 blocks (~12 hrs).
    boundary_window = min(12, actual_blocks)   # additive window (recent blocks)
    ratio_window = min(32, actual_blocks)       # ratio window (wider = more representative)
    boundary_bias_mw = 0.0
    level_ratio = 1.0
    if boundary_window > 0:
        recent_actual = actual_partial[-boundary_window:]
        recent_baseline = baseline_hist[actual_blocks - boundary_window:actual_blocks]
        boundary_bias_mw = float(np.median(recent_actual - recent_baseline))
    if ratio_window > 0:
        ratio_actual = actual_partial[-ratio_window:]
        ratio_baseline = baseline_hist[actual_blocks - ratio_window:actual_blocks]
        _valid = ratio_baseline > 100
        if _valid.sum() >= 2:
            level_ratio = float(np.clip(
                np.median(ratio_actual[_valid] / ratio_baseline[_valid]),
                0.75, 1.25,
            ))

    # Level-adjust baseline_hist only for the actual period so the Pattern Fit chart
    # shows baseline aligned with actual. Forecast period baseline is left unchanged
    # (the boundary_correction below handles the smooth level transition there).
    if level_ratio != 1.0 and actual_blocks > 0:
        for b in range(actual_blocks):
            baseline_hist[b] = baseline_hist[b] * level_ratio
        safe_baseline = np.maximum(baseline_hist, 1.0)

    # Additive and proportional boundary corrections for forecast period.
    # The additive term handles MW offset; the proportional term handles days
    # where live actuals are consistently below/above the LSTM baseline.
    # On drastic-weather days the decay half-life is halved (24 blocks instead
    # of 48) so the anchor to known actuals persists deeper into the peak window.
    boundary_correction = np.zeros(96, dtype=float)
    boundary_level_correction = np.zeros(96, dtype=float)
    _decay_half = 24.0 if _drastic_weather else 48.0
    for b in range(actual_blocks, 96):
        distance = b - actual_blocks + 1
        decay = np.exp(-0.693 * distance / _decay_half)
        boundary_correction[b] = boundary_bias_mw * decay
        if actual_blocks > 0 and abs(level_ratio - 1.0) > 1e-4:
            boundary_level_correction[b] = baseline_hist[b] * (level_ratio - 1.0) * decay

    sim_residual_shape = np.zeros(96, dtype=float)

    india_adj = np.zeros(96, dtype=float)
    india_meta = {}

    _step("3/3 assembling final forecast...")
    forecast = np.zeros(96, dtype=float)
    for b in range(96):
        if b < actual_blocks:
            forecast[b] = actual[b]
        else:
            forecast[b] = (
                baseline_hist[b]
                + calendar_offset[b]
                + boundary_level_correction[b]
            )
    forecast = np.maximum(forecast, 0)
    _diag_vec("forecast_pre_stitch", forecast)

    if "hybrid_ai_forecast_vec" not in locals():
        hybrid_ai_forecast_vec = None


    # ── Legacy variables for downstream KPI reporting ──────────────────
    boundary_ratio = float(np.median(actual_partial / safe_baseline[:actual_blocks]) if actual_blocks > 0 else 1.0)
    hybrid = baseline_hist + boundary_level_correction  # approximate hybrid for KPIs
    hybrid_pre_calendar = hybrid.copy()
    calendar_adjustment_block = calendar_offset
    bias_factor = boundary_ratio
    bias_applied = (baseline_hist + boundary_level_correction).copy()
    alpha = 1.0
    beta = 0.0
    trend_slope = float(trend_correction[48]) if len(trend_correction) > 48 else 0.0
    trend_component = np.zeros(actual_blocks)
    trend_full = trend_correction.copy()
    rain_coeff = cfg["rain_coeffs"].get(season, 20.0)
    # Rain impact via log-saturation
    rain_impact_vec = np.full(96, rain_coeff, dtype=float)
    rain_impact_vec[69:] *= 1.8  # evening boost
    precip_raw = target_df["precipitation"].to_numpy(dtype=float) if "precipitation" in target_df.columns else np.zeros(96, dtype=float)
    precip_saturated = np.where(precip_raw > 0, np.log1p(precip_raw) * (5.0 / max(np.log1p(5.0), 1e-9)), 0.0)
    rain_impact = rain_impact_vec * precip_saturated

    # ── Wind-gust dip ──────────────────────────────────────────────────────────
    # High wind gusts reduce load: outdoor activities halt, industrial plant
    # run at reduced capacity, and comfort-cooling drops (wind chill substitutes).
    # Effect is proportional to excess wind above threshold, log-saturated.
    # Peak-hour blocks (08:00–19:00, blocks 33–76) take a larger hit because
    # commercial/industrial load is highest then.
    _wind_gust_thresh = float(cfg.get("wind_gust_threshold_kmh", 30.0))
    _wind_coeffs_cfg  = cfg.get("wind_coeffs", {"winter": 8.0, "spring": 10.0,
                                                  "summer": 12.0, "fall": 10.0})
    _wind_coeff = float(_wind_coeffs_cfg.get(season, 10.0))
    if "wind_speed_10m" in target_df.columns:
        wind_raw = target_df.sort_values("time_block")["wind_speed_10m"].to_numpy(dtype=float)
        if len(wind_raw) != 96:
            wind_raw = np.resize(wind_raw, 96)
    else:
        wind_raw = np.zeros(96, dtype=float)
    wind_excess    = np.maximum(0.0, wind_raw - _wind_gust_thresh)
    _norm          = max(np.log1p(30.0), 1e-9)           # ~1.0 at 30 km/h excess (≈ 60 km/h total)
    wind_saturated = np.log1p(wind_excess) / _norm
    wind_impact_vec = np.full(96, _wind_coeff, dtype=float)
    wind_impact_vec[32:76] *= 1.5                         # peak-hour multiplier
    wind_impact = wind_impact_vec * wind_saturated        # MW dip, positive = load reduction

    # SLDC grid-security shedding correction (80m wind).
    # When hub-height wind exceeds the SLDC protection threshold, feeders are cut —
    # this is a supply-side event independent of demand.  Correction is flat across
    # all blocks (SLDC curtailment applies round-the-clock once triggered).
    _sldc_correction = np.zeros(96, dtype=float)
    if _high_wind_risk:
        _sldc_excess = max(0.0, _wx_wind_80m - _sldc_threshold)
        _sldc_mw = min(30.0 * np.log1p(_sldc_excess / 10.0) * 10.0, 600.0)
        _sldc_correction[:] = _sldc_mw
        logger.info(
            "[pipeline:%s] SLDC shedding correction: -%.0f MW flat (80m_wind=%.1f km/h, excess=%.1f km/h)",
            target_date, _sldc_mw, _wx_wind_80m, _sldc_excess,
        )

    pattern_adjustment = sim_residual_shape.copy()
    residual_correction = boundary_correction.copy()
    residuals = actual_partial - hybrid[:actual_blocks]

    ramp_weather_meta: Dict[str, Any] = {"enabled": False}
    ramp_adjustment = np.zeros(_BLOCK_COUNT, dtype=float)
    block_bias_meta: Dict[str, Any] = {"enabled": False}
    block_bias_applied = np.zeros(_BLOCK_COUNT, dtype=float)
    spline_residual_meta: Dict[str, Any] = {"enabled": False}
    spline_residual_adjustment = np.zeros(_BLOCK_COUNT, dtype=float)
    calibration_meta: Dict[str, Any] = {"enabled": False}
    calibration_applied = np.zeros(_BLOCK_COUNT, dtype=float)
    momentum_meta: Dict[str, Any] = {"enabled": False}
    momentum_adjustment = np.zeros(_BLOCK_COUNT, dtype=float)
    ramp_limit_meta: Dict[str, Any] = {"enabled": False}
    ramp_limit_adjustment = np.zeros(_BLOCK_COUNT, dtype=float)

    # ── Stitch: exact actuals + smooth 8-block cosine taper into forecast ─────
    stitched = forecast.copy()
    stitched[:actual_blocks] = actual_partial
    taper_len = min(8, 96 - actual_blocks)
    if taper_len > 0 and actual_blocks > 0 and len(actual_partial) > 0:
        for t in range(taper_len):
            b = actual_blocks + t
            w = 0.5 * (1.0 - np.cos(np.pi * (t + 1) / (taper_len + 1)))  # cosine ease-in
            stitched[b] = actual_partial[-1] * (1.0 - w) + forecast[b] * w

    # ── Evening storm pre-trigger (blocks 73-88, 18:00-22:00) ────────────────
    # Analysis of May-June 2022-2025 shows wind_speed coefficient grew from
    # -11 to -88 MW/(km/h) in the 18:00-21:00 window. Wind arrives 20-40 min
    # before rain — the April 27 thunderstorm gap (-5000 MW) was detectable in
    # wind data before rain hit. This trigger fires when the NWP forecast shows
    # sustained wind >25 km/h in the 17:30-19:00 window (blocks 69-76) with
    # any precipitation, applying a pre-emptive flat -60 MW/(km/h) correction
    # to blocks 73-88. Enabled only when actual data hasn't covered those blocks.
    _evening_storm_meta: Dict[str, Any] = {"enabled": False}
    _evening_storm_adjustment = np.zeros(_BLOCK_COUNT, dtype=float)
    _STORM_WIND_THRESH_KMH = float(accuracy_cfg.get("evening_storm_wind_threshold_kmh", 25.0))
    _STORM_COEFF_MW_KMH    = float(accuracy_cfg.get("evening_storm_wind_coeff_mw_kmh", 60.0))
    _STORM_PRECIP_THRESH   = float(accuracy_cfg.get("evening_storm_precip_threshold_mm", 2.0))
    if actual_blocks < 73 and "wind_speed_10m" in target_df.columns:
        try:
            tdf_sorted = target_df.sort_values("time_block")
            # Pre-storm window: blocks 69-76 (17:15-19:00)
            _prestorm_wind = tdf_sorted[
                tdf_sorted["time_block"].between(69, 76)
            ]["wind_speed_10m"].to_numpy(dtype=float)
            _prestorm_rain = tdf_sorted[
                tdf_sorted["time_block"].between(69, 88)
            ]["precipitation"].to_numpy(dtype=float) if "precipitation" in tdf_sorted.columns else np.zeros(1)
            _avg_prestorm_wind = float(np.nanmean(_prestorm_wind)) if len(_prestorm_wind) else 0.0
            _total_prestorm_rain = float(np.nansum(_prestorm_rain))
            # Trigger: wind sustained >25 km/h AND any precip forecast in blocks 69-88
            if _avg_prestorm_wind > _STORM_WIND_THRESH_KMH and _total_prestorm_rain > _STORM_PRECIP_THRESH:
                # Apply -60 MW/(km/h) × excess wind to blocks 73-88
                _excess_wind = max(0.0, _avg_prestorm_wind - _STORM_WIND_THRESH_KMH)
                _storm_cut_mw = _STORM_COEFF_MW_KMH * _excess_wind
                _storm_cut_mw = min(_storm_cut_mw, 2000.0)  # hard cap at 2000 MW
                # Taper: full cut at block 73, fades to 50% by block 88
                _storm_blocks = np.arange(73, 89)  # blocks 73-88
                _storm_taper = np.linspace(1.0, 0.5, len(_storm_blocks))
                for _bi, _blk in enumerate(_storm_blocks):
                    if _blk > actual_blocks:
                        _idx = _blk - 1
                        _storm_cut = _storm_cut_mw * _storm_taper[_bi]
                        _evening_storm_adjustment[_idx] = -_storm_cut
                        stitched[_idx] = max(stitched[_idx] - _storm_cut, 0.0)
                _evening_storm_meta = {
                    "enabled": True,
                    "triggered": True,
                    "avg_prestorm_wind_kmh": round(_avg_prestorm_wind, 1),
                    "total_prestorm_rain_mm": round(_total_prestorm_rain, 1),
                    "storm_cut_mw": round(_storm_cut_mw, 1),
                    "blocks_adjusted": "73-88",
                }
                logger.warning(
                    "[storm-trigger] Evening storm pre-trigger FIRED: wind=%.1f km/h, "
                    "rain=%.1fmm → cutting blocks 73-88 by up to %.0f MW",
                    _avg_prestorm_wind, _total_prestorm_rain, _storm_cut_mw,
                )
            else:
                _evening_storm_meta = {
                    "enabled": True, "triggered": False,
                    "avg_prestorm_wind_kmh": round(_avg_prestorm_wind, 1),
                    "total_prestorm_rain_mm": round(_total_prestorm_rain, 1),
                }
        except Exception as _storm_err:
            _evening_storm_meta = {"enabled": True, "triggered": False, "error": str(_storm_err)}

    # Direct wind dip is diagnostic by default. Applying it on top of weather_base
    # double-counts wind/rain effects and was a major source of over-correction.
    if (
        bool(accuracy_cfg.get("direct_wind_adjustment_enabled", False))
        and actual_blocks < 96
        and wind_impact[actual_blocks:].sum() > 0
    ):
        stitched[actual_blocks:] = np.maximum(
            stitched[actual_blocks:] - wind_impact[actual_blocks:], 0.0
        )

    # SLDC feeder-cut correction (80m wind, supply-side).
    # Applied to all future blocks regardless of actual_blocks; unlike the
    # demand-side wind_impact this is a flat curtailment across the whole day.
    if _high_wind_risk and actual_blocks < 96:
        stitched[actual_blocks:] = np.maximum(
            stitched[actual_blocks:] - _sldc_correction[actual_blocks:], 0.0
        )

    # Haryana afternoon ramp correction from the Excel accuracy strategy.
    if build_afternoon_ramp_adjustment is not None and bool(accuracy_cfg.get("afternoon_ramp_enabled", True)):
        try:
            _ramp = build_afternoon_ramp_adjustment(
                baseline_96=baseline_hist,
                target_df=target_df,
                season=season,
                region=hb_region,
                enabled=True,
            )
            ramp_adjustment = np.asarray(_ramp.adjustment_mw, dtype=float)
            ramp_weather_meta = dict(_ramp.metadata)
            if actual_blocks < _BLOCK_COUNT:
                stitched[actual_blocks:] = np.maximum(stitched[actual_blocks:] + ramp_adjustment[actual_blocks:], 0.0)
                temperature_impact_block = temperature_impact_block + ramp_adjustment
                weather_impact_block = weather_impact_block + ramp_adjustment
                weather_baseline = weather_baseline + ramp_adjustment
        except Exception as _ramp_err:
            ramp_weather_meta = {"enabled": True, "applied": False, "reason": str(_ramp_err)}

    # Block-level systematic bias table. Sign convention is forecast - actual,
    # so apply_bias_correction subtracts positive over-forecast bias and adds
    # MW when recent history shows under-forecast.
    if compute_bias_table is not None and apply_bias_correction is not None and bool(accuracy_cfg.get("block_bias_enabled", True)):
        try:
            _day_type_flag = "weekend" if pd.Timestamp(str(target_date)[:10]).dayofweek >= 5 else "weekday"
            _bias_table, _bias_meta = compute_bias_table(
                df,
                target_date=str(target_date),
                window_days=int(accuracy_cfg.get("block_bias_window_days", 30)),
                day_type=_day_type_flag,
                horizon=str(cfg.get("forecast_horizon", "t1")),
                smooth_window=int(accuracy_cfg.get("block_bias_smooth_window", 4)),
            )
            _bias_result = apply_bias_correction(stitched, _bias_table, actual_blocks=actual_blocks, enabled=True)
            block_bias_applied = _bias_result.applied_mw
            live_bias_decay = np.zeros(_BLOCK_COUNT, dtype=float)
            live_bias_decay[actual_blocks:] = boundary_correction[actual_blocks:]
            final_bias_applied = (0.60 * block_bias_applied) + (0.40 * live_bias_decay)
            if actual_blocks < _BLOCK_COUNT:
                stitched[actual_blocks:] = np.maximum(stitched[actual_blocks:] + final_bias_applied[actual_blocks:], 0.0)
            block_bias_meta = {
                **_bias_meta,
                **_bias_result.metadata,
                "hierarchy": "single_final_bias",
                "formula": "0.6 * rolling_block_bias + 0.4 * recent_live_bias_decay",
                "applied_mw": final_bias_applied.tolist(),
            }
            block_bias_applied = final_bias_applied
            residual_correction = final_bias_applied
        except Exception as _bias_err:
            block_bias_meta = {"enabled": True, "applied": False, "reason": str(_bias_err)}

    # Spline residual shape carry. This replaces a flat future error extension
    # when there are enough live actual points to infer an intraday residual shape.
    if bool(accuracy_cfg.get("spline_residual_enabled", True)) and actual_blocks >= 6:
        try:
            known_x = np.arange(actual_blocks, dtype=float)
            known_y = np.asarray(actual_partial - hybrid[:actual_blocks], dtype=float)
            fut_x = np.arange(actual_blocks, _BLOCK_COUNT, dtype=float)
            if fut_x.size and np.isfinite(known_y).sum() >= 4:
                try:
                    from scipy.interpolate import CubicSpline as _CubicSpline
                    _spl = _CubicSpline(known_x, known_y, extrapolate=True)
                    pred_resid = np.asarray(_spl(fut_x), dtype=float)
                except Exception:
                    deg = min(3, max(1, actual_blocks - 1))
                    coeff = np.polyfit(known_x, known_y, deg=deg)
                    pred_resid = np.polyval(coeff, fut_x)
                taper = np.linspace(1.0, 0.30, fut_x.size)
                cap = np.maximum(0.08 * np.maximum(stitched[actual_blocks:], 1.0), 200.0)
                adj = np.clip(pred_resid * taper, -cap, cap)
                spline_residual_adjustment[actual_blocks:] = adj
                stitched[actual_blocks:] = np.maximum(stitched[actual_blocks:] + adj, 0.0)
                pattern_adjustment = pattern_adjustment + spline_residual_adjustment
                spline_residual_meta = {
                    "enabled": True,
                    "applied": bool(np.any(np.abs(adj) > 1e-6)),
                    "known_blocks": int(actual_blocks),
                    "mean_applied_mw": round(float(np.mean(adj)), 3),
                    "max_abs_applied_mw": round(float(np.max(np.abs(adj))), 3),
                    "applied_mw": spline_residual_adjustment.tolist(),
                }
        except Exception as _spl_err:
            logger.warning("Spline residual adjustment skipped: %s", _spl_err)
            spline_residual_meta = {"enabled": True, "applied": False, "reason": str(_spl_err)}

    shrink_cfg = cfg.get("forecast_shrinkage", {}) if isinstance(cfg.get("forecast_shrinkage", {}), dict) else {}
    offpeak_weight = float(np.clip(float(shrink_cfg.get("offpeak_weight", 0.50)), 0.0, 1.0))
    peak_weight = float(np.clip(float(shrink_cfg.get("peak_weight", 0.25)), 0.0, 1.0))
    known_blocks_bonus = float(np.clip(float(shrink_cfg.get("known_blocks_bonus", 0.20)), 0.0, 0.5))
    peak_start_idx = int(np.clip(int(shrink_cfg.get("peak_start_block", 48)) - 1, 0, _BLOCK_COUNT - 1))
    peak_end_idx = int(np.clip(int(shrink_cfg.get("peak_end_block", 72)), peak_start_idx + 1, _BLOCK_COUNT))

    shrink_weights = np.full(_BLOCK_COUNT, offpeak_weight, dtype=float)
    shrink_weights[peak_start_idx:peak_end_idx] = peak_weight
    if actual_blocks > 0:
        progress = float(np.clip(actual_blocks / _BLOCK_COUNT, 0.0, 1.0))
        shrink_weights = np.clip(shrink_weights + (known_blocks_bonus * progress), 0.0, 1.0)
    if bool(accuracy_cfg.get("dynamic_shrinkage_enabled", True)):
        try:
            _hist = df[df["date"].astype(str) < str(target_date)].copy()
            _dates = sorted(_hist["date"].dropna().astype(str).unique().tolist())
            _hist = _hist[_hist["date"].astype(str).isin(set(_dates[-14:]))]
            if not _hist.empty:
                _grp = _hist.groupby("time_block")["total_drawal"]
                _mean = _grp.mean().reindex(range(1, 97)).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
                _std = _grp.std().reindex(range(1, 97)).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
                _vol = np.divide(_std, np.maximum(_mean, 1.0), out=np.zeros_like(_std), where=_mean > 0)
                # Gentler dynamic shrinkage: avoid pulling hot peak blocks back
                # to baseline so hard that afternoon peaks under-forecast.
                _dynamic_weight = np.clip(0.98 - (_vol * 3.0), 0.80, 0.98)
                _dynamic_weight[peak_start_idx:peak_end_idx] = np.clip(
                    _dynamic_weight[peak_start_idx:peak_end_idx] + 0.05,
                    0.80,
                    0.98,
                )
                shrink_weights = np.minimum(shrink_weights, _dynamic_weight)
        except Exception:
            pass
    shrink_weights[:actual_blocks] = 1.0

    stitched_raw = stitched.copy()
    stitched = (shrink_weights * stitched_raw) + ((1.0 - shrink_weights) * hybrid)
    stitched[:actual_blocks] = actual_partial

    helper_ai_result = None

    # ── Base-day → T+1 seam smoother ──────────────────────────────────────────
    # When the user is producing a pure forward forecast (no partial actuals
    # for the target day), the very first blocks of the forecast often have a
    # visible jump from the previous day's last block.  Bridge them: linearly
    # taper the first `seam_window` blocks from a continuation of the prior
    # day's tail toward the model forecast.  Off when actual_blocks > 0 (we
    # already have a real anchor inside the day).
    base_seam_cfg = cfg.get("base_seam_bridge", {}) if isinstance(cfg.get("base_seam_bridge", {}), dict) else {}
    base_seam_window = int(np.clip(int(base_seam_cfg.get("window_blocks", 6)), 0, _BLOCK_COUNT))
    if actual_blocks == 0 and base_seam_window > 0:
        try:
            _all_dates = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
            _target_str = str(target_date)
            _prior_dates = [d for d in _all_dates if d < _target_str]
            if _prior_dates:
                _prev = _prior_dates[-1]
                _prev_df = df[df["date"].astype(str) == _prev]
                if not _prev_df.empty and "total_drawal" in _prev_df.columns:
                    _prev_curve = (
                        _prev_df.groupby("time_block")["total_drawal"]
                        .mean()
                        .reindex(range(1, 97))
                        .ffill()
                        .bfill()
                        .to_numpy(dtype=float)
                    )
                    if np.isfinite(_prev_curve).any():
                        # Bridge value at block 1 = prior block 96 + (block 1 - block 96) shape
                        # of similar days picks up DOW seasonality automatically; here we only
                        # need to remove the discontinuity at the seam.
                        anchor_prev_last = float(_prev_curve[-1])
                        forecast_block1 = float(stitched[0]) if np.isfinite(stitched[0]) else anchor_prev_last
                        # Allowed jump per block (MW) — clamp drift so we never produce a
                        # ridge if the prior day was anomalous.
                        max_step = max(0.05 * max(anchor_prev_last, forecast_block1, 1.0), 25.0)
                        target_block1 = anchor_prev_last + np.clip(
                            forecast_block1 - anchor_prev_last, -max_step, max_step
                        )
                        # Linear taper: block i in [0, base_seam_window) gets weight i/window
                        # toward the model forecast, (1 - i/window) toward target_block1.
                        for i in range(base_seam_window):
                            w_model = (i + 1) / float(base_seam_window + 1)
                            stitched[i] = (1.0 - w_model) * target_block1 + w_model * float(stitched[i])
                        logger.info(
                            "✓ Base→T+1 seam smoothed: prev-last=%.1f, fc[0]=%.1f, target=%.1f, window=%d",
                            anchor_prev_last, forecast_block1, target_block1, base_seam_window,
                        )
        except Exception as _seam_err:
            logger.debug(f"Base→T+1 seam smoother skipped: {_seam_err}")

    # ── Savitzky-Golay smoothing on forecasted blocks ─────────────────────────
    # Removes per-block jaggedness introduced by independent ML predictions
    # without shifting peaks (unlike causal smoothers).  Applied only to the
    # blocks ahead of actuals, so real measurements are never modified.
    # Gated by config["sg_smooth"]["enabled"] (default True).
    _sg_cfg = cfg.get("sg_smooth", {}) if isinstance(cfg.get("sg_smooth", {}), dict) else {}
    if bool(_sg_cfg.get("enabled", True)) and actual_blocks < _BLOCK_COUNT:
        try:
            _sg_window = int(_sg_cfg.get("window", 5))
            _sg_order = int(_sg_cfg.get("polyorder", 2))
            # window must be odd and > polyorder
            if _sg_window % 2 == 0:
                _sg_window += 1
            _sg_window = max(_sg_window, _sg_order + 2 if _sg_order % 2 == 0 else _sg_order + 2)
            try:
                from scipy.signal import savgol_filter as _savgol
                _smooth_region = _savgol(
                    stitched[actual_blocks:],
                    window_length=_sg_window,
                    polyorder=_sg_order,
                    mode="nearest",
                )
            except ImportError:
                # Fallback: symmetric uniform moving average (no phase lag)
                _pad = _sg_window // 2
                _padded = np.pad(stitched[actual_blocks:], _pad, mode="edge")
                _smooth_region = np.convolve(
                    _padded, np.ones(_sg_window) / _sg_window, mode="valid"
                )[:len(stitched) - actual_blocks]
            # Blend 80/20: more aggressive smoothing reduces jagged spikes
            # while Savitzky-Golay still preserves peak positions (FIX 6)
            stitched[actual_blocks:] = (
                0.80 * stitched[actual_blocks:] + 0.20 * np.maximum(_smooth_region, 0)
            )
            # Preserve seam continuity: actual tail must remain unchanged
            if actual_blocks > 0:
                stitched[:actual_blocks] = actual_partial
        except Exception as _sg_err:
            logger.debug("SG smoother skipped: %s", _sg_err)

    horizon_calibration_correction = np.zeros(_BLOCK_COUNT, dtype=float)
    horizon_calibration_meta: Dict[str, Any] = {"enabled": False}
    if actual_blocks < _BLOCK_COUNT:
        stitched, horizon_calibration_correction, horizon_calibration_meta = _recent_horizon_calibration(
            df=df,
            target_date=str(target_date),
            forecast=stitched,
            actual_blocks=int(actual_blocks),
            actual_partial=actual_partial,
            cfg=cfg,
        )
        if actual_blocks > 0:
            stitched[:actual_blocks] = actual_partial
        _diag_vec("stitched_post_horizon_calibration", stitched)

    if apply_block_type_calibration is not None and bool(accuracy_cfg.get("calibration_enabled", True)):
        try:
            _cal = apply_block_type_calibration(
                stitched,
                target_date=str(target_date),
                history_df=df,
                actual_blocks=int(actual_blocks),
                window_days=int(accuracy_cfg.get("calibration_window_days", 21)),
                min_confidence=float(accuracy_cfg.get("calibration_min_confidence", 0.80)),
                enabled=True,
            )
            stitched = _cal.forecast
            calibration_applied = _cal.applied_mw
            calibration_meta = dict(_cal.metadata)
            if actual_blocks > 0:
                stitched[:actual_blocks] = actual_partial
        except Exception as _cal_err:
            logger.error("Calibration layer failed: %s", _cal_err, exc_info=True)
            calibration_meta = {"enabled": True, "applied": False, "reason": str(_cal_err)}

    if bool(accuracy_cfg.get("momentum_enabled", True)) and actual_blocks >= 5 and actual_blocks < _BLOCK_COUNT:
        try:
            momentum = float(actual_partial[-1] - actual_partial[-5])
            decay_blocks = max(float(accuracy_cfg.get("momentum_decay_blocks", 8.0)), 1.0)
            for i in range(actual_blocks, _BLOCK_COUNT):
                dist = i - actual_blocks + 1
                momentum_adjustment[i] = momentum * float(np.exp(-dist / decay_blocks))
            cap = np.maximum(0.03 * np.maximum(stitched, 1.0), 75.0)
            momentum_adjustment = np.clip(momentum_adjustment, -cap, cap)
            stitched[actual_blocks:] = np.maximum(stitched[actual_blocks:] + momentum_adjustment[actual_blocks:], 0.0)
            momentum_meta = {
                "enabled": True,
                "applied": bool(np.any(np.abs(momentum_adjustment[actual_blocks:]) > 1e-6)),
                "raw_momentum_mw": round(momentum, 3),
                "decay_blocks": float(decay_blocks),
                "max_abs_applied_mw": round(float(np.max(np.abs(momentum_adjustment[actual_blocks:]))), 3),
                "applied_mw": momentum_adjustment.tolist(),
            }
        except Exception as _mom_err:
            momentum_meta = {"enabled": True, "applied": False, "reason": str(_mom_err)}

    if bool(accuracy_cfg.get("ramp_limit_enabled", True)):
        try:
            _limited, _ramp_applied, _ramp_meta = _apply_ramp_limit(
                stitched,
                actual_blocks=int(actual_blocks),
                max_ramp_mw=float(accuracy_cfg.get("max_ramp_mw", 300.0)),
            )
            stitched = _limited
            ramp_limit_adjustment = _ramp_applied
            ramp_limit_meta = _ramp_meta
        except Exception as _ramp_limit_err:
            ramp_limit_meta = {"enabled": True, "applied": False, "reason": str(_ramp_limit_err)}

    _diag_vec("stitched_final", stitched)

    baseline_vec = np.asarray(stitched, dtype=float)
    if "temperature_impact_block" not in locals():
        temperature_impact_block = np.zeros(_BLOCK_COUNT, dtype=float)
    if "temp_delta" not in locals():
        temp_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "hum_delta" not in locals():
        hum_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "rain_delta" not in locals():
        rain_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "apparent_delta" not in locals():
        apparent_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "cloud_delta" not in locals():
        cloud_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "sun_delta" not in locals():
        sun_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "rad_delta" not in locals():
        rad_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "wind_delta" not in locals():
        wind_delta = np.zeros(_BLOCK_COUNT, dtype=float)
    if "coef_hum" not in locals():
        coef_hum = 0.0
    if "coef_rain" not in locals():
        coef_rain = 0.0
    if "temp_coeff_vec" not in locals():
        temp_coeff_vec = np.zeros(_BLOCK_COUNT, dtype=float)
    if "temp_up_coeff" not in locals():
        temp_up_coeff = np.zeros(_BLOCK_COUNT, dtype=float)
    if "temp_down_coeff" not in locals():
        temp_down_coeff = np.zeros(_BLOCK_COUNT, dtype=float)
    weather_base = np.divide(
        weather_impact_block,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(weather_impact_block, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    temperature_base = np.divide(
        temperature_impact_block,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(temp_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    humidity_base = np.divide(
        hum_delta * coef_hum,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(hum_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    precipitation_base = np.divide(
        rain_delta * coef_rain,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(rain_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    apparent_base = np.divide(
        apparent_delta * float(weather_coefs.get("apparent_temp", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(apparent_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    cloud_base = np.divide(
        cloud_delta * float(weather_coefs.get("cloud", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(cloud_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    sun_base = np.divide(
        sun_delta * float(weather_coefs.get("sunshine", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(sun_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    radiation_base = np.divide(
        rad_delta * float(weather_coefs.get("radiation", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(rad_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    wind_base = np.divide(
        wind_delta * float(weather_coefs.get("wind", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(wind_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    daytype_base = np.asarray(cal_daytype_factor) - 1.0
    holiday_base = np.asarray(cal_holiday_factor) - 1.0
    manual_base = _as_96_vector(cfg.get("manual_base", np.zeros(96, dtype=float)))
    driver_layer = build_block_driver_matrix(
        target_date=target_date,
        baseline_load=baseline_vec,
        weather_base=weather_base,
        daytype_base=daytype_base,
        holiday_base=holiday_base,
        manual_base=manual_base,
        temperature_base=temperature_base,
        humidity_base=humidity_base,
        precipitation_base=precipitation_base,
        apparent_base=apparent_base,
        cloud_base=cloud_base,
        sun_base=sun_base,
        radiation_base=radiation_base,
        wind_base=wind_base,
        sliders=cfg.get("driver_sliders"),
        selection=cfg.get("selection"),
        smooth_edges=bool(cfg.get("selection_smoothing", False)),
    )
    block_driver_matrix = driver_layer["matrix_df"]
    selection_mask = np.asarray(driver_layer["selection_mask"], dtype=float)
    final_load = np.asarray(driver_layer["final_load"], dtype=float)

    weather_scaled = block_driver_matrix["weather_scaled"].to_numpy(dtype=float)
    temperature_scaled = block_driver_matrix["temperature_scaled"].to_numpy(dtype=float)
    humidity_scaled = block_driver_matrix["humidity_scaled"].to_numpy(dtype=float)
    precipitation_scaled = block_driver_matrix["precipitation_scaled"].to_numpy(dtype=float)
    apparent_scaled = block_driver_matrix["apparent_scaled"].to_numpy(dtype=float) if "apparent_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    cloud_scaled = block_driver_matrix["cloud_scaled"].to_numpy(dtype=float) if "cloud_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    sun_scaled = block_driver_matrix["sun_scaled"].to_numpy(dtype=float) if "sun_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    rad_scaled = block_driver_matrix["rad_scaled"].to_numpy(dtype=float) if "rad_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    wind_scaled = block_driver_matrix["wind_scaled"].to_numpy(dtype=float) if "wind_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    daytype_scaled = block_driver_matrix["daytype_scaled"].to_numpy(dtype=float)
    holiday_scaled = block_driver_matrix["holiday_scaled"].to_numpy(dtype=float)
    manual_scaled = block_driver_matrix["manual_scaled"].to_numpy(dtype=float)
    clamped_impact = block_driver_matrix["final_impact"].to_numpy(dtype=float)

    temperature_contrib_mw = baseline_vec * temperature_scaled * selection_mask
    humidity_contrib_mw = baseline_vec * humidity_scaled * selection_mask
    precipitation_contrib_mw = baseline_vec * precipitation_scaled * selection_mask
    apparent_contrib_mw = baseline_vec * apparent_scaled * selection_mask
    cloud_contrib_mw = baseline_vec * cloud_scaled * selection_mask
    sun_contrib_mw = baseline_vec * sun_scaled * selection_mask
    radiation_contrib_mw = baseline_vec * rad_scaled * selection_mask
    wind_contrib_mw = baseline_vec * wind_scaled * selection_mask
    daytype_contrib_mw = baseline_vec * daytype_scaled * selection_mask
    holiday_contrib_mw = baseline_vec * holiday_scaled * selection_mask
    manual_contrib_mw = baseline_vec * manual_scaled * selection_mask
    net_contribution_mw = final_load - baseline_vec
    applied_impact_pct = clamped_impact * selection_mask * 100.0

    residual_std = float(np.std(residuals)) if len(residuals) >= 2 else 0.0
    similar_residual_std = np.zeros(96, dtype=float)
    if similar_dates:
        similar_residuals = []
        for d in similar_dates:
            day_series = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            if len(day_series) == 96:
                similar_residuals.append(day_series - baseline_hist)
        if similar_residuals:
            similar_residual_std = np.std(np.vstack(similar_residuals), axis=0)

    # Tail Decay (User Fix #5)
    # Decay momentum (trend) at night (Block > 84)
    trend_decay_mask = np.ones(96, dtype=float)
    for b in range(84, 96):
        dist = b - 84
        trend_decay_mask[b] = np.exp(-dist / 6.0)
        
    trend_weight = float(cfg.get("trend_weight", 0.05))
    trend_applied = trend_full * trend_weight * trend_decay_mask
    sigma_block = np.sqrt((similar_residual_std ** 2) + (residual_std ** 2))
    sigma_block = sigma_block + (np.abs(pattern_adjustment) * 0.15)
    sigma_floor = np.maximum(np.abs(net_contribution_mw) * 0.08, 1.0)
    sigma_block = np.maximum(sigma_block, sigma_floor)
    z_val = 1.2816
    p50 = stitched.copy()
    p10 = np.maximum(p50 - (z_val * sigma_block), 0.0)
    p90 = p50 + (z_val * sigma_block)
    uncertainty_width_mw = p90 - p10
    uncertainty_width_pct = (uncertainty_width_mw / np.maximum(p50, 1e-6)) * 100.0
    forecast_confidence = np.clip(1.0 - (uncertainty_width_pct / 100.0), 0.05, 0.99)

    weather_sens = np.abs(weather_scaled)
    calendar_sens = np.abs(daytype_scaled) + np.abs(holiday_scaled)
    direct_rain_for_sensitivity = rain_impact if bool(accuracy_cfg.get("direct_rain_adjustment_enabled", False)) else 0.0
    operational_sens = np.abs((trend_applied + pattern_adjustment - direct_rain_for_sensitivity) / np.maximum(baseline_vec, 1e-6))
    manual_sens = np.abs(manual_scaled)
    sens_stack = np.vstack([weather_sens, calendar_sens, operational_sens, manual_sens])
    dominant_idx = np.argmax(sens_stack, axis=0)
    family_labels = np.array(["weather", "calendar", "operational", "manual"])
    dominant_family = family_labels[dominant_idx]
    sensitivity_score = np.sum(sens_stack, axis=0)

    block_contributors = []
    slot_sensitivity_profile = []
    forecast_uncertainty = []
    decision_signals = []
    top_contributor_slots = []
    for i in range(96):
        contrib_map = {
            "temperature": float(temperature_contrib_mw[i]),
            "humidity": float(humidity_contrib_mw[i]),
            "precipitation": float(precipitation_contrib_mw[i]),
            "wind": float(wind_contrib_mw[i]),
            "apparent_temperature": float(apparent_contrib_mw[i]),
            "cloud_cover": float(cloud_contrib_mw[i]),
            "sunshine_duration": float(sun_contrib_mw[i]),
            "direct_radiation": float(radiation_contrib_mw[i]),
            "daytype": float(daytype_contrib_mw[i]),
            "holiday": float(holiday_contrib_mw[i]),
            "manual": float(manual_contrib_mw[i]),
        }
        ranked = _rank_block_contributors(contrib_map, baseline_mw=float(baseline_vec[i]))
        abs_vals = np.asarray([abs(v) for v in contrib_map.values()], dtype=float)
        abs_sum = float(np.sum(abs_vals)) if np.sum(abs_vals) > 0 else 1.0
        dominance = float(np.max(abs_vals) / abs_sum)
        contributor_confidence = float(np.clip(0.35 + (0.40 * dominance) + (0.25 * forecast_confidence[i]), 0.05, 0.99))

        risk_score = (0.55 * abs(applied_impact_pct[i])) + (0.45 * uncertainty_width_pct[i])
        if risk_score >= 25:
            risk_flag = "high"
        elif risk_score >= 12:
            risk_flag = "medium"
        else:
            risk_flag = "low"

        primary_driver = ranked[0]["feature"] if ranked else "none"
        secondary_driver = ranked[1]["feature"] if len(ranked) > 1 else primary_driver
        expected_gain = abs(float(net_contribution_mw[i])) * (0.35 if risk_flag == "high" else 0.20 if risk_flag == "medium" else 0.08)

        block_contributors.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "baseline_mw": float(baseline_vec[i]),
            "forecast_mw": float(final_load[i]),
            "net_impact_pct": float(applied_impact_pct[i]),
            "net_contribution_mw": float(net_contribution_mw[i]),
            "forecast_confidence": float(forecast_confidence[i]),
            "contributor_confidence": contributor_confidence,
            "contributors_ranked": ranked,
        })

        slot_sensitivity_profile.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "weather_sensitivity": float(weather_sens[i]),
            "calendar_sensitivity": float(calendar_sens[i]),
            "operational_sensitivity": float(operational_sens[i]),
            "manual_sensitivity": float(manual_sens[i]),
            "dominant_family": str(dominant_family[i]),
            "sensitivity_score": float(sensitivity_score[i]),
        })

        forecast_uncertainty.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "p10_mw": float(p10[i]),
            "p50_mw": float(p50[i]),
            "p90_mw": float(p90[i]),
            "uncertainty_width_mw": float(uncertainty_width_mw[i]),
            "uncertainty_width_pct": float(uncertainty_width_pct[i]),
            "forecast_confidence": float(forecast_confidence[i]),
        })

        decision_signals.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "risk_flag": risk_flag,
            "risk_score": float(risk_score),
            "dominant_family": str(dominant_family[i]),
            "primary_driver": primary_driver,
            "secondary_driver": secondary_driver,
            "recommended_action": _recommended_action(primary_driver, risk_flag, float(applied_impact_pct[i])),
            "expected_gain_mw": float(expected_gain),
            "net_impact_pct": float(applied_impact_pct[i]),
            "uncertainty_pct": float(uncertainty_width_pct[i]),
            "confidence": float(min(forecast_confidence[i], contributor_confidence)),
        })

        top_contributor_slots.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "abs_net_contribution_mw": float(abs(net_contribution_mw[i])),
            "primary_driver": primary_driver,
            "risk_flag": risk_flag,
        })
    top_contributor_slots = sorted(top_contributor_slots, key=lambda x: x["abs_net_contribution_mw"], reverse=True)[:12]

    # Peak-level weather + calendar impact diagnostics
    hist_peak_idx = int(np.argmax(baseline_hist))
    weather_peak_idx = int(np.argmax(weather_baseline))
    hybrid_pre_cal_peak_idx = int(np.argmax(hybrid_pre_calendar))
    hybrid_peak_idx = int(np.argmax(hybrid))
    forecast_peak_idx = int(np.argmax(stitched))
    peak_i = forecast_peak_idx
    peak_impact = {
        "calendar_config": cfg.get("calendar_config"),
        "baseline_peak": {
            "time_block": hist_peak_idx + 1,
            "mw": float(np.max(baseline_hist)),
        },
        "weather_peak": {
            "time_block": weather_peak_idx + 1,
            "mw": float(np.max(weather_baseline)),
            "delta_vs_baseline_mw": float(np.max(weather_baseline) - np.max(baseline_hist)),
        },
        "hybrid_pre_calendar_peak": {
            "time_block": hybrid_pre_cal_peak_idx + 1,
            "mw": float(np.max(hybrid_pre_calendar)),
            "delta_vs_baseline_mw": float(np.max(hybrid_pre_calendar) - np.max(baseline_hist)),
        },
        "hybrid_peak": {
            "time_block": hybrid_peak_idx + 1,
            "mw": float(np.max(hybrid)),
            "calendar_effect_at_peak_mw": float(hybrid[hybrid_peak_idx] - hybrid_pre_calendar[hybrid_peak_idx]),
        },
        "forecast_peak": {
            "time_block": forecast_peak_idx + 1,
            "mw": float(np.max(stitched)),
            "weather_effect_mw": float(weather_baseline[peak_i] - baseline_hist[peak_i]),
            "calendar_effect_mw": float(calendar_adjustment_block[peak_i]),
            "net_weather_calendar_effect_mw": float(hybrid[peak_i] - baseline_hist[peak_i]),
        },
    }

    driver_table = []

    window_start = max(0, actual_blocks)
    window_len = max(1, 96 - window_start)
    temp_impact = float(np.sum(temperature_impact_block[window_start:])) / window_len
    hum_impact = float(np.sum((hum_delta * coef_hum)[window_start:])) / window_len
    rain_impact_total = float(np.sum(rain_impact[window_start:])) / window_len
    wind_impact_total = float(np.sum(wind_impact[window_start:])) / window_len
    bias_mw = float(np.mean(residuals)) if len(residuals) else 0.0
    trend_amp = float(np.mean(np.abs(trend_component))) if len(trend_component) else 0.0
    trend_mw = trend_amp if trend_slope >= 0 else -trend_amp
    pattern_mw = float(np.sum(pattern_adjustment[window_start:])) / window_len
    residual_corr_mw = float(np.sum(residual_correction[window_start:])) / window_len

    driver_values = [
        ("Temperature impact", temp_impact),
        ("Humidity impact", hum_impact),
        ("Rain impact (diagnostic)", -rain_impact_total),
        ("Wind gust dip (diagnostic)", -wind_impact_total),
        ("Bias MW", bias_mw),
        ("Trend MW", trend_mw),
        ("Pattern MW", pattern_mw),
        ("Residual correction", residual_corr_mw),
        ("Human behaviour", float(np.sum(human_behaviour_adj[window_start:])) / window_len),
    ]
    total_driver = sum(abs(v) for _, v in driver_values) or 1.0

    for label, val in driver_values:
        driver_table.append({
            "factor": label,
            "mw": round(val, 2),
            "pct": round(val / total_driver * 100, 2),
        })

    rows = []
    for i in range(96):
        rows.append({
            "date": target_date,
            "time_block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "historical_baseline": float(baseline_hist[i]),
            "weather_impact": float(weather_impact_block[i]),
            "weather_baseline": float(weather_baseline[i]),
            "hybrid_baseline": float(hybrid[i]),
            "calendar_adjustment": float(calendar_adjustment_block[i]),
            "bias_applied": float(bias_applied[i]),
            "trend_component": float(trend_applied[i]),
            "rain_adjustment": float(rain_impact[i]),
            "wind_adjustment": float(wind_impact[i]),
            "ramp_weather_adjustment": float(ramp_adjustment[i]),
            "block_bias_correction": float(block_bias_applied[i]),
            "spline_residual_adjustment": float(spline_residual_adjustment[i]),
            "calibration_adjustment": float(calibration_applied[i]),
            "momentum_adjustment": float(momentum_adjustment[i]),
            "ramp_limit_adjustment": float(ramp_limit_adjustment[i]),
            "pattern_adjustment": float(pattern_adjustment[i]),
            "residual_correction": float(residual_correction[i]),
            "human_behaviour": float(human_behaviour_adj[i]),
            "forecast": float(stitched[i]),
            "actual": float(actual_full[i]) if (i < actual_blocks and i < len(actual_full)) else None,
            "is_actual": i < actual_blocks,
        })

    # Dip explanations (forecast window)
    forecast_vals = stitched
    window_start = max(0, actual_blocks)
    mean_forecast = float(np.mean(forecast_vals[window_start:])) if window_start < 96 else float(np.mean(forecast_vals))
    dip_threshold_mw = max(40.0, mean_forecast * 0.015)
    dip_threshold_pct = 1.5
    dip_explanations = []
    for i in range(window_start + 1, 96):
        prev_val = forecast_vals[i - 1]
        curr_val = forecast_vals[i]
        if prev_val <= 0:
            continue
        diff = curr_val - prev_val
        diff_pct = (diff / prev_val) * 100
        if diff > -dip_threshold_mw and diff_pct > -dip_threshold_pct:
            continue

        bias_component = bias_applied[i] - hybrid[i]
        trend_component_block = trend_full[i]
        rain_component = -rain_impact[i]
        weather_component = weather_pred[i] - baseline_hist[i]

        components = {
            "Bias MW": bias_component,
            "Trend MW": trend_component_block,
            "Rain impact": rain_component,
            "Weather shift": weather_component,
        }
        primary = min(components.items(), key=lambda kv: kv[1])[0]

        dip_explanations.append({
            "time_block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "drop_mw": round(float(diff), 2),
            "drop_pct": round(float(diff_pct), 2),
            "primary_driver": primary,
            "components": {k: round(float(v), 2) for k, v in components.items()},
        })

    dip_explanations = sorted(dip_explanations, key=lambda d: d["drop_mw"])[:6]

    same_month_last_year_diag = None
    try:
        tdt = pd.to_datetime(target_date, errors="coerce")
        if pd.notna(tdt):
            last_year = int(tdt.year) - 1
            month = int(tdt.month)
            diag = df.copy()
            diag["date"] = diag["date"].astype(str)
            diag["_dt"] = pd.to_datetime(diag["date"], errors="coerce")
            diag = diag[pd.notna(diag["_dt"])]
            diag["_year"] = diag["_dt"].dt.year
            diag["_month"] = diag["_dt"].dt.month
            ly = diag[(diag["_year"] == last_year) & (diag["_month"] == month)]
            if not ly.empty:
                is_weekend = int(tdt.dayofweek >= 5)
                ly["_is_weekend"] = (ly["_dt"].dt.dayofweek >= 5).astype(int)
                ly = ly[ly["_is_weekend"] == is_weekend]
                by_day = ly.groupby("date")["total_drawal"].agg(["mean", "max"]).reset_index()
                rain_col = "rain" if "rain" in ly.columns else ("precipitation" if "precipitation" in ly.columns else None)
                rainy_days = None
                if rain_col:
                    rain_by_day = ly.groupby("date")[rain_col].sum()
                    rainy_days = int((rain_by_day > 0.1).sum())
                same_month_last_year_diag = {
                    "year": last_year,
                    "month": month,
                    "n_days": int(by_day["date"].nunique()),
                    "avg_peak_mw": float(by_day["max"].mean()) if len(by_day) else None,
                    "avg_mean_mw": float(by_day["mean"].mean()) if len(by_day) else None,
                    "rainy_days": rainy_days,
                }
    except Exception:
        same_month_last_year_diag = None

    response = {
        "date": target_date,
        "metadata": {
            "season": season,
            "day_type": day_type,
            "actual_blocks": actual_blocks,
            "rain_coeff": rain_coeff,
            "best_baseline_window": best_window,
            "best_baseline_mape": round(best_mape, 2) if best_mape is not None else None,
            "similar_days_count": int(len(similar_days)) if not similar_days.empty else 0,
            "region": hb_region,
            "human_behaviour_profile": hb_profile_label,
            "human_behaviour_weight": hb_weight,
            "behaviour_source": (
                "CalibratedCyclicData"
                if (_CALIBRATED_BEHAVIOUR and (INDIAN_STATE_REGIONS.get(hb_region.lower().strip(), hb_region.lower().strip()), season, hb_day_type) in _CALIBRATED_BEHAVIOUR)
                else "Hardcoded"
            ),
            "hybrid_ai_engine": {
                "enabled": hybrid_ai_forecast_vec is not None,
                "variant": ai_cfg.get("variant", "attention_seq2seq"),
                "blend_weight": float(ai_cfg.get("blend_weight", 0.35)) if hybrid_ai_forecast_vec is not None else 0.0
            },
            "hybrid_ai_explanation": "Attention-based Seq2Seq residual corrector applied." if hybrid_ai_forecast_vec is not None else "Hybrid engine unavailable.",
            "horizon_calibration": horizon_calibration_meta,
            "bias_correction": block_bias_meta,
            "ramp_weather": ramp_weather_meta,
            "spline_residual": spline_residual_meta,
            "calibration_layer": calibration_meta,
            "momentum": momentum_meta,
            "ramp_limit": ramp_limit_meta,
            "evening_storm_trigger": _evening_storm_meta,

            "same_month_last_year": same_month_last_year_diag,
            "human_behaviour_learning": _CALIBRATED_BEHAVIOUR_META.get(
                (INDIAN_STATE_REGIONS.get(hb_region.lower().strip(), hb_region.lower().strip()), season, hb_day_type),
                {"source": "hardcoded_profiles"},
            ),
            "forecast_shrinkage": {
                "offpeak_weight": round(float(offpeak_weight), 3),
                "peak_weight": round(float(peak_weight), 3),
                "known_blocks_bonus": round(float(known_blocks_bonus), 3),
                "peak_blocks": [int(peak_start_idx + 1), int(peak_end_idx)],
            },
            "target_day_type_extended": hb_day_type if 'hb_day_type' in dir() else day_type,
            "holiday_flags": target_holiday_flags,
            "weather_divergence_gate": (
                float(np.mean(locals()["divergence_gate"])) if "divergence_gate" in locals() else None
            ),
            "avg_temp_divergence_c": (
                float(locals()["avg_temp_divergence"]) if "avg_temp_divergence" in locals() else None
            ),
            "model_scope": "global",
            "feature_schema": {
                "cross_day_lags": ["lag_1", "lag_7"],
                "intraday_lags": ["lag_block_1", "lag_block_4"],
                "rolling_windows": ["rolling_4", "rolling_12"],
                "time_encoding": ["block_sin", "block_cos"],
            },
            "feature_count": int(len(_get_short_term_feature_columns(weather_training_frame))),
            "similarity_weights": cfg.get("similarity_weights"),
            "require_rain_match": bool(cfg.get("require_rain_match", True)),
            "similarity_profile_source": similarity_profile.get("source"),
            "similarity_profile_diagnostics": similarity_profile.get("diagnostics", {}),
            "temperature_delta_profile": {
                "source": temp_profile.get("source"),
                "prior_coeff": float(temp_profile.get("prior_coeff", coef_temp)),
                "global_increase_coeff": float(temp_profile.get("global_increase_coeff", coef_temp)),
                "global_reduction_coeff": float(temp_profile.get("global_reduction_coeff", coef_temp)),
                "global_increase_samples": int(temp_profile.get("global_increase_samples", 0)),
                "global_reduction_samples": int(temp_profile.get("global_reduction_samples", 0)),
                "diagnostics": temp_profile.get("diagnostics", {}),
            },
            "driver_sliders": cfg.get("driver_sliders"),
            "selection": cfg.get("selection"),
            "selection_smoothing": bool(cfg.get("selection_smoothing", False)),
            "decision_mode": "block_exogenous_attribution_v1",
            "similar_days": similar_days[[c for c in ["date", "similarity_score", "temp_mean", "hum_mean", "rain_any", "pool", "is_last_year_obs", "avg_load"] if c in similar_days.columns]].to_dict("records") if not similar_days.empty else [],
            "weather_coefs": {k: round(float(v), 4) for k, v in weather_coefs.items()} if weather_coefs else {},
            "drivers": driver_table,
        },
        "weights": {
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "weather_deviation": round(weather_dev, 3),
        },
        "calendar_factor": calendar_factor_total.tolist() if calendar_factor_total is not None else None,
        "series": {
            "blocks": list(range(1, 97)),
            "historical_baseline": baseline_hist.tolist(),
            "weather_feature_deltas": {
                "temperature": temp_delta.tolist(),
                "humidity": hum_delta.tolist(),
                "precipitation": rain_delta.tolist(),
                "apparent_temperature": apparent_delta.tolist(),
                "cloud_cover": cloud_delta.tolist(),
                "sunshine_duration": sun_delta.tolist(),
                "direct_radiation": rad_delta.tolist(),
                "wind_speed_10m": wind_delta.tolist(),
            },
            "temperature_effective_coeff": temp_coeff_vec.tolist(),
            "temperature_increase_coeff": temp_up_coeff.tolist(),
            "temperature_reduction_coeff": temp_down_coeff.tolist(),
            "weather_impact": weather_impact_block.tolist(),
            "weather_baseline": weather_baseline.tolist(),
            "hybrid_baseline": hybrid.tolist(),
            "calendar_adjustment": calendar_adjustment_block.tolist(),
            "cal_daytype_factor": cal_daytype_factor.tolist(),
            "cal_holiday_factor": cal_holiday_factor.tolist(),
            "cal_transition_factor": cal_transition_factor.tolist(),
            "bias_applied": bias_applied.tolist(),
            "trend": trend_full.tolist(),
            "trend_component": (trend_full * trend_weight).tolist(),
            "rain_adjustment": rain_impact.tolist(),
            "wind_adjustment": wind_impact.tolist(),
            "ramp_weather_adjustment": ramp_adjustment.tolist(),
            "block_bias_correction": block_bias_applied.tolist(),
            "spline_residual_adjustment": spline_residual_adjustment.tolist(),
            "calibration_adjustment": calibration_applied.tolist(),
            "momentum_adjustment": momentum_adjustment.tolist(),
            "ramp_limit_adjustment": ramp_limit_adjustment.tolist(),
            "pattern_adjustment": pattern_adjustment.tolist(),
            "residual_correction": residual_correction.tolist(),
            "horizon_calibration_correction": horizon_calibration_correction.tolist(),
            "human_behaviour": human_behaviour_adj.tolist(),
            "temperature_contrib_mw": temperature_contrib_mw.tolist(),
            "humidity_contrib_mw": humidity_contrib_mw.tolist(),
            "precipitation_contrib_mw": precipitation_contrib_mw.tolist(),
            "wind_contrib_mw": wind_contrib_mw.tolist(),
            "apparent_contrib_mw": apparent_contrib_mw.tolist(),
            "cloud_contrib_mw": cloud_contrib_mw.tolist(),
            "sunshine_contrib_mw": sun_contrib_mw.tolist(),
            "radiation_contrib_mw": radiation_contrib_mw.tolist(),
            "weather_feature_impacts_pct": {
                "temperature": (temperature_scaled * selection_mask * 100.0).tolist(),
                "humidity": (humidity_scaled * selection_mask * 100.0).tolist(),
                "precipitation": (precipitation_scaled * selection_mask * 100.0).tolist(),
                "wind": (wind_scaled * selection_mask * 100.0).tolist(),
                "apparent_temperature": (apparent_scaled * selection_mask * 100.0).tolist(),
                "cloud_cover": (cloud_scaled * selection_mask * 100.0).tolist(),
                "sunshine_duration": (sun_scaled * selection_mask * 100.0).tolist(),
                "direct_radiation": (rad_scaled * selection_mask * 100.0).tolist(),
                "weather_total": (weather_scaled * selection_mask * 100.0).tolist(),
            },
            "forecast_raw_before_shrinkage": stitched_raw.tolist(),
            "forecast_after_shrinkage": stitched_raw.tolist(),
            "forecast": stitched.tolist(),
            "hybrid_ai_forecast": hybrid_ai_forecast_vec.tolist() if hybrid_ai_forecast_vec is not None else stitched.tolist(),
            "final_load": final_load.tolist(),

            # debug_stages diagnostics are emitted as log lines when
            # config["debug_stages"]=True; no extra keys added to response.
            "selection_mask": selection_mask.tolist(),
            "actual": [float(actual_full[i]) if i < actual_blocks else None for i in range(len(actual_full))],
        },
        "explanation": "Forecast generated by statistical pipeline.",
        "bias_factor": round(bias_factor, 4),
        "trend_curve": trend_full.tolist(),
        "rain_adjustment": rain_impact.tolist(),
        "wind_adjustment": wind_impact.tolist(),
        "driver_contributions": driver_table,
        "similar_days": similar_days[["date", "similarity_score", "temp_diff", "hum_diff", "rain_match"]].to_dict("records") if not similar_days.empty else [],
        "peak_impact": peak_impact,
        "forecast_df": rows,
        "dip_explanations": dip_explanations,
        "block_driver_matrix": block_driver_matrix.to_dict("records"),
        "block_contributors": block_contributors,
        "slot_sensitivity_profile": slot_sensitivity_profile,
        "forecast_uncertainty": forecast_uncertainty,
        "decision_signals": decision_signals,
        "india_intelligence": india_meta,
        "top_contributor_slots": top_contributor_slots,
        # ── Weather anomaly flags (populated when detect_weather_anomalies ran) ──
        "weather_anomaly_flags": (
            {
                "severity":               _weather_result.severity,
                "description":            _weather_result.description,
                "load_impact_estimate_mw": _weather_result.load_impact_estimate_mw,
                "day_level":              _weather_result.day_flags,
                "block_level":            _weather_result.block_flags,
            }
            if _weather_result is not None
            else {"severity": "NONE", "description": "No anomaly", "day_level": {}, "block_level": {}}
        ),
        # ── T+2 forecast (96 blocks, next calendar day) ───────────────────────
        "t2_forecast": (
            {
                "date": str((pd.Timestamp(str(target_date)[:10]) + pd.Timedelta(days=1)).date()),
                "blocks": _lstm_t2_vec.tolist() if _lstm_t2_vec is not None else [],
                "weather_anomaly_flags": {"severity": "NONE", "description": "Not computed"},
            }
            if _lstm_t2_vec is not None and len(_lstm_t2_vec) == 96
            else None
        ),
    }

    _step("✓ DONE — pipeline complete")
    return response


# ── Weather DB fetch helper ───────────────────────────────────────────────────
class WeatherDataUnavailable(Exception):
    """Raised when MySQL weather data is unavailable (no fallback)."""
    pass


def _recompute_precipitation_if_split(wdf: pd.DataFrame) -> pd.DataFrame:
    """If rain/showers/snowfall columns are present, set precipitation to
    their per-block sum.  This is more accurate than the upstream
    "precipitation" field for providers that publish the sum with rounding,
    and lets downstream code keep referencing a single "precipitation"
    aggregate while we surface the components separately for the UI."""
    if wdf is None or wdf.empty:
        return wdf
    components = [c for c in ("rain", "showers", "snowfall") if c in wdf.columns]
    if not components:
        return wdf
    summed = sum(pd.to_numeric(wdf[c], errors="coerce").fillna(0.0) for c in components)
    wdf = wdf.copy()
    wdf["precipitation"] = summed
    return wdf


def _fetch_t2_weather_from_db(t2_date: str, region: str) -> Optional[pd.DataFrame]:
    """
    Fetch block-level weather from MySQL (via Django pipeline API).
    MySQL is the ONLY source — no SQLite/CSV fallback.

    For Haryana, this also attempts a per-location fetch via
    `/weather/by_location` and aggregates with circle-load weights from
    `haryana_circle_weights.py` so a localised event (e.g. rain only in
    Gurugram) only moves the state-level forecast in proportion to that
    circle's share of total load.  Falls back to `/weather/mean` if the
    per-location endpoint isn't available.

    Raises WeatherDataUnavailable if MySQL is unreachable or has no data
    for the requested (region, date) pair.  Caller must surface this as
    a user-facing error.
    """
    _log = logging.getLogger(__name__)
    state = region.upper().replace(" ", "_").replace("-", "_")

    import requests as _req
    pipeline_base = os.environ.get("PIPELINE_API_BASE_URL", "http://3.108.54.200:8001").rstrip("/")
    api_token     = os.environ.get("PIPELINE_API_TOKEN", os.environ.get("API_SECRET_KEY", "flagbearer"))
    headers = {"Authorization": f"Bearer {api_token}"}

    # ── Per-location weighted aggregation (if applicable) ─────────────────────
    # HARYANA  → hand-curated CIRCLE_CONTRIBUTION_PCT (haryana_circle_weights).
    # OTHERS   → learned weights from exports/location_weights_<state>.json
    #            (precomputed via scripts/learn_location_weights.py).
    aggregator = None
    aggregator_label = ""
    if state == "HARYANA":
        try:
            from .haryana_circle_weights import aggregate_weather_by_circle  # type: ignore
        except Exception:
            try:
                from haryana_circle_weights import aggregate_weather_by_circle  # type: ignore
            except Exception:
                aggregate_weather_by_circle = None  # type: ignore
        if aggregate_weather_by_circle is not None:
            aggregator = lambda loc_df: aggregate_weather_by_circle(  # noqa: E731
                loc_df, location_col="location", block_col="time_block"
            )
            aggregator_label = "haryana_circle_weights"
    else:
        try:
            from .learned_location_weights import (  # type: ignore
                load_cached_weights,
                aggregate_with_learned_weights,
            )
        except Exception:
            try:
                from learned_location_weights import (  # type: ignore
                    load_cached_weights,
                    aggregate_with_learned_weights,
                )
            except Exception:
                load_cached_weights = None  # type: ignore
                aggregate_with_learned_weights = None  # type: ignore
        if load_cached_weights is not None and aggregate_with_learned_weights is not None:
            cached = load_cached_weights(state)
            if cached:
                aggregator = lambda loc_df, _w=cached: aggregate_with_learned_weights(  # noqa: E731
                    loc_df, _w, location_col="location", block_col="time_block"
                )
                aggregator_label = f"learned_weights({len(cached)} locs)"

    if aggregator is not None:
            try:
                resp_loc = _req.get(
                    f"{pipeline_base}/weather/by_location",
                    params={"state": state, "date": t2_date, "limit": 5000},
                    headers=headers,
                    timeout=10,
                )
                if resp_loc.status_code == 200:
                    loc_rows = resp_loc.json()
                    if isinstance(loc_rows, list) and loc_rows:
                        loc_df = pd.DataFrame(loc_rows)
                        rename = {
                            "temperature_2m": "temperature",
                            "relative_humidity_2m": "humidity",
                        }
                        loc_df = loc_df.rename(
                            columns={k: v for k, v in rename.items() if k in loc_df.columns}
                        )
                        if "location" not in loc_df.columns:
                            for alt in ("city", "station", "name"):
                                if alt in loc_df.columns:
                                    loc_df = loc_df.rename(columns={alt: "location"})
                                    break
                        loc_df["time_block"] = pd.to_numeric(
                            loc_df.get("time_block", loc_df.get("block", pd.Series(dtype=float))),
                            errors="coerce",
                        ).astype("Int64")
                        loc_df = loc_df[loc_df["time_block"].between(1, 96)].copy()
                        if not loc_df.empty and "location" in loc_df.columns:
                            wdf = aggregator(loc_df)
                            if wdf is not None and not wdf.empty and len(wdf) >= 10:
                                wdf["date"] = t2_date
                                wdf["time_block"] = wdf["time_block"].astype("Int64")
                                wdf = wdf.sort_values("time_block").reset_index(drop=True)
                                wdf = _recompute_precipitation_if_split(wdf)
                                _log.info(
                                    "[weather] MySQL per-location aggregate: %d blocks for %s %s "
                                    "(method=%s)",
                                    len(wdf), region, t2_date, aggregator_label,
                                )
                                return wdf
            except _req.exceptions.RequestException as exc:
                _log.info("[weather] per-location endpoint not available, falling back to /weather/mean: %s", exc)
            except Exception as exc:
                _log.warning("[weather] per-location aggregation failed, falling back: %s", exc)

    try:
        resp = _req.get(
            f"{pipeline_base}/weather/mean",
            params={"state": state, "date": t2_date, "limit": 100},
            headers=headers,
            timeout=10,
        )
    except _req.exceptions.RequestException as exc:
        raise WeatherDataUnavailable(
            f"MySQL weather API unreachable ({pipeline_base}): {exc}"
        ) from exc

    if resp.status_code != 200:
        raise WeatherDataUnavailable(
            f"MySQL weather API returned {resp.status_code} for {state} {t2_date}"
        )

    try:
        rows = resp.json()
    except ValueError as exc:
        raise WeatherDataUnavailable(f"MySQL weather API returned non-JSON: {exc}") from exc

    if not isinstance(rows, list) or not rows:
        raise WeatherDataUnavailable(
            f"No weather data in MySQL for {region} on {t2_date}"
        )

    wdf = pd.DataFrame(rows)
    rename = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "rain_mm": "rain",
        "showers_mm": "showers",
        "snowfall_cm": "snowfall",
    }
    wdf = wdf.rename(columns={k: v for k, v in rename.items() if k in wdf.columns})
    wdf["date"]       = pd.to_datetime(wdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wdf["time_block"] = pd.to_numeric(
        wdf.get("time_block", wdf.get("block", pd.Series(dtype=float))),
        errors="coerce",
    ).astype("Int64")
    wdf = wdf[wdf["time_block"].between(1, 96)].sort_values("time_block").reset_index(drop=True)

    if len(wdf) < 10:
        raise WeatherDataUnavailable(
            f"MySQL weather for {region} {t2_date} has only {len(wdf)} blocks (need ≥10)"
        )

    wdf = _recompute_precipitation_if_split(wdf)

    _log.info("[weather] MySQL: %d blocks for %s %s", len(wdf), region, t2_date)
    return wdf


def _inject_weather_into_rows(target_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite weather columns in target_df with values from weather_df,
    matched by time_block.  Any column present in weather_df but not in
    target_df is added.  time_block must be an integer key in both frames.
    """
    WEATHER_COLS = [
        "temperature", "humidity",
        # Precipitation aggregate + decomposed forms.  Open-Meteo and similar
        # providers expose rain, showers, and snowfall separately; their sum
        # equals "precipitation".  We surface all four so the UI can show the
        # right form (rain vs snowfall has very different load impact) and
        # so future coefficient layers can treat them differently.
        "precipitation", "rain", "showers", "snowfall",
        "apparent_temperature", "cloud_cover", "cloud_cover_low",
        "sunshine_duration", "direct_radiation", "wind_speed_10m",
        "wind_speed_80m",   # hub-height wind — SLDC feeder-cut detection
    ]

    result = target_df.copy()
    result["time_block"] = pd.to_numeric(result["time_block"], errors="coerce").astype("Int64")
    wdf = weather_df.copy()
    wdf["time_block"] = pd.to_numeric(wdf["time_block"], errors="coerce").astype("Int64")

    available = [c for c in WEATHER_COLS if c in wdf.columns]
    if not available:
        return result

    # Reindex weather to cover all 96 blocks, forward-fill gaps
    wdf_full = wdf.set_index("time_block")[available].reindex(range(1, 97)).ffill().bfill()

    for col in available:
        vals = wdf_full[col].values
        if col not in result.columns:
            result[col] = 0.0
        # Align by time_block index
        tb = result["time_block"].to_numpy(dtype=int)
        result[col] = [float(vals[b - 1]) if 1 <= b <= 96 else float(vals[0]) for b in tb]

    return result


# ── T+2 (Day-After-Tomorrow) Forecast Pipeline ────────────────────────────────
def _get_lag7_row(df: pd.DataFrame, target_date: str) -> Optional[pd.DataFrame]:
    """Return the 96-block actual for the same weekday 7 days before target_date.

    Used as a real-data anchor for T+2 so the pipeline does not propagate
    T+1 forecast error into T+2's baseline.  Returns None when lag-7 data
    is missing or incomplete (< 48 blocks).
    """
    try:
        lag7_date = (pd.Timestamp(target_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        rows = df[df["date"].astype(str) == lag7_date].copy()
        if rows.empty or int(rows["time_block"].nunique()) < 48:
            return None
        return rows
    except Exception:
        return None


def run_t2_pipeline(
    df: pd.DataFrame,
    t1_date: str,
    config: Optional[Dict] = None,
    region: Optional[str] = None,
) -> Dict:
    """
    Run the short-term pipeline for T+2 (the day after t1_date).

    Strategy:
    - Forecast T+1 first with actual_blocks=0.
    - Replace the T+1 day in the history panel with the forecasted T+1 curve so
      the T+2 run sees a sequential prior day rather than hidden actuals or a
      separate baseline lookup.
    - Forecast T+2 with actual_blocks=0.
    - Apply a short seam taper so T+1 block 96 and T+2 block 1 remain continuous.
    """
    df = _ensure_time_block_1_based(df)

    try:
        t1_dt = pd.Timestamp(t1_date)
    except Exception:
        raise ValueError(f"Invalid t1_date: {t1_date!r}")

    # ── Data availability check ───────────────────────────────────────────────
    # Layout:
    #   base_date  (April 24) = last date with real actuals in df
    #   t1_date    (April 25) = T+1 target — forecasted, no real actuals
    #   t2_date    (April 26) = T+2 target — forecasted, no real actuals
    #
    # We need at least some data for the base_date so the pipeline has recent
    # context.  If base_date is completely missing, reject early with a clear
    # message.  Partial data (< 96 blocks) is allowed — the pipeline handles it.
    import logging as _log_mod
    _avail_logger = _log_mod.getLogger(__name__)

    base_date_str = (t1_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_day_df = df[df["date"].astype(str) == base_date_str]
    base_blocks = int(base_day_df["time_block"].nunique()) if not base_day_df.empty else 0

    # Also find the true latest date in the df (may be newer than base_date_str)
    available_dates = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
    latest_date = available_dates[-1] if available_dates else None
    latest_blocks = (
        int(df[df["date"].astype(str) == latest_date]["time_block"].nunique())
        if latest_date else 0
    )

    if base_blocks == 0 and latest_blocks == 0:
        raise ValueError(
            "No historical data available. Please load the latest data and retry."
        )

    if base_blocks == 0 and latest_date:
        raise ValueError(
            f"No data for reference date {base_date_str}. "
            f"Latest available: {latest_date} with {latest_blocks}/96 blocks. "
            "Please add the latest data and retry."
        )

    _avail_logger.info(
        "[T+2] base=%s %d/96 blocks | t1=%s | t2=%s",
        base_date_str, base_blocks, t1_date, (t1_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    t2_dt = t1_dt + pd.Timedelta(days=1)
    t2_date = t2_dt.strftime("%Y-%m-%d")

    cfg = dict(config or {})
    cfg.setdefault("forecast_horizon", "t2")
    cfg.setdefault("t2_level_norm", {"enabled": True, "blend": 0.45, "ratio_cap": 1.12})
    # Ensure T+2 bias correction is always on unless explicitly disabled
    if "accuracy_corrections" not in cfg or not isinstance(cfg.get("accuracy_corrections"), dict):
        cfg["accuracy_corrections"] = {}
    cfg["accuracy_corrections"].setdefault("t2_night_bias_enabled", True)
    accuracy_cfg = cfg["accuracy_corrections"]
    _hc = cfg.get("horizon_calibration", {}) if isinstance(cfg.get("horizon_calibration", {}), dict) else {}
    _hc.setdefault("enabled", True)
    _hc.setdefault("t2_blend", 0.72)
    _hc.setdefault("t2_segment_blend", 0.45)
    _hc.setdefault("t2_ratio_cap", 0.08)
    _hc.setdefault("t2_live_decay_half_blocks", 12.0)
    cfg["horizon_calibration"] = _hc
    # Signal to sub-pipelines whether the base date has complete data
    cfg["_t2_base_blocks"] = base_blocks
    cfg["_t2_base_complete"] = base_blocks == 96

    seam_cfg = cfg.get("t2_seam_bridge", {}) if isinstance(cfg.get("t2_seam_bridge", {}), dict) else {}
    # Smaller seam window so T+1 forecast error doesn't bleed into T+2's morning.
    # Each T+2 day now runs its own similar-day selection (Phase 1 rewrite),
    # so it doesn't need a wide bridge to look "continuous".
    seam_window = int(np.clip(int(seam_cfg.get("window_blocks", 4)), 1, _BLOCK_COUNT))

    # Infer region from df if not explicitly passed (used for weather DB fetch).
    _region = region
    if not _region and "state" in df.columns:
        _region = str(df["state"].dropna().iloc[0]) if not df["state"].dropna().empty else None
    if not _region and "region" in df.columns:
        _region = str(df["region"].dropna().iloc[0]) if not df["region"].dropna().empty else None

    # T+1 template rows: may not exist in the historical dataframe (future forecast).
    # If missing, synthesize a placeholder day from the latest available day and
    # inject weather for the target date when possible.
    t1_rows = df[df["date"].astype(str) == t1_date].copy()
    if t1_rows.empty:
        dates_in_df = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
        if not dates_in_df:
            raise ValueError("No dates available in dataframe; cannot synthesise T+1/T+2.")

        # Prefer the latest date strictly before t1_date as template; else fallback to latest.
        try:
            t1_ts = pd.to_datetime(t1_date, errors="coerce")
        except Exception:
            t1_ts = pd.NaT

        template_date = dates_in_df[-1]
        if pd.notna(t1_ts):
            dts = pd.to_datetime(pd.Series(dates_in_df), errors="coerce")
            prior = dts[dts < t1_ts]
            if not prior.empty:
                template_date = str(pd.Series(dates_in_df).iloc[int(prior.index.max())])

        template_rows = df[df["date"].astype(str) == template_date].copy()
        if template_rows.empty:
            raise ValueError(f"Template date {template_date} not found; cannot synthesise T+1 day {t1_date}.")

        t1_rows = _build_synthetic_forecast_day(
            template_df=template_rows,
            target_date=t1_date,
            forecast=np.zeros(_BLOCK_COUNT, dtype=float),
            zero_actuals=True,
        )
        t1_rows["_is_synthetic"] = 1

        # Fetch real weather for T+1 if available; else keep template weather.
        try:
            if _region:
                t1_weather_df = _fetch_t2_weather_from_db(t1_date, _region)
            else:
                t1_weather_df = None
        except Exception:
            t1_weather_df = None

        if t1_weather_df is not None:
            t1_rows = _inject_weather_into_rows(t1_rows, t1_weather_df)

        # Ensure the pipeline sees the target_date rows.
        df = pd.concat([df, t1_rows], ignore_index=True)

    # ── Step 1: Forecast T+1 (April 25) with zero actuals ───────────────────
    # T+1 is a future date — it has no real actuals yet.  The pipeline uses the
    # full historical df (which now includes the complete base-date / April 24 data)
    # as context.  actual_blocks=0 tells it: "produce a pure forward forecast."
    # DO NOT pass the synthetic t1_rows block count — those are placeholder zeros.
    t1_result = run_short_term_pipeline(
        df=df,
        target_date=t1_date,
        actual_blocks=0,
        config=cfg,
    )
    t1_forecast = _normalize_block_vector(
        ((t1_result.get("series") or {}).get("forecast") if isinstance(t1_result, dict) else None),
        length=_BLOCK_COUNT,
        fill_value=0.0,
    )

    # ── Step 2: Build T+1 synthetic row for T+2 context ─────────────────────
    # FIX 1: Use lag-7 real actuals as the T+1 load anchor instead of the T+1
    # forecast.  This breaks the error-propagation chain: if T+1 forecast is
    # wrong, T+2 no longer inherits that error through its "yesterday" context.
    # The seam continuity correction (below) still uses t1_forecast to ensure
    # the T+2 curve starts at the right absolute level — only the baseline
    # SHAPE context switches to a real historical anchor.
    _lag7_rows = _get_lag7_row(df, t1_date)
    if _lag7_rows is not None:
        _lag7_load = (
            _lag7_rows.groupby("time_block")["total_drawal"]
            .mean()
            .reindex(range(1, 97))
            .ffill().bfill()
            .to_numpy(dtype=float)
        )
        t1_synthetic = _build_synthetic_forecast_day(
            template_df=t1_rows,
            target_date=t1_date,
            forecast=_lag7_load,
            zero_actuals=False,
        )
        _avail_logger.info(
            "[T+2] Using lag-7 real anchor (%s) as T+1 synthetic load for T+2 context.",
            (pd.Timestamp(t1_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        )
    else:
        # Lag-7 not available — fall back to T+1 forecast (old behaviour)
        t1_synthetic = _build_synthetic_forecast_day(
            template_df=t1_rows,
            target_date=t1_date,
            forecast=t1_forecast,
            zero_actuals=False,
        )
        _avail_logger.warning("[T+2] Lag-7 data unavailable — using T+1 forecast as synthetic anchor.")
    t1_synthetic["_is_synthetic"] = 1

    # ── Step 3: Build T+2 rows — use DB weather if available ─────────────────
    dates_in_df = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []

    # Fetch real T+2 weather from MySQL via Django API
    t2_weather_df = None
    _t2_wx_log = logging.getLogger(__name__)
    if _region:
        t2_weather_df = _fetch_t2_weather_from_db(t2_date, _region)
        if t2_weather_df is not None:
            _t2_wx_log.info(
                "[T+2] Fetched %d weather blocks for %s %s from DB",
                len(t2_weather_df), _region, t2_date,
            )
        else:
            # DB fetch failed — build a smarter proxy from historical same-weekday
            # same-season averages rather than copying T+1 verbatim.
            _t2_wx_log.warning(
                "[T+2] No DB weather for %s %s — building climatological proxy",
                _region, t2_date,
            )
            try:
                _t2_dt_obj = pd.Timestamp(t2_date)
                _wx_cols = [c for c in [
                    "temperature", "humidity", "precipitation",
                    "apparent_temperature", "cloud_cover", "wind_speed_10m",
                ] if c in df.columns]
                if _wx_cols and "time_block" in df.columns:
                    _hist_wx = df.copy()
                    _hist_wx["date"] = _hist_wx["date"].astype(str)
                    _hist_wx = _hist_wx[_hist_wx["date"] < t2_date]
                    # Same season + same weekday (±1 day) for last 60 days
                    try:
                        _hist_wx["_season"] = _hist_wx["date"].apply(_season)
                        _hist_wx = _hist_wx[_hist_wx["_season"] == _season(t2_date)]
                    except Exception:
                        pass
                    _hist_wx_dates = sorted(_hist_wx["date"].dropna().unique().tolist())
                    _hist_wx = _hist_wx[_hist_wx["date"].isin(set(_hist_wx_dates[-60:]))]
                    _hist_wx["_dow"] = pd.to_datetime(_hist_wx["date"], errors="coerce").dt.dayofweek
                    _target_dow = int(_t2_dt_obj.dayofweek)
                    _same_dow = _hist_wx[_hist_wx["_dow"].isin([(_target_dow - 1) % 7, _target_dow, (_target_dow + 1) % 7])]
                    if len(_same_dow["date"].unique()) >= 3:
                        _hist_wx = _same_dow
                    proxy_wx = (
                        _hist_wx.groupby("time_block")[_wx_cols]
                        .mean()
                        .reindex(range(1, 97))
                        .ffill()
                        .bfill()
                        .reset_index()
                        .rename(columns={"index": "time_block"})
                    )
                    proxy_wx["date"] = t2_date
                    t2_weather_df = proxy_wx
                    _t2_wx_log.info("[T+2] Climatological proxy built from %d history days", len(_hist_wx["date"].unique()))
            except Exception as _px_err:
                _t2_wx_log.warning("[T+2] Proxy weather build failed: %s — T+1 weather will be copied", _px_err)
    else:
        _t2_wx_log.warning("[T+2] region unknown — skipping DB weather fetch for %s", t2_date)

    if t2_date in dates_in_df:
        # T+2 already in df — use it but override weather with fresh DB data
        t2_rows = df[df["date"].astype(str) == t2_date].copy()
        if t2_weather_df is not None:
            t2_rows = _inject_weather_into_rows(t2_rows, t2_weather_df)
    else:
        # T+2 not in df — synthesise from T+1 template, then inject DB/proxy weather
        t2_rows = _build_synthetic_forecast_day(
            template_df=t1_rows,
            target_date=t2_date,
            forecast=np.zeros(_BLOCK_COUNT, dtype=float),
            zero_actuals=True,
        )
        if t2_weather_df is not None:
            t2_rows = _inject_weather_into_rows(t2_rows, t2_weather_df)

    # ── Step 4: Build extended history — replace T+1 actual with T+1 forecast -
    #   History up to (not including) T+1, then T+1-as-forecast, then T+2.
    history_without_t1 = df[df["date"].astype(str) != t1_date].copy()
    history_without_t1_t2 = history_without_t1[history_without_t1["date"].astype(str) != t2_date].copy()
    df_extended = pd.concat([history_without_t1_t2, t1_synthetic, t2_rows], ignore_index=True)

    result = run_short_term_pipeline(
        df=df_extended,
        target_date=t2_date,
        actual_blocks=0,
        config=cfg,
    )

    result_series = result.get("series", {}) if isinstance(result, dict) else {}
    t2_forecast = _normalize_block_vector(result_series.get("forecast"), length=_BLOCK_COUNT, fill_value=0.0)

    # ── T+2 Load-Level Normalisation (optional) ──────────────────────────────
    # When T+1 overnight forecast is meaningfully higher than T+2 overnight
    # (i.e., T+1 "sees" higher current-year load than T+2's similar-day pool),
    # gently scale T+2 up.  Only applies in the up direction — avoids amplifying
    # over-forecasts.  Conservative blend (0.30) keeps the correction subtle.
    # Enabled by route defaults for T+2; can be disabled via cfg["t2_level_norm"]["enabled"]=false.
    _t2_norm_cfg     = cfg.get("t2_level_norm", {})
    _t2_norm_enabled = bool(_t2_norm_cfg.get("enabled", False))
    _t2_norm_blend   = float(_t2_norm_cfg.get("blend", 0.30))
    _t2_norm_cap     = float(_t2_norm_cfg.get("ratio_cap", 1.05))
    _t2_level_ratio  = 1.0
    if _t2_norm_enabled:
        try:
            _t1_night = t1_forecast[:24]
            _t2_night = t2_forecast[:24]
            _t1_night_mean = float(np.mean(_t1_night[_t1_night > 0])) if (_t1_night > 0).any() else 0.0
            _t2_night_mean = float(np.mean(_t2_night[_t2_night > 0])) if (_t2_night > 0).any() else 0.0
            if _t2_night_mean > 100.0 and _t1_night_mean > 100.0:
                raw_ratio = _t1_night_mean / _t2_night_mean

                # Secondary midday calibration anchor (blocks 41-48, 10:15-12:00).
                # Night-only normalisation fails for the midday ramp because T+2's
                # similar-day pool may have a different load level than T+1 around noon.
                _t1_mid = t1_forecast[40:48]
                _t2_mid = t2_forecast[40:48]
                _t1_mid_mean = float(np.mean(_t1_mid[_t1_mid > 0])) if (_t1_mid > 0).any() else 0.0
                _t2_mid_mean = float(np.mean(_t2_mid[_t2_mid > 0])) if (_t2_mid > 0).any() else 0.0
                if _t2_mid_mean > 500.0 and _t1_mid_mean > 500.0:
                    mid_ratio = _t1_mid_mean / _t2_mid_mean
                    # Blend night and midday ratios 70/30 to preserve night dominance
                    raw_ratio = 0.70 * raw_ratio + 0.30 * mid_ratio

                # Only scale up (fix under-forecast) — never scale down
                if raw_ratio > 1.0:
                    capped_ratio = float(np.clip(raw_ratio, 1.0, _t2_norm_cap))
                    _t2_level_ratio = capped_ratio
                    t2_scaled   = t2_forecast * capped_ratio
                    t2_forecast = (1.0 - _t2_norm_blend) * t2_forecast + _t2_norm_blend * t2_scaled
                    _avail_logger.info(
                        "[T+2] Level norm (up-only): T+1_night=%.0f T+2_night=%.0f "
                        "ratio=%.3f (incl midday blend) blend=%.2f",
                        _t1_night_mean, _t2_night_mean, capped_ratio, _t2_norm_blend,
                    )
        except Exception as _norm_err:
            _avail_logger.warning("[T+2] Load-level normalisation failed: %s", _norm_err)

    t2_bias_meta: Dict[str, Any] = {"enabled": False}
    t2_bias_applied = np.zeros(_BLOCK_COUNT, dtype=float)
    if (
        build_t2_bias_correction is not None
        and apply_t2_night_correction is not None
        and bool(accuracy_cfg.get("t2_night_bias_enabled", True))
    ):
        try:
            t2_corr, t2_corr_meta = build_t2_bias_correction(
                df,
                fallback_night_mw=float(accuracy_cfg.get("t2_fallback_night_mw", 250.0)),
                min_samples=int(accuracy_cfg.get("t2_bias_min_samples", 7)),
            )
            t2_bias_result = apply_t2_night_correction(t2_forecast, t2_corr, enabled=True)
            t2_forecast = t2_bias_result.forecast
            t2_bias_applied = t2_bias_result.applied_mw
            t2_bias_meta = {**t2_corr_meta, **t2_bias_result.metadata}
            _avail_logger.info(
                "[T+2] Night bias correction applied: mean night %.1f MW",
                float(np.mean(t2_bias_applied[:32])),
            )
        except Exception as _t2_bias_err:
            t2_bias_meta = {"enabled": True, "applied": False, "reason": str(_t2_bias_err)}

    # Seam anchor: prefer lag-7 actual tail over T+1 forecast tail.
    # Using T+1 forecast (which may be 15% wrong) as the anchor propagates that error
    # into the first few T+2 blocks via the cosine taper in _apply_seam_continuity.
    # Using the real lag-7 load level breaks this chain.
    _seam_tail = int(np.clip(int(seam_cfg.get("anchor_tail_blocks", 4)), 1, min(12, _BLOCK_COUNT)))
    _lag7_anchor_rows = _get_lag7_row(df, t1_date)
    if _lag7_anchor_rows is not None:
        try:
            _lag7_tail = (
                _lag7_anchor_rows.groupby("time_block")["total_drawal"]
                .mean()
                .reindex(range(1, 97))
                .ffill().bfill()
                .to_numpy(dtype=float)
            )
            # Blend 60% lag-7 real + 40% T+1 forecast to preserve intraday level shift
            _anchor_vec = 0.60 * _lag7_tail + 0.40 * t1_forecast
        except Exception:
            _anchor_vec = t1_forecast
    else:
        _anchor_vec = t1_forecast
    _tail_slice = _anchor_vec[-_seam_tail:]
    _tail_weights = np.exp(np.linspace(0.0, 1.0, len(_tail_slice)))
    _tail_weights /= _tail_weights.sum()
    _seam_anchor = float(np.dot(_tail_weights, _tail_slice))
    t2_adjusted, seam_gap_before, seam_gap_after = _apply_seam_continuity(
        forecast=t2_forecast,
        previous_terminal_mw=_seam_anchor,
        window=seam_window,
    )

    if isinstance(result_series, dict):
        result_series["forecast"] = t2_adjusted.tolist()
        if "final_load" in result_series:
            result_series["final_load"] = t2_adjusted.tolist()
        if "hybrid_ai_forecast" in result_series:
            result_series["hybrid_ai_forecast"] = t2_adjusted.tolist()
        result_series["t2_bias_correction"] = t2_bias_applied.tolist()
        
        # Inject T+1 forecast as "yesterday" for UI comparison
        result_series["yesterday"] = t1_forecast.tolist()

    if isinstance(result.get("forecast_df"), list):
        for idx, row in enumerate(result["forecast_df"][:_BLOCK_COUNT]):
            if isinstance(row, dict):
                row["forecast"] = float(t2_adjusted[idx])
                row["forecast_mw"] = float(t2_adjusted[idx])
                row["final_load"] = float(t2_adjusted[idx])
                # Also inject into the tabular view
                row["yesterday"] = float(t1_forecast[idx])
                row["t2_bias_correction"] = float(t2_bias_applied[idx])

    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(metadata, dict):
        metadata["sequential_forecast"] = {
            "enabled": True,
            "t1_seed_date": str(t1_date),
            "t2_target_date": str(t2_date),
            "seam_window_blocks": int(seam_window),
            "seam_gap_before_mw": round(float(seam_gap_before), 3),
            "seam_gap_after_mw": round(float(seam_gap_after), 3),
            "t2_level_norm_ratio": round(float(_t2_level_ratio), 4),
            "t2_level_norm_enabled": _t2_norm_enabled,
        }
        metadata["t2_bias_correction"] = t2_bias_meta

    result["horizon"] = "t2"
    result["t1_date"] = t1_date
    result["t2_date"] = t2_date
    result["t1_forecast"] = t1_forecast.tolist()

    # Build weather_analysis from T+2 weather data so the UI can display
    # avg temperature, humidity, precipitation metrics on the T+2 tab.
    if t2_weather_df is not None and not t2_weather_df.empty:
        try:
            _wdf = t2_weather_df.copy()
            if "time_block" in _wdf.columns:
                _wdf = _wdf.set_index("time_block").reindex(range(1, 97)).ffill().bfill()
            _w_cols = ["temperature", "humidity", "precipitation", "wind_speed",
                       "cloud_cover", "direct_radiation", "sunshine_duration", "solar_radiation"]
            for _c in _w_cols:
                if _c not in _wdf.columns:
                    _wdf[_c] = 0.0
            if "wind_speed" not in _wdf.columns and "wind_speed_10m" in _wdf.columns:
                _wdf["wind_speed"] = _wdf["wind_speed_10m"]
            if "solar_radiation" not in _wdf.columns and "direct_radiation" in _wdf.columns:
                _wdf["solar_radiation"] = _wdf["direct_radiation"]
            # Use T+1 rows as "normal" baseline for T+2 intraday comparison
            _t1_rows = df[df["date"].astype(str) == t1_date].copy()
            if not _t1_rows.empty:
                # Deduplicate time_block to avoid non-unique index error
                _t1_rows = _t1_rows.sort_values("time_block").drop_duplicates("time_block", keep="last")
                _normal = _t1_rows.set_index("time_block").reindex(range(1, 97)).ffill().bfill()
            else:
                _normal = _wdf.copy()
            _intra = {}
            for _col in ["temperature", "humidity", "precipitation", "cloud_cover",
                         "solar_radiation", "wind_speed"]:
                _intra[_col] = {
                    "actual": [round(float(v), 2) for v in _wdf[_col].tolist()],
                    "normal": [round(float(v), 2) for v in (_normal[_col].tolist() if _col in _normal.columns else _wdf[_col].tolist())],
                    "delta":  [round(float(a - n), 2) for a, n in zip(_wdf[_col].tolist(), (_normal[_col].tolist() if _col in _normal.columns else [0]*96))],
                }
            _precip = _wdf["precipitation"]
            result["weather_analysis"] = {
                "intraday": _intra,
                "peak_windows": {},
                "rain_metrics": {
                    "total_mm": round(float(_precip.sum()), 1),
                    "max_intensity": round(float(_precip.max()), 1),
                    "rain_hours": round(float((_precip > 0).sum() * 0.25), 1),
                    "load_impact_mw": 0.0,
                    "cloud_factor": 0.0,
                },
                "dod_changes": {},
                "dod_series": {},
                "sensitivity": {},
            }
        except Exception as _wx_err:
            _log.warning("[T+2] weather_analysis build failed: %s", _wx_err)

    return result
