"""
ML Baseline Forecaster
======================
Trains a gradient-boosted model (XGBoost → LightGBM → Ridge fallback) on
the available history and predicts a 96-block baseline load curve for a
target date.

Architecture role
-----------------
This module implements **Option 1** of the hybrid upgrade plan:

    [Load + Weather + Time Features]
              ↓
        MLBaselineForecaster
              ↓
        baseline_load  (96 blocks)
              ↓
      run_block_driver_weight_delta_engine   ← unchanged
              ↓
        Final Forecast  (explainable + accurate)

The delta engine remains the control layer. ML only learns
"what is the normal expected load?" — it does not replace
weather sensitivity, manual overrides, or the weight matrix.

Caching
-------
A module-level LRU cache avoids re-training on every API call.
The cache key is (number_of_rows, last_date_in_history).  When new
data arrives the cache is automatically invalidated.

Usage
-----
    from backend.ml_baseline import compute_ml_baseline

    ml_baseline, meta = compute_ml_baseline(df, target_date="2024-06-15")
    # ml_baseline: np.ndarray shape (96,)
    # meta: dict with model_name, feature_importance, train_rows, ...
"""

from __future__ import annotations

import logging
import hashlib
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional boosting libraries (same try/except pattern as the pipeline)
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor as _XGB
except Exception:
    _XGB = None

try:
    from lightgbm import LGBMRegressor as _LGB
except Exception:
    _LGB = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BLOCK_COUNT = 96
_WEATHER_COLS = [
    "temperature", "humidity", "precipitation",
    "apparent_temperature", "cloud_cover", "sunshine_duration",
    "direct_radiation", "wind_speed_10m",
]
# Blend weight for ML prediction vs similar-day baseline.
# 0.0 = pure similar-day (safe), 1.0 = pure ML.
# Start conservative; raise after validation.
ML_BLEND_WEIGHT: float = 0.45
# Minimum training rows required before the ML model is used.
_MIN_TRAIN_ROWS: int = 96 * 14  # 14 days × 96 blocks


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _season_int(month: int) -> int:
    if month in (11, 12, 1, 2):
        return 0   # winter
    if month in (3, 4, 5, 6):
        return 1   # summer
    return 2       # monsoon


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy DataFrame with one row per (date, time_block).

    Columns added
    -------------
    Time:   block_sin, block_cos, hour_sin, hour_cos,
            dow_sin, dow_cos, month_sin, month_cos,
            is_weekend, season_enc
    Lags:   load_lag_1d, load_lag_7d, load_lag_14d, load_rolling_4w
    Target: total_drawal   (untouched — used as y during training)
    """
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce")
    df["total_drawal"] = pd.to_numeric(df["total_drawal"], errors="coerce")
    for col in _WEATHER_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = (
        df.dropna(subset=["date", "time_block", "total_drawal"])
        .sort_values(["date", "time_block"])
        .reset_index(drop=True)
    )
    df = df[df["time_block"].between(1, _BLOCK_COUNT)]
    df["time_block"] = df["time_block"].astype(int)

    dt = pd.to_datetime(df["date"], errors="coerce")
    block = df["time_block"].astype(float)
    hour = (block - 1) * 0.25
    dow = dt.dt.dayofweek.astype(float).fillna(0.0)
    month = dt.dt.month.astype(float).fillna(6.0)

    df["block_sin"] = np.sin(2 * np.pi * block / _BLOCK_COUNT)
    df["block_cos"] = np.cos(2 * np.pi * block / _BLOCK_COUNT)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["is_weekend"] = (dow >= 5).astype(float)
    df["season_enc"] = month.apply(lambda m: float(_season_int(int(m))))

    # -------------------------------------------------------------------
    # Load lags — shift by 96 rows = 1 day (assuming sorted, dense data).
    # Using groupby on time_block ensures the shift is block-aligned even
    # if the global index has gaps.
    # -------------------------------------------------------------------
    def _block_lag(series: pd.Series, n_days: int) -> pd.Series:
        return series.shift(n_days)

    gb = df.groupby("time_block")["total_drawal"]
    df["load_lag_1d"] = gb.shift(1)
    df["load_lag_7d"] = gb.shift(7)
    df["load_lag_14d"] = gb.shift(14)
    df["load_rolling_4w"] = gb.transform(
        lambda x: x.shift(1).rolling(28, min_periods=4).mean()
    )

    return df


_FEATURE_COLS = [
    # weather
    "temperature", "humidity", "precipitation",
    "apparent_temperature", "cloud_cover", "sunshine_duration",
    "direct_radiation", "wind_speed_10m",
    # time
    "block_sin", "block_cos", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "season_enc",
    # lags
    "load_lag_1d", "load_lag_7d", "load_lag_14d", "load_rolling_4w",
    # block index (lets the model learn block-specific intercepts)
    "time_block",
]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model() -> Any:
    """Return the best available gradient-boosted regressor."""
    if _XGB is not None:
        return _XGB(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    if _LGB is not None:
        return _LGB(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    # Ridge fallback: scale + linear model
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def _model_name(model: Any) -> str:
    if _XGB is not None and isinstance(model, _XGB):
        return "xgboost"
    if _LGB is not None and isinstance(model, _LGB):
        return "lightgbm"
    return "ridge"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[Any, Dict]] = {}


def _cache_key(df: pd.DataFrame) -> str:
    """Cheap fingerprint: row count + last date. Invalidates when new data arrives."""
    last_date = str(df["date"].max()) if "date" in df.columns else "unknown"
    raw = f"{len(df)}|{last_date}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ml_baseline(
    df: pd.DataFrame,
    target_date: str,
    blend_weight: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Train (or retrieve from cache) an ML baseline model and predict 96-block
    baseline load for *target_date*.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical data including the target date's weather rows.
        Must contain columns: date, time_block, total_drawal, and the
        8 weather columns in _WEATHER_COLS.
    target_date : str
        ISO date string, e.g. "2024-06-15".
    blend_weight : float, optional
        Override for ML_BLEND_WEIGHT (0 = pure similar-day, 1 = pure ML).

    Returns
    -------
    ml_baseline : np.ndarray, shape (96,)
        96-block predicted baseline. All values are finite and positive.
    meta : dict
        Keys: model_name, train_rows, feature_importance (top 10),
              is_cached, n_features, blend_weight.
    """
    bw = ML_BLEND_WEIGHT if blend_weight is None else float(blend_weight)
    key = _cache_key(df)
    target_date_str = str(target_date)

    # -----------------------------------------------------------------------
    # Build feature DataFrame
    # -----------------------------------------------------------------------
    feat_df = build_features(df)

    train_df = feat_df[feat_df["date"] < target_date_str].dropna(subset=_FEATURE_COLS + ["total_drawal"])
    pred_df = feat_df[feat_df["date"] == target_date_str].dropna(subset=_FEATURE_COLS)

    n_train = len(train_df)
    meta_base: Dict[str, Any] = {
        "blend_weight": bw,
        "train_rows": n_train,
        "model_name": "skipped",
        "is_cached": False,
        "feature_importance": {},
        "n_features": len(_FEATURE_COLS),
    }

    # Bail early if not enough training data or no target rows
    if n_train < _MIN_TRAIN_ROWS:
        logger.debug(
            "ML baseline skipped: only %d training rows, need %d.", n_train, _MIN_TRAIN_ROWS
        )
        return np.zeros(_BLOCK_COUNT, dtype=float), {**meta_base, "model_name": "insufficient_data"}

    if pred_df.empty:
        logger.warning("ML baseline: no feature rows found for target_date=%s.", target_date_str)
        return np.zeros(_BLOCK_COUNT, dtype=float), {**meta_base, "model_name": "no_target_rows"}

    # -----------------------------------------------------------------------
    # Train (or use cache)
    # -----------------------------------------------------------------------
    cache_entry = _CACHE.get(key)
    if cache_entry is not None:
        model, fit_meta = cache_entry
        is_cached = True
        logger.debug("ML baseline: using cached model (%s).", fit_meta.get("model_name"))
    else:
        X_train = train_df[_FEATURE_COLS].to_numpy(dtype=float)
        y_train = train_df["total_drawal"].to_numpy(dtype=float)

        model = _build_model()
        model_label = _model_name(model)
        logger.info("ML baseline: fitting %s on %d training rows...", model_label, n_train)
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning("ML baseline training failed: %s", exc)
            return np.zeros(_BLOCK_COUNT, dtype=float), {**meta_base, "model_name": "train_error"}

        fit_meta = {"model_name": model_label}
        if len(_CACHE) >= 8:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (model, fit_meta)
        is_cached = False
        logger.info("ML baseline: %s trained ✓ (%d rows)", model_label, n_train)

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------
    X_pred = pred_df[_FEATURE_COLS].to_numpy(dtype=float)
    try:
        y_pred = model.predict(X_pred)
    except Exception as exc:
        logger.warning("ML baseline prediction failed: %s", exc)
        return np.zeros(_BLOCK_COUNT, dtype=float), {**meta_base, "model_name": "predict_error"}

    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0.0)  # no negative load

    # Align predictions to 96-block vector by time_block index
    ml_baseline = np.zeros(_BLOCK_COUNT, dtype=float)
    pred_blocks = pred_df["time_block"].to_numpy(dtype=int)
    for i, blk in enumerate(pred_blocks):
        idx = int(blk) - 1
        if 0 <= idx < _BLOCK_COUNT:
            ml_baseline[idx] = float(y_pred[i])

    # Fill any zero gaps (missing blocks) with neighbour interpolation
    zero_mask = ml_baseline == 0.0
    if zero_mask.any() and not zero_mask.all():
        indices = np.arange(_BLOCK_COUNT, dtype=float)
        known = ~zero_mask
        ml_baseline[zero_mask] = np.interp(indices[zero_mask], indices[known], ml_baseline[known])

    # -----------------------------------------------------------------------
    # Feature importance (top 10 for transparency)
    # -----------------------------------------------------------------------
    fi: Dict[str, float] = {}
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            ranked = sorted(zip(_FEATURE_COLS, imp), key=lambda t: t[1], reverse=True)
            fi = {k: round(float(v), 5) for k, v in ranked[:10]}
    except Exception:
        pass

    return ml_baseline, {
        "blend_weight": bw,
        "train_rows": n_train,
        "model_name": fit_meta["model_name"],
        "is_cached": is_cached,
        "feature_importance": fi,
        "n_features": len(_FEATURE_COLS),
    }


def blend_baselines(
    ml_baseline: np.ndarray,
    stat_baseline: np.ndarray,
    blend_weight: float = ML_BLEND_WEIGHT,
) -> np.ndarray:
    """
    Weighted blend of ML and statistical (similar-day) baselines.

        result = blend_weight * ml + (1 - blend_weight) * stat

    If the ML baseline is all-zero (model skipped/failed) the statistical
    baseline is returned unchanged.
    """
    ml = np.asarray(ml_baseline, dtype=float)
    stat = np.asarray(stat_baseline, dtype=float)
    if not np.any(ml > 0):
        return stat
    w = float(np.clip(blend_weight, 0.0, 1.0))
    return w * ml + (1.0 - w) * stat
