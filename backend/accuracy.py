"""
accuracy.py — Adaptive rolling bias-correction layer for short-term load forecasting.

Six public functions imported by short_term_pipeline.py:
    compute_bias_table            — rolling mean (forecast−actual) per block, T+1
    apply_bias_correction         — subtract bias_table from future blocks
    apply_block_type_calibration  — ratio calibration by time-of-day segment
    build_afternoon_ramp_adjustment — weather-aware afternoon upward ramp fix
    build_t2_bias_correction      — rolling + static segment correction for T+2
    apply_t2_night_correction     — add T+2 correction vector to forecast

Sign convention (shared with pipeline):
    bias_table[i]  = mean(forecast_i − actual_i)
    Positive → model over-forecasts → apply_bias_correction subtracts it
    Negative → model under-forecasts → apply_bias_correction adds it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_BLOCKS = 96

# ---------------------------------------------------------------------------
# Static T+2 segment-level fallback correction
# Derived from Haryana April 2026 MAPE analysis (2208 block observations).
# Values = -(mean signed error) so adding them to the T+2 forecast is corrective.
# Indexed by 2-hour segment start block (1-based, step 8).
# ---------------------------------------------------------------------------
_T2_STATIC_CORRECTION_BY_SEG: Dict[int, float] = {
    1:  +71,   # 00h: T+2 under-predicts night by 71 MW  → add 71
    9:  +96,   # 02h: T+2 under-predicts by 96 MW
    17: +75,   # 04h: T+2 under-predicts by 75 MW
    25: -133,  # 06h: T+2 over-predicts by 166 MW  → subtract 133 (capped)
    33:  -1,   # 08h: near-zero, ignore
    41: +28,   # 10h: slight under
    49: +200,  # 12h: T+2 under-predicts by 239 MW → add 200 (capped at 84%)
    57: +260,  # 14h: T+2 under-predicts by 318 MW → add 260 (capped at 82%)
    65:  -7,   # 16h: near-zero, ignore
    73: -280,  # 18h: T+2 over-predicts by 349 MW → subtract 280 (capped)
    81:  -90,  # 20h: T+2 over-predicts by 108 MW → subtract 90
    89: +200,  # 22h: T+2 under-predicts by 246 MW → add 200
}

# Segment length (blocks)
_SEG = 8


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class CorrectionResult:
    forecast: np.ndarray
    applied_mw: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Circular uniform smooth over a 96-block daily vector."""
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    # Pad circularly so the daily wrap-around is smooth
    pad = window // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:_BLOCKS]


def _build_static_t2_vector(blend: float = 1.0) -> np.ndarray:
    """
    Build a 96-element correction vector from the static segment table.
    blend ∈ [0, 1]: how much of the static correction to apply.
    """
    vec = np.zeros(_BLOCKS, dtype=float)
    for seg_start, corr in _T2_STATIC_CORRECTION_BY_SEG.items():
        seg_end = min(seg_start + _SEG, _BLOCKS + 1)  # 1-based inclusive
        # Convert to 0-based index range
        i0 = seg_start - 1
        i1 = min(seg_end - 1, _BLOCKS)
        vec[i0:i1] = corr * blend
    return _smooth(vec, window=6)


# ---------------------------------------------------------------------------
# 1. compute_bias_table  (T+1 rolling block-level bias)
# ---------------------------------------------------------------------------

def compute_bias_table(
    df: pd.DataFrame,
    target_date: str,
    window_days: int = 30,
    day_type: str = "weekday",
    horizon: str = "t1",
    smooth_window: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate a 96-block rolling bias table from historical actual data.

    Without stored forecast history we approximate forecast bias as the
    mean difference between a lag-7 load and the actual load for the same
    block.  This captures the systematic over/under-shoot of a calendar-lag
    baseline — a conservative proxy for the pipeline's own bias.

    Returns (bias_table, meta) where bias_table[i] = mean(forecast_i − actual_i).
    Positive → over-forecast → apply_bias_correction will subtract it.
    """
    meta: Dict[str, Any] = {"enabled": True, "applied": False, "method": "lag7_proxy"}
    bias = np.zeros(_BLOCKS, dtype=float)

    try:
        if df is None or df.empty or "date" not in df.columns:
            meta["reason"] = "no dataframe"
            return bias, meta

        df2 = df.copy()
        df2["date"] = df2["date"].astype(str)
        target_str = str(target_date)[:10]
        target_ts = pd.Timestamp(target_str)

        # Select same day-type within the window
        hist = df2[df2["date"] < target_str].copy()
        hist["_ts"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist[hist["_ts"] >= target_ts - pd.Timedelta(days=window_days)]
        if day_type == "weekend":
            hist = hist[hist["_ts"].dt.dayofweek >= 5]
        else:
            hist = hist[hist["_ts"].dt.dayofweek < 5]

        if hist.empty or "total_drawal" not in hist.columns or "time_block" not in hist.columns:
            meta["reason"] = "insufficient history"
            return bias, meta

        # Lag-7 proxy: for each historical day, find the day 7 days prior
        dates = sorted(hist["date"].unique())
        if len(dates) < 7:
            meta["reason"] = f"only {len(dates)} history days (need ≥7)"
            return bias, meta

        errors: list[np.ndarray] = []
        for d in dates:
            lag_date = (pd.Timestamp(d) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            lag_rows = df2[df2["date"] == lag_date].copy()
            actual_rows = hist[hist["date"] == d].copy()
            if lag_rows.empty or actual_rows.empty:
                continue
            lag_rows = lag_rows.sort_values("time_block")
            actual_rows = actual_rows.sort_values("time_block")
            if lag_rows["time_block"].nunique() < _BLOCKS // 2:
                continue
            lag_series = (
                lag_rows.groupby("time_block")["total_drawal"]
                .mean()
                .reindex(range(1, _BLOCKS + 1))
                .ffill()
                .bfill()
                .to_numpy()
            )
            actual_series = (
                actual_rows.groupby("time_block")["total_drawal"]
                .mean()
                .reindex(range(1, _BLOCKS + 1))
                .ffill()
                .bfill()
                .to_numpy()
            )
            # bias = lag7 − actual  (positive when lag over-estimates)
            errors.append(lag_series - actual_series)

        if not errors:
            meta["reason"] = "no valid lag pairs found"
            return bias, meta

        bias = np.nanmean(errors, axis=0)
        bias = np.nan_to_num(bias, nan=0.0)
        bias = _smooth(bias, smooth_window)

        # Cap: ±8% of recent mean actual or 400 MW whichever is smaller
        recent_mean = hist.groupby("time_block")["total_drawal"].mean().reindex(range(1, _BLOCKS + 1)).ffill().bfill().to_numpy()
        cap = np.minimum(np.maximum(recent_mean * 0.08, 50.0), 400.0)
        bias = np.clip(bias, -cap, cap)

        # Dampen: only apply 50% of the lag-7 proxy bias (conservative)
        bias *= 0.50

        meta.update({
            "applied": True,
            "n_days": len(errors),
            "mean_abs_bias_mw": round(float(np.mean(np.abs(bias))), 1),
            "max_abs_bias_mw": round(float(np.max(np.abs(bias))), 1),
        })

    except Exception as exc:
        _log.warning("[accuracy] compute_bias_table failed: %s", exc)
        meta["reason"] = str(exc)
        bias = np.zeros(_BLOCKS, dtype=float)

    return bias, meta


# ---------------------------------------------------------------------------
# 2. apply_bias_correction  (T+1)
# ---------------------------------------------------------------------------

def apply_bias_correction(
    forecast: np.ndarray,
    bias_table: np.ndarray,
    actual_blocks: int = 0,
    enabled: bool = True,
) -> CorrectionResult:
    """
    Subtract bias_table[actual_blocks:] from forecast future blocks.

    Positive bias_table → over-forecast → we subtract → lowers forecast.
    Negative bias_table → under-forecast → we add → raises forecast.

    A linear taper is applied so corrections fade to 0 by block 96,
    limiting over-reach on long forecasts.
    """
    fc = np.asarray(forecast, dtype=float).copy()
    applied = np.zeros(_BLOCKS, dtype=float)

    if not enabled or actual_blocks >= _BLOCKS:
        return CorrectionResult(fc, applied, {"enabled": False})

    try:
        bt = np.asarray(bias_table, dtype=float)
        if bt.shape[0] != _BLOCKS:
            bt = np.zeros(_BLOCKS, dtype=float)

        n_future = _BLOCKS - actual_blocks
        # Taper: full correction at actual_blocks, 30% at block 96
        taper = np.linspace(1.0, 0.30, n_future)

        # Cap correction at ±15% of forecast value or 350 MW
        correction = bt[actual_blocks:] * taper
        cap = np.minimum(np.maximum(fc[actual_blocks:] * 0.15, 50.0), 350.0)
        correction = np.clip(correction, -cap, cap)

        fc[actual_blocks:] = np.maximum(fc[actual_blocks:] - correction, 0.0)
        applied[actual_blocks:] = -correction  # negative = we subtracted

    except Exception as exc:
        _log.warning("[accuracy] apply_bias_correction failed: %s", exc)
        return CorrectionResult(
            np.asarray(forecast, dtype=float),
            np.zeros(_BLOCKS, dtype=float),
            {"enabled": True, "applied": False, "reason": str(exc)},
        )

    return CorrectionResult(
        fc,
        applied,
        {
            "enabled": True,
            "applied": True,
            "actual_blocks": actual_blocks,
            "mean_abs_correction_mw": round(float(np.mean(np.abs(applied[actual_blocks:]))), 1),
        },
    )


# ---------------------------------------------------------------------------
# 3. apply_block_type_calibration  (T+1)
# ---------------------------------------------------------------------------

# Six time-of-day segments (0-based block indices)
_TOD_SEGMENTS = {
    "night":     (0,  32),   # 00–08h
    "morning":   (32, 40),   # 08–10h
    "midday":    (40, 48),   # 10–12h
    "afternoon": (48, 64),   # 12–16h
    "evening":   (64, 80),   # 16–20h
    "latenight": (80, 96),   # 20–24h
}


def apply_block_type_calibration(
    forecast: np.ndarray,
    target_date: str,
    history_df: pd.DataFrame,
    actual_blocks: int = 0,
    window_days: int = 21,
    min_confidence: float = 0.80,
    enabled: bool = True,
) -> CorrectionResult:
    """
    Apply a segment-level ratio calibration using recent historical actuals.

    For each TOD segment compute:
        ratio = mean_actual_segment / mean_lag7_segment

    Then blend the ratio correction into the forecast:
        corrected = forecast * (1 + blend * (ratio − 1))

    where blend = min(confidence, 0.30).  This prevents over-correction.
    """
    fc = np.asarray(forecast, dtype=float).copy()
    applied = np.zeros(_BLOCKS, dtype=float)

    if not enabled or actual_blocks >= _BLOCKS:
        return CorrectionResult(fc, applied, {"enabled": False})

    try:
        if history_df is None or history_df.empty or "date" not in history_df.columns:
            return CorrectionResult(fc, applied, {"enabled": True, "applied": False, "reason": "no history"})

        df2 = history_df.copy()
        df2["date"] = df2["date"].astype(str)
        target_str = str(target_date)[:10]
        target_ts = pd.Timestamp(target_str)

        hist = df2[
            (df2["date"] < target_str) &
            (pd.to_datetime(df2["date"], errors="coerce") >= target_ts - pd.Timedelta(days=window_days))
        ].copy()

        if hist.empty or "total_drawal" not in hist.columns:
            return CorrectionResult(fc, applied, {"enabled": True, "applied": False, "reason": "no drawal data"})

        hist["_tb"] = hist["time_block"].astype(int)
        actual_by_block = (
            hist.groupby("_tb")["total_drawal"]
            .mean()
            .reindex(range(1, _BLOCKS + 1))
            .ffill()
            .bfill()
            .to_numpy()
        )

        seg_meta: Dict[str, Any] = {}
        for seg_name, (i0, i1) in _TOD_SEGMENTS.items():
            mean_actual = float(np.nanmean(actual_by_block[i0:i1]))
            mean_fc = float(np.nanmean(fc[i0:i1]))
            if mean_fc < 100 or mean_actual < 100:
                continue
            ratio = mean_actual / mean_fc
            # Only apply if ratio is meaningfully off (>2%) and within range
            if abs(ratio - 1.0) < 0.02 or ratio < 0.70 or ratio > 1.30:
                seg_meta[seg_name] = {"ratio": round(ratio, 4), "applied": False}
                continue
            # Confidence: number of days with data in this segment
            n_days = hist["date"].nunique()
            confidence = min(float(n_days) / 14.0, 1.0)
            if confidence < min_confidence * 0.5:  # relaxed threshold
                seg_meta[seg_name] = {"ratio": round(ratio, 4), "applied": False, "reason": "low_confidence"}
                continue
            blend = min(confidence * 0.25, 0.25)  # max 25% blend
            correction = fc[i0:i1] * (ratio - 1.0) * blend
            # Only apply to future blocks
            future_mask = np.arange(i0, i1) >= actual_blocks
            fc[i0:i1][future_mask] += correction[future_mask]
            fc[i0:i1] = np.maximum(fc[i0:i1], 0.0)
            applied[i0:i1] += correction * future_mask.astype(float)
            seg_meta[seg_name] = {
                "ratio": round(ratio, 4),
                "blend": round(blend, 3),
                "applied": True,
                "mean_correction_mw": round(float(np.mean(correction)), 1),
            }

    except Exception as exc:
        _log.warning("[accuracy] apply_block_type_calibration failed: %s", exc)
        return CorrectionResult(
            np.asarray(forecast, dtype=float),
            np.zeros(_BLOCKS, dtype=float),
            {"enabled": True, "applied": False, "reason": str(exc)},
        )

    return CorrectionResult(
        fc,
        applied,
        {"enabled": True, "applied": True, "segments": seg_meta},
    )


# ---------------------------------------------------------------------------
# 4. build_afternoon_ramp_adjustment  (T+1 weather-aware)
# ---------------------------------------------------------------------------

_AFTERNOON_BLOCKS = slice(48, 64)  # 12h–16h (0-based)

# Season → mean under-prediction magnitude in MW (recalibrated from April 2026 actuals)
# April 2026 block-level data shows +250–330 MW bias at 14:45–16:00; old values (180/260) undershot.
_AFTERNOON_STATIC_MW: Dict[str, float] = {
    "summer": 380.0,
    "spring": 300.0,
    "winter":  40.0,
    "fall":    60.0,
}


def build_afternoon_ramp_adjustment(
    baseline_96: np.ndarray,
    target_df: pd.DataFrame,
    season: str,
    region: str,
    enabled: bool = True,
) -> CorrectionResult:
    """
    Build an afternoon (12–16h) upward correction for hot-season under-prediction.

    The magnitude is weather-aware: scales with peak afternoon temperature above
    a threshold (35°C for summer, 32°C for spring).  Falls back to a static
    fraction of the baseline if no temperature data is available.
    """
    adj = np.zeros(_BLOCKS, dtype=float)
    meta: Dict[str, Any] = {"enabled": enabled, "applied": False}

    if not enabled:
        return CorrectionResult(np.asarray(baseline_96, dtype=float), adj, meta)

    try:
        base_corr = _AFTERNOON_STATIC_MW.get(str(season).lower(), 80.0)

        # Try to get afternoon temperature from target_df
        temp_boost = 1.0
        if target_df is not None and not target_df.empty and "temperature" in target_df.columns:
            aft_rows = target_df[target_df["time_block"].astype(int).between(49, 64)]
            if not aft_rows.empty:
                peak_temp = float(aft_rows["temperature"].max())
                thresh = 35.0 if season == "summer" else 32.0
                # Hysteresis band ±0.75°C to prevent binary flip from small weather forecast errors.
                # Below lower band: no boost. Inside band: linear ramp 0→full. Above upper band: full boost.
                thresh_lo, thresh_hi = thresh - 0.75, thresh + 0.75
                if peak_temp > thresh_hi:
                    temp_boost = 1.0 + min((peak_temp - thresh) * 0.08, 0.60)
                elif peak_temp > thresh_lo:
                    blend = (peak_temp - thresh_lo) / (thresh_hi - thresh_lo)  # 0→1 inside band
                    temp_boost = 1.0 + blend * min((peak_temp - thresh_lo) * 0.08, 0.30)

        correction = base_corr * temp_boost
        # Taper: ramp up from 12h, peak at 14h, ramp down to 16h
        taper = np.array([0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0,
                          0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        adj[_AFTERNOON_BLOCKS] = correction * taper

        meta.update({
            "applied": True,
            "season": season,
            "base_correction_mw": round(base_corr, 1),
            "temp_boost": round(temp_boost, 3),
            "mean_adj_mw": round(float(np.mean(adj[_AFTERNOON_BLOCKS])), 1),
        })

    except Exception as exc:
        _log.warning("[accuracy] build_afternoon_ramp_adjustment failed: %s", exc)
        meta["reason"] = str(exc)
        adj = np.zeros(_BLOCKS, dtype=float)

    return CorrectionResult(np.asarray(baseline_96, dtype=float), adj, meta)


# ---------------------------------------------------------------------------
# 5. build_t2_bias_correction
# ---------------------------------------------------------------------------

def build_t2_bias_correction(
    df: Optional[pd.DataFrame],
    fallback_night_mw: float = 250.0,
    min_samples: int = 7,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a 96-block T+2 correction vector.

    Strategy (in priority order):
    1. If df has enough same-weekday history (≥min_samples days with full blocks),
       compute rolling mean (actual − lag14) as a data-driven proxy correction,
       blended 40/60 with the static correction.
    2. Otherwise, return the static correction derived from Haryana April 2026 analysis.

    The vector is ADDITIVE: apply_t2_night_correction adds it to the T+2 forecast.
    """
    meta: Dict[str, Any] = {"enabled": True, "applied": False, "method": "static_fallback"}
    static_vec = _build_static_t2_vector(blend=0.80)

    if df is None or df.empty or "date" not in df.columns or "total_drawal" not in df.columns:
        meta.update({"method": "static_only", "applied": True, "reason": "no df provided"})
        return static_vec, meta

    try:
        df2 = df.copy()
        df2["date"] = df2["date"].astype(str)
        all_dates = sorted(df2["date"].unique())
        if len(all_dates) < min_samples + 14:
            meta["reason"] = f"only {len(all_dates)} dates in df"
            meta["applied"] = True
            return static_vec, meta

        # Use the last `window` days that have full-block actual data
        window = min(21, len(all_dates) - 14)
        recent_dates = all_dates[-(window + 14): -14]  # leave 14-day gap for lag

        errors: list[np.ndarray] = []
        for d in recent_dates:
            lag_date = (pd.Timestamp(d) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            actual_rows = df2[df2["date"] == d]
            lag_rows = df2[df2["date"] == lag_date]
            if actual_rows.empty or lag_rows.empty:
                continue
            if actual_rows["time_block"].nunique() < _BLOCKS // 2:
                continue
            actual_s = (
                actual_rows.groupby("time_block")["total_drawal"]
                .mean()
                .reindex(range(1, _BLOCKS + 1))
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )
            lag_s = (
                lag_rows.groupby("time_block")["total_drawal"]
                .mean()
                .reindex(range(1, _BLOCKS + 1))
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )
            # correction = actual − lag14  (positive → lag under-predicted → add to forecast)
            errors.append(actual_s - lag_s)

        if len(errors) < min_samples:
            meta["reason"] = f"only {len(errors)} valid lag pairs (need {min_samples})"
            meta.update({"applied": True, "method": "static_only"})
            return static_vec, meta

        data_vec = np.nanmean(errors, axis=0)
        data_vec = np.nan_to_num(data_vec, nan=0.0)
        data_vec = _smooth(data_vec, window=6)

        # Cap data-driven component
        cap = 500.0
        data_vec = np.clip(data_vec, -cap, cap)

        # Blend: 60% static + 40% data-driven
        combined = 0.60 * static_vec + 0.40 * data_vec
        combined = _smooth(combined, window=4)

        meta.update({
            "applied": True,
            "method": "blended_static_data",
            "n_pairs": len(errors),
            "data_mean_abs_mw": round(float(np.mean(np.abs(data_vec))), 1),
            "static_mean_abs_mw": round(float(np.mean(np.abs(static_vec))), 1),
            "combined_mean_abs_mw": round(float(np.mean(np.abs(combined))), 1),
        })
        return combined, meta

    except Exception as exc:
        _log.warning("[accuracy] build_t2_bias_correction failed: %s — using static fallback", exc)
        meta.update({"applied": True, "method": "static_fallback_after_error", "reason": str(exc)})
        return static_vec, meta


# ---------------------------------------------------------------------------
# 6. apply_t2_night_correction
# ---------------------------------------------------------------------------

def apply_t2_night_correction(
    forecast: np.ndarray,
    correction: np.ndarray,
    enabled: bool = True,
) -> CorrectionResult:
    """
    Add correction vector to T+2 forecast (all 96 blocks, no actual-block gate).

    Correction is capped at ±20% of forecast value per block or 400 MW,
    whichever is smaller.  Result is clipped to ≥0.
    """
    fc = np.asarray(forecast, dtype=float).copy()
    applied = np.zeros(_BLOCKS, dtype=float)

    if not enabled:
        return CorrectionResult(fc, applied, {"enabled": False})

    try:
        corr = np.asarray(correction, dtype=float)
        if corr.shape[0] != _BLOCKS:
            corr = np.zeros(_BLOCKS, dtype=float)

        cap = np.minimum(np.maximum(fc * 0.20, 80.0), 400.0)
        corr_clipped = np.clip(corr, -cap, cap)
        fc = np.maximum(fc + corr_clipped, 0.0)
        applied = corr_clipped

    except Exception as exc:
        _log.warning("[accuracy] apply_t2_night_correction failed: %s", exc)
        return CorrectionResult(
            np.asarray(forecast, dtype=float),
            np.zeros(_BLOCKS, dtype=float),
            {"enabled": True, "applied": False, "reason": str(exc)},
        )

    return CorrectionResult(
        fc,
        applied,
        {
            "enabled": True,
            "applied": True,
            "mean_abs_correction_mw": round(float(np.mean(np.abs(applied))), 1),
            "night_correction_mw": round(float(np.mean(applied[:32])), 1),
            "afternoon_correction_mw": round(float(np.mean(applied[48:64])), 1),
            "evening_correction_mw": round(float(np.mean(applied[64:80])), 1),
        },
    )
