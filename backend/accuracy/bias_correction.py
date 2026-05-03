"""Rolling 96-block bias correction for day-ahead forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96


@dataclass
class BiasCorrectionResult:
    forecast: np.ndarray
    applied_mw: np.ndarray
    metadata: dict[str, Any]


def _as_96(values: Any, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    out = np.full(BLOCKS_PER_DAY, float(fill), dtype=float)
    if arr.size:
        n = min(BLOCKS_PER_DAY, arr.size)
        out[:n] = arr[:n]
        if n < BLOCKS_PER_DAY:
            out[n:] = out[n - 1]
    return out


def _smooth_96(values: np.ndarray, window: int = 4) -> np.ndarray:
    vec = _as_96(values)
    window = max(1, int(window))
    if window <= 1:
        return vec
    kernel = np.ones(window, dtype=float) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(vec, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:BLOCKS_PER_DAY]


def _day_type_mask(dates: pd.Series, day_type: str) -> pd.Series:
    dt = pd.to_datetime(dates, errors="coerce")
    day_type = str(day_type or "all").lower().strip()
    if day_type == "weekday":
        return dt.dt.dayofweek < 5
    if day_type == "weekend":
        return dt.dt.dayofweek >= 5
    return pd.Series(True, index=dates.index)


def _forecast_col(df: pd.DataFrame, horizon: str) -> Optional[str]:
    horizon = str(horizon or "t1").lower().strip()
    candidates = (
        [f"forecast_{horizon}", f"{horizon}_forecast", "forecast", "predicted", "prediction"]
        if horizon
        else ["forecast", "predicted", "prediction"]
    )
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _actual_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("actual", "total_drawal", "load", "y"):
        if col in df.columns:
            return col
    return None


def _proxy_error_frame(df: pd.DataFrame, target_date: Optional[str], window_days: int) -> pd.DataFrame:
    """Build forecast-actual proxy errors using a causal recent-day block baseline."""
    work = df.copy()
    work["date"] = work["date"].astype(str).str[:10]
    if target_date:
        work = work[work["date"] < str(target_date)[:10]]
    work["time_block"] = pd.to_numeric(work["time_block"], errors="coerce")
    work["total_drawal"] = pd.to_numeric(work["total_drawal"], errors="coerce")
    work = work.dropna(subset=["date", "time_block", "total_drawal"])
    work = work[work["time_block"].between(1, BLOCKS_PER_DAY)]
    dates = sorted(work["date"].dropna().unique().tolist())
    keep = set(dates[-max(3, int(window_days) + 7):])
    work = work[work["date"].isin(keep)].sort_values(["date", "time_block"])
    if work.empty:
        return pd.DataFrame(columns=["date", "time_block", "error"])

    piv = (
        work.pivot_table(index="date", columns="time_block", values="total_drawal", aggfunc="mean")
        .reindex(columns=range(1, BLOCKS_PER_DAY + 1))
        .sort_index()
    )

    # Night anomaly guard: if a day's night blocks (1-32) deviated > 15% from
    # the per-block rolling mean, exclude that day from the error table.
    # Without this, a grid outage or demand collapse on one night poisons the
    # block bias table with a large apparent overforecast, causing the NEXT
    # normal night to be systematically underforecast.
    _NIGHT_ANOMALY_THRESH = 0.15
    _night_cols = [c for c in range(1, 33) if c in piv.columns]
    _night_rolling_mean = piv[_night_cols].rolling(7, min_periods=2).mean().shift(1)
    _night_dev = (piv[_night_cols] - _night_rolling_mean).abs() / _night_rolling_mean.replace(0, np.nan)
    _night_anomaly_dates = set(
        _night_dev[_night_dev.mean(axis=1) > _NIGHT_ANOMALY_THRESH].index.astype(str)
    )

    rows = []
    for i, date in enumerate(piv.index):
        if str(date) in _night_anomaly_dates:
            continue  # skip days with anomalous night load — don't pollute bias table
        hist = piv.iloc[max(0, i - int(window_days)):i]
        if len(hist) < 3:
            continue
        proxy_forecast = hist.mean(axis=0)
        actual = piv.loc[date]
        err = proxy_forecast - actual
        for block, value in err.items():
            if pd.notna(value):
                rows.append({"date": str(date), "time_block": int(block), "error": float(value)})
    return pd.DataFrame(rows)


def compute_bias_table(
    historical_df: pd.DataFrame,
    *,
    target_date: Optional[str] = None,
    window_days: int = 30,
    day_type: str = "all",
    horizon: str = "t1",
    smooth_window: int = 4,
    seasonal_window_months: tuple = (4, 5, 6),
    seasonal_window_days: int = 14,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return median forecast-actual bias per block.

    If forecast-error columns are unavailable, this uses a causal rolling block
    baseline as a proxy forecast, so the correction remains non-leaky.
    """
    if historical_df is None or historical_df.empty:
        return np.zeros(BLOCKS_PER_DAY, dtype=float), {
            "enabled": True,
            "applied": False,
            "reason": "empty_history",
            "source": "none",
        }

    work = historical_df.copy()
    if "date" not in work.columns or "time_block" not in work.columns:
        return np.zeros(BLOCKS_PER_DAY, dtype=float), {
            "enabled": True,
            "applied": False,
            "reason": "missing_date_or_block",
            "source": "none",
        }

    work["date"] = work["date"].astype(str).str[:10]
    if target_date:
        work = work[work["date"] < str(target_date)[:10]]

    # Shorten bias window for transitional months to avoid cross-season contamination
    if target_date:
        try:
            _ref_month = pd.to_datetime(str(target_date)[:10], errors="coerce").month
            if pd.notna(_ref_month) and int(_ref_month) in seasonal_window_months:
                window_days = seasonal_window_days
        except Exception:
            pass

    f_col = _forecast_col(work, horizon)
    a_col = _actual_col(work)
    if f_col and a_col:
        err = work[["date", "time_block", f_col, a_col]].copy()
        err["time_block"] = pd.to_numeric(err["time_block"], errors="coerce")
        err[f_col] = pd.to_numeric(err[f_col], errors="coerce")
        err[a_col] = pd.to_numeric(err[a_col], errors="coerce")
        err = err.dropna(subset=["date", "time_block", f_col, a_col])
        dates = sorted(err["date"].dropna().unique().tolist())
        err = err[err["date"].isin(set(dates[-int(window_days):]))]
        err["error"] = err[f_col] - err[a_col]
        source = f"saved_{horizon}_forecast_error"
    else:
        err = _proxy_error_frame(work, target_date, window_days)
        source = "causal_block_baseline_proxy"

    if err.empty:
        return np.zeros(BLOCKS_PER_DAY, dtype=float), {
            "enabled": True,
            "applied": False,
            "reason": "insufficient_error_history",
            "source": source,
        }

    mask = _day_type_mask(err["date"], day_type)
    filtered = err[mask]
    if filtered["date"].nunique() >= 3:
        err = filtered

    raw = (
        err.groupby("time_block")["error"]
        .median()
        .reindex(range(1, BLOCKS_PER_DAY + 1))
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    smoothed = _smooth_96(raw, smooth_window)
    return smoothed, {
        "enabled": True,
        "applied": bool(np.any(np.abs(smoothed) > 1e-6)),
        "source": source,
        "horizon": horizon,
        "day_type": day_type,
        "window_days": int(window_days),
        "history_days": int(err["date"].nunique()),
        "mean_bias_mw": round(float(np.mean(smoothed)), 3),
        "max_abs_bias_mw": round(float(np.max(np.abs(smoothed))), 3),
    }


def apply_bias_correction(
    forecast_96: Any,
    bias_table: Any,
    *,
    actual_blocks: int = 0,
    enabled: bool = True,
) -> BiasCorrectionResult:
    forecast = _as_96(forecast_96)
    bias = _as_96(bias_table)
    start = int(np.clip(int(actual_blocks), 0, BLOCKS_PER_DAY))
    corrected = forecast.copy()
    applied = np.zeros(BLOCKS_PER_DAY, dtype=float)
    if enabled and start < BLOCKS_PER_DAY:
        applied[start:] = -bias[start:]
        corrected[start:] = np.maximum(corrected[start:] + applied[start:], 0.0)
    return BiasCorrectionResult(
        forecast=corrected,
        applied_mw=applied,
        metadata={
            "enabled": bool(enabled),
            "applied": bool(np.any(np.abs(applied[start:]) > 1e-6)) if start < BLOCKS_PER_DAY else False,
            "actual_blocks": start,
            "mean_applied_mw": round(float(np.mean(applied[start:])), 3) if start < BLOCKS_PER_DAY else 0.0,
            "max_abs_applied_mw": round(float(np.max(np.abs(applied[start:]))), 3) if start < BLOCKS_PER_DAY else 0.0,
            "applied_mw": applied.tolist(),
        },
    )
