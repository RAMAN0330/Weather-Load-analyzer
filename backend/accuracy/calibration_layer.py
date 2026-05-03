"""Lightweight post-pipeline block-type calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96


@dataclass
class CalibrationResult:
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


def _block_type(block: int, weekend: bool) -> str:
    hour = ((int(block) - 1) * 15) // 60
    if hour < 6:
        part = "night"
    elif hour < 12:
        part = "morning_peak"
    elif hour < 18:
        part = "afternoon"
    else:
        part = "evening"
    return f"{part}_{'weekend' if weekend else 'weekday'}"


def _learn_bias_by_type(history: pd.DataFrame, target_date: Optional[str], window_days: int) -> dict[str, float]:
    if history is None or history.empty or "date" not in history.columns or "time_block" not in history.columns:
        return {}
    df = history.copy()
    df["date"] = df["date"].astype(str).str[:10]
    if target_date:
        df = df[df["date"] < str(target_date)[:10]]
    f_col = next((c for c in ("forecast", "forecast_t1", "predicted") if c in df.columns), None)
    a_col = next((c for c in ("actual", "total_drawal") if c in df.columns), None)
    if not f_col or not a_col:
        return {}
    dates = sorted(df["date"].dropna().unique().tolist())
    df = df[df["date"].isin(set(dates[-int(window_days):]))]
    if df.empty:
        return {}
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["_weekend"] = dt.dt.dayofweek >= 5
    df["_block_type"] = [
        _block_type(block, weekend)
        for block, weekend in zip(pd.to_numeric(df["time_block"], errors="coerce").fillna(1), df["_weekend"])
    ]
    df["_err"] = pd.to_numeric(df[f_col], errors="coerce") - pd.to_numeric(df[a_col], errors="coerce")
    return df.groupby("_block_type")["_err"].median().dropna().to_dict()


def apply_block_type_calibration(
    forecast_96: Any,
    *,
    target_date: str,
    forecast_confidence: Optional[Any] = None,
    history_df: Optional[pd.DataFrame] = None,
    calibration_table: Optional[Mapping[str, float]] = None,
    actual_blocks: int = 0,
    window_days: int = 21,
    min_confidence: float = 0.80,
    enabled: bool = True,
) -> CalibrationResult:
    forecast = _as_96(forecast_96)
    confidence = _as_96(forecast_confidence if forecast_confidence is not None else np.ones(BLOCKS_PER_DAY), 1.0)
    start = int(np.clip(actual_blocks, 0, BLOCKS_PER_DAY))
    try:
        weekend = pd.Timestamp(str(target_date)[:10]).dayofweek >= 5
    except Exception:
        weekend = False

    table = dict(calibration_table or {})
    if not table and history_df is not None:
        table = _learn_bias_by_type(history_df, target_date, window_days)

    applied = np.zeros(BLOCKS_PER_DAY, dtype=float)
    if enabled and table:
        for idx in range(start, BLOCKS_PER_DAY):
            if confidence[idx] < float(min_confidence):
                continue
            key = _block_type(idx + 1, weekend)
            if key in table:
                applied[idx] = -float(table[key])

    calibrated = np.maximum(forecast + applied, 0.0)
    calibrated[:start] = forecast[:start]
    return CalibrationResult(
        forecast=calibrated,
        applied_mw=applied,
        metadata={
            "enabled": bool(enabled),
            "applied": bool(np.any(np.abs(applied[start:]) > 1e-6)) if start < BLOCKS_PER_DAY else False,
            "source": "rolling_block_type_bias" if table else "none",
            "window_days": int(window_days),
            "min_confidence": float(min_confidence),
            "actual_blocks": start,
            "mean_applied_mw": round(float(np.mean(applied[start:])), 3) if start < BLOCKS_PER_DAY else 0.0,
            "max_abs_applied_mw": round(float(np.max(np.abs(applied[start:]))), 3) if start < BLOCKS_PER_DAY else 0.0,
            "applied_mw": applied.tolist(),
        },
    )
