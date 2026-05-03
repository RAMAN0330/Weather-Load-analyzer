"""Dedicated T+2 night-block bias correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96


@dataclass
class T2BiasCorrectionResult:
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


def _taper() -> np.ndarray:
    taper = np.ones(BLOCKS_PER_DAY, dtype=float)
    taper[32:48] = np.linspace(1.0, 0.30, 16)
    taper[48:] = 0.0
    return taper


def build_t2_bias_correction(
    t2_error_history: Optional[Any] = None,
    *,
    fallback_night_mw: float = 350.0,
    min_samples: int = 7,
    fallback_early_night_scale: float = 1.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build additive T+2 correction in MW.

    Error convention is forecast - actual. A negative median error therefore
    becomes a positive additive correction.
    """
    correction = np.zeros(BLOCKS_PER_DAY, dtype=float)
    samples = np.zeros(BLOCKS_PER_DAY, dtype=int)
    source = "fallback_haryana_night_bias"

    if isinstance(t2_error_history, pd.DataFrame) and not t2_error_history.empty:
        df = t2_error_history.copy()
        if "time_block" in df.columns:
            block_col = "time_block"
        elif "block" in df.columns:
            block_col = "block"
        else:
            block_col = None
        if block_col:
            if "error" not in df.columns:
                f_col = next((c for c in ("forecast_t2", "t2_forecast", "forecast") if c in df.columns), None)
                a_col = next((c for c in ("actual", "total_drawal") if c in df.columns), None)
                if f_col and a_col:
                    df["error"] = pd.to_numeric(df[f_col], errors="coerce") - pd.to_numeric(df[a_col], errors="coerce")
            if "error" in df.columns:
                df[block_col] = pd.to_numeric(df[block_col], errors="coerce")
                df["error"] = pd.to_numeric(df["error"], errors="coerce")
                df = df.dropna(subset=[block_col, "error"])
                grouped = df.groupby(block_col)["error"]
                med = grouped.median().reindex(range(1, BLOCKS_PER_DAY + 1))
                cnt = grouped.count().reindex(range(1, BLOCKS_PER_DAY + 1)).fillna(0).astype(int)
                valid = cnt.to_numpy(dtype=int) >= int(min_samples)
                correction[valid] = -med.fillna(0.0).to_numpy(dtype=float)[valid]
                samples = cnt.to_numpy(dtype=int)
                source = "historical_t2_error"
    elif isinstance(t2_error_history, Mapping):
        for block, errors in t2_error_history.items():
            idx = int(block) - 1
            if 0 <= idx < BLOCKS_PER_DAY:
                arr = np.asarray(errors if errors is not None else [], dtype=float)
                arr = arr[np.isfinite(arr)]
                samples[idx] = int(arr.size)
                if arr.size >= int(min_samples):
                    correction[idx] = -float(np.median(arr))
                    source = "historical_t2_error"

    if not np.any(np.abs(correction[:32]) > 1e-6) and fallback_night_mw:
        # blocks 1-16 (00:00-04:00): deeper trough, scale up by fallback_early_night_scale
        correction[:16] = float(fallback_night_mw) * float(fallback_early_night_scale)
        # blocks 17-32 (04:00-08:00): standard fallback
        correction[16:32] = float(fallback_night_mw)

    correction = correction * _taper()
    return correction, {
        "enabled": True,
        "applied": bool(np.any(np.abs(correction) > 1e-6)),
        "source": source,
        "fallback_night_mw": float(fallback_night_mw),
        "fallback_early_night_scale": float(fallback_early_night_scale),
        "min_samples": int(min_samples),
        "sample_count_min": int(samples[:32].min()) if samples.size else 0,
        "night_mean_correction_mw": round(float(np.mean(correction[:32])), 3),
        "max_abs_correction_mw": round(float(np.max(np.abs(correction))), 3),
        "correction_mw": correction.tolist(),
    }


def apply_t2_night_correction(forecast_96: Any, correction_96: Any, *, enabled: bool = True) -> T2BiasCorrectionResult:
    forecast = _as_96(forecast_96)
    correction = _as_96(correction_96)
    applied = correction if enabled else np.zeros(BLOCKS_PER_DAY, dtype=float)
    adjusted = np.maximum(forecast + applied, 0.0)
    return T2BiasCorrectionResult(
        forecast=adjusted,
        applied_mw=applied,
        metadata={
            "enabled": bool(enabled),
            "applied": bool(np.any(np.abs(applied) > 1e-6)),
            "night_mean_applied_mw": round(float(np.mean(applied[:32])), 3),
            "max_abs_applied_mw": round(float(np.max(np.abs(applied))), 3),
            "applied_mw": applied.tolist(),
        },
    )
