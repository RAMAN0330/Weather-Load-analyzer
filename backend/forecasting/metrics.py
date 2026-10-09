"""Vectorised forecast accuracy metrics.

Inputs are (n_days × 96) arrays (or anything reshapeable to that). Cells where
the actual or the point forecast is NaN are excluded from every metric.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import BLOCKS_PER_DAY

DEFAULT_PEAK_BLOCKS: tuple[int, int] = (69, 92)  # 17:00–23:00 IST, inclusive


def _as_days(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return a.reshape(-1, BLOCKS_PER_DAY)


def _nan(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def wape(actual: ArrayLike, pred: ArrayLike) -> float:
    """Weighted absolute percentage error (%) = Σ|y−ŷ| / Σ|y| · 100."""
    y, p = np.asarray(actual, float).ravel(), np.asarray(pred, float).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    den = np.abs(y[m]).sum()
    return float(np.abs(y[m] - p[m]).sum() / den * 100.0) if den > 0 else float("nan")


def pinball_loss(actual: ArrayLike, q_pred: ArrayLike, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> float:
    """Mean pinball loss averaged across quantiles. ``q_pred``: (..., n_q)."""
    y = np.asarray(actual, float).reshape(-1, 1)
    q = np.asarray(q_pred, float).reshape(-1, len(quantiles))
    a = np.asarray(quantiles, float)[None, :]
    m = np.isfinite(y[:, 0]) & np.isfinite(q).all(axis=1)
    diff = y[m] - q[m]
    loss = np.maximum(a * diff, (a - 1.0) * diff)
    return float(loss.mean()) if loss.size else float("nan")


def compute_metrics(
    actual: ArrayLike,
    p50: ArrayLike,
    p10: ArrayLike | None = None,
    p90: ArrayLike | None = None,
    peak_blocks: tuple[int, int] = DEFAULT_PEAK_BLOCKS,
) -> dict[str, float | None]:
    """Aggregate metrics in the contract's shape (NaN → None)."""
    y, p = _as_days(actual), _as_days(p50)
    m = np.isfinite(y) & np.isfinite(p)
    err = np.where(m, p - y, np.nan)
    abs_err = np.abs(err[m])
    w = wape(y, p)
    lo, hi = peak_blocks
    pk = slice(lo - 1, hi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        day_ok = m.any(axis=1)
        peak_err = np.abs(np.nanmax(np.where(m, p, np.nan), axis=1) - np.nanmax(np.where(m, y, np.nan), axis=1))
    out: dict[str, float | None] = {
        "wape": _nan(w),
        "accuracy_wape": _nan(100.0 - w),
        "mae": _nan(abs_err.mean() if abs_err.size else np.nan),
        "rmse": _nan(np.sqrt((abs_err**2).mean()) if abs_err.size else np.nan),
        "bias_mw": _nan(err[m].mean() if abs_err.size else np.nan),
        "peak_wape": _nan(wape(y[:, pk], p[:, pk])),
        "daily_peak_error_mw": _nan(peak_err[day_ok].mean() if day_ok.any() else np.nan),
        "p90_abs_error_mw": _nan(np.percentile(abs_err, 90) if abs_err.size else np.nan),
        "p95_abs_error_mw": _nan(np.percentile(abs_err, 95) if abs_err.size else np.nan),
        "pinball_loss": None,
        "interval_coverage_pct": None,
    }
    if p10 is not None and p90 is not None:
        lo_q, hi_q = _as_days(p10), _as_days(p90)
        out["pinball_loss"] = _nan(pinball_loss(y, np.stack([lo_q, p, hi_q], axis=-1)))
        mi = m & np.isfinite(lo_q) & np.isfinite(hi_q)
        inside = (y[mi] >= lo_q[mi]) & (y[mi] <= hi_q[mi])
        out["interval_coverage_pct"] = _nan(inside.mean() * 100.0 if inside.size else np.nan)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def daily_metrics(actual: ArrayLike, p50: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Per-day WAPE (%), MAE and |peak error| (MW); NaN for days without actuals."""
    y, p = _as_days(actual), _as_days(p50)
    m = np.isfinite(y) & np.isfinite(p)
    ae = np.where(m, np.abs(p - y), 0.0)
    cnt = m.sum(axis=1)
    den = np.where(m, np.abs(y), 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "wape": np.where(den > 0, ae.sum(axis=1) / den * 100.0, np.nan),
            "mae": np.where(cnt > 0, ae.sum(axis=1) / cnt, np.nan),
            "peak_error_mw": np.abs(np.nanmax(np.where(m, p, np.nan), axis=1) - np.nanmax(np.where(m, y, np.nan), axis=1)),
        }


__all__ = ["DEFAULT_PEAK_BLOCKS", "wape", "pinball_loss", "compute_metrics", "daily_metrics"]
