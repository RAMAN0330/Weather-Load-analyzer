"""Vectorised thermodynamic / thermal-comfort transforms.

All functions accept scalars or numpy arrays (any shape) and propagate NaN:
missing inputs yield missing outputs, never zeros.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Stull (2011) empirical validity envelope.
STULL_RH_RANGE: tuple[float, float] = (5.0, 99.0)
STULL_T_RANGE: tuple[float, float] = (-20.0, 50.0)

# Magnus coefficients (Alduchov & Eskridge 1996).
MAGNUS_A: float = 17.625
MAGNUS_B: float = 243.04

BLOCK_HOURS: float = 15.0 / 60.0


def wet_bulb_stull(t_c: ArrayLike, rh_pct: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Wet-bulb temperature (°C) after Stull (2011), J. Appl. Meteor. Climatol.

    Parameters
    ----------
    t_c : air temperature in °C.
    rh_pct : relative humidity in percent (0–100).

    Returns
    -------
    (tw, valid) : wet-bulb temperature (NaN outside the validity envelope
        RH 5–99 %, T −20..50 °C, or where inputs are missing) and the boolean
        validity mask.
    """
    t = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)
    valid = (
        np.isfinite(t)
        & np.isfinite(rh)
        & (rh >= STULL_RH_RANGE[0])
        & (rh <= STULL_RH_RANGE[1])
        & (t >= STULL_T_RANGE[0])
        & (t <= STULL_T_RANGE[1])
    )
    with np.errstate(invalid="ignore"):
        rh_safe = np.where(valid, rh, 50.0)
        t_safe = np.where(valid, t, 20.0)
        tw = (
            t_safe * np.arctan(0.151977 * np.sqrt(rh_safe + 8.313659))
            + np.arctan(t_safe + rh_safe)
            - np.arctan(rh_safe - 1.676331)
            + 0.00391838 * np.power(rh_safe, 1.5) * np.arctan(0.023101 * rh_safe)
            - 4.686035
        )
    return np.where(valid, tw, np.nan), valid


def dew_point_magnus(t_c: ArrayLike, rh_pct: ArrayLike) -> NDArray[np.float64]:
    """Dew point (°C) via the Magnus formula (a=17.625, b=243.04 °C).

    Returns NaN where inputs are missing or RH ≤ 0.
    """
    t = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh_pct, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(rh) & (rh > 0)
    rh_safe = np.where(ok, np.minimum(rh, 100.0), 50.0)
    t_safe = np.where(ok, t, 20.0)
    gamma = np.log(rh_safe / 100.0) + MAGNUS_A * t_safe / (MAGNUS_B + t_safe)
    td = MAGNUS_B * gamma / (MAGNUS_A - gamma)
    return np.where(ok, td, np.nan)


def cooling_degree(t_c: ArrayLike, base_c: float = 24.0) -> NDArray[np.float64]:
    """Instantaneous cooling degree ``max(T − base, 0)`` (°C); NaN stays NaN."""
    t = np.asarray(t_c, dtype=np.float64)
    return np.where(np.isfinite(t), np.maximum(t - base_c, 0.0), np.nan)


def rolling_degree_hours(
    degree: ArrayLike,
    window_blocks: int,
    *,
    min_fraction: float = 0.75,
    block_hours: float = BLOCK_HOURS,
) -> NDArray[np.float64]:
    """Trailing rolling cooling-degree-hours along axis 0 (time).

    ``degree`` is a (time,) or (time × district) array on a *complete* 15-min
    grid. The window ending at t covers ``window_blocks`` blocks [t−k+1, t].
    Missing values are excluded; if at least ``min_fraction`` of the window is
    present the sum is rescaled to the full window, otherwise NaN.
    Implemented with cumulative sums (O(n), fully vectorised).
    """
    x = np.asarray(degree, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    k = int(window_blocks)
    n = x.shape[0]
    out = np.full_like(x, np.nan)
    if k <= 0 or n < k:
        return out[:, 0] if squeeze else out
    present = np.isfinite(x)
    zeros = np.zeros((1, x.shape[1]))
    csum = np.concatenate([zeros, np.cumsum(np.where(present, x, 0.0), axis=0)])
    ccnt = np.concatenate([zeros, np.cumsum(present, axis=0, dtype=np.float64)])
    win_sum = csum[k:] - csum[:-k]
    win_cnt = ccnt[k:] - ccnt[:-k]
    enough = win_cnt >= min_fraction * k
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = win_sum * (k / win_cnt) * block_hours
    out[k - 1 :] = np.where(enough, scaled, np.nan)
    return out[:, 0] if squeeze else out


__all__ = [
    "wet_bulb_stull",
    "dew_point_magnus",
    "cooling_degree",
    "rolling_degree_hours",
    "BLOCK_HOURS",
]
