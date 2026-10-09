"""Origin-safe baseline forecasts on a (date × block) load matrix.

Both baselines are computed for many target dates at once by indexing rows of
the load matrix; for target row ``i`` and horizon ``h`` only rows ≤ ``i − h``
(the origin) are touched.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def take_rows(mat: NDArray[np.float64], rows: NDArray[np.int64]) -> NDArray[np.float64]:
    """``mat[rows]`` with out-of-range rows returned as NaN."""
    rows = np.asarray(rows, dtype=np.int64)
    ok = (rows >= 0) & (rows < mat.shape[0])
    out = np.full((rows.shape[0], mat.shape[1]), np.nan)
    out[ok] = mat[rows[ok]]
    return out


def seasonal_naive(load_mat: NDArray[np.float64], target_rows: NDArray[np.int64], season_days: int = 7) -> NDArray[np.float64]:
    """Same block, ``season_days`` earlier (D−7). Valid for horizons ≤ season_days."""
    return take_rows(load_mat, np.asarray(target_rows) - season_days)


def recent_day(
    load_mat: NDArray[np.float64],
    target_rows: NDArray[np.int64],
    horizon: int,
    off_day: NDArray[np.bool_],
    k: int = 3,
    lookback_days: int = 14,
) -> NDArray[np.float64]:
    """Mean of the same block over the ``k`` most recent same-day-type days.

    Candidates are the ``lookback_days`` days ending at the origin
    (``target − horizon``). Day type is ``off_day`` (weekend/holiday) vs
    working day. If fewer than ``k`` same-type days with data exist, falls back
    to the plain last ``k`` days ending at the origin.

    Parameters
    ----------
    load_mat : (n_days × 96) load matrix (NaN = missing).
    target_rows : row indices of target dates.
    horizon : days between origin and target.
    off_day : (n_days,) boolean day-type per row (may extend beyond load rows).
    """
    tr = np.asarray(target_rows, dtype=np.int64)
    origin = tr - int(horizon)
    lags = np.arange(lookback_days, dtype=np.int64)
    cand = origin[:, None] - lags[None, :]  # (n, L), most recent first
    n_rows = load_mat.shape[0]
    if n_rows == 0 or tr.size == 0:
        return np.full((tr.size, load_mat.shape[1]), np.nan)
    in_range = (cand >= 0) & (cand < n_rows)
    safe = np.clip(cand, 0, n_rows - 1)
    vals = np.where(in_range[..., None], load_mat[safe], np.nan)  # (n, L, 96)
    has_data = np.isfinite(vals).any(axis=2)

    off = np.asarray(off_day, dtype=bool)
    tgt_type = off[np.clip(tr, 0, len(off) - 1)]
    cand_type = off[np.clip(cand, 0, len(off) - 1)]
    match = has_data & in_range & (cand_type == tgt_type[:, None])
    rank = np.cumsum(match, axis=1)
    sel = match & (rank <= k)
    enough = sel.sum(axis=1) >= k

    plain = np.zeros_like(sel)
    plain[:, :k] = True
    plain &= in_range
    chosen = np.where(enough[:, None], sel, plain)

    picked = np.where(chosen[..., None], vals, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(picked, axis=1)


__all__ = ["take_rows", "seasonal_naive", "recent_day"]
