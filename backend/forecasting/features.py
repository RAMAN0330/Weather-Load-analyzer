"""Origin-safe supervised design matrix.

For a target date ``D`` and horizon ``h`` the forecast origin is
``O = D − h days``. Every *load* feature for ``D`` is read from rows of the
(date × block) load matrix with date ≤ ``O``; this is enforced structurally by
row-shifting the matrix (no per-date loops), so the full training design for
all dates is built in a handful of numpy operations.

Weather features are taken for the target date ``D`` itself (weather valid at
the target time). Caveat: historical weather is the stored (possibly observed)
value, not the forecast that was available at the origin.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import BLOCKS_PER_DAY
from .baselines import recent_day, take_rows

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "base": (
        "lag_h_load",
        "lag_7d_load",
        "lag_14d_load",
        "roll7_mean_load",
        "roll7_std_load",
        "origin_daily_mean_load",
        "origin_last_block_load",
        "recent_day_baseline",
        "block_idx",
        "block_sin",
        "block_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
        "days_to_holiday",
        "days_since_holiday",
    ),
    "wx_mean": ("temp_mean", "hum_mean"),
    "wx_weighted": (
        "temp_wmean",
        "hum_wmean",
        "wb_wmean",
        "dew_wmean",
        "precip_wmean",
        "cloud_wmean",
        "wind_wmean",
        "rad_wmean",
        "temp_wmean_delta_origin",
    ),
    "wx_extremes": ("temp_p90", "temp_max", "temp_std", "hot_share", "wb_p90", "wb_max", "wb_hot_share"),
    "wx_cdh": ("cdh6_wmean", "cdh24_wmean", "cdh24_p90", "cdh24_delta_origin"),
    "wx_night": ("night_min_temp", "hot_day_streak", "prev_day_temp_wmax"),
}
ALL_GROUPS: tuple[str, ...] = tuple(FEATURE_GROUPS)
HOLIDAY_CAP_DAYS: int = 7
ID_COLUMNS: tuple[str, ...] = ("date", "block", "origin_date", "load")


@dataclass(frozen=True)
class LoadMatrix:
    """Dense (n_days × 96) load matrix on a complete daily grid (NaN = missing)."""

    days: pd.DatetimeIndex
    values: NDArray[np.float64]

    def row(self, day: pd.Timestamp) -> int:
        return int(self.days.get_loc(pd.Timestamp(day).normalize()))


def normalise_load_frame(load_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to columns (date: datetime64 normalised, block: int, load: float)."""
    if load_df is None or load_df.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "block": pd.Series(dtype=np.int64), "load": pd.Series(dtype=np.float64)})
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(load_df["date"]).dt.normalize(),
            "block": pd.to_numeric(load_df["block"], errors="coerce"),
            "load": pd.to_numeric(load_df["load"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "block"])
    out["block"] = out["block"].astype(np.int64)
    return out


def build_load_matrix(
    load_df: pd.DataFrame,
    days: pd.DatetimeIndex,
    until: pd.Timestamp | None = None,
) -> LoadMatrix:
    """Scatter a long load frame onto ``days`` × 96 (duplicates averaged).

    Rows dated after ``until`` are dropped before scattering (hard information
    cut-off at the forecast origin).
    """
    lf = normalise_load_frame(load_df)
    if until is not None:
        lf = lf[lf["date"] <= pd.Timestamp(until).normalize()]
    vals = lf["load"].to_numpy(dtype=np.float64)
    d_idx = days.get_indexer(lf["date"])
    blk = lf["block"].to_numpy(dtype=np.int64)
    ok = (d_idx >= 0) & (blk >= 1) & (blk <= BLOCKS_PER_DAY) & np.isfinite(vals)
    flat = d_idx[ok] * BLOCKS_PER_DAY + blk[ok] - 1
    size = len(days) * BLOCKS_PER_DAY
    sums = np.bincount(flat, weights=vals[ok], minlength=size)
    cnts = np.bincount(flat, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        dense = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return LoadMatrix(days=days, values=dense.reshape(len(days), BLOCKS_PER_DAY))


@lru_cache(maxsize=32)
def _holiday_ordinals(years: tuple[int, ...]) -> NDArray[np.int64]:
    """Sorted ordinal days of Indian holidays (Haryana subdivision if available)."""
    try:
        import holidays as _hol

        try:
            cal = _hol.India(years=list(years), subdiv="HR")
        except (NotImplementedError, KeyError, ValueError):
            cal = _hol.India(years=list(years))
        days = sorted(cal.keys())
    except Exception:  # pragma: no cover - package missing
        days = [pd.Timestamp(y, m, d).date() for y in years for (m, d) in ((1, 26), (8, 15), (10, 2))]
    return np.array([pd.Timestamp(d).toordinal() for d in days], dtype=np.int64)


def calendar_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-day calendar attributes (vectorised)."""
    if len(days) == 0:
        return pd.DataFrame(index=days)
    years = tuple(range(int(days.year.min()) - 1, int(days.year.max()) + 2))
    hol = _holiday_ordinals(years)
    # Ordinal = days since 0001-01-01 (day 1); computed without a Python loop.
    ords = (days.normalize().to_numpy().astype("datetime64[D]").astype(np.int64) + 719163).astype(np.int64)
    cap = HOLIDAY_CAP_DAYS
    if len(hol) == 0:
        is_hol = np.zeros(len(ords), dtype=bool)
        nxt = prv = np.full(len(ords), cap)
    else:
        pos = np.searchsorted(hol, ords, side="left")  # first holiday ≥ day
        nxt = np.where(pos < len(hol), hol[np.minimum(pos, len(hol) - 1)] - ords, cap)
        is_hol = nxt == 0
        pos_r = np.searchsorted(hol, ords, side="right") - 1  # last holiday ≤ day
        prv = np.where(pos_r >= 0, ords - hol[np.maximum(pos_r, 0)], cap)
    dow = days.dayofweek.to_numpy()
    is_weekend = dow >= 5
    return pd.DataFrame(
        {
            "dow": dow,
            "month": days.month.to_numpy(),
            "is_weekend": is_weekend.astype(np.float64),
            "is_holiday": is_hol.astype(np.float64),
            "days_to_holiday": np.minimum(nxt, HOLIDAY_CAP_DAYS).astype(np.float64),
            "days_since_holiday": np.minimum(prv, HOLIDAY_CAP_DAYS).astype(np.float64),
            "off_day": is_weekend | is_hol,
        },
        index=days,
    )


def feature_columns(groups: Iterable[str] = ALL_GROUPS) -> list[str]:
    """Ordered feature names for the selected groups ("base" is always included)."""
    sel = ["base", *[g for g in groups if g != "base"]]
    unknown = [g for g in sel if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")
    cols: list[str] = []
    for g in sel:
        cols.extend(c for c in FEATURE_GROUPS[g] if c not in cols)
    return cols


def _wx_block(wx: pd.DataFrame | None, days: pd.DatetimeIndex, col: str) -> NDArray[np.float64]:
    """(len(days) × 96) array of a weather column (NaN where unavailable)."""
    if wx is None or wx.empty or col not in wx.columns:
        return np.full((len(days), BLOCKS_PER_DAY), np.nan)
    idx = pd.MultiIndex.from_product([days, np.arange(1, BLOCKS_PER_DAY + 1)], names=["date", "block"])
    return wx[col].reindex(idx).to_numpy(dtype=np.float64).reshape(len(days), BLOCKS_PER_DAY)


def build_design(
    lm: LoadMatrix,
    wx: pd.DataFrame | None,
    horizon: int,
    target_days: Sequence | pd.DatetimeIndex | None = None,
    groups: Iterable[str] = ALL_GROUPS,
) -> pd.DataFrame:
    """Supervised design matrix for ``horizon`` (one row per target date × block).

    Columns: ``date``, ``block``, ``origin_date``, ``load`` (target; NaN when
    unknown) followed by :func:`feature_columns`. Target days must lie on the
    load matrix grid (extend the grid with NaN rows to forecast the future).
    """
    h = int(horizon)
    if not 1 <= h <= 7:
        raise ValueError("horizon must be between 1 and 7 days")
    days = lm.days if target_days is None else pd.DatetimeIndex(pd.to_datetime(list(target_days))).normalize()
    rows = lm.days.get_indexer(days)
    if (rows < 0).any():
        raise ValueError("target days must lie on the load-matrix grid")
    L = lm.values
    n = len(rows)
    origin_rows = rows - h

    lag_h = take_rows(L, origin_rows)
    stack7 = np.stack([take_rows(L, origin_rows - j) for j in range(7)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        roll_mean = np.nanmean(stack7, axis=1)
        roll_std = np.nanstd(stack7, axis=1)
        origin_mean = np.nanmean(lag_h, axis=1)

    cal_all = calendar_frame(lm.days)
    cal = cal_all.iloc[rows]
    off = cal_all["off_day"].to_numpy()

    def per_day(v: NDArray) -> NDArray[np.float64]:
        return np.repeat(np.asarray(v, dtype=np.float64)[:, None], BLOCKS_PER_DAY, axis=1)

    blocks = np.arange(1, BLOCKS_PER_DAY + 1, dtype=np.float64)
    ang_b = 2 * np.pi * (blocks - 1) / BLOCKS_PER_DAY
    ang_d = 2 * np.pi * cal["dow"].to_numpy() / 7.0
    ang_m = 2 * np.pi * (cal["month"].to_numpy() - 1) / 12.0
    tile = lambda v: np.broadcast_to(v, (n, BLOCKS_PER_DAY))  # noqa: E731

    f: dict[str, NDArray[np.float64]] = {
        "lag_h_load": lag_h,
        "lag_7d_load": take_rows(L, rows - 7),
        "lag_14d_load": take_rows(L, rows - 14),
        "roll7_mean_load": roll_mean,
        "roll7_std_load": roll_std,
        "origin_daily_mean_load": per_day(origin_mean),
        "origin_last_block_load": per_day(lag_h[:, -1]),
        "recent_day_baseline": recent_day(L, rows, h, off),
        "block_idx": tile(blocks),
        "block_sin": tile(np.sin(ang_b)),
        "block_cos": tile(np.cos(ang_b)),
        "dow_sin": per_day(np.sin(ang_d)),
        "dow_cos": per_day(np.cos(ang_d)),
        "month_sin": per_day(np.sin(ang_m)),
        "month_cos": per_day(np.cos(ang_m)),
        "is_weekend": per_day(cal["is_weekend"].to_numpy()),
        "is_holiday": per_day(cal["is_holiday"].to_numpy()),
        "days_to_holiday": per_day(cal["days_to_holiday"].to_numpy()),
        "days_since_holiday": per_day(cal["days_since_holiday"].to_numpy()),
    }

    cols = feature_columns(groups)
    origin_days = days - pd.Timedelta(days=h)
    wx_needed = [c for c in cols if c not in f]
    for c in wx_needed:
        if c == "temp_wmean_delta_origin":
            f[c] = _wx_block(wx, days, "temp_wmean") - _wx_block(wx, origin_days, "temp_wmean")
        elif c == "cdh24_delta_origin":
            f[c] = _wx_block(wx, days, "cdh24_wmean") - _wx_block(wx, origin_days, "cdh24_wmean")
        else:
            f[c] = _wx_block(wx, days, c)

    out: dict[str, object] = {
        "date": np.repeat(days.to_numpy(), BLOCKS_PER_DAY),
        "block": np.tile(np.arange(1, BLOCKS_PER_DAY + 1, dtype=np.int64), n),
        "origin_date": np.repeat(origin_days.to_numpy(), BLOCKS_PER_DAY),
        "load": take_rows(L, rows).ravel(),
    }
    for c in cols:
        out[c] = np.ascontiguousarray(f[c], dtype=np.float64).ravel()
    return pd.DataFrame(out)


__all__ = [
    "FEATURE_GROUPS",
    "ALL_GROUPS",
    "LoadMatrix",
    "normalise_load_frame",
    "build_load_matrix",
    "calendar_frame",
    "feature_columns",
    "build_design",
]
