"""District → state weather aggregation.

The primary spatial weighting is **demand-share weighting**: each district's
weather is weighted by the load share of the distribution circle it maps to
(``haryana_circle_weights``). Weights are renormalised at every timestamp over
the districts that actually reported, and the reported weight share is kept as
a coverage diagnostic. A plain equal mean is also produced for comparison.

Non-linear physics (wet-bulb, dew point, cooling-degree-hours) is computed per
district *before* aggregation, because f(mean(x)) ≠ mean(f(x)).

All computations are vectorised over a (time × district) matrix built on a
complete 15-minute grid; missing values stay NaN (never zero-filled).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import BLOCKS_PER_DAY
from .physics import cooling_degree, dew_point_magnus, rolling_degree_hours, wet_bulb_stull

try:
    from ..haryana_circle_weights import CIRCLE_CONTRIBUTION_PCT, LOCATION_TO_CIRCLE
except ImportError:  # running with backend/ on sys.path (script mode)
    from haryana_circle_weights import CIRCLE_CONTRIBUTION_PCT, LOCATION_TO_CIRCLE

# Weather-location aliases seen in upstream feeds → canonical district names.
DISTRICT_ALIASES: dict[str, str] = {
    "GURGAON": "GURUGRAM",
    "KURUSHETRA": "KURUKSHETRA",
    "NARNAUL": "MAHENDRAGARH",
    "MEWAT": "NUH",
    "SONEPAT": "SONIPAT",
    "YAMUNA NAGAR": "YAMUNANAGAR",
    "HISSAR": "HISAR",
}

# District → load share (%). Fatehabad has its own (tiny) circle but no entry
# in LOCATION_TO_CIRCLE, so it is added explicitly.
_HARYANA_DISTRICT_TO_CIRCLE: dict[str, str] = {
    **LOCATION_TO_CIRCLE,
    "FATEHABAD": "FATEHABAD CIRCLE LOAD",
}
HARYANA_DISTRICT_WEIGHT_PCT: dict[str, float] = {
    d: float(CIRCLE_CONTRIBUTION_PCT.get(c, 0.0)) for d, c in _HARYANA_DISTRICT_TO_CIRCLE.items()
}

WEATHER_VARS: tuple[str, ...] = (
    "temperature",
    "humidity",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "direct_radiation",
)

# Output columns that are diagnostics, not model features.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("wx_weight_coverage", "wx_district_count")


@dataclass(frozen=True)
class SpatialConfig:
    """Thresholds for extreme / persistence features."""

    hot_temp_c: float = 35.0
    hot_wet_bulb_c: float = 26.0
    cdh_base_c: float = 24.0
    hot_day_temp_c: float = 38.0
    night_blocks: tuple[int, int] = (1, 24)


def canonical_district(name: object) -> str:
    """Normalise a location/district label to an upper-case canonical name."""
    s = re.sub(r"[\s_\-]+", " ", str(name).strip().upper()).strip()
    return DISTRICT_ALIASES.get(s, s)


def is_haryana(region: str) -> bool:
    return str(region).strip().lower() in {"haryana", "hr", "hry"}


def expected_districts(region: str) -> list[str]:
    """Districts expected to report weather for ``region`` (empty = unknown)."""
    return sorted(HARYANA_DISTRICT_WEIGHT_PCT) if is_haryana(region) else []


def district_weights(region: str, districts: Sequence[str]) -> tuple[NDArray[np.float64], float]:
    """Demand-share weights for ``districts`` and the total expected weight.

    Returns ``(w, w_total)`` where ``w`` aligns with ``districts`` (unknown
    districts get 0) and ``w_total`` is the sum over all *expected* districts,
    used as the coverage denominator. Regions without a load-share table fall
    back to equal weights over the districts seen.
    """
    if is_haryana(region):
        w = np.array([HARYANA_DISTRICT_WEIGHT_PCT.get(d, 0.0) for d in districts], dtype=np.float64)
        total = float(sum(HARYANA_DISTRICT_WEIGHT_PCT.values()))
        if w.sum() > 0:
            return w, total
    w = np.ones(len(districts), dtype=np.float64)
    return w, float(max(len(districts), 1))


def day_grid(dates: Iterable) -> pd.DatetimeIndex:
    """Complete daily DatetimeIndex spanning the given dates."""
    d = pd.to_datetime(pd.Index(list(dates))).normalize()
    if len(d) == 0:
        return pd.DatetimeIndex([], name="date")
    return pd.date_range(d.min(), d.max(), freq="D", name="date")


def long_to_matrix(
    long: pd.DataFrame,
    var: str,
    days: pd.DatetimeIndex,
    districts: Sequence[str],
) -> NDArray[np.float64]:
    """Scatter a long frame into a (n_days·96 × n_districts) matrix (NaN = missing).

    Duplicate (date, block, district) rows are averaged.
    """
    n_t = len(days) * BLOCKS_PER_DAY
    out = np.full((n_t, len(districts)), np.nan)
    if var not in long.columns or long.empty or n_t == 0:
        return out
    vals = pd.to_numeric(long[var], errors="coerce").to_numpy(dtype=np.float64)
    day_idx = days.get_indexer(pd.to_datetime(long["date"]).dt.normalize())
    blk = long["block"].to_numpy(dtype=np.int64)
    d_idx = pd.Index(districts).get_indexer(long["district"])
    ok = (day_idx >= 0) & (d_idx >= 0) & (blk >= 1) & (blk <= BLOCKS_PER_DAY) & np.isfinite(vals)
    t_idx = day_idx[ok] * BLOCKS_PER_DAY + (blk[ok] - 1)
    flat = t_idx * len(districts) + d_idx[ok]
    sums = np.bincount(flat, weights=vals[ok], minlength=out.size)
    cnts = np.bincount(flat, minlength=out.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        dense = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return dense.reshape(out.shape)


def weighted_mean(m: NDArray[np.float64], w: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Row-wise weighted mean renormalised over non-NaN cells.

    Returns ``(mean, weight_present)``; ``mean`` is NaN where no district reported.
    """
    present = np.isfinite(m)
    wm = np.where(present, w[None, :], 0.0)
    wsum = wm.sum(axis=1)
    num = (np.where(present, m, 0.0) * wm).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, num / wsum, np.nan)
    return mean, wsum


def weighted_share_at_or_above(m: NDArray[np.float64], w: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Share (0..1) of reporting load-weight whose value is ≥ ``threshold``."""
    present = np.isfinite(m)
    wm = np.where(present, w[None, :], 0.0)
    wsum = wm.sum(axis=1)
    with np.errstate(invalid="ignore"):
        hot = (wm * (np.where(present, m, -np.inf) >= threshold)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, hot / wsum, np.nan)


def nan_percentile_rows(m: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Row-wise percentile ignoring NaN (linear interpolation, numpy default).

    Fully vectorised replacement for ``np.nanpercentile(m, q, axis=1)``, which
    falls back to a per-row Python loop.
    """
    if m.shape[1] == 0:
        return np.full(m.shape[0], np.nan)
    s = np.sort(m, axis=1)  # NaN sorted to the end
    n = np.isfinite(m).sum(axis=1)
    pos = (np.maximum(n, 1) - 1) * (q / 100.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    frac = pos - lo
    v_lo = np.take_along_axis(s, lo[:, None], axis=1)[:, 0]
    v_hi = np.take_along_axis(s, hi[:, None], axis=1)[:, 0]
    return np.where(n > 0, v_lo + (v_hi - v_lo) * frac, np.nan)


def _nan_reduce(fn, m: NDArray[np.float64], **kw) -> NDArray[np.float64]:
    """Apply a nan-aware reducer along axis 1, silencing all-NaN warnings."""
    if m.shape[1] == 0:
        return np.full(m.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(m, axis=1, **kw)


def _consecutive_true(flags: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Run length of consecutive True values ending at each position."""
    c = np.cumsum(flags.astype(np.int64))
    reset = np.maximum.accumulate(np.where(~flags, c, 0))
    return c - reset


def normalise_weather_long(weather_long: pd.DataFrame) -> pd.DataFrame:
    """Coerce a long weather frame to canonical dtypes / district names."""
    if weather_long is None or weather_long.empty:
        return pd.DataFrame(columns=["date", "block", "district", *WEATHER_VARS])
    w = weather_long.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w["block"] = pd.to_numeric(w["block"], errors="coerce").astype("Int64")
    w = w.dropna(subset=["date", "block"])
    w["block"] = w["block"].astype(np.int64)
    w["district"] = w["district"].map(canonical_district)
    for v in WEATHER_VARS:
        w[v] = pd.to_numeric(w[v], errors="coerce") if v in w.columns else np.nan
    return w


def build_state_weather_features(
    weather_long: pd.DataFrame,
    region: str = "haryana",
    config: SpatialConfig = SpatialConfig(),
    days: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """State-level weather feature frame indexed by (date, block).

    Columns (NaN where unobserved):
      equal mean      temp_mean, hum_mean
      weighted        temp_wmean, hum_wmean, wb_wmean, dew_wmean, precip_wmean,
                      cloud_wmean, wind_wmean, rad_wmean
      extremes        temp_p90, temp_max, temp_std, hot_share, wb_p90, wb_max,
                      wb_hot_share
      degree-hours    cdh6_wmean, cdh24_wmean, cdh24_p90
      persistence     night_min_temp, hot_day_streak, prev_day_temp_wmax
      diagnostics     wx_weight_coverage (0..1 of expected load share),
                      wx_district_count
    """
    w = normalise_weather_long(weather_long)
    if days is None:
        days = day_grid(w["date"]) if not w.empty else pd.DatetimeIndex([], name="date")
    index = pd.MultiIndex.from_product([days, np.arange(1, BLOCKS_PER_DAY + 1)], names=["date", "block"])
    seen = sorted(w["district"].dropna().unique().tolist())
    districts = sorted(set(seen) | set(expected_districts(region)))
    weights, w_total = district_weights(region, districts)

    mats = {v: long_to_matrix(w, v, days, districts) for v in WEATHER_VARS}
    t, rh = mats["temperature"], mats["humidity"]

    # Per-district physics before aggregation.
    wb, _ = wet_bulb_stull(t, rh)
    dew = dew_point_magnus(t, rh)
    cd = cooling_degree(t, config.cdh_base_c)
    cdh6 = rolling_degree_hours(cd, 24)
    cdh24 = rolling_degree_hours(cd, 96)

    ones = np.ones(len(districts))
    temp_wmean, w_present = weighted_mean(t, weights)
    feats: dict[str, NDArray[np.float64]] = {
        "temp_mean": weighted_mean(t, ones)[0],
        "hum_mean": weighted_mean(rh, ones)[0],
        "temp_wmean": temp_wmean,
        "hum_wmean": weighted_mean(rh, weights)[0],
        "wb_wmean": weighted_mean(wb, weights)[0],
        "dew_wmean": weighted_mean(dew, weights)[0],
        "precip_wmean": weighted_mean(mats["precipitation"], weights)[0],
        "cloud_wmean": weighted_mean(mats["cloud_cover"], weights)[0],
        "wind_wmean": weighted_mean(mats["wind_speed_10m"], weights)[0],
        "rad_wmean": weighted_mean(mats["direct_radiation"], weights)[0],
        "temp_p90": nan_percentile_rows(t, 90),
        "temp_max": _nan_reduce(np.nanmax, t),
        "temp_std": _nan_reduce(np.nanstd, t),
        "hot_share": weighted_share_at_or_above(t, weights, config.hot_temp_c),
        "wb_p90": nan_percentile_rows(wb, 90),
        "wb_max": _nan_reduce(np.nanmax, wb),
        "wb_hot_share": weighted_share_at_or_above(wb, weights, config.hot_wet_bulb_c),
        "cdh6_wmean": weighted_mean(cdh6, weights)[0],
        "cdh24_wmean": weighted_mean(cdh24, weights)[0],
        "cdh24_p90": nan_percentile_rows(cdh24, 90),
    }

    # Persistence features (per date, broadcast to the 96 blocks).
    n_days = len(days)
    tw_daily = temp_wmean.reshape(n_days, BLOCKS_PER_DAY) if n_days else np.empty((0, BLOCKS_PER_DAY))
    lo, hi = config.night_blocks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        night_min = np.nanmin(tw_daily[:, lo - 1 : hi], axis=1) if n_days else np.empty(0)
        day_max = np.nanmax(tw_daily, axis=1) if n_days else np.empty(0)
    streak = _consecutive_true(np.nan_to_num(day_max, nan=-np.inf) >= config.hot_day_temp_c).astype(np.float64)
    prev_max = np.concatenate([[np.nan], day_max[:-1]]) if n_days else np.empty(0)
    feats["night_min_temp"] = np.repeat(night_min, BLOCKS_PER_DAY)
    feats["hot_day_streak"] = np.repeat(streak, BLOCKS_PER_DAY)
    feats["prev_day_temp_wmax"] = np.repeat(prev_max, BLOCKS_PER_DAY)

    feats["wx_weight_coverage"] = w_present / w_total if w_total > 0 else np.zeros_like(w_present)
    feats["wx_district_count"] = np.isfinite(t).sum(axis=1).astype(np.float64)
    return pd.DataFrame(feats, index=index)


__all__ = [
    "SpatialConfig",
    "HARYANA_DISTRICT_WEIGHT_PCT",
    "WEATHER_VARS",
    "DIAGNOSTIC_COLUMNS",
    "canonical_district",
    "expected_districts",
    "district_weights",
    "day_grid",
    "long_to_matrix",
    "weighted_mean",
    "weighted_share_at_or_above",
    "normalise_weather_long",
    "build_state_weather_features",
]
