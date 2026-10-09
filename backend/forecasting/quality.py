"""Input data-quality checks and the output validation gate.

Status semantics (contract): ``ok`` | ``degraded`` (forecast produced but inputs
below standard — warnings must be shown) | ``failed``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import BLOCKS_PER_DAY
from .features import normalise_load_frame
from .spatial import expected_districts, normalise_weather_long

STATUS_ORDER: dict[str, int] = {"ok": 0, "degraded": 1, "failed": 2}

LOAD_DEGRADED_BELOW: float = 0.99  # share of expected blocks present
LOAD_FAILED_BELOW: float = 0.50
WEATHER_DEGRADED_BELOW_PCT: float = 90.0
DEFAULT_RAMP_THRESHOLD_MW: float = 1000.0


def worst_status(*statuses: str) -> str:
    """Most severe of the given statuses."""
    return max(statuses, key=lambda s: STATUS_ORDER.get(s, 2)) if statuses else "ok"


def _date_range(date_from: pd.Timestamp | str, date_to: pd.Timestamp | str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(date_from).normalize(), pd.Timestamp(date_to).normalize(), freq="D")


def validate_load(
    load_df: pd.DataFrame,
    date_from: pd.Timestamp | str,
    date_to: pd.Timestamp | str,
) -> tuple[dict[str, Any], str, list[str]]:
    """Check the 96-block-per-date contract over [date_from, date_to].

    Returns ``(load_section, status, warnings)`` where ``load_section`` holds
    the contract keys: expected_blocks, present_blocks, duplicate_blocks,
    nonpositive_blocks, missing_dates.
    """
    days = _date_range(date_from, date_to)
    lf = normalise_load_frame(load_df)
    lf = lf[lf["date"].isin(days) & lf["block"].between(1, BLOCKS_PER_DAY)]
    src_dupes = int(getattr(load_df, "attrs", {}).get("source_duplicate_blocks", 0) or 0) if load_df is not None else 0
    dupes = int(lf.duplicated(subset=["date", "block"]).sum()) + src_dupes
    finite = lf[np.isfinite(lf["load"].to_numpy(dtype=np.float64))]
    uniq = finite.drop_duplicates(subset=["date", "block"])
    present = int(len(uniq))
    nonpos = int((uniq["load"] <= 0).sum())
    expected = len(days) * BLOCKS_PER_DAY
    per_day = uniq.groupby("date").size().reindex(days, fill_value=0)
    missing_dates = [d.strftime("%Y-%m-%d") for d in per_day.index[per_day.to_numpy() == 0]]
    partial = int(((per_day > 0) & (per_day < BLOCKS_PER_DAY)).sum())

    warnings_: list[str] = []
    ratio = present / expected if expected else 0.0
    status = "ok"
    if ratio < LOAD_FAILED_BELOW:
        status = "failed"
        warnings_.append(f"Only {present} of {expected} load blocks present ({ratio:.0%}).")
    elif ratio < LOAD_DEGRADED_BELOW:
        status = "degraded"
        warnings_.append(f"{expected - present} of {expected} load blocks missing.")
    if missing_dates:
        status = worst_status(status, "degraded")
        warnings_.append(f"{len(missing_dates)} date(s) have no load data.")
    if partial:
        warnings_.append(f"{partial} date(s) have fewer than {BLOCKS_PER_DAY} load blocks.")
    if nonpos:
        status = worst_status(status, "degraded")
        warnings_.append(f"{nonpos} load block(s) are non-positive.")
    if dupes:
        warnings_.append(f"{dupes} duplicate load block(s) were averaged.")
    section = {
        "expected_blocks": expected,
        "present_blocks": present,
        "duplicate_blocks": dupes,
        "nonpositive_blocks": nonpos,
        "missing_dates": missing_dates,
    }
    return section, status, warnings_


def validate_weather(
    weather_long: pd.DataFrame,
    date_from: pd.Timestamp | str,
    date_to: pd.Timestamp | str,
    region: str,
) -> tuple[dict[str, Any], str, list[str]]:
    """District coverage of temperature over [date_from, date_to].

    ``coverage_pct`` = present (district, date, block) temperature cells of the
    expected districts / all expected cells × 100.
    """
    days = _date_range(date_from, date_to)
    w = normalise_weather_long(weather_long)
    w = w[w["date"].isin(days) & w["block"].between(1, BLOCKS_PER_DAY)]
    w = w[np.isfinite(w["temperature"].to_numpy(dtype=np.float64))]
    seen = sorted(w["district"].unique().tolist())
    expected = expected_districts(region) or seen
    w_exp = w[w["district"].isin(expected)].drop_duplicates(subset=["date", "block", "district"])
    denom = len(expected) * len(days) * BLOCKS_PER_DAY
    coverage = float(len(w_exp) / denom * 100.0) if denom else 0.0
    missing = sorted(set(expected) - set(seen))
    seen_expected = len(set(expected) & set(seen))

    warnings_: list[str] = []
    status = "ok"
    if not seen:
        status = "degraded"
        warnings_.append("No district weather available; model ran without weather signal.")
    else:
        if missing:
            warnings_.append(
                f"Weather for {len(missing)} of {len(expected)} districts missing; weights renormalised."
            )
        if coverage < WEATHER_DEGRADED_BELOW_PCT:
            status = "degraded"
            warnings_.append(f"Weather coverage {coverage:.1f}% is below {WEATHER_DEGRADED_BELOW_PCT:.0f}%.")
    section = {
        "districts_expected": len(expected),
        "districts_seen": seen_expected,
        "coverage_pct": round(coverage, 1),
        "missing_districts": missing,
    }
    return section, status, warnings_


def data_quality_report(
    region: str,
    load_df: pd.DataFrame,
    weather_long: pd.DataFrame,
    date_from: pd.Timestamp | str,
    date_to: pd.Timestamp | str,
) -> dict[str, Any]:
    """Data-quality payload for ``GET /api/v3/data-quality``."""
    load_sec, ls, lw = validate_load(load_df, date_from, date_to)
    wx_sec, ws, ww = validate_weather(weather_long, date_from, date_to, region)
    return {
        "region": region,
        "from": pd.Timestamp(date_from).strftime("%Y-%m-%d"),
        "to": pd.Timestamp(date_to).strftime("%Y-%m-%d"),
        "load": load_sec,
        "weather": wx_sec,
        "status": worst_status(ls, ws),
        "warnings": lw + ww,
    }


def validate_forecast_blocks(
    blocks: Sequence[Mapping[str, Any]],
    ramp_threshold_mw: float = DEFAULT_RAMP_THRESHOLD_MW,
) -> tuple[str, list[str]]:
    """Output gate: exactly 96 blocks, finite, p10 ≤ p50 ≤ p90, non-negative.

    Block-to-block p50 ramps above ``ramp_threshold_mw`` produce warnings only.
    """
    if len(blocks) != BLOCKS_PER_DAY:
        return "failed", [f"Forecast has {len(blocks)} blocks; expected {BLOCKS_PER_DAY}."]
    q = np.array(
        [[np.nan if b.get(k) is None else float(b[k]) for k in ("p10", "p50", "p90")] for b in blocks],
        dtype=np.float64,
    )
    issues: list[str] = []
    if not np.isfinite(q).all():
        issues.append(f"{int((~np.isfinite(q)).any(axis=1).sum())} block(s) have non-finite quantiles.")
    with np.errstate(invalid="ignore"):
        if ((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])).any():
            issues.append("Quantile crossing detected (p10 ≤ p50 ≤ p90 violated).")
        if (q < 0).any():
            issues.append("Negative load forecast detected.")
    if issues:
        return "failed", issues
    ramps = np.abs(np.diff(q[:, 1]))
    big = np.flatnonzero(ramps > ramp_threshold_mw)
    warnings_: list[str] = []
    if big.size:
        warnings_.append(
            f"{big.size} block-to-block ramp(s) exceed {ramp_threshold_mw:.0f} MW (max {ramps.max():.0f} MW at block {int(big[np.argmax(ramps[big])]) + 2})."
        )
    return "ok", warnings_


__all__ = [
    "worst_status",
    "validate_load",
    "validate_weather",
    "data_quality_report",
    "validate_forecast_blocks",
    "DEFAULT_RAMP_THRESHOLD_MW",
]
