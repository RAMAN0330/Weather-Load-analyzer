"""Haryana per-circle load contribution weights and weather→circle mapping.

Used by the forecasting pipeline to build a load-weighted weather time series:
weather features (temperature, humidity, precipitation, etc.) are aggregated
across circles using CIRCLE_CONTRIBUTION_PCT so a localised event (e.g. rain
only in Gurugram) only affects the state-level forecast in proportion to that
circle's share of total Haryana load.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

# ============================================================
# 1) Per-circle contribution to Haryana total load (percent)
# ============================================================
CIRCLE_CONTRIBUTION_PCT: Dict[str, float] = {
    "GURUGRAM CIRCLE LOAD": 16.68,
    "FARIDABAD CIRCLE LOAD": 10.24,
    "HISAR CIRCLE LOAD": 7.04,
    "SONIPAT CIRCLE LOAD": 6.32,
    "BHIWANI CIRCLE LOAD": 5.53,
    "KARNAL CIRCLE LOAD": 5.19,
    "ROHTAK CIRCLE LOAD": 4.96,
    "REWARI CIRCLE LOAD": 3.95,
    "JHAJJAR CIRCLE LOAD": 3.71,
    "PANIPAT CIRCLE LOAD": 3.64,
    "JIND CIRCLE LOAD": 3.45,
    "PALWAL CIRCLE LOAD": 3.04,
    "KURUSHETRA CIRCLE LOAD": 2.99,
    "YAMUNANAGAR CIRCLE LOAD": 2.91,
    "SIRSA CIRCLE LOAD": 2.76,
    "NUH CIRCLE LOAD": 2.50,
    "PANCHKULA CIRCLE LOAD": 2.27,
    "KAITHAL CIRCLE LOAD": 2.22,
    "AMBALA CIRCLE LOAD": 2.00,
    "NARNAUL CIRCLE LOAD": 1.87,
    "FATEHABAD CIRCLE LOAD": 0.09,
}

# ============================================================
# 2) Weather-location -> circle mapping
# ============================================================
LOCATION_TO_CIRCLE: Dict[str, str] = {
    "AMBALA": "AMBALA CIRCLE LOAD",
    "BHIWANI": "BHIWANI CIRCLE LOAD",
    "FARIDABAD": "FARIDABAD CIRCLE LOAD",
    "GURUGRAM": "GURUGRAM CIRCLE LOAD",
    "HISAR": "HISAR CIRCLE LOAD",
    "JHAJJAR": "JHAJJAR CIRCLE LOAD",
    "JIND": "JIND CIRCLE LOAD",
    "KAITHAL": "KAITHAL CIRCLE LOAD",
    "KARNAL": "KARNAL CIRCLE LOAD",
    "KURUKSHETRA": "KURUSHETRA CIRCLE LOAD",
    "MAHENDRAGARH": "NARNAUL CIRCLE LOAD",
    "NUH": "NUH CIRCLE LOAD",
    "PALWAL": "PALWAL CIRCLE LOAD",
    "PANCHKULA": "PANCHKULA CIRCLE LOAD",
    "PANIPAT": "PANIPAT CIRCLE LOAD",
    "REWARI": "REWARI CIRCLE LOAD",
    "ROHTAK": "ROHTAK CIRCLE LOAD",
    "SIRSA": "SIRSA CIRCLE LOAD",
    "SONIPAT": "SONIPAT CIRCLE LOAD",
    "YAMUNANAGAR": "YAMUNANAGAR CIRCLE LOAD",
}

# Only the circles that have a corresponding weather location actually
# contribute weighted weather; other circles get implicit zero weather weight
# (their weather is unobserved, so we cannot attribute anything to them).
LOCATION_WEIGHT_PCT: Dict[str, float] = {
    loc: CIRCLE_CONTRIBUTION_PCT.get(circle, 0.0)
    for loc, circle in LOCATION_TO_CIRCLE.items()
}

# Sum of weights covered by observed locations.  We renormalise so the
# weighted mean still represents 100% of the *observed* load — otherwise
# every weather signal would be muted by the uncovered share.
_TOTAL_OBSERVED_PCT: float = float(sum(LOCATION_WEIGHT_PCT.values()))


def _normalise_location(name: str) -> str:
    return str(name).strip().upper().replace("-", "_").replace(" ", "_").replace("__", "_")


def location_weight(location: str, *, renormalise: bool = True) -> float:
    """Return the load-weight (0..1) for a single weather location."""
    key = _normalise_location(location).replace("_", "")
    # Tolerant matching: try a couple of forms.
    for candidate in (location.upper().strip(), key, _normalise_location(location).replace("_", " ")):
        if candidate in LOCATION_WEIGHT_PCT:
            pct = LOCATION_WEIGHT_PCT[candidate]
            denom = _TOTAL_OBSERVED_PCT if renormalise else 100.0
            return float(pct / denom) if denom > 0 else 0.0
    # Try fuzzy lookup
    for k, v in LOCATION_WEIGHT_PCT.items():
        if k.replace(" ", "") == key:
            denom = _TOTAL_OBSERVED_PCT if renormalise else 100.0
            return float(v / denom) if denom > 0 else 0.0
    return 0.0


WEATHER_COLS_FOR_WEIGHTING = (
    "temperature",
    "humidity",
    "precipitation",
    "apparent_temperature",
    "cloud_cover",
    "cloud_cover_low",
    "sunshine_duration",
    "direct_radiation",
    "wind_speed_10m",
)


def aggregate_weather_by_circle(
    per_location_df: pd.DataFrame,
    *,
    location_col: str = "location",
    block_col: str = "time_block",
    weather_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Collapse a per-location weather frame into a single Haryana-level frame.

    Each location's weather row is multiplied by its renormalised load-weight
    (so locations with no weather observation are excluded from the denominator
    rather than dragging the average to zero).  Locations not in
    LOCATION_TO_CIRCLE get weight 0 — they're just dropped silently.
    """
    if per_location_df is None or per_location_df.empty:
        return pd.DataFrame(columns=[block_col, *list(weather_cols or WEATHER_COLS_FOR_WEIGHTING)])

    cols = [c for c in (weather_cols or WEATHER_COLS_FOR_WEIGHTING) if c in per_location_df.columns]
    if not cols:
        return pd.DataFrame(columns=[block_col])

    df = per_location_df.copy()
    df[location_col] = df[location_col].astype(str).str.strip().str.upper()
    df["__weight"] = df[location_col].map(lambda s: location_weight(s, renormalise=True))

    df = df[df["__weight"] > 0.0].copy()
    if df.empty:
        return pd.DataFrame(columns=[block_col, *cols])

    # Per-block weighted mean: sum(weight * value) / sum(weight)
    out = pd.DataFrame({block_col: sorted(df[block_col].dropna().unique())}).set_index(block_col)
    weight_sum = df.groupby(block_col)["__weight"].sum()
    for c in cols:
        weighted_vals = df["__weight"] * pd.to_numeric(df[c], errors="coerce")
        weighted_sum = weighted_vals.groupby(df[block_col]).sum()
        out[c] = (weighted_sum / weight_sum.replace(0, np.nan)).reindex(out.index)
    return out.reset_index()


__all__ = [
    "CIRCLE_CONTRIBUTION_PCT",
    "LOCATION_TO_CIRCLE",
    "LOCATION_WEIGHT_PCT",
    "WEATHER_COLS_FOR_WEIGHTING",
    "location_weight",
    "aggregate_weather_by_circle",
]
