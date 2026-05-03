"""
Weather anomaly detection for the short-term load forecasting pipeline.

Produces a WeatherAnomalyResult for a target day's 96-block weather DataFrame.
All thresholds are configurable via the config dict passed to detect_weather_anomalies().

Anomaly types:
  rain_alert          — moderate rain/showers (>3 mm block-avg)
  heavy_rain_alert    — heavy rain (>8 mm block-avg or any block >15 mm)
  wind_alert          — sustained wind >25 km/h on any block
  gusty_wind_alert    — sustained wind >40 km/h on any block
  heat_alert          — temperature >42°C on any block
  cold_alert          — temperature <8°C on any block
  severe_cold_alert   — temperature <4°C on any block
  dust_storm_alert    — composite: wind spike + cloud surge + radiation drop + humidity jump
                        all within the same 8-block (2-hour) window
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Default thresholds ─────────────────────────────────────────────────────────
_DEFAULTS = {
    # Precipitation (mm averaged over 96 blocks)
    "rain_mm_avg":            3.0,
    "heavy_rain_mm_avg":      8.0,
    "heavy_rain_mm_peak":    15.0,   # single-block trigger

    # Wind (km/h, any single block)
    "wind_kmh_alert":        25.0,
    "wind_kmh_gusty":        40.0,

    # Temperature (°C, any single block)
    "heat_c":                42.0,
    "cold_c":                 8.0,
    "severe_cold_c":          4.0,

    # Dust-storm composite (all conditions must fire within window_blocks)
    "dust_wind_kmh":         25.0,
    "dust_cloud_rise_pct":   30.0,   # cloud_cover increase vs prev block
    "dust_rad_drop_pct":     40.0,   # direct_radiation drop vs prev block (%)
    "dust_humidity_rise_pct":20.0,   # relative humidity rise vs prev block
    "dust_window_blocks":     8,

    # Rough MW impact coefficients (negative = suppression)
    "rain_mw_per_mm":       -80.0,
    "wind_mw_per_kmh":       -5.0,
    "heat_mw_per_c":         60.0,   # above 42°C each extra degree adds load
    "cold_mw_per_c":        -40.0,   # below 8°C each degree drop reduces load
}

# Severity ladder
_SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _max_severity(a: str, b: str) -> str:
    return _SEVERITY_ORDER[max(_SEVERITY_ORDER.index(a), _SEVERITY_ORDER.index(b))]


@dataclass
class WeatherAnomalyResult:
    day_flags: Dict[str, bool] = field(default_factory=dict)
    block_flags: Dict[str, List[int]] = field(default_factory=dict)
    severity: str = "NONE"
    description: str = ""
    load_impact_estimate_mw: float = 0.0
    clamp_sigma_override: float = 2.5
    correction_decay_override: float = 0.997


def detect_weather_anomalies(
    target_df: pd.DataFrame,
    config: dict = None,
) -> WeatherAnomalyResult:
    """
    Analyse weather for the target day and return a WeatherAnomalyResult.

    Parameters
    ----------
    target_df : DataFrame with columns including temperature, rain, showers,
                wind_speed_10m, cloud_cover, direct_radiation, humidity (or
                relative_humidity_2m).  Must have time_block (1-96) or be
                already sorted in block order (96 rows).
    config    : Optional dict to override default thresholds (keys match _DEFAULTS).
    """
    cfg = {**_DEFAULTS, **(config or {})}

    # ── Guard: empty input ─────────────────────────────────────────────────────
    if target_df is None or len(target_df) == 0:
        return WeatherAnomalyResult()

    # ── Normalise input ────────────────────────────────────────────────────────
    df = target_df.copy()
    if "time_block" in df.columns:
        df = df.sort_values("time_block").reset_index(drop=True)

    def _col(name: str, alt: str = None, default: float = 0.0) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy()
        if alt and alt in df.columns:
            return pd.to_numeric(df[alt], errors="coerce").fillna(default).to_numpy()
        return np.full(len(df), default)

    temp     = _col("temperature", default=25.0)
    rain     = _col("rain", default=0.0)
    showers  = _col("showers", default=0.0)
    snowfall = _col("snowfall", default=0.0)
    wind     = _col("wind_speed_10m", default=0.0)
    cloud    = _col("cloud_cover", default=0.0)
    rad      = _col("direct_radiation", default=0.0)
    humidity = _col("humidity", alt="relative_humidity_2m", default=50.0)

    precip = rain + showers   # combined precipitation per block
    n      = len(df)

    # ── Precipitation flags ────────────────────────────────────────────────────
    precip_avg  = float(np.mean(precip))
    precip_peak = float(np.max(precip))

    rain_alert       = precip_avg  >= cfg["rain_mm_avg"]
    heavy_rain_alert = (precip_avg  >= cfg["heavy_rain_mm_avg"] or
                        precip_peak >= cfg["heavy_rain_mm_peak"])
    rain_blocks      = [i + 1 for i in range(n) if precip[i] > 1.0]

    # ── Wind flags ─────────────────────────────────────────────────────────────
    wind_max      = float(np.max(wind))
    wind_alert    = wind_max >= cfg["wind_kmh_alert"]
    gusty_alert   = wind_max >= cfg["wind_kmh_gusty"]
    wind_blocks   = [i + 1 for i in range(n) if wind[i] >= cfg["wind_kmh_alert"]]

    # ── Temperature flags ──────────────────────────────────────────────────────
    temp_max      = float(np.max(temp))
    temp_min      = float(np.min(temp))
    heat_alert    = temp_max >= cfg["heat_c"]
    cold_alert    = temp_min <= cfg["cold_c"]
    severe_cold   = temp_min <= cfg["severe_cold_c"]
    heat_blocks   = [i + 1 for i in range(n) if temp[i] >= cfg["heat_c"]]
    cold_blocks   = [i + 1 for i in range(n) if temp[i] <= cfg["cold_c"]]

    # ── Dust-storm composite ───────────────────────────────────────────────────
    win   = int(cfg["dust_window_blocks"])
    dust_storm_alert = False
    dust_blocks: List[int] = []

    if n >= win + 1:
        cloud_diff = np.diff(cloud, prepend=cloud[0])
        rad_pct_drop = np.where(
            rad[:-1] > 10,
            (rad[:-1] - rad[1:]) / rad[:-1].clip(min=1) * 100,
            0.0,
        )
        rad_drop = np.r_[0.0, rad_pct_drop]
        hum_diff = np.diff(humidity, prepend=humidity[0])

        for start in range(n - win):
            sl = slice(start, start + win)
            if (
                np.max(wind[sl])       >= cfg["dust_wind_kmh"]
                and np.max(cloud_diff[sl]) >= cfg["dust_cloud_rise_pct"]
                and np.max(rad_drop[sl])   >= cfg["dust_rad_drop_pct"]
                and np.max(hum_diff[sl])   >= cfg["dust_humidity_rise_pct"]
            ):
                dust_storm_alert = True
                dust_blocks = [i + 1 for i in range(start, start + win)]
                break

    # ── Load impact estimate ───────────────────────────────────────────────────
    impact = 0.0
    if rain_alert:
        impact += cfg["rain_mw_per_mm"] * precip_avg
    if wind_alert:
        excess_wind = max(0.0, wind_max - cfg["wind_kmh_alert"])
        impact += cfg["wind_mw_per_kmh"] * excess_wind
    if heat_alert:
        impact += cfg["heat_mw_per_c"] * (temp_max - cfg["heat_c"])
    if cold_alert:
        impact += cfg["cold_mw_per_c"] * (cfg["cold_c"] - temp_min)

    # ── Severity ───────────────────────────────────────────────────────────────
    sev = "NONE"
    if rain_alert or wind_alert or cold_alert:
        sev = _max_severity(sev, "MEDIUM")
    if heavy_rain_alert or gusty_alert or heat_alert or severe_cold or dust_storm_alert:
        sev = _max_severity(sev, "HIGH")
    if heavy_rain_alert and gusty_alert:
        sev = "CRITICAL"

    # ── Clamp / decay overrides ────────────────────────────────────────────────
    clamp_sigma = {
        "NONE":     2.5,
        "LOW":      2.75,
        "MEDIUM":   3.0,
        "HIGH":     4.0,
        "CRITICAL": 5.0,
    }[sev]

    decay_override = {
        "NONE":     0.997,
        "LOW":      0.997,
        "MEDIUM":   0.998,
        "HIGH":     0.999,
        "CRITICAL": 0.999,
    }[sev]

    # ── Description ───────────────────────────────────────────────────────────
    parts = []
    if heavy_rain_alert:
        parts.append(f"Heavy rain ({precip_avg:.1f} mm avg)")
    elif rain_alert:
        parts.append(f"Rain ({precip_avg:.1f} mm avg)")
    if gusty_alert:
        parts.append(f"Gusty winds ({wind_max:.0f} km/h peak)")
    elif wind_alert:
        parts.append(f"Strong winds ({wind_max:.0f} km/h peak)")
    if heat_alert:
        parts.append(f"Extreme heat ({temp_max:.1f}°C)")
    if severe_cold:
        parts.append(f"Severe cold ({temp_min:.1f}°C)")
    elif cold_alert:
        parts.append(f"Cold ({temp_min:.1f}°C)")
    if dust_storm_alert:
        parts.append("Dust storm composite")
    description = " + ".join(parts) if parts else "No anomaly"

    return WeatherAnomalyResult(
        day_flags={
            "rain_alert":        rain_alert,
            "heavy_rain_alert":  heavy_rain_alert,
            "wind_alert":        wind_alert,
            "gusty_wind_alert":  gusty_alert,
            "heat_alert":        heat_alert,
            "cold_alert":        cold_alert,
            "severe_cold_alert": severe_cold,
            "dust_storm_alert":  dust_storm_alert,
        },
        block_flags={
            "rain_blocks":       rain_blocks,
            "wind_blocks":       wind_blocks,
            "heat_blocks":       heat_blocks,
            "cold_blocks":       cold_blocks,
            "dust_storm_blocks": dust_blocks,
        },
        severity=sev,
        description=description,
        load_impact_estimate_mw=round(impact, 1),
        clamp_sigma_override=clamp_sigma,
        correction_decay_override=decay_override,
    )
