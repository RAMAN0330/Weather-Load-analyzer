"""Haryana afternoon ramp and weather regime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BLOCKS_PER_DAY = 96


@dataclass
class RampAdjustmentResult:
    adjustment_mw: np.ndarray
    heat_index: np.ndarray
    multiplier: np.ndarray
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


def compute_heat_index(temperature_c: Any, humidity_pct: Any) -> np.ndarray:
    t = _as_96(temperature_c, 25.0)
    h = _as_96(humidity_pct, 50.0)
    return (
        -8.78
        + 1.611 * t
        + 2.338 * h
        - 0.146 * t * h
        - 0.0123 * (t**2)
        - 0.0164 * (h**2)
        + 0.00221 * (t**2) * h
        + 0.000725 * t * (h**2)
        - 0.00000358 * (t**2) * (h**2)
    )


def classify_weather_regime(target_df: pd.DataFrame, *, region: str = "haryana") -> str:
    if target_df is None or target_df.empty:
        return "NORMAL"
    temp = pd.to_numeric(target_df.get("temperature", pd.Series([25.0])), errors="coerce").fillna(25.0)
    hum = pd.to_numeric(target_df.get("humidity", pd.Series([50.0])), errors="coerce").fillna(50.0)
    apparent = pd.to_numeric(target_df.get("apparent_temperature", temp), errors="coerce").fillna(temp)
    wind = pd.to_numeric(target_df.get("wind_speed_10m", pd.Series([0.0])), errors="coerce").fillna(0.0)
    rain = pd.to_numeric(target_df.get("rain", target_df.get("precipitation", pd.Series([0.0]))), errors="coerce").fillna(0.0)
    showers = pd.to_numeric(target_df.get("showers", pd.Series([0.0])), errors="coerce").fillna(0.0)
    cloud = pd.to_numeric(target_df.get("cloud_cover", pd.Series([0.0])), errors="coerce").fillna(0.0)

    if float((rain + showers).mean()) > 1.0 or float((rain + showers).max()) > 5.0:
        return "RAINY"
    if float(apparent.max()) > 42.0 and float(hum.mean()) < 35.0 and float(wind.max()) > 25.0:
        return "DUSTY_AANDHI"
    if float(apparent.max()) > 42.0 or (float(temp.max()) > 40.0 and float(hum.mean()) < 45.0):
        return "PRE_MONSOON_HOT"
    if float(cloud.mean()) > 75.0 and float(wind.max()) > 25.0:
        return "DUSTY_AANDHI"
    return "NORMAL"


def build_afternoon_ramp_adjustment(
    baseline_96: Any,
    target_df: pd.DataFrame,
    *,
    season: str,
    region: str = "haryana",
    enabled: bool = True,
) -> RampAdjustmentResult:
    baseline = _as_96(baseline_96)
    if not enabled or target_df is None or target_df.empty:
        z = np.zeros(BLOCKS_PER_DAY, dtype=float)
        return RampAdjustmentResult(z, z, np.ones(BLOCKS_PER_DAY), {"enabled": bool(enabled), "applied": False})

    ordered = target_df.copy()
    if "time_block" in ordered.columns:
        ordered["time_block"] = pd.to_numeric(ordered["time_block"], errors="coerce")
        ordered = ordered.dropna(subset=["time_block"]).sort_values("time_block")
    temp = _as_96(pd.to_numeric(ordered.get("temperature", pd.Series([25.0])), errors="coerce").fillna(25.0).to_numpy())
    hum = _as_96(pd.to_numeric(ordered.get("humidity", pd.Series([50.0])), errors="coerce").fillna(50.0).to_numpy())
    apparent = _as_96(pd.to_numeric(ordered.get("apparent_temperature", pd.Series(temp)), errors="coerce").fillna(pd.Series(temp)).to_numpy())
    heat_index = compute_heat_index(temp, hum)
    cooling_load = np.maximum(0.0, temp - 24.0) ** 1.5
    heating_load = np.maximum(0.0, 18.0 - temp) ** 1.3
    heat_index_effect = temp * (1.0 + (hum / 100.0))
    regime = classify_weather_regime(ordered, region=region)

    blocks = np.arange(1, BLOCKS_PER_DAY + 1)
    hours = ((blocks - 1) * 15) // 60
    multiplier = np.ones(BLOCKS_PER_DAY, dtype=float)
    adjustment_pct = np.zeros(BLOCKS_PER_DAY, dtype=float)
    is_haryana = str(region or "").lower().strip() in {"haryana", "hr"}
    hot_month_season = str(season).lower().strip() in {"spring", "summer", "pre_monsoon"}

    afternoon = (hours >= 13) & (hours <= 17)
    morning = (hours >= 10) & (hours <= 12)
    if is_haryana and hot_month_season:
        hot_excess = np.maximum(0.0, apparent - 38.0)
        heat_boost = np.minimum(1.6, 1.0 + (hot_excess * 0.04))
        ramp_shape = 1.0 + 0.3 * np.sin(np.pi * np.clip((hours - 13) / 4.0, 0.0, 1.0))
        multiplier[afternoon] = heat_boost[afternoon] * ramp_shape[afternoon]
        multiplier[morning] = np.maximum(multiplier[morning], 1.15)

        if regime == "PRE_MONSOON_HOT":
            adjustment_pct[afternoon] = np.clip((multiplier[afternoon] - 1.0) * 0.055, 0.0, 0.08)
        elif regime == "DUSTY_AANDHI":
            adjustment_pct[afternoon] = -0.04
        elif regime == "RAINY":
            adjustment_pct[afternoon] = -0.025
        else:
            adjustment_pct[afternoon] = np.clip((multiplier[afternoon] - 1.0) * 0.025, 0.0, 0.035)
        nonlinear_heat = np.clip((cooling_load / 120.0) + (heat_index_effect - 36.0) / 1200.0, 0.0, 0.035)
        adjustment_pct[afternoon] = np.clip(adjustment_pct[afternoon] + nonlinear_heat[afternoon], -0.05, 0.10)

    if np.any(heating_load > 0):
        cold = temp <= 18.0
        adjustment_pct[cold] = np.clip(adjustment_pct[cold] + (heating_load[cold] / 180.0), -0.05, 0.06)

    # Wind delta dampening: suppress ramp adjustment on anomalous wind days.
    # wind_max_delta_vs_trailing7 > 8 km/h correlates with systematic overforecast
    # because wind-driven convective cooling suppresses AC load even on hot days.
    try:
        _wind_delta_col = ordered.get("wind_max_delta_vs_trailing7", pd.Series([0.0]))
        wind_delta = float(
            pd.to_numeric(_wind_delta_col, errors="coerce").fillna(0.0).max()
        )
        if wind_delta > 8.0:
            wind_scale = max(0.88, 1.0 - (wind_delta - 8.0) * 0.015)
            adjustment_pct = adjustment_pct * wind_scale
        else:
            wind_scale = 1.0
    except Exception:
        wind_delta = 0.0
        wind_scale = 1.0

    adjustment_mw = baseline * adjustment_pct
    return RampAdjustmentResult(
        adjustment_mw=adjustment_mw,
        heat_index=heat_index,
        multiplier=multiplier,
        metadata={
            "enabled": bool(enabled),
            "applied": bool(np.any(np.abs(adjustment_mw) > 1e-6)),
            "regime": regime,
            "season": str(season),
            "region": str(region),
            "mean_heat_index": round(float(np.nanmean(heat_index)), 3),
            "max_apparent_temp": round(float(np.nanmax(apparent)), 3),
            "mean_cooling_load": round(float(np.nanmean(cooling_load)), 3),
            "mean_heating_load": round(float(np.nanmean(heating_load)), 3),
            "mean_heat_index_effect": round(float(np.nanmean(heat_index_effect)), 3),
            "wind_delta_vs_trailing7": round(wind_delta, 3),
            "wind_scale_applied": round(wind_scale, 4),
            "mean_adjustment_mw": round(float(np.mean(adjustment_mw)), 3),
            "max_abs_adjustment_mw": round(float(np.max(np.abs(adjustment_mw))), 3),
            "multiplier": multiplier.tolist(),
            "adjustment_mw": adjustment_mw.tolist(),
        },
    )
