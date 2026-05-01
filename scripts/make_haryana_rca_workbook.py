"""Build Haryana forecast RCA workbook from Haryana_mape.xlsx plus API data."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
XLSX_PATH = ROOT / "Haryana_mape.xlsx"
OUT_PATH = ROOT / "exports" / "Haryana_Forecast_RCA.xlsx"
SIMPLE_OUT_PATH = ROOT / "exports" / "Haryana_RCA_OPEN_THIS.xlsx"
STATE = "HARYANA"


def _load_env() -> None:
    for env_path in (ROOT / ".env", ROOT / "env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _api_get(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    base = os.environ.get("PIPELINE_API_BASE_URL", "http://3.108.54.200:8001").rstrip("/")
    token = os.environ.get("PIPELINE_API_TOKEN") or os.environ.get("API_SECRET_KEY") or "flagbearer"
    response = requests.get(
        f"{base}{path}",
        params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def _parse_date_value(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, str):
        text = value.strip()
        if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", text):
            return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", text):
            return pd.to_datetime(text, format="%m-%d-%Y", errors="coerce")
        return pd.to_datetime(text, errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _load_mape_workbook() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", dtype={"date": object})
    block = raw[["date", "time_block", "Actual", "T+1", "T+2"]].copy()
    block = block.dropna(subset=["time_block"]).reset_index(drop=True)
    block["time_block"] = pd.to_numeric(block["time_block"], errors="coerce").astype("Int64")
    block = block.dropna(subset=["time_block"]).copy()
    block["time_block"] = block["time_block"].astype(int)

    # The workbook contains one Excel-parsed date as 2026-12-04, but the row
    # sequence is Apr 09..Apr 27. Use contiguous block groups to preserve order.
    assigned_dates: list[pd.Timestamp] = []
    current_date = None
    for idx, row in block.iterrows():
        raw_date = _parse_date_value(row["date"])
        starts_day = int(row["time_block"]) == 1 or idx == 0
        if idx == 0:
            current_date = raw_date.normalize()
        elif starts_day:
            expected = current_date + pd.Timedelta(days=1)
            if pd.isna(raw_date) or abs((raw_date.normalize() - expected).days) > 1:
                current_date = expected
            else:
                current_date = raw_date.normalize()
        assigned_dates.append(current_date)
    block["date"] = pd.to_datetime(assigned_dates).strftime("%Y-%m-%d")

    for col in ("Actual", "T+1", "T+2"):
        block[col] = pd.to_numeric(block[col], errors="coerce")

    summary = raw[["Unnamed: 12", "Unnamed: 13", "Unnamed: 14"]].copy()
    summary = summary.dropna(how="all")
    summary = summary[summary["Unnamed: 12"].astype(str).str.lower().str.strip() != "date"].copy()
    summary = summary.rename(
        columns={
            "Unnamed: 12": "date",
            "Unnamed: 13": "t1_accuracy_pct_source",
            "Unnamed: 14": "t2_accuracy_pct_source",
        }
    )
    summary["date"] = summary["date"].map(_parse_date_value)
    summary = summary.dropna(subset=["date"])
    summary["date"] = pd.to_datetime(summary["date"]).dt.strftime("%Y-%m-%d")
    summary["t1_accuracy_pct_source"] = pd.to_numeric(summary["t1_accuracy_pct_source"], errors="coerce")
    summary["t2_accuracy_pct_source"] = pd.to_numeric(summary["t2_accuracy_pct_source"], errors="coerce")
    summary = summary.drop_duplicates(subset=["date"], keep="first")
    return block, summary


def _clean_load(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "time_block", "total_drawal"])
    df = pd.DataFrame(rows)
    tb_col = "time_block" if "time_block" in df.columns else "block"
    load_col = next((c for c in ("total_drawal", "load_mw", "drawal_mw", "demand_mw", "mw") if c in df.columns), None)
    if load_col is None:
        return pd.DataFrame(columns=["date", "time_block", "total_drawal"])
    out = df[["date", tb_col, load_col]].rename(columns={tb_col: "time_block", load_col: "total_drawal"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["time_block"] = pd.to_numeric(out["time_block"], errors="coerce")
    out["total_drawal"] = pd.to_numeric(out["total_drawal"], errors="coerce")
    out = out.dropna(subset=["date", "time_block", "total_drawal"])
    out["time_block"] = out["time_block"].astype(int)
    return out.sort_values(["date", "time_block"]).reset_index(drop=True)


def _merge_actual_load(sldc: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    if sldc.empty:
        return load
    if load.empty:
        return sldc
    rows: list[dict[str, Any]] = []
    dates = sorted(set(sldc["date"]) | set(load["date"]))
    for date in dates:
        s = sldc[sldc["date"] == date].set_index("time_block")["total_drawal"]
        l = load[load["date"] == date].set_index("time_block")["total_drawal"]
        last_sldc = int(s.dropna().index.max()) if not s.dropna().empty else 0
        for block in range(1, 97):
            value = s.get(block) if block <= last_sldc else l.get(block)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = l.get(block, s.get(block))
            if value is not None and not pd.isna(value):
                rows.append({"date": date, "time_block": block, "total_drawal": float(value)})
    return pd.DataFrame(rows)


def _clean_weather(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "time_block"])
    df = pd.DataFrame(rows)
    rename = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "rain_mm": "rain",
        "showers_mm": "showers",
        "snowfall_cm": "snowfall",
        "direct_radiation_instant": "direct_radiation",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "time_block" not in df.columns and "block" in df.columns:
        df = df.rename(columns={"block": "time_block"})
    keep = [
        "date",
        "time_block",
        "temperature",
        "humidity",
        "precipitation",
        "rain",
        "showers",
        "cloud_cover",
        "sunshine_duration",
        "direct_radiation",
        "wind_speed_10m",
        "apparent_temperature",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["time_block"] = pd.to_numeric(out["time_block"], errors="coerce")
    out = out.dropna(subset=["date", "time_block"])
    out["time_block"] = out["time_block"].astype(int)
    for col in keep[2:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["date", "time_block"]).reset_index(drop=True)


def _fetch_context(from_date: str, to_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    params = {"state": STATE, "from_date": from_date, "to_date": to_date, "limit": 200000}
    status: dict[str, Any] = {"from_date": from_date, "to_date": to_date, "state": STATE}
    sldc_rows = _api_get("/sldc", params)
    load_rows = _api_get("/load", params)
    weather_rows = _api_get("/weather/mean", params)
    sldc = _clean_load(sldc_rows)
    load = _clean_load(load_rows)
    actual = _merge_actual_load(sldc, load)
    weather = _clean_weather(weather_rows)
    status.update(
        {
            "sldc_rows": len(sldc),
            "load_rows": len(load),
            "actual_rows_after_merge": len(actual),
            "weather_rows": len(weather),
        }
    )
    return actual, weather, status


def _accuracy(actual: pd.Series, forecast: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(forecast, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(f) & (a > 0)
    if not mask.any():
        return np.nan
    return float(100.0 - np.mean(np.abs(f[mask] - a[mask]) / a[mask]) * 100.0)


def _segment_name(block: int) -> str:
    if block <= 32:
        return "night"
    if block <= 48:
        return "morning"
    if block <= 72:
        return "afternoon/ramp"
    return "evening"


def _join_unique(items: list[str]) -> str:
    return " ".join(dict.fromkeys([item for item in items if item]))


def _daily_context(actual: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if actual.empty:
        load_daily = pd.DataFrame(columns=["date"])
    else:
        load_daily = actual.groupby("date").agg(
            actual_blocks=("time_block", "nunique"),
            actual_mean_mw=("total_drawal", "mean"),
            actual_peak_mw=("total_drawal", "max"),
            actual_min_mw=("total_drawal", "min"),
            actual_energy_mwh=("total_drawal", lambda s: float(np.sum(s) * 0.25)),
            actual_max_ramp_mw=("total_drawal", lambda s: float(pd.to_numeric(s, errors="coerce").diff().abs().max())),
        ).reset_index()
    if weather.empty:
        weather_daily = pd.DataFrame(columns=["date"])
    else:
        weather_daily = weather.groupby("date").agg(
            weather_blocks=("time_block", "nunique"),
            temp_mean_c=("temperature", "mean"),
            temp_max_c=("temperature", "max"),
            humidity_mean_pct=("humidity", "mean"),
            precipitation_sum_mm=("precipitation", "sum"),
            rain_sum_mm=("rain", "sum"),
            wind_max=("wind_speed_10m", "max"),
            apparent_temp_max_c=("apparent_temperature", "max"),
        ).reset_index()
    daily = load_daily.merge(weather_daily, on="date", how="outer")
    daily = daily.sort_values("date")
    daily["trailing7_actual_mean_mw"] = daily["actual_mean_mw"].shift(1).rolling(7, min_periods=3).mean()
    daily["actual_vs_trailing7_pct"] = (
        (daily["actual_mean_mw"] - daily["trailing7_actual_mean_mw"])
        / daily["trailing7_actual_mean_mw"].replace(0, np.nan)
        * 100.0
    )
    if "temp_max_c" in daily.columns and "humidity_mean_pct" in daily.columns:
        daily["heat_index_proxy"] = daily["temp_max_c"] * (1.0 + (daily["humidity_mean_pct"] / 100.0))
    else:
        daily["heat_index_proxy"] = np.nan
    for col in (
        "temp_max_c",
        "apparent_temp_max_c",
        "humidity_mean_pct",
        "rain_sum_mm",
        "wind_max",
        "heat_index_proxy",
    ):
        if col not in daily.columns:
            daily[col] = np.nan
        base = daily[col].shift(1).rolling(7, min_periods=3)
        mean_col = f"{col}_trailing7"
        delta_col = f"{col}_delta_vs_trailing7"
        z_col = f"{col}_z_vs_trailing7"
        daily[mean_col] = base.mean()
        daily[delta_col] = daily[col] - daily[mean_col]
        daily[z_col] = daily[delta_col] / base.std().replace(0, np.nan)
    return daily


def _abnormal_weather_factor(row: pd.Series) -> str:
    factors: list[str] = []
    temp_max = row.get("temp_max_c", np.nan)
    temp_delta = row.get("temp_max_c_delta_vs_trailing7", np.nan)
    apparent_max = row.get("apparent_temp_max_c", np.nan)
    apparent_delta = row.get("apparent_temp_max_c_delta_vs_trailing7", np.nan)
    humidity = row.get("humidity_mean_pct", np.nan)
    humidity_delta = row.get("humidity_mean_pct_delta_vs_trailing7", np.nan)
    rain_sum = row.get("rain_sum_mm", row.get("precipitation_sum_mm", np.nan))
    rain_delta = row.get("rain_sum_mm_delta_vs_trailing7", np.nan)
    wind_max = row.get("wind_max", np.nan)
    wind_delta = row.get("wind_max_delta_vs_trailing7", np.nan)
    heat_index = row.get("heat_index_proxy", np.nan)
    heat_index_delta = row.get("heat_index_proxy_delta_vs_trailing7", np.nan)

    if pd.notna(temp_max) and (temp_max >= 38.0 or (pd.notna(temp_delta) and temp_delta >= 3.0)):
        suffix = f", +{temp_delta:.1f}C vs trailing 7d" if pd.notna(temp_delta) else ""
        factors.append(f"Heat abnormality: max temp {temp_max:.1f}C{suffix}.")
    if pd.notna(apparent_max) and (apparent_max >= 40.0 or (pd.notna(apparent_delta) and apparent_delta >= 3.0)):
        suffix = f", +{apparent_delta:.1f}C vs trailing 7d" if pd.notna(apparent_delta) else ""
        factors.append(f"Apparent-temperature abnormality: {apparent_max:.1f}C{suffix}.")
    if pd.notna(humidity) and (humidity >= 70.0 or (pd.notna(humidity_delta) and humidity_delta >= 8.0)):
        suffix = f", +{humidity_delta:.1f} points vs trailing 7d" if pd.notna(humidity_delta) else ""
        factors.append(f"Humidity abnormality: mean humidity {humidity:.1f}%{suffix}.")
    if pd.notna(heat_index) and (heat_index >= 60.0 or (pd.notna(heat_index_delta) and heat_index_delta >= 7.0)):
        suffix = f", +{heat_index_delta:.1f} vs trailing 7d" if pd.notna(heat_index_delta) else ""
        factors.append(f"Heat-index interaction abnormality: proxy {heat_index:.1f}{suffix}.")
    if pd.notna(rain_sum) and (rain_sum >= 1.0 or (pd.notna(rain_delta) and rain_delta >= 1.0)):
        suffix = f", +{rain_delta:.1f} mm vs trailing 7d" if pd.notna(rain_delta) else ""
        factors.append(f"Rain abnormality: daily rain {rain_sum:.1f} mm{suffix}.")
    if pd.notna(wind_max) and (wind_max >= 18.0 or (pd.notna(wind_delta) and wind_delta >= 5.0)):
        suffix = f", +{wind_delta:.1f} vs trailing 7d" if pd.notna(wind_delta) else ""
        factors.append(f"Wind abnormality: max wind {wind_max:.1f}{suffix}.")
    if not factors:
        return "No clear abnormal weather factor from fetched weather versus trailing 7-day context."
    return _join_unique(factors)


def _model_bias_reason(row: pd.Series, horizon_key: str) -> str:
    prefix = "t1" if horizon_key == "T+1" else "t2"
    mean_error = row.get(f"{prefix}_mean_error_mw", np.nan)
    energy_error = row.get(f"{prefix}_energy_error_pct", np.nan)
    worst_segment = row.get(f"{prefix}_worst_segment", np.nan)
    worst_segment_mape = row.get(f"{prefix}_worst_segment_mape_pct", np.nan)
    under_blocks = row.get(f"{prefix}_underforecast_blocks_pct", np.nan)
    if pd.isna(mean_error):
        return f"{horizon_key}: no block-level forecast rows available; daily summary only."

    direction = "underforecast" if mean_error < 0 else "overforecast"
    parts = [
        f"{horizon_key}: model {direction} on average by {abs(mean_error):.0f} MW",
    ]
    if pd.notna(energy_error):
        parts.append(f"energy error {energy_error:+.1f}%.")
    if pd.notna(under_blocks):
        parts.append(f"Underforecast blocks {under_blocks:.0f}%." if mean_error < 0 else f"Overforecast blocks {100.0 - under_blocks:.0f}%.")
    if pd.notna(worst_segment):
        segment_text = str(worst_segment)
        mape_text = f" ({worst_segment_mape:.1f}% segment MAPE)" if pd.notna(worst_segment_mape) else ""
        parts.append(f"Worst segment: {segment_text}{mape_text}.")

    abnormal = _abnormal_weather_factor(row)
    if direction == "underforecast":
        if "Heat" in abnormal or "Apparent" in abnormal or "Heat-index" in abnormal:
            parts.append("Likely missed nonlinear cooling demand from heat/apparent-temperature.")
        if "Humidity" in abnormal:
            parts.append("Humidity likely amplified cooling load beyond linear temperature response.")
        if "Rain" in abnormal:
            parts.append("Rain may have changed agricultural/urban demand timing; check whether model damped too much or missed rebound.")
        if "Wind" in abnormal:
            parts.append("Wind/weather instability may have created non-normal load shape.")
        if "No clear abnormal weather" in abnormal:
            parts.append("Weather does not explain the miss; likely baseline, live-bias, holiday, feeder, or industrial schedule issue.")
    else:
        if "Rain" in abnormal:
            parts.append("Rain/cloud likely suppressed cooling demand more than model expected.")
        elif "Wind" in abnormal:
            parts.append("Wind/storm conditions may have reduced load or shifted demand.")
        elif "Heat" in abnormal or "Humidity" in abnormal:
            parts.append("Weather was hot/humid, so overforecast points to correction stacking or baseline too high.")
        else:
            parts.append("No strong weather abnormality; overforecast likely from high baseline or excessive positive bias correction.")
    return " ".join(parts)


def _make_reason(row: pd.Series, block_rows: pd.DataFrame) -> tuple[str, str, str, str, str, str]:
    reasons: list[str] = []
    ask: list[str] = []
    learning: list[str] = []
    date = str(row["date"])
    t1_acc = row.get("t1_accuracy_pct", np.nan)
    t2_acc = row.get("t2_accuracy_pct", np.nan)
    actual_shift = row.get("actual_vs_trailing7_pct", np.nan)
    max_ramp = row.get("actual_max_ramp_mw", np.nan)
    actual_blocks = row.get("actual_blocks", np.nan)
    weather_blocks = row.get("weather_blocks", np.nan)
    ts = pd.Timestamp(date)
    weekend = ts.dayofweek >= 5
    holiday_like = date in {"2026-04-14"}
    abnormal_weather = _abnormal_weather_factor(row)
    t1_model_reason = _model_bias_reason(row, "T+1")
    t2_model_reason = _model_bias_reason(row, "T+2")

    reasons.append(abnormal_weather)
    reasons.append(t1_model_reason)
    reasons.append(t2_model_reason)

    if pd.notna(actual_blocks) and actual_blocks < 96:
        reasons.append(f"Actual load coverage is incomplete ({int(actual_blocks)}/96 blocks).")
        ask.append("Confirm revised/missing SLDC actual blocks for this date.")
    if pd.notna(weather_blocks) and weather_blocks < 96:
        reasons.append(f"Weather coverage is incomplete ({int(weather_blocks)}/96 blocks).")
        ask.append("Confirm weather feed completeness or station outage.")

    if "Heat" in abnormal_weather or "Apparent" in abnormal_weather or "Heat-index" in abnormal_weather:
        learning.append("Use nonlinear cooling-load and heat-index multipliers for April-June afternoons.")
        learning.append("Track apparent temperature separately from dry-bulb temperature.")
    if "Humidity" in abnormal_weather:
        learning.append("Treat humidity as a heat-load amplifier, not only a standalone linear feature.")
    if "Rain" in abnormal_weather:
        learning.append("Avoid double counting rain; use one weather channel plus regime flag.")
    if "Wind" in abnormal_weather:
        learning.append("Keep wind as diagnostic unless load drop is proven in backtest.")
    if pd.notna(actual_shift) and abs(actual_shift) >= 4:
        direction = "higher" if actual_shift > 0 else "lower"
        reasons.append(f"Actual average load was {abs(actual_shift):.1f}% {direction} than trailing 7-day level.")
        learning.append("Adaptive baseline should exclude anomaly days and shorten window in April-June.")
    if pd.notna(max_ramp) and max_ramp >= 350:
        reasons.append(f"Actual load had sharp ramp of {max_ramp:.0f} MW between 15-minute blocks.")
        learning.append("Apply ramp-aware forecast limit before export and monitor ramp misses.")
    if weekend:
        reasons.append("Weekend profile can differ from weekday industrial/agricultural consumption.")
        learning.append("Keep separate weekday/weekend bias tables.")
    if holiday_like:
        reasons.append("Holiday/bridge-day effect likely changed normal working-day profile.")
        learning.append("Keep Haryana/national holiday and bridge-day profile classes.")

    if not block_rows.empty:
        for horizon in ("T+1", "T+2"):
            err = pd.to_numeric(block_rows[horizon], errors="coerce") - pd.to_numeric(block_rows["Actual"], errors="coerce")
            denom = pd.to_numeric(block_rows["Actual"], errors="coerce").replace(0, np.nan)
            mape = (err.abs() / denom) * 100.0
            seg = block_rows.assign(_mape=mape, _err=err, _segment=block_rows["time_block"].map(_segment_name))
            seg_score = seg.groupby("_segment")["_mape"].mean().dropna()
            if seg_score.empty:
                continue
            worst_segment = str(seg_score.idxmax())
            mean_err = float(err.mean()) if err.notna().any() else np.nan
            direction = "over-forecast" if mean_err > 0 else "under-forecast"
            if horizon == "T+2" and worst_segment == "night":
                reasons.append(f"{horizon} error concentrated in night blocks; {direction} bias visible.")
                learning.append("Maintain dedicated T+2 night correction for blocks 1-32.")
            elif worst_segment == "afternoon/ramp":
                reasons.append(f"{horizon} worst error is in afternoon/ramp blocks; {direction} dominates.")
                learning.append("Tune afternoon ramp weights around blocks 52-68.")
            elif seg_score.max() >= 4:
                reasons.append(f"{horizon} highest MAPE segment is {worst_segment}; {direction} dominates.")
    else:
        reasons.append("No block-level forecast rows available in workbook for this date; only daily summary is available.")

    poor = (pd.notna(t1_acc) and t1_acc < 95.0) or (pd.notna(t2_acc) and t2_acc < 94.0)
    if poor and not ask and not reasons:
        ask.append("Ask SLDC if there was outage, grid restriction, industrial shutdown, agricultural feeder change, or revised schedule.")
    if poor and not ask:
        ask.append("Confirm whether any non-weather operational event occurred: outage, feeder shutdown, industry holiday, grid restriction, or schedule revision.")
    if not reasons:
        reasons.append("No clear weather, load-ramp, calendar, or data-quality driver identified from available data.")
        ask.append("Ask SLDC for operational/event reason.")
        learning.append("Create manual event flag if SLDC confirms a non-modelled cause.")

    # Keep cells readable.
    return (
        _join_unique(reasons),
        abnormal_weather,
        t1_model_reason,
        t2_model_reason,
        _join_unique(ask),
        _join_unique(learning),
    )


def _build_outputs(block: pd.DataFrame, summary: pd.DataFrame, actual: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_context = _daily_context(actual, weather)

    daily_from_blocks = []
    for date, grp in block.groupby("date", sort=True):
        row = {
            "date": date,
            "workbook_blocks": int(grp["time_block"].nunique()),
            "t1_accuracy_pct_calc": _accuracy(grp["Actual"], grp["T+1"]),
            "t2_accuracy_pct_calc": _accuracy(grp["Actual"], grp["T+2"]),
            "t1_energy_error_pct": (
                (grp["T+1"].sum() - grp["Actual"].sum()) / grp["Actual"].sum() * 100.0
                if grp["Actual"].sum() else np.nan
            ),
            "t2_energy_error_pct": (
                (grp["T+2"].sum() - grp["Actual"].sum()) / grp["Actual"].sum() * 100.0
                if grp["Actual"].sum() else np.nan
            ),
            "t1_mean_error_mw": float((grp["T+1"] - grp["Actual"]).mean()),
            "t2_mean_error_mw": float((grp["T+2"] - grp["Actual"]).mean()),
        }
        for horizon in ("T+1", "T+2"):
            signed_err = grp[horizon] - grp["Actual"]
            abs_err = signed_err.abs()
            prefix = "t1" if horizon == "T+1" else "t2"
            row[f"{prefix}_underforecast_blocks_pct"] = float((signed_err < 0).mean() * 100.0)
            seg = grp.assign(
                _segment=grp["time_block"].map(_segment_name),
                _mape=(abs_err / grp["Actual"].replace(0, np.nan)) * 100.0,
                _err=signed_err,
            )
            seg_mape = seg.groupby("_segment")["_mape"].mean().dropna()
            if not seg_mape.empty:
                worst_segment = str(seg_mape.idxmax())
                row[f"{prefix}_worst_segment"] = worst_segment
                row[f"{prefix}_worst_segment_mape_pct"] = float(seg_mape.max())
                row[f"{prefix}_worst_segment_mean_error_mw"] = float(
                    seg.loc[seg["_segment"] == worst_segment, "_err"].mean()
                )
            err = abs_err
            worst = grp.loc[err.idxmax()] if err.notna().any() else None
            if worst is not None:
                row[f"{prefix}_worst_block"] = int(worst["time_block"])
                row[f"{prefix}_worst_abs_error_mw"] = float(abs(worst[horizon] - worst["Actual"]))
        daily_from_blocks.append(row)

    block_daily = pd.DataFrame(daily_from_blocks)
    all_dates = pd.DataFrame({"date": sorted(set(summary["date"]) | set(block["date"]))})
    rca = all_dates.merge(summary, on="date", how="left").merge(block_daily, on="date", how="left").merge(daily_context, on="date", how="left")
    rca["t1_accuracy_pct"] = rca["t1_accuracy_pct_calc"].combine_first(rca["t1_accuracy_pct_source"])
    rca["t2_accuracy_pct"] = rca["t2_accuracy_pct_calc"].combine_first(rca["t2_accuracy_pct_source"])
    rca["T+1"] = rca["t1_accuracy_pct"].map(lambda v: f"{v:.2f}% accuracy" if pd.notna(v) else "No forecast blocks")
    rca["T+2"] = rca["t2_accuracy_pct"].map(lambda v: f"{v:.2f}% accuracy" if pd.notna(v) else "No forecast blocks")

    reasons = []
    for _, row in rca.iterrows():
        grp = block[block["date"] == row["date"]]
        reason, abnormal_weather, t1_model_reason, t2_model_reason, ask, learning = _make_reason(row, grp)
        reasons.append((reason, abnormal_weather, t1_model_reason, t2_model_reason, ask, learning))
    rca["Reason why it happen"] = [r[0] for r in reasons]
    rca["Abnormal weather factor"] = [r[1] for r in reasons]
    rca["T+1 model under/overforecast reason"] = [r[2] for r in reasons]
    rca["T+2 model under/overforecast reason"] = [r[3] for r in reasons]
    rca["Ask SLDC"] = [r[4] for r in reasons]
    rca["Assumption / Learning"] = [r[5] for r in reasons]

    ordered_cols = [
        "date",
        "T+1",
        "T+2",
        "Abnormal weather factor",
        "T+1 model under/overforecast reason",
        "T+2 model under/overforecast reason",
        "Reason why it happen",
        "Ask SLDC",
        "Assumption / Learning",
        "t1_accuracy_pct",
        "t2_accuracy_pct",
        "workbook_blocks",
        "actual_blocks",
        "weather_blocks",
        "actual_mean_mw",
        "actual_peak_mw",
        "actual_energy_mwh",
        "actual_vs_trailing7_pct",
        "actual_max_ramp_mw",
        "temp_max_c",
        "apparent_temp_max_c",
        "humidity_mean_pct",
        "humidity_mean_pct_delta_vs_trailing7",
        "rain_sum_mm",
        "rain_sum_mm_delta_vs_trailing7",
        "wind_max",
        "wind_max_delta_vs_trailing7",
        "heat_index_proxy",
        "heat_index_proxy_delta_vs_trailing7",
        "t1_energy_error_pct",
        "t2_energy_error_pct",
        "t1_mean_error_mw",
        "t2_mean_error_mw",
        "t1_underforecast_blocks_pct",
        "t2_underforecast_blocks_pct",
        "t1_worst_segment",
        "t2_worst_segment",
        "t1_worst_segment_mape_pct",
        "t2_worst_segment_mape_pct",
        "t1_worst_block",
        "t2_worst_block",
        "t1_worst_abs_error_mw",
        "t2_worst_abs_error_mw",
    ]
    rca = rca.reindex(columns=ordered_cols)

    block_diag = block.merge(weather, on=["date", "time_block"], how="left")
    block_diag = block_diag.merge(actual.rename(columns={"total_drawal": "actual_load_fetched"}), on=["date", "time_block"], how="left")
    block_diag["t1_error_mw"] = block_diag["T+1"] - block_diag["Actual"]
    block_diag["t2_error_mw"] = block_diag["T+2"] - block_diag["Actual"]
    block_diag["t1_mape_pct"] = block_diag["t1_error_mw"].abs() / block_diag["Actual"].replace(0, np.nan) * 100.0
    block_diag["t2_mape_pct"] = block_diag["t2_error_mw"].abs() / block_diag["Actual"].replace(0, np.nan) * 100.0
    return rca, block_diag, daily_context


def main() -> int:
    _load_env()
    block, summary = _load_mape_workbook()
    min_date = min(pd.to_datetime(summary["date"]).min(), pd.to_datetime(block["date"]).min())
    max_date = max(pd.to_datetime(summary["date"]).max(), pd.to_datetime(block["date"]).max())
    from_date = (min_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    to_date = max_date.strftime("%Y-%m-%d")

    actual, weather, status = _fetch_context(from_date, to_date)
    rca, block_diag, daily_context = _build_outputs(block, summary, actual, weather)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        rca.to_excel(writer, sheet_name="RCA_Summary", index=False)
        block_diag.to_excel(writer, sheet_name="Block_Diagnostics", index=False)
        daily_context.to_excel(writer, sheet_name="Daily_Context", index=False)
        pd.DataFrame([status]).to_excel(writer, sheet_name="Data_Status", index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                header = str(col_cells[0].value or "")
                width = min(max(len(header) + 2, 12), 55)
                ws.column_dimensions[col_cells[0].column_letter].width = width

    simple_cols = [
        "date",
        "T+1",
        "T+2",
        "Abnormal weather factor",
        "T+1 model under/overforecast reason",
        "T+2 model under/overforecast reason",
        "Reason why it happen",
        "Ask SLDC",
        "Assumption / Learning",
        "t1_accuracy_pct",
        "t2_accuracy_pct",
        "actual_mean_mw",
        "actual_peak_mw",
        "temp_max_c",
        "humidity_mean_pct",
        "rain_sum_mm",
        "wind_max",
        "heat_index_proxy",
    ]
    with pd.ExcelWriter(SIMPLE_OUT_PATH, engine="openpyxl") as writer:
        rca.reindex(columns=simple_cols).to_excel(writer, sheet_name="RCA", index=False)

    print(f"Written: {OUT_PATH}")
    print(f"Simple Excel: {SIMPLE_OUT_PATH}")
    print(f"RCA rows: {len(rca)} | block rows: {len(block_diag)}")
    print(
        f"Fetched actual rows: {status['actual_rows_after_merge']} | "
        f"weather rows: {status['weather_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
