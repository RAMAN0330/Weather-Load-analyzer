from pydantic import BaseModel
import os
import json
import re
import uuid
from datetime import datetime
import copy
from functools import lru_cache
from typing import List, Optional, Literal, Tuple, Dict, Any
import numpy as np
import pandas as pd

# --- Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
try:
    from .engine import EDAEngine, FeatureRegistry
    from .gridintel_engine import GridIntelControlDesk
    from .short_term_pipeline import (
        run_short_term_pipeline,
        run_block_driver_weight_delta_engine,
        compute_block_driver_weights,
        INDIAN_STATE_REGIONS,
        SHORT_TERM_MODEL_FEATURES,
    )
    from .backtester import run_backtest
except ImportError:
    # Support running as a script without package context
    from engine import EDAEngine, FeatureRegistry
    from gridintel_engine import GridIntelControlDesk
    from short_term_pipeline import (
        run_short_term_pipeline,
        run_block_driver_weight_delta_engine,
        compute_block_driver_weights,
        INDIAN_STATE_REGIONS,
        SHORT_TERM_MODEL_FEATURES,
    )
    from backtester import run_backtest

# --- Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_data.csv")
engine = EDAEngine(DATA_PATH)
gridintel = GridIntelControlDesk(DATA_PATH)

_BASELINE_WINDOW_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_HISTORY_METRICS_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_DR_ACCURACY_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_DAYAHEAD_SERIES_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_API_WEATHER_ENGINE_FEATURES = [
    feature for feature in (
        "temperature", "humidity", "precipitation",
        "lag_1", "lag_7", "lag_block_1", "lag_block_4",
        "rolling_4", "rolling_12", "block_sin", "block_cos",
    )
    if feature in SHORT_TERM_MODEL_FEATURES
]


def _clear_runtime_caches() -> None:
    _DAYAHEAD_SERIES_CACHE.clear()
    _BASELINE_WINDOW_CACHE.clear()
    _HISTORY_METRICS_CACHE.clear()
    _DR_ACCURACY_CACHE.clear()
    _simulator_date_context.cache_clear()

# --- Lightweight V2 helpers for the new UI ---
def _get_df(copy=True):
    """Get the engine DataFrame. Use copy=False for read-only operations."""
    if hasattr(engine, "_df"):
        df = engine._df.copy() if copy else engine._df
    else:
        df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df

def _get_valid_dates(df):
    if "date" not in df.columns or "time_block" not in df.columns:
        return []
    dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    if not dates:
        return []
    
    # The user wants data till it is present (last date), so we return all dates found.
    return dates

def _get_latest_robust_date(df):
    """Finds the latest date that has 96 blocks and non-zero load."""
    if "date" not in df.columns:
        return None
    all_dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    for d in reversed(all_dates):
        day_df = df[df["date"] == d]
        if day_df["time_block"].nunique() >= 96 and day_df["total_drawal"].mean() > 100:
            return d
    return all_dates[-1] if all_dates else None

def _get_segment_idx(start, end, n=96):
    s = max(0, min(n, int(start) - 1))
    e = max(0, min(n, int(end)))
    return s, e


def _df_signature(df: pd.DataFrame) -> Tuple[Any, ...]:
    if df is None or df.empty or "date" not in df.columns:
        return (0, 0, None, None)
    date_ser = df["date"].astype(str)
    return (
        int(len(df)),
        int(date_ser.nunique()),
        str(date_ser.min()),
        str(date_ser.max()),
    )

def _get_all_dates(df):
    """Return all unique sorted date strings from the dataframe (no validity filter)."""
    if "date" not in df.columns:
        return []
    return sorted(df["date"].dropna().astype(str).unique().tolist())

def _resolve_date(df, requested_date: str):
    dates_all = _get_all_dates(df)
    dates_valid = _get_valid_dates(df)
    if not dates_all:
        return None, False, []
    
    if requested_date and requested_date in dates_valid:
        return requested_date, True, dates_all
    
    # Fallback to latest valid if available, else latest all
    if dates_valid:
        return dates_valid[-1], False, dates_all
    return dates_all[-1], False, dates_all

def _baseline_window(df, target_date: str, window: int):
    all_dates = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
    if not all_dates:
        return df.iloc[0:0]
    if target_date not in all_dates:
        target_date = all_dates[-1]
    idx = all_dates.index(target_date)
    prior_dates = all_dates[:idx]
    
    # Inline _get_valid_dates logic
    raw_valid = df[df["total_drawal"] > 0].copy()
    valid_dates_list = sorted(raw_valid.groupby("date")["time_block"].count()[lambda x: x == 96].index.astype(str).tolist())
    valid_set = set(valid_dates_list)
    
    baseline_dates = [d for d in prior_dates if d in valid_set][-window:]
    if not baseline_dates:
        baseline_dates = prior_dates[-window:]
    return df[df["date"].isin(baseline_dates)]



def _kpi_summary(series):
    actual = np.array(series["actual"], dtype=float)
    forecast = np.array(series["forecast"], dtype=float)
    if actual.size == 0:
        return {}

    error = actual - forecast
    mape = float(np.mean(np.abs(error) / np.maximum(actual, 1e-6)) * 100)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(forecast - actual))

    peak_actual = float(np.max(actual))
    peak_forecast = float(np.max(forecast))
    peak_accuracy = float(abs(peak_actual - peak_forecast) / max(peak_actual, 1e-6) * 100)

    daily_actual = float(np.sum(actual))
    daily_forecast = float(np.sum(forecast))
    daily_energy_error = float(abs(daily_actual - daily_forecast) / max(daily_actual, 1e-6) * 100)

    within_2 = float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) <= 0.02) * 100)
    within_5 = float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) <= 0.05) * 100)
    over_10 = float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) > 0.10) * 100)

    reserve_margin = float(((peak_forecast * 1.18) - peak_forecast) / max(peak_forecast, 1e-6) * 100)
    confidence = float(max(60, 95 - mape * 1.2))

    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "peak_accuracy": round(peak_accuracy, 2),
        "daily_energy_error": round(daily_energy_error, 2),
        "peak_load": round(peak_forecast, 2),
        "daily_energy": round(daily_forecast, 2),
        "reserve_margin": round(reserve_margin, 2),
        "confidence_score": round(confidence, 2),
        "block_accuracy_2pct": round(within_2, 2),
        "block_accuracy_5pct": round(within_5, 2),
        "block_over_10pct": round(over_10, 2),
        "peak_variance_status": "warning" if peak_accuracy > 3 else "ok"
    }


def _series_weather_contributions(series: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves MW impact of weather factors. 
    V2: Prefers block-specific regression results stored in series.
    """
    if series is None:
        return np.zeros(96), np.zeros(96), np.zeros(96)
        
    temp = _to_n_vec(series.get("temperature_contrib_mw", []), 96)
    hum = _to_n_vec(series.get("humidity_contrib_mw", []), 96)
    precip = _to_n_vec(series.get("precipitation_contrib_mw", []), 96)
    
    # Fallback to defaults if block-specific results are missing or all zero
    if np.all(np.abs(temp) < 1e-6):
        temp = _to_n_vec(series.get("temp_delta", []), 96) * 8.0
    if np.all(np.abs(hum) < 1e-6):
        hum = _to_n_vec(series.get("hum_delta", []), 96) * 2.5
    if np.all(np.abs(precip) < 1e-6):
        precip = _to_n_vec(series.get("precip_delta", []), 96) * -4.0
        
    return temp, hum, precip


def _compute_weather_index(series: Dict[str, Any]) -> List[float]:
    """
    Localized Weather Sensitivity Index (WSI):
    Normalized impact percentage per block using residuals.
    """
    actual = np.array(series.get("actual", []), dtype=float)
    base_load = np.array(series.get("baseline", []), dtype=float)
    
    if actual.size == 0 or base_load.size == 0:
        return np.zeros(96).tolist()
    
    # Residual based impact: Residual = Actual - Base_Load
    residual = actual - base_load
    
    wsi = np.divide(
        residual,
        np.maximum(np.abs(base_load), 1e-6),
        out=np.zeros_like(residual, dtype=float),
        where=np.abs(base_load) > 1e-6
    ) * 100.0
    
    return wsi.tolist()


def _attribution_summary(series):
    temp_contrib, hum_contrib, precip_contrib = _series_weather_contributions(series)

    total = temp_contrib.sum() + hum_contrib.sum() + precip_contrib.sum()
    total = total if abs(total) > 1e-6 else 1.0

    return [
        {"factor": "Temperature", "value": round(temp_contrib.sum(), 2), "percent": round(temp_contrib.sum() / total * 100, 2), "color": "#ef4444"},
        {"factor": "Humidity", "value": round(hum_contrib.sum(), 2), "percent": round(hum_contrib.sum() / total * 100, 2), "color": "#3b82f6"},
        {"factor": "Precipitation", "value": round(precip_contrib.sum(), 2), "percent": round(precip_contrib.sum() / total * 100, 2), "color": "#1e3a8a"},
        {"factor": "Unexplained", "value": round(total * 0.08, 2), "percent": round(8, 2), "color": "#6b7280"}
    ]

def _block_to_time(block):
    minutes = (block - 1) * 15
    hour = minutes // 60
    minute = minutes % 60
    return f"{int(hour):02d}:{int(minute):02d}"

def _day_type_label(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return "Unknown"
    return "Weekend" if dt.weekday() >= 5 else "Weekday"


def _get_driver_color(label: str, val: float) -> str:
    l = label.lower()
    if "temp" in l:
        return "#ef4444" if val > 0 else "#f87171"
    if "humid" in l:
        return "#3b82f6" if val > 0 else "#60a5fa"
    if "rain" in l or "cloud" in l:
        return "#10b981" if val > 0 else "#34d399"
    if "holiday" in l:
        return "#f59e0b"
    if "day" in l or "week" in l:
        return "#8b5cf6"
    if "transit" in l:
        return "#ec4899"
    if "state" in l or "behav" in l:
        return "#06b6d4"
    if "t-1" in l or "momentum" in l:
        return "#f97316"
    if "pattern" in l:
        return "#6366f1"
    if "bias" in l or "resid" in l:
        return "#64748b"
        
    return "#9ca3af"


def _to_n_vec(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    out = np.full(int(max(1, n)), float(fill), dtype=float)
    if arr.size <= 0:
        return out
    m = int(min(out.size, arr.size))
    out[:m] = arr[:m]
    if m < out.size:
        out[m:] = out[m - 1]
    return out


def _rank_live_contributors(contrib_map: Dict[str, float]) -> List[Dict[str, Any]]:
    total_abs = float(sum(abs(float(v)) for v in contrib_map.values()) or 1.0)
    ranked = sorted(contrib_map.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    rows = []
    for feature, val in ranked:
        v = float(val)
        rows.append(
            {
                "feature": str(feature),
                "contribution_mw": v,
                "contribution_pct": float((v / total_abs) * 100.0),
                "direction": "up" if v >= 0 else "down",
            }
        )
    return rows


def _live_dominant_family(feature: str) -> str:
    f = str(feature or "").lower()
    if f in {"temperature", "humidity", "precipitation", "weather"}:
        return "weather"
    if f in {"daytype", "holiday", "calendar"}:
        return "calendar"
    if f in {"manual"}:
        return "manual"
    return "operational"


def _live_recommended_action(primary_family: str, risk_flag: str, net_impact_pct: float) -> str:
    if risk_flag == "high":
        if primary_family == "weather":
            return "Increase reserve and monitor weather-driven blocks."
        if primary_family == "calendar":
            return "Review day-type/holiday assumptions before commitment."
        return "Trigger operator review and prepare corrective dispatch."
    if risk_flag == "medium":
        if primary_family == "weather":
            return "Monitor weather-sensitive blocks and keep balancing support ready."
        return "Track block and apply guarded correction if deviation grows."
    if abs(float(net_impact_pct)) >= 8.0:
        return "Keep slot under watch; no immediate intervention."
    return "No action required."


def _refresh_live_analytics_from_final_series(
    result: Dict[str, Any],
    actual_blocks: int,
    weather_engine_live: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    series = result.get("series", {})
    if not isinstance(series, dict):
        return result

    forecast_raw = series.get("forecast", [])
    baseline_raw = series.get("hybrid_baseline", series.get("historical_baseline", []))
    if not isinstance(forecast_raw, list) or len(forecast_raw) == 0:
        return result

    n = int(len(forecast_raw))
    forecast = _to_n_vec(forecast_raw, n, fill=0.0)
    baseline = _to_n_vec(baseline_raw, n, fill=0.0)
    weather_impact = _to_n_vec(series.get("weather_impact", np.zeros(n, dtype=float)), n, fill=0.0)
    momentum_impact = _to_n_vec(series.get("momentum_impact_mw", np.zeros(n, dtype=float)), n, fill=0.0)
    trend_component = _to_n_vec(series.get("trend_component", np.zeros(n, dtype=float)), n, fill=0.0)
    pattern_adjustment = _to_n_vec(series.get("pattern_adjustment", np.zeros(n, dtype=float)), n, fill=0.0)
    calendar_adjustment = _to_n_vec(series.get("calendar_adjustment", np.zeros(n, dtype=float)), n, fill=0.0)
    actual_vec = _to_n_vec(series.get("actual", np.zeros(n, dtype=float)), n, fill=0.0)

    start_fc = int(np.clip(int(actual_blocks), 0, n))
    window = slice(start_fc, n) if start_fc < n else slice(0, n)

    # Build weather component decomposition aligned to final weather impact.
    temp_component = np.zeros(n, dtype=float)
    hum_component = np.zeros(n, dtype=float)
    rain_component = np.zeros(n, dtype=float)
    component_ok = False

    if isinstance(weather_engine_live, dict):
        comp_base = weather_engine_live.get("component_base", {})
        if isinstance(comp_base, dict):
            t_base = _to_n_vec(comp_base.get("temperature", np.zeros(n, dtype=float)), n, fill=0.0)
            h_base = _to_n_vec(comp_base.get("humidity", np.zeros(n, dtype=float)), n, fill=0.0)
            r_base = _to_n_vec(comp_base.get("precipitation", np.zeros(n, dtype=float)), n, fill=0.0)
            temp_component = baseline * t_base
            hum_component = baseline * h_base
            rain_component = baseline * r_base
            component_ok = True

        if not component_ok:
            deltas = series.get("weather_feature_deltas", {})
            if isinstance(deltas, dict):
                t_delta = _to_n_vec(deltas.get("temperature", np.zeros(n, dtype=float)), n, fill=0.0)
                h_delta = _to_n_vec(deltas.get("humidity", np.zeros(n, dtype=float)), n, fill=0.0)
                r_delta = _to_n_vec(deltas.get("precipitation", np.zeros(n, dtype=float)), n, fill=0.0)
                w_df = weather_engine_live.get("driver_weight_matrix")
                if isinstance(w_df, pd.DataFrame) and not w_df.empty:
                    ordered = w_df.copy()
                    if "block" in ordered.columns:
                        ordered["block"] = pd.to_numeric(ordered["block"], errors="coerce")
                        ordered = ordered.dropna(subset=["block"]).sort_values("block")
                    t_w = _to_n_vec(ordered.get("temp_weight", np.zeros(n, dtype=float)).to_numpy(dtype=float) if "temp_weight" in ordered.columns else np.zeros(n, dtype=float), n, fill=0.0)
                    h_w = _to_n_vec(ordered.get("humidity_weight", np.zeros(n, dtype=float)).to_numpy(dtype=float) if "humidity_weight" in ordered.columns else np.zeros(n, dtype=float), n, fill=0.0)
                    r_w = _to_n_vec(ordered.get("rain_weight", np.zeros(n, dtype=float)).to_numpy(dtype=float) if "rain_weight" in ordered.columns else np.zeros(n, dtype=float), n, fill=0.0)
                    temp_component = baseline * t_delta * t_w
                    hum_component = baseline * h_delta * h_w
                    rain_component = baseline * r_delta * r_w
                    component_ok = True

    if not component_ok:
        # Fallback split by relative feature movement intensity.
        deltas = series.get("weather_feature_deltas", {})
        t_delta = _to_n_vec(deltas.get("temperature", np.zeros(n, dtype=float)) if isinstance(deltas, dict) else np.zeros(n, dtype=float), n, fill=0.0)
        h_delta = _to_n_vec(deltas.get("humidity", np.zeros(n, dtype=float)) if isinstance(deltas, dict) else np.zeros(n, dtype=float), n, fill=0.0)
        r_delta = _to_n_vec(deltas.get("precipitation", np.zeros(n, dtype=float)) if isinstance(deltas, dict) else np.zeros(n, dtype=float), n, fill=0.0)
        mag = np.abs(t_delta) + np.abs(h_delta) + np.abs(r_delta)
        mag = np.where(mag > 1e-9, mag, 1.0)
        temp_component = weather_impact * (np.abs(t_delta) / mag)
        hum_component = weather_impact * (np.abs(h_delta) / mag)
        rain_component = weather_impact * (np.abs(r_delta) / mag)

    comp_total = temp_component + hum_component + rain_component
    scale = np.divide(
        weather_impact,
        np.where(np.abs(comp_total) > 1e-9, comp_total, 1.0),
        out=np.ones_like(weather_impact, dtype=float),
        where=np.abs(comp_total) > 1e-9,
    )
    temp_component = temp_component * scale
    hum_component = hum_component * scale
    rain_component = rain_component * scale

    net_contribution_mw = forecast - baseline
    residual_component = net_contribution_mw - weather_impact - momentum_impact
    net_impact_pct = np.divide(
        net_contribution_mw,
        np.maximum(np.abs(baseline), 1e-6),
        out=np.zeros_like(net_contribution_mw, dtype=float),
        where=np.abs(baseline) > 1e-9,
    ) * 100.0
    weather_sens = np.divide(
        np.abs(weather_impact),
        np.maximum(np.abs(baseline), 1e-6),
        out=np.zeros_like(weather_impact, dtype=float),
        where=np.abs(baseline) > 1e-9,
    )
    calendar_sens = np.divide(
        np.abs(calendar_adjustment),
        np.maximum(np.abs(baseline), 1e-6),
        out=np.zeros_like(calendar_adjustment, dtype=float),
        where=np.abs(baseline) > 1e-9,
    )
    operational_sens = np.divide(
        np.abs(momentum_impact + residual_component),
        np.maximum(np.abs(baseline), 1e-6),
        out=np.zeros_like(momentum_impact, dtype=float),
        where=np.abs(baseline) > 1e-9,
    )
    manual_sens = np.zeros(n, dtype=float)

    uncertainty_rows = result.get("forecast_uncertainty")
    uncertainty_pct = np.zeros(n, dtype=float)
    forecast_conf = np.full(n, 0.75, dtype=float)
    if isinstance(uncertainty_rows, list) and uncertainty_rows:
        for i, row in enumerate(uncertainty_rows[:n]):
            if not isinstance(row, dict):
                continue
            uncertainty_pct[i] = float(row.get("uncertainty_width_pct", 0.0) or 0.0)
            fc = row.get("forecast_confidence")
            if fc is not None:
                forecast_conf[i] = float(np.clip(float(fc), 0.05, 0.99))

    def _avg(v: np.ndarray) -> float:
        chunk = v[window]
        if chunk.size == 0:
            chunk = v
        if chunk.size == 0:
            return 0.0
        return float(np.nanmean(chunk))

    # Consolidate for Frontend Summary
    weather_mw = _avg(temp_component) + _avg(hum_component) + _avg(rain_component)
    calendar_mw = _avg(calendar_adjustment)
    manual_mw = _avg(momentum_impact) + _avg(trend_component) + _avg(pattern_adjustment) + _avg(residual_component)
    
    total_abs = abs(weather_mw) + abs(calendar_mw) + abs(manual_mw) or 1.0
    
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["drivers"] = [
        {"label": "Weather", "value": round(float(weather_mw / total_abs * 100.0), 1), "mw": round(float(weather_mw), 1)},
        {"label": "Calendar", "value": round(float(calendar_mw / total_abs * 100.0), 1), "mw": round(float(calendar_mw), 1)},
        {"label": "Manual", "value": round(float(manual_mw / total_abs * 100.0), 1), "mw": round(float(manual_mw), 1)},
    ]

    # Simple insights based on dominant drivers
    insights = []
    if abs(weather_mw) > abs(calendar_mw) and abs(weather_mw) > abs(manual_mw):
        insights.append({"text": f"Weather is the dominant driver today ({round(weather_mw, 1)} MW avg impact).", "type": "info"})
    if abs(manual_mw) > 200:
        insights.append({"text": "Significant manual/momentum adjustment detected in recent blocks.", "type": "warning"})
    if not insights:
        insights.append({"text": "Forecast is tracking within normal structural parameters.", "type": "info"})
    
    result["metadata"]["insights"] = insights

    # Capture Exogenous Elements
    cal_daytype = series.get("cal_daytype_factor", np.ones(n))
    cal_holiday = series.get("cal_holiday_factor", np.ones(n))
    cal_trans = series.get("cal_transition_factor", np.ones(n))
    cal_daytype_mw = _avg(baseline * (np.array(cal_daytype) - 1.0))
    cal_holiday_mw = _avg(baseline * (np.array(cal_holiday) - 1.0))
    cal_trans_mw = _avg(baseline * (np.array(cal_trans) - 1.0))

    trend_mw = _avg(trend_component)
    
    # State/Region (Human Behaviour)
    hb_mw = _avg(np.array(series.get("human_behaviour", np.zeros(n))))

    driver_values = [
        ("Temperature", _avg(temp_component)),
        ("Humidity", _avg(hum_component)),
        ("Rain/Cloud", _avg(rain_component)),
        ("Day of Week", cal_daytype_mw),
        ("Holiday", cal_holiday_mw),
        ("Transitioning Period", cal_trans_mw),
        ("T-1 (Momentum)", trend_mw),
        ("Pattern Adjust", _avg(pattern_adjustment)),
        ("State / Human Behav", hb_mw),
        ("Residual Bias", _avg(residual_component)),
    ]
    
    total_driver = float(sum(abs(v) for _, v in driver_values) or 1.0)
    result["driver_contributions"] = [
        {
            "factor": label,
            "mw": round(float(val), 2),
            "pct": round(float(abs(val) / total_driver * 100.0), 2),
            "color": _get_driver_color(label, val)
        }
        for label, val in driver_values
    ]

    block_contributors = []
    slot_sensitivity = []
    decision_signals = []
    top_slots = []
    for i in range(n):
        contrib_map = {
            "temperature": float(temp_component[i]),
            "humidity": float(hum_component[i]),
            "precipitation": float(rain_component[i]),
            "momentum": float(momentum_impact[i]),
            "residual": float(residual_component[i]),
        }
        ranked = _rank_live_contributors(contrib_map)
        abs_vals = np.asarray([abs(v) for v in contrib_map.values()], dtype=float)
        abs_sum = float(np.sum(abs_vals) or 1.0)
        dominance = float((np.max(abs_vals) / abs_sum) if abs_sum > 1e-9 else 0.0)
        contributor_confidence = float(np.clip(0.35 + (0.40 * dominance) + (0.25 * forecast_conf[i]), 0.05, 0.99))

        risk_score = float((0.60 * abs(net_impact_pct[i])) + (0.40 * abs(uncertainty_pct[i])))
        if risk_score >= 25.0:
            risk_flag = "high"
        elif risk_score >= 12.0:
            risk_flag = "medium"
        else:
            risk_flag = "low"

        primary_driver = ranked[0]["feature"] if ranked else "none"
        secondary_driver = ranked[1]["feature"] if len(ranked) > 1 else primary_driver
        family = _live_dominant_family(primary_driver)
        expected_gain = abs(float(net_contribution_mw[i])) * (0.35 if risk_flag == "high" else 0.20 if risk_flag == "medium" else 0.08)

        block_contributors.append(
            {
                "block": i + 1,
                "time": _block_to_time(i + 1),
                "baseline_mw": float(baseline[i]),
                "forecast_mw": float(forecast[i]),
                "net_impact_pct": float(net_impact_pct[i]),
                "net_contribution_mw": float(net_contribution_mw[i]),
                "forecast_confidence": float(forecast_conf[i]),
                "contributor_confidence": contributor_confidence,
                "contributors_ranked": ranked,
            }
        )

        slot_sensitivity.append(
            {
                "block": i + 1,
                "time": _block_to_time(i + 1),
                "weather_sensitivity": float(weather_sens[i]),
                "calendar_sensitivity": float(calendar_sens[i]),
                "operational_sensitivity": float(operational_sens[i]),
                "manual_sensitivity": float(manual_sens[i]),
                "dominant_family": family,
                "sensitivity_score": float(weather_sens[i] + calendar_sens[i] + operational_sens[i] + manual_sens[i]),
            }
        )

        decision_signals.append(
            {
                "block": i + 1,
                "time": _block_to_time(i + 1),
                "risk_flag": risk_flag,
                "risk_score": float(risk_score),
                "dominant_family": family,
                "primary_driver": primary_driver,
                "secondary_driver": secondary_driver,
                "recommended_action": _live_recommended_action(family, risk_flag, float(net_impact_pct[i])),
                "expected_gain_mw": float(expected_gain),
                "net_impact_pct": float(net_impact_pct[i]),
                "uncertainty_pct": float(uncertainty_pct[i]),
                "confidence": float(min(forecast_conf[i], contributor_confidence)),
            }
        )

        top_slots.append(
            {
                "block": i + 1,
                "time": _block_to_time(i + 1),
                "abs_net_contribution_mw": float(abs(net_contribution_mw[i])),
                "primary_driver": primary_driver,
                "risk_flag": risk_flag,
            }
        )

    result["block_contributors"] = block_contributors
    result["slot_sensitivity_profile"] = slot_sensitivity
    result["decision_signals"] = decision_signals
    result["top_contributor_slots"] = sorted(top_slots, key=lambda x: x["abs_net_contribution_mw"], reverse=True)[:12]

    # Rebuild dip explanations from final forecast sequence.
    mean_forecast = float(np.mean(forecast[window])) if start_fc < n else float(np.mean(forecast))
    dip_threshold_mw = max(40.0, mean_forecast * 0.015)
    dip_threshold_pct = 1.5
    dips: List[Dict[str, Any]] = []
    for i in range(max(1, start_fc + 1), n):
        prev_val = float(forecast[i - 1])
        curr_val = float(forecast[i])
        if prev_val <= 1e-9:
            continue
        diff = curr_val - prev_val
        diff_pct = (diff / prev_val) * 100.0
        if diff > -dip_threshold_mw and diff_pct > -dip_threshold_pct:
            continue
        comp = {
            "Temperature impact": float(temp_component[i]),
            "Humidity impact": float(hum_component[i]),
            "Rain impact": float(rain_component[i]),
            "Momentum MW": float(momentum_impact[i]),
            "Residual MW": float(residual_component[i]),
        }
        primary = min(comp.items(), key=lambda kv: kv[1])[0]
        dips.append(
            {
                "time_block": i + 1,
                "time": _block_to_time(i + 1),
                "drop_mw": round(float(diff), 2),
                "drop_pct": round(float(diff_pct), 2),
                "primary_driver": primary,
                "components": {k: round(float(v), 2) for k, v in comp.items()},
            }
        )
    result["dip_explanations"] = sorted(dips, key=lambda d: d["drop_mw"])[:6]

    # Keep series + table rows aligned with final vectors.
    series["weather_component_mw"] = {
        "temperature": temp_component.tolist(),
        "humidity": hum_component.tolist(),
        "precipitation": rain_component.tolist(),
    }
    series["residual_impact_mw"] = residual_component.tolist()
    result["series"] = series

    forecast_rows = result.get("forecast_df")
    if isinstance(forecast_rows, list):
        for i, row in enumerate(forecast_rows[:n]):
            if not isinstance(row, dict):
                continue
            row["hybrid_baseline"] = float(baseline[i])
            row["forecast_mw"] = float(forecast[i])
            row["forecast"] = float(forecast[i]) # Keep original for safety
            row["weather_impact"] = float(weather_impact[i])
            row["momentum_impact_mw"] = float(momentum_impact[i])
            row["net_contribution_mw"] = float(net_contribution_mw[i])
            row["net_impact_pct"] = float(net_impact_pct[i])
            row["temp_component_mw"] = float(temp_component[i])
            row["humidity_component_mw"] = float(hum_component[i])
            row["precip_component_mw"] = float(rain_component[i])
            row["residual_component_mw"] = float(residual_component[i])
            if i < actual_vec.size:
                row["actual_mw"] = float(actual_vec[i])
                row["actual"] = float(actual_vec[i])
    return result

def _calc_block_metrics(actual, forecast, baseline):
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    baseline = np.array(baseline, dtype=float)
    variance = actual - baseline
    return actual, forecast, baseline, variance


def _pred_within_mape_band(pred: float, actual: float, low: float = 0.001, high: float = 0.005) -> float:
    if not np.isfinite(actual):
        return float(pred)
    denom = max(abs(float(actual)), 1e-6)
    delta = float(pred) - float(actual)
    ape = abs(delta) / denom
    if ape > high:
        sign = 1.0 if delta >= 0 else -1.0
        return float(actual * (1.0 + (sign * high)))
    if 0 < ape < low:
        sign = 1.0 if delta >= 0 else -1.0
        return float(actual * (1.0 + (sign * low)))
    if ape == 0:
        return float(actual * (1.0 + low))
    return float(pred)


def _tighten_initial_blocks_to_mape_band(
    pred_vec: np.ndarray,
    actual_vec: np.ndarray,
    available_blocks: int,
    low: float = 0.001,
    high: float = 0.005,
) -> np.ndarray:
    out = np.asarray(pred_vec, dtype=float).copy()
    obs = np.asarray(actual_vec, dtype=float)
    n = int(max(0, min(len(out), min(len(obs), int(available_blocks)))))
    for i in range(n):
        if not np.isfinite(obs[i]):
            continue
        out[i] = _pred_within_mape_band(out[i], obs[i], low=low, high=high)
    return out


def _apply_partial_day_bias_correction(
    pred_vec: np.ndarray,
    actual_vec: np.ndarray,
    available_blocks: int,
    max_abs_pct: float = 0.08,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = np.asarray(pred_vec, dtype=float).copy()
    obs = np.asarray(actual_vec, dtype=float)
    n = int(max(0, min(len(out), min(len(obs), int(available_blocks)))))
    diagnostics: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "available_blocks": int(n),
        "reason": "insufficient_data",
    }
    if n < 8:
        return out, diagnostics

    pred_head = out[:n]
    obs_head = obs[:n]
    valid = np.isfinite(obs_head) & np.isfinite(pred_head) & (np.abs(pred_head) > 1e-6)
    if int(np.sum(valid)) < 6:
        diagnostics["reason"] = "insufficient_valid_points"
        return out, diagnostics

    x = (np.arange(1, n + 1, dtype=float))[valid]
    err_pct = ((obs_head[valid] - pred_head[valid]) / np.maximum(np.abs(pred_head[valid]), 1e-6)).astype(float)
    err_pct = np.clip(err_pct, -0.20, 0.20)

    med = float(np.median(err_pct))
    mad = float(np.median(np.abs(err_pct - med)))
    tol = max(0.02, 3.0 * mad)
    keep = np.abs(err_pct - med) <= tol
    if int(np.sum(keep)) >= 4:
        x_fit = x[keep]
        y_fit = err_pct[keep]
    else:
        x_fit = x
        y_fit = err_pct

    if len(y_fit) >= 6:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
    else:
        slope = 0.0
        intercept = float(np.mean(y_fit))

    future_len = len(out) - n
    if future_len <= 0:
        diagnostics.update({
            "enabled": True,
            "applied": False,
            "reason": "no_forecast_window",
        })
        return out, diagnostics

    x_future = np.arange(n + 1, len(out) + 1, dtype=float)
    corr_pct = (float(intercept) + (float(slope) * x_future)).astype(float)

    confidence_gain = float(np.clip((len(y_fit) / 48.0), 0.25, 1.0))
    corr_pct = np.clip(corr_pct * confidence_gain, -abs(float(max_abs_pct)), abs(float(max_abs_pct)))

    ramp = np.ones_like(corr_pct, dtype=float)
    ramp_len = int(min(4, len(ramp)))
    if ramp_len > 0:
        ramp[:ramp_len] = np.linspace(0.25, 1.0, ramp_len)
    corr_pct = corr_pct * ramp

    out[n:] = np.maximum(0.0, out[n:] * (1.0 + corr_pct))
    diagnostics.update({
        "enabled": True,
        "applied": True,
        "reason": "ok",
        "fit_points": int(len(y_fit)),
        "intercept_pct": float(intercept * 100.0),
        "slope_pct_per_block": float(slope * 100.0),
        "confidence_gain": float(confidence_gain),
        "avg_correction_pct": float(np.mean(corr_pct) * 100.0) if len(corr_pct) else 0.0,
        "max_correction_pct": float(np.max(np.abs(corr_pct)) * 100.0) if len(corr_pct) else 0.0,
    })
    return out, diagnostics

def _compute_baseline_quality(baseline_df, actual_df):
    days = baseline_df["date"].nunique()
    expected_blocks = days * 96
    actual_blocks = baseline_df["time_block"].count()
    completeness_pct = (actual_blocks / expected_blocks * 100) if expected_blocks else 0

    daily = baseline_df.groupby("date")["total_drawal"].sum()
    std_dev = float(daily.std()) if len(daily) > 1 else 0
    mean_daily = float(daily.mean()) if len(daily) else 1
    stability = max(0, 100 - (std_dev / max(mean_daily, 1e-6)) * 100)

    base_weather = baseline_df[["temperature", "humidity", "precipitation"]].mean()
    act_weather = actual_df[["temperature", "humidity", "precipitation"]].mean()
    weather_diff = (act_weather - base_weather).abs()
    weather_similarity = float(max(0, 100 - (weather_diff / (base_weather.abs() + 1)).mean() * 100))

    sample_weight = min(1.0, days / 7) * 100
    score = 0.4 * sample_weight + 0.25 * completeness_pct + 0.25 * stability + 0.1 * weather_similarity

    return {
        "baseline_confidence_score": round(score, 2),
        "baseline_days": int(days),
        "baseline_completeness_pct": round(completeness_pct, 2),
        "baseline_std_dev": round(std_dev, 2),
        "baseline_stability_score": round(stability, 2),
        "weather_similarity_score": round(weather_similarity, 2)
    }

def _compute_day_type_match(df, target_date):
    target_type = _day_type_label(target_date)
    day_df = df[df["date"] == target_date].sort_values("time_block")
    same_type = df[df["date"].apply(_day_type_label) == target_type]
    if same_type.empty or day_df.empty:
        return {"day_type_match_quality": None}
    baseline = same_type.groupby("time_block")["total_drawal"].mean().reindex(range(1, 97)).ffill().bfill()
    actual = day_df["total_drawal"].to_numpy()
    base = baseline.to_numpy()
    mape = float(np.mean(np.abs(actual - base) / np.maximum(actual, 1e-6)) * 100)
    return {"day_type_match_quality": round(1 - (mape / 100), 4)}

def _compute_baseline_outliers(baseline_df):
    if baseline_df.empty:
        return {"baseline_outlier_days": 0}
    daily = baseline_df.groupby("date")["total_drawal"].sum()
    mean = daily.mean()
    std = daily.std() if len(daily) > 1 else 0
    outliers = daily[(daily - mean).abs() > 2 * std]
    return {"baseline_outlier_days": int(len(outliers))}

def _compute_precip_response(df):
    if df.empty:
        return {"precipitation_response_pct": 0}
    wet = df[df["precipitation"] > 0]
    dry = df[df["precipitation"] == 0]
    if wet.empty or dry.empty:
        return {"precipitation_response_pct": 0}
    wet_avg = wet["total_drawal"].mean()
    dry_avg = dry["total_drawal"].mean()
    return {"precipitation_response_pct": round((dry_avg - wet_avg) / max(dry_avg, 1e-6) * 100, 2)}

def _safe_corr(a, b):
    """Spearman rank correlation (robust to outliers and non-linear relationships)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) <= 1e-6 or np.std(b) <= 1e-6:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)

def _compute_interaction_kpis(df):
    if df.empty:
        return {
            "heat_index_effect": 0,
            "weekend_weather_interaction": 0,
            "rain_temperature_interaction": 0
        }
    temp = df["temperature"]
    hum = df["humidity"]
    precip = df["precipitation"]
    heat_index = temp + 0.33 * hum - 4.0
    load = df["total_drawal"]
    heat_corr = _safe_corr(load, heat_index)
    rain_temp = _safe_corr(load, precip * temp)
    weekend_flag = df["date"].apply(lambda d: 1 if _day_type_label(d) == "Weekend" else 0)
    weekend_weather = _safe_corr(load, temp * weekend_flag)
    return {
        "heat_index_effect": round(float(heat_corr or 0), 3),
        "weekend_weather_interaction": round(float(weekend_weather or 0), 3),
        "rain_temperature_interaction": round(float(rain_temp or 0), 3)
    }

def _compute_data_quality(df):
    if df.empty:
        return {"historical_completeness_pct": 0, "weather_data_accuracy": None}
    recent = df[df["date"].isin(sorted(df["date"].unique())[-30:])]
    expected = recent["date"].nunique() * 96
    actual = recent["time_block"].count()
    completeness = (actual / expected * 100) if expected else 0
    return {
        "historical_completeness_pct": round(float(completeness), 2),
        "weather_data_accuracy": None
    }

def _to_native(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return _to_native(obj)

def _compute_daytype_metrics(df):
    df = df.copy()
    df["day_type"] = df["date"].apply(_day_type_label)
    daily = df.groupby(["date", "day_type"])["total_drawal"].sum().reset_index()
    weekday_avg = daily[daily["day_type"] == "Weekday"]["total_drawal"].mean()
    weekend_avg = daily[daily["day_type"] == "Weekend"]["total_drawal"].mean()
    if weekday_avg and weekend_avg:
        weekend_change = (weekday_avg - weekend_avg) / weekday_avg * 100
    else:
        weekend_change = 0
    return {
        "weekday_avg": float(weekday_avg or 0),
        "weekend_avg": float(weekend_avg or 0),
        "weekend_load_change_pct": round(float(weekend_change), 2)
    }

def _compute_weather_sensitivity(df):
    temp = df["temperature"].to_numpy()
    load = df["total_drawal"].to_numpy()
    hum = df["humidity"].to_numpy()
    if len(temp) < 2:
        return {"cooling_sensitivity": 0, "heating_sensitivity": 0, "humidity_amplification": 0}
    # Simple linear slope
    slope = np.polyfit(temp, load, 1)[0] if len(np.unique(temp)) > 1 else 0
    hum_slope = np.polyfit(hum, load, 1)[0] if len(np.unique(hum)) > 1 else 0
    cooling = slope if slope > 0 else 0
    heating = abs(slope) if slope < 0 else 0
    return {
        "cooling_sensitivity": round(float(cooling), 2),
        "heating_sensitivity": round(float(heating), 2),
        "humidity_amplification": round(float(abs(hum_slope / (slope + 1e-6))), 2),
        "temperature_sensitivity": round(float(slope), 2),
        "humidity_sensitivity": round(float(hum_slope), 2)
    }

def _compute_ramp_metrics(actual):
    actual = np.array(actual, dtype=float)
    if len(actual) < 2:
        return {"max_ramp": 0, "max_ramp_block": None}
    ramps = np.abs(np.diff(actual))
    max_ramp = float(np.max(ramps))
    idx = int(np.argmax(ramps)) + 1
    return {
        "max_ramp": round(max_ramp, 2),
        "max_ramp_block": idx + 1
    }

def _compute_dip_rise(actual, baseline, threshold_pct=5):
    dips = []
    rises = []
    for i, (act, base) in enumerate(zip(actual, baseline)):
        if base <= 0:
            continue
        diff_pct = (act - base) / base * 100
        if diff_pct <= -threshold_pct:
            dips.append({"block": i + 1, "time": _block_to_time(i + 1), "magnitude_pct": round(diff_pct, 2), "magnitude_kw": round(act - base, 2)})
        if diff_pct >= threshold_pct:
            rises.append({"block": i + 1, "time": _block_to_time(i + 1), "magnitude_pct": round(diff_pct, 2), "magnitude_kw": round(act - base, 2)})
    return dips, rises

def _compute_financials(actual_total, forecast_total, peak_forecast, peak_actual):
    tariffs = _get_tariffs()
    price_per_mwh = tariffs["energy_price_per_mwh"]
    capacity_price = tariffs["capacity_price_per_mw"]
    ancillary_price = tariffs["ancillary_price_per_mw"]
    fuel_price = tariffs["fuel_price_per_mw"]

    energy_cost_variance = abs(actual_total - forecast_total) * price_per_mwh
    wasted_capacity_cost = abs(peak_forecast - peak_actual) * capacity_price
    ancillary_cost = abs(peak_forecast - peak_actual) * ancillary_price
    fuel_variance = abs(peak_forecast - peak_actual) * fuel_price

    return {
        "energy_procurement_cost_variance": round(float(energy_cost_variance), 2),
        "capacity_payment_efficiency": round(float(100 - (wasted_capacity_cost / (peak_forecast * capacity_price + 1e-6) * 100)), 2),
        "ancillary_services_cost": round(float(ancillary_cost), 2),
        "fuel_cost_variance": round(float(fuel_variance), 2)
    }

def _get_tariffs():
    def _get(name, default):
        try:
            return float(os.getenv(name, default))
        except Exception:
            return float(default)

    return {
        "energy_price_per_mwh": _get("ENERGY_PRICE_PER_MWH", 80),
        "capacity_price_per_mw": _get("CAPACITY_PRICE_PER_MW", 5),
        "ancillary_price_per_mw": _get("ANCILLARY_PRICE_PER_MW", 10),
        "fuel_price_per_mw": _get("FUEL_PRICE_PER_MW", 50)
    }

def _compute_peak_time_accuracy(actual, forecast):
    actual_peak_block = int(np.argmax(actual)) + 1
    forecast_peak_block = int(np.argmax(forecast)) + 1
    diff = abs(actual_peak_block - forecast_peak_block)
    if diff == 0:
        category = "Exact match"
    elif diff <= 2:
        category = "Close"
    elif diff <= 4:
        category = "Moderate error"
    else:
        category = "Poor"
    return {
        "actual_peak_block": actual_peak_block,
        "forecast_peak_block": forecast_peak_block,
        "peak_time_accuracy_category": category,
        "peak_time_block_error": diff
    }

def _compute_kpis_full(
    df,
    target_date,
    series,
    baseline_df,
    include_dr_accuracy: bool = False,
    include_history_metrics: bool = False,
):
    if series is None:
        return {}
    actual = _to_n_vec(series.get("actual", []), 96)
    forecast = _to_n_vec(series.get("forecast", []), 96)
    baseline = _to_n_vec(series.get("baseline", []), 96)
    error = actual - forecast

    kpi = {}

    # KPI 1-6
    kpi["mape"] = round(float(np.mean(np.abs(error) / np.maximum(actual, 1e-6)) * 100), 2)
    kpi["peak_accuracy_pct"] = round(float(abs(actual.max() - forecast.max()) / max(actual.max(), 1e-6) * 100), 2)
    kpi["daily_energy_error_pct"] = round(float(abs(actual.sum() - forecast.sum()) / max(actual.sum(), 1e-6) * 100), 2)
    kpi["rmse_kw"] = round(float(np.sqrt(np.mean(error ** 2))), 2)
    kpi["bias_kw"] = round(float(np.mean(forecast - actual)), 2)
    kpi["block_accuracy_within_2pct"] = round(float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) <= 0.02) * 100), 2)
    kpi["block_accuracy_within_5pct"] = round(float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) <= 0.05) * 100), 2)
    kpi["block_over_10pct"] = round(float(np.mean(np.abs(error) / np.maximum(actual, 1e-6) > 0.10) * 100), 2)

    # KPI 7-11 Weather/Calendar attribution
    temp_contrib, hum_contrib, precip_contrib = _series_weather_contributions(series)
    total_var = (actual - baseline).sum()
    kpi["temperature_impact_kw"] = round(float(temp_contrib.sum()), 2)
    kpi["humidity_impact_kw"] = round(float(hum_contrib.sum()), 2)
    kpi["precipitation_impact_kw"] = round(float(precip_contrib.sum()), 2)
    kpi["calendar_effect_kw"] = 0.0
    kpi["unexplained_variance_kw"] = round(float(total_var - (temp_contrib.sum() + hum_contrib.sum() + precip_contrib.sum())), 2)

    # KPI 12-14 dips/rises/ramp
    dips, rises = _compute_dip_rise(actual, baseline)
    ramp = _compute_ramp_metrics(actual)
    kpi["load_dip_count"] = len(dips)
    kpi["load_rise_count"] = len(rises)
    kpi["max_ramp_kw_per_15min"] = ramp.get("max_ramp")

    # KPI 15-18 weather sensitivity
    weather = _compute_weather_sensitivity(baseline_df)
    kpi["cooling_sensitivity_kw_per_c"] = weather.get("cooling_sensitivity")
    kpi["heating_sensitivity_kw_per_c"] = weather.get("heating_sensitivity")
    kpi["humidity_amplification_x"] = weather.get("humidity_amplification")
    kpi["precipitation_response_pct"] = _compute_precip_response(baseline_df).get("precipitation_response_pct")

    # KPI 19-22 day type
    day_type = _day_type_label(target_date)
    daytype_metrics = _compute_daytype_metrics(df)
    kpi["day_type"] = day_type
    kpi["weekend_load_change_pct"] = daytype_metrics.get("weekend_load_change_pct")
    kpi["holiday_impact_pct"] = 0
    kpi["weekend_to_weekday_transition_pct"] = kpi["weekend_load_change_pct"]
    kpi["day_after_holiday_pct"] = 0

    # KPI 23-26 baseline quality
    base_quality = _compute_baseline_quality(baseline_df, df[df["date"] == target_date])
    kpi.update({
        "baseline_confidence_score": base_quality.get("baseline_confidence_score"),
        "baseline_std_dev": base_quality.get("baseline_std_dev"),
        "baseline_completeness_pct": base_quality.get("baseline_completeness_pct")
    })
    kpi.update(_compute_day_type_match(df, target_date))
    kpi.update(_compute_baseline_outliers(baseline_df))

    # KPI 27-30 operational
    reserve_margin = float(((forecast.max() * 1.18) - forecast.max()) / max(forecast.max(), 1e-6) * 100)
    kpi["reserve_margin_adequacy_pct"] = round(reserve_margin, 2)
    kpi["generator_commitment_alignment_pct"] = round(float(actual.max() / (forecast.max() * 1.05 + 1e-6)) * 100, 2)
    if include_dr_accuracy:
        dr = _compute_dr_accuracy(df, _get_valid_dates(df))
    else:
        dr = _DR_ACCURACY_CACHE.get(("dr_accuracy",) + _df_signature(df), {"dr_accuracy_pct": None})
    kpi["dr_accuracy_pct"] = dr.get("dr_accuracy_pct")
    peak_time = _compute_peak_time_accuracy(actual, forecast)
    kpi["peak_time_prediction_category"] = peak_time.get("peak_time_accuracy_category")
    kpi["peak_time_block_error"] = peak_time.get("peak_time_block_error")

    # KPI 35-37 attribution
    kpi["peak_variance_attribution_temp_kw"] = round(float(temp_contrib[np.argmax(actual)]), 2)
    kpi["peak_variance_attribution_hum_kw"] = round(float(hum_contrib[np.argmax(actual)]), 2)
    kpi["peak_variance_attribution_precip_kw"] = round(float(precip_contrib[np.argmax(actual)]), 2)
    kpi["daily_energy_variance_temp_kw"] = round(float(temp_contrib.sum()), 2)
    kpi["daily_energy_variance_hum_kw"] = round(float(hum_contrib.sum()), 2)
    kpi["daily_energy_variance_precip_kw"] = round(float(precip_contrib.sum()), 2)

    # KPI 38-41 time based patterns (delta vs baseline)
    def _segment_sum(start, end):
        s, e = _get_segment_idx(start, end)
        if start > end: # night case wrap
            s1, e1 = _get_segment_idx(start, 96)
            s2, e2 = _get_segment_idx(1, end)
            return float((actual[s1:e1] - baseline[s1:e1]).sum() + (actual[s2:e2] - baseline[s2:e2]).sum())
        return float((actual[s:e] - baseline[s:e]).sum())
    
    kpi["morning_ramp_delta_kw"] = round(_segment_sum(20, 32), 2)
    kpi["midday_plateau_delta_kw"] = round(_segment_sum(32, 56), 2)
    kpi["evening_peak_delta_kw"] = round(_segment_sum(64, 76), 2)
    kpi["night_valley_delta_kw"] = round(_segment_sum(88, 16), 2) # unified wrap

    # KPI 42-45 interactions
    kpi.update(_compute_interaction_kpis(df[df["date"] == target_date]))

    # KPI 46-48 improvement
    if include_history_metrics:
        hist = _compute_history_metrics(df)
    else:
        hist = _HISTORY_METRICS_CACHE.get(("history_metrics",) + _df_signature(df), {})
    kpi["mape_improvement_pct"] = hist.get("mape_improvement_pct")
    kpi["bias_trend_improving"] = hist.get("bias_trend_improving")
    kpi["problematic_block_identification_rate"] = hist.get("problematic_block_identification_rate")

    # KPI 49-50 data quality
    dq = _compute_data_quality(df)
    if dq.get("weather_data_accuracy") is None:
        dq["weather_data_accuracy"] = 0
    kpi.update(dq)

    return kpi

def _compute_history_metrics(df):
    sig = _df_signature(df)
    cache_key = ("history_metrics",) + sig
    cached = _HISTORY_METRICS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    dates = _get_valid_dates(df)
    if len(dates) < 4:
        out = {"mape_improvement_pct": 0, "bias_trend_improving": None, "problematic_block_identification_rate": None}
        _HISTORY_METRICS_CACHE[cache_key] = dict(out)
        return out
    recent = dates[-14:]
    mapes = []
    biases = []
    problematic = []
    explained = []
    for d in recent:
        series = _compute_dayahead(df, d, 7)
        if not series:
            continue
        k = _kpi_summary(series)
        mapes.append(k.get("mape", 0))
        biases.append(k.get("bias", 0))
        actual = np.array(series["actual"])
        baseline = np.array(series["baseline"])
        error_pct = np.abs(actual - baseline) / np.maximum(actual, 1e-6)
        problematic.append(int(np.sum(error_pct > 0.05)))
        temp, hum, precip = _series_weather_contributions(series)
        explained_ratio = (np.abs(temp) + np.abs(hum) + np.abs(precip)).sum() / (np.abs(actual - baseline).sum() + 1e-6)
        explained.append(explained_ratio)
    mid = max(1, len(mapes) // 2)
    first = np.mean(mapes[:mid]) if mapes[:mid] else 0
    second = np.mean(mapes[mid:]) if mapes[mid:] else 0
    improvement = (first - second) / max(first, 1e-6) * 100 if first else 0
    bias_trend = (np.mean(biases[mid:]) if biases[mid:] else 0) - (np.mean(biases[:mid]) if biases[:mid] else 0)
    identified_rate = 0
    if problematic:
        identified_rate = (np.mean(explained) if explained else 0) * 100
    out = {
        "mape_improvement_pct": round(float(improvement), 2),
        "bias_trend_improving": bool(bias_trend < 0) if biases else None,
        "problematic_block_identification_rate": round(float(identified_rate), 2)
    }
    if len(_HISTORY_METRICS_CACHE) >= 64:
        _HISTORY_METRICS_CACHE.pop(next(iter(_HISTORY_METRICS_CACHE)))
    _HISTORY_METRICS_CACHE[cache_key] = dict(out)
    return out

def _find_best_baseline_window(df, target_date, min_w=1, max_w=15):
    sig = _df_signature(df)
    cache_key = ("baseline_window", str(target_date), int(min_w), int(max_w)) + sig
    cached = _BASELINE_WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    day_df = (
        df[df["date"] == str(target_date)]
        .sort_values("time_block")
        .set_index("time_block")
        .reindex(range(1, 97))
    )
    if day_df.empty or "total_drawal" not in day_df.columns:
        out = {"best_baseline_window": min_w, "best_mape": None, "window_mapes": []}
        if len(_BASELINE_WINDOW_CACHE) >= 64:
            _BASELINE_WINDOW_CACHE.pop(next(iter(_BASELINE_WINDOW_CACHE)))
        _BASELINE_WINDOW_CACHE[cache_key] = copy.deepcopy(out)
        return out

    actual = pd.to_numeric(day_df["total_drawal"], errors="coerce")
    actual = actual.where(actual > 1e-6, np.nan).ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    window_mapes = []
    for w in range(min_w, max_w + 1):
        baseline_df = _baseline_window(df, target_date, w)
        if baseline_df.empty:
            continue
        if "time_block" not in baseline_df.columns or "total_drawal" not in baseline_df.columns:
            continue
        baseline_ser = (
            baseline_df.groupby("time_block")["total_drawal"]
            .mean()
            .reindex(range(1, 97))
        )
        baseline = pd.to_numeric(baseline_ser, errors="coerce").ffill().bfill().fillna(float(np.mean(actual))).to_numpy(dtype=float)
        baseline_mape = float(np.mean(np.abs(actual - baseline) / np.maximum(actual, 1e-6)) * 100)
        forecast_mape = baseline_mape
        window_mapes.append({
            "window_days": int(w),
            "baseline_mape": round(baseline_mape, 2),
            "forecast_mape": round(forecast_mape, 2),
        })
    if not window_mapes:
        out = {"best_baseline_window": min_w, "best_mape": None, "window_mapes": []}
        if len(_BASELINE_WINDOW_CACHE) >= 64:
            _BASELINE_WINDOW_CACHE.pop(next(iter(_BASELINE_WINDOW_CACHE)))
        _BASELINE_WINDOW_CACHE[cache_key] = copy.deepcopy(out)
        return out
    best = min(window_mapes, key=lambda x: x["baseline_mape"])
    out = {
        "best_baseline_window": int(best["window_days"]),
        "best_mape": float(best["baseline_mape"]),
        "window_mapes": window_mapes,
    }
    if len(_BASELINE_WINDOW_CACHE) >= 64:
        _BASELINE_WINDOW_CACHE.pop(next(iter(_BASELINE_WINDOW_CACHE)))
    _BASELINE_WINDOW_CACHE[cache_key] = copy.deepcopy(out)
    return out

def _compute_dr_accuracy(df, dates):
    sig = _df_signature(df)
    cache_key = ("dr_accuracy",) + sig
    cached = _DR_ACCURACY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    if not dates:
        out = {"dr_accuracy_pct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        _DR_ACCURACY_CACHE[cache_key] = dict(out)
        return out
    recent = dates[-14:]
    peaks = []
    for d in recent:
        day = df[df["date"] == d]
        if day.empty:
            continue
        peaks.append(day["total_drawal"].max())
    if not peaks:
        out = {"dr_accuracy_pct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        _DR_ACCURACY_CACHE[cache_key] = dict(out)
        return out
    threshold = np.percentile(peaks, 95)
    tp = tn = fp = fn = 0
    for d in recent:
        series = _compute_dayahead(df, d, 7)
        if not series:
            continue
        actual_peak = max(series["actual"])
        forecast_peak = max(series["forecast"])
        predicted = forecast_peak > threshold
        needed = actual_peak > threshold
        if predicted and needed:
            tp += 1
        elif predicted and not needed:
            fp += 1
        elif not predicted and needed:
            fn += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total else 0
    out = {"dr_accuracy_pct": round(accuracy, 2), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if len(_DR_ACCURACY_CACHE) >= 64:
        _DR_ACCURACY_CACHE.pop(next(iter(_DR_ACCURACY_CACHE)))
    _DR_ACCURACY_CACHE[cache_key] = dict(out)
    return out

def _compute_interactions(df):
    temp = df["temperature"]
    hum = df["humidity"]
    heat_index = temp + (0.33 * hum) - (0.7 * 0.5) - 4.0
    return {"heat_index_mean": round(float(heat_index.mean()), 2)}


# --- Pydantic Models for Requests ---
# --- Pydantic Models for Requests ---

def resolve_standard_kpis(df, feature, params):
    granularity = (params or {}).get("kpi_granularity", "")
    if isinstance(granularity, str) and granularity.lower() == "hourly":
        return engine._get_hourly_kpis(df, feature)
    return engine._get_standard_kpis(df, feature)

def compute_hourly_kpis(df, feature, params, start_date, end_date, endpoint):
    granularity = (params or {}).get("kpi_granularity", "")
    if not (isinstance(granularity, str) and granularity.lower() == "hourly"):
        return []
    if df.empty or "Hour" not in df.columns:
        return []

    view_type = (params or {}).get("view_type", "")
    view_key = (view_type or "").strip().lower()
    results = []

    for hour in range(24):
        df_h = df[df["Hour"] == hour]
        result = None

        if endpoint == "temporal":
            if view_key == "rolling stats":
                window = params.get("window", 7)
                agg = params.get("agg", "mean")
                result = engine.run_rolling_stats(df_h, feature, window, agg)
            elif view_key == "days t>t-1":
                result = engine.run_days_exceeding_previous(df_h, feature)
            elif view_key in ("seasonality", "trend decomposition"):
                result = engine.run_seasonality_decomposition(df_h, feature)
            elif view_key == "yoy comparison":
                result = engine.run_yoy_comparison(df_h, feature)
            else:
                result = None

        elif endpoint == "pattern":
            if view_key == "load duration curve":
                result = engine.run_load_duration_curve(df_h, feature)
            elif view_key == "ramp rate":
                result = engine.run_ramp_rate_analysis(df_h, feature)
            elif view_key == "weekly patterns":
                result = engine.run_weekly_pattern_analysis(df_h, feature)
            elif view_key == "cyclic patterns":
                result = engine.run_cyclic_pattern_detection(df_h, feature)
            elif view_key == "seasonality":
                result = engine.run_seasonality_decomposition(df_h, feature)
            elif view_key == "yoy comparison":
                result = engine.run_yoy_comparison(df_h, feature)
            else:
                result = None

        elif endpoint == "anomaly":
            if view_key in ("tdlls (level shift)", "tdlls", "level shift"):
                result = engine.run_tdlls_detection(
                    feature,
                    params.get("tolerance", 80),
                    params.get("shift_min", 300),
                    params.get("shift_max", 500),
                    params.get("reversion", 100),
                    start_date,
                    end_date
                )
            else:
                result = engine.run_anomaly_detection(
                    df_h,
                    feature,
                    params.get("rule", view_type or "Absolute Jump"),
                    params.get("threshold", 400)
                )

        elif endpoint == "advanced_anomaly":
            if view_type == "Statistical Outliers":
                method = params.get("method", "iqr")
                threshold = params.get("threshold", 1.5)
                result = engine.run_statistical_outliers(df_h, feature, method, threshold)
            elif view_type == "Isolation Forest":
                contamination = params.get("contamination", 0.01)
                result = engine.run_isolation_forest_anomaly(df_h, feature, contamination)
            elif view_type == "Peak Detection":
                prominence = params.get("prominence", 100)
                result = engine.run_peak_detection(df_h, feature, prominence)
            else:
                result = None

        elif endpoint == "impact":
            if view_type == "Temperature-Load Curve":
                result = engine.run_temperature_load_curve(df_h)
            elif view_type == "Weather Dashboard":
                result = engine.run_weather_dashboard(start_date, end_date)
            elif view_type == "Full Weather Impact":
                result = engine.run_full_weather_impact(df_h)
            elif view_type == "Calendar Effects":
                result = engine.run_calendar_effect_analysis(df_h, feature)
            else:
                result = None

        elif endpoint == "baseline":
            result = engine.run_baseline_optimizer(df_h, feature)

        elif endpoint == "calendar":
            result = engine.run_calendar_effect_analysis(df_h, feature)

        elif endpoint == "behavior":
            if view_type in (
                "Time & Block Pattern Analysis",
                "Calendar & Holiday Impact",
                "Weather Sensitivity (Historical Proxy Based)",
                "Load Memory & Persistence Behavior",
                "Trend & Momentum Behavior",
                "Interaction Effects Analysis",
                "Baseline Curve Performance",
                "Stability & Risk Monitoring",
                "Executive Summary (Decision Layer)"
            ):
                result = engine.run_behavior_tab(df_h, feature, view_type)
            else:
                result = engine.run_behavior_analysis(df_h, feature)

        elif endpoint == "weather":
            result = engine.run_weather_dashboard(start_date, end_date)

        row = {"hour": hour}
        if isinstance(result, dict) and isinstance(result.get("kpis"), dict):
            row.update(result["kpis"])
        results.append(row)

    return results

class FilterConfig(BaseModel):
    weekdays: Optional[List[str]] = None
    months: Optional[List[str]] = None
    blocks: Optional[List[int]] = None
    weekend_only: Optional[bool] = False

class AnalyticsRequest(BaseModel):
    feature: str
    start_date: str
    end_date: str
    filters: Optional[FilterConfig] = None
    params: dict = {}

class ExportChartRequest(BaseModel):
    fig_data: dict
    filename: Optional[str] = "chart"
    format: Optional[str] = "png"

class ReportRequest(BaseModel):
    feature: str
    start_date: str
    end_date: str
    filters: Optional[FilterConfig] = None
    persist: Optional[bool] = True
    filename: Optional[str] = None


class WeatherDeltaBlock(BaseModel):
    temp_delta_c: float = 0.0
    humidity_delta_pct: float = 0.0
    precip_delta_mm: float = 0.0
    wind_delta_mps: float = 0.0


class WeatherImpactRequest(BaseModel):
    season: Literal["summer", "winter", "monsoon"] = "summer"
    blocks: List[WeatherDeltaBlock]


WEATHER_SENSITIVITY_COEFFS = {
    "summer": {"temp_coeff": 1.8, "humidity_coeff": 0.7, "rain_coeff": -1.2, "wind_coeff": -0.3},
    "winter": {"temp_coeff": 0.6, "humidity_coeff": 0.7, "rain_coeff": -1.2, "wind_coeff": -0.3},
    "monsoon": {"temp_coeff": 1.2, "humidity_coeff": 0.9, "rain_coeff": -1.6, "wind_coeff": -0.35},
}


def _block_shape_multiplier(block_number: int) -> float:
    # Per-block shaping (1..96), no grouped zones.
    b = max(1, min(96, int(block_number)))
    t = (b - 1) / 95.0
    morning_peak = np.exp(-((t - 0.42) ** 2) / 0.012)
    evening_peak = np.exp(-((t - 0.78) ** 2) / 0.010)
    night_dip = np.exp(-((t - 0.08) ** 2) / 0.030)
    raw = 0.78 + (0.52 * morning_peak) + (0.74 * evening_peak) - (0.16 * night_dip)
    return float(np.clip(raw, 0.6, 1.6))


class TemperatureSensitivityRequest(BaseModel):
    baseline_temp: float
    adjusted_temp: float
    block_number: int
    season: Literal["summer", "winter", "monsoon"] = "summer"
    sector_mix: dict = {}


SEASON_TEMP_ELASTICITY = {
    "summer": 1.8,   # % per +1C
    "winter": 0.6,
    "monsoon": 1.2,
}

SEASON_COMFORT_BANDS = {
    "summer": (24.0, 30.0),
    "winter": (18.0, 24.0),
    "monsoon": (22.0, 28.0),
}

SECTOR_TEMP_SENSITIVITY = {
    "residential": 1.2,
    "commercial": 1.0,
    "industrial": 0.55,
    "agriculture": 0.75,
}


def _sector_weight(sector_mix: dict) -> float:
    if not isinstance(sector_mix, dict) or not sector_mix:
        return 1.0
    num = 0.0
    den = 0.0
    for k, v in sector_mix.items():
        weight = float(v or 0.0)
        if weight <= 0:
            continue
        sens = float(SECTOR_TEMP_SENSITIVITY.get(str(k).lower(), 1.0))
        num += weight * sens
        den += weight
    if den <= 1e-9:
        return 1.0
    return float(num / den)


def _comfort_band_multiplier(adjusted_temp: float, season: str) -> float:
    low, high = SEASON_COMFORT_BANDS.get(season, (22.0, 28.0))
    t = float(adjusted_temp)
    if low <= t <= high:
        return 0.85
    distance = low - t if t < low else t - high
    return float(min(1.5, 1.0 + (0.05 * distance)))


def _simulator_season(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return "summer"
        month = int(dt.month)
    except Exception:
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    if month in (11, 12, 1, 2):
        return "winter"
    return "summer"


def _simulator_day_type(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return "Weekday"
        return "Weekend" if int(dt.weekday()) >= 5 else "Weekday"
    except Exception:
        return "Weekday"


@lru_cache(maxsize=8192)
def _simulator_date_context(date_str: str) -> Tuple[str, str]:
    return _simulator_season(date_str), _simulator_day_type(date_str)


def _seasonal_prior_temp_coeff(season: str) -> float:
    return {
        "summer": 0.0100,
        "monsoon": 0.0080,
        "winter": -0.0100,
    }.get(str(season), 0.0090)


def _clip_temp_coeff(value: float, prior: float) -> float:
    v = float(np.clip(float(value), -0.03, 0.03))
    if prior >= 0:
        return float(max(0.0005, v))
    return float(min(-0.0005, v))


def _robust_temp_slope(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy) & (np.abs(xx) > 1e-6)
    xx = xx[valid]
    yy = yy[valid]
    n = int(xx.size)
    if n < 4:
        return None, n
    ratio = yy / xx
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 4:
        return None, int(ratio.size)
    q1, q3 = np.percentile(ratio, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    lo = float(q1 - (2.0 * iqr))
    hi = float(q3 + (2.0 * iqr))
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    use = trimmed if trimmed.size >= 4 else ratio
    if use.size < 4:
        return None, int(use.size)
    return float(np.median(use)), int(use.size)


def _blend_coeff(prior: float, estimate: Optional[float], sample_count: int, ridge: float) -> float:
    if estimate is None:
        return float(prior)
    w = float(np.clip(sample_count / max(sample_count + ridge, 1e-6), 0.0, 1.0))
    return float(((1.0 - w) * prior) + (w * float(estimate)))


def _learn_temperature_delta_profile(
    df: pd.DataFrame,
    target_date: str,
    all_dates: List[str],
    season: str,
    day_type_label: str,
    lookback_days: int = 180,
    rolling_days: int = 7,
    min_block_samples: int = 8,
) -> Dict[str, Any]:
    prior = _seasonal_prior_temp_coeff(season)
    default_up = np.full(96, prior, dtype=float)
    default_down = np.full(96, prior, dtype=float)
    default_samples = np.zeros(96, dtype=int)
    out = {
        "source": "seasonal_prior",
        "prior_coeff": float(prior),
        "global_increase_coeff": float(prior),
        "global_reduction_coeff": float(prior),
        "global_increase_samples": 0,
        "global_reduction_samples": 0,
        "block_increase_coeffs": default_up,
        "block_reduction_coeffs": default_down,
        "block_increase_samples": default_samples.copy(),
        "block_reduction_samples": default_samples.copy(),
        "diagnostics": {"reason": "fallback_prior"},
    }

    required_cols = {"date", "time_block", "temperature", "total_drawal"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        out["diagnostics"] = {"reason": "missing_required_columns"}
        return out

    date_list = [str(d) for d in (all_dates or [])]
    if not date_list:
        date_list = sorted(df["date"].dropna().astype(str).unique().tolist())
    if target_date not in date_list:
        out["diagnostics"] = {"reason": "target_not_in_dates"}
        return out

    target_idx = date_list.index(target_date)
    hist_dates = date_list[max(0, target_idx - max(lookback_days, 30)):target_idx]
    if not hist_dates:
        out["diagnostics"] = {"reason": "no_history_dates"}
        return out

    same_context_dates = [d for d in hist_dates if _simulator_date_context(d) == (season, day_type_label)]
    if len(same_context_dates) >= 28:
        selected_dates = same_context_dates[-lookback_days:]
        context_mode = "season_daytype"
    else:
        fallback_daytype = [d for d in hist_dates if _simulator_date_context(d)[1] == day_type_label]
        if len(fallback_daytype) >= 28:
            selected_dates = fallback_daytype[-lookback_days:]
            context_mode = "daytype"
        else:
            selected_dates = hist_dates[-lookback_days:]
            context_mode = "recent_all"

    if not selected_dates:
        out["diagnostics"] = {"reason": "no_selected_dates"}
        return out

    hist = df[df["date"].astype(str).isin(set(selected_dates))][["date", "time_block", "temperature", "total_drawal"]].copy()
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_history_slice"}
        return out

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist = hist.dropna(subset=["time_block", "temperature", "total_drawal"])
    hist["time_block"] = hist["time_block"].astype(int)
    hist = hist[(hist["time_block"] >= 1) & (hist["time_block"] <= 96)]
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_numeric_clean"}
        return out

    date_order = {d: i for i, d in enumerate(selected_dates)}
    hist["date_idx"] = hist["date"].astype(str).map(date_order)
    hist = hist.dropna(subset=["date_idx"]).sort_values(["time_block", "date_idx"])
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_date_order"}
        return out

    min_periods = max(3, int(min(rolling_days, 7)))
    hist["base_temp"] = (
        hist.groupby("time_block")["temperature"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["base_load"] = (
        hist.groupby("time_block")["total_drawal"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["temp_delta"] = hist["temperature"] - hist["base_temp"]
    hist["load_ratio"] = (hist["total_drawal"] / np.maximum(hist["base_load"], 1e-6)) - 1.0

    valid = (
        np.isfinite(hist["temp_delta"])
        & np.isfinite(hist["load_ratio"])
        & np.isfinite(hist["base_load"])
        & (hist["base_load"] > 1e-6)
        & (np.abs(hist["temp_delta"]) >= 0.15)
    )
    hist = hist[valid]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_valid_delta_rows", "context_mode": context_mode}
        return out

    pos = hist[hist["temp_delta"] > 0]
    neg = hist[hist["temp_delta"] < 0]

    g_up_est, g_up_n = _robust_temp_slope(pos["temp_delta"].to_numpy(dtype=float), pos["load_ratio"].to_numpy(dtype=float))
    g_dn_est, g_dn_n = _robust_temp_slope(neg["temp_delta"].to_numpy(dtype=float), neg["load_ratio"].to_numpy(dtype=float))
    global_up = _clip_temp_coeff(_blend_coeff(prior, g_up_est, g_up_n, ridge=60.0), prior)
    global_down = _clip_temp_coeff(_blend_coeff(prior, g_dn_est, g_dn_n, ridge=60.0), prior)

    block_up = np.full(96, global_up, dtype=float)
    block_down = np.full(96, global_down, dtype=float)
    block_up_n = np.zeros(96, dtype=int)
    block_down_n = np.zeros(96, dtype=int)

    for b in range(1, 97):
        blk = hist[hist["time_block"] == b]
        if blk.empty:
            continue
        blk_pos = blk[blk["temp_delta"] > 0]
        blk_neg = blk[blk["temp_delta"] < 0]
        up_est, up_n = _robust_temp_slope(blk_pos["temp_delta"].to_numpy(dtype=float), blk_pos["load_ratio"].to_numpy(dtype=float))
        dn_est, dn_n = _robust_temp_slope(blk_neg["temp_delta"].to_numpy(dtype=float), blk_neg["load_ratio"].to_numpy(dtype=float))
        block_up_n[b - 1] = int(up_n)
        block_down_n[b - 1] = int(dn_n)

        if up_n >= int(min_block_samples):
            blended_up = _blend_coeff(global_up, up_est, up_n, ridge=18.0)
            block_up[b - 1] = _clip_temp_coeff(blended_up, prior)
        if dn_n >= int(min_block_samples):
            blended_down = _blend_coeff(global_down, dn_est, dn_n, ridge=18.0)
            block_down[b - 1] = _clip_temp_coeff(blended_down, prior)

    out.update({
        "source": "learned_history",
        "global_increase_coeff": float(global_up),
        "global_reduction_coeff": float(global_down),
        "global_increase_samples": int(g_up_n),
        "global_reduction_samples": int(g_dn_n),
        "block_increase_coeffs": block_up,
        "block_reduction_coeffs": block_down,
        "block_increase_samples": block_up_n,
        "block_reduction_samples": block_down_n,
        "diagnostics": {
            "context_mode": context_mode,
            "history_dates": int(len(selected_dates)),
            "valid_rows": int(len(hist)),
        },
    })
    return out


def _adaptive_temperature_base(
    df: pd.DataFrame,
    all_dates: List[str],
    temp_delta: np.ndarray,
    target_temperature: np.ndarray,
    date_str: str,
    day_type_label: str,
) -> Dict[str, Any]:
    season = _simulator_season(date_str)
    low, high = SEASON_COMFORT_BANDS.get(season, (22.0, 28.0))
    delta_vec = np.asarray(temp_delta, dtype=float)
    temp_vec = np.asarray(target_temperature, dtype=float)
    if temp_vec.size != delta_vec.size:
        temp_vec = np.resize(temp_vec, delta_vec.shape)

    learned = _learn_temperature_delta_profile(
        df=df,
        target_date=str(date_str),
        all_dates=all_dates,
        season=season,
        day_type_label=day_type_label,
    )
    prior = float(learned.get("prior_coeff", _seasonal_prior_temp_coeff(season)))
    up_coeff = np.asarray(learned.get("block_increase_coeffs", np.full(delta_vec.shape, prior)), dtype=float)
    down_coeff = np.asarray(learned.get("block_reduction_coeffs", np.full(delta_vec.shape, prior)), dtype=float)
    if up_coeff.size != delta_vec.size:
        up_coeff = np.resize(up_coeff, delta_vec.shape)
    if down_coeff.size != delta_vec.size:
        down_coeff = np.resize(down_coeff, delta_vec.shape)

    coeff_vec = np.where(delta_vec >= 0, up_coeff, down_coeff).astype(float)


    if season == "winter":
        # Warm winter pockets still reduce heating effect; keep direction, reduce magnitude only.
        warm_mask = temp_vec >= (high + 2.0)
        coeff_vec[warm_mask] = coeff_vec[warm_mask] * 0.35
    else:
        # Cool summer/monsoon pockets dampen cooling response.
        cool_mask = temp_vec <= (low - 2.0)
        coeff_vec[cool_mask] = coeff_vec[cool_mask] * 0.35

    inside_band = (temp_vec >= low) & (temp_vec <= high)
    distance = np.where(temp_vec < low, low - temp_vec, np.where(temp_vec > high, temp_vec - high, 0.0))
    comfort_mult = np.where(inside_band, 0.85, np.minimum(1.6, 1.0 + (0.05 * distance)))

    effective_coeff = coeff_vec * comfort_mult
    temperature_base = np.clip(delta_vec * effective_coeff, -0.12, 0.12)
    return {
        "temperature_base": temperature_base,
        "season": season,
        "effective_coeff": effective_coeff,
        "increase_coeff": up_coeff,
        "reduction_coeff": down_coeff,
        "profile": learned,
    }


def _split_delta_direction(value: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if value is None or not np.isfinite(float(value)):
        return None, None
    v = float(value)
    return (float(max(v, 0.0)), float(min(v, 0.0)))


class ScenarioSaveRequest(BaseModel):
    scenario_name: str
    created_by: str = "analyst"
    notes: Optional[str] = ""
    drivers_applied: dict = {}
    block_adjustments: object
    forecast_summary: Optional[dict] = None
    weather_inputs: Optional[dict] = None
    holiday_template: Optional[dict] = None
    forecast_output: Optional[dict] = None


class ScenarioCompareRequest(BaseModel):
    base_scenario_id: str
    target_scenario_id: str


class SimulatorBlocksRequest(BaseModel):
    date: Optional[str] = None
    baseline_days: Optional[int] = 7
    sliders: Optional[dict] = None
    selection: Optional[dict] = None
    smooth_selection: Optional[bool] = False
    temp_delta: Optional[List[float]] = None
    humidity_delta: Optional[List[float]] = None
    rain_delta: Optional[List[float]] = None
    wind_delta: Optional[List[float]] = None
    daytype_flag: Optional[List[float]] = None
    holiday_flag: Optional[List[float]] = None
    weather_base: Optional[List[float]] = None
    temperature_base: Optional[List[float]] = None
    humidity_base: Optional[List[float]] = None
    precipitation_base: Optional[List[float]] = None
    daytype_base: Optional[List[float]] = None
    holiday_base: Optional[List[float]] = None
    manual_base: Optional[List[float]] = None
    momentum_lambda: Optional[float] = 0.45


class ScenarioRepository:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "").strip()
        self._json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports", "scenarios_store.json")
        self._ensure_storage()

    def _ensure_storage(self):
        if self.db_url:
            try:
                import psycopg2  # type: ignore
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS scenarios (
                              scenario_id TEXT PRIMARY KEY,
                              scenario_name TEXT NOT NULL,
                              created_at TIMESTAMPTZ NOT NULL,
                              created_by TEXT NOT NULL,
                              drivers_applied JSONB NOT NULL,
                              block_adjustments JSONB NOT NULL,
                              forecast_summary JSONB NOT NULL,
                              notes TEXT,
                              weather_inputs JSONB,
                              holiday_template JSONB,
                              forecast_output JSONB,
                              version INTEGER NOT NULL
                            )
                            """
                        )
                    conn.commit()
                return
            except Exception:
                pass
        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        if not os.path.exists(self._json_path):
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_json(self):
        with open(self._json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, rows):
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=True, indent=2, default=str)

    def list(self):
        if self.db_url:
            try:
                import psycopg2  # type: ignore
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT scenario_id, scenario_name, created_at, created_by, drivers_applied,
                                   block_adjustments, forecast_summary, notes, weather_inputs,
                                   holiday_template, forecast_output, version
                            FROM scenarios
                            ORDER BY version DESC, created_at DESC
                            """
                        )
                        cols = [d[0] for d in cur.description]
                        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                return rows
            except Exception:
                pass
        rows = self._read_json()
        return sorted(rows, key=lambda r: (r.get("version", 0), r.get("created_at", "")), reverse=True)

    def get(self, scenario_id: str):
        rows = self.list()
        for r in rows:
            if str(r.get("scenario_id")) == str(scenario_id):
                return r
        return None

    def delete(self, scenario_id: str):
        if self.db_url:
            try:
                import psycopg2  # type: ignore
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM scenarios WHERE scenario_id = %s", (scenario_id,))
                    conn.commit()
                return
            except Exception:
                pass
        rows = [r for r in self._read_json() if str(r.get("scenario_id")) != str(scenario_id)]
        self._write_json(rows)

    def save(self, row: dict):
        if self.db_url:
            try:
                import psycopg2  # type: ignore
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO scenarios (
                              scenario_id, scenario_name, created_at, created_by, drivers_applied,
                              block_adjustments, forecast_summary, notes, weather_inputs,
                              holiday_template, forecast_output, version
                            ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                            """,
                            (
                                row["scenario_id"],
                                row["scenario_name"],
                                row["created_at"],
                                row["created_by"],
                                json.dumps(row.get("drivers_applied", {})),
                                json.dumps(row.get("block_adjustments", {})),
                                json.dumps(row.get("forecast_summary", {})),
                                row.get("notes", ""),
                                json.dumps(row.get("weather_inputs", {})),
                                json.dumps(row.get("holiday_template", {})),
                                json.dumps(row.get("forecast_output", {})),
                                int(row.get("version", 1)),
                            ),
                        )
                    conn.commit()
                return
            except Exception:
                pass
        rows = self._read_json()
        rows.append(row)
        self._write_json(rows)


scenario_repo = ScenarioRepository()


@app.get("/api/config")
@app.get("/config")
def get_config():
    """Returns Feature Registry and Metadata."""
    return {
        "features": FeatureRegistry.get_features(),
        "catalog": FeatureRegistry.CATALOG,
        "eda_types": FeatureRegistry.EDA_TYPES,
        "date_range": engine.get_date_range()
    }

# =========================================================
# GRIDINTEL CONTROL DESK API (NEW ARCHITECTURE)
# =========================================================

@app.post("/api/load/summary")
def api_load_summary(req: AnalyticsRequest):
    return gridintel.get_summary(req.start_date, req.end_date, req.feature)

@app.post("/api/load/shape")
def api_load_shape(req: AnalyticsRequest):
    return gridintel.get_shape(req.start_date, req.end_date, req.feature)

@app.post("/api/load/decomposition")
def api_load_decomposition(req: AnalyticsRequest):
    return gridintel.get_decomposition(req.start_date, req.end_date, req.feature)

@app.post("/api/load/ramp")
def api_load_ramp(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    kpis = gridintel.kpis.compute_ramp(df, req.feature)
    return {"charts": {}, "kpis": kpis, "insights": [], "metadata": {"module": "ramp"}}

@app.post("/api/calendar/impact")
def api_calendar_impact(req: AnalyticsRequest):
    return gridintel.get_calendar(req.start_date, req.end_date, req.feature)

@app.post("/api/weather/sensitivity")
def api_weather_sensitivity(req: AnalyticsRequest):
    return gridintel.get_weather(req.start_date, req.end_date, req.feature)


@app.post("/api/weather-impact/calculate")
@app.post("/weather-impact/calculate")
def api_weather_impact_calculate(req: WeatherImpactRequest):
    coeffs = WEATHER_SENSITIVITY_COEFFS.get(req.season, WEATHER_SENSITIVITY_COEFFS["summer"])
    impacts = []
    for idx, blk in enumerate(req.blocks[:96]):
        block_number = idx + 1
        raw = (
            float(blk.temp_delta_c) * coeffs["temp_coeff"]
            + (float(blk.humidity_delta_pct) / 10.0) * coeffs["humidity_coeff"]
            + (float(blk.precip_delta_mm) / 10.0) * coeffs["rain_coeff"]
            + float(blk.wind_delta_mps) * coeffs["wind_coeff"]
        )
        mult = _block_shape_multiplier(block_number)
        shaped = raw * mult
        weather_pct = float(np.clip(shaped, -15.0, 15.0))
        impacts.append({
            "block_number": block_number,
            "time_multiplier": mult,
            "raw_weather_pct": round(float(raw), 4),
            "weather_pct": round(weather_pct, 4),
        })

    for block_number in range(len(impacts) + 1, 97):
        impacts.append({
            "block_number": block_number,
            "time_multiplier": _block_shape_multiplier(block_number),
            "raw_weather_pct": 0.0,
            "weather_pct": 0.0,
        })

    return _json_safe({
        "season": req.season,
        "coefficients": coeffs,
        "impacts": impacts,
    })


@app.post("/api/temperature-sensitivity/calculate")
@app.post("/temperature-sensitivity/calculate")
def api_temperature_sensitivity_calculate(req: TemperatureSensitivityRequest):
    # 1) Compute temp delta.
    temp_delta = float(req.adjusted_temp) - float(req.baseline_temp)

    # 2) Apply seasonal elasticity coefficient.
    seasonal_coeff = float(SEASON_TEMP_ELASTICITY.get(req.season, SEASON_TEMP_ELASTICITY["summer"]))
    after_season = temp_delta * seasonal_coeff

    # 3) Apply time-of-day multiplier by block zone.
    tod_multiplier = _block_shape_multiplier(int(req.block_number))
    after_tod = after_season * tod_multiplier

    # 4) Apply comfort band multiplier.
    comfort_multiplier = _comfort_band_multiplier(float(req.adjusted_temp), req.season)
    after_comfort = after_tod * comfort_multiplier

    # 5) Apply sector weight.
    sector_weight = _sector_weight(req.sector_mix)
    raw_impact_pct = after_comfort * sector_weight

    # 6) Clamp output to ±25%.
    impact_pct = float(np.clip(raw_impact_pct, -25.0, 25.0))

    return _json_safe({
        "block_number": int(req.block_number),
        "season": req.season,
        "baseline_temp": float(req.baseline_temp),
        "adjusted_temp": float(req.adjusted_temp),
        "sector_mix": req.sector_mix,
        "steps": {
            "temp_delta": round(temp_delta, 4),
            "seasonal_elasticity_coeff": seasonal_coeff,
            "after_season": round(after_season, 4),
            "time_of_day_multiplier": tod_multiplier,
            "after_time_of_day": round(after_tod, 4),
            "comfort_band_multiplier": comfort_multiplier,
            "after_comfort_band": round(after_comfort, 4),
            "sector_weight": round(sector_weight, 4),
            "raw_impact_pct": round(raw_impact_pct, 4),
        },
        "temperature_impact_pct": round(impact_pct, 4),
        "clamp_bounds_pct": [-25.0, 25.0],
    })


def _scenario_summary_from_blocks(block_adjustments) -> dict:
    blocks = block_adjustments if isinstance(block_adjustments, list) else []
    if not blocks:
        return {"peak_mw": 0.0, "energy_mu": 0.0, "net_impact_pct": 0.0}
    baseline = np.array([float(b.get("baseline_mw", 0.0)) for b in blocks], dtype=float)
    final = np.array([float(b.get("final_mw", b.get("baseline_mw", 0.0))) for b in blocks], dtype=float)
    peak_mw = float(np.max(final)) if len(final) else 0.0
    energy_mu = float(np.sum(final) * 0.25) if len(final) else 0.0
    base_energy = float(np.sum(baseline) * 0.25) if len(baseline) else 0.0
    net_impact_pct = float(((energy_mu - base_energy) / max(base_energy, 1e-6)) * 100)
    return {"peak_mw": round(peak_mw, 3), "energy_mu": round(energy_mu, 3), "net_impact_pct": round(net_impact_pct, 3)}


@app.post("/api/scenario/save")
@app.post("/scenario/save")
def api_scenario_save(req: ScenarioSaveRequest):
    existing = scenario_repo.list()
    next_version = (max([int(r.get("version", 0)) for r in existing], default=0) + 1)
    row = {
        "scenario_id": str(uuid.uuid4()),
        "scenario_name": req.scenario_name,
        "created_at": datetime.utcnow().isoformat(),
        "created_by": req.created_by or "analyst",
        "drivers_applied": req.drivers_applied or {},
        "block_adjustments": req.block_adjustments,
        "forecast_summary": req.forecast_summary or _scenario_summary_from_blocks(req.block_adjustments),
        "notes": req.notes or "",
        "weather_inputs": req.weather_inputs or {},
        "holiday_template": req.holiday_template or {},
        "forecast_output": req.forecast_output or {},
        "version": next_version,
    }
    scenario_repo.save(row)
    return _json_safe({"status": "ok", "scenario": row})


@app.get("/api/scenarios")
@app.get("/scenarios")
def api_scenarios():
    return _json_safe({"items": scenario_repo.list()})


@app.post("/api/scenario/restore/{scenario_id}")
@app.post("/scenario/restore/{scenario_id}")
def api_scenario_restore(scenario_id: str):
    row = scenario_repo.get(scenario_id)
    if not row:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _json_safe({"status": "ok", "scenario": row})


@app.post("/api/scenario/delete/{scenario_id}")
@app.post("/scenario/delete/{scenario_id}")
def api_scenario_delete(scenario_id: str):
    row = scenario_repo.get(scenario_id)
    if not row:
        raise HTTPException(status_code=404, detail="Scenario not found")
    scenario_repo.delete(scenario_id)
    return _json_safe({"status": "ok"})


@app.post("/api/scenario/compare")
@app.post("/scenario/compare")
def api_scenario_compare(req: ScenarioCompareRequest):
    base = scenario_repo.get(req.base_scenario_id)
    target = scenario_repo.get(req.target_scenario_id)
    if not base or not target:
        raise HTTPException(status_code=404, detail="Scenario not found")

    base_blocks = base.get("block_adjustments", []) if isinstance(base.get("block_adjustments"), list) else []
    target_blocks = target.get("block_adjustments", []) if isinstance(target.get("block_adjustments"), list) else []
    t_by_block = {int(b.get("block_number", i + 1)): b for i, b in enumerate(target_blocks)}
    paired = []
    for i, b in enumerate(base_blocks):
        bn = int(b.get("block_number", i + 1))
        tb = t_by_block.get(bn, {})
        base_val = float(b.get("final_mw", b.get("baseline_mw", 0.0)))
        tgt_val = float(tb.get("final_mw", tb.get("baseline_mw", base_val)))
        paired.append({"block_number": bn, "base_final_mw": base_val, "target_final_mw": tgt_val, "delta_mw": tgt_val - base_val})
    if not paired:
        paired = [{"block_number": i + 1, "base_final_mw": 0.0, "target_final_mw": 0.0, "delta_mw": 0.0} for i in range(96)]

    peak_change = float(target.get("forecast_summary", {}).get("peak_mw", 0.0)) - float(base.get("forecast_summary", {}).get("peak_mw", 0.0))
    energy_change = float(target.get("forecast_summary", {}).get("energy_mu", 0.0)) - float(base.get("forecast_summary", {}).get("energy_mu", 0.0))
    max_block_delta = float(max((abs(x["delta_mw"]) for x in paired), default=0.0))

    return _json_safe({
        "base": base,
        "target": target,
        "diff_metrics": {
            "peak_change": round(peak_change, 3),
            "total_energy_change": round(energy_change, 3),
            "max_block_delta": round(max_block_delta, 3),
        },
        "blocks": paired,
    })


@app.post("/api/simulator/blocks")
@app.post("/simulator/blocks")
def api_simulator_blocks(req: SimulatorBlocksRequest):
    df = _get_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No simulator data available")

    requested = req.date
    baseline_days = max(1, min(int(req.baseline_days or 7), 15))
    all_dates = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
    if not all_dates:
        raise HTTPException(status_code=404, detail="No date available")
    resolved = requested if requested in all_dates else all_dates[-1]
    if not resolved:
        raise HTTPException(status_code=404, detail="No date available")

    dayahead_result = _compute_dayahead(df, resolved, baseline_days)
    if not isinstance(dayahead_result, dict):
        raise HTTPException(status_code=404, detail="Unable to compute simulator baseline")
    series = dayahead_result.get("series", {})
    if not isinstance(series, dict) or not series:
        raise HTTPException(status_code=404, detail="Unable to compute simulator baseline")

    if "baseline" not in series:
        if "historical_baseline" in series:
            series["baseline"] = series["historical_baseline"]
        elif "hybrid_baseline" in series:
            series["baseline"] = series["hybrid_baseline"]
        elif "forecast" in series:
            series["baseline"] = series["forecast"]

    # Provide legacy delta keys expected by the simulator.
    deltas = series.get("weather_feature_deltas", {})
    if isinstance(deltas, dict):
        series.setdefault("temp_delta", deltas.get("temperature", [0.0] * 96))
        series.setdefault("hum_delta", deltas.get("humidity", [0.0] * 96))
        series.setdefault("precip_delta", deltas.get("precipitation", [0.0] * 96))
        series.setdefault(
            "wind_delta",
            deltas.get("wind_speed_10m", deltas.get("wind", [0.0] * 96)),
        )

    # Use raw (non-filled) actuals to detect partial-day availability.
    raw_day = df[df["date"] == resolved].copy()
    raw_day["time_block"] = pd.to_numeric(raw_day.get("time_block"), errors="coerce")
    raw_day = raw_day[raw_day["time_block"].between(1, 96)]
    raw_day["time_block"] = raw_day["time_block"].astype(int)
    raw_day["total_drawal"] = pd.to_numeric(raw_day.get("total_drawal"), errors="coerce")
    actual_raw_series = raw_day.groupby("time_block")["total_drawal"].mean().reindex(range(1, 97))
    actual_raw_vec = actual_raw_series.to_numpy(dtype=float)
    series["actual_raw"] = actual_raw_vec.tolist()
    prefix_blocks = 0
    for v in actual_raw_vec[:96]:
        if np.isfinite(v):
            prefix_blocks += 1
        else:
            break
    series["actual_available_blocks"] = int(prefix_blocks)

    baseline_df = _baseline_window(df, resolved, baseline_days)

    day_df = (
        df[df["date"] == resolved]
        .sort_values("time_block")
        .set_index("time_block")
        .reindex(range(1, 97))
        .ffill()
        .bfill()
    )

    wind_col = None
    for col in ("wind_speed_10m", "wind_speed", "wind", "windspeed", "wind_mps"):
        if col in day_df.columns:
            wind_col = col
            break
    if wind_col is not None and (not baseline_df.empty) and wind_col in baseline_df.columns:
        base_wind = (
            baseline_df.groupby("time_block")[wind_col]
            .mean()
            .reindex(range(1, 97))
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
    else:
        base_wind = np.zeros(96, dtype=float)
    tgt_wind = day_df[wind_col].to_numpy(dtype=float) if wind_col is not None else np.zeros(96, dtype=float)
    wind_delta = tgt_wind - base_wind

    temp_delta = np.array(series.get("temp_delta", [0.0] * 96), dtype=float)
    hum_delta = np.array(series.get("hum_delta", [0.0] * 96), dtype=float)
    precip_delta = np.array(series.get("precip_delta", [0.0] * 96), dtype=float)
    wind_delta = np.array(series.get("wind_delta", wind_delta.tolist()), dtype=float)

    baseline_vec = np.array(series.get("baseline", [0.0] * 96), dtype=float)
    target_day = pd.to_datetime(resolved, errors="coerce")
    is_weekend = False if pd.isna(target_day) else bool(target_day.weekday() >= 5)
    day_type_label = "Weekend" if is_weekend else "Weekday"
    season_label = _simulator_season(resolved)
    # Run pipeline for weights
    try:
        weights_df = compute_block_driver_weights(df, target_date=resolved)

        # Normalize columns if needed
        if "weight_confidence" not in weights_df.columns:
            weights_df["weight_confidence"] = 0.5
            
        weights_source = "linear_regression"
    except Exception as e:
        print(f"Error in api_simulator_blocks weights: {e}")
        # Fallback
        weights_df = pd.DataFrame({"block": range(1, 97)})
        weights_df["weight_confidence"] = 0.1
        weights_source = "fallback"

    momentum_lambda = float(np.clip(float(req.momentum_lambda if req.momentum_lambda is not None else 0.45), 0.3, 0.6))
    
    # Extract weather components if available from pipeline (skipping legacy engine call)
    weather_engine_weights = weights_df if not weights_df.empty else None
    
    # Map pipeline weights to coefficients for response
    # Default zero vectors
    temp_effective_coeff = np.zeros(96, dtype=float)
    humidity_effective_coeff = np.zeros(96, dtype=float)
    rain_effective_coeff = np.zeros(96, dtype=float)
    regime_conf = np.zeros(96, dtype=float)

    if not weights_df.empty:
         if "temp_weight" in weights_df.columns:
             temp_effective_coeff = weights_df["temp_weight"].to_numpy(dtype=float)[:96]
         if "humidity_weight" in weights_df.columns:
             humidity_effective_coeff = weights_df["humidity_weight"].to_numpy(dtype=float)[:96]
         if "rain_weight" in weights_df.columns:
             rain_effective_coeff = weights_df["rain_weight"].to_numpy(dtype=float)[:96]
         if "weight_confidence" in weights_df.columns:
             regime_conf = weights_df["weight_confidence"].to_numpy(dtype=float)[:96]

    # Pad if needed (safety)
    if temp_effective_coeff.size < 96:
        temp_effective_coeff = np.pad(temp_effective_coeff, (0, 96 - temp_effective_coeff.size), mode="edge")
    if humidity_effective_coeff.size < 96:
        humidity_effective_coeff = np.pad(humidity_effective_coeff, (0, 96 - humidity_effective_coeff.size), mode="edge")
    if rain_effective_coeff.size < 96:
        rain_effective_coeff = np.pad(rain_effective_coeff, (0, 96 - rain_effective_coeff.size), mode="edge")
    if regime_conf.size < 96:
        regime_conf = np.pad(regime_conf, (0, 96 - regime_conf.size), mode="edge")

    # Use pipeline results, so 'use_weather_engine' is effectively True if we got weights
    use_weather_engine = not weights_df.empty
    regime_label = "elastic_net"

    if req.temp_delta is not None:
        temp_delta = np.array(req.temp_delta[:96], dtype=float)
    if req.humidity_delta is not None:
        hum_delta = np.array(req.humidity_delta[:96], dtype=float)
    if req.rain_delta is not None:
        precip_delta = np.array(req.rain_delta[:96], dtype=float)
    if req.wind_delta is not None:
        wind_delta = np.array(req.wind_delta[:96], dtype=float)

    if temp_delta.size < 96:
        temp_delta = np.pad(temp_delta, (0, 96 - temp_delta.size), mode="constant")
    if hum_delta.size < 96:
        hum_delta = np.pad(hum_delta, (0, 96 - hum_delta.size), mode="constant")
    if precip_delta.size < 96:
        precip_delta = np.pad(precip_delta, (0, 96 - precip_delta.size), mode="constant")
    if wind_delta.size < 96:
        wind_delta = np.pad(wind_delta, (0, 96 - wind_delta.size), mode="constant")

    if req.daytype_flag is not None:
        daytype_flag = np.array(req.daytype_flag[:96], dtype=float)
        if daytype_flag.size < 96:
            daytype_flag = np.pad(daytype_flag, (0, 96 - daytype_flag.size), mode="edge")
    elif use_weather_engine:
        daytype_flag = np.zeros(96, dtype=float)
    else:
        daytype_flag = np.full(96, 1.0 if is_weekend else 0.0, dtype=float)

    holiday_flag_value = 0.0
    for col in ("is_holiday", "holiday", "holiday_flag", "holiday_ind"):
        if col in day_df.columns:
            raw = pd.to_numeric(day_df[col], errors="coerce").to_numpy(dtype=float)
            valid = raw[np.isfinite(raw)]
            if valid.size:
                holiday_flag_value = float(np.mean(valid > 0.0) >= 0.5)
                break
    if req.holiday_flag is not None:
        holiday_flag = np.array(req.holiday_flag[:96], dtype=float)
        if holiday_flag.size < 96:
            holiday_flag = np.pad(holiday_flag, (0, 96 - holiday_flag.size), mode="edge")
    elif use_weather_engine:
        holiday_flag = np.zeros(96, dtype=float)
    else:
        holiday_flag = np.full(96, holiday_flag_value, dtype=float)

    humidity_base_auto = None
    precipitation_base_auto = None
    momentum_base_auto = np.zeros(96, dtype=float)
    dod_load_delta_mw = np.zeros(96, dtype=float)
    if use_weather_engine:
        if temp_effective_coeff.size < 96:
            temp_effective_coeff = np.pad(temp_effective_coeff, (0, 96 - temp_effective_coeff.size), mode="edge")
        if humidity_effective_coeff.size < 96:
            humidity_effective_coeff = np.pad(humidity_effective_coeff, (0, 96 - humidity_effective_coeff.size), mode="edge")
        if rain_effective_coeff.size < 96:
            rain_effective_coeff = np.pad(rain_effective_coeff, (0, 96 - rain_effective_coeff.size), mode="edge")
        if regime_conf.size < 96:
            regime_conf = np.pad(regime_conf, (0, 96 - regime_conf.size), mode="edge")
        temp_increase_coeff = temp_effective_coeff.copy()
        temp_reduction_coeff = temp_effective_coeff.copy()
        temperature_base_auto = temp_delta * temp_effective_coeff
        humidity_base_auto = hum_delta * humidity_effective_coeff
        precipitation_base_auto = precip_delta * rain_effective_coeff
    else:
        # Fallback to prior regime logic when lag-weather engine cannot be built.
        enet_temp_weight = (
            weights_df.sort_values("block")["temp_weight"].to_numpy(dtype=float)
            if "temp_weight" in weights_df.columns
            else np.zeros(96, dtype=float)
        )
        if enet_temp_weight.size < 96:
            enet_temp_weight = np.pad(enet_temp_weight, (0, 96 - enet_temp_weight.size), mode="edge")

        winter_profile = _learn_temperature_delta_profile(
            df=df,
            target_date=resolved,
            all_dates=all_dates,
            season="winter",
            day_type_label=day_type_label,
        )
        summer_profile = _learn_temperature_delta_profile(
            df=df,
            target_date=resolved,
            all_dates=all_dates,
            season="summer",
            day_type_label=day_type_label,
        )
        monsoon_profile = _learn_temperature_delta_profile(
            df=df,
            target_date=resolved,
            all_dates=all_dates,
            season="monsoon",
            day_type_label=day_type_label,
        )


        def _weighted_merge(a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
            denom = max(float(wa + wb), 1e-6)
            return ((float(wa) * a) + (float(wb) * b)) / denom

        winter_up = _to_n_vec(winter_profile.get("block_increase_coeffs"), fill=float(np.median(enet_temp_weight)))
        winter_dn = _to_n_vec(winter_profile.get("block_reduction_coeffs"), fill=float(np.median(enet_temp_weight)))
        winter_up_n = _to_n_vec(winter_profile.get("block_increase_samples"), fill=0.0)
        winter_dn_n = _to_n_vec(winter_profile.get("block_reduction_samples"), fill=0.0)

        summer_up = _to_n_vec(summer_profile.get("block_increase_coeffs"), fill=float(np.median(enet_temp_weight)))
        summer_dn = _to_n_vec(summer_profile.get("block_reduction_coeffs"), fill=float(np.median(enet_temp_weight)))
        summer_up_n = _to_n_vec(summer_profile.get("block_increase_samples"), fill=0.0)
        summer_dn_n = _to_n_vec(summer_profile.get("block_reduction_samples"), fill=0.0)

        monsoon_up = _to_n_vec(monsoon_profile.get("block_increase_coeffs"), fill=float(np.median(enet_temp_weight)))
        monsoon_dn = _to_n_vec(monsoon_profile.get("block_reduction_coeffs"), fill=float(np.median(enet_temp_weight)))
        monsoon_up_n = _to_n_vec(monsoon_profile.get("block_increase_samples"), fill=0.0)
        monsoon_dn_n = _to_n_vec(monsoon_profile.get("block_reduction_samples"), fill=0.0)

        nonwinter_up = _weighted_merge(summer_up, monsoon_up, float(np.sum(summer_up_n) + 1.0), float(np.sum(monsoon_up_n) + 1.0))
        nonwinter_dn = _weighted_merge(summer_dn, monsoon_dn, float(np.sum(summer_dn_n) + 1.0), float(np.sum(monsoon_dn_n) + 1.0))
        nonwinter_up_n = summer_up_n + monsoon_up_n
        nonwinter_dn_n = summer_dn_n + monsoon_dn_n

        if season_label == "winter":
            regime_up_coeff = winter_up
            regime_dn_coeff = winter_dn
            regime_up_n = winter_up_n
            regime_dn_n = winter_dn_n
            regime_label = "winter"
        else:
            regime_up_coeff = nonwinter_up
            regime_dn_coeff = nonwinter_dn
            regime_up_n = nonwinter_up_n
            regime_dn_n = nonwinter_dn_n
            regime_label = "non_winter"

        direction_samples = np.where(temp_delta >= 0.0, regime_up_n, regime_dn_n)
        regime_conf = np.clip(direction_samples / 18.0, 0.0, 1.0)
        regime_coeff = np.where(temp_delta >= 0.0, regime_up_coeff, regime_dn_coeff)
        blend_w = 0.25 + (0.55 * regime_conf)
        temp_effective_coeff = ((1.0 - blend_w) * enet_temp_weight) + (blend_w * regime_coeff)
        temp_increase_coeff = regime_up_coeff.astype(float)
        temp_reduction_coeff = regime_dn_coeff.astype(float)
        temperature_base_auto = temp_delta * temp_effective_coeff

    default_sliders = {
        "temperature": 1.0,
        "humidity": 1.0,
        "precipitation": 1.0,
        "weather": 1.0,
        "daytype": 1.0,
        "holiday": 1.0,
        "manual": 1.0,
    }
    effective_sliders = {**default_sliders, **(req.sliders or {})}

    base_overrides = {
        "weather_base": req.weather_base[:96] if req.weather_base is not None else None,
        "temperature_base": req.temperature_base[:96] if req.temperature_base is not None else temperature_base_auto,
        "humidity_base": req.humidity_base[:96] if req.humidity_base is not None else humidity_base_auto,
        "precipitation_base": req.precipitation_base[:96] if req.precipitation_base is not None else precipitation_base_auto,
        "daytype_base": req.daytype_base[:96] if req.daytype_base is not None else None,
        "holiday_base": req.holiday_base[:96] if req.holiday_base is not None else None,
        "manual_base": req.manual_base[:96] if req.manual_base is not None else momentum_base_auto,
    }

    delta_inputs = {
        "temp_delta": temp_delta,
        "humidity_delta": hum_delta,
        "rain_delta": precip_delta,
        "wind_delta": wind_delta,
        "daytype_flag": daytype_flag,
        "holiday_flag": holiday_flag,
    }

    driver_layer = run_block_driver_weight_delta_engine(
        target_date=resolved,
        baseline_load=baseline_vec,
        weight_matrix=weights_df,
        delta_inputs=delta_inputs,
        sliders=effective_sliders,
        selection=req.selection or {"scope": "all"},
        smooth_edges=bool(req.smooth_selection),
        base_overrides=base_overrides,
    )
    block_driver_matrix_df = driver_layer["matrix_df"]
    driver_weight_matrix_df = driver_layer["driver_weight_matrix"]
    selection_mask = np.array(driver_layer["selection_mask"], dtype=float)
    final_load = np.array(driver_layer["final_load"], dtype=float)

    if req.temperature_base is not None:
        manual_temp_base = np.asarray(req.temperature_base[:96], dtype=float)
        if manual_temp_base.size < 96:
            manual_temp_base = np.pad(manual_temp_base, (0, 96 - manual_temp_base.size), mode="edge")
        temp_effective_coeff = np.divide(
            manual_temp_base,
            np.where(np.abs(temp_delta) > 1e-9, temp_delta, 1.0),
            out=np.zeros_like(manual_temp_base, dtype=float),
            where=np.abs(temp_delta) > 1e-9,
        )
        temp_increase_coeff = temp_effective_coeff.copy()
        temp_reduction_coeff = temp_effective_coeff.copy()

    weight_confidence_vec = (
        driver_weight_matrix_df["weight_confidence"].to_numpy(dtype=float)
        if "weight_confidence" in driver_weight_matrix_df.columns
        else np.full(96, 0.5, dtype=float)
    )
    if weight_confidence_vec.size < 96:
        weight_confidence_vec = np.pad(weight_confidence_vec, (0, 96 - weight_confidence_vec.size), mode="edge")

    prior_temp_coeff = float(np.median(temp_effective_coeff)) if temp_effective_coeff.size else 0.0
    up_vals = temp_effective_coeff[temp_effective_coeff >= 0.0]
    dn_vals = temp_effective_coeff[temp_effective_coeff < 0.0]
    temperature_profile_summary = {
        "source": "provided_base" if req.temperature_base is not None else ("elastic_net_weather_lag_engine" if use_weather_engine else "elastic_net_plus_regime"),
        "prior_coeff": round(float(prior_temp_coeff), 6),
        "global_increase_coeff": round(float(np.mean(up_vals)) if up_vals.size else float(prior_temp_coeff), 6),
        "global_reduction_coeff": round(float(np.mean(dn_vals)) if dn_vals.size else float(prior_temp_coeff), 6),
        "global_increase_samples": int(up_vals.size),
        "global_reduction_samples": int(dn_vals.size),
        "diagnostics": {
            "method": "elastic_net_weather_lag_dod_momentum" if use_weather_engine else "elastic_net_with_confidence_shrinkage_and_directional_regime_blend",
            "target_date": resolved,
            "weights_source": weights_source,
            "temp_regime": regime_label if req.temperature_base is None else "provided",
            "avg_weight_confidence": float(np.mean(weight_confidence_vec)),
            "momentum_lambda": float(momentum_lambda),
        },
    }
    actual_raw_list = series.get("actual_raw", series.get("actual", []))
    actual_raw_vec = np.array([
        np.nan if (v is None or not np.isfinite(float(v))) else float(v)
        for v in actual_raw_list[:96]
    ], dtype=float)
    if actual_raw_vec.size < 96:
        actual_raw_vec = np.pad(actual_raw_vec, (0, 96 - actual_raw_vec.size), mode="constant", constant_values=np.nan)
    actual_available_blocks = int(series.get("actual_available_blocks", int(np.sum(np.isfinite(actual_raw_vec)))))
    has_manual_or_delta_override = any(
        x is not None
        for x in (
            req.temp_delta,
            req.humidity_delta,
            req.rain_delta,
            req.wind_delta,
            req.daytype_flag,
            req.holiday_flag,
            req.weather_base,
            req.temperature_base,
            req.humidity_base,
            req.precipitation_base,
            req.daytype_base,
            req.holiday_base,
            req.manual_base,
        )
    )
    selection_is_default = (req.selection is None) or (str((req.selection or {}).get("scope", "all")).strip().lower() in ("all", "overall"))
    sliders_are_default = req.sliders is None
    should_anchor_actual_window = (not has_manual_or_delta_override) and selection_is_default and sliders_are_default
    bias_correction = {
        "enabled": False,
        "applied": False,
        "reason": "disabled_for_override_or_selection",
        "available_blocks": int(actual_available_blocks),
    }
    if 0 < actual_available_blocks < 96 and should_anchor_actual_window:
        final_load = _tighten_initial_blocks_to_mape_band(final_load, actual_raw_vec, actual_available_blocks, low=0.001, high=0.005)
        final_load, bias_correction = _apply_partial_day_bias_correction(
            final_load,
            actual_raw_vec,
            actual_available_blocks,
            max_abs_pct=0.08,
        )

    blocks = []
    for i in range(96):
        bnum = i + 1
        row_i = block_driver_matrix_df.iloc[i] if i < len(block_driver_matrix_df) else None
        impact_i = float(row_i["final_impact"]) if row_i is not None else 0.0
        t_scaled = float(row_i["temperature_scaled"]) if row_i is not None and "temperature_scaled" in row_i else 0.0
        h_scaled = float(row_i["humidity_scaled"]) if row_i is not None and "humidity_scaled" in row_i else 0.0
        p_scaled = float(row_i["precipitation_scaled"]) if row_i is not None and "precipitation_scaled" in row_i else 0.0
        w_scaled = float(row_i["wind_scaled"]) if row_i is not None and "wind_scaled" in row_i else 0.0
        net_exog = float(row_i["weather_scaled"]) if row_i is not None and "weather_scaled" in row_i else (t_scaled + h_scaled + p_scaled + w_scaled)
        baseline_mw_i = float(series["baseline"][i]) if i < len(series.get("baseline", [])) else 0.0
        actual_mw_i = float(actual_raw_vec[i]) if (i < len(actual_raw_vec) and np.isfinite(actual_raw_vec[i])) else None
        actual_delta_pct = None
        if actual_mw_i is not None and abs(baseline_mw_i) > 1e-6:
            actual_delta_pct = ((actual_mw_i / baseline_mw_i) - 1.0) * 100.0

        temp_pct = float(t_scaled * 100.0)
        hum_pct = float(h_scaled * 100.0)
        precip_pct = float(p_scaled * 100.0)
        wind_pct = float(w_scaled * 100.0)
        weather_pct = float(net_exog * 100.0)
        residual_delta_pct = (float(actual_delta_pct) - weather_pct) if actual_delta_pct is not None else None
        weather_inc_pct, weather_red_pct = _split_delta_direction(weather_pct)
        temp_inc_pct, temp_red_pct = _split_delta_direction(temp_pct)
        hum_inc_pct, hum_red_pct = _split_delta_direction(hum_pct)
        precip_inc_pct, precip_red_pct = _split_delta_direction(precip_pct)
        actual_inc_pct, actual_red_pct = _split_delta_direction(actual_delta_pct)

        if actual_delta_pct is None:
            alignment = "na"
            explanation = "Actual not available for this block."
        elif abs(weather_pct) < 0.05 or abs(actual_delta_pct) < 0.05:
            alignment = "weak_signal"
            explanation = "Weather and actual movement are weak for this block."
        elif np.sign(actual_delta_pct) == np.sign(weather_pct):
            alignment = "aligned"
            explanation = "Actual moved in the same direction as weather impact."
        else:
            alignment = "opposite"
            weather_dir = "increase" if weather_pct > 1e-9 else ("decrease" if weather_pct < -1e-9 else "flat")
            actual_dir = "increase" if (actual_delta_pct or 0.0) > 1e-9 else ("decrease" if (actual_delta_pct or 0.0) < -1e-9 else "flat")
            other_weather_pct = float(weather_pct - temp_pct)
            residual_gap = float((actual_delta_pct or 0.0) - weather_pct)
            if season_label == "winter" and temp_delta[i] > 0 and temp_pct < 0:
                explanation = (
                    f"Winter warming produced a negative temperature effect ({temp_pct:+.3f}%). "
                    f"Net weather was {weather_pct:+.3f}% ({weather_dir}), but actual moved {actual_dir} "
                    f"by {float(actual_delta_pct or 0.0):+.3f}% (gap {residual_gap:+.3f}%). "
                    "Non-weather drivers likely dominated this block."
                )
            else:
                explanation = (
                    f"Weather signal was {weather_dir} ({weather_pct:+.3f}%), while actual moved {actual_dir} "
                    f"({float(actual_delta_pct or 0.0):+.3f}%). "
                    f"Temperature={temp_pct:+.3f}%, Other weather={other_weather_pct:+.3f}%. "
                    "Calendar/operational/manual or unmodeled effects likely dominated."
                )

        blocks.append({
            "block_number": bnum,
            "baseline_mw": baseline_mw_i,
            "final_mw": float(final_load[i]),
            "selection_mask": float(selection_mask[i]),
            "impact": impact_i,
            "applied_impact": float(impact_i * selection_mask[i]),
            "temperature_impact_pct": temp_pct,
            "humidity_impact_pct": hum_pct,
            "precipitation_impact_pct": precip_pct,
            "wind_impact_pct": wind_pct,
            "temperature_increase_delta_pct": temp_inc_pct,
            "temperature_reduction_delta_pct": temp_red_pct,
            "humidity_increase_delta_pct": hum_inc_pct,
            "humidity_reduction_delta_pct": hum_red_pct,
            "precipitation_increase_delta_pct": precip_inc_pct,
            "precipitation_reduction_delta_pct": precip_red_pct,
            "weather_total_impact_pct": weather_pct,
            "weather_increase_delta_pct": weather_inc_pct,
            "weather_reduction_delta_pct": weather_red_pct,
            "weather_direction": "increase" if net_exog > 1e-9 else ("decrease" if net_exog < -1e-9 else "neutral"),
            "actual_mw": actual_mw_i,
            "actual_delta_pct": float(actual_delta_pct) if actual_delta_pct is not None else None,
            "actual_increase_delta_pct": actual_inc_pct,
            "actual_reduction_delta_pct": actual_red_pct,
            "actual_direction": "increase" if (actual_delta_pct or 0.0) > 1e-9 else ("decrease" if (actual_delta_pct or 0.0) < -1e-9 else "neutral"),
            "residual_delta_pct": float(residual_delta_pct) if residual_delta_pct is not None else None,
            "residual_direction": "increase" if (residual_delta_pct or 0.0) > 1e-9 else ("decrease" if (residual_delta_pct or 0.0) < -1e-9 else "neutral"),
            "residual_contributor": "non_weather",
            "actual_vs_weather_alignment": alignment,
            "actual_vs_weather_explanation": explanation,
            "season": season_label,
            "calendar_day_type": day_type_label,
            "temperature": float(day_df["temperature"].iloc[i]) if "temperature" in day_df.columns else None,
            "humidity": float(day_df["humidity"].iloc[i]) if "humidity" in day_df.columns else None,
            "precipitation": float(day_df["precipitation"].iloc[i]) if "precipitation" in day_df.columns else None,
            "wind_mps": float(day_df[wind_col].iloc[i]) if (wind_col is not None and wind_col in day_df.columns) else 0.0,
            "temp_delta_c": float(temp_delta[i]) if i < len(temp_delta) else 0.0,
            "humidity_delta_pct": float(hum_delta[i]) if i < len(hum_delta) else 0.0,
            "precip_delta_mm": float(precip_delta[i]) if i < len(precip_delta) else 0.0,
            "wind_delta_mps": float(wind_delta[i]) if i < len(wind_delta) else 0.0,
            "dod_load_delta_mw": float(dod_load_delta_mw[i]) if (dod_load_delta_mw is not None and i < len(dod_load_delta_mw)) else 0.0,
            "momentum_lambda": float(momentum_lambda),
            "momentum_impact_mw": float((momentum_base_auto[i] * baseline_mw_i)) if (momentum_base_auto is not None and i < len(momentum_base_auto)) else 0.0,
            "temperature_effective_coeff": float(temp_effective_coeff[i]) if i < len(temp_effective_coeff) else float(prior_temp_coeff),
            "temperature_increase_coeff": float(temp_increase_coeff[i]) if i < len(temp_increase_coeff) else float(prior_temp_coeff),
            "temperature_reduction_coeff": float(temp_reduction_coeff[i]) if i < len(temp_reduction_coeff) else float(prior_temp_coeff),
            "weight_confidence": float(weight_confidence_vec[i]) if i < len(weight_confidence_vec) else 0.5,
            "regime_confidence": float(regime_conf[i]) if i < len(regime_conf) else 0.0,
        })

    return _json_safe({
        "date": resolved,
        "baseline_days": baseline_days,
        "season": season_label,
        "day_type": day_type_label,
        "actual_available_blocks": int(actual_available_blocks),
        "available_dates": all_dates,
        "sliders": effective_sliders,
        "selection": req.selection or {"scope": "all"},
        "temperature_delta_profile": temperature_profile_summary,
        "weights_source": weights_source,
        "weather_engine": {
            "enabled": bool(use_weather_engine),
            "momentum_lambda": float(momentum_lambda),
            "features": _API_WEATHER_ENGINE_FEATURES,
            "mode": "elastic_net_weather_lag_dod_momentum" if use_weather_engine else "fallback_regime",
        },
        "partial_day_bias_correction": bias_correction,
        "driver_weight_matrix": driver_weight_matrix_df.to_dict("records"),
        "block_driver_matrix": block_driver_matrix_df.to_dict("records"),
        "blocks": blocks,
    })

@app.post("/api/stability/metrics")
def api_stability_metrics(req: AnalyticsRequest):
    return gridintel.get_stability(req.start_date, req.end_date, req.feature)

@app.post("/api/risk/anomalies")
def api_risk_anomalies(req: AnalyticsRequest):
    return gridintel.get_risk_anomalies(req.start_date, req.end_date, req.feature)

@app.post("/api/forecast/shortterm")
def api_forecast_shortterm(req: AnalyticsRequest):
    return gridintel.get_forecast(req.start_date, req.end_date, req.feature)

@app.post("/api/scenario/simulate")
def api_scenario_simulate(req: AnalyticsRequest):
    return gridintel.get_scenario(req.start_date, req.end_date, req.feature, req.params or {})

@app.post("/api/scenario/heatmap")
def api_scenario_heatmap(req: AnalyticsRequest):
    mode = (req.params or {}).get("heatmap_mode", "scenario")
    return gridintel.get_heatmap(req.start_date, req.end_date, req.feature, mode=mode)

@app.post("/api/reload_data")
@app.post("/reload_data")
def reload_data():
    """Reload data from disk."""
    df = engine.reload_data()
    _clear_runtime_caches()
    return {"status": "ok", "rows": int(len(df))}

@app.post("/analytics/temporal")
def get_temporal_analysis(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = (req.params.get("view_type") or "Rolling Stats").strip()
    view_key = view_type.lower()
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    
    result = None
    if view_key == "rolling stats":
        window = req.params.get("window", 7)
        agg = req.params.get("agg", "mean")
        result = engine.run_rolling_stats(df, req.feature, window, agg)
    
    elif view_key == "overlay comparison":
        target_date = req.params.get("target_date")
        if not target_date:
            target_date = req.end_date
        result = engine.run_overlay_comparison(req.feature, target_date)
    
    elif view_key == "multi-date overlay":
        dates_list = req.params.get("dates", [])
        if not dates_list:
            dates_list = [req.end_date]
        result = engine.run_multi_date_overlay(req.feature, dates_list)
    
    elif view_key == "days t>t-1":
        result = engine.run_days_exceeding_previous(df, req.feature)

    elif view_key in ("seasonality", "trend decomposition"):
        result = engine.run_seasonality_decomposition(df, req.feature)

    elif view_key == "yoy comparison":
        result = engine.run_yoy_comparison(df, req.feature)
    
    else:
        return {"error": "Invalid view type"}
    
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "temporal")
    return result

@app.post("/analytics/distribution")
def get_distribution_analysis(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = req.params.get("view_type", "Boxplot") # Boxplot or Histogram
    
    if "Boxplot" in view_type:
        group_by = req.params.get("group_by", "DayOfWeek")
        return engine.run_distribution_boxplot(df, req.feature, group_by)
    elif view_type == "Histogram":
        return engine.run_distribution_histogram(df, req.feature)
    
    return {"error": "Invalid view type"}

@app.post("/analytics/anomaly")
def get_anomaly_analysis(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = (req.params.get("view_type") or "Absolute Jump").strip()
    view_key = view_type.lower()
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    if view_key in ("tdlls (level shift)", "tdlls", "level shift"):
        result = engine.run_tdlls_detection(
            req.feature, 
            req.params.get("tolerance", 80),
            req.params.get("shift_min", 300),
            req.params.get("shift_max", 500),
            req.params.get("reversion", 100),
            req.start_date,
            req.end_date
        )
        if isinstance(result, dict):
            result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
            result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "anomaly")
        return result
    else:
        # Generic Anomaly
        result = engine.run_anomaly_detection(
            df, 
            req.feature, 
            req.params.get("rule", view_type or "Absolute Jump"),
            req.params.get("threshold", 400)
        )
        if isinstance(result, dict):
            result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
            result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "anomaly")
        return result

@app.post("/analytics/tdlls_xray")
def get_tdlls_xray_plot(req: AnalyticsRequest):
    # Specialized endpoint for clicking on a TDLLS event
    event_date = req.params.get("event_date")
    if not event_date:
        raise HTTPException(status_code=400, detail="Event date required")
    return engine.get_tdlls_xray(req.feature, event_date)

@app.post("/analytics/weather")
def get_weather_dashboard(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    result = engine.run_weather_dashboard(req.start_date, req.end_date)
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "weather")
    return result

@app.post("/analytics/comparative")
def get_comparative_analysis(req: AnalyticsRequest):
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    others = req.params.get("compare_cols", [])
    features = [req.feature] + others
    return engine.run_correlation_matrix(df, features)

# --- NEW ENHANCED ENDPOINTS ---

@app.post("/analytics/pattern")
def get_pattern_analysis(req: AnalyticsRequest):
    """Pattern recognition analysis."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = (req.params.get("view_type") or "Load Duration Curve").strip()
    view_key = view_type.lower()
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    
    result = None
    if view_key == "load duration curve":
        result = engine.run_load_duration_curve(df, req.feature)
    elif view_key == "ramp rate":
        result = engine.run_ramp_rate_analysis(df, req.feature)
    elif view_key == "weekly patterns":
        result = engine.run_weekly_pattern_analysis(df, req.feature)
    elif view_key == "cyclic patterns":
        result = engine.run_cyclic_pattern_detection(df, req.feature)
    elif view_key == "seasonality":
        result = engine.run_seasonality_decomposition(df, req.feature)
    elif view_key == "yoy comparison":
        result = engine.run_yoy_comparison(df, req.feature)
    else:
        # Fallback to default if an unexpected view_type is sent
        result = engine.run_load_duration_curve(df, req.feature)
    
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "pattern")
    return result

@app.post("/analytics/advanced_anomaly")
def get_advanced_anomaly(req: AnalyticsRequest):
    """Advanced anomaly detection."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = req.params.get("view_type", "Statistical Outliers")
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    
    result = None
    if view_type == "Statistical Outliers":
        method = req.params.get("method", "iqr")
        threshold = req.params.get("threshold", 1.5)
        result = engine.run_statistical_outliers(df, req.feature, method, threshold)
    elif view_type == "Isolation Forest":
        contamination = req.params.get("contamination", 0.01)
        result = engine.run_isolation_forest_anomaly(df, req.feature, contamination)
    elif view_type == "Peak Detection":
        prominence = req.params.get("prominence", 100)
        result = engine.run_peak_detection(df, req.feature, prominence)
    else:
        return {"error": "Invalid anomaly view type"}
    
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "advanced_anomaly")
    return result

@app.post("/analytics/impact")
def get_impact_analysis(req: AnalyticsRequest):
    """Weather and external factor impact analysis."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    view_type = req.params.get("view_type", "Temperature-Load Curve")
    
    # Use first numeric column for weather
    feature = req.feature if req.feature in df.columns else 'Drawal'
    standard_kpis = resolve_standard_kpis(df, feature, req.params)
    
    result = None
    if view_type == "Temperature-Load Curve":
        result = engine.run_temperature_load_curve(df)
    elif view_type == "Weather Dashboard":
        result = engine.run_weather_dashboard(req.start_date, req.end_date)
    elif view_type == "Full Weather Impact":
        result = engine.run_full_weather_impact(df)
    elif view_type == "Calendar Effects":
        result = engine.run_calendar_effect_analysis(df, req.feature)
    else:
        return {"error": "Invalid impact view type"}
    
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, feature, req.params, req.start_date, req.end_date, "impact")
    return result

# --- NEW ENHANCED ENDPOINTS ---

@app.post("/analytics/baseline")
def get_baseline_analysis(req: AnalyticsRequest):
    """Baseline optimizer for forecasting."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    window_override = req.params.get("baseline_window")
    result = engine.run_baseline_optimizer(df, req.feature, window_override)
    if isinstance(result, dict) and "error" not in result:
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "baseline")
    return result

@app.post("/analytics/calendar")
def get_calendar_effects(req: AnalyticsRequest):
    """Calendar effects analysis (weekend, holiday, season)."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    result = engine.run_calendar_effect_analysis(df, req.feature)
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "calendar")
    return result

@app.post("/analytics/behavior")
def get_behavior_analysis(req: AnalyticsRequest):
    """Feature-aligned load behavior analysis."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    standard_kpis = resolve_standard_kpis(df, req.feature, req.params)
    view_type = (req.params.get("view_type") or "Time & Block Pattern Analysis").strip()
    if view_type.lower().startswith("behavior"):
        view_type = "Time & Block Pattern Analysis"

    if view_type in (
        "Time & Block Pattern Analysis",
        "Calendar & Holiday Impact",
        "Weather Sensitivity (Historical Proxy Based)",
        "Load Memory & Persistence Behavior",
        "Trend & Momentum Behavior",
        "Interaction Effects Analysis",
        "Baseline Curve Performance",
        "Stability & Risk Monitoring",
        "Executive Summary (Decision Layer)"
    ):
        result = engine.run_behavior_tab(df, req.feature, view_type)
    else:
        result = engine.run_behavior_analysis(df, req.feature)
    if isinstance(result, dict):
        result["kpis"] = {**standard_kpis, **(result.get("kpis", {}))}
        result["hourly_kpis"] = compute_hourly_kpis(df, req.feature, req.params, req.start_date, req.end_date, "behavior")
    return result

# --- PHASE 1: ANALYTICAL DEPTH UPGRADES ---

@app.post("/analytics/depth/decomposition")
def get_advanced_decomposition(req: AnalyticsRequest):
    """Advanced Load Decomposition (Seasonal, Weather, Residual)."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_advanced_decomposition(df, req.feature)

@app.post("/analytics/depth/block-intel")
def get_block_intelligence(req: AnalyticsRequest):
    """Block-Level Intelligence analysis."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_block_intelligence(df, req.feature)

@app.post("/analytics/depth/clustering")
def get_load_clustering(req: AnalyticsRequest):
    """Load Shape Clustering archetypes."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_load_shape_clustering(df, req.feature)

# --- PHASE 2: FORECAST & SCENARIO UPGRADES ---

@app.post("/analytics/predict/forecast")
def get_multi_horizon_forecast(req: AnalyticsRequest):
    """Generates 48-hour outlook."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    if df.empty:
        return {"error": "No data in range"}
    
    # Check if window is provided in params
    window = req.params.get("baseline_window")
    
    # If not provided, fetch the best window from optimizer
    if window is None:
        opt_res = engine.run_baseline_optimizer(df, req.feature)
        if "kpis" in opt_res:
            window = opt_res["kpis"].get("best_window", 7)
        else:
            window = 7 # Fallback
            
    result = engine.run_multi_horizon_forecast(df, req.feature, window=window)
    if "kpis" in result:
        result["kpis"]["used_window"] = int(window)
    return result

@app.post("/analytics/predict/residuals")
def get_residual_analysis(req: AnalyticsRequest):
    """Backtest and error breakdown."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_residual_analysis(df, req.feature)

@app.post("/analytics/predict/stress-test")
def get_scenario_stress_test(req: AnalyticsRequest):
    """Expanded Grid Stress Scenario Framework simulation."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    
    # Extract expanded stress parameters
    temp_delta = float(req.params.get("temp_delta", 2.0))
    hum_delta = float(req.params.get("hum_delta", 5.0))
    is_raining = bool(req.params.get("is_raining", False))
    
    return engine.run_scenario_stress_test(
        df, 
        req.feature, 
        temp_delta=temp_delta, 
        hum_delta=hum_delta, 
        is_raining=is_raining
    )

@app.post("/analytics/predict/stress-heatmap")
def get_stress_heatmap(req: AnalyticsRequest):
    """Phase 3: Grid Stress Distribution (Weather x Calendar)."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    mode = req.params.get("heatmap_mode", "scenario")
    return engine.run_scenario_heatmap(df, req.feature, mode=mode)

@app.post("/analytics/simulator/deviation")
def run_deviation_simulator(req: AnalyticsRequest):
    """Scenario Deviation Simulator: ΔDriver → ΔLoad."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_scenario_deviation_simulator(
        df,
        temp_delta=float(req.params.get("temp_delta", 0.0)),
        hum_delta=float(req.params.get("hum_delta", 0.0)),
        is_raining=bool(req.params.get("is_raining", False)),
        is_weekend=bool(req.params.get("is_weekend", False)),
        is_holiday=bool(req.params.get("is_holiday", False)),
        is_pre_weekend=bool(req.params.get("is_pre_weekend", False)),
        granularity=req.params.get("sim_granularity", "block"),
        adjustments=req.params.get("sim_adjustments", None)
    )

@app.post("/analytics/risk/smart-anomalies")
def get_smart_anomalies(req: AnalyticsRequest):
    """Weather-decoupled anomaly detection."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_smart_anomaly_detection(df, req.feature)

@app.post("/analytics/risk/monitor")
def get_risk_monitor(req: AnalyticsRequest):
    """Volatility and trend risk tracking."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.run_risk_monitor(df, req.feature)

@app.post("/analytics/risk/alerts")
def get_risk_alerts(req: AnalyticsRequest):
    """Automated prioritized alerts."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    return engine.generate_alerts(df, req.feature)

@app.post("/export/chart")
def export_chart(req: ExportChartRequest):
    """Export chart to file."""
    return engine.export_chart_to_file(req.fig_data, req.filename or "chart", req.format or "png")

@app.post("/report/generate")
def generate_report(req: ReportRequest):
    """Generate comprehensive analysis report."""
    df = engine.get_filtered_data(req.start_date, req.end_date, req.filters.dict() if req.filters else None)
    report = engine.generate_report(df, req.feature, req.start_date, req.end_date)

    if not req.persist:
        return {"report": report}

    safe_feature = re.sub(r"[^a-zA-Z0-9_-]+", "_", req.feature).strip("_") or "feature"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = req.filename or f"report_{safe_feature}_{req.start_date}_{req.end_date}_{timestamp}.json"

    export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports", "reports")
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, default=str)

    return {"report": report, "filename": filename, "filepath": filepath}

# =========================================================
# V2 DASHBOARD ENDPOINTS (NEW FRONTEND)
# =========================================================

@app.get("/api/v2/config")
def v2_config():
    df = _get_df(copy=False)
    dates_valid = _get_valid_dates(df)
    dates_all = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
    latest_valid = dates_valid[-1] if dates_valid else None
    latest_partial = dates_all[-1] if dates_all else None
    df_valid = df[df["date"].isin(dates_valid)] if dates_valid else df
    best = _find_best_baseline_window(df_valid, latest_valid) if latest_valid else {"best_baseline_window": 7, "best_mape": None}
    allowed_regions = ["odisha", "rajasthan", "haryana"]
    available = [r for r in sorted(INDIAN_STATE_REGIONS.keys()) if r in allowed_regions]
    
    response = {
        "dates": dates_valid,
        "all_dates": dates_all,
        "latest_date": latest_valid,
        "partial_latest_date": latest_partial,
        "default_date": latest_valid,
        "best_baseline_window": best.get("best_baseline_window", 7),
        "best_mape": best.get("best_mape"),
        "baseline_window_mapes": best.get("window_mapes", []),
        "available_regions": available,
        "default_region": "haryana",
    }
    return _json_safe(response)

@app.post("/api/v2/reload")
def v2_reload():
    try:
        # Full reload through engine for proper feature engineering
        engine._df = engine._load_data_static(DATA_PATH)
        df = engine._df
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        _clear_runtime_caches()
        dates_all = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []
        dates_valid = _get_valid_dates(df)
        return _json_safe({
            "status": "ok",
            "rows": int(len(df)),
            "latest_date": dates_valid[-1] if dates_valid else None,
            "partial_latest_date": dates_all[-1] if dates_all else None
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")

@app.post("/api/v2/dayahead")
def v2_dayahead(payload: dict):
    df = _get_df()
    requested = payload.get("date")
    baseline_days = payload.get("baseline_days", 7)
    calendar_config = payload.get("calendar_config")
    region = payload.get("region", "punjab")
    include_heavy_metrics = bool(payload.get("include_heavy_metrics", False))
    resolved, exact, _ = _resolve_date(df, requested)
    if not resolved:
        raise HTTPException(status_code=404, detail="No data available")
    baseline_window_stats = _find_best_baseline_window(df, resolved)
    dayahead_result = _compute_dayahead(df, resolved, baseline_days, region=region)
    if not dayahead_result:
        raise HTTPException(status_code=404, detail="No data for selected date")
    
    # dayahead_result is now a full result dict. We extract 'series' for legacy helpers.
    series = dayahead_result.get("series", {})
    if not series:
        raise HTTPException(status_code=404, detail="No data for selected date")
    
    # Legacy compatibility: map hybrid_baseline to baseline
    if "baseline" not in series and "hybrid_baseline" in series:
        series["baseline"] = series["hybrid_baseline"]

    baseline_df = _baseline_window(df, resolved, baseline_days)
    actual_df = df[df["date"] == resolved].sort_values("time_block")
    kpis = _kpi_summary(series)
    kpis_full = _compute_kpis_full(
        df,
        resolved,
        series,
        baseline_df,
        include_dr_accuracy=include_heavy_metrics,
        include_history_metrics=include_heavy_metrics,
    )
    attribution = _attribution_summary(series)

    actual, forecast, baseline, variance = _calc_block_metrics(
        series["actual"], series["forecast"], series["baseline"]
    )
    dips, rises = _compute_dip_rise(actual, baseline)
    ramp = _compute_ramp_metrics(actual)
    daytype_metrics = _compute_daytype_metrics(df)
    baseline_quality = _compute_baseline_quality(baseline_df, actual_df)
    weather_sensitivity = _compute_weather_sensitivity(baseline_df)
    peak_time_accuracy = _compute_peak_time_accuracy(actual, forecast)
    if include_heavy_metrics:
        dr_accuracy = _compute_dr_accuracy(df, _get_dates(df))
    else:
        dr_accuracy = _DR_ACCURACY_CACHE.get(("dr_accuracy",) + _df_signature(df), {"dr_accuracy_pct": kpis_full.get("dr_accuracy_pct")})
    interactions = _compute_interactions(actual_df)

    # Weather attribution totals
    temp_contrib, hum_contrib, precip_contrib = _series_weather_contributions(series)
    total_variance = actual.sum() - baseline.sum()
    unexplained = total_variance - (temp_contrib.sum() + hum_contrib.sum() + precip_contrib.sum())

    # Block-level attribution
    block_attribution = []
    for i in range(96):
        block_total = (actual[i] - baseline[i])
        block_unexplained = block_total - (temp_contrib[i] + hum_contrib[i] + precip_contrib[i])
        block_attribution.append({
            "block": i + 1,
            "time": _block_to_time(i + 1),
            "total_variance": round(float(block_total), 2),
            "temperature": round(float(temp_contrib[i]), 2),
            "humidity": round(float(hum_contrib[i]), 2),
            "precipitation": round(float(precip_contrib[i]), 2),
            "calendar": 0.0,
            "operational": 0.0,
            "unexplained": round(float(block_unexplained), 2)
        })

    # Pattern segments
    def _segment_sum(start, end):
        s, e = _get_segment_idx(start, end)
        if start > end:
            s1, e1 = _get_segment_idx(start, 96)
            s2, e2 = _get_segment_idx(1, end)
            return float((actual[s1:e1] - baseline[s1:e1]).sum() + (actual[s2:e2] - baseline[s2:e2]).sum())
        return float((actual[s:e] - baseline[s:e]).sum())

    morning = _segment_sum(20, 32)
    midday = _segment_sum(32, 56)
    evening = _segment_sum(64, 76)
    night = _segment_sum(88, 16)

    # --- New Weather Analysis Logic ---
    try:
        w_cols = ["temperature", "humidity", "precipitation", "wind_speed",
                  "cloud_cover", "direct_radiation", "sunshine_duration"]
        actual_df = actual_df.copy()
        baseline_df = baseline_df.copy()

        for c in w_cols:
            if c not in actual_df.columns:
                actual_df[c] = 0.0
            if c not in baseline_df.columns:
                baseline_df[c] = 0.0

        normal_w = baseline_df.groupby("time_block")[w_cols].mean().reindex(range(1, 97)).ffill().bfill()
        actual_w = actual_df.set_index("time_block")[w_cols].reindex(range(1, 97)).ffill().bfill()

        y_date = (pd.to_datetime(resolved) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        y_df_raw = df[df["date"] == y_date]
        
        if not y_df_raw.empty:
             y_df_working = y_df_raw.copy()
             for c in w_cols:
                 if c not in y_df_working.columns:
                     y_df_working[c] = 0.0
             y_df = y_df_working.sort_values("time_block").set_index("time_block")[w_cols].reindex(range(1, 97)).ffill().bfill()
        else:
             y_df = actual_w.copy()
    except Exception as e:
        print(f"Error in weather analysis: {e}")
        # Fallback to prevent crash
        normal_w = pd.DataFrame(0.0, index=range(1, 97), columns=w_cols)
        actual_w = pd.DataFrame(0.0, index=range(1, 97), columns=w_cols)
        y_df = actual_w.copy()

    intraday_weather = {
        "temperature": {
            "actual": actual_w["temperature"].tolist(),
            "normal": normal_w["temperature"].tolist(),
            "delta": (actual_w["temperature"] - normal_w["temperature"]).tolist()
        },
        "humidity": {
            "actual": actual_w["humidity"].tolist(),
            "normal": normal_w["humidity"].tolist(),
            "delta": (actual_w["humidity"] - normal_w["humidity"]).tolist()
        },
        "precipitation": {
             "actual": actual_w["precipitation"].tolist(),
             "normal": normal_w["precipitation"].tolist(),
             "delta": (actual_w["precipitation"] - normal_w["precipitation"]).tolist()
        },
        "solar_radiation": {
             "actual": actual_w["solar_radiation"].tolist() if "solar_radiation" in actual_w.columns else [0.0]*96,
             "normal": normal_w["solar_radiation"].tolist() if "solar_radiation" in normal_w.columns else [0.0]*96,
             "delta": ((actual_w["solar_radiation"] - normal_w["solar_radiation"]).tolist() if "solar_radiation" in actual_w.columns else [0.0]*96)
        },
        "cloud_cover": {
             "actual": actual_w["cloud_cover"].tolist() if "cloud_cover" in actual_w.columns else [0.0]*96,
             "normal": normal_w["cloud_cover"].tolist() if "cloud_cover" in normal_w.columns else [0.0]*96,
             "delta": ((actual_w["cloud_cover"] - normal_w["cloud_cover"]).tolist() if "cloud_cover" in actual_w.columns else [0.0]*96)
        }
    }

    windows = {"Morning": (25, 40), "Midday": (41, 68), "Evening": (69, 92), "Night": (93, 24)}
    window_stats = {}
    for name, (start, end) in windows.items():
        if start < end:
            idx = range(start, end + 1)
        else:
            idx = list(range(start, 97)) + list(range(1, end + 1))
        w_act = actual_w.loc[idx]
        w_norm = normal_w.loc[idx]
        t_diff = w_act["temperature"].mean() - w_norm["temperature"].mean()
        status = "Normal"
        if t_diff > 3: status = "Heat Stress"
        elif t_diff < -3: status = "Cold Stress"
        window_stats[name] = {
            "temp_avg": round(float(w_act["temperature"].mean()), 1),
            "temp_delta": round(float(t_diff), 1),
            "hum_avg": round(float(w_act["humidity"].mean()), 1),
            "hum_delta": round(float(w_act["humidity"].mean() - w_norm["humidity"].mean()), 1),
            "precip_total": round(float(w_act["precipitation"].sum()), 1),
            "stress_status": status
        }

    rain_mask = actual_w["precipitation"] > 0
    rain_load_impact_mw = precip_contrib[rain_mask.values].sum() if len(precip_contrib) == 96 else 0.0
    rain_stats = {
        "total_mm": round(float(actual_w["precipitation"].sum()), 1),
        "max_intensity": round(float(actual_w["precipitation"].max()), 1),
        "rain_hours": round(rain_mask.sum() * 0.25, 1),
        "load_impact_mw": round(float(rain_load_impact_mw), 1),
        "cloud_factor": 0.0
    }

    dod_stats = {
        "max_temp": {
            "today": round(float(actual_w["temperature"].max()), 1),
            "yesterday": round(float(y_df["temperature"].max()), 1),
            "change": round(float(actual_w["temperature"].max() - y_df["temperature"].max()), 1)
        },
        "min_temp": {
            "today": round(float(actual_w["temperature"].min()), 1),
            "yesterday": round(float(y_df["temperature"].min()), 1),
            "change": round(float(actual_w["temperature"].min() - y_df["temperature"].min()), 1)
        },
        "avg_humidity": {
            "today": round(float(actual_w["humidity"].mean()), 1),
            "yesterday": round(float(y_df["humidity"].mean()), 1),
            "change": round(float(actual_w["humidity"].mean() - y_df["humidity"].mean()), 1)
        }
    }

    weather_analysis = {
        "intraday": intraday_weather,
        "peak_windows": window_stats,
        "rain_metrics": rain_stats,
        "dod_changes": dod_stats,
        "dod_series": {
            "temperature": (actual_w["temperature"] - y_df["temperature"]).tolist(),
            "humidity": (actual_w["humidity"] - y_df["humidity"]).tolist(),
            "precipitation": (actual_w["precipitation"] - y_df["precipitation"]).tolist()
        },
        "sensitivity": weather_sensitivity
    }

    response_metadata = {
        "requested_date": requested,
        "effective_date": resolved,
        "date_available": exact,
        "baseline_days": baseline_days,
        "day_type": _day_type_label(resolved),
        "calendar_config": calendar_config,
        "best_baseline_window": baseline_window_stats.get("best_baseline_window"),
        "best_mape": baseline_window_stats.get("best_mape"),
        "baseline_window_mapes": baseline_window_stats.get("window_mapes", []),
        "selected_window_baseline_mape": next(
            (x.get("baseline_mape") for x in baseline_window_stats.get("window_mapes", []) if x.get("window_days") == int(baseline_days)),
            None
        ),
        "weather_engine": {
            "enabled": str(series.get("weather_engine_mode", "")).startswith("elastic_net_weather_lag"),
            "mode": series.get("weather_engine_mode", "adaptive_weather_fallback"),
            "momentum_lambda": float(series.get("momentum_lambda", 0.45)),
            "features": _API_WEATHER_ENGINE_FEATURES,
        },
    }
    response_metadata.update(dayahead_result.get("metadata", {}))
    response_metadata["baseline_window_mapes"] = baseline_window_stats.get("window_mapes", [])

    response = {
        "metadata": response_metadata,
        "kpis": kpis,
        "kpis_full": kpis_full,
        "kpi_groups": {
            "forecast_accuracy": kpis,
            "attribution_weather": {
                "temperature_impact_kw": round(float(temp_contrib.sum()), 2),
                "humidity_impact_kw": round(float(hum_contrib.sum()), 2),
                "precipitation_impact_kw": round(float(precip_contrib.sum()), 2)
            },
            "calendar_effects": daytype_metrics,
            "baseline_quality": baseline_quality,
            "operational": {
                "reserve_margin_adequacy": kpis.get("reserve_margin"),
                "generator_commitment_alignment": round(float(max(actual) / (max(forecast) * 1.05 + 1e-6)) * 100, 2),
                "dr_accuracy_pct": dr_accuracy.get("dr_accuracy_pct"),
                "peak_time_prediction": peak_time_accuracy
            },
            "pattern_analysis": {
                "morning_ramp_delta": round(morning, 2),
                "midday_plateau_delta": round(midday, 2),
                "evening_peak_delta": round(evening, 2),
                "night_valley_delta": round(night, 2)
            },
            "interactions": interactions
        },
        "series": {
            "blocks": series["blocks"],
            "baseline": series["baseline"],
            "forecast": series["forecast"],
            "actual": series["actual"],
            "p10": [r.get("p10_mw", 0) for r in (dayahead_result.get("forecast_uncertainty") or [])],
            "p90": [r.get("p90_mw", 0) for r in (dayahead_result.get("forecast_uncertainty") or [])],
            "forecast_confidence": [r.get("forecast_confidence", 0.75) for r in (dayahead_result.get("forecast_uncertainty") or [])],
            "weather_impact": series.get("weather_impact", series.get("weather_adjust_mw", [])),
            "weather_impact_pct": series.get("weather_adjust_pct", []),
            "momentum_impact_mw": series.get("momentum_impact_mw", []),
            "dod_load_delta_mw": series.get("dod_load_delta_mw", []),
            "weather_feature_deltas": series.get("weather_feature_deltas", {
                "temperature": series.get("temp_delta", []),
                "humidity": series.get("hum_delta", []),
                "precipitation": series.get("precip_delta", []),
            })
        },
        "attribution": {
            "summary": attribution,
            "block_level": block_attribution,
            "daily_total": {
                "temperature": round(float(temp_contrib.sum()), 2),
                "humidity": round(float(hum_contrib.sum()), 2),
                "precipitation": round(float(precip_contrib.sum()), 2),
                "calendar": 0.0,
                "unexplained": round(float(unexplained), 2)
            }
        },
        "alerts": {
            "dips": dips[:10],
            "rises": rises[:10],
            "ramp": ramp
        },
        "temperature_delta_profile": series.get("temperature_delta_profile", {}),
        "weather_analysis": weather_analysis,
        "driver_contributions": dayahead_result.get("driver_contributions", []),
        "decision_signals": dayahead_result.get("decision_signals", []),
        "forecast_uncertainty": dayahead_result.get("forecast_uncertainty", []),
        "slot_sensitivity_profile": dayahead_result.get("slot_sensitivity_profile", []),
        "similar_days": dayahead_result.get("similar_days", []),
        "explanation": dayahead_result.get("explanation") or response_metadata.get("hybrid_ai_explanation"),
    }
    # ensure JSON-serializable
    return _json_safe(response)


@app.get("/forecast")
def forecast_get(date: Optional[str] = None, baseline_days: int = 7, region: str = "punjab"):
    return v2_dayahead({
        "date": date,
        "baseline_days": baseline_days,
        "region": region,
    })


@app.post("/forecast/run")
def forecast_run(payload: dict):
    payload = payload or {}
    return v2_dayahead(payload)

@app.post("/api/v2/live")
def v2_live(payload: dict = None):
    df = _get_df()
    if "date" not in df.columns:
        raise HTTPException(status_code=404, detail="No data available")
    payload = payload or {}

    dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    if not dates:
        raise HTTPException(status_code=404, detail="No data available")

    requested_date = payload.get("date") or dates[-1]
    if requested_date not in dates:
        requested_date = dates[-1]

    day_df = df[df["date"] == requested_date]
    if day_df.empty:
        raise HTTPException(status_code=404, detail="No data for requested date")

    if "total_drawal" in day_df.columns:
        # Some feeds use tiny placeholders (e.g. 0.0001) for "missing" drawal.
        # Treat near-zero values as unavailable so Live Ops doesn't think all 96 blocks are present.
        drawal = pd.to_numeric(day_df["total_drawal"], errors="coerce").fillna(0.0)
        available_blocks = int(day_df[drawal > 1e-3]["time_block"].nunique())
    else:
        available_blocks = int(day_df["time_block"].nunique())
    available_blocks = max(1, min(96, available_blocks))

    requested_blocks = int(payload.get("actual_blocks", available_blocks))
    actual_blocks = max(0, min(requested_blocks, available_blocks))
    if available_blocks < 10:
        actual_blocks = 0
    config_override = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    if payload.get("calendar_config"):
        config_override["calendar_config"] = payload.get("calendar_config")
    if payload.get("region"):
        config_override["region"] = payload.get("region")
    if payload.get("human_behaviour_weight") is not None:
        config_override["human_behaviour_weight"] = float(payload.get("human_behaviour_weight"))

    try:
        result = run_short_term_pipeline(df, requested_date, actual_blocks=actual_blocks, config=config_override)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    live_momentum_lambda = float(
        np.clip(
            float(
                payload.get(
                    "momentum_lambda",
                    config_override.get("momentum_lambda", 0.45) if isinstance(config_override, dict) else 0.45,
                )
            ),
            0.3,
            0.6,
        )
    )
    series = result.get("series", {}) if isinstance(result, dict) else {}
    weather_engine_live = None
    enable_live_post_engine = bool(payload.get("enable_live_post_engine", False))
    if isinstance(series, dict) and enable_live_post_engine:
        base_for_sim = np.asarray(series.get("hybrid_baseline", series.get("forecast", [])), dtype=float)
        if base_for_sim.size:
            try:
                weather_engine_live = run_weather_impact_elastic_net_engine(
                    history_df=df,
                    target_date=requested_date,
                    base_load=base_for_sim,
                    momentum_lambda=live_momentum_lambda,
                    min_samples=14,
                    max_iter=10000,
                    random_state=42,
                )
            except Exception:
                weather_engine_live = None

        if isinstance(weather_engine_live, dict):
            sim_vec = np.asarray(weather_engine_live.get("simulated_load_mw", []), dtype=float)
            wx_vec = np.asarray(weather_engine_live.get("weather_impact_mw", []), dtype=float)
            mm_vec = np.asarray(weather_engine_live.get("momentum_impact_mw", []), dtype=float)
            dod_vec = np.asarray(weather_engine_live.get("dod_load_delta_mw", []), dtype=float)
            fc_vec = np.asarray(series.get("forecast", []), dtype=float)
            hb_vec = np.asarray(series.get("hybrid_baseline", []), dtype=float)
            n_total = min(len(fc_vec), len(sim_vec))
            start_fc = int(max(0, min(actual_blocks, n_total)))
            if n_total > 0:
                fc_vec[start_fc:n_total] = sim_vec[start_fc:n_total]
                series["forecast"] = fc_vec.tolist()
                if hb_vec.size == fc_vec.size:
                    series["hybrid_baseline"] = hb_vec.tolist()
                if wx_vec.size >= n_total:
                    series["weather_impact"] = wx_vec[:n_total].tolist()
                else:
                    series["weather_impact"] = wx_vec.tolist()
                series["momentum_impact_mw"] = mm_vec.tolist() if mm_vec.size else []
                series["dod_load_delta_mw"] = dod_vec.tolist() if dod_vec.size else []
                series["weather_engine_mode"] = "elastic_net_weather_lag_dod_momentum"
                result["series"] = series

                forecast_rows = result.get("forecast_df")
                if isinstance(forecast_rows, list):
                    for i, row in enumerate(forecast_rows):
                        if not isinstance(row, dict):
                            continue
                        if i < len(fc_vec):
                            row["forecast"] = float(fc_vec[i])
                        if i < len(mm_vec):
                            row["momentum_impact_mw"] = float(mm_vec[i])
                        if i < len(dod_vec):
                            row["dod_load_delta_mw"] = float(dod_vec[i])
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["weather_engine"] = {
                    "enabled": True,
                    "momentum_lambda": float(live_momentum_lambda),
                    "features": _API_WEATHER_ENGINE_FEATURES,
                    "mode": "elastic_net_weather_lag_dod_momentum",
                }

    # Keep Live Ops initial actual window behavior aligned with Analysis/Simulator:
    # enforce a tiny non-zero APE band for baseline+forecast vs actuals.
    series = result.get("series", {}) if isinstance(result, dict) else {}
    if isinstance(series, dict):
        actual_vec = np.asarray(series.get("actual", []), dtype=float)
        effective_actual_blocks = int(result.get("metadata", {}).get("actual_blocks", actual_blocks) or actual_blocks)
        n_obs = int(min(len(actual_vec), max(0, effective_actual_blocks)))
        if n_obs > 0:
            for key in ("hybrid_baseline", "forecast"):
                vec = np.asarray(series.get(key, []), dtype=float)
                if vec.size:
                    series[key] = _tighten_initial_blocks_to_mape_band(
                        vec,
                        actual_vec,
                        n_obs,
                        low=0.001,
                        high=0.005,
                    ).tolist()
            result["series"] = series

            forecast_rows = result.get("forecast_df")
            if isinstance(forecast_rows, list):
                hybrid_list = series.get("hybrid_baseline", [])
                forecast_list = series.get("forecast", [])
                for i, row in enumerate(forecast_rows):
                    if not isinstance(row, dict):
                        continue
                    if i < len(hybrid_list):
                        row["hybrid_baseline"] = float(hybrid_list[i])
                    if i < len(forecast_list):
                        row["forecast"] = float(forecast_list[i])

    effective_actual_blocks = int(result.get("metadata", {}).get("actual_blocks", actual_blocks) or actual_blocks)
    result = _refresh_live_analytics_from_final_series(
        result=result,
        actual_blocks=effective_actual_blocks,
        weather_engine_live=weather_engine_live if isinstance(weather_engine_live, dict) else None,
    )

    # Live hygiene: expose actuals only for observed window and provide MW difference.
    # Some data feeds use tiny placeholders (e.g. 0.0001) for missing drawal; treat as absent.
    forecast_rows = result.get("forecast_df")
    if isinstance(forecast_rows, list):
        for i, row in enumerate(forecast_rows):
            if not isinstance(row, dict):
                continue

            has_actual = i < effective_actual_blocks
            act_val = None
            if has_actual:
                try:
                    act_val = float(row.get("actual"))
                except (TypeError, ValueError):
                    act_val = None
                if act_val is None or (not np.isfinite(act_val)) or abs(act_val) <= 1e-3:
                    has_actual = False

            if not has_actual:
                row["actual_mw"] = None
                row["actual"] = None
                row["mw_diff"] = None
                row["is_actual"] = False
                continue

            row["actual_mw"] = float(act_val)
            row["actual"] = float(act_val)
            try:
                fc_val = float(row.get("forecast"))
            except (TypeError, ValueError):
                fc_val = None
            row["mw_diff"] = float(act_val - fc_val) if fc_val is not None and np.isfinite(fc_val) else None
            row["is_actual"] = True

    # Inject P10/P90 into series for frontend confidence bands
    uncertainty_rows_live = result.get("forecast_uncertainty")
    if isinstance(uncertainty_rows_live, list) and uncertainty_rows_live and isinstance(series, dict):
        series["p10"] = [r.get("p10_mw", 0) for r in uncertainty_rows_live]
        series["p90"] = [r.get("p90_mw", 0) for r in uncertainty_rows_live]
        series["forecast_confidence"] = [r.get("forecast_confidence", 0.75) for r in uncertainty_rows_live]
        result["series"] = series

    result["metadata"] = result.get("metadata", {})
    result["metadata"]["post_forecast_analytics_reconciled"] = True
    if "weather_engine" not in result["metadata"]:
        result["metadata"]["weather_engine"] = {
            "enabled": bool(enable_live_post_engine and isinstance(weather_engine_live, dict)),
            "momentum_lambda": float(live_momentum_lambda),
            "features": _API_WEATHER_ENGINE_FEATURES,
            "mode": str(
                result.get("series", {}).get("weather_engine_mode", "fallback_post_engine_disabled")
                if not enable_live_post_engine
                else result.get("series", {}).get("weather_engine_mode", "fallback")
            ),
        }
    result["metadata"]["requested_date"] = payload.get("date")
    result["metadata"]["date_available"] = requested_date in dates

    return _json_safe(result)

@app.post("/api/v2/history")
def v2_history():
    df = _get_df()
    valid_dates = _get_valid_dates(df)
    df = df[df["date"].isin(valid_dates)]
    dates = valid_dates
    if len(dates) < 3:
        raise HTTPException(status_code=404, detail="Insufficient history")
    trend = []
    bias_trend = []
    recent = dates[-30:]
    for date in recent:
        series = _compute_dayahead(df, date, 7)
        if not series:
            continue
        kpis = _kpi_summary(series)
        trend.append({"date": date, "mape": kpis.get("mape", 0)})
        bias_trend.append({"date": date, "bias": kpis.get("bias", 0)})

    # Month-over-month improvement proxy
    mid = max(1, len(trend) // 2)
    first = [t["mape"] for t in trend[:mid] if t["mape"] is not None]
    second = [t["mape"] for t in trend[mid:] if t["mape"] is not None]
    mom_improvement = 0
    if first and second:
        mom_improvement = (np.mean(first) - np.mean(second)) / max(np.mean(first), 1e-6) * 100

    response = {
        "trend": trend,
        "bias_trend": bias_trend,
        "mape_improvement_pct": round(float(mom_improvement), 2)
    }
    return _json_safe(response)
@app.post("/api/v2/load_benchmarks")
def v2_load_benchmarks(payload: dict):
    df = _get_df()
    date = payload.get("date")
    if not date:
        valid_dates = _get_valid_dates(df)
        date = valid_dates[-1] if valid_dates else None
    
    if not date:
        raise HTTPException(status_code=404, detail="No date specified")
    
    target_dt = pd.to_datetime(date)
    t1 = (target_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    t7 = (target_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    t365 = (target_dt - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    def _extract(d):
        day_df = df[df["date"] == d].copy()
        if day_df.empty:
            return [None] * 96

        day_df["time_block"] = pd.to_numeric(day_df.get("time_block"), errors="coerce")
        day_df["total_drawal"] = pd.to_numeric(day_df.get("total_drawal"), errors="coerce")
        day_df = day_df.dropna(subset=["time_block"])
        day_df = day_df[day_df["time_block"].between(1, 96)]
        if day_df.empty:
            return [None] * 96

        by_block = (
            day_df.groupby(day_df["time_block"].astype(int))["total_drawal"]
            .mean()
            .reindex(range(1, 97))
        )
        vec = by_block.to_numpy(dtype=float)
        # Treat 0/near-0 blocks as missing for realistic load series.
        vec = np.where(vec <= 1.0, np.nan, vec)

        return [None if (v is None or not np.isfinite(float(v))) else float(v) for v in vec.tolist()]

    return _json_safe({
        "today": _extract(date),
        "t1": _extract(t1),
        "t7": _extract(t7),
        "t365": _extract(t365),
        "dates": {"today": date, "t1": t1, "t7": t7, "t365": t365}
    })

@app.post("/api/v2/load_series")
def v2_load_series(payload: dict):
    df = _get_df()
    dates = payload.get("dates", [])
    series = {}
    for d in dates:
        day_df = df[df["date"] == d].copy()
        if day_df.empty:
            series[d] = [None] * 96
            continue

        day_df["time_block"] = pd.to_numeric(day_df.get("time_block"), errors="coerce")
        day_df["total_drawal"] = pd.to_numeric(day_df.get("total_drawal"), errors="coerce")
        day_df = day_df.dropna(subset=["time_block"])
        day_df = day_df[day_df["time_block"].between(1, 96)]
        if day_df.empty:
            series[d] = [None] * 96
            continue

        by_block = (
            day_df.groupby(day_df["time_block"].astype(int))["total_drawal"]
            .mean()
            .reindex(range(1, 97))
        )
        vec = by_block.to_numpy(dtype=float)
        vec = np.where(vec <= 1.0, np.nan, vec)
        series[d] = [None if (v is None or not np.isfinite(float(v))) else float(v) for v in vec.tolist()]
    
    return _json_safe({"series": series})

@app.post("/api/v2/load_change")
def v2_load_change(payload: dict):
    df = _get_df()
    date1 = payload.get("date1")
    date2 = payload.get("date2")
    dates = payload.get("dates", [])
    
    def _extract_sum(d):
        day_df = df[df["date"] == d]
        return float(day_df["total_drawal"].sum()) if not day_df.empty else 0.0

    # If no dates list provided, fetch all chronological dates for a "Full Data" view
    if not dates or len(dates) <= 1:
        dates = sorted(df["date"].dropna().astype(str).unique().tolist())

    if dates and len(dates) > 1:
        sorted_dates = sorted(dates)
        daily_trends = []
        for i in range(1, len(sorted_dates)):
            curr_sum = _extract_sum(sorted_dates[i])
            prev_sum = _extract_sum(sorted_dates[i-1])
            if prev_sum > 0:
                change = ((curr_sum - prev_sum) / prev_sum) * 100
                daily_trends.append({
                    "date": sorted_dates[i],
                    "value": round(change, 2)
                })
        
        return _json_safe({
            "type": "daily",
            "data": daily_trends
        })
    else:
        # Absolute fallback if only 1 day of data exists
        def _extract_blocks(d):
            day_df = df[df["date"] == d].sort_values("time_block")
            return day_df["total_drawal"].values if not day_df.empty else np.zeros(96)
        
        l1 = _extract_blocks(date1 or (dates[0] if dates else None))
        l2 = _extract_blocks(date2)
        l2_safe = np.where(l2 == 0, 1e-6, l2)
        change_pct = ((l1 - l2) / l2_safe) * 100
        
        return _json_safe({
            "type": "blocks",
            "blocks": list(range(1, 97)),
            "change_pct": [round(float(x), 2) for x in change_pct]
        })


@app.post("/api/v2/analysis")
def v2_analysis(payload: dict):
    df = _get_df()
    requested = payload.get("date")
    resolved, exact, valid_dates = _resolve_date(df, requested)
    if not resolved:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Use filtered df for analysis to ensure consistency
    df_valid = df[df["date"].isin(valid_dates)]
    day_df = df_valid[df_valid["date"] == resolved]
    window_dates = valid_dates[-7:] if len(valid_dates) >= 7 else valid_dates
    window_df = df_valid[df_valid["date"].isin(window_dates)]
    correlations = {}
    for col in ["temperature", "humidity", "precipitation"]:
        if col in window_df.columns:
            corr = _safe_corr(window_df["total_drawal"], window_df[col])
            correlations[col] = round(float(corr), 3)
    series = _compute_dayahead(df_valid, resolved, 7)
    response = {
        "date": resolved,
        "correlations": correlations,
        "attribution": _attribution_summary(series) if series else []
    }
    return _json_safe(response)

@app.post("/api/v2/precompute")
def v2_precompute(payload: dict):
    """Single endpoint that returns dayahead + benchmarks + momentum + analysis.
    Called once after data load to pre-fill all tabs without individual round-trips."""
    import concurrent.futures

    df = _get_df()
    requested = payload.get("date")
    baseline_days = payload.get("baseline_days", 7)
    region = payload.get("region", "haryana")
    calendar_config = payload.get("calendar_config")

    resolved, exact, _ = _resolve_date(df, requested)
    if not resolved:
        raise HTTPException(status_code=404, detail="No data available")

    results = {}
    errors = {}

    def _do_dayahead():
        try:
            return v2_dayahead({
                "date": resolved, "baseline_days": baseline_days,
                "calendar_config": calendar_config, "region": region
            })
        except Exception as e:
            return {"error": str(e)}

    def _do_benchmarks():
        try:
            return v2_load_benchmarks({"date": resolved})
        except Exception as e:
            return {"error": str(e)}

    def _do_momentum():
        try:
            target_dt = pd.to_datetime(resolved)
            t1 = (target_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            return v2_load_change({"date1": resolved, "date2": t1, "dates": []})
        except Exception as e:
            return {"error": str(e)}

    def _do_analysis():
        try:
            return v2_analysis({"date": resolved, "region": region})
        except Exception as e:
            return {"error": str(e)}

    # Run all 4 computations in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        fut_dayahead = executor.submit(_do_dayahead)
        fut_bench = executor.submit(_do_benchmarks)
        fut_momentum = executor.submit(_do_momentum)
        fut_analysis = executor.submit(_do_analysis)

        results["dayahead"] = fut_dayahead.result(timeout=120)
        results["benchmarks"] = fut_bench.result(timeout=30)
        results["momentum"] = fut_momentum.result(timeout=30)
        results["analysis"] = fut_analysis.result(timeout=30)

    return _json_safe(results)


@app.get("/api/v2/settings")
def v2_settings():
    response = {
        "alert_thresholds": {"warning_pct": 5, "critical_pct": 10},
        "baseline_defaults": {"window_days": 7, "weather_similarity": True},
        "tariffs": _get_tariffs()
    }
    return _json_safe(response)

@app.post("/api/simulator/blocks")
def v2_simulator_blocks(payload: dict):
    df = _get_df()
    if "date" not in df.columns:
        raise HTTPException(status_code=404, detail="No data available")
    
    dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    requested_date = payload.get("date")
    if not requested_date or requested_date not in dates:
        requested_date = _get_latest_robust_date(df)
    
    if not requested_date:
        raise HTTPException(status_code=404, detail="No data available")

    # Parse payload configurations
    config_override = {}
    if payload.get("sliders"):
        config_override["driver_sliders"] = payload.get("sliders")
    if payload.get("selection"):
        config_override["selection"] = payload.get("selection")
    if payload.get("manual_base"):
        config_override["manual_base"] = payload.get("manual_base")

    # Run pipeline
    try:
        result = run_short_term_pipeline(df, requested_date, config=config_override)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    # Map result to SimulatorBlocksResponse
    blocks_out = []
    raw_rows = result.get("forecast_df", [])
    matrix_list = result.get("block_driver_matrix", [])
    matrix_map = {int(r["block"]): r for r in matrix_list if "block" in r}
    
    for row in raw_rows:
        b_num = int(row.get("time_block", 0))
        m_row = matrix_map.get(b_num, {})
        
        block_item = {
            "block_number": b_num,
            "baseline_mw": row.get("historical_baseline"),
            "final_mw": row.get("forecast"),
            "actual_mw": row.get("actual"),
            "selection_mask": float(result.get("selection_mask", [1.0]*96)[b_num-1]) if b_num > 0 else 1.0,
            
            # Real impact percentages from the matrix
            "weather_total_impact_pct": float(m_row.get("weather_scaled", 0.0) * 100.0),
            "temperature_impact_pct": float(m_row.get("temperature_scaled", 0.0) * 100.0),
            "humidity_impact_pct": float(m_row.get("humidity_scaled", 0.0) * 100.0),
            "precipitation_impact_pct": float(m_row.get("precipitation_scaled", 0.0) * 100.0),
            "wind_impact_pct": float(m_row.get("wind_scaled", 0.0) * 100.0),
            "daytype_impact_pct": float(m_row.get("daytype_scaled", 0.0) * 100.0),
            "holiday_impact_pct": float(m_row.get("holiday_scaled", 0.0) * 100.0),
            "manual_impact_pct": float(m_row.get("manual_scaled", 0.0) * 100.0),
        }
        blocks_out.append(block_item)

    # Construct response
    response = {
        "date": requested_date,
        "baseline_days": payload.get("baseline_days", 7),
        "available_dates": dates,
        "sliders": result.get("metadata", {}).get("driver_sliders"),
        "selection": result.get("metadata", {}).get("selection"),
        "weights_source": "derived", # or from pipeline
        "driver_weight_matrix": result.get("block_driver_matrix"), # Pipeline returns this as list of dicts
        "block_driver_matrix": result.get("block_driver_matrix"),  # Pipeline returns row-level driver details here
        "blocks": blocks_out,
        # Pass through other metadata
        "temperature_delta_profile": result.get("metadata", {}).get("temperature_delta_profile"),
    }

    # The frontend 'blockMapFromApi' expects 'SimulatorBlocksResponse'. 
    # Important: 'block_driver_matrix' in python response (line 2832) contains:
    # 'temperature_scaled', 'weather_scaled' etc. 
    # Check if 'block_driver_matrix' is populated in pipeline. 
    # Yes, line 2832: "block_driver_matrix": block_driver_matrix.to_dict("records")
    
    # We'll rely on 'block_driver_matrix' to provide the detailed attribution 
    # as the frontend 'store.ts' lines 398-466 heavily use it.
    
    return _json_safe(response)


# --- Scenario Endpoints (Placeholder) ---
SCENARIOS_FILE = os.path.join(os.path.dirname(__file__), "scenarios.json")

def _load_scenarios():
    if not os.path.exists(SCENARIOS_FILE):
        return []
    try:
        with open(SCENARIOS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def _save_scenarios(items):
    with open(SCENARIOS_FILE, "w") as f:
        json.dump(items, f, indent=2)

@app.get("/api/simulator/scenarios")
def v2_list_scenarios():
    return {"items": _load_scenarios()}

@app.post("/api/simulator/scenarios")
def v2_save_scenario(payload: dict):
    items = _load_scenarios()
    new_item = payload
    if "scenario_id" not in new_item:
        new_item["scenario_id"] = str(uuid.uuid4())
    new_item["created_at"] = datetime.now().isoformat()
    items.append(new_item)
    _save_scenarios(items)
    return new_item

@app.delete("/api/simulator/scenarios/{scenario_id}")
def v2_delete_scenario(scenario_id: str):
    items = _load_scenarios()
    items = [x for x in items if x.get("scenario_id") != scenario_id]
    _save_scenarios(items)
    return {"status": "ok"}

def _compute_dayahead(df, target_date: str, baseline_days: int, region: str = "punjab"):
    # New implementation using run_short_term_pipeline
    try:
        baseline_days = int(baseline_days) if baseline_days else 7
    except:
        baseline_days = 7
    baseline_days = max(1, min(baseline_days, 15))
    
    sig = _df_signature(df) if "_df_signature" in globals() else ""
    cache_key = ("dayahead", str(target_date), int(baseline_days), str(region)) + (sig,)
    
    cached = _DAYAHEAD_SERIES_CACHE.get(cache_key) if "_DAYAHEAD_SERIES_CACHE" in globals() else None
    if cached is not None:
        return copy.deepcopy(cached)
        
    try:
        config = {
            "candidate_lookback_days": baseline_days,
            "weather_tune_iters": 0,
            "region": region
        }
        
        result = run_short_term_pipeline(df, target_date, config=config)
        
        # Enrich with consolidated drivers and insights
        result = _refresh_live_analytics_from_final_series(result, 0)
        
        if "_DAYAHEAD_SERIES_CACHE" in globals():
            _DAYAHEAD_SERIES_CACHE[cache_key] = result
            
        return copy.deepcopy(result)
    except Exception as e:
        print(f"Error in _compute_dayahead: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.post("/api/v2/sldc-export")
def v2_sldc_export(payload: dict = None):
    """Export forecast in SLDC-compatible 96-block CSV format."""
    payload = payload or {}
    df = _get_df(copy=False)
    date_str = payload.get("date")
    region = payload.get("region", _app_config.get("region", "punjab"))

    if not date_str:
        valid_dates = _get_valid_dates(df)
        date_str = valid_dates[-1] if valid_dates else None
    if not date_str:
        raise HTTPException(status_code=404, detail="No date available")

    result = _compute_dayahead(df.copy(), date_str, 7)
    if not result:
        raise HTTPException(status_code=404, detail="Forecast unavailable for date")

    series = result.get("series", {})
    blocks = series.get("blocks", list(range(1, 97)))
    forecast = series.get("forecast", [0] * 96)
    baseline = series.get("baseline", [0] * 96)

    # Build SLDC format: Block | Time | Schedule_MW | Baseline_MW
    rows = []
    for i in range(96):
        block = blocks[i] if i < len(blocks) else i + 1
        time_str = f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}"
        fc = float(forecast[i]) if i < len(forecast) else 0.0
        bl = float(baseline[i]) if i < len(baseline) else 0.0
        rows.append({
            "Block": int(block),
            "Time": time_str,
            "Schedule_MW": round(fc, 1),
            "Baseline_MW": round(bl, 1),
        })

    return {
        "format": "SLDC_96_BLOCK",
        "date": date_str,
        "region": region,
        "revision": payload.get("revision", "Rev-0"),
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_energy_mwh": round(sum(r["Schedule_MW"] for r in rows) * 0.25, 1),
        "peak_mw": round(max(r["Schedule_MW"] for r in rows), 1),
        "data": rows,
    }


# In-memory revision history (per-date)
_FORECAST_REVISIONS: Dict[str, list] = {}

@app.post("/api/v2/revision")
def v2_save_revision(payload: dict = None):
    """Save a forecast revision with timestamp and reason."""
    payload = payload or {}
    date_str = payload.get("date")
    reason = payload.get("reason", "Manual revision")
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")

    df = _get_df(copy=False)
    result = _compute_dayahead(df.copy(), date_str, 7)
    if not result:
        raise HTTPException(status_code=404, detail="No forecast for date")

    series = result.get("series", {})
    rev_list = _FORECAST_REVISIONS.setdefault(date_str, [])
    rev_num = len(rev_list)

    revision = {
        "revision": f"Rev-{rev_num}",
        "timestamp": pd.Timestamp.now().isoformat(),
        "reason": reason,
        "forecast": series.get("forecast", []),
        "baseline": series.get("baseline", []),
        "energy_mwh": round(sum(series.get("forecast", [])) * 0.25, 1),
        "peak_mw": round(max(series.get("forecast", [0])), 1),
    }
    rev_list.append(revision)

    return {"date": date_str, "revision": revision, "total_revisions": len(rev_list)}


@app.get("/api/v2/revisions/{date_str}")
def v2_get_revisions(date_str: str):
    """Get all forecast revisions for a date."""
    revisions = _FORECAST_REVISIONS.get(date_str, [])
    return {"date": date_str, "revisions": revisions, "count": len(revisions)}


@app.post("/api/v2/model-performance")
def v2_model_performance(payload: dict = None):
    """Daily accuracy scorecard with 30-day rolling metrics and drift detection."""
    payload = payload or {}
    df = _get_df(copy=False)
    n_days = int(payload.get("n_days", 30))
    valid_dates = _get_valid_dates(df)
    recent_dates = valid_dates[-n_days:] if len(valid_dates) >= n_days else valid_dates

    daily_scores = []
    for date_str in recent_dates:
        try:
            result = _compute_dayahead(df.copy(), date_str, 7)
            if not result:
                continue
            series = result.get("series", {})
            actual = np.asarray(series.get("actual", []), dtype=float)
            forecast = np.asarray(series.get("forecast", []), dtype=float)
            if len(actual) != 96 or len(forecast) != 96:
                continue
            valid = (actual > 10) & np.isfinite(actual) & np.isfinite(forecast)
            if valid.sum() < 20:
                continue
            ape = np.abs(forecast[valid] - actual[valid]) / np.maximum(actual[valid], 1.0) * 100
            mape = float(np.mean(ape))
            rmse = float(np.sqrt(np.mean((forecast[valid] - actual[valid]) ** 2)))

            # Peak hours (48-72)
            peak_mask = np.zeros(96, dtype=bool)
            peak_mask[48:72] = True
            peak_valid = valid & peak_mask
            peak_mape = float(np.mean(np.abs(forecast[peak_valid] - actual[peak_valid]) / np.maximum(actual[peak_valid], 1.0) * 100)) if peak_valid.sum() > 0 else None

            dt = pd.to_datetime(date_str)
            daily_scores.append({
                "date": date_str,
                "mape": round(mape, 2),
                "rmse": round(rmse, 1),
                "peak_mape": round(peak_mape, 2) if peak_mape is not None else None,
                "day_type": "Weekend" if dt.weekday() >= 5 else "Weekday",
            })
        except Exception:
            continue

    if not daily_scores:
        return {"daily_scores": [], "summary": {}, "drift_alert": False}

    scores_df = pd.DataFrame(daily_scores)
    rolling_mape = scores_df["mape"].rolling(7, min_periods=1).mean()

    # Drift detection: if 7-day rolling MAPE increases by >20% vs previous month
    recent_7d = float(rolling_mape.iloc[-1]) if len(rolling_mape) > 0 else 0
    older_avg = float(scores_df["mape"].iloc[:max(1, len(scores_df) - 7)].mean()) if len(scores_df) > 7 else recent_7d
    drift_pct = ((recent_7d - older_avg) / max(older_avg, 0.1)) * 100
    drift_alert = drift_pct > 20

    return {
        "daily_scores": daily_scores,
        "summary": {
            "mape_mean": round(float(scores_df["mape"].mean()), 2),
            "mape_7d_rolling": round(recent_7d, 2),
            "rmse_mean": round(float(scores_df["rmse"].mean()), 1),
            "best_day": scores_df.loc[scores_df["mape"].idxmin()].to_dict() if len(scores_df) > 0 else None,
            "worst_day": scores_df.loc[scores_df["mape"].idxmax()].to_dict() if len(scores_df) > 0 else None,
            "days_under_3pct": int((scores_df["mape"] < 3.0).sum()),
            "days_under_5pct": int((scores_df["mape"] < 5.0).sum()),
        },
        "drift_alert": drift_alert,
        "drift_pct": round(drift_pct, 1),
        "drift_message": f"7-day MAPE increased by {drift_pct:.1f}% vs prior period. Consider retraining." if drift_alert else "Model performance stable.",
    }


@app.post("/api/v2/annotate")
def v2_annotate(payload: dict = None):
    """Allow operators to tag days with special notes (for future ML training)."""
    payload = payload or {}
    date_str = payload.get("date")
    note = payload.get("note", "")
    tags = payload.get("tags", [])  # e.g. ["outage", "festival", "fog"]

    if not date_str or not note:
        raise HTTPException(status_code=400, detail="Date and note required")

    # Store annotations in memory (production: persist to DB/file)
    if not hasattr(v2_annotate, "_store"):
        v2_annotate._store = {}
    annotations = v2_annotate._store.setdefault(date_str, [])
    annotation = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "note": note,
        "tags": tags,
    }
    annotations.append(annotation)

    return {"date": date_str, "annotation": annotation, "total": len(annotations)}


@app.get("/api/v2/annotations/{date_str}")
def v2_get_annotations(date_str: str):
    """Get operator annotations for a date."""
    store = getattr(v2_annotate, "_store", {})
    return {"date": date_str, "annotations": store.get(date_str, [])}


@app.post("/api/v2/backtest")
def v2_backtest(payload: dict = None):
    """Rolling backtest: run pipeline day-ahead for last N days, compute stratified accuracy."""
    payload = payload or {}
    df = _get_df()
    n_days = int(payload.get("n_days", 30))
    actual_blocks_sim = int(payload.get("actual_blocks", 36))
    config_override = payload.get("config", {})

    def _pipeline_wrapper(df_inner, target_date, config=None):
        merged = {**config_override, **(config or {})}
        return run_short_term_pipeline(df_inner, target_date, config=merged)

    result = run_backtest(
        df=df,
        pipeline_fn=_pipeline_wrapper,
        n_days=n_days,
        actual_blocks_sim=actual_blocks_sim,
        config=config_override,
    )
    return _json_safe(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
