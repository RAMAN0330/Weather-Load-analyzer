"""Data-driven per-location weights for non-Haryana states.

Idea: for each weather location L in a state, fit a small ridge model
    state_load ~ L's [temperature, humidity, precipitation, cloud_cover, ...]
on historical data, score it on a held-out tail, and use the held-out R²
(clipped at 0) as L's raw weight.  The state-level weighted weather is then
    weighted_W[col] = sum_L (w_L * W_L[col])  /  sum_L w_L
i.e. locations whose weather better explains state load get more say.

This is a one-shot offline computation per state — results are cached to
`exports/location_weights_<state>.json`.  Forecast-time we just look the
weights up; no model fitting in the request path.

Cache schema:
    {
      "state": "CHHATTISGARH",
      "computed_at": "2026-04-25T12:00:00",
      "lookback_days": 120,
      "weights": {
        "RAIPUR": 0.42,
        "BILASPUR": 0.21,
        ...
      },
      "method": "ridge_holdout_r2_v1"
    }

`weights` always sum to 1.0 over the keys present.  Locations not in the
dict get weight 0 at fetch time.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _REPO_ROOT / "exports"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_FEATURES = (
    "temperature",
    "humidity",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "direct_radiation",
    "apparent_temperature",
)


def _cache_path(state: str) -> Path:
    return _CACHE_DIR / f"location_weights_{state.lower()}.json"


def load_cached_weights(state: str, max_age_days: float = 14.0) -> Optional[Dict[str, float]]:
    """Return cached {location: weight} for `state`, or None if missing/stale."""
    p = _cache_path(state)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[learned_weights] cache unreadable for %s: %s", state, exc)
        return None
    try:
        ts = datetime.fromisoformat(payload.get("computed_at", ""))
        age_days = (datetime.utcnow() - ts).total_seconds() / 86400.0
        if age_days > max_age_days:
            logger.info("[learned_weights] cache for %s is %.1fd old (>%.1fd) — ignoring",
                        state, age_days, max_age_days)
            return None
    except Exception:
        pass
    weights = payload.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return None
    return {str(k).upper(): float(v) for k, v in weights.items()}


def save_weights(state: str, weights: Dict[str, float], lookback_days: int) -> Path:
    p = _cache_path(state)
    payload = {
        "state": state.upper(),
        "computed_at": datetime.utcnow().isoformat(timespec="seconds"),
        "lookback_days": int(lookback_days),
        "weights": {str(k).upper(): float(v) for k, v in weights.items()},
        "method": "ridge_holdout_r2_v1",
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _fit_holdout_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, holdout_frac: float = 0.2) -> float:
    """Closed-form ridge on the train tail, score on the held-out tail.

    Returns R² clipped at 0.  No sklearn dependency.
    """
    if X.size == 0 or y.size == 0:
        return 0.0
    n = X.shape[0]
    if n < 30:
        return 0.0
    h = max(int(round(n * holdout_frac)), 10)
    h = min(h, n - 20)
    if h <= 0:
        return 0.0
    X_tr, X_te = X[:-h], X[-h:]
    y_tr, y_te = y[:-h], y[-h:]

    # Standardise X on the train portion only (no leakage).
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    # Add intercept column.
    X_tr_b = np.column_stack([np.ones(len(X_tr)), X_tr])
    X_te_b = np.column_stack([np.ones(len(X_te)), X_te])
    p = X_tr_b.shape[1]

    # Closed-form ridge: β = (X'X + αI)^-1 X'y, with intercept un-penalised.
    A = X_tr_b.T @ X_tr_b
    reg = alpha * np.eye(p)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(A + reg, X_tr_b.T @ y_tr)
    except np.linalg.LinAlgError:
        return 0.0

    y_hat = X_te_b @ beta
    ss_res = float(np.sum((y_te - y_hat) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    if ss_tot < 1e-9:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(0.0, r2))


def compute_location_weights(
    state: str,
    df_load: pd.DataFrame,
    df_weather_loc: pd.DataFrame,
    *,
    features: Iterable[str] = WEATHER_FEATURES,
    lookback_days: int = 120,
    persist: bool = True,
) -> Dict[str, float]:
    """Fit per-location ridge models against state load and return normalised weights.

    Inputs:
      df_load: [date, time_block, total_drawal]  (state-level)
      df_weather_loc: [date, time_block, location, <weather features>]
      lookback_days: cap on df rows used (latest N days).

    Output:
      {location_upper: weight in [0, 1]} summing to 1.0 over locations that
      produced any signal (R² > 0).  Locations with R² == 0 are dropped.
      If no location produces signal, returns equal weights so the caller
      degrades to a plain mean.
    """
    if df_load is None or df_load.empty or df_weather_loc is None or df_weather_loc.empty:
        return {}

    load = df_load[["date", "time_block", "total_drawal"]].copy()
    load["date"] = pd.to_datetime(load["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    load["time_block"] = pd.to_numeric(load["time_block"], errors="coerce").astype("Int64")
    load = load.dropna()

    wdf = df_weather_loc.copy()
    wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wdf["time_block"] = pd.to_numeric(wdf["time_block"], errors="coerce").astype("Int64")
    wdf["location"] = wdf["location"].astype(str).str.strip().str.upper()
    wdf = wdf.dropna(subset=["date", "time_block", "location"])

    # Trim to recent window for both frames.
    if lookback_days and lookback_days > 0:
        all_dates = sorted(set(load["date"].unique()) & set(wdf["date"].unique()))
        if all_dates:
            cutoff = all_dates[-lookback_days:]
            load = load[load["date"].isin(cutoff)]
            wdf = wdf[wdf["date"].isin(cutoff)]

    feat_cols = [c for c in features if c in wdf.columns]
    if not feat_cols:
        logger.warning("[learned_weights] no usable feature columns in weather frame")
        return {}

    # State-level baseline: per-block mean over the same date range, so the
    # residual we regress against represents "weather-driven deviations".
    block_means = load.groupby("time_block")["total_drawal"].mean()
    load["resid"] = load["total_drawal"] - load["time_block"].map(block_means)

    raw_weights: Dict[str, float] = {}
    for loc, loc_df in wdf.groupby("location"):
        merged = loc_df.merge(load[["date", "time_block", "resid"]], on=["date", "time_block"], how="inner")
        if len(merged) < 200:
            continue
        merged = merged.sort_values(["date", "time_block"])
        X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        y = merged["resid"].to_numpy(dtype=float)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if mask.sum() < 200:
            continue
        r2 = _fit_holdout_r2(X[mask], y[mask])
        if r2 > 0.0:
            raw_weights[loc] = r2

    if not raw_weights:
        # Fall back to equal weights across whatever locations we saw.
        seen = sorted(wdf["location"].unique())
        if not seen:
            return {}
        equal = 1.0 / len(seen)
        weights = {loc: equal for loc in seen}
    else:
        total = float(sum(raw_weights.values()))
        weights = {loc: float(w / total) for loc, w in raw_weights.items()}

    if persist:
        try:
            save_weights(state, weights, lookback_days)
        except Exception as exc:
            logger.warning("[learned_weights] could not persist cache for %s: %s", state, exc)

    return weights


def aggregate_with_learned_weights(
    per_location_df: pd.DataFrame,
    weights: Dict[str, float],
    *,
    location_col: str = "location",
    block_col: str = "time_block",
    weather_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Same shape as haryana_circle_weights.aggregate_weather_by_circle, but
    using a learned-weights dict instead of a hand-curated CIRCLE_CONTRIBUTION_PCT.
    Locations not in `weights` are dropped (equivalent to weight 0)."""
    if per_location_df is None or per_location_df.empty or not weights:
        return pd.DataFrame()

    cols = [c for c in (weather_cols or WEATHER_FEATURES) if c in per_location_df.columns]
    if not cols:
        return pd.DataFrame()

    df = per_location_df.copy()
    df[location_col] = df[location_col].astype(str).str.strip().str.upper()
    df["__weight"] = df[location_col].map(weights).astype(float)
    df = df[df["__weight"].fillna(0.0) > 0.0].copy()
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({block_col: sorted(df[block_col].dropna().unique())}).set_index(block_col)
    weight_sum = df.groupby(block_col)["__weight"].sum()
    for c in cols:
        weighted_vals = df["__weight"] * pd.to_numeric(df[c], errors="coerce")
        weighted_sum = weighted_vals.groupby(df[block_col]).sum()
        out[c] = (weighted_sum / weight_sum.replace(0, np.nan)).reindex(out.index)
    return out.reset_index()


__all__ = [
    "WEATHER_FEATURES",
    "load_cached_weights",
    "save_weights",
    "compute_location_weights",
    "aggregate_with_learned_weights",
]
