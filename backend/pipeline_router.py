"""Pipeline data endpoints (``/api/pipeline/*``) served from PostgreSQL.

Used by the analytics dashboards, Similar Days and Weather Locations pages.
Response shapes match what the frontend consumes (formerly served by Django).
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

try:
    from . import db
except ImportError:  # running as a script without package context
    import db  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

SIMILARITY_VARS = ("temperature", "humidity", "precipitation")
WEIGHTED_SIMILARITY = {"temperature": 1.0, "humidity": 0.5, "precipitation": 0.3}


def _clean(rows: list[dict]) -> list[dict]:
    """NaN/inf -> None so the JSON is valid."""
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
    return rows


def _rows(table: str, state: str, days: int, from_date: Optional[str], to_date: Optional[str]) -> list[dict]:
    """Rows for the last ``days`` days of ``table`` (or an explicit date window)."""
    try:
        if not from_date:
            _, latest = db.date_range(state, table)
            if not latest:
                return []
            from_date = (date.fromisoformat(latest) - timedelta(days=max(int(days or 60), 1) - 1)).isoformat()
        return _clean(db.table_rows(table, state, from_date=from_date, to_date=to_date, limit=None))
    except db.DataSourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


@router.get("/states")
def pipeline_states():
    try:
        return db.states()
    except db.DataSourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


@router.get("/weather/{state}")
def pipeline_weather(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _rows("weather_mean", state, days, from_date, to_date)


@router.get("/weather-loc/{state}")
def pipeline_weather_loc(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _rows("weather_loc", state, days, from_date, to_date)


@router.get("/circle-impact/{state}")
def pipeline_circle_impact(state: str, days: int = Query(365)):
    # Per-circle load sensitivity needs circle-level feeder load, which the
    # database does not hold; return the empty-but-valid shape the UI expects.
    return {
        "state": state.upper(),
        "days": days,
        "circles": [],
        "impact": [],
        "available": False,
        "message": "Circle impact analytics need circle-level load data, which is not loaded.",
    }


def _reshape(rows: list[dict], value_col: str, out_col: str, block_col: str = "time_block") -> list[dict]:
    return [{"date": r["date"], "time_block": r[block_col], out_col: r.get(value_col)} for r in rows]


@router.get("/load/{state}")
def pipeline_load(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _reshape(_rows("load", state, days, from_date, to_date), "total_drawal", "load")


@router.get("/sldc/{state}")
def pipeline_sldc(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _reshape(_rows("sldc", state, days, from_date, to_date), "total_drawal", "actual")


@router.get("/forecast/{state}")
def pipeline_forecast(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _reshape(_rows("forecast", state, days, from_date, to_date), "forecast_mw", "forecasted", block_col="block")


@router.get("/sldc-forecast/{state}")
def pipeline_sldc_forecast(state: str, days: int = Query(60), from_date: Optional[str] = None, to_date: Optional[str] = None):
    return _reshape(_rows("sldc_forecast", state, days, from_date, to_date), "forecast_mw", "forecast")


@router.get("/similarity/{state}")
def pipeline_similarity(
    state: str,
    target_date: str = Query(...),
    method: str = Query("euclidean", pattern="^(euclidean|weighted|cosine)$"),
    top_k: int = Query(5, ge=1, le=50),
):
    """Days whose weather profile is closest to ``target_date``.

    Only days *before* the target are candidates (no future analogues). Each
    variable is standardised (z-score over the candidate pool) so temperature,
    humidity and rain contribute on comparable scales; missing blocks are
    skipped rather than treated as 0. ``method``: ``euclidean`` (RMS distance
    of z-scores), ``weighted`` (temperature 1.0, humidity 0.5, rain 0.3) or
    ``cosine`` (1 - cosine similarity of the z-scored profiles).
    """
    try:
        df = db.weather_mean_frame(state, to_date=target_date)
    except db.DataSourceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    if df.empty or target_date not in set(df["date"]):
        return {"target_date": target_date, "similar": [], "profiles": {}, "method": method}

    cube = {}
    for v in SIMILARITY_VARS:
        piv = df.pivot_table(index="date", columns="time_block", values=v, aggfunc="mean")
        cube[v] = piv.reindex(columns=range(1, 97))
    dates = cube["temperature"].index
    candidates = [d for d in dates if d < target_date]
    if not candidates:
        return {"target_date": target_date, "similar": [], "profiles": {}, "method": method}

    weights = WEIGHTED_SIMILARITY if method == "weighted" else {v: 1.0 for v in SIMILARITY_VARS}
    z_target, z_cand, w_vec = [], [], []
    for v in SIMILARITY_VARS:
        vals = cube[v].to_numpy(dtype=float)
        pool = cube[v].loc[candidates].to_numpy(dtype=float)
        mu, sd = np.nanmean(pool), np.nanstd(pool)
        sd = sd if sd > 1e-9 else 1.0
        z = (vals - mu) / sd
        z_target.append(z[dates.get_loc(target_date)])
        z_cand.append(z[[dates.get_loc(d) for d in candidates]])
        w_vec.append(np.full(96, weights[v]))
    t = np.concatenate(z_target)                  # (3*96,)
    c = np.concatenate(z_cand, axis=1)            # (n, 3*96)
    w = np.concatenate(w_vec)
    valid = np.isfinite(c) & np.isfinite(t)[None, :]

    if method == "cosine":
        tc, cc = np.where(valid, t[None, :], 0.0), np.where(valid, c, 0.0)
        num = (tc * cc).sum(axis=1)
        den = np.sqrt((tc ** 2).sum(axis=1) * (cc ** 2).sum(axis=1))
        dist = 1.0 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    else:
        sq = np.where(valid, w[None, :] * (c - t[None, :]) ** 2, 0.0)
        wsum = np.where(valid, w[None, :], 0.0).sum(axis=1)
        dist = np.sqrt(np.divide(sq.sum(axis=1), wsum, out=np.full(len(c), np.inf), where=wsum > 0))
    coverage = valid.mean(axis=1)
    dist = np.where(coverage >= 0.5, dist, np.inf)  # need at least half the blocks to compare

    order = np.argsort(dist)[:top_k]
    similar = [
        {"date": candidates[i], "distance": round(float(dist[i]), 4),
         "day_of_week": pd.Timestamp(candidates[i]).day_name()}
        for i in order if np.isfinite(dist[i])
    ]
    profiles = {}
    for d in [target_date] + [s["date"] for s in similar]:
        profiles[d] = {
            "temp": [None if pd.isna(x) else float(x) for x in cube["temperature"].loc[d]],
            "humidity": [None if pd.isna(x) else float(x) for x in cube["humidity"].loc[d]],
            "precip": [None if pd.isna(x) else float(x) for x in cube["precipitation"].loc[d]],
        }
    return {"target_date": target_date, "similar": similar, "profiles": profiles, "method": method}
