"""Data access for Forecast Engine v3.

``load_region_frames`` pulls SLDC + load + per-district weather from the
PostgreSQL database (``backend/db.py``) and returns two
tidy frames:

* ``load_df``       — date, block, load  (SLDC preferred, gaps filled from /load)
* ``weather_long``  — date, block, district, temperature, humidity,
                      precipitation, cloud_cover, wind_speed_10m, direct_radiation

Missing values stay NaN (no zero-filling). Results are cached on disk under
``backend/.cache/forecasting/`` for ``FORECAST_CACHE_TTL_S`` seconds (default
900). The service accepts any callable with the same signature, so tests can
inject synthetic data.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import FEATURE_VERSION
from .spatial import WEATHER_VARS, canonical_district

logger = logging.getLogger(__name__)

CACHE_DIR: Path = Path(__file__).resolve().parent.parent / ".cache" / "forecasting"
LoaderFn = Callable[[str, str, str], tuple[pd.DataFrame, pd.DataFrame]]

# API / Open-Meteo column names → canonical names.
_WEATHER_RENAME: dict[str, str] = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "rain": "precipitation",
    "cloudcover": "cloud_cover",
    "windspeed_10m": "wind_speed_10m",
    "direct_radiation_instant": "direct_radiation",
}
_LOAD_VALUE_COLS: tuple[str, ...] = ("total_drawal", "load_mw", "drawal_mw", "demand_mw", "mw", "load")


class DataSourceError(RuntimeError):
    """The upstream data source failed or returned unusable data."""


def region_to_state(region: str) -> str:
    return str(region).strip().upper().replace(" ", "_").replace("-", "_")


def _block_column(df: pd.DataFrame) -> pd.Series:
    """Extract block 1..96 from ``time_block``/``block`` or a timestamp column."""
    for c in ("time_block", "block"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    for c in ("datetime", "timestamp", "time"):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce")
            return (ts.dt.hour * 4 + ts.dt.minute // 15 + 1).astype("Float64")
    return pd.Series(np.nan, index=df.index)


def normalise_load_rows(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """API rows → (date, block, load), duplicates averaged (count kept in attrs)."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty or "date" not in df.columns:
        out = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "block": pd.Series(dtype=np.int64), "load": pd.Series(dtype=np.float64)})
        out.attrs["source_duplicate_blocks"] = 0
        return out
    val_col = next((c for c in _LOAD_VALUE_COLS if c in df.columns), None)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "block": _block_column(df),
            "load": pd.to_numeric(df[val_col], errors="coerce") if val_col else np.nan,
        }
    ).dropna(subset=["date", "block"])
    out["block"] = out["block"].astype(np.int64)
    out = out[out["block"].between(1, 96)]
    dupes = int(out.duplicated(subset=["date", "block"]).sum())
    out = out.groupby(["date", "block"], as_index=False)["load"].mean()
    out.attrs["source_duplicate_blocks"] = dupes
    return out


def merge_sldc_load(sldc: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    """Vectorised SLDC-priority merge (same rule as ``build_final_data.merge_load_sldc``).

    Per date, blocks up to the last valid SLDC block use SLDC (falling back to
    /load where SLDC is NaN); later blocks use /load (falling back to SLDC).
    """
    m = pd.merge(
        sldc.rename(columns={"load": "sldc"}),
        load.rename(columns={"load": "aux"}),
        on=["date", "block"],
        how="outer",
    )
    if m.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "block": pd.Series(dtype=np.int64), "load": pd.Series(dtype=np.float64)})
    last_sldc = m["block"].where(m["sldc"].notna()).groupby(m["date"]).transform("max").fillna(0)
    in_zone = m["block"] <= last_sldc
    value = np.where(in_zone, m["sldc"].fillna(m["aux"]), m["aux"].fillna(m["sldc"]))
    out = pd.DataFrame({"date": m["date"], "block": m["block"].astype(np.int64), "load": value.astype(np.float64)})
    out = out.sort_values(["date", "block"]).reset_index(drop=True)
    out.attrs["source_duplicate_blocks"] = int(sldc.attrs.get("source_duplicate_blocks", 0)) + int(
        load.attrs.get("source_duplicate_blocks", 0)
    )
    return out


def normalise_weather_rows(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """API rows → long weather frame with canonical columns (NaN kept)."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    cols = ["date", "block", "district", *WEATHER_VARS]
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={k: v for k, v in _WEATHER_RENAME.items() if k in df.columns and v not in df.columns})
    if "temperature" not in df.columns and "apparent_temperature" in df.columns:
        df = df.rename(columns={"apparent_temperature": "temperature"})
    dist_col = next((c for c in ("district", "location", "location_name", "city") if c in df.columns), None)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "block": _block_column(df),
            "district": df[dist_col].map(canonical_district) if dist_col else "STATE",
        }
    )
    for v in WEATHER_VARS:
        out[v] = pd.to_numeric(df[v], errors="coerce") if v in df.columns else np.nan
    out = out.dropna(subset=["date", "block"])
    out["block"] = out["block"].astype(np.int64)
    return out[cols].reset_index(drop=True)


def _import_db():
    """PostgreSQL data access (``backend/db.py``)."""
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import db  # noqa: WPS433

    return db


def _cache_key(region: str, from_date: str, to_date: str) -> str:
    raw = f"{region.lower()}|{from_date}|{to_date}|{FEATURE_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _cache_ttl() -> float:
    try:
        return float(os.getenv("FORECAST_CACHE_TTL_S", "900"))
    except ValueError:
        return 900.0


def _write_frame(df: pd.DataFrame, path_stem: Path) -> None:
    """Parquet when an engine is installed, else pickle (attrs preserved)."""
    try:
        df.to_parquet(path_stem.with_suffix(".parquet"), index=False)
    except (ImportError, ValueError):
        df.to_pickle(path_stem.with_suffix(".pkl"))


def _read_frame(path_stem: Path, ttl: float) -> pd.DataFrame | None:
    for suffix, reader in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        p = path_stem.with_suffix(suffix)
        if p.exists() and (time.time() - p.stat().st_mtime) <= ttl:
            try:
                return reader(p)
            except Exception:  # corrupt cache → refetch
                return None
    return None


def load_region_frames(region: str, from_date: str, to_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (load_df, weather_long) for ``region`` over [from_date, to_date].

    Raises :class:`DataSourceError` when the upstream API fails.
    """
    ttl = _cache_ttl()
    stem = CACHE_DIR / _cache_key(region, from_date, to_date)
    if ttl > 0:
        cl, cw = _read_frame(stem.with_name(stem.name + "_load"), ttl), _read_frame(stem.with_name(stem.name + "_wx"), ttl)
        if cl is not None and cw is not None:
            return cl, cw

    state = region_to_state(region)
    try:
        source = _import_db()
        sldc = normalise_load_rows(source.load_frame(state, from_date, to_date, table="sldc"))
        aux = normalise_load_rows(source.load_frame(state, from_date, to_date, table="load"))
        wx = normalise_weather_rows(source.weather_loc_frame(state, from_date, to_date))
    except Exception as exc:  # generator / parse failures
        raise DataSourceError(f"Data source unavailable for {state}: {type(exc).__name__}: {exc}") from exc
    load_df = merge_sldc_load(sldc, aux)
    if load_df.empty:
        raise DataSourceError(f"No load data returned for {state} between {from_date} and {to_date}.")

    if ttl > 0:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _write_frame(load_df, stem.with_name(stem.name + "_load"))
            _write_frame(wx, stem.with_name(stem.name + "_wx"))
        except OSError as exc:  # cache is best-effort
            logger.warning("forecast cache write failed: %s", exc)
    return load_df, wx


__all__ = [
    "CACHE_DIR",
    "DataSourceError",
    "LoaderFn",
    "load_region_frames",
    "merge_sldc_load",
    "normalise_load_rows",
    "normalise_weather_rows",
    "region_to_state",
]
