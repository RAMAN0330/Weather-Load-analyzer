"""PostgreSQL data access for Forecast Studio.

All pipeline data (actual load, SLDC load, state-mean and district weather,
external forecasts) lives in PostgreSQL. The schema is created by
``db/init/01_schema.sql`` and filled with dummy data by ``db/seed.py``.

The connection string comes from ``DATABASE_URL``, e.g.
``postgresql+psycopg2://forecast:forecast@postgres:5432/forecast_studio``.

The public helpers mirror the shapes the app used before (row dicts ordered by
date then block, with ``date`` as ``YYYY-MM-DD`` strings), so callers only
change their import.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_URL = "postgresql+psycopg2://forecast:forecast@localhost:5432/forecast_studio"

# table -> (block column, value columns)
TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "load": ("time_block", ("total_drawal",)),
    "sldc": ("time_block", ("total_drawal",)),
    "weather_mean": (
        "time_block",
        ("temperature", "humidity", "precipitation", "cloud_cover", "cloud_cover_low",
         "sunshine_duration", "direct_radiation", "wind_speed_10m"),
    ),
    "weather_loc": (
        "time_block",
        ("location", "temperature", "humidity", "precipitation", "cloud_cover", "cloud_cover_low",
         "sunshine_duration", "direct_radiation", "wind_speed_10m"),
    ),
    "forecast": ("block", ("forecast_mw",)),
    "sldc_forecast": ("time_block", ("forecast_mw",)),
}


class DataSourceUnavailable(RuntimeError):
    """The database could not be reached or queried."""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


@lru_cache(maxsize=1)
def engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)


def _norm_state(state: object) -> str:
    return str(state or "HARYANA").strip().upper().replace(" ", "_").replace("-", "_")


def _iso(d: object) -> Optional[str]:
    if d is None or d == "":
        return None
    if isinstance(d, (date, datetime)):
        return d.isoformat()[:10]
    return str(d)[:10]


def _query(sql: str, params: dict) -> pd.DataFrame:
    try:
        with engine().connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)
    except Exception as exc:  # connection refused, bad credentials, missing table …
        raise DataSourceUnavailable(f"{type(exc).__name__}: {exc}") from exc


def read_frame(
    table: str,
    state: object,
    date: object = None,
    from_date: object = None,
    to_date: object = None,
    location: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Rows of ``table`` for ``state`` filtered by date, ordered by date then block."""
    if table not in TABLES:
        raise KeyError(f"unknown table {table!r}")
    block_col, value_cols = TABLES[table]
    where = ["state = :state"]
    params: dict = {"state": _norm_state(state)}
    if date is not None:
        where.append("date = :d")
        params["d"] = _iso(date)
    else:
        if from_date:
            where.append("date >= :f")
            params["f"] = _iso(from_date)
        if to_date:
            where.append("date <= :t")
            params["t"] = _iso(to_date)
    if location and table == "weather_loc":
        where.append("location = :loc")
        params["loc"] = str(location).strip().upper()
    order = f"date, {block_col}" + (", location" if table == "weather_loc" else "")
    sql = (
        f"SELECT state, to_char(date, 'YYYY-MM-DD') AS date, {block_col}, {', '.join(value_cols)} "
        f"FROM {table} WHERE {' AND '.join(where)} ORDER BY {order}"
    )
    if limit is not None:
        sql += " LIMIT :lim"
        params["lim"] = int(limit)
    return _query(sql, params)


def table_rows(table: str, state: object, date=None, from_date=None, to_date=None,
               limit: Optional[int] = 200, location: Optional[str] = None) -> list[dict]:
    df = read_frame(table, state, date=date, from_date=from_date, to_date=to_date, location=location, limit=limit)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load_frame(state: object, from_date=None, to_date=None, table: str = "load") -> pd.DataFrame:
    return read_frame(table, state, from_date=from_date, to_date=to_date)


def weather_loc_frame(state: object, from_date=None, to_date=None) -> pd.DataFrame:
    return read_frame("weather_loc", state, from_date=from_date, to_date=to_date)


def weather_mean_frame(state: object, from_date=None, to_date=None) -> pd.DataFrame:
    return read_frame("weather_mean", state, from_date=from_date, to_date=to_date)


def forecast_frame(state: object, from_date=None, to_date=None, table: str = "forecast") -> pd.DataFrame:
    return read_frame(table, state, from_date=from_date, to_date=to_date)


def states() -> list[str]:
    df = _query("SELECT DISTINCT state FROM load ORDER BY state", {})
    return df["state"].tolist()


def locations(state: object) -> list[str]:
    df = _query("SELECT DISTINCT location FROM weather_loc WHERE state = :s ORDER BY location", {"s": _norm_state(state)})
    return df["location"].tolist()


def date_range(state: object, table: str = "load") -> tuple[Optional[str], Optional[str]]:
    if table not in TABLES:
        raise KeyError(table)
    df = _query(
        f"SELECT to_char(MIN(date), 'YYYY-MM-DD') AS lo, to_char(MAX(date), 'YYYY-MM-DD') AS hi FROM {table} WHERE state = :s",
        {"s": _norm_state(state)},
    )
    return (df.at[0, "lo"], df.at[0, "hi"]) if not df.empty else (None, None)


def load_dates(state: object) -> list[str]:
    df = _query("SELECT DISTINCT to_char(date, 'YYYY-MM-DD') AS d FROM load WHERE state = :s ORDER BY d", {"s": _norm_state(state)})
    return df["d"].tolist()


def table_count(table: str, state: object) -> int:
    if table not in TABLES:
        raise KeyError(table)
    df = _query(f"SELECT COUNT(*) AS c FROM {table} WHERE state = :s", {"s": _norm_state(state)})
    return int(df.at[0, "c"])


__all__: Iterable[str] = [
    "DataSourceUnavailable", "TABLES", "database_url", "engine", "read_frame", "table_rows",
    "load_frame", "weather_loc_frame", "weather_mean_frame", "forecast_frame",
    "states", "locations", "date_range", "load_dates", "table_count",
]
