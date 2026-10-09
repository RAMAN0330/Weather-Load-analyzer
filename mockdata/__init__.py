"""Deterministic mock data source for Forecast Studio (replaces the MySQL database).

Every table the app used to read (``load``, ``sldc``, ``weather_mean``,
``weather_loc``, ``forecast``, ``sldc_forecast``) is generated on demand from
seeded, physically plausible models of Indian state demand and district
weather. The same (state, date) always yields the same numbers, so pages,
backtests and screenshots are reproducible.

Public API:
  - :func:`table_rows`   row dicts shaped like the old MySQL rows (Django views)
  - :func:`load_frame`, :func:`weather_loc_frame`, :func:`weather_mean_frame`,
    :func:`forecast_frame` vectorised DataFrames (FastAPI / forecasting engine)
  - :func:`states`, :func:`locations`, :func:`load_dates`, :func:`date_range`
"""
from .generator import (  # noqa: F401
    DATA_START,
    TABLES,
    date_range,
    forecast_frame,
    load_dates,
    load_frame,
    locations,
    states,
    table_count,
    table_rows,
    weather_loc_frame,
    weather_mean_frame,
)

SOURCE_NAME = "mock"
