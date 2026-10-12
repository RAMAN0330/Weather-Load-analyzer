"""Daily forecasting cycle and dummy-data feed.

Every ``DAILY_JOB_INTERVAL_S`` seconds (default 900) the background loop:

1. **Tops up the dummy data feed** (when ``DUMMY_DATA_FEED`` is not "false"):
   appends the newly settled load blocks for today and weather / external
   forecasts up to today + 2 from the ``mockdata`` generator, so the seeded
   database keeps moving forward like a live feed. Any insert clears the
   query cache and triggers ``on_new_data`` (reloads the legacy engine).
2. **Issues today's forecasts**: T+1 (tomorrow) and T+2 (day after) for the
   app region with the default model, once per operating day, cached in memory
   and written as immutable forecast records by the service.

Forecasting is only for those two targets; past dates are evaluated with the
backtest endpoints instead.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

try:
    from . import db
except ImportError:  # script mode
    import db  # type: ignore

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
APP_REGION = os.environ.get("APP_REGION", "haryana").lower()
DEFAULT_MODEL = "lgbm_residual"
HORIZONS = (1, 2)


def ops_date() -> date:
    """Operating day (IST)."""
    return datetime.now(IST).date()


def target_for(horizon: int, today: Optional[date] = None) -> date:
    return (today or ops_date()) + timedelta(days=int(horizon))


def allowed_targets(today: Optional[date] = None) -> dict[int, str]:
    return {h: target_for(h, today).isoformat() for h in HORIZONS}


# ── Daily forecast cache ─────────────────────────────────────────────────────
class DailyForecasts:
    """One forecast per (operating day, horizon, model), computed on demand and cached."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[tuple, dict] = {}
        self._inflight: dict[tuple, threading.Event] = {}

    def get(self, service: Any, horizon: int, model: str = DEFAULT_MODEL, refresh: bool = False) -> dict:
        today = ops_date()
        key = (today.isoformat(), int(horizon), model)
        while True:
            with self._lock:
                if not refresh and key in self._results:
                    return self._results[key]
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    break
            event.wait(timeout=600)  # another thread is computing the same forecast
            refresh = False
        try:
            result = service.forecast(APP_REGION, target_for(horizon, today).isoformat(), int(horizon), model)
            result = {**result, "issued_for": today.isoformat(), "issued_at": datetime.now(IST).isoformat(timespec="seconds")}
            with self._lock:
                # keep only today's entries
                self._results = {k: v for k, v in self._results.items() if k[0] == key[0]}
                self._results[key] = result
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    def invalidate(self) -> None:
        with self._lock:
            self._results.clear()

    def status(self) -> dict:
        with self._lock:
            return {"cached": sorted(f"{k[0]} h{k[1]} {k[2]}" for k in self._results)}


daily_forecasts = DailyForecasts()


# ── Dummy data feed ──────────────────────────────────────────────────────────
def _import_mockdata():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import mockdata  # noqa: WPS433

    return mockdata


_FEED_COLUMNS = {
    "load": ["state", "date", "time_block", "total_drawal"],
    "sldc": ["state", "date", "time_block", "total_drawal"],
    "weather_mean": ["state", "date", "time_block", "temperature", "humidity", "precipitation", "cloud_cover",
                     "cloud_cover_low", "sunshine_duration", "direct_radiation", "wind_speed_10m"],
    "weather_loc": ["state", "date", "time_block", "location", "temperature", "humidity", "precipitation", "cloud_cover",
                    "cloud_cover_low", "sunshine_duration", "direct_radiation", "wind_speed_10m"],
    "forecast": ["state", "date", "block", "forecast_mw"],
    "sldc_forecast": ["state", "date", "time_block", "forecast_mw"],
}


def feed_enabled() -> bool:
    return os.environ.get("DUMMY_DATA_FEED", "true").strip().lower() not in ("0", "false", "no")


def topup_dummy_data() -> int:
    """Insert generated rows newer than what the database holds. Returns rows inserted."""
    if not feed_enabled():
        return 0
    from psycopg2.extras import execute_values  # noqa: WPS433

    mock = _import_mockdata()
    frames = {
        "load": lambda s, f, t: mock.load_frame(s, f, t, table="load"),
        "sldc": lambda s, f, t: mock.load_frame(s, f, t, table="sldc"),
        "weather_mean": mock.weather_mean_frame,
        "weather_loc": mock.weather_loc_frame,
        "forecast": lambda s, f, t: mock.forecast_frame(s, f, t, table="forecast"),
        "sldc_forecast": lambda s, f, t: mock.forecast_frame(s, f, t, table="sldc_forecast"),
    }
    inserted = 0
    raw = db.engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("SELECT DISTINCT state FROM load")
            states = [r[0] for r in cur.fetchall()]
        for state in states:
            for table, cols in _FEED_COLUMNS.items():
                with raw.cursor() as cur:
                    cur.execute(f"SELECT MAX(date) FROM {table} WHERE state = %s", (state,))
                    last = cur.fetchone()[0]
                if last is None:
                    continue
                _, end = mock.date_range(state, table)
                if end < last:
                    continue
                df = frames[table](state, last.isoformat(), end.isoformat())
                if df.empty:
                    continue
                rows = list(zip(*(df[c].tolist() for c in cols)))  # python scalars for psycopg2
                count_sql = f"SELECT COUNT(*) FROM {table} WHERE state = %s AND date >= %s"
                with raw.cursor() as cur:
                    cur.execute(count_sql, (state, last))
                    before = cur.fetchone()[0]
                    execute_values(
                        cur,
                        f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s ON CONFLICT DO NOTHING",
                        rows, page_size=5000,
                    )
                    cur.execute(count_sql, (state, last))  # rowcount only covers the last page
                    inserted += cur.fetchone()[0] - before
        raw.commit()
    finally:
        raw.close()
    if inserted:
        db.clear_cache()
        logger.info("[daily] dummy feed inserted %d rows", inserted)
    return inserted


# ── Background loop ──────────────────────────────────────────────────────────
_status: dict[str, Any] = {"last_run": None, "last_inserted": 0, "last_error": None, "ops_date": None}


def job_status() -> dict:
    return {**_status, "feed_enabled": feed_enabled(), "targets": allowed_targets(), **daily_forecasts.status(),
            "db_cache": db.cache_stats()}


def run_cycle(get_service: Callable[[], Any], on_new_data: Optional[Callable[[], None]] = None) -> None:
    today = ops_date()
    try:
        inserted = topup_dummy_data()
        _status["last_inserted"] = inserted
        if inserted and on_new_data is not None:
            on_new_data()
        if _status.get("ops_date") != today.isoformat():
            daily_forecasts.invalidate()  # new operating day: issue fresh forecasts
            _status["ops_date"] = today.isoformat()
        for h in HORIZONS:
            daily_forecasts.get(get_service(), h)
        _status["last_error"] = None
    except Exception as exc:  # keep the loop alive; surface in /status
        logger.warning("[daily] cycle failed: %s", exc)
        _status["last_error"] = f"{type(exc).__name__}: {exc}"
    _status["last_run"] = datetime.now(IST).isoformat(timespec="seconds")


async def scheduler(get_service: Callable[[], Any], on_new_data: Optional[Callable[[], None]] = None) -> None:
    interval = float(os.environ.get("DAILY_JOB_INTERVAL_S", "900"))
    while True:
        t0 = time.time()
        await asyncio.to_thread(run_cycle, get_service, on_new_data)
        await asyncio.sleep(max(30.0, interval - (time.time() - t0)))
