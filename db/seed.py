"""Fill PostgreSQL with dummy data for Forecast Studio.

    python -m db.seed            # seed if the load table is empty
    python -m db.seed --force    # wipe and re-seed

Data comes from the deterministic generator in ``mockdata`` (realistic state
demand driven by district weather; see mockdata/generator.py). Environment:

  DATABASE_URL   postgresql+psycopg2://user:pass@host:5432/db
  SEED_STATES    comma list (default: all generator states)
  SEED_FROM      first date (default 2023-01-01)
  DEMO_USERNAME / DEMO_EMAIL / DEMO_PASSWORD   demo login (default demo / demo12345)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mockdata  # noqa: E402
from mockdata.generator import TABLES as MOCK_TABLES  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

COLUMNS = {
    "load": ["state", "date", "time_block", "total_drawal"],
    "sldc": ["state", "date", "time_block", "total_drawal"],
    "weather_mean": ["state", "date", "time_block", *mockdata.generator.WEATHER_COLS],
    "weather_loc": ["state", "date", "time_block", "location", *mockdata.generator.WEATHER_COLS],
    "forecast": ["state", "date", "block", "forecast_mw"],
    "sldc_forecast": ["state", "date", "time_block", "forecast_mw"],
}
FRAMES = {
    "load": lambda s, f, t: mockdata.load_frame(s, f, t, table="load"),
    "sldc": lambda s, f, t: mockdata.load_frame(s, f, t, table="sldc"),
    "weather_mean": mockdata.weather_mean_frame,
    "weather_loc": mockdata.weather_loc_frame,
    "forecast": lambda s, f, t: mockdata.forecast_frame(s, f, t, table="forecast"),
    "sldc_forecast": lambda s, f, t: mockdata.forecast_frame(s, f, t, table="sldc_forecast"),
}
CHUNK_DAYS = 120


def _copy(raw_conn, table: str, df) -> None:
    buf = io.StringIO()
    df[COLUMNS[table]].to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with raw_conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({', '.join(COLUMNS[table])}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)


def seed(force: bool = False) -> None:
    url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://forecast:forecast@localhost:5432/forecast_studio")
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text((ROOT / "db" / "init" / "01_schema.sql").read_text()))
        existing = conn.execute(text("SELECT COUNT(*) FROM load")).scalar()
        if existing and not force:
            print(f"[seed] load already has {existing} rows; skipping data (use --force to re-seed).")
            _seed_demo_user()
            return
        conn.execute(text("TRUNCATE " + ", ".join(MOCK_TABLES)))

    states = [s.strip().upper() for s in os.environ.get("SEED_STATES", ",".join(mockdata.states())).split(",") if s.strip()]
    start = os.environ.get("SEED_FROM", mockdata.DATA_START.isoformat())
    import pandas as pd

    raw = eng.raw_connection()
    try:
        for state in states:
            for table in MOCK_TABLES:
                lo, hi = mockdata.date_range(state, table)
                t0, n = time.time(), 0
                for chunk_start in pd.date_range(max(pd.Timestamp(start), pd.Timestamp(lo)), hi, freq=f"{CHUNK_DAYS}D"):
                    chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS - 1), pd.Timestamp(hi))
                    df = FRAMES[table](state, chunk_start.date().isoformat(), chunk_end.date().isoformat())
                    if not df.empty:
                        _copy(raw, table, df)
                        n += len(df)
                raw.commit()
                print(f"[seed] {state:<13} {table:<14} {n:>9,} rows  {time.time() - t0:5.1f}s", flush=True)
    finally:
        raw.close()
    with eng.begin() as conn:
        conn.execute(text("ANALYZE"))
    _seed_demo_user()


def _seed_demo_user() -> None:
    """Create the demo login in the auth tables (same database)."""
    os.environ.setdefault("AUTH_DB_URL", os.environ.get("DATABASE_URL", ""))
    from backend.auth_db import SessionLocal, create_user, get_user_by_username  # noqa: WPS433
    from backend.auth_router import _hash_pw  # noqa: WPS433

    username = os.environ.get("DEMO_USERNAME", "demo")
    with SessionLocal() as db:
        if get_user_by_username(db, username):
            print(f"[seed] demo user '{username}' exists.")
            return
        create_user(db, username, os.environ.get("DEMO_EMAIL", "demo@forecast.studio"),
                    _hash_pw(os.environ.get("DEMO_PASSWORD", "demo12345")))
        print(f"[seed] created demo user '{username}'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="wipe and re-seed")
    seed(force=ap.parse_args().force)
