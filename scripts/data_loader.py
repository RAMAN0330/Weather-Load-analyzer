"""
data_loader.py — Query PostgreSQL weather_mean + sldc for HARYANA May-June 2022-2025.
Writes two parquet cache files read by all 4 agents.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
import db  # noqa: E402  (PostgreSQL via DATABASE_URL)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'cache')
WEATHER_PARQUET = os.path.join(CACHE_DIR, 'may_jun_weather.parquet')
LOAD_PARQUET    = os.path.join(CACHE_DIR, 'may_jun_load.parquet')


def _query(sql: str) -> pd.DataFrame:
    return db._query(sql, {})


def load_weather() -> pd.DataFrame:
    query = """
        SELECT
            date, time_block,
            temperature,
            humidity,
            precipitation, precipitation AS rain, 0.0 AS showers,
            wind_speed_10m      AS wind_speed,
            cloud_cover,
            sunshine_duration,
            direct_radiation
        FROM weather_mean
        WHERE state = 'HARYANA'
          AND EXTRACT(MONTH FROM date) IN (5, 6)
          AND EXTRACT(YEAR FROM date) BETWEEN 2022 AND 2025
        ORDER BY date, time_block
    """
    df = _query(query)
    df['date'] = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df


def load_sldc() -> pd.DataFrame:
    query = """
        SELECT date, time_block, total_drawal AS load_mw
        FROM sldc
        WHERE state = 'HARYANA'
          AND EXTRACT(MONTH FROM date) IN (5, 6)
          AND EXTRACT(YEAR FROM date) BETWEEN 2022 AND 2025
        ORDER BY date, time_block
    """
    df = _query(query)
    df['date'] = pd.to_datetime(df['date'])
    return df


def build_cache(force: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and os.path.exists(WEATHER_PARQUET) and os.path.exists(LOAD_PARQUET):
        print("Cache already exists. Use force=True to rebuild.")
        return

    print("Loading weather from PostgreSQL...")
    weather = load_weather()
    print(f"  {len(weather):,} weather rows")

    print("Loading load from PostgreSQL...")
    load = load_sldc()
    print(f"  {len(load):,} load rows")

    # Join on date + time_block
    merged = weather.merge(load, on=['date', 'time_block'], how='inner')
    merged = merged.dropna(subset=['load_mw'])
    print(f"  {len(merged):,} joined rows after inner join")

    # Save separately so agents can read just what they need
    weather_cols = ['date','time_block','year','month','temperature','humidity',
                    'precipitation','rain','showers','wind_speed','cloud_cover',
                    'sunshine_duration','direct_radiation']
    merged[weather_cols].to_parquet(WEATHER_PARQUET, index=False)

    load_cols = ['date','time_block','year','month','load_mw']
    merged[load_cols].to_parquet(LOAD_PARQUET, index=False)

    print(f"Cache written:\n  {WEATHER_PARQUET}\n  {LOAD_PARQUET}")


if __name__ == '__main__':
    build_cache(force='--force' in sys.argv)
