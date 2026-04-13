"""
build_final_data.py
────────────────────────────────────────────────────────────────────────────
Fetches data from the pipeline API and builds a final_data.csv-compatible
DataFrame for any state.

Column output (matches final_data.csv exactly):
  date, time_block, total_drawal, Datetime,
  temperature, humidity, precipitation,
  cloud_cover, cloud_cover_low, sunshine_duration,
  direct_radiation, wind_speed_10m,
  hour, minute, day_of_week, day_of_month, month,
  is_weekend, is_holiday,
  hour_cos, dow_sin, dow_cos, block_sin, block_cos,
  is_peak_hour, is_night, is_business_hour,
  temp_squared, temp_cubed, temperature_sin

Load priority rule:
  For each day, use SLDC rows where available (up to the last SLDC block),
  then fill the remainder with `load` table rows.

Usage:
    python build_final_data.py --state HARYANA --out final_data_haryana.csv
    python build_final_data.py --state CHHATTISGARH --days 60
"""

import argparse
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ── Load env from ./env file ──────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / "env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_BASE = os.environ.get("PIPELINE_API_BASE_URL", "http://3.108.54.200:8001").rstrip("/")
API_TOKEN = os.environ.get("PIPELINE_API_TOKEN", os.environ.get("API_SECRET_KEY", "flagbearer"))
TIMEOUT = 60

INDIAN_HOLIDAYS_STATIC = {
    2023: [(1,26),(3,8),(3,30),(4,4),(4,6),(4,14),(4,21),(5,1),(6,29),(8,15),(8,31),(9,19),(10,2),(10,24),(11,12),(11,13),(12,25)],
    2024: [(1,26),(3,8),(3,25),(3,29),(4,11),(4,14),(4,17),(5,1),(6,17),(8,15),(8,19),(9,7),(10,2),(10,31),(11,1),(11,15),(12,25)],
    2025: [(1,26),(3,8),(3,14),(4,10),(4,13),(4,18),(5,1),(6,6),(8,15),(8,27),(9,27),(10,2),(10,20),(11,4),(12,25)],
    2026: [(1,26),(3,30),(4,2),(4,14),(5,1),(8,15),(10,2),(12,25)],
}

# ── API helpers ────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {API_TOKEN}"
    s.headers["Accept"] = "application/json"
    return s


_sess = _session()


def _get(path: str, params: dict) -> list[dict]:
    """GET from API, return list of row dicts."""
    url = f"{API_BASE}{path}"
    resp = _sess.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


def _fetch_table(
    path: str,
    state: str,
    from_date: str = None,
    to_date: str = None,
    days: int = 120,
) -> list[dict]:
    """
    Fetch rows from the API with optional date-range filtering.
    If from_date/to_date are provided they are sent as query params so the
    server only returns the relevant window.  The row limit is derived from
    the actual window size rather than a fixed multiplier.
    """
    params: dict = {"state": state.upper()}

    if from_date:
        params["from_date"] = from_date[:10]
    if to_date:
        params["to_date"] = to_date[:10]

    # Derive limit from window; add headroom for partial days
    window_days = _date_range_to_days(from_date, to_date, default=days)
    params["limit"] = window_days * 96 + 500

    return _get(path, params)


def _norm_load_df(rows: list[dict]) -> pd.DataFrame:
    """Parse raw API rows into a clean (date, time_block, total_drawal) frame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # time_block column name varies
    tb_col = next((c for c in ["time_block", "block"] if c in df.columns), None)
    if tb_col:
        df["time_block"] = pd.to_numeric(df[tb_col], errors="coerce").astype("Int64")
    # total_drawal column name varies
    for col in ["load_mw", "total_drawal", "drawal_mw", "demand_mw", "mw"]:
        if col in df.columns:
            df = df.rename(columns={col: "total_drawal"})
            break
    return df


def fetch_weather_mean(
    state: str,
    days: int = 120,
    from_date: str = None,
    to_date: str = None,
) -> pd.DataFrame:
    rows = _fetch_table("/weather/mean", state, from_date, to_date, days)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time_block"] = pd.to_numeric(
        df.get("time_block", df.get("block", pd.Series(dtype=float))),
        errors="coerce",
    ).astype("Int64")
    return df


def fetch_sldc(
    state: str,
    days: int = 120,
    from_date: str = None,
    to_date: str = None,
) -> pd.DataFrame:
    rows = _fetch_table("/sldc", state, from_date, to_date, days)
    return _norm_load_df(rows)


def fetch_load(
    state: str,
    days: int = 120,
    from_date: str = None,
    to_date: str = None,
) -> pd.DataFrame:
    rows = _fetch_table("/load", state, from_date, to_date, days)
    return _norm_load_df(rows)


# ── Load merge: SLDC preferred, fallback to load ──────────────────────────

def merge_load_sldc(sldc_df: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date: use SLDC rows first (up to the last valid SLDC block),
    then fill remaining blocks from the load table.

    Rule: on a given date if SLDC has data through block N, use blocks 1..N
    from SLDC and blocks N+1..96 from load (if available).
    """
    needed = ["date", "time_block", "total_drawal"]

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["date", "time_block", "total_drawal"]].copy()
        df["total_drawal"] = pd.to_numeric(df["total_drawal"], errors="coerce")
        df = df.dropna(subset=["date", "time_block", "total_drawal"])
        df["time_block"] = df["time_block"].astype(int)
        return df.sort_values(["date", "time_block"])

    sldc = _clean(sldc_df) if not sldc_df.empty else pd.DataFrame(columns=needed)
    load = _clean(load_df) if not load_df.empty else pd.DataFrame(columns=needed)

    if sldc.empty and load.empty:
        return pd.DataFrame(columns=needed)
    if sldc.empty:
        return load
    if load.empty:
        return sldc

    # Get all unique dates from both sources
    all_dates = sorted(set(sldc["date"].unique()) | set(load["date"].unique()))
    parts = []

    for d in all_dates:
        s = sldc[sldc["date"] == d].set_index("time_block")["total_drawal"]
        l = load[load["date"] == d].set_index("time_block")["total_drawal"]

        # Last block with a valid SLDC reading (non-NaN)
        valid_sldc = s.dropna()
        last_sldc_block = int(valid_sldc.index.max()) if not valid_sldc.empty else 0

        rows = []
        for blk in range(1, 97):
            if blk <= last_sldc_block:
                # SLDC zone: use SLDC value if present, else load fallback
                val = s.get(blk, None)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    rows.append({"date": d, "time_block": blk, "total_drawal": float(val)})
                elif blk in l.index:
                    rows.append({"date": d, "time_block": blk, "total_drawal": float(l[blk])})
            else:
                # Beyond SLDC: use load table
                if blk in l.index:
                    rows.append({"date": d, "time_block": blk, "total_drawal": float(l[blk])})
                elif blk in s.index:
                    # load missing but SLDC has it anyway
                    rows.append({"date": d, "time_block": blk, "total_drawal": float(s[blk])})

        if rows:
            parts.append(pd.DataFrame(rows))

    if not parts:
        return pd.DataFrame(columns=needed)

    return pd.concat(parts, ignore_index=True).sort_values(["date", "time_block"]).reset_index(drop=True)


# ── Holiday set ────────────────────────────────────────────────────────────

def _build_holiday_set() -> set:
    hols = set()
    for year in range(2020, 2031):
        hols.add(pd.Timestamp(year, 1, 26))
        hols.add(pd.Timestamp(year, 8, 15))
        hols.add(pd.Timestamp(year, 10, 2))
        hols.add(pd.Timestamp(year, 12, 25))
    for yr, dates in INDIAN_HOLIDAYS_STATIC.items():
        for m, d in dates:
            hols.add(pd.Timestamp(yr, m, d))

    try:
        import holidays as _hlib
        for yr in range(2020, 2031):
            for ts in _hlib.country_holidays("IN", years=yr):
                hols.add(pd.Timestamp(ts))
    except Exception:
        pass

    return hols


# ── Feature engineering ────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: date, time_block, total_drawal, + weather columns.
    Adds all engineered features to match final_data.csv schema.
    """
    df = df.copy()

    # ── Datetime ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["time_block"] = df["time_block"].astype(int)
    df["Datetime"] = df["date"] + pd.to_timedelta((df["time_block"]) * 15, unit="m")

    # ── Basic time components ──────────────────────────────────────────────
    df["hour"]        = df["Datetime"].dt.hour
    df["minute"]      = df["Datetime"].dt.minute
    df["day_of_week"] = df["Datetime"].dt.dayofweek      # 0=Mon … 6=Sun
    df["day_of_month"]= df["Datetime"].dt.day
    df["month"]       = df["Datetime"].dt.month

    # ── Calendar flags ─────────────────────────────────────────────────────
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    holiday_set = _build_holiday_set()
    df["is_holiday"]  = df["date"].dt.normalize().isin(holiday_set).astype(int)

    # ── Sinusoidal encodings ───────────────────────────────────────────────
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    block_theta = 2 * np.pi * df["time_block"] / 96.0
    df["block_sin"] = np.sin(block_theta)
    df["block_cos"] = np.cos(block_theta)

    # ── Period flags ───────────────────────────────────────────────────────
    df["is_peak_hour"]     = (((df["hour"] >= 6)  & (df["hour"] < 10)) |
                              ((df["hour"] >= 18) & (df["hour"] < 22))).astype(int)
    df["is_night"]         = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_business_hour"] = ((df["hour"] >= 9) & (df["hour"] < 18) & (df["is_weekend"] == 0)).astype(int)

    # ── Polynomial / trig weather features ────────────────────────────────
    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce")
        df["temp_squared"]    = t ** 2
        df["temp_cubed"]      = t ** 3
        df["temperature_sin"] = np.sin(2 * np.pi * t / 40.0)  # normalise over ~0-40°C range
    else:
        df["temp_squared"] = df["temp_cubed"] = df["temperature_sin"] = np.nan

    return df


# ── Weather column normalisation ───────────────────────────────────────────

WEATHER_COL_MAP = {
    # API name            → final_data.csv name
    "temperature_2m":       "temperature",
    "apparent_temperature": "temperature",
    "relative_humidity_2m": "humidity",
    "cloud_cover_low":      "cloud_cover_low",
    "cloud_cover_mid":      "cloud_cover_mid",
    "sunshine_duration":    "sunshine_duration",
    "direct_radiation":     "direct_radiation",
    "direct_radiation_instant": "direct_radiation",
    "wind_speed_10m":       "wind_speed_10m",
}

WEATHER_KEEP = [
    "temperature", "humidity", "precipitation",
    "cloud_cover", "cloud_cover_low",
    "sunshine_duration", "direct_radiation", "wind_speed_10m",
]


def normalise_weather(wdf: pd.DataFrame) -> pd.DataFrame:
    """Rename API columns → final_data.csv names. Fill zeros for missing cols."""
    w = wdf.copy()
    for src, dst in WEATHER_COL_MAP.items():
        if src in w.columns and dst not in w.columns:
            w = w.rename(columns={src: dst})
    # ensure all needed cols exist
    for c in WEATHER_KEEP:
        if c not in w.columns:
            w[c] = np.nan
    for c in WEATHER_KEEP:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w["precipitation"] = w["precipitation"].fillna(0.0)
    return w[["date", "time_block"] + WEATHER_KEEP]


# ── Main pipeline ──────────────────────────────────────────────────────────

FINAL_COLUMNS = [
    "date", "time_block", "total_drawal", "Datetime",
    "temperature", "humidity", "precipitation",
    "cloud_cover", "cloud_cover_low", "sunshine_duration",
    "direct_radiation", "wind_speed_10m",
    "hour", "minute", "day_of_week", "day_of_month", "month",
    "is_weekend", "is_holiday",
    "hour_cos", "dow_sin", "dow_cos", "block_sin", "block_cos",
    "is_peak_hour", "is_night", "is_business_hour",
    "temp_squared", "temp_cubed", "temperature_sin",
]


def _date_range_to_days(from_date: str = None, to_date: str = None, default: int = 120) -> int:
    """Convert a from/to date string pair into a `days` integer for the API limit calculation."""
    try:
        if from_date and to_date:
            d0 = datetime.strptime(from_date[:10], "%Y-%m-%d")
            d1 = datetime.strptime(to_date[:10], "%Y-%m-%d")
            return max(1, (d1 - d0).days + 2)
        if from_date:
            d0 = datetime.strptime(from_date[:10], "%Y-%m-%d")
            return max(1, (datetime.today() - d0).days + 2)
    except Exception:
        pass
    return default


def build_final_data(
    state: str,
    days: int = 120,
    from_date: str = None,
    to_date: str = None,
) -> pd.DataFrame:
    """
    Full pipeline:
      1. Fetch SLDC + load from API → merge with SLDC priority
      2. Fetch weather_mean from API → normalise columns
      3. Join load + weather on (date, time_block)
      4. Add engineered features
      5. Filter to [from_date, to_date] if provided
      6. Return DataFrame with FINAL_COLUMNS column order
    """
    # Compute effective days limit from date range if given
    effective_days = _date_range_to_days(from_date, to_date, default=days)

    print(f"[1/4] Fetching load data for {state.upper()} (days={effective_days}) ...")
    sldc_df = fetch_sldc(state, effective_days, from_date=from_date, to_date=to_date)
    load_df = fetch_load(state, effective_days, from_date=from_date, to_date=to_date)

    print(f"      SLDC rows: {len(sldc_df)}  |  Load rows: {len(load_df)}")
    load_merged = merge_load_sldc(sldc_df, load_df)
    print(f"      Merged load rows: {len(load_merged)}")

    if load_merged.empty:
        raise ValueError(f"No load/SLDC data found for state={state!r}. Check API connectivity.")

    print(f"[2/4] Fetching weather_mean for {state.upper()} ...")
    wdf = fetch_weather_mean(state, effective_days, from_date=from_date, to_date=to_date)
    print(f"      Weather rows: {len(wdf)}")

    if wdf.empty:
        print("      WARNING: No weather data — weather columns will be NaN.")
        weather_norm = pd.DataFrame(columns=["date", "time_block"] + WEATHER_KEEP)
    else:
        weather_norm = normalise_weather(wdf)

    print(f"[3/4] Joining load + weather ...")
    weather_norm["date"] = pd.to_datetime(weather_norm["date"], errors="coerce")
    weather_norm["time_block"] = pd.to_numeric(weather_norm["time_block"], errors="coerce").astype("Int64")

    load_merged["date"] = pd.to_datetime(load_merged["date"], errors="coerce")
    load_merged["time_block"] = load_merged["time_block"].astype("Int64")

    merged = load_merged.merge(
        weather_norm,
        on=["date", "time_block"],
        how="left",
    )

    # ── Apply date range filter ────────────────────────────────────────────
    filtered = merged.copy()
    if from_date:
        filtered = filtered[filtered["date"] >= pd.Timestamp(from_date)]
    if to_date:
        filtered = filtered[filtered["date"] <= pd.Timestamp(to_date)]

    if filtered.empty and not merged.empty:
        db_min = merged["date"].min().date()
        db_max = merged["date"].max().date()
        print(
            f"      WARNING: No data in range [{from_date} to {to_date}]. "
            f"DB contains [{db_min} to {db_max}]. Returning all available data."
        )
        filtered = merged

    merged = filtered

    print(f"[4/4] Engineering features ...")
    result = add_features(merged)

    # Reorder to final column set, fill any missing with NaN
    for col in FINAL_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[FINAL_COLUMNS].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["time_block"] = result["time_block"].astype(int)

    result = result.sort_values(["date", "time_block"]).reset_index(drop=True)

    print(f"Done. Shape: {result.shape}  |  Dates: {result['date'].min()} to {result['date'].max()}")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build final_data.csv from pipeline API")
    parser.add_argument("--state",  default="HARYANA",  help="State slug (e.g. HARYANA, CHHATTISGARH)")
    parser.add_argument("--days",   type=int, default=120, help="Number of days to fetch when no date range given (default 120)")
    parser.add_argument("--from",   dest="from_date", default=None, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--to",     dest="to_date",   default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out",    default="",        help="Output CSV path (default: final_data_<STATE>.csv)")
    args = parser.parse_args()

    state = args.state.upper()
    out_path = args.out or f"final_data_{state.lower()}.csv"

    df = build_final_data(state=state, days=args.days, from_date=args.from_date, to_date=args.to_date)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows, {df['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
