"""Precompute and cache learned per-location weights for a given state.

Usage:
    python scripts/learn_location_weights.py --state CHHATTISGARH
    python scripts/learn_location_weights.py --state ODISHA --lookback-days 180

The result is written to `exports/location_weights_<state>.json` and looked
up at forecast time by `_fetch_t2_weather_from_db` for non-Haryana states.

Inputs the script needs:
  - State load history. Auto-resolved from final_data.csv (or --load-csv path).
  - Per-location weather history. Fetched from
    `${PIPELINE_API_BASE_URL}/weather/by_location?state=...&start=...&end=...`
    using PIPELINE_API_TOKEN, OR loaded from --weather-csv if you have an export.

If the upstream endpoint isn't reachable for the given window, the script
exits non-zero with a clear message.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.learned_location_weights import (  # noqa: E402
    compute_location_weights,
    save_weights,
)


def _fetch_per_location(state: str, start: str, end: str, base_url: str, token: str) -> pd.DataFrame:
    rows: list[dict] = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        date_str = cur.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                f"{base_url.rstrip('/')}/weather/by_location",
                params={"state": state, "date": date_str, "limit": 5000},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            print(f"  [warn] {date_str}: {exc}", file=sys.stderr)
            cur += pd.Timedelta(days=1)
            continue
        if resp.status_code != 200:
            print(f"  [warn] {date_str}: HTTP {resp.status_code}", file=sys.stderr)
            cur += pd.Timedelta(days=1)
            continue
        try:
            payload = resp.json()
        except ValueError:
            cur += pd.Timedelta(days=1)
            continue
        if isinstance(payload, list) and payload:
            for r in payload:
                r["date"] = date_str
                rows.append(r)
        cur += pd.Timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    rename = {"temperature_2m": "temperature", "relative_humidity_2m": "humidity"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "location" not in df.columns:
        for alt in ("city", "station", "name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "location"})
                break
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--load-csv", default=str(ROOT / "final_data.csv"))
    parser.add_argument("--weather-csv", default=None,
                        help="Optional pre-exported per-location weather CSV (skips API fetch)")
    parser.add_argument("--api-base", default=os.environ.get("PIPELINE_API_BASE_URL", "http://3.108.54.200:8001"))
    parser.add_argument("--api-token", default=os.environ.get("PIPELINE_API_TOKEN", os.environ.get("API_SECRET_KEY", "flagbearer")))
    args = parser.parse_args()

    state = args.state.upper()
    print(f"[learn] state={state}, lookback={args.lookback_days}d")

    load_path = Path(args.load_csv)
    if not load_path.exists():
        print(f"  [err] load CSV not found: {load_path}", file=sys.stderr)
        return 2
    df_load = pd.read_csv(load_path, usecols=["date", "time_block", "total_drawal"])
    df_load["date"] = pd.to_datetime(df_load["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_load = df_load.dropna()
    if df_load.empty:
        print("  [err] load CSV is empty after parse", file=sys.stderr)
        return 2

    end = pd.to_datetime(df_load["date"]).max()
    start = end - pd.Timedelta(days=args.lookback_days)
    print(f"  [load] {len(df_load):,} rows, {start.date()} → {end.date()}")

    if args.weather_csv:
        df_loc = pd.read_csv(args.weather_csv)
        print(f"  [weather] from CSV: {args.weather_csv} ({len(df_loc):,} rows)")
    else:
        print(f"  [weather] fetching from {args.api_base} ...")
        df_loc = _fetch_per_location(state, str(start.date()), str(end.date()), args.api_base, args.api_token)
        print(f"  [weather] fetched {len(df_loc):,} rows, {df_loc.get('location', pd.Series()).nunique()} locations")

    if df_loc.empty:
        print("  [err] no per-location weather data — cannot fit weights", file=sys.stderr)
        return 3

    weights = compute_location_weights(
        state=state,
        df_load=df_load,
        df_weather_loc=df_loc,
        lookback_days=args.lookback_days,
        persist=True,
    )

    if not weights:
        print("  [err] compute_location_weights returned empty", file=sys.stderr)
        return 4

    print(f"  [weights]")
    for loc, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"    {loc:24s} {100*w:6.2f}%")
    print(f"  [done] cached to exports/location_weights_{state.lower()}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
