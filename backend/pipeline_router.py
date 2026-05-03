"""
pipeline_router.py
FastAPI router that exposes the pipeline.db (SQLite) data.
Ported from Raman_pipeline_db/dashboard/backend/app.py.
Mount point: /api/pipeline  (included in main.py)
"""
import os
import sqlite3

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.linear_model import LinearRegression

try:
    import holidays as holidays_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# ── DB path ────────────────────────────────────────────────────────────────
# Priority: PIPELINE_DB_PATH env var → sibling-folder default (dev machine)
_DEFAULT_DB = os.path.abspath(os.path.join(
    os.path.dirname(__file__),          # backend/
    "..",                                # RD root
    "..", "Haryana",                     # Desktop/Haryana
    "Raman_pipeline_db", "pipeline.db"
))
DB_PATH = os.getenv("PIPELINE_DB_PATH", _DEFAULT_DB)


# ── Helpers ────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def filter_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep only rows within the last `days` days based on the 'date' column."""
    if days <= 0 or 'date' not in df.columns:
        return df
    from datetime import date as date_type, timedelta
    cutoff = (date_type.today() - timedelta(days=days)).isoformat()
    df['date'] = df['date'].astype(str)
    return df[df['date'] >= cutoff]


def get_all_tables():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column aliases to standard names."""
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    for alias in ['block', 'time_block', 'block_no', 'blockno', 'hour_block']:
        if alias in cols_lower and 'time_block' not in [c.lower() for c in df.columns]:
            rename_map[cols_lower[alias]] = 'time_block'
            break

    for alias in ['total_drawal', 'drawal', 'mw', 'demand', 'actual', 'actual_load', 'load_mw']:
        if alias in cols_lower and 'load' not in [c.lower() for c in df.columns]:
            rename_map[cols_lower[alias]] = 'load'
            break

    for alias in ['forecast', 'forecasted_load', 'predicted', 'forecast_mw']:
        if alias in cols_lower and 'forecasted' not in [c.lower() for c in df.columns]:
            rename_map[cols_lower[alias]] = 'forecasted'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'time_block' in df.columns:
        df['time_block'] = pd.to_numeric(df['time_block'], errors='coerce')

    return df


def get_day_category(date_str: str, state: str) -> str:
    try:
        from datetime import date as date_type
        d = date_type.fromisoformat(date_str)
        if HAS_HOLIDAYS:
            country_holidays = holidays_lib.India(years=d.year)
            if d in country_holidays:
                return "holiday"
        if d.weekday() == 6:
            return "sunday"
        if d.weekday() == 5:
            return "holiday"
        return "working"
    except Exception:
        return "working"


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/states")
def pipeline_get_states():
    tables = get_all_tables()
    known_suffixes = ['load', 'sldc', 'weather_mean', 'weather_loc', 'sldc_forecast', 'forecast']
    states = set()
    for t in tables:
        for suffix in known_suffixes:
            if t.lower().endswith('_' + suffix):
                state = t[:-(len(suffix) + 1)].upper()
                states.add(state)
                break
    return sorted(list(states))


@router.get("/weather/{state}")
def pipeline_get_weather(state: str, days: int = Query(60)):
    table = f"{state.upper()}_weather_mean"
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]
    if table.lower() not in tables_lower:
        return []
    actual_table = tables[tables_lower.index(table.lower())]
    try:
        conn = get_conn()
        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', conn)
        conn.close()
        df = normalize_df(df)
        if 'date' not in df.columns:
            for c in df.columns:
                if 'date' in c.lower():
                    df = df.rename(columns={c: 'date'})
                    break
        df = filter_last_days(df, days)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather-loc/{state}")
def pipeline_get_weather_loc(state: str, days: int = Query(60)):
    table = f"{state.upper()}_weather_loc"
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]
    if table.lower() not in tables_lower:
        return []
    actual_table = tables[tables_lower.index(table.lower())]
    try:
        conn = get_conn()
        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', conn)
        conn.close()
        df = normalize_df(df)
        if 'date' not in df.columns:
            for c in df.columns:
                if 'date' in c.lower():
                    df = df.rename(columns={c: 'date'})
                    break
        df = filter_last_days(df, days)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circle-impact/{state}")
def pipeline_get_circle_impact(state: str, days: int = Query(365)):
    """
    Haryana-specific (but works if table exists for any state):
    Rank circles by impact on total state load using history.

    Returns per-circle:
      - avg_share_pct (energy share proxy)
      - corr_daily_mean (circle vs total daily mean)
      - beta_mw_per_mw (slope from linear regression total~circle)
      - r2
    """
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]

    state_norm = state.upper().replace(" ", "_").replace("-", "_")
    circle_candidates = [
        f"{state_norm}_circle_load",
        f"{state_norm}_circleload",
        f"{state.lower()}_circle_load",
        f"{state.lower()}_circleload",
    ]
    circle_table = None
    for c in circle_candidates:
        if c.lower() in tables_lower:
            circle_table = tables[tables_lower.index(c.lower())]
            break
    if circle_table is None:
        return {"state": state_norm, "available": False, "circles": []}

    load_candidates = [
        f"{state_norm}_sldc", f"{state_norm}_load",
        f"{state.lower()}_sldc", f"{state.lower()}_load",
    ]
    load_table = None
    for c in load_candidates:
        if c.lower() in tables_lower:
            load_table = tables[tables_lower.index(c.lower())]
            break
    if load_table is None:
        raise HTTPException(status_code=404, detail=f"No load table found for state '{state}'")

    try:
        conn = get_conn()
        cdf = pd.read_sql_query(f'SELECT * FROM "{circle_table}"', conn)
        ldf = pd.read_sql_query(f'SELECT * FROM "{load_table}"', conn)
        conn.close()

        cdf = normalize_df(cdf)
        ldf = normalize_df(ldf)

        # Ensure date column name
        if "date" not in cdf.columns:
            for col in cdf.columns:
                if "date" in col.lower():
                    cdf = cdf.rename(columns={col: "date"})
                    break
        if "date" not in ldf.columns:
            for col in ldf.columns:
                if "date" in col.lower():
                    ldf = ldf.rename(columns={col: "date"})
                    break

        cdf = filter_last_days(cdf, days)
        ldf = filter_last_days(ldf, days)

        # Circle name column detection
        circle_col = None
        for cand in ["circle", "circle_name", "zone", "area", "location", "discom_circle"]:
            if cand in cdf.columns:
                circle_col = cand
                break
        if circle_col is None:
            non_key = [c for c in cdf.columns if c not in {"date", "time_block"}]
            # Prefer first object-like column
            for c in non_key:
                if cdf[c].dtype == object:
                    circle_col = c
                    break
        if circle_col is None:
            raise HTTPException(status_code=500, detail="Could not detect circle name column in circle_load table.")

        # Circle load value detection
        value_col = None
        for cand in ["total_drawal", "load", "load_mw", "mw", "demand", "value"]:
            if cand in cdf.columns:
                value_col = cand
                break
        if value_col is None:
            numeric_cols = [c for c in cdf.columns if c not in {"date", "time_block", circle_col} and pd.api.types.is_numeric_dtype(cdf[c])]
            if numeric_cols:
                value_col = numeric_cols[0]
        if value_col is None:
            raise HTTPException(status_code=500, detail="Could not detect circle load value column in circle_load table.")

        # Total load value
        total_col = "total_drawal" if "total_drawal" in ldf.columns else ("load" if "load" in ldf.columns else None)
        if total_col is None:
            raise HTTPException(status_code=500, detail="Could not detect total load column in load table.")

        cdf["date"] = cdf["date"].astype(str)
        ldf["date"] = ldf["date"].astype(str)
        cdf["time_block"] = pd.to_numeric(cdf.get("time_block"), errors="coerce")
        ldf["time_block"] = pd.to_numeric(ldf.get("time_block"), errors="coerce")
        cdf[value_col] = pd.to_numeric(cdf[value_col], errors="coerce")
        ldf[total_col] = pd.to_numeric(ldf[total_col], errors="coerce")

        cdf = cdf.dropna(subset=["date", "time_block", circle_col, value_col])
        ldf = ldf.dropna(subset=["date", "time_block", total_col])
        cdf["time_block"] = cdf["time_block"].astype(int)
        ldf["time_block"] = ldf["time_block"].astype(int)

        merged = pd.merge(
            cdf[["date", "time_block", circle_col, value_col]],
            ldf[["date", "time_block", total_col]],
            on=["date", "time_block"],
            how="inner",
        )
        if merged.empty:
            return {"state": state_norm, "available": True, "circles": []}

        merged = merged.rename(columns={circle_col: "circle", value_col: "circle_load", total_col: "total_load"})

        # Daily aggregates for correlation/regression
        daily = merged.groupby(["date", "circle"], as_index=False)[["circle_load", "total_load"]].mean()

        out = []
        for circle, grp in daily.groupby("circle"):
            x = grp["circle_load"].to_numpy(dtype=float).reshape(-1, 1)
            y = grp["total_load"].to_numpy(dtype=float).reshape(-1)
            if len(y) < 8 or not np.isfinite(y).all() or not np.isfinite(x).all():
                continue
            # correlation on daily means
            corr = float(np.corrcoef(x.reshape(-1), y)[0, 1]) if len(y) >= 2 else 0.0
            reg = LinearRegression()
            reg.fit(x, y)
            r2 = float(reg.score(x, y))
            beta = float(reg.coef_.reshape(-1)[0])

            # share (energy proxy): sum(circle)/sum(total) over merged history
            share = float(
                merged.loc[merged["circle"] == circle, "circle_load"].sum()
                / max(float(merged["total_load"].sum()), 1e-9)
                * 100.0
            )
            out.append(
                {
                    "circle": str(circle),
                    "n_days": int(grp["date"].nunique()),
                    "avg_share_pct": round(share, 3),
                    "corr_daily_mean": round(corr, 4),
                    "beta_mw_per_mw": round(beta, 4),
                    "r2": round(r2, 4),
                }
            )

        out = sorted(out, key=lambda r: (r.get("avg_share_pct", 0.0), r.get("r2", 0.0)), reverse=True)
        return {
            "state": state_norm,
            "available": True,
            "table": circle_table,
            "days_window": int(days),
            "circles": out,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load/{state}")
def pipeline_get_load(state: str, days: int = Query(60)):
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]
    candidates = [
        f"{state.upper()}_sldc", f"{state.upper()}_load",
        f"{state.lower()}_sldc", f"{state.lower()}_load",
    ]
    actual_table = None
    for c in candidates:
        if c.lower() in tables_lower:
            actual_table = tables[tables_lower.index(c.lower())]
            break
    if actual_table is None:
        return []
    try:
        conn = get_conn()
        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', conn)
        conn.close()
        df = normalize_df(df)
        if 'date' not in df.columns:
            for c in df.columns:
                if 'date' in c.lower():
                    df = df.rename(columns={c: 'date'})
                    break
        df = filter_last_days(df, days)
        cols = ['date', 'time_block']
        if 'load' in df.columns:
            cols.append('load')
        df_out = df[[c for c in cols if c in df.columns]]
        return df_out.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{state}")
def pipeline_get_forecast(state: str, days: int = Query(60)):
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]
    candidates = [
        f"{state.upper()}_sldc_forecast", f"{state.upper()}_forecast",
        f"{state.lower()}_sldc_forecast", f"{state.lower()}_forecast",
    ]
    actual_table = None
    for c in candidates:
        if c.lower() in tables_lower:
            actual_table = tables[tables_lower.index(c.lower())]
            break
    if actual_table is None:
        return []
    try:
        conn = get_conn()
        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', conn)
        conn.close()
        df = normalize_df(df)
        if 'date' not in df.columns:
            for c in df.columns:
                if 'date' in c.lower():
                    df = df.rename(columns={c: 'date'})
                    break
        df = filter_last_days(df, days)
        cols = ['date', 'time_block']
        if 'forecasted' in df.columns:
            cols.append('forecasted')
        df_out = df[[c for c in cols if c in df.columns]]
        return df_out.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similarity/{state}")
def pipeline_get_similarity(
    state: str,
    target_date: str = Query(...),
    method: str = Query("euclidean"),
    top_k: int = Query(5),
):
    table = f"{state.upper()}_weather_mean"
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]
    if table.lower() not in tables_lower:
        raise HTTPException(status_code=404, detail=f"Table {table} not found")
    actual_table = tables[tables_lower.index(table.lower())]
    try:
        conn = get_conn()
        df = pd.read_sql_query(f'SELECT * FROM "{actual_table}"', conn)
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = normalize_df(df)
    if 'date' not in df.columns:
        for c in df.columns:
            if 'date' in c.lower():
                df = df.rename(columns={c: 'date'})
                break

    required_cols = ['temperature_2m', 'relative_humidity_2m', 'precipitation']
    available_cols = [c for c in required_cols if c in df.columns]
    if not available_cols:
        raise HTTPException(status_code=400, detail="No weather feature columns found")

    df['date'] = df['date'].astype(str)
    if 'time_block' in df.columns:
        df = df.sort_values(['date', 'time_block'])

    grouped = df.groupby('date')
    daily_profiles = {}
    for d, grp in grouped:
        profile = {}
        for col in available_cols:
            vals = grp[col].dropna().tolist()
            profile[col] = vals
        daily_profiles[d] = profile

    if target_date not in daily_profiles:
        raise HTTPException(status_code=404, detail=f"Target date {target_date} not found in data")

    target_profile = daily_profiles[target_date]

    def profile_to_vec(profile):
        vecs = []
        for col in available_cols:
            vals = profile.get(col, [])
            if len(vals) == 0:
                vecs.extend([0.0] * 96)
            elif len(vals) >= 96:
                vecs.extend(vals[:96])
            else:
                padded = vals + [vals[-1]] * (96 - len(vals))
                vecs.extend(padded)
        return np.array(vecs, dtype=float)

    target_vec = profile_to_vec(target_profile)
    weights_map = {'temperature_2m': 1.0, 'relative_humidity_2m': 0.5, 'precipitation': 0.3}

    results = []
    for d, profile in daily_profiles.items():
        if d == target_date:
            continue
        vec = profile_to_vec(profile)
        if len(vec) != len(target_vec):
            continue
        if method == "weighted":
            w = []
            for col in available_cols:
                wt = weights_map.get(col, 1.0)
                w.extend([wt] * 96)
            w = np.array(w[:len(target_vec)])
            dist = float(np.sqrt(np.sum(w * (target_vec - vec) ** 2)))
        else:
            dist = float(np.sqrt(np.sum((target_vec - vec) ** 2)))
        results.append({'date': d, 'distance': round(dist, 4)})

    results.sort(key=lambda x: x['distance'])
    top_results = results[:top_k]

    from datetime import date as date_type
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for r in top_results:
        try:
            d_obj = date_type.fromisoformat(r['date'])
            r['day_of_week'] = day_names[d_obj.weekday()]
        except Exception:
            r['day_of_week'] = 'Unknown'
        r['category'] = get_day_category(r['date'], state)

    profile_dates = [target_date] + [r['date'] for r in top_results]
    profiles_out = {}
    for d in profile_dates:
        if d in daily_profiles:
            p = daily_profiles[d]
            profiles_out[d] = {
                'temp': p.get('temperature_2m', []),
                'humidity': p.get('relative_humidity_2m', []),
                'precip': p.get('precipitation', []),
            }

    return {
        'target_date': target_date,
        'similar': top_results,
        'profiles': profiles_out,
    }


@router.get("/engine-data/{state}")
def pipeline_get_engine_data(state: str):
    """
    Return a merged load + weather DataFrame in the exact format that engine.py
    expects from final_data.csv:
      date, time_block, total_drawal, temperature, humidity, precipitation,
      cloud_cover, cloud_cover_low, wind_speed_10m, sunshine_duration,
      direct_radiation
    The caller (main.py) can load this into the EDAEngine directly.
    """
    state_upper = state.upper()
    tables = get_all_tables()
    tables_lower = [t.lower() for t in tables]

    # ── Find load table ────────────────────────────────────────────────────
    load_candidates = [
        f"{state_upper}_load", f"{state_upper}_sldc",
        f"{state.lower()}_load", f"{state.lower()}_sldc",
    ]
    load_table = None
    for c in load_candidates:
        if c.lower() in tables_lower:
            load_table = tables[tables_lower.index(c.lower())]
            break
    if load_table is None:
        raise HTTPException(status_code=404, detail=f"No load table found for state '{state}'")

    # ── Find weather table ─────────────────────────────────────────────────
    weather_candidates = [f"{state_upper}_weather_mean", f"{state.lower()}_weather_mean"]
    weather_table = None
    for c in weather_candidates:
        if c.lower() in tables_lower:
            weather_table = tables[tables_lower.index(c.lower())]
            break

    try:
        conn = get_conn()

        # Load data
        load_df = pd.read_sql_query(f'SELECT * FROM "{load_table}"', conn)
        # Normalise load column → total_drawal
        lcols = {c.lower(): c for c in load_df.columns}
        for alias in ['total_drawal', 'load_mw', 'load', 'drawal', 'mw', 'demand', 'actual']:
            if alias in lcols and alias != 'total_drawal':
                load_df = load_df.rename(columns={lcols[alias]: 'total_drawal'})
                break
            elif alias == 'total_drawal' and alias in lcols:
                break
        load_df['date'] = load_df['date'].astype(str)
        load_df['time_block'] = pd.to_numeric(load_df['time_block'], errors='coerce')
        load_df = load_df[['date', 'time_block', 'total_drawal']].dropna(subset=['time_block', 'total_drawal'])

        # Weather data
        if weather_table:
            wx_df = pd.read_sql_query(f'SELECT * FROM "{weather_table}"', conn)
            wx_df['date'] = wx_df['date'].astype(str)
            wx_df['time_block'] = pd.to_numeric(wx_df['time_block'], errors='coerce')

            # Rename weather columns to match engine.py expectations
            wx_rename = {
                'temperature_2m': 'temperature',
                'relative_humidity_2m': 'humidity',
            }
            wx_df = wx_df.rename(columns=wx_rename)

            wx_cols = ['date', 'time_block', 'temperature', 'humidity',
                       'precipitation', 'cloud_cover', 'cloud_cover_low',
                       'wind_speed_10m', 'wind_speed_80m',
                       'sunshine_duration', 'direct_radiation']
            wx_df = wx_df[[c for c in wx_cols if c in wx_df.columns]]

            # Dates with actual load → historical weather (training).
            # Dates beyond the last load date → forecasted weather (T+1/T+2 target).
            load_dates = set(load_df['date'].astype(str).unique())
            last_load_date = max(load_dates)

            # Always ensure we cover exactly 2 days ahead from today:
            #   T+1 = today (or next day if today already has load)
            #   T+2 = today + 1
            from datetime import date as _date, timedelta as _td
            _today = _date.today()
            _t1_str = _today.isoformat()
            _t2_str = (_today + _td(days=1)).isoformat()
            _forecast_dates = {_t1_str, _t2_str}

            hist_wx = wx_df[wx_df['date'] <= last_load_date].copy()

            # fore_wx = weather rows for dates that are either:
            #   (a) strictly after last_load_date, OR
            #   (b) one of the 2 explicit forward dates (T+1 / T+2)
            # This guarantees both T+1 and T+2 weather are always present
            # regardless of where last_load_date falls relative to today.
            fore_mask = (wx_df['date'] > last_load_date) | (wx_df['date'].isin(_forecast_dates))
            # Exclude dates that already have actual load so we don't double-append
            fore_mask &= ~wx_df['date'].isin(load_dates)
            fore_wx = wx_df[fore_mask].copy()

            # Merge historical weather with load (training data)
            merged = pd.merge(load_df, hist_wx, on=['date', 'time_block'], how='left')

            # Append forecasted weather rows for T+1/T+2 with no actual load.
            # The pipeline will pick these up as target_df weather for T+1/T+2.
            if not fore_wx.empty:
                fore_wx = fore_wx.copy()
                fore_wx['total_drawal'] = np.nan
                merged = pd.concat([merged, fore_wx], ignore_index=True)
        else:
            merged = load_df

        conn.close()

        # Build Datetime column
        merged['Datetime'] = pd.to_datetime(merged['date']) + pd.to_timedelta(
            (merged['time_block'] - 1) * 15, unit='m'
        )
        merged = merged.sort_values('Datetime').reset_index(drop=True)

        return merged.to_dict(orient='records')

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
