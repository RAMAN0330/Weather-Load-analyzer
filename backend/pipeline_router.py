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
                       'wind_speed_10m', 'sunshine_duration', 'direct_radiation']
            wx_df = wx_df[[c for c in wx_cols if c in wx_df.columns]]

            # Dates with actual load → historical weather (training).
            # Dates beyond the last load date → forecasted weather (T+1/T+2 target).
            load_dates = set(load_df['date'].astype(str).unique())
            last_load_date = max(load_dates)

            hist_wx = wx_df[wx_df['date'] <= last_load_date].copy()
            fore_wx = wx_df[wx_df['date'] > last_load_date].copy()

            # Merge historical weather with load (training data)
            merged = pd.merge(load_df, hist_wx, on=['date', 'time_block'], how='left')

            # Append forecasted weather rows for future dates with no actual load.
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
