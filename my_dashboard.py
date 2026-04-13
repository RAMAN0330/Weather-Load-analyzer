import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from typing import Any
from datetime import date, timedelta, datetime

# =========================
# CONFIG
# =========================
API_BASE_URL = os.environ.get("PIPELINE_API_BASE_URL", "http://3.108.54.200:8001").strip().rstrip("/")
API_TOKEN = os.environ.get("PIPELINE_API_TOKEN", os.environ.get("API_SECRET_KEY", "flagbearer")).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


API_READ_LIMIT = _env_int("DASHBOARD_API_LIMIT", 200000)
API_TIMEOUT_SEC = _env_float("DASHBOARD_API_TIMEOUT", 45.0)

_API_ENDPOINT_PATHS = {
    "weather_mean": "/weather/mean",
    "weather_loc": "/weather/loc",
    "load": "/load",
    "forecast": "/forecast",
    "sldc": "/sldc",
    "sldc_forecast": "/sldc/forecast",
}

_TABLE_ENDPOINT_ALIASES = [
    ("_SLDC_FORECAST", "sldc_forecast"),
    ("_WEATHER_MEAN", "weather_mean"),
    ("_WEATHER_LOC", "weather_loc"),
    ("_SLDC", "sldc"),
    ("_FORECAST", "forecast"),
    ("_LOAD_MEAN", "load"),
    ("_DEMAND_MEAN", "load"),
    ("_DEMAND", "load"),
    ("_LOAD", "load"),
]


def _is_api_configured() -> bool:
    return bool(API_BASE_URL and API_TOKEN)


@st.cache_resource(show_spinner=False)
def _get_api_session(base_url: str, token: str) -> requests.Session:
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def _api_get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    if not API_BASE_URL:
        raise RuntimeError("PIPELINE_API_BASE_URL is not configured.")
    if not API_TOKEN:
        raise RuntimeError("PIPELINE_API_TOKEN is not configured.")

    session = _get_api_session(API_BASE_URL, API_TOKEN)
    resp = session.get(f"{API_BASE_URL}{path}", params=params, timeout=API_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def _parse_state_table_name(table: str) -> tuple[str | None, str | None]:
    table_name = str(table or "").strip().upper()
    for suffix, endpoint in _TABLE_ENDPOINT_ALIASES:
        if table_name.endswith(suffix):
            state_slug = table_name[:-len(suffix)].strip("_")
            if state_slug:
                return state_slug, endpoint
    return None, None


@st.cache_data(show_spinner=False, ttl=60)
def _api_healthcheck() -> tuple[bool, str]:
    try:
        _api_get_json("/")
        return True, ""
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False, ttl=300)
def _read_api_endpoint(endpoint: str, state_slug: str, limit: int = API_READ_LIMIT) -> pd.DataFrame:
    if endpoint not in _API_ENDPOINT_PATHS:
        return pd.DataFrame()

    params = {"state": state_slug.upper(), "limit": max(1, int(limit))}
    payload = _api_get_json(_API_ENDPOINT_PATHS[endpoint], params=params)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=120)
def _api_has_rows(endpoint: str, state_slug: str) -> bool:
    df = _read_api_endpoint(endpoint, state_slug, limit=1)
    return not df.empty

# Supported states
STATE_OPTIONS = {
    "HARYANA": {"display": "Haryana", "holiday_code": "HR"},
    "CHHATTISGARH": {"display": "Chhattisgarh", "holiday_code": "CG"},
    "RAJASTHAN": {"display": "Rajasthan", "holiday_code": "RJ"},
    "PUNJAB": {"display": "Punjab", "holiday_code": "PB"},
    "DELHI": {"display": "Delhi", "holiday_code": "DL"},
    "UTTAR_PRADESH": {"display": "Uttar Pradesh", "holiday_code": "UP"},
    "UTTARAKHAND": {"display": "Uttarakhand", "holiday_code": "UK"},
    "HIMACHAL_PRADESH": {"display": "Himachal Pradesh", "holiday_code": "HP"},
    "GUJARAT": {"display": "Gujarat", "holiday_code": "GJ"},
    "MADHYA_PRADESH": {"display": "Madhya Pradesh", "holiday_code": "MP"},
    "MAHARASHTRA": {"display": "Maharashtra", "holiday_code": "MH"},
    "BIHAR": {"display": "Bihar", "holiday_code": "BR"},
    "JHARKHAND": {"display": "Jharkhand", "holiday_code": "JH"},
    "ODISHA": {"display": "Odisha", "holiday_code": "OD"},
    "WEST_BENGAL": {"display": "West Bengal", "holiday_code": "WB"},
    "ASSAM": {"display": "Assam", "holiday_code": "AS"},
    "TAMIL_NADU": {"display": "Tamil Nadu", "holiday_code": "TN"},
    "KARNATAKA": {"display": "Karnataka", "holiday_code": "KA"},
    "TELANGANA": {"display": "Telangana", "holiday_code": "TS"},
    "ANDHRA_PRADESH": {"display": "Andhra Pradesh", "holiday_code": "AP"},
    "KERALA": {"display": "Kerala", "holiday_code": "KL"},
}
DEFAULT_STATE_SLUG = "CHHATTISGARH"

# WEATHER defaults
DEFAULT_TS_DAYS    = 5
DEFAULT_OVERLAY_N  = 2

# LOAD defaults
DEFAULT_LOAD_TS_DAYS = 5

# PEAK WINDOWS (block ranges, inclusive)
MORNING_BLOCKS = (24, 40)
SOLAR_BLOCKS   = (40, 72)
EVENING_BLOCKS = (73, 92)


def _slug_to_display(slug: str) -> str:
    return STATE_OPTIONS.get(slug, {}).get("display", slug.replace("_", " ").title())


def _get_state_meta(slug: str) -> dict:
    meta = STATE_OPTIONS.get(slug, {})
    return {
        "slug": slug,
        "display": meta.get("display", _slug_to_display(slug)),
        "holiday_code": meta.get("holiday_code"),
    }


def _scan_db_state_slugs() -> list[str]:
    try:
        states = _api_get_json("/states")
        found = {
            str(s).strip().upper()
            for s in (states if isinstance(states, list) else [])
            if str(s).strip()
        }
    except Exception:
        found = set()

    if not found:
        return list(STATE_OPTIONS.keys())

    known = [slug for slug in STATE_OPTIONS.keys() if slug in found]
    unknown = sorted(slug for slug in found if slug not in STATE_OPTIONS)
    return known + unknown


def _set_active_state(state_slug: str):
    global STATE_SLUG, STATE_DISPLAY, STATE_HOLIDAY_CODE
    meta = _get_state_meta(state_slug)
    STATE_SLUG = meta["slug"]
    STATE_DISPLAY = meta["display"]
    STATE_HOLIDAY_CODE = meta["holiday_code"]


def _get_india_holidays(years: list[int], holiday_code: str | None):
    try:
        import holidays as _holidays
    except Exception:
        return {}

    years = [int(y) for y in years if pd.notna(y)]
    if not years:
        return {}

    if holiday_code:
        try:
            return _holidays.country_holidays("IN", subdiv=holiday_code, years=years)
        except TypeError:
            pass
        except Exception:
            pass

        try:
            return _holidays.country_holidays("IN", state=holiday_code, years=years)
        except Exception:
            pass

    try:
        return _holidays.country_holidays("IN", years=years)
    except Exception:
        return {}


ACTIVE_STATE_OPTIONS = _scan_db_state_slugs()
if not ACTIVE_STATE_OPTIONS:
    ACTIVE_STATE_OPTIONS = [DEFAULT_STATE_SLUG]
if DEFAULT_STATE_SLUG not in ACTIVE_STATE_OPTIONS:
    DEFAULT_STATE_SLUG = ACTIVE_STATE_OPTIONS[0]

_set_active_state(DEFAULT_STATE_SLUG)

st.set_page_config(page_title="State Dashboard — Weather & Load", page_icon="⚡", layout="wide")

# =========================
# DB HELPERS
# =========================

def _table_exists(table: str) -> bool:
    state_slug, endpoint = _parse_state_table_name(table)
    if not state_slug or not endpoint:
        return False
    try:
        return _api_has_rows(endpoint, state_slug)
    except Exception:
        return False

def _find_first_existing_table(candidates: list[str]) -> str | None:
    for table in candidates:
        if _table_exists(table):
            return table
    return None

@st.cache_data(show_spinner=False, ttl=300)
def _read_table(table: str) -> pd.DataFrame:
    state_slug, endpoint = _parse_state_table_name(table)
    if not state_slug or not endpoint:
        return pd.DataFrame()
    try:
        df = _read_api_endpoint(endpoint, state_slug, limit=API_READ_LIMIT)
    except Exception:
        df = pd.DataFrame()
    return df

# =========================
# SHARED HELPERS
# =========================

def _overlay_init(session_key: str, defaults: list[date]):
    if session_key not in st.session_state:
        st.session_state[session_key] = defaults.copy()
        st.session_state[session_key + "_touched"] = False

def _overlay_add(session_key: str, defaults: list[date], new_date: date):
    touched_key = session_key + "_touched"
    cur = st.session_state.get(session_key, [])
    if (not st.session_state.get(touched_key, False)) or (set(cur) == set(defaults)):
        st.session_state[session_key] = []
    st.session_state[touched_key] = True
    if new_date not in st.session_state[session_key]:
        st.session_state[session_key].append(new_date)
        st.session_state[session_key] = sorted(st.session_state[session_key])

def _dt_from_date_block(d: date, block: int) -> pd.Timestamp:
    return pd.to_datetime(d) + pd.to_timedelta((int(block) - 1) * 15, unit='m')

def _nearest_available(target: date, avail: list[date]) -> date:
    if not avail:
        return target
    arr = np.array(avail, dtype="datetime64[D]")
    t = np.datetime64(target)
    idx = np.searchsorted(arr, t)
    if idx == 0:
        return avail[0]
    if idx == len(arr):
        return avail[-1]
    before = avail[idx - 1]
    after = avail[idx]
    return before if abs(t - np.datetime64(before)) <= abs(np.datetime64(after) - t) else after

def _symmetric_window(avail_dates: list[date], anchor_day: date, n: int) -> list[date]:
    if not avail_dates or n <= 0:
        return []
    if anchor_day not in avail_dates:
        anchor_day = _nearest_available(anchor_day, avail_dates)
    ai = avail_dates.index(anchor_day)
    pre = n // 2
    post = n - pre - 1
    start = max(0, ai - pre)
    end = min(len(avail_dates) - 1, ai + post)
    while (end - start + 1) < n:
        if start > 0:
            start -= 1
        elif end < len(avail_dates) - 1:
            end += 1
        else:
            break
    return avail_dates[start:end + 1]

def _nearest_or_prev(target: date, avail: list[date]) -> date:
    if not avail:
        return target
    if target in avail:
        return target
    earlier = [d for d in avail if d <= target]
    if earlier:
        return earlier[-1]
    return avail[-1]


# =========================
# COMMON DATA LOADERS (from DB)
# =========================

def _resolve_state_slug(state_slug: str | None = None) -> str:
    return state_slug or STATE_SLUG


def _normalize_weather_df(df: pd.DataFrame, keep_location: bool = False, location_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "humidity" not in df.columns:
        for c in ["relative_humidity_2m", "rh", "rel_humidity"]:
            if c in df.columns:
                df = df.rename(columns={c: "humidity"})
                break

    if "precipitation" not in df.columns:
        for c in ["rain", "precip", "precipitation_sum", "total_precipitation"]:
            if c in df.columns:
                df = df.rename(columns={c: "precipitation"})
                break

    if "date" not in df.columns:
        for c in ["datetime", "Date", "DATE"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "time_block" not in df.columns:
        for c in ["block", "timeblock", "time_Block", "TIME_BLOCK"]:
            if c in df.columns:
                df = df.rename(columns={c: "time_block"})
                break

    if "date" not in df.columns or "time_block" not in df.columns:
        return pd.DataFrame()

    if keep_location and location_col:
        if location_col != "location" and location_col in df.columns:
            df = df.rename(columns={location_col: "location"})
        if "location" in df.columns:
            df["location"] = df["location"].astype(str).str.strip()
            df = df[df["location"].ne("")]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "time_block"])
    if df.empty:
        return df

    df["Datetime"] = (
        pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d"), errors="coerce")
        + pd.to_timedelta((df["time_block"] - 1).fillna(0) * 15, unit="m")
    )
    df["date"] = df["date"].dt.date

    if "time_block" not in df.columns or df["time_block"].isna().any():
        dt = pd.to_datetime(df["Datetime"], errors="coerce")
        df["time_block"] = ((dt.dt.hour * 60 + dt.dt.minute) // 15 + 1).astype("Int64")

    subset_cols = ["Datetime", "time_block"]
    if keep_location and "location" in df.columns:
        subset_cols.append("location")
    df = df.dropna(subset=subset_cols).sort_values(subset_cols).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def _load_weather_from_db(state_slug: str | None = None) -> pd.DataFrame:
    state_slug = _resolve_state_slug(state_slug)
    table = f"{state_slug}_weather_mean"
    return _normalize_weather_df(_read_table(table), keep_location=False)


@st.cache_data(show_spinner=False, ttl=300)
def _load_weather_loc_from_db(state_slug: str | None = None) -> pd.DataFrame:
    state_slug = _resolve_state_slug(state_slug)
    table = f"{state_slug}_weather_loc"
    df = _read_table(table)
    if df.empty:
        return df

    location_col = None
    preferred = ["location", "district", "city", "station", "place", "name", "zone", "area", "subdivision"]
    for c in preferred:
        if c in df.columns:
            location_col = c
            break

    if location_col is None:
        candidate_cols = [
            c for c in df.columns
            if c not in {"date", "datetime", "Date", "DATE", "time_block", "block", "state", "STATE"}
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if candidate_cols:
            location_col = candidate_cols[0]

    if location_col is None:
        return pd.DataFrame()

    return _normalize_weather_df(df, keep_location=True, location_col=location_col)


@st.cache_data(show_spinner=False, ttl=300)
def _load_sldc_from_db(state_slug: str | None = None) -> pd.DataFrame:
    state_slug = _resolve_state_slug(state_slug)
    table = _find_first_existing_table([
        f"{state_slug}_sldc",
        f"{state_slug}_load",
        f"{state_slug}_load_mean",
        f"{state_slug}_demand",
        f"{state_slug}_demand_mean",
    ])
    if table is None:
        return pd.DataFrame()

    df = _read_table(table)
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns:
        for c in ["Date", "DATE", "datetime"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "time_block" not in df.columns:
        for c in ["block", "timeblock", "TIME_BLOCK", "Block"]:
            if c in df.columns:
                df = df.rename(columns={c: "time_block"})
                break

    if "date" not in df.columns or "time_block" not in df.columns:
        return pd.DataFrame()

    val_col = None
    for c in ["load_mw", "Load_MW", "load", "total_drawal", "demand", "actual_mw", "actual_load", "drawal_mw"]:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce").astype("Int64")
    df["load"] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=["date", "time_block", "load"])
    if df.empty:
        return df

    df = df[(df["time_block"] >= 1) & (df["time_block"] <= 96)]
    df["Datetime"] = df.apply(lambda r: _dt_from_date_block(r["date"], int(r["time_block"])), axis=1)
    df = df.dropna(subset=["Datetime", "date", "time_block", "load"]).sort_values("Datetime").reset_index(drop=True)
    return df[["Datetime", "date", "time_block", "load"]]


@st.cache_data(show_spinner=False, ttl=300)
def _load_actual_from_db(state_slug: str | None = None) -> pd.DataFrame:
    state_slug = _resolve_state_slug(state_slug)
    table = _find_first_existing_table([
        f"{state_slug}_sldc",
        f"{state_slug}_load",
        f"{state_slug}_load_mean",
        f"{state_slug}_demand",
        f"{state_slug}_demand_mean",
    ])
    if table is None:
        return pd.DataFrame()

    df = _read_table(table)
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns:
        for c in ["Date", "DATE", "datetime"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "time_block" not in df.columns:
        for c in ["block", "timeblock", "TIME_BLOCK", "Block"]:
            if c in df.columns:
                df = df.rename(columns={c: "time_block"})
                break

    if "date" not in df.columns or "time_block" not in df.columns:
        return pd.DataFrame()

    val_col = None
    for c in ["total_drawal", "load_mw", "Load_MW", "load", "demand", "actual_mw", "actual_load", "drawal_mw"]:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce").astype("Int64")
    df["total_drawal"] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=["date", "time_block", "total_drawal"])
    if df.empty:
        return df

    df = df[(df["time_block"] >= 1) & (df["time_block"] <= 96)]
    df["Datetime"] = df.apply(lambda r: _dt_from_date_block(r["date"], int(r["time_block"])), axis=1)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df[["date", "time_block", "Datetime", "total_drawal"]]


@st.cache_data(show_spinner=False, ttl=300)
def _load_forecast_from_db(state_slug: str | None = None) -> pd.DataFrame:
    state_slug = _resolve_state_slug(state_slug)
    table = _find_first_existing_table([
        f"{state_slug}_sldc_forecast",
        # f"{state_slug}_forecast",
    ])
    if table is None:
        return pd.DataFrame()

    df = _read_table(table)
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns:
        for c in ["Date", "DATE", "datetime"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    if "time_block" not in df.columns:
        for c in ["block", "timeblock", "TIME_BLOCK", "Block"]:
            if c in df.columns:
                df = df.rename(columns={c: "time_block"})
                break

    if "date" not in df.columns or "time_block" not in df.columns:
        return pd.DataFrame()

    val_col = None
    for c in ["forecast_mw", "forecasted", "forecast", "load_forecast", "demand_forecast", "predicted_load", "prediction"]:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce").astype("Int64")
    df["forecasted"] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=["date", "time_block", "forecasted"])
    if df.empty:
        return df

    df = df[(df["time_block"] >= 1) & (df["time_block"] <= 96)]
    df["Datetime"] = df.apply(lambda r: _dt_from_date_block(r["date"], int(r["time_block"])), axis=1)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df[["date", "time_block", "Datetime", "forecasted"]]


def render_combined_dashboard():
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def available_weather_vars(df: pd.DataFrame) -> list[str]:
        candidates = []
        for c in ["temperature_2m", "apparent_temperature", "temperature", "precipitation", "humidity"]:
            if c in df.columns:
                candidates.append(c)
        return candidates

    def available_dates(df: pd.DataFrame) -> list[date]:
        if df.empty:
            return []
        return sorted(pd.Series(df["date"]).dropna().unique().tolist())

    def available_locations(df: pd.DataFrame) -> list[str]:
        if df.empty or "location" not in df.columns:
            return []
        return sorted(pd.Series(df["location"]).dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())

    def make_ts_fig(df: pd.DataFrame, y_cols: list[str], start_d: date, end_d: date, title: str):
        mask = (df["date"] >= start_d) & (df["date"] <= end_d)
        cols = ["Datetime"] + y_cols
        plot_df = df.loc[mask, cols].copy()
        if plot_df.empty:
            return None
        long_df = plot_df.melt(id_vars="Datetime", value_vars=y_cols, var_name="Variable", value_name="Value")
        fig = px.line(long_df, x="Datetime", y="Value", color="Variable", title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
        return fig

    def make_overlay_by_block(df: pd.DataFrame, value_cols: list[str], days: list[date], title: str):
        if not days:
            return None
        plot_df = df[df["date"].isin(days)].copy()
        if plot_df.empty:
            return None
        plot_df["block"] = ((plot_df["Datetime"].dt.hour * 60 + plot_df["Datetime"].dt.minute) // 15) + 1
        plot_df["block"] = plot_df["block"].astype(int)
        keep_cols = ["block", "date"] + value_cols
        plot_df = plot_df[keep_cols]
        long_df = plot_df.melt(id_vars=["block", "date"], value_vars=value_cols,
                               var_name="Variable", value_name="Value")
        long_df["Series"] = long_df["Variable"] + " | " + long_df["date"].astype(str)
        fig = px.line(long_df, x="block", y="Value", color="Series", title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis=dict(dtick=4))
        return fig

    def make_overlay_load_vs_weather(
        load_df: pd.DataFrame,
        wx_df: pd.DataFrame,
        days: list[date],
        wx_var: str,
        weather_name: str | None = None,
        figure_title: str | None = None,
    ):
        if not days:
            return None
        L = load_df[load_df["date"].isin(days)].copy()
        W = wx_df[wx_df["date"].isin(days)].copy()
        if L.empty or W.empty or wx_var not in W.columns:
            return None
        L = L.drop_duplicates(subset=["date", "time_block"], keep="last")[["date", "time_block", "load"]]
        W = W.drop_duplicates(subset=["date", "time_block"], keep="last")[["date", "time_block", wx_var]]
        M = pd.merge(L, W, on=["date", "time_block"], how="inner").dropna()
        if M.empty:
            return None
        M["block"] = M["time_block"].astype(int)
        M["date_str"] = M["date"].astype(str)
        wx_series_name = weather_name or wx_var
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for d in days:
            ds = str(d)
            m = M[M["date_str"] == ds].sort_values("block")
            if m.empty:
                continue
            fig.add_trace(
                go.Scatter(x=m["block"], y=m["load"], mode="lines", name=f"Load | {ds}"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=m["block"], y=m[wx_var], mode="lines", name=f"{wx_series_name} | {ds}"),
                secondary_y=True,
            )
        fig.update_layout(
            title=figure_title or f"Overlay: Load + {wx_series_name} (dual-axis) — selected dates",
            margin=dict(l=10, r=10, t=50, b=10), height=460,
            xaxis=dict(title="Block (1–96)", dtick=4),
        )
        fig.update_yaxes(title_text="Load (MW)", secondary_y=False)
        fig.update_yaxes(title_text=wx_var, secondary_y=True)
        return fig

    # ---- Load data from DB ----
    wx = _load_weather_from_db(STATE_SLUG)
    wx_loc = _load_weather_loc_from_db(STATE_SLUG)
    ld = _load_sldc_from_db(STATE_SLUG)
    if wx.empty:
        st.warning(f"Weather data not found in DB for {STATE_DISPLAY}.")
        return
    if ld.empty:
        st.warning(f"SLDC load data not found in DB for {STATE_DISPLAY}.")
        return

    wx_avail = available_dates(wx)
    wx_loc_avail = available_dates(wx_loc)
    ld_avail = available_dates(ld)
    union_avail = sorted(set(wx_avail) | set(ld_avail) | set(wx_loc_avail))
    if not union_avail:
        st.warning("No dates available.")
        return

    today = pd.Timestamp.today().date()
    anchor = _nearest_available(today, union_avail)

    wx_vars = available_weather_vars(wx)
    if not wx_vars:
        st.warning("No weather variables found.")
        return

    default_wx_var = "temperature_2m" if "temperature_2m" in wx_vars else wx_vars[0]
    default_wx_vars = [v for v in ["temperature_2m"] if v in wx_vars] or [default_wx_var]

    wx_vars_ts = st.multiselect(
        "A) Weather variables for Time-Series",
        options=wx_vars, default=default_wx_vars, key="cmb_wx_vars_ts",
    )
    if not wx_vars_ts:
        st.warning("Select at least one weather variable for Weather Time-Series.")
        return

    min_d, max_d = union_avail[0], union_avail[-1]
    ts_days = _symmetric_window(union_avail, anchor, 6)
    ts_default_range = (ts_days[0], ts_days[-1]) if ts_days else (min_d, max_d)

    st.subheader("A) Weather Time-Series")
    ts_range = st.date_input(
        "Date range (shared for both time-series)",
        value=ts_default_range, min_value=min_d, max_value=max_d,
        format="YYYY-MM-DD", key="cmb_ts_range",
    )
    if isinstance(ts_range, tuple) and len(ts_range) == 2:
        start_d, end_d = ts_range
    else:
        start_d = ts_range if isinstance(ts_range, date) else max_d
        end_d = start_d
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    start_w = _nearest_available(start_d, wx_avail) if wx_avail else start_d
    end_w = _nearest_available(end_d, wx_avail) if wx_avail else end_d

    fig_wx_ts = make_ts_fig(wx, wx_vars_ts, start_w, end_w, title=f"Weather: {start_w} -> {end_w}")
    if fig_wx_ts is not None:
        st.plotly_chart(fig_wx_ts, width='stretch')
    else:
        st.warning("No weather rows in selected range.")

    st.markdown("---")

    st.subheader("B) Load Time-Series (SLDC)")
    start_l = _nearest_available(start_d, ld_avail) if ld_avail else start_d
    end_l = _nearest_available(end_d, ld_avail) if ld_avail else end_d
    fig_ld_ts = make_ts_fig(
        ld.rename(columns={"load": "Load_MW"}), ["Load_MW"],
        start_l, end_l, title=f"Load: {start_l} -> {end_l}"
    )
    if fig_ld_ts is not None:
        st.plotly_chart(fig_ld_ts, width='stretch')
    else:
        st.warning("No load rows in selected range.")

    st.markdown("---")

    overlay_pool = sorted(set(ld_avail).intersection(set(wx_avail) | set(wx_loc_avail)))
    if not overlay_pool:
        overlay_pool = sorted(set(wx_avail).intersection(set(ld_avail))) or union_avail
    if not overlay_pool:
        st.warning("No dates available for overlays.")
        return

    overlay_defaults = _symmetric_window(
        overlay_pool,
        _nearest_or_prev(today - timedelta(days=1), overlay_pool),
        DEFAULT_OVERLAY_N,
    )
    _overlay_init("cmb_overlay_days", overlay_defaults)

    st.subheader("Overlay Date Picker (used for C/D/E/F)")
    c_ov1, c_ov2, c_ov3 = st.columns([1.2, 0.6, 0.4])
    with c_ov1:
        add_overlay_date = st.date_input(
            "Add comparison date",
            value=_nearest_or_prev(today - timedelta(days=1), overlay_pool),
            min_value=overlay_pool[0], max_value=overlay_pool[-1],
            format="YYYY-MM-DD", key="cmb_overlay_add_date",
        )
    with c_ov2:
        if st.button("Add date", key="cmb_overlay_add_btn"):
            d = _nearest_or_prev(add_overlay_date, overlay_pool)
            _overlay_add("cmb_overlay_days", overlay_defaults, d)
    with c_ov3:
        if st.button("Clear", key="cmb_overlay_clear_btn"):
            st.session_state["cmb_overlay_days"] = []
            st.session_state["cmb_overlay_days_touched"] = True

    days = st.session_state["cmb_overlay_days"]
    if days:
        st.caption("Comparing: " + ", ".join([d.strftime("%Y-%m-%d") for d in days]))

    st.markdown("---")

    st.subheader("C) Overlay of Load (blocks 1-96)")
    fig_ld_ov = make_overlay_by_block(
        ld.rename(columns={"load": "Load_MW"}), ["Load_MW"], days,
        title="Daily Overlay - Load (SLDC)"
    )
    if fig_ld_ov is not None:
        st.plotly_chart(fig_ld_ov, width='stretch')
    else:
        st.warning("No load overlay rows for selected dates.")

    st.markdown("---")

    st.subheader("D) Overlay of Weather Variables (blocks 1-96)")
    wx_vars_overlay = st.multiselect(
        "D) Weather variables for Weather Overlay",
        options=wx_vars, default=default_wx_vars, key="cmb_wx_vars_overlay",
    )
    if not wx_vars_overlay:
        st.warning("Select at least one weather variable for Weather Overlay.")
    else:
        fig_wx_ov = make_overlay_by_block(wx, wx_vars_overlay, days, title="Daily Overlay - Weather")
        if fig_wx_ov is not None:
            st.plotly_chart(fig_wx_ov, width='stretch')
        else:
            st.warning("No weather overlay rows for selected dates.")

    st.markdown("---")

    st.subheader("E) Overlay: Load + Weather on same plot (dual-axis)")
    wx_var_for_joint = st.selectbox(
        "E) Select weather variable to overlay with Load",
        options=wx_vars,
        index=wx_vars.index(default_wx_var) if default_wx_var in wx_vars else 0,
        key="cmb_joint_wx_var",
    )
    fig_joint = make_overlay_load_vs_weather(ld, wx, days, wx_var_for_joint)
    if fig_joint is not None:
        st.plotly_chart(fig_joint, width='stretch')
    else:
        st.warning("Joint overlay needs matching (date,time_block) rows in both datasets for selected dates.")

    st.markdown("---")

    st.subheader("F) Overlay: Load + Location Weather on same plot (dual-axis)")
    if wx_loc.empty:
        st.warning(f"Location-level weather data not found in DB for {STATE_DISPLAY}.")
        return

    loc_options = available_locations(wx_loc)
    loc_wx_vars = available_weather_vars(wx_loc)

    if not loc_options:
        st.warning("No locations found in the state weather location table.")
        return
    if not loc_wx_vars:
        st.warning("No weather variables found in the state weather location table.")
        return

    loc_default = loc_options[0]
    loc_default_var = "temperature_2m" if "temperature_2m" in loc_wx_vars else loc_wx_vars[0]
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        selected_location = st.selectbox(
            "F) Select location present in this state",
            options=loc_options,
            index=0,
            key="cmb_joint_loc_name",
        )
    with c_f2:
        selected_loc_wx_var = st.selectbox(
            "F) Select location weather variable to overlay with Load",
            options=loc_wx_vars,
            index=loc_wx_vars.index(loc_default_var) if loc_default_var in loc_wx_vars else 0,
            key="cmb_joint_loc_wx_var",
        )

    wx_loc_filtered = wx_loc[wx_loc["location"] == selected_location].copy()
    fig_joint_loc = make_overlay_load_vs_weather(
        ld,
        wx_loc_filtered,
        days,
        selected_loc_wx_var,
        weather_name=f"{selected_location} — {selected_loc_wx_var}",
        figure_title=f"Overlay: Load + Location Weather ({selected_location}) (dual-axis) — selected dates",
    )
    if fig_joint_loc is not None:
        st.plotly_chart(fig_joint_loc, width='stretch')
    else:
        st.warning(
            "Location weather overlay needs matching (date,time_block) rows in both load and selected location weather data for the chosen dates."
        )


# =========================
# WEATHER DASHBOARD
# =========================
def render_weather_dashboard():
    import plotly.express as px

    def available_dates(df: pd.DataFrame) -> list[date]:
        if df.empty:
            return []
        return sorted(pd.Series(df["date"]).dropna().unique().tolist())

    def available_weather_vars(df: pd.DataFrame) -> list[str]:
        candidates = []
        for c in ["temperature_2m", "apparent_temperature", "temperature", "precipitation", "humidity"]:
            if c in df.columns:
                candidates.append(c)
        return candidates

    def make_time_series_fig_multi(df: pd.DataFrame, y_cols: list[str], start_d: date, end_d: date):
        mask = (df["date"] >= start_d) & (df["date"] <= end_d)
        plot_df = df.loc[mask, ["Datetime"] + y_cols].copy()
        if plot_df.empty:
            return None
        long_df = plot_df.melt(id_vars="Datetime", value_vars=y_cols, var_name="Variable", value_name="Value")
        fig = px.line(long_df, x="Datetime", y="Value", color="Variable",
                      title=f"Time-series ({start_d} -> {end_d})")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        return fig

    def make_overlay_fig_multi(df: pd.DataFrame, y_cols: list[str], days: list[date]):
        if not days:
            return None
        plot_df = df[df["date"].isin(days)].copy()
        if plot_df.empty:
            return None
        plot_df["block"] = ((plot_df["Datetime"].dt.hour * 60 + plot_df["Datetime"].dt.minute) // 15) + 1
        plot_df["block"] = plot_df["block"].astype(int)
        keep_cols = ["block", "date"] + y_cols
        plot_df = plot_df[keep_cols]
        long_df = plot_df.melt(id_vars=["block", "date"], value_vars=y_cols,
                               var_name="Variable", value_name="Value")
        long_df["Series"] = long_df["Variable"] + " | " + long_df["date"].astype(str)
        fig = px.line(long_df, x="block", y="Value", color="Series",
                      title="Daily Overlay by 15-min Blocks")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis=dict(dtick=4))
        return fig

    def minmax_for_variable(day_df: pd.DataFrame, col: str):
        if col not in day_df.columns or day_df[col].dropna().empty:
            return None
        idxmin = day_df[col].idxmin()
        idxmax = day_df[col].idxmax()
        rmin = day_df.loc[idxmin]; rmax = day_df.loc[idxmax]
        tmin = pd.to_datetime(rmin["Datetime"]); tmax = pd.to_datetime(rmax["Datetime"])
        return (
            (float(rmin[col]), tmin.strftime("%H:%M"), int(rmin["time_block"])),
            (float(rmax[col]), tmax.strftime("%H:%M"), int(rmax["time_block"]))
        )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Weather Controls")
        st.caption("Pick variables to plot (supports multiple).")

    df = _load_weather_from_db()
    if df.empty:
        st.warning(f"Weather data not found in DB for {STATE_DISPLAY}.")
        st.stop()

    avail = available_dates(df)
    min_d, max_d = (avail[0], avail[-1]) if avail else (date.today(), date.today())
    today = pd.Timestamp.today().date()
    anchor = _nearest_available(today, avail) if avail else today

    vars_available = available_weather_vars(df)
    default_vars = [v for v in ["temperature_2m"] if v in vars_available] or vars_available[:1]
    var_multi = st.multiselect("Variables", options=vars_available, default=default_vars, key="wx_vars_multi")
    if not var_multi:
        st.warning("Select at least one variable.")
        st.stop()

    ts_days = _symmetric_window(avail, anchor, DEFAULT_TS_DAYS) if avail else []
    ts_default_range = (ts_days[0], ts_days[-1]) if ts_days else (min_d, max_d)
    overlay_default_days = _symmetric_window(avail, anchor, DEFAULT_OVERLAY_N) if avail else []
    summary_default_date = anchor

    # Time-Series
    st.subheader("Time-Series")
    ts_range = st.date_input(
        "Date range", value=ts_default_range,
        min_value=min_d, max_value=max_d,
        format="YYYY-MM-DD", key="wx_ts_range",
    )
    if isinstance(ts_range, tuple) and len(ts_range) == 2:
        start_d, end_d = ts_range
    else:
        start_d = ts_range if isinstance(ts_range, date) else max_d
        end_d = start_d

    if start_d not in avail:
        start_d = _nearest_available(start_d, avail)
    if end_d not in avail:
        end_d = _nearest_available(end_d, avail)
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    fig_ts = make_time_series_fig_multi(df, var_multi, start_d, end_d)
    if fig_ts is not None:
        st.plotly_chart(fig_ts, width='stretch')
    else:
        st.warning("No rows in the selected range.")
    st.markdown("---")

    # Overlay days
    st.subheader("Overlay Days (1-96 blocks)")
    _overlay_init("overlay_days", overlay_default_days)

    c_ov1, c_ov2, c_ov3 = st.columns([1.2, 0.6, 0.4])
    with c_ov1:
        add_overlay_date = st.date_input(
            "Add comparison date", value=anchor,
            min_value=min_d, max_value=max_d,
            format="YYYY-MM-DD", key="wx_overlay_add_date",
        )
    with c_ov2:
        if st.button("Add date", key="wx_add_btn"):
            d = _nearest_available(add_overlay_date, avail)
            _overlay_add("overlay_days", overlay_default_days, d)
    with c_ov3:
        if st.button("Clear", key="wx_clear_btn"):
            st.session_state["overlay_days"] = []
            st.session_state["overlay_days_touched"] = True

    days_for_plot = st.session_state["overlay_days"]
    if days_for_plot:
        st.caption("Comparing: " + ", ".join([d.strftime("%Y-%m-%d") for d in days_for_plot]))

    fig_overlay = make_overlay_fig_multi(df, var_multi, st.session_state["overlay_days"])
    if fig_overlay is not None:
        st.plotly_chart(fig_overlay, width='stretch')
    else:
        st.warning("Pick at least one date in data.")
    st.markdown("---")

    # Daily Stats & Extremes
    st.subheader("Daily Stats & Extremes (multiple dates)")
    if "stats_dates" not in st.session_state:
        st.session_state.stats_dates = [summary_default_date]

    c_sd1, c_sd2, c_sd3 = st.columns([1.2, 0.6, 0.4])
    with c_sd1:
        add_stats_date = st.date_input(
            "Add date for stats", value=summary_default_date,
            min_value=min_d, max_value=max_d,
            format="YYYY-MM-DD", key="wx_stats_add_date",
        )
    with c_sd2:
        if st.button("Add stats date", key="wx_stats_add_btn"):
            d = _nearest_available(add_stats_date, avail)
            if d not in st.session_state.stats_dates:
                st.session_state.stats_dates.append(d)
                st.session_state.stats_dates = sorted(st.session_state.stats_dates)
    with c_sd3:
        if st.button("Clear stats dates", key="wx_stats_clear_btn"):
            st.session_state.stats_dates = [summary_default_date]

    unit_lookup = {
        "temperature_2m": "C", "apparent_temperature": "C",
        "temperature": "C", "precipitation": "mm", "humidity": "%"
    }

    for d in st.session_state.stats_dates:
        day_df = df[df["date"] == d]
        st.markdown(f"**{d}**")
        if day_df.empty:
            st.warning(f"No data for {d}.")
            st.divider()
            continue
        for col in var_multi:
            if col not in day_df.columns or day_df[col].dropna().empty:
                st.warning(f"No {col} data for {d}."); st.divider(); continue
            unit = unit_lookup.get(col, "")
            res = minmax_for_variable(day_df, col)
            if res is not None:
                (vmin, tmin, bmin), (vmax, tmax, bmax) = res
            else:
                vmin = vmax = np.nan; tmin = tmax = "--"; bmin = bmax = 0
            c_avg, c_min, c_max = st.columns(3)
            with c_avg:
                if col == "precipitation":
                    total = float(day_df[col].sum())
                    st.metric(f"Total {col}", f"{total:.2f} {unit}")
                else:
                    avg_val = float(day_df[col].mean())
                    st.metric(f"Avg {col}", f"{avg_val:.2f} {unit}")
            with c_min:
                st.metric(f"Min {col}", f"{(vmin if pd.notna(vmin) else 0):.2f} {unit}")
                st.caption(f"Min at **{tmin}** (Block **{bmin-1}**)")
            with c_max:
                st.metric(f"Max {col}", f"{(vmax if pd.notna(vmax) else 0):.2f} {unit}")
                st.caption(f"Max at **{tmax}** (Block **{bmax-1}**)")
            st.divider()


# =========================
# LOAD DASHBOARD (Actual + Forecast)
# =========================
def render_load_dashboard():
    import plotly.express as px

    def available_dates(df: pd.DataFrame) -> list[date]:
        if df.empty:
            return []
        return sorted(pd.Series(df["date"]).dropna().unique().tolist())

    def complete_dates_96(df: pd.DataFrame, val_col: str) -> set[date]:
        if df.empty:
            return set()
        g = (df.dropna(subset=[val_col])
               .drop_duplicates(subset=["date", "time_block"])
               .groupby("date")["time_block"].nunique())
        return set(g[g == 96].index.tolist())

    act = _load_actual_from_db(STATE_SLUG)
    if act.empty:
        st.warning(f"Load (actual) data not found in DB for {STATE_DISPLAY}.")
        st.stop()
    fc = _load_forecast_from_db(STATE_SLUG)
    have_fc = not fc.empty

    avail_a = available_dates(act)
    min_a, max_a = (avail_a[0], avail_a[-1]) if avail_a else (date.today(), date.today())
    avail_f = available_dates(fc) if have_fc else []
    avail_ts = sorted(set(avail_a) | set(avail_f))
    min_ts, max_ts = (avail_ts[0], avail_ts[-1]) if avail_ts else (date.today(), date.today())

    anchor_ts = max_ts
    ts_days = _symmetric_window(avail_ts, anchor_ts, DEFAULT_LOAD_TS_DAYS) if avail_ts else []
    ts_default_range = (ts_days[0], ts_days[-1]) if ts_days else (min_ts, max_ts)

    st.subheader("Time-Series — Actual vs Forecast")
    ts_range = st.date_input(
        "Date range", value=ts_default_range,
        min_value=min_ts, max_value=max_ts,
        format="YYYY-MM-DD", key="ld_ts_range",
    )
    if isinstance(ts_range, tuple) and len(ts_range) == 2:
        start_d, end_d = ts_range
    else:
        start_d = ts_range if isinstance(ts_range, date) else max_ts
        end_d = start_d
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    maskA = (act["date"] >= start_d) & (act["date"] <= end_d)
    a_plot = act.loc[maskA, ["Datetime", "total_drawal"]].copy()
    a_plot["Series"] = "Actual"

    if have_fc:
        maskF = (fc["date"] >= start_d) & (fc["date"] <= end_d)
        f_plot = fc.loc[maskF, ["Datetime", "forecasted"]].rename(columns={"forecasted": "total_drawal"})
        f_plot["Series"] = "Forecast"
        plot_df = pd.concat(
            [a_plot.rename(columns={"total_drawal": "value"}),
             f_plot.rename(columns={"total_drawal": "value"})],
            ignore_index=True
        )
    else:
        plot_df = a_plot.rename(columns={"total_drawal": "value"})

    if plot_df.empty:
        st.warning("No rows in the selected range.")
    else:
        fig = px.line(
            plot_df, x="Datetime", y="value", color="Series",
            labels={"Datetime": "Datetime", "value": "Load (MW)"},
            title=f"Actual{' vs Forecast' if have_fc else ''}: {start_d} -> {end_d}",
        )
        tick_start = pd.Timestamp(start_d)
        last_midday = pd.Timestamp(end_d) + pd.Timedelta(hours=12)
        ticks = pd.date_range(tick_start, last_midday, freq="12h")
        ticktexts = []
        for t in ticks:
            if t.hour == 0:
                day_no_leading_zero = t.strftime("%d").lstrip("0")
                date_label = f"{t.strftime('%b')} {day_no_leading_zero}, {t.strftime('%Y')}"
                ticktexts.append(f"{t.strftime('%H:%M')}<br>{date_label}")
            else:
                ticktexts.append(f"{t.strftime('%H:%M')}<br>\u00A0")
        fig.update_xaxes(tickvals=ticks, ticktext=ticktexts)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=440)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ---------- Combined Overlay (Actual + Forecast) ----------
    st.subheader("Daily Overlay by 15-min Blocks (Load)")
    overlay_mode = st.radio(
        "Overlay source",
        ["Actual", "Forecast", "Actual + Forecast"],
        index=0, horizontal=True, key="ld_overlay_source",
    )
    if overlay_mode in ["Forecast", "Actual + Forecast"] and not have_fc:
        st.info("Forecast data not found — switching overlay to Actual.")
        overlay_mode = "Actual"

    if overlay_mode == "Actual":
        src_df = act.rename(columns={"total_drawal": "value"}).copy()
        s_avail = available_dates(src_df)
    elif overlay_mode == "Forecast":
        src_df = fc.rename(columns={"forecasted": "value"}).copy()
        s_avail = available_dates(src_df)
    else:
        s_avail = sorted(set(available_dates(act)) | set(available_dates(fc)))

    if not s_avail:
        st.warning("No data found for overlay.")
    else:
        yesterday = pd.Timestamp.today().date() - timedelta(days=1)
        anchor_overlay = _nearest_or_prev(yesterday, s_avail)

        ov_key = f"ld_overlay_days_{overlay_mode.replace(' ','_').lower()}"
        overlay_defaults = _symmetric_window(s_avail, anchor_overlay, DEFAULT_OVERLAY_N)
        _overlay_init(ov_key, overlay_defaults)

        c_ov1, c_ov2, c_ov3 = st.columns([1.2, 0.6, 0.8])
        with c_ov1:
            add_overlay_date = st.date_input(
                f"Add comparison date ({overlay_mode})",
                value=anchor_overlay, min_value=s_avail[0], max_value=s_avail[-1],
                format="YYYY-MM-DD", key=f"ld_overlay_add_{overlay_mode.replace(' ','_').lower()}",
            )
        with c_ov2:
            if st.button("Add date", key=f"ld_overlay_add_btn_{overlay_mode.replace(' ','_').lower()}"):
                d = _nearest_or_prev(add_overlay_date, s_avail)
                _overlay_add(ov_key, overlay_defaults, d)
        with c_ov3:
            st.markdown("**Options**")
            show_actual = st.checkbox("Show Actual", value=True, key="ov_show_actual")
            show_forecast = st.checkbox("Show Forecast", value=(overlay_mode != "Actual"), key="ov_show_forecast")
            st.caption("Toggle series visibility for selected dates.")
            if st.button("Clear dates", key=f"ld_overlay_clear_btn_{overlay_mode.replace(' ','_').lower()}"):
                st.session_state[ov_key] = []
                st.session_state[ov_key + "_touched"] = True

        days = st.session_state[ov_key]
        if days:
            st.caption("Comparing: " + ", ".join([d.strftime("%Y-%m-%d") for d in days]))

        if overlay_mode in ["Actual", "Forecast"]:
            plot_df = src_df[src_df["date"].isin(days)].copy()
            if plot_df.empty:
                st.warning("No rows for the selected dates.")
            else:
                plot_df["block"] = ((plot_df["Datetime"].dt.hour * 60 + plot_df["Datetime"].dt.minute) // 15) + 1
                plot_df["block"] = plot_df["block"].astype(int)
                fig_overlay = px.line(
                    plot_df, x="block", y="value",
                    color=plot_df["date"].astype(str), markers=False,
                    title=f"Daily Overlay — {overlay_mode}",
                    labels={"block": "Block (1-96)", "value": "Load (MW)", "color": "Date"},
                )
                fig_overlay.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis=dict(dtick=4))
                st.plotly_chart(fig_overlay, width='stretch')
        else:
            if not days:
                st.warning("Pick at least one date.")
            else:
                parts = []
                if show_actual:
                    aa = act[act["date"].isin(days)].copy()
                    aa["value"] = aa["total_drawal"]; aa["Source"] = "Actual"
                    parts.append(aa[["date", "time_block", "Datetime", "value", "Source"]])
                if show_forecast:
                    ff = fc[fc["date"].isin(days)].copy()
                    ff["value"] = ff["forecasted"]; ff["Source"] = "Forecast"
                    parts.append(ff[["date", "time_block", "Datetime", "value", "Source"]])
                if not parts:
                    st.warning("Enable at least one of Actual or Forecast.")
                else:
                    J = pd.concat(parts, ignore_index=True)
                    J["block"] = ((J["Datetime"].dt.hour * 60 + J["Datetime"].dt.minute) // 15) + 1
                    J["block"] = J["block"].astype(int)
                    J["Series"] = J["Source"] + " | " + J["date"].astype(str)
                    fig_overlay = px.line(
                        J, x="block", y="value", color="Series",
                        title="Daily Overlay — Actual + Forecast",
                        labels={"block": "Block (1-96)", "value": "Load (MW)", "Series": "Series"},
                    )
                    fig_overlay.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis=dict(dtick=4))
                    st.plotly_chart(fig_overlay, width='stretch')

    st.markdown("---")

    # ---------- Daily Summary ----------
    st.subheader("Daily Summary (multiple dates)")
    summary_source = st.radio(
        "Summary source", ["Actual", "Forecast"],
        index=0, horizontal=True, key="ld_summary_source",
    )
    if summary_source == "Forecast" and (not have_fc or fc.empty):
        st.info("Forecast data not found or empty — switched to Actual.")
        summary_source = "Actual"

    if summary_source == "Actual":
        s_avail = available_dates(act)
        min_s, max_s = (s_avail[0], s_avail[-1]) if s_avail else (date.today(), date.today())
        anchor_summary = _nearest_or_prev(pd.Timestamp.today().date() - timedelta(days=1), s_avail) if s_avail else date.today()
        value_col = "total_drawal"
        src = act
    else:
        s_avail = available_dates(fc)
        min_s, max_s = (s_avail[0], s_avail[-1]) if s_avail else (date.today(), date.today())
        anchor_summary = _nearest_or_prev(pd.Timestamp.today().date() - timedelta(days=1), s_avail) if s_avail else date.today()
        value_col = "forecasted"
        src = fc

    dates_key = f"load_stats_dates_{summary_source.lower()}"
    if dates_key not in st.session_state or not st.session_state[dates_key]:
        st.session_state[dates_key] = [anchor_summary]

    c1, c2, c3 = st.columns([1.2, 0.6, 0.4])
    with c1:
        add_sum_date = st.date_input(
            f"Add date for summary ({summary_source})",
            value=anchor_summary, min_value=min_s, max_value=max_s,
            key=f"ld_stats_add_date_{summary_source.lower()}", format="YYYY-MM-DD",
        )
    with c2:
        if st.button("Add summary date", key=f"ld_stats_add_btn_{summary_source.lower()}"):
            d = _nearest_or_prev(add_sum_date, s_avail)
            if d not in st.session_state[dates_key]:
                st.session_state[dates_key].append(d)
                st.session_state[dates_key] = sorted(st.session_state[dates_key])
    with c3:
        if st.button("Clear summary dates", key=f"ld_stats_clear_btn_{summary_source.lower()}"):
            st.session_state[dates_key] = [anchor_summary]

    def _peak_in_window(df: pd.DataFrame, start_b: int, end_b: int, vcol: str):
        win = df[(df["time_block"] >= start_b) & (df["time_block"] <= end_b)]
        if win.empty:
            return (np.nan, "--:--", 0)
        i = win[vcol].idxmax()
        r = win.loc[i]
        t = pd.to_datetime(r["Datetime"]).strftime("%H:%M")
        return (float(r[vcol]), t, int(r["time_block"]))

    for d in st.session_state[dates_key]:
        day = src[src["date"] == d].copy()
        st.markdown(f"**{d}**")
        if day.empty:
            st.warning(f"No {summary_source.lower()} data for {d}")
            st.divider()
            continue

        if "Datetime" not in day.columns:
            day["Datetime"] = day.apply(lambda r: _dt_from_date_block(r["date"], int(r["time_block"])), axis=1)

        m_val, m_time, m_b = _peak_in_window(day, *MORNING_BLOCKS, vcol=value_col)
        s_val, s_time, s_b = _peak_in_window(day, *SOLAR_BLOCKS,   vcol=value_col)
        e_val, e_time, e_b = _peak_in_window(day, *EVENING_BLOCKS, vcol=value_col)
        avg_val = float(day[value_col].mean())

        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(f"Morning Peak (MW) — {summary_source}", f"{(0 if pd.isna(m_val) else m_val):,.2f}")
            st.caption(f"{MORNING_BLOCKS[0]}-{MORNING_BLOCKS[1]} at **{m_time}** (Block {max(m_b-1,0)})")
            st.metric(f"Solar-Hour Peak (MW) — {summary_source}", f"{(0 if pd.isna(s_val) else s_val):,.2f}")
            st.caption(f"{SOLAR_BLOCKS[0]}-{SOLAR_BLOCKS[1]} at **{s_time}** (Block {max(s_b-1,0)})")
        with cc2:
            st.metric(f"Evening Peak (MW) — {summary_source}", f"{(0 if pd.isna(e_val) else e_val):,.2f}")
            st.caption(f"{EVENING_BLOCKS[0]}-{EVENING_BLOCKS[1]} at **{e_time}** (Block {max(e_b-1,0)})")
            st.metric(f"Average Load (MW) — {summary_source}", f"{avg_val:,.2f}")
        st.divider()

    st.markdown("---")

    # ---------- Forecast Error ----------
    st.subheader("Forecast Error — dates with full 96 blocks (Actual x Forecast)")

    if not have_fc:
        st.info("Forecast data not found — metrics need forecast data to enable this section.")
        return

    complete_a = complete_dates_96(act, "total_drawal")
    complete_f = complete_dates_96(fc, "forecasted")
    common_complete = sorted(complete_a.intersection(complete_f))
    if not common_complete:
        st.info("No date where both Actual and Forecast have all 96 blocks.")
        return

    min_cc, max_cc = common_complete[0], common_complete[-1]
    yesterday = pd.Timestamp.today().date() - timedelta(days=1)
    default_err_date = _nearest_or_prev(yesterday, common_complete)

    if "error_dates" not in st.session_state or not st.session_state.error_dates:
        st.session_state.error_dates = [default_err_date]

    c1, c2, c3 = st.columns([1.2, 0.6, 0.4])
    with c1:
        pick_date = st.date_input(
            "Add date for error metrics (complete dates only)",
            value=default_err_date, min_value=min_cc, max_value=max_cc,
            key="ld_err_add_date", format="YYYY-MM-DD",
        )
    with c2:
        if st.button("Add error date", key="ld_err_add_btn"):
            d = _nearest_or_prev(pick_date, common_complete)
            if d not in common_complete:
                st.info(f"{pick_date} is not a complete date; snapped to nearest available: {d}")
            if d not in st.session_state.error_dates:
                st.session_state.error_dates.append(d)
                st.session_state.error_dates = sorted(st.session_state.error_dates)
    with c3:
        if st.button("Clear error dates", key="ld_err_clear_btn"):
            st.session_state.error_dates = [default_err_date]

    if st.session_state.error_dates:
        st.caption("Evaluating: " + ", ".join([str(d) for d in st.session_state.error_dates]))

    rows = []
    for d in st.session_state.error_dates:
        a_day = (act[act["date"] == d]
                 .drop_duplicates(subset=["time_block"], keep="last")
                 .loc[:, ["time_block", "total_drawal"]])
        f_day = (fc[fc["date"] == d]
                 .drop_duplicates(subset=["time_block"], keep="last")
                 .loc[:, ["time_block", "forecasted"]])
        merged = pd.merge(a_day, f_day, on="time_block", how="inner").sort_values("time_block")
        n_blocks = len(merged)

        if n_blocks == 0:
            rows.append({"date": d, "blocks": 0, "MAPE_%": np.nan, "RMSE_MW": np.nan, "MAD_MW": np.nan, "Count_|e|>500": 0})
            continue

        err = merged["forecasted"] - merged["total_drawal"]
        abs_err = err.abs()

        mask_nonzero = merged["total_drawal"] != 0
        mape = (abs_err[mask_nonzero] / merged.loc[mask_nonzero, "total_drawal"]).mean() * 100.0 if mask_nonzero.any() else np.nan
        rmse = np.sqrt((err ** 2).mean())
        mad  = abs_err.mean()
        over_500 = int((abs_err > 500).sum())

        rows.append({
            "date": d, "blocks": n_blocks,
            "MAPE_%": round(0 if pd.isna(mape) else float(mape), 3),
            "RMSE_MW": round(float(rmse), 2),
            "MAD_MW": round(float(mad), 2),
            "Count_|e|>500": over_500
        })

    metrics_df = pd.DataFrame(rows).sort_values("date")
    st.dataframe(metrics_df, width="stretch")

    if len(st.session_state.error_dates) == 1:
        rec = metrics_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("MAPE (|e|/actual x100)", f"{rec['MAPE_%']:.2f}%")
        with c2: st.metric("RMSE (MW)", f"{rec['RMSE_MW']:.2f}")
        with c3: st.metric("MAD (MW)", f"{rec['MAD_MW']:.2f}")
        with c4: st.metric("Count |Error| > 500", f"{int(rec['Count_|e|>500'])}")

    st.caption("MAPE uses absolute percent error; RMSE uses squared error; MAD uses absolute error. Computed cumulatively per selected date (96 blocks).")


# =========================
# SIMILARITY DASHBOARD — DATA LOADERS (module-level for cache)
# =========================

@st.cache_data(show_spinner=False)
def _sim_load_weather_data(state: str) -> pd.DataFrame:
    df = _read_api_endpoint("weather_mean", state, limit=API_READ_LIMIT)
    if df.empty:
        raise ValueError(f"No weather_mean data found for {state}.")

    df.columns = df.columns.str.strip()
    date_col = next((c for c in ["datetime", "date"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("No 'datetime' or 'date' column in weather table.")
    df["date"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    block_col = next((c for c in ["block", "time_block"] if c in df.columns), None)
    if block_col is None:
        raise ValueError("No 'block' or 'time_block' column in weather table.")
    df.rename(columns={block_col: "time_block"}, inplace=True)
    df = df.dropna(subset=["date", "time_block"])
    df.sort_values(["date", "time_block"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def _sim_load_load_data(state: str):
    df = _read_api_endpoint("sldc", state, limit=API_READ_LIMIT)
    if df.empty:
        df = _read_api_endpoint("load", state, limit=API_READ_LIMIT)
    if df.empty:
        raise ValueError(f"No load/sldc data found for {state}.")

    df.columns = df.columns.str.strip()
    date_col = next((c for c in ["datetime", "date"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("No date column found in load table.")
    df["date"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True, errors="coerce")
    block_col = next((c for c in ["block", "time_block"] if c in df.columns), None)
    if block_col is None:
        raise ValueError("No time_block/block column found in load table.")
    df.rename(columns={block_col: "time_block"}, inplace=True)
    skip = {"date", "time_block", date_col, block_col, "datetime", "state", "id"}
    load_cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not load_cols:
        raise ValueError("No numeric load/demand column found.")
    df = df.dropna(subset=["date", "time_block"])
    df.sort_values(["date", "time_block"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, load_cols


# =========================
# SIMILARITY DASHBOARD
# =========================

def render_similarity_dashboard():
    import calendar as _cal
    from scipy.spatial.distance import euclidean
    from dtw import dtw as _dtw_calc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    KEY_FEATURES   = ["temperature_2m", "relative_humidity_2m", "precipitation"]
    FEATURE_LABELS = ["Temperature (°C)", "Humidity (%)", "Precipitation (mm)"]
    CATEGORY_LABELS = {"working": "📅 Working", "sunday": "🟡 Sunday", "holiday": "🎉 Holiday"}
    PALETTE = [
        "#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
        "#06B6D4", "#E11D48", "#7C3AED", "#059669", "#D97706",
        "#4F46E5", "#DB2777", "#0D9488", "#EA580C", "#9333EA",
    ]

    with st.sidebar:
        st.divider()
        st.subheader("Similarity Settings")
        method = st.selectbox(
            "Distance method",
            ["Euclidean", "Dynamic Time Warping (DTW)", "Weighted Euclidean"],
            key="sim_method",
        )
        top_k = st.slider("Top K similar days", min_value=1, max_value=20, value=5, key="sim_top_k")
        if method == "Weighted Euclidean":
            st.markdown("**Feature weights**")
            w_temp = st.slider("Temperature weight",   0.1, 5.0, 2.0, 0.1, key="sim_w_temp")
            w_hum  = st.slider("Humidity weight",      0.1, 5.0, 1.0, 0.1, key="sim_w_hum")
            w_prec = st.slider("Precipitation weight", 0.1, 5.0, 1.5, 0.1, key="sim_w_prec")
            feature_weights = np.array([w_temp, w_hum, w_prec])
        else:
            feature_weights = None

    st.subheader("🌤️⚡ Similar Weather Day Finder")
    st.caption("Find historically similar weather days and compare electricity load profiles.")

    # ── Helper functions ───────────────────────────────────────────────────────
    def _categorise_dates(dates, holiday_set):
        cat = {}
        for d in dates:
            dt = d if isinstance(d, datetime) else pd.Timestamp(d)
            if dt.date() in holiday_set:
                cat[d] = "holiday"
            elif dt.weekday() == 6:
                cat[d] = "sunday"
            else:
                cat[d] = "working"
        return cat

    def _build_profiles(df, cat_map):
        profiles = {}
        for dt, grp in df.groupby("date"):
            grp = grp.sort_values("time_block")
            profiles[dt] = {
                "profile":  grp[KEY_FEATURES].values.astype(float),
                "category": cat_map.get(dt, "working"),
            }
        return profiles

    def _compute_stats(df):
        return {f: {"mean": df[f].mean(), "std": df[f].std()} for f in KEY_FEATURES}

    def _normalise(profile, stats):
        normed = profile.copy()
        for i, f in enumerate(KEY_FEATURES):
            s = stats[f]["std"]
            normed[:, i] = (normed[:, i] - stats[f]["mean"]) / s if s else 0.0
        return normed

    def _calc_distance(a, b, meth, weights=None):
        if meth == "Euclidean":
            mn = min(len(a), len(b))
            return euclidean(a[:mn].flatten(), b[:mn].flatten())
        elif meth == "Dynamic Time Warping (DTW)":
            return _dtw_calc(a, b, dist_method="euclidean").distance
        else:
            mn = min(len(a), len(b))
            diff = (a[:mn] - b[:mn]) ** 2 * weights[np.newaxis, :]
            return float(np.sqrt(diff.sum()))

    def _find_similar(target_date, profiles, stats, meth, weights, k):
        t_info = profiles[target_date]
        t_norm = _normalise(t_info["profile"], stats)
        t_cat  = t_info["category"]
        t_month = target_date.month
        valid_months = {(t_month - 2) % 12 + 1, t_month, t_month % 12 + 1}
        results = []
        for dt, info in profiles.items():
            if dt == target_date or info["category"] != t_cat or dt.month not in valid_months:
                continue
            dist = _calc_distance(t_norm, _normalise(info["profile"], stats), meth, weights)
            results.append((dt, dist))
        results.sort(key=lambda x: x[1])
        return results[:k]

    def _plot_multi_overlay(date_list, profiles, target_date):
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=FEATURE_LABELS,
        )
        for idx, dt in enumerate(date_list):
            prof   = profiles[dt]["profile"]
            blocks = np.arange(1, len(prof) + 1)
            color  = PALETTE[idx % len(PALETTE)]
            is_tgt = (dt == target_date)
            label  = f"⭐ {dt.strftime('%d-%b-%Y (%a)')}" if is_tgt else dt.strftime("%d-%b-%Y (%a)")
            for feat_i in range(3):
                fig.add_trace(
                    go.Scatter(
                        x=blocks, y=prof[:, feat_i], mode="lines",
                        name=label,
                        line=dict(color=color, width=3 if is_tgt else 2, dash="solid" if is_tgt else "dot"),
                        legendgroup=label, showlegend=(feat_i == 0),
                    ),
                    row=feat_i + 1, col=1,
                )
        fig.update_layout(
            height=750, template="plotly_white",
            xaxis3_title="Time Block (15-min interval)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        return fig

    def _build_feature_summary(target_date, results, profiles):
        all_dates = [target_date] + [r[0] for r in results]
        tables = {}
        for feat_i, feat_name in enumerate(FEATURE_LABELS):
            rows = []
            for dt in all_dates:
                prof = profiles[dt]["profile"][:, feat_i]
                rows.append({"Date": dt.strftime("%d-%b-%Y"), "Day": dt.strftime("%A"),
                             "Avg": round(prof.mean(), 2), "Min": round(prof.min(), 2), "Max": round(prof.max(), 2)})
            tables[feat_name] = pd.DataFrame(rows)
            tables[feat_name].index = ["⭐ Target"] + [f"#{i}" for i in range(1, len(results) + 1)]
            tables[feat_name].index.name = ""
        return tables

    def _get_day_load(load_df, dt, load_col):
        mask = load_df["date"].dt.normalize() == pd.Timestamp(dt).normalize()
        day  = load_df[mask].sort_values("time_block")
        return day[load_col].values if not day.empty else None

    def _get_day_weather(dt, var_col):
        mask = df["date"].dt.normalize() == pd.Timestamp(dt).normalize()
        day  = df[mask].sort_values("time_block")
        return day[var_col].values if not day.empty else None

    def _plot_load_overlay(dates_to_plot, load_df, load_col):
        fig = go.Figure()
        for _, dt, color, lw in dates_to_plot:
            vals = _get_day_load(load_df, dt, load_col)
            if vals is not None:
                fig.add_trace(go.Scatter(
                    x=np.arange(1, len(vals) + 1), y=vals, mode="lines",
                    name=f"{load_col} | {pd.Timestamp(dt).strftime('%Y-%m-%d')}",
                    line=dict(color=color, width=lw),
                ))
        fig.update_layout(
            height=480, template="plotly_white",
            xaxis_title="block", yaxis_title="Value",
            legend=dict(title="Series", orientation="v", yanchor="middle", y=0.5,
                        xanchor="left", x=1.01, bordercolor="#ccc", borderwidth=1),
            margin=dict(r=200),
        )
        return fig

    def _highlight_pct(series):
        return [
            "" if pd.isna(v)
            else "background-color: #d4edda; color: #155724" if v >= 0
            else "background-color: #f8d7da; color: #721c24"
            for v in series
        ]

    def _pct_change_section(arr_a, arr_b, date_a, date_b, y_label, plot_key, tbl_key):
        n = min(len(arr_a), len(arr_b), 96)
        a_arr, b_arr = arr_a[:n], arr_b[:n]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(b_arr != 0, (a_arr - b_arr) / np.abs(b_arr) * 100, np.nan)
        tbl = pd.DataFrame({
            "Block": np.arange(1, n + 1),
            f"Date B  {pd.Timestamp(date_b).strftime('%d-%b-%Y')}": np.round(b_arr, 2),
            f"Date A  {pd.Timestamp(date_a).strftime('%d-%b-%Y')}": np.round(a_arr, 2),
            "% Change (A vs B)": np.round(pct, 2),
        })
        valid_pct = pd.Series(pct).dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg % Change", f"{valid_pct.mean():.2f}%")
        m2.metric("Max Increase", f"{valid_pct.max():.2f}%")
        m3.metric("Max Decrease", f"{valid_pct.min():.2f}%")
        m4.metric("Blocks ↑ / ↓", f"{(valid_pct > 0).sum()} / {(valid_pct < 0).sum()}")
        st.dataframe(tbl.style.apply(_highlight_pct, subset=["% Change (A vs B)"]),
                     width='stretch', key=tbl_key)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(b_arr) + 1), y=b_arr, mode="lines",
                                 name=f"Date B  {pd.Timestamp(date_b).strftime('%d-%b-%Y (%a)')}",
                                 line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=np.arange(1, len(a_arr) + 1), y=a_arr, mode="lines",
                                 name=f"Date A  {pd.Timestamp(date_a).strftime('%d-%b-%Y (%a)')}",
                                 line=dict(color="#EF4444", width=2)))
        fig.update_layout(height=380, template="plotly_white",
                          xaxis_title="Time Block (15-min interval)", yaxis_title=y_label,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig, width='stretch', key=plot_key)

    # ── Load weather data ──────────────────────────────────────────────────────
    try:
        df = _sim_load_weather_data(STATE_SLUG)
    except Exception as e:
        st.error(f"Failed to load weather data: {e}")
        st.stop()

    st.success(f"Loaded **{len(df):,}** rows · **{df['date'].nunique()}** days · State: **{STATE_DISPLAY}**")
    col_a, col_b = st.columns(2)
    col_a.metric("From", df["date"].min().strftime("%d-%b-%Y"))
    col_b.metric("To",   df["date"].max().strftime("%d-%b-%Y"))

    # ── Categorise ────────────────────────────────────────────────────────────
    years   = df["date"].dt.year.unique()
    h_set   = _get_india_holidays(list(years), STATE_HOLIDAY_CODE)
    cat_map = _categorise_dates(df["date"].unique(), h_set)
    df["day_category"] = df["date"].map(cat_map)
    cat_counts = df.groupby("date").first()["day_category"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Working days", cat_counts.get("working", 0))
    c2.metric("🟡 Sundays",      cat_counts.get("sunday",  0))
    c3.metric("🎉 Holidays",     cat_counts.get("holiday", 0))

    _skip_wx = {"date", "time_block", "day_category", "datetime"}
    weather_vars = [c for c in df.columns if c not in _skip_wx and pd.api.types.is_numeric_dtype(df[c])]

    profiles = _build_profiles(df, cat_map)
    stats    = _compute_stats(df)

    # ── Target date picker ────────────────────────────────────────────────────
    st.divider()
    all_dates = sorted(profiles.keys())
    min_dt = all_dates[0].date()  if hasattr(all_dates[0],  "date") else all_dates[0]
    max_dt = all_dates[-1].date() if hasattr(all_dates[-1], "date") else all_dates[-1]
    _today      = datetime.today().date()
    _default_dt = max(min_dt, min(_today, max_dt))

    target_input = st.date_input(
        "Select target date", value=_default_dt,
        min_value=min_dt, max_value=max_dt, key="sim_target_date",
    )
    target_date = next((d for d in all_dates if (d.date() if hasattr(d, "date") else d) == target_input), None)
    if target_date is None:
        st.warning(f"No weather data for **{target_input.strftime('%d-%b-%Y')}**. Pick another date.")
        st.stop()

    _prev_m = (target_input.month - 2) % 12 + 1
    _next_m = target_input.month % 12 + 1
    _window_str = f"{_cal.month_abbr[_prev_m]}, {_cal.month_abbr[target_input.month]}, {_cal.month_abbr[_next_m]}"
    st.info(
        f"**{target_input.strftime('%d-%b-%Y (%A)')}**  →  Category: "
        f"**{CATEGORY_LABELS[cat_map[target_date]]}**  ·  Search window: **{_window_str}** (all years)"
    )

    # ── Find Similar Days ─────────────────────────────────────────────────────
    if st.button("🔍 Find Similar Days", type="primary", width='stretch', key="sim_find_btn"):
        with st.spinner("Calculating distances…"):
            _results = _find_similar(target_date, profiles, stats, method, feature_weights, top_k)
        if not _results:
            st.warning("No similar days found in the same category.")
        else:
            st.session_state["sim_results"]     = _results
            st.session_state["sim_result_target_date"] = target_date

    # ── Tab navigation ────────────────────────────────────────────────────────
    _TAB_WEATHER     = "🌤️  Similar Weather"
    _TAB_WEATHER_VAR = "🌡️  Weather"
    _TAB_LOAD        = "⚡  Load"

    active_tab = st.radio(
        "Navigation", [_TAB_WEATHER, _TAB_WEATHER_VAR, _TAB_LOAD],
        horizontal=True, key="sim_active_tab", label_visibility="collapsed",
    )
    st.divider()

    # ═══ Tab 1 — Similar Weather ═══════════════════════════════════════════════
    if active_tab == _TAB_WEATHER:
        if "sim_results" not in st.session_state:
            st.info("Click **🔍 Find Similar Days** above to see similarity results.")
        else:
            results     = st.session_state["sim_results"]
            target_date = st.session_state["sim_result_target_date"]

            st.subheader(
                f"Top {len(results)} similar days to "
                f"{target_date.strftime('%d-%b-%Y')}  ({CATEGORY_LABELS[cat_map[target_date]]})"
            )
            res_df = pd.DataFrame(results, columns=["Date", "Distance"])
            res_df.index      = range(1, len(res_df) + 1)
            res_df.index.name = "Rank"
            res_df["Day"]      = res_df["Date"].apply(lambda d: d.strftime("%A"))
            res_df["Category"] = [CATEGORY_LABELS[profiles[r[0]]["category"]] for r in results]
            res_df["Distance"] = res_df["Distance"].round(4)
            res_df["Date"]     = res_df["Date"].apply(lambda d: d.strftime("%d-%b-%Y"))
            st.dataframe(res_df[["Date", "Day", "Category", "Distance"]], width='stretch')

            st.divider()
            st.subheader("📊 Multi-Day Weather Comparison")
            st.caption("Target = solid line · Similar days = dotted lines")
            match_dates = [r[0] for r in results]
            all_options = [target_date] + match_dates
            option_labels = {
                d: f"{'⭐ TARGET — ' if d == target_date else ''}{d.strftime('%d-%b-%Y (%A)')}"
                for d in all_options
            }
            selected_w = st.multiselect(
                "Days to plot", options=all_options,
                default=[target_date, match_dates[0]],
                format_func=lambda d: option_labels[d],
                key="sim_weather_multi_select",
            )
            if selected_w:
                st.plotly_chart(_plot_multi_overlay(selected_w, profiles, target_date),
                                width='stretch', key="sim_weather_overlay")
            else:
                st.info("Select at least one day to plot.")

            st.divider()
            st.subheader("📋 Feature-wise Summary  (Target + Similar Days)")
            for feat_name, tbl in _build_feature_summary(target_date, results, profiles).items():
                st.markdown(f"**{feat_name}**")
                st.dataframe(tbl, width='stretch', key=f"sim_wtbl_{feat_name}")

    # ═══ Tab 2 — Load ══════════════════════════════════════════════════════════
    elif active_tab == _TAB_LOAD:
        if "sim_results" not in st.session_state:
            st.info("Run **🔍 Find Similar Days** first, then come back here for load analysis.")
        else:
            results     = st.session_state["sim_results"]
            target_date = st.session_state["sim_result_target_date"]
            st.subheader("⚡ Load Profile Comparison")

            load_df, load_cols_list = None, []
            try:
                load_df, load_cols_list = _sim_load_load_data(STATE_SLUG)
            except Exception as e:
                st.error(f"Could not load demand data: {e}")

            if load_df is not None:
                load_col = (
                    st.selectbox("Select load column", load_cols_list, key="sim_load_col_sel")
                    if len(load_cols_list) > 1
                    else load_cols_list[0]
                )
                if len(load_cols_list) == 1:
                    st.caption(f"Load column: **{load_col}**")

                load_date_set = set(load_df["date"].dt.date.unique())
                load_min, load_max = min(load_date_set), max(load_date_set)

                st.divider()
                st.subheader("C) Overlay of Load (blocks 1-96)")
                st.markdown("**Daily Overlay - Load (SLDC)**")

                _seed_key = f"sim_plot_dates_{target_date}"
                if st.session_state.get("_sim_plot_dates_seed") != _seed_key:
                    st.session_state["sim_plot_dates"]      = (
                        [target_date] if _get_day_load(load_df, target_date, load_col) is not None else []
                    )
                    st.session_state["_sim_plot_dates_seed"] = _seed_key

                pc1, pc2, pc3 = st.columns([5, 1, 1])
                new_date = pc1.date_input("Add comparison date", value=None,
                                          min_value=load_min, max_value=load_max, key="sim_new_plot_date")
                if pc2.button("Add date", key="sim_btn_add_date"):
                    if new_date is not None:
                        existing = [pd.Timestamp(d).date() for d in st.session_state["sim_plot_dates"]]
                        if new_date not in existing:
                            if new_date in load_date_set:
                                st.session_state["sim_plot_dates"].append(pd.Timestamp(new_date))
                            else:
                                st.warning(f"No load data for {pd.Timestamp(new_date).strftime('%d-%b-%Y')}.")
                if pc3.button("Clear", key="sim_btn_clear_dates"):
                    st.session_state["sim_plot_dates"] = []

                if st.session_state["sim_plot_dates"]:
                    st.caption("Comparing: " + ", ".join(
                        pd.Timestamp(d).strftime("%Y-%m-%d") for d in sorted(st.session_state["sim_plot_dates"])
                    ))

                st.divider()
                plot_entries = [
                    (None, dt, PALETTE[i % len(PALETTE)],
                     2.5 if pd.Timestamp(dt).date() == pd.Timestamp(target_date).date() else 1.8)
                    for i, dt in enumerate(sorted(st.session_state["sim_plot_dates"]))
                ]
                if plot_entries:
                    st.plotly_chart(_plot_load_overlay(plot_entries, load_df, load_col),
                                    width='stretch', key="sim_load_main_plot")
                else:
                    st.info("Add at least one date using the picker above.")

                st.divider()
                st.subheader("📋 Block-wise % Change")
                st.caption("% Change = (Date A − Date B) / |Date B| × 100  ·  🟢 increase  🔴 decrease")
                tc1, tc2 = st.columns(2)
                date_a = tc1.date_input("Date A  (compare)",   value=None, min_value=load_min, max_value=load_max, key="sim_tbl_date_a")
                date_b = tc2.date_input("Date B  (reference)", value=None, min_value=load_min, max_value=load_max, key="sim_tbl_date_b")

                if date_a is None or date_b is None:
                    st.info("Select both Date A and Date B above to see the % change table.")
                elif date_a == date_b:
                    st.warning("Date A and Date B must be different.")
                else:
                    load_a = _get_day_load(load_df, date_a, load_col)
                    load_b = _get_day_load(load_df, date_b, load_col)
                    if load_a is None:
                        st.warning(f"No load data for Date A: {pd.Timestamp(date_a).strftime('%d-%b-%Y')}.")
                    elif load_b is None:
                        st.warning(f"No load data for Date B: {pd.Timestamp(date_b).strftime('%d-%b-%Y')}.")
                    else:
                        st.markdown("**Load profile: Date A vs Date B**")
                        _pct_change_section(load_a, load_b, date_a, date_b,
                                            f"Load  [{load_col}]", "sim_load_tbl_plot", "sim_load_tbl_custom")

    # ═══ Tab 3 — Weather Variable Overlay ══════════════════════════════════════
    elif active_tab == _TAB_WEATHER_VAR:
        st.subheader("🌡️ Weather Variable Overlay")
        if not weather_vars:
            st.error("No numeric weather columns detected in the table.")
        else:
            wx_var = st.selectbox(
                "Select weather variable", weather_vars,
                format_func=lambda c: c.replace("_", " ").title(),
                key="sim_wx_var_sel",
            )
            wx_date_set = set(df["date"].dt.date.unique())
            wx_min, wx_max = min(wx_date_set), max(wx_date_set)

            _wx_seed_key = f"sim_wx_dates_{target_date}"
            if st.session_state.get("_sim_wx_dates_seed") != _wx_seed_key:
                st.session_state["sim_wx_dates"]       = [target_date]
                st.session_state["_sim_wx_dates_seed"] = _wx_seed_key

            wc1, wc2, wc3 = st.columns([5, 1, 1])
            wx_new_date = wc1.date_input("Add comparison date", value=None,
                                         min_value=wx_min, max_value=wx_max, key="sim_wx_new_date")
            if wc2.button("Add date", key="sim_wx_btn_add"):
                if wx_new_date is not None:
                    existing = [pd.Timestamp(d).date() for d in st.session_state["sim_wx_dates"]]
                    if wx_new_date not in existing:
                        if wx_new_date in wx_date_set:
                            st.session_state["sim_wx_dates"].append(pd.Timestamp(wx_new_date))
                        else:
                            st.warning(f"No weather data for {pd.Timestamp(wx_new_date).strftime('%d-%b-%Y')}.")
            if wc3.button("Clear", key="sim_wx_btn_clear"):
                st.session_state["sim_wx_dates"] = []

            if st.session_state["sim_wx_dates"]:
                st.caption("Comparing: " + ", ".join(
                    pd.Timestamp(d).strftime("%Y-%m-%d") for d in sorted(st.session_state["sim_wx_dates"])
                ))

            st.divider()
            st.subheader(f"C) Overlay of {wx_var.replace('_', ' ').title()} (blocks 1-96)")
            st.markdown(f"**Daily Overlay — {wx_var}**")

            if not st.session_state["sim_wx_dates"]:
                st.info("Add at least one date using the picker above.")
            else:
                fig_wx = go.Figure()
                for idx, dt in enumerate(sorted(st.session_state["sim_wx_dates"])):
                    vals = _get_day_weather(dt, wx_var)
                    if vals is not None:
                        fig_wx.add_trace(go.Scatter(
                            x=np.arange(1, len(vals) + 1), y=vals, mode="lines",
                            name=f"{wx_var} | {pd.Timestamp(dt).strftime('%Y-%m-%d')}",
                            line=dict(
                                color=PALETTE[idx % len(PALETTE)],
                                width=2.5 if pd.Timestamp(dt).date() == pd.Timestamp(target_date).date() else 1.8,
                            ),
                        ))
                fig_wx.update_layout(
                    height=480, template="plotly_white",
                    xaxis_title="block", yaxis_title=wx_var.replace("_", " ").title(),
                    legend=dict(title="Series", orientation="v", yanchor="middle", y=0.5,
                                xanchor="left", x=1.01, bordercolor="#ccc", borderwidth=1),
                    margin=dict(r=200),
                )
                st.plotly_chart(fig_wx, width='stretch', key="sim_wx_overlay_plot")

            st.divider()
            st.subheader("📋 Block-wise % Change")
            st.caption("% Change = (Date A − Date B) / |Date B| × 100  ·  🟢 increase  🔴 decrease")
            wt1, wt2 = st.columns(2)
            wx_date_a = wt1.date_input("Date A  (compare)",   value=None, min_value=wx_min, max_value=wx_max, key="sim_wx_tbl_date_a")
            wx_date_b = wt2.date_input("Date B  (reference)", value=None, min_value=wx_min, max_value=wx_max, key="sim_wx_tbl_date_b")

            if wx_date_a is None or wx_date_b is None:
                st.info("Select both dates above to see the % change table.")
            elif wx_date_a == wx_date_b:
                st.warning("Date A and Date B must be different.")
            else:
                arr_a = _get_day_weather(wx_date_a, wx_var)
                arr_b = _get_day_weather(wx_date_b, wx_var)
                if arr_a is None:
                    st.warning(f"No data for Date A: {pd.Timestamp(wx_date_a).strftime('%d-%b-%Y')}.")
                elif arr_b is None:
                    st.warning(f"No data for Date B: {pd.Timestamp(wx_date_b).strftime('%d-%b-%Y')}.")
                else:
                    st.markdown(f"**{wx_var.replace('_', ' ').title()}: Date A vs Date B**")
                    _pct_change_section(arr_a, arr_b, wx_date_a, wx_date_b,
                                        wx_var.replace("_", " ").title(), "sim_wx_tbl_plot", "sim_wx_tbl_custom")


# =========================
# MAIN
# =========================
if st.session_state.get("selected_state_slug"):
    _set_active_state(st.session_state["selected_state_slug"])

st.title(f"{STATE_DISPLAY} Dashboard")
st.caption(f"Weather and Electricity Load dashboards for **{STATE_DISPLAY}**. Use the sidebar to switch views.")

api_configured = _is_api_configured()

with st.sidebar:
    st.header("⚡ State Dashboard")
    if not api_configured:
        st.error("FastAPI URL/token is missing.")
        st.caption("Set `PIPELINE_API_BASE_URL` and `PIPELINE_API_TOKEN`.")
        st.stop()
    api_ok, api_err = _api_healthcheck()
    if not api_ok:
        st.error("FastAPI endpoint is not reachable.")
        st.caption(api_err)
        st.stop()
    st.caption("Data source: FastAPI (read-only)")

    # state_options = ACTIVE_STATE_OPTIONS if ACTIVE_STATE_OPTIONS else [DEFAULT_STATE_SLUG]
    state_options = [s for s in ['HARYANA', 'CHHATTISGARH', 'DELHI', 'PUNJAB', 'UTTAR_PRADESH'] if s in ACTIVE_STATE_OPTIONS] or ['HARYANA', 'CHHATTISGARH', 'DELHI', 'PUNJAB', 'UTTAR_PRADESH']
    default_index = state_options.index(DEFAULT_STATE_SLUG) if DEFAULT_STATE_SLUG in state_options else 0
    selected_state_slug = st.selectbox(
        "Select state",
        options=state_options,
        index=default_index,
        format_func=lambda slug: _get_state_meta(slug)["display"],
        key="selected_state_slug",
    )

    if st.session_state.get("_active_state_slug") != selected_state_slug:
        keep_keys = {"selected_state_slug", "_active_state_slug"}
        for key in list(st.session_state.keys()):
            if key not in keep_keys:
                del st.session_state[key]
        st.session_state["_active_state_slug"] = selected_state_slug
        _set_active_state(selected_state_slug)
        st.rerun()

    st.session_state["_active_state_slug"] = selected_state_slug
    _set_active_state(selected_state_slug)

    st.info(f"**State:** {STATE_DISPLAY}")
    dash_choice = st.radio(
        "Select dashboard",
        ["Weather", "Load (Actual + Forecast)", "Combined (Weather + Load)", "Similar Weather Days"],
        index=1, key="dash_choice"
    )
    st.markdown("---")

if dash_choice == "Weather":
    render_weather_dashboard()
elif dash_choice == "Load (Actual + Forecast)":
    render_load_dashboard()
elif dash_choice == "Similar Weather Days":
    render_similarity_dashboard()
else:
    render_combined_dashboard()
