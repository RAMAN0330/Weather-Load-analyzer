"""Seeded generator for state electricity demand and district weather.

Model (per state, per 15-minute block):

* **Weather** per district = monthly climatology (temperature, humidity,
  cloud), a diurnal cycle, a smooth state-wide synoptic anomaly (sum of slow
  sinusoids with state-specific phases, so heat waves and cool spells span
  days) and district offsets. Rain days follow a monthly probability (monsoon
  Jul-Aug); storms arrive in the afternoon/evening, hit a random subset of
  districts, cool the air and raise humidity.
* **Load** = seasonal level (growth trend, summer / paddy-season uplift)
  x daily shape (morning and evening peaks, night trough; summer nights stay
  high from agricultural pumping) + a weather response to the *temperature
  anomaly* (cooling in summer, mild heating in winter) - rain relief - weekend
  and holiday dips + small day-level and block-level noise.
* **Forecast tables** are the noise-free expected load plus a persistent
  daily bias and block noise, so they behave like an imperfect external
  forecast.

Availability mirrors production: actual load up to the last settled block of
today (IST); weather and forecasts up to today + 2 days.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
BLOCKS = 96
DATA_START = date(2023, 1, 1)
LEAD_DAYS = 2  # weather + forecast tables extend to today + 2

TABLES = ("load", "sldc", "weather_mean", "weather_loc", "forecast", "sldc_forecast")
WEATHER_COLS = (
    "temperature",
    "humidity",
    "precipitation",
    "cloud_cover",
    "cloud_cover_low",
    "sunshine_duration",
    "direct_radiation",
    "wind_speed_10m",
)

# ── State profiles ───────────────────────────────────────────────────────────
# base: typical Feb/Mar demand (MW); summer_gain: uplift at the seasonal peak;
# temp_coef: MW per °C of temperature anomaly in summer; night_agri: share of
# load added at night in the paddy season; temp/humid offsets vs. Haryana.
STATE_PROFILES: dict[str, dict] = {
    "HARYANA": dict(base=6300.0, summer_gain=0.62, temp_coef=140.0, night_agri=0.10, growth=0.055, temp=0.0, humid=0.0, rain=1.0),
    "PUNJAB": dict(base=6800.0, summer_gain=0.78, temp_coef=155.0, night_agri=0.16, growth=0.05, temp=-0.6, humid=3.0, rain=1.05),
    "RAJASTHAN": dict(base=11800.0, summer_gain=0.22, temp_coef=165.0, night_agri=0.12, growth=0.06, temp=1.6, humid=-8.0, rain=0.6),
    "ODISHA": dict(base=4400.0, summer_gain=0.24, temp_coef=60.0, night_agri=0.02, growth=0.05, temp=1.4, humid=12.0, rain=1.4),
    "CHHATTISGARH": dict(base=4200.0, summer_gain=0.20, temp_coef=58.0, night_agri=0.04, growth=0.05, temp=1.2, humid=4.0, rain=1.2),
}

# District name -> (temperature offset °C, humidity offset %, rain participation 0..1)
LOCATIONS: dict[str, dict[str, tuple[float, float, float]]] = {
    "HARYANA": {
        "AMBALA": (-0.7, 5.0, 0.80), "BHIWANI": (0.9, -5.0, 0.50), "FARIDABAD": (0.3, 0.0, 0.62),
        "FATEHABAD": (1.1, -6.0, 0.48), "GURUGRAM": (0.2, -1.0, 0.60), "HISAR": (1.0, -6.0, 0.48),
        "JHAJJAR": (0.4, -2.0, 0.55), "JIND": (0.5, -2.0, 0.56), "KAITHAL": (0.0, 1.0, 0.64),
        "KARNAL": (-0.3, 3.0, 0.70), "KURUKSHETRA": (-0.4, 4.0, 0.74), "MAHENDRAGARH": (0.8, -6.0, 0.45),
        "NUH": (0.5, -2.0, 0.55), "PALWAL": (0.4, 0.0, 0.58), "PANCHKULA": (-1.6, 7.0, 0.88),
        "PANIPAT": (-0.1, 2.0, 0.66), "REWARI": (0.6, -4.0, 0.50), "ROHTAK": (0.3, -1.0, 0.58),
        "SIRSA": (1.2, -7.0, 0.44), "SONIPAT": (0.0, 1.0, 0.64), "YAMUNANAGAR": (-0.9, 6.0, 0.85),
    },
    "PUNJAB": {
        "AMRITSAR": (-0.4, 3.0, 0.70), "LUDHIANA": (0.0, 2.0, 0.66), "JALANDHAR": (-0.3, 3.0, 0.70),
        "PATIALA": (0.1, 2.0, 0.65), "BATHINDA": (1.0, -5.0, 0.48), "MOHALI": (-0.8, 5.0, 0.80),
    },
    "RAJASTHAN": {
        "JAIPUR": (0.0, 0.0, 0.60), "JODHPUR": (1.2, -6.0, 0.40), "BIKANER": (1.5, -8.0, 0.35),
        "UDAIPUR": (-1.0, 6.0, 0.70), "KOTA": (0.6, 2.0, 0.62), "AJMER": (0.2, -2.0, 0.55),
    },
    "ODISHA": {
        "BHUBANESWAR": (0.0, 0.0, 0.70), "CUTTACK": (0.1, 1.0, 0.70), "ROURKELA": (-0.5, -3.0, 0.62),
        "SAMBALPUR": (0.4, -4.0, 0.60), "BERHAMPUR": (-0.2, 4.0, 0.72), "BALASORE": (-0.1, 3.0, 0.74),
    },
    "CHHATTISGARH": {
        "RAIPUR": (0.0, 0.0, 0.65), "BILASPUR": (0.2, -1.0, 0.62), "DURG": (0.1, -1.0, 0.62),
        "KORBA": (0.3, -2.0, 0.60), "JAGDALPUR": (-1.2, 6.0, 0.75), "AMBIKAPUR": (-1.5, 3.0, 0.70),
    },
}

# Monthly climatology for north India (Jan..Dec), shifted per state by offsets.
_T_MEAN = np.array([13.5, 16.5, 22.0, 28.0, 33.0, 34.0, 31.0, 30.0, 29.5, 25.5, 19.5, 14.5])
_T_AMP = np.array([6.0, 6.5, 7.0, 7.5, 7.0, 5.5, 3.5, 3.0, 3.5, 6.0, 6.5, 6.0])  # diurnal half-range
_RH = np.array([72.0, 65.0, 52.0, 35.0, 30.0, 42.0, 72.0, 78.0, 70.0, 55.0, 60.0, 70.0])
_CLOUD = np.array([22.0, 25.0, 20.0, 15.0, 15.0, 32.0, 66.0, 70.0, 45.0, 12.0, 10.0, 20.0])
_P_RAIN = np.array([0.06, 0.07, 0.06, 0.04, 0.06, 0.16, 0.46, 0.45, 0.26, 0.03, 0.02, 0.04])
# Demand seasonal index (0 = Feb/Mar trough, 1 = Jul peak): AC + paddy pumping.
_SEASON = np.array([0.10, 0.00, 0.06, 0.28, 0.58, 0.95, 1.00, 0.90, 0.74, 0.36, 0.12, 0.16])

# Fixed national holidays (+ approximate festival dates) for the demand dip.
_HOLIDAYS = {
    "01-26", "08-15", "10-02",
    "2023-03-08", "2023-10-24", "2023-11-12", "2024-03-25", "2024-10-12", "2024-10-31",
    "2025-03-14", "2025-10-02", "2025-10-20", "2026-03-04", "2026-10-20", "2026-11-08",
}

_HOURS = (np.arange(BLOCKS) + 0.5) / 4.0  # block centre, hours since midnight


# ── Small helpers ────────────────────────────────────────────────────────────
def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.md5("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _monthly(table: np.ndarray, d: date) -> float:
    """Smooth periodic interpolation of a Jan..Dec table at day ``d`` (values at mid-month)."""
    pos = (d.timetuple().tm_yday - 15.2) / 30.44  # 0 = mid-Jan
    i0 = int(np.floor(pos)) % 12
    frac = pos - np.floor(pos)
    return float(table[i0] * (1 - frac) + table[(i0 + 1) % 12] * frac)


def _anomaly(state: str, t_days: np.ndarray) -> np.ndarray:
    """Smooth state-wide temperature anomaly (°C) at fractional day ``t_days``."""
    rng = np.random.default_rng(_seed(state, "synoptic"))
    out = np.zeros_like(t_days, dtype=np.float64)
    for amp, period in ((1.7, 5.3), (1.2, 9.7), (0.9, 17.1), (0.7, 31.0)):
        out += amp * np.sin(2 * np.pi * t_days / period + rng.uniform(0, 2 * np.pi))
    return out


def _as_date(d: object) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _norm_state(state: object) -> str:
    s = str(state or "HARYANA").strip().upper().replace(" ", "_").replace("-", "_")
    return s if s in STATE_PROFILES else "HARYANA"


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def settled_blocks_today() -> int:
    n = now_ist()
    return min(BLOCKS, (n.hour * 60 + n.minute) // 15)


# ── Public metadata ──────────────────────────────────────────────────────────
def states() -> list[str]:
    return sorted(STATE_PROFILES)


def locations(state: object) -> list[str]:
    return sorted(LOCATIONS[_norm_state(state)])


def date_range(state: object, table: str = "load") -> tuple[date, date]:
    end = today_ist() if table in ("load", "sldc") else today_ist() + timedelta(days=LEAD_DAYS)
    return DATA_START, end


def load_dates(state: object) -> list[str]:
    start, end = date_range(state, "load")
    return [d.isoformat() for d in pd.date_range(start, end, freq="D").date]


# ── Weather (per day, all districts) ─────────────────────────────────────────
@lru_cache(maxsize=900)
def _weather_day(state: str, iso: str) -> dict[str, np.ndarray]:
    """{var: array (n_locations, 96)} for one day; float32 to keep the cache small."""
    d = date.fromisoformat(iso)
    prof = STATE_PROFILES[state]
    locs = LOCATIONS[state]
    names = sorted(locs)
    off_t = np.array([locs[n][0] for n in names])[:, None]
    off_h = np.array([locs[n][1] for n in names])[:, None]
    part = np.array([locs[n][2] for n in names])
    rng = np.random.default_rng(_seed(state, iso, "wx"))

    t_mean = _monthly(_T_MEAN, d) + prof["temp"]
    t_amp = _monthly(_T_AMP, d)
    rh_mean = _monthly(_RH, d) + prof["humid"]
    cloud_base = _monthly(_CLOUD, d)

    epoch = (d - DATA_START).days
    anom = _anomaly(state, epoch + _HOURS / 24.0)[None, :]
    # Diurnal cycle: minimum ~05:30, maximum ~15:00 (asymmetric).
    phase = 2 * np.pi * (_HOURS - 15.0) / 24.0
    diurnal = np.cos(phase) + 0.18 * np.cos(2 * phase + 0.6)
    temp = t_mean + anom + off_t + t_amp * diurnal[None, :] + rng.normal(0, 0.35, (len(names), 1))
    hum = rh_mean + off_h - 0.9 * (temp - t_mean - off_t) * (rh_mean / 60.0)
    cloud = np.clip(cloud_base + rng.normal(0, 8, (len(names), 1)) + 6 * np.sin(phase)[None, :], 0, 100)

    precip = np.zeros_like(temp)
    p_rain = min(0.95, _monthly(_P_RAIN, d) * prof["rain"])
    if rng.random() < p_rain:
        start = int(np.clip(rng.normal(64, 10), 36, 88))  # ~16:00 typical
        dur = int(rng.integers(4, 17))
        intensity = rng.gamma(1.3, 1.4)  # mm per 15 min at the storm core
        hit = rng.random(len(names)) < np.clip(part * (0.75 + 0.5 * p_rain), 0, 0.97)
        scale = rng.lognormal(0.0, 0.45, len(names)) * hit
        idx = np.arange(BLOCKS)
        core = np.exp(-0.5 * ((idx - (start + dur / 2)) / max(dur / 2.5, 1.0)) ** 2)
        precip = (intensity * scale)[:, None] * core[None, :]
        wet = np.clip(np.cumsum(precip, axis=1), 0, 12.0) / 12.0  # 0..1 since storm began
        decay = np.where(idx[None, :] >= start, 1.0, 0.0) * hit[:, None]
        temp = temp - 4.0 * wet * decay
        hum = hum + 22.0 * wet * decay
        cloud = np.clip(cloud + 55.0 * np.clip(core * 3, 0, 1)[None, :] * hit[:, None], 0, 100)
    hum = np.clip(hum, 8, 99)

    sunrise = 6.4 - 0.9 * np.cos(2 * np.pi * (d.timetuple().tm_yday - 172) / 365.0)
    sunset = 18.6 + 0.9 * np.cos(2 * np.pi * (d.timetuple().tm_yday - 172) / 365.0)
    day_frac = np.clip((_HOURS - sunrise) / (sunset - sunrise), 0, 1)
    elev = np.where((_HOURS > sunrise) & (_HOURS < sunset), np.sin(np.pi * day_frac), 0.0)[None, :]
    sunshine = 900.0 * (elev > 0) * np.clip(1 - 0.9 * cloud / 100, 0, 1)
    radiation = 920.0 * elev * np.clip(1 - 0.8 * cloud / 100, 0, 1)
    wind = np.clip(7.5 + 3.0 * np.sin(phase)[None, :] + rng.normal(0, 1.2, (len(names), 1)) + 4.0 * np.sqrt(precip), 0.5, 60)

    out = {
        "temperature": temp,
        "humidity": hum,
        "precipitation": precip,
        "cloud_cover": cloud,
        "cloud_cover_low": np.clip(cloud * 0.55, 0, 100),
        "sunshine_duration": sunshine,
        "direct_radiation": radiation,
        "wind_speed_10m": wind,
    }
    return {k: np.round(v, 2).astype(np.float32) for k, v in out.items()}


def _weather_mean_day(state: str, iso: str) -> dict[str, np.ndarray]:
    return {k: v.mean(axis=0) for k, v in _weather_day(state, iso).items()}


# ── Load (per day) ───────────────────────────────────────────────────────────
def _is_holiday(d: date) -> bool:
    return d.isoformat() in _HOLIDAYS or d.strftime("%m-%d") in _HOLIDAYS


def _bump(center: float, width: float) -> np.ndarray:
    dist = np.minimum(np.abs(_HOURS - center), 24 - np.abs(_HOURS - center))
    return np.exp(-0.5 * (dist / width) ** 2)


@lru_cache(maxsize=1500)
def _expected_load(state: str, iso: str) -> np.ndarray:
    """Noise-free expected demand (MW, 96 blocks) given that day's weather."""
    d = date.fromisoformat(iso)
    prof = STATE_PROFILES[state]
    season = _monthly(_SEASON, d)
    years = (d - DATA_START).days / 365.25
    level = prof["base"] * (1 + prof["growth"]) ** years * (1 + prof["summer_gain"] * season)

    winter = 1.0 - season
    shape = (
        1.0
        + 0.11 * _bump(9.75, 1.6) * (0.6 + 0.4 * winter)     # morning peak
        + 0.15 * _bump(19.5, 1.5) * (0.5 + 0.5 * winter)     # evening lighting peak
        + 0.12 * _bump(22.5, 2.0) * season                  # summer late-evening AC peak
        + 0.07 * _bump(15.0, 2.2) * season                  # summer afternoon AC
        - 0.20 * _bump(4.0, 2.2) * (1 - 0.6 * season)        # night trough (shallower in summer)
        + prof["night_agri"] * _bump(1.5, 3.0) * season      # paddy-season night pumping
    )
    load = level * shape

    wx = _weather_mean_day(state, iso)
    t_clim = _monthly(_T_MEAN, d) + prof["temp"] + _monthly(_T_AMP, d) * (
        np.cos(2 * np.pi * (_HOURS - 15.0) / 24.0) + 0.18 * np.cos(4 * np.pi * (_HOURS - 15.0) / 24.0 + 0.6)
    )
    t_anom = wx["temperature"].astype(np.float64) - t_clim
    coef = prof["temp_coef"] * np.where(t_anom > 0, 0.25 + 0.75 * season, -0.25 * winter + 0.6 * season)
    load = load + coef * t_anom

    rain_today = float(wx["precipitation"].sum())
    prev = (d - timedelta(days=1)).isoformat()
    rain_prev = float(_weather_mean_day(state, prev)["precipitation"].sum()) if d > DATA_START else 0.0
    relief = min(0.12, 0.010 * rain_today + 0.005 * rain_prev) * (0.4 + 0.6 * season)
    load = load * (1 - relief)

    dow = d.weekday()
    load = load * (0.94 if dow == 6 else 0.975 if dow == 5 else 1.0) * (0.92 if _is_holiday(d) else 1.0)
    return load


@lru_cache(maxsize=1500)
def _actual_load(state: str, iso: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(state, iso, "load"))
    day_noise = rng.normal(0, 0.012)
    ar = np.zeros(BLOCKS)
    eps = rng.normal(0, 0.005, BLOCKS)
    for i in range(1, BLOCKS):
        ar[i] = 0.85 * ar[i - 1] + eps[i]
    return np.round(_expected_load(state, iso) * (1 + day_noise + ar), 2)


def _forecast_day(state: str, iso: str, tag: str, bias_sd: float, noise_sd: float) -> np.ndarray:
    rng = np.random.default_rng(_seed(state, iso, tag))
    bias = rng.normal(0, bias_sd)
    return np.round(_expected_load(state, iso) * (1 + bias + rng.normal(0, noise_sd, BLOCKS)), 2)


# ── Day iteration ────────────────────────────────────────────────────────────
def _days(state: str, table: str, day=None, from_date=None, to_date=None) -> list[date]:
    lo, hi = date_range(state, table)
    if day is not None:
        d = _as_date(day)
        return [d] if lo <= d <= hi else []
    start = max(lo, _as_date(from_date)) if from_date else lo
    end = min(hi, _as_date(to_date)) if to_date else hi
    if end < start:
        return []
    return list(pd.date_range(start, end, freq="D").date)


def _blocks_available(table: str, d: date) -> int:
    if table in ("load", "sldc") and d == today_ist():
        return settled_blocks_today()
    return BLOCKS


def _day_frame(state: str, table: str, d: date, location: str | None = None) -> pd.DataFrame:
    iso = d.isoformat()
    n = _blocks_available(table, d)
    if n <= 0:
        return pd.DataFrame()
    blocks = np.arange(1, n + 1)
    if table in ("load", "sldc"):
        vals = _actual_load(state, iso)
        if table == "sldc":  # SLDC metering differs very slightly from the load table
            vals = np.round(vals * (1 + np.random.default_rng(_seed(state, iso, "sldc")).normal(0, 0.002, BLOCKS)), 2)
        return pd.DataFrame({"state": state, "date": iso, "time_block": blocks, "total_drawal": vals[:n]})
    if table == "forecast":
        return pd.DataFrame({"state": state, "date": iso, "block": blocks, "forecast_mw": _forecast_day(state, iso, "fc", 0.015, 0.010)})
    if table == "sldc_forecast":
        return pd.DataFrame({"state": state, "date": iso, "time_block": blocks, "forecast_mw": _forecast_day(state, iso, "sfc", 0.022, 0.012)})
    if table == "weather_mean":
        wx = _weather_mean_day(state, iso)
        return pd.DataFrame({"state": state, "date": iso, "time_block": blocks, **{k: np.round(wx[k].astype(float), 2) for k in WEATHER_COLS}})
    if table == "weather_loc":
        names = sorted(LOCATIONS[state])
        wx = _weather_day(state, iso)
        sel = [i for i, nme in enumerate(names) if location is None or nme == str(location).strip().upper()]
        if not sel:
            return pd.DataFrame()
        k_sel = len(sel)
        return pd.DataFrame({
            "state": state,
            "date": iso,
            "time_block": np.tile(blocks, k_sel),
            "location": np.repeat(np.array(names, dtype=object)[sel], n),
            **{k: np.round(wx[k][sel, :n].astype(np.float64), 2).ravel() for k in WEATHER_COLS},
        })
    raise KeyError(f"unknown mock table {table!r}")


def _frames(state: str, table: str, days: Iterable[date], location: str | None = None) -> Iterator[pd.DataFrame]:
    for d in days:
        f = _day_frame(state, table, d, location)
        if not f.empty:
            yield f


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Public data API ──────────────────────────────────────────────────────────
def table_rows(table: str, state: object, date=None, from_date=None, to_date=None,
               limit: int | None = 200, location: str | None = None) -> list[dict]:
    """Rows shaped like the old MySQL tables, ordered by date then block (ascending)."""
    st = _norm_state(state)
    out: list[dict] = []
    for f in _frames(st, table, _days(st, table, date, from_date, to_date), location):
        out.extend(f.to_dict("records"))
        if limit is not None and len(out) >= limit:
            return out[:limit]
    return out


def table_count(table: str, state: object) -> int:
    st = _norm_state(state)
    days = _days(st, table)
    per_day = BLOCKS * (len(LOCATIONS[st]) if table == "weather_loc" else 1)
    full = len(days) * per_day
    if table in ("load", "sldc") and days and days[-1] == today_ist():
        full -= BLOCKS - settled_blocks_today()
    return full


def load_frame(state: object, from_date=None, to_date=None, table: str = "load") -> pd.DataFrame:
    st = _norm_state(state)
    return _concat(_frames(st, table, _days(st, table, None, from_date, to_date)))


def weather_loc_frame(state: object, from_date=None, to_date=None) -> pd.DataFrame:
    """All districts, long format; built from stacked arrays (one DataFrame, not one per day)."""
    st = _norm_state(state)
    days = _days(st, "weather_loc", None, from_date, to_date)
    if not days:
        return pd.DataFrame()
    names = np.array(sorted(LOCATIONS[st]), dtype=object)
    per_day = len(names) * BLOCKS
    cols = {k: np.empty(len(days) * per_day, dtype=np.float32) for k in WEATHER_COLS}
    for j, d in enumerate(days):
        wx = _weather_day(st, d.isoformat())
        for k in WEATHER_COLS:
            cols[k][j * per_day:(j + 1) * per_day] = wx[k].ravel()
    return pd.DataFrame({
        "state": st,
        "date": np.repeat(np.array([d.isoformat() for d in days], dtype=object), per_day),
        "time_block": np.tile(np.arange(1, BLOCKS + 1), len(days) * len(names)),
        "location": np.tile(np.repeat(names, BLOCKS), len(days)),
        **{k: np.round(v.astype(np.float64), 2) for k, v in cols.items()},
    })


def weather_mean_frame(state: object, from_date=None, to_date=None) -> pd.DataFrame:
    st = _norm_state(state)
    return _concat(_frames(st, "weather_mean", _days(st, "weather_mean", None, from_date, to_date)))


def forecast_frame(state: object, from_date=None, to_date=None, table: str = "forecast") -> pd.DataFrame:
    st = _norm_state(state)
    return _concat(_frames(st, table, _days(st, table, None, from_date, to_date)))
