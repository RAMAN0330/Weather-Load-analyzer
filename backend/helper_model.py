"""
Encoder-decoder LSTM with Bahdanau attention for 96-block day-ahead load forecasting.
Rotated-day frame: block 33 (08:00 IST) = rotated day start, avoiding the midnight seam.

Full feature set:
  Encoder (HIST): load residual + baseline + y + load lags + 10 weather raw +
                  8 weather deltas + 7 thermal/solar + 8 time + 6 holiday  = 46
  Decoder (FUT):  baseline + 10 weather raw + 8 weather deltas + 6 thermal/solar +
                  8 time + 4 holiday                                         = 37

Hypertune v2:
  - precip_severity + wind_severity features (extreme-weather response)
  - Post-inference sanity clamp (blocks >2.5σ from K-day baseline pulled back)
  - Drastic-day flag persisted for T+2 restabilization
  - Heavier sample weights for rain/gusty-wind days
  - DERIV_LAMBDA 0.3→0.45 (smoother peak-block curves)
  - Schema-version check forces retrain when features change

Entry point:  run_helper_forecast(df, target_date, region, ...)
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional  # noqa: F401 — kept for any external callers that import it

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── TF import ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras import callbacks as C
    from tensorflow.keras import layers as L
    from tensorflow.keras import optimizers as O
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 96
HORIZON      = 96
FORECAST_DAYS = 2          # K=2: decoder forecasts T+1 and T+2 simultaneously
EPOCHS       = 40
PATIENCE     = 10
BATCH_SIZE   = 16
SEED         = 42
BASELINE_K   = 2
_BASELINE_POOL = 7      # candidate window when anomaly is detected
_OUTLIER_SIGMA = 1.8    # days deviating >1.8σ (daily mean) are treated as anomalous
DERIV_LAMBDA = 0.45       # ↑ from 0.3 — smoother peak-block curves (blocks 60-90)
WT_TEMP      = 0.8
WT_PRECIP    = 1.8        # ↑ from 1.2 — heavier weight on rainy training days
WT_WIND      = 1.5        # NEW — weight for gusty-wind days
WT_SNOW      = 2.5        # NEW — weight for snowfall days (rare in Haryana, impactful)
WT_SUNDAY    = 0.8
WT_HOLIDAY   = 1.2
W_MAX        = 6.0        # ↑ from 5.0 — allow higher weights for drastic-weather days
START_BLOCK  = 33        # 08:00 IST

# Sanity-clamp thresholds for post-inference correction
_CLAMP_SIGMA      = 2.5   # start pulling back when deviation exceeds 2.5σ from K-day baseline
_CLAMP_MAX_PULL   = 0.70  # at extreme deviation, pull up to 70% toward K-day mean
_DRASTIC_SCORE_THR= 0.55  # mean per-block σ-deviation above which we flag a day as drastic
# Schema version — bump this string whenever HIST_COLS or FUT_COLS change dimensions
_SCHEMA_VERSION   = "v3_48x39"

HP = dict(
    enc_units=256, enc_layers=2,
    enc_activation="tanh", enc_recurrent_activation="sigmoid",
    dec_units=256, attn_units=256,
    exog_embed=64, exog_activation="relu",
    dec_activation="tanh", dec_recurrent_activation="sigmoid",
    dropout=0.1, lr=3e-4,
)

# Weather columns pulled from the pipeline df
_WX_RAW = [
    "temperature", "humidity",
    "rain", "showers", "snowfall",
    "apparent_temperature", "cloud_cover",
    "sunshine_duration", "direct_radiation", "wind_speed_10m",
]
# Deltas: today vs same block yesterday (column names produced by _weather_deltas())
_WX_DELTA = [
    "temp_delta", "humidity_delta", "rain_delta",
    "apparent_delta", "cloud_delta", "sun_delta", "rad_delta", "wind_delta",
]
_THERMAL = ["CDD", "HDD", "wbgt", "cdh_24h", "temp_momentum_3d", "solar_index",
            "precip_severity", "wind_severity",
            "is_extreme_heat", "is_extreme_cold"]
# Load lags (MW domain, scaled separately)
_LOAD_LAGS = ["lag_1d", "lag_7d", "load_rolling_7d"]
# Time
_TIME = ["rot_block_sin", "rot_block_cos", "dow_sin", "dow_cos",
         "is_sunday", "is_weekend", "is_peak_hour", "is_night"]
# Holiday
_HOL_HIST = ["is_holiday", "bridge_day", "long_weekend",
             "days_to_holiday", "days_since_holiday", "mid_week_holiday"]
_HOL_FUT  = ["is_holiday", "bridge_day", "long_weekend", "mid_week_holiday"]

_HIST_COLS = (["residual", "baseline", "y"]
              + _LOAD_LAGS
              + _WX_RAW + _WX_DELTA
              + _THERMAL
              + _TIME + _HOL_HIST)          # 48 total

_FUT_COLS  = (["baseline"]
              + _WX_RAW + _WX_DELTA
              + ["CDD", "HDD", "wbgt", "solar_index", "precip_severity", "wind_severity",
                 "is_extreme_heat", "is_extreme_cold"]
              + _TIME + _HOL_FUT)           # 39 total

# Columns scaled by the load MinMaxScaler
_LOAD_SCALE_COLS = ["y", "baseline", "lag_1d", "lag_7d", "load_rolling_7d"]
# Columns scaled by the exog StandardScaler (everything continuous except binary/cyclical)
_EXOG_SCALE_COLS = (_WX_RAW
                    + ["CDD", "HDD", "wbgt", "cdh_24h", "temp_momentum_3d", "solar_index",
                       "precip_severity", "wind_severity"]
                    + ["days_to_holiday", "days_since_holiday"])


@dataclass
class _Artifacts:
    load_scaler: object
    exog_scaler: object
    schema_version: str = ""   # _SCHEMA_VERSION at training time; mismatch → retrain


# ── Indian holidays ────────────────────────────────────────────────────────────
_HOL_FIXED = {
    2023: [(1,26),(3,8),(3,30),(4,4),(4,6),(4,14),(4,21),(5,1),(6,29),(8,15),(8,31),
           (9,19),(10,2),(10,24),(11,12),(11,13),(12,25)],
    2024: [(1,26),(3,8),(3,25),(3,29),(4,11),(4,14),(4,17),(5,1),(6,17),(8,15),(8,19),
           (9,7),(10,2),(10,31),(11,1),(11,15),(12,25)],
    2025: [(1,26),(3,8),(3,14),(4,10),(4,13),(4,18),(5,1),(6,6),(8,15),(8,27),(9,27),
           (10,2),(10,20),(11,4),(12,25)],
    2026: [(1,26),(3,8),(3,21),(4,3),(4,5),(4,14),(5,1),(5,27),(8,15),(9,16),(10,2),
           (10,9),(10,20),(11,4),(11,23),(12,25)],
}

def _holidays_set() -> set:
    h = set()
    for yr in range(2020, 2031):
        for mo, dy in [(1,26),(8,15),(10,2),(12,25)]:
            h.add(pd.Timestamp(yr, mo, dy))
    for yr, dates in _HOL_FIXED.items():
        for mo, dy in dates:
            h.add(pd.Timestamp(yr, mo, dy))
    return h


# ── Rotated-day helpers ────────────────────────────────────────────────────────
def _rot_anchor(rot_date, sb: int = START_BLOCK) -> pd.Timestamp:
    return pd.Timestamp(rot_date) + pd.Timedelta(minutes=(sb - 1) * 15)

def _assign_rot_date(idx: pd.DatetimeIndex, sb: int = START_BLOCK) -> pd.Series:
    blk  = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    base = idx.normalize()
    return pd.Series(base - pd.to_timedelta((blk < sb).astype(int), unit="D"), index=idx)

def _rot_block_num(idx: pd.DatetimeIndex, sb: int = START_BLOCK) -> np.ndarray:
    blk = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    return ((blk - sb) % 96) + 1

def _rot_block_from_idx(idx: pd.DatetimeIndex, sb: int = START_BLOCK) -> np.ndarray:
    blk = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    return ((blk - sb) % 96) + 1


# ── Feature engineering ────────────────────────────────────────────────────────

def _time_features(idx: pd.DatetimeIndex, sb: int = START_BLOCK) -> pd.DataFrame:
    rot_date  = _assign_rot_date(idx, sb)
    rot_block = _rot_block_num(idx, sb)
    theta_b   = 2 * np.pi * (rot_block - 1) / 96.0
    dow       = pd.DatetimeIndex(rot_date).dayofweek.astype(int)
    theta_d   = 2 * np.pi * dow / 7.0
    hour      = idx.hour
    return pd.DataFrame({
        "rot_block_sin": np.sin(theta_b),
        "rot_block_cos": np.cos(theta_b),
        "dow_sin":       np.sin(theta_d),
        "dow_cos":       np.cos(theta_d),
        "is_sunday":     (dow == 6).astype(int),
        "is_weekend":    (dow >= 5).astype(int),
        "is_peak_hour":  hour.isin([7,8,9,18,19,20]).astype(int),
        "is_night":      hour.isin([0,1,2,3,4,5,6]).astype(int),
    }, index=idx)


def _holiday_features(idx: pd.DatetimeIndex, sb: int = START_BLOCK) -> pd.DataFrame:
    """Per-block holiday flags derived from the rotated-date."""
    holidays = _holidays_set()
    rot_date_norm = pd.DatetimeIndex(_assign_rot_date(idx, sb)).normalize()
    unique_dates  = sorted(set(rot_date_norm.date))

    flags = {}
    for d in unique_dates:
        ts  = pd.Timestamp(d)
        dow = ts.dayofweek
        is_hol  = ts in holidays
        is_wknd = dow >= 5
        prev    = ts - pd.Timedelta(days=1)
        nxt     = ts + pd.Timedelta(days=1)

        bridge      = (not is_hol and not is_wknd
                       and ((prev in holidays or prev.dayofweek >= 5)
                            and (nxt  in holidays or nxt.dayofweek  >= 5)))
        long_wknd   = is_wknd and (prev in holidays or nxt in holidays)
        mid_wk_hol  = is_hol and 1 <= dow <= 3

        days_to   = next((n for n in range(1, 30) if (ts + pd.Timedelta(days=n)) in holidays), 999)
        days_since= next((n for n in range(1, 30) if (ts - pd.Timedelta(days=n)) in holidays), 999)

        flags[ts] = dict(
            is_holiday        = int(is_hol),
            bridge_day        = int(bridge),
            long_weekend      = int(long_wknd),
            mid_week_holiday  = int(mid_wk_hol),
            days_to_holiday   = float(days_to),
            days_since_holiday= float(days_since),
        )

    rows = [flags.get(ts, {k:0 for k in ["is_holiday","bridge_day","long_weekend",
                                          "mid_week_holiday","days_to_holiday","days_since_holiday"]})
            for ts in rot_date_norm]
    return pd.DataFrame(rows, index=idx)


def _thermal_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """CDD, HDD, wbgt, cdh_24h, temp_momentum_3d, solar_index + precip_severity + wind_severity."""
    T  = df["temperature"].fillna(25.0)
    H  = df["humidity"].fillna(50.0)
    R  = df.get("direct_radiation", pd.Series(0.0, index=df.index)).fillna(0.0)
    CC = df.get("cloud_cover",      pd.Series(0.0, index=df.index)).fillna(0.0)

    cdd = np.maximum(0.0, T - 24.0)
    hdd = np.maximum(0.0, 15.0 - T)

    # Stull (2011) wet-bulb approximation
    wb = (T * np.arctan(0.151977 * np.sqrt(H + 8.313659))
          + np.arctan(T + H)
          - np.arctan(H - 1.676331)
          + 0.00391838 * H**1.5 * np.arctan(0.023101 * H)
          - 4.686035)
    wbgt = 0.7 * wb + 0.3 * T

    cdh_24h = cdd.rolling(96, min_periods=1).sum()
    temp_mom_3d = T - T.shift(96 * 3)

    max_rad = max(float(R.max()), 1.0)
    solar_idx = (R / max_rad) * (1.0 - CC.clip(0, 100) / 100.0)

    # Precipitation severity: log-scale composite of rain, showers, snowfall.
    # Snowfall weighted 5× (rare but high-impact in North India).
    RAIN = df.get("rain",     pd.Series(0.0, index=df.index)).fillna(0.0)
    SHOW = df.get("showers",  pd.Series(0.0, index=df.index)).fillna(0.0)
    SNOW = df.get("snowfall", pd.Series(0.0, index=df.index)).fillna(0.0)
    precip_raw = RAIN + SHOW * 1.5 + SNOW * 5.0
    precip_severity = np.log1p(precip_raw) / np.log1p(10.0)          # 0–1 normalised
    precip_severity = precip_severity.clip(0.0, 1.0)

    # Wind severity: excess above 15 km/h comfort threshold, normalised to 1 at 45 km/h.
    WIND = df.get("wind_speed_10m", pd.Series(0.0, index=df.index)).fillna(0.0)
    wind_severity = np.clip((WIND - 15.0) / 30.0, 0.0, 1.0)

    return pd.DataFrame({
        "CDD":              cdd,
        "HDD":              hdd,
        "wbgt":             wbgt,
        "cdh_24h":          cdh_24h,
        "temp_momentum_3d": temp_mom_3d,
        "solar_index":      solar_idx,
        "precip_severity":  precip_severity,
        "wind_severity":    wind_severity,
        "is_extreme_heat":  (T >= 42.0).astype(float),
        "is_extreme_cold":  (T <= 8.0).astype(float),
    }, index=df.index)


def _weather_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Same-block delta vs previous calendar day (96-period shift)."""
    src = {
        "temp_delta":     "temperature",
        "humidity_delta": "humidity",
        "rain_delta":     "rain",
        "apparent_delta": "apparent_temperature",
        "cloud_delta":    "cloud_cover",
        "sun_delta":      "sunshine_duration",
        "rad_delta":      "direct_radiation",
        "wind_delta":     "wind_speed_10m",
    }
    out = {}
    for dst, col in src.items():
        if col in df.columns:
            out[dst] = df[col] - df[col].shift(96)
        else:
            out[dst] = pd.Series(0.0, index=df.index)
    return pd.DataFrame(out, index=df.index)


def _day_anomaly_mask(series: pd.Series, blocks_per_day: int = 96,
                      pool: int = 14, sigma: float = _OUTLIER_SIGMA) -> pd.Series:
    """Return a boolean Series (same index) — True where a 96-block day is anomalous.

    A day is anomalous when its mean load deviates more than `sigma` standard
    deviations from the rolling `pool`-day mean (computed from prior days only,
    so the mask is causal and can be applied during training without leakage).
    """
    daily_mean = series.groupby(series.index // blocks_per_day).transform("mean")
    rolling_mu  = daily_mean.shift(blocks_per_day).rolling(pool * blocks_per_day, min_periods=blocks_per_day).mean()
    rolling_std = daily_mean.shift(blocks_per_day).rolling(pool * blocks_per_day, min_periods=blocks_per_day).std()
    with_std = rolling_std.fillna(0)
    return (daily_mean - rolling_mu).abs() > sigma * with_std.clip(lower=1.0)


def _load_lags(series: pd.Series) -> pd.DataFrame:
    """lag_1d, lag_7d, load_rolling_7d — with anomalous T-1 days replaced by T-7."""
    lag1  = series.shift(96)
    lag7  = series.shift(672)
    roll7 = lag1.rolling(672, min_periods=1).mean()

    # Where yesterday was anomalous, substitute lag_7d so the bad day doesn't
    # propagate as a direct feature input.
    integer_idx = pd.RangeIndex(len(series))
    lag1_daily  = lag1.reset_index(drop=True)
    lag7_daily  = lag7.reset_index(drop=True)

    # Build a positional day-mean series for the shifted-by-1-day values
    blocks       = 96
    day_pos      = integer_idx // blocks
    day_means_l1 = lag1_daily.groupby(day_pos).transform("mean")
    day_means_l7 = lag7_daily.groupby(day_pos).transform("mean")

    # Rolling reference mean from past 14 days (in lag1 space)
    ref_mu  = day_means_l1.rolling(14 * blocks, min_periods=blocks).mean()
    ref_std = day_means_l1.rolling(14 * blocks, min_periods=blocks).std().fillna(0).clip(lower=1.0)
    anomalous_day = (day_means_l1 - ref_mu).abs() > _OUTLIER_SIGMA * ref_std

    lag1_clean = lag1_daily.where(~anomalous_day, other=lag7_daily)
    lag1_clean.index = series.index

    return pd.DataFrame({"lag_1d": lag1_clean, "lag_7d": lag7, "load_rolling_7d": roll7},
                        index=series.index)


# ── Causal baseline ────────────────────────────────────────────────────────────

def _causal_baseline(df_load: pd.DataFrame, K: int = BASELINE_K, sb: int = START_BLOCK) -> pd.Series:
    """Rolling K-day mean baseline with outlier days excluded from the window.

    Uses the same logic as _baseline_for_future so training and inference see
    a consistent baseline definition — critical to avoid train/inference mismatch.
    """
    df = df_load.copy()
    df["Datetime"] = df["Date"] + pd.to_timedelta((df["block"] - 1) * 15, unit="m")
    df = df.sort_values(["Date", "block"])
    first_day = df["Date"].min()
    df = df.loc[~((df["Date"] == first_day) & (df["block"] <= sb - 1))].copy()
    idx = pd.DatetimeIndex(df["Datetime"])
    df["rot_date"]  = _assign_rot_date(idx, sb).to_numpy()
    df["rot_block"] = _rot_block_num(idx, sb)
    piv = df.pivot_table(index="rot_date", columns="rot_block",
                         values="total_drawal_adj", aggfunc="mean").sort_index()

    # Compute clean rolling baseline row-by-row with outlier filtering
    day_means = piv.mean(axis=1)
    baselines = []
    for i, rot_day in enumerate(piv.index):
        pool_slice = piv.iloc[max(0, i - _BASELINE_POOL):i]
        if len(pool_slice) == 0:
            baselines.append(pd.Series([float("nan")] * piv.shape[1], index=piv.columns))
            continue
        pm = day_means.iloc[max(0, i - _BASELINE_POOL):i]
        if len(pm) >= 3:
            mu, sigma = pm.mean(), pm.std()
            if sigma > 0:
                clean = pool_slice.loc[(pm - mu).abs() <= _OUTLIER_SIGMA * sigma]
                pool_slice = clean if len(clean) >= 2 else pool_slice
        hist = pool_slice.tail(K)
        baselines.append(hist.mean(axis=0))

    base_rot = pd.DataFrame(baselines, index=piv.index, columns=piv.columns)
    bl = base_rot.stack(dropna=False).rename("baseline").reset_index()
    bl["Datetime"] = (pd.to_datetime(bl["rot_date"]).apply(lambda d: _rot_anchor(d, sb))
                      + pd.to_timedelta((bl["rot_block"] - 1) * 15, unit="m"))
    return bl.set_index("Datetime")["baseline"].sort_index()


def _baseline_for_future(df_hist: pd.DataFrame, rot_day: pd.Timestamp,
                          K: int = BASELINE_K, sb: int = START_BLOCK) -> pd.Series:
    df = df_hist.copy()
    df["Datetime"] = df["Date"] + pd.to_timedelta((df["block"] - 1) * 15, unit="m")
    ser  = df.set_index("Datetime")["total_drawal_adj"].sort_index()
    idx  = ser.index
    rdat = _assign_rot_date(idx, sb).values
    rblk = _rot_block_from_idx(idx, sb)
    piv  = (pd.DataFrame({"rot_date": rdat, "rot_block": rblk, "y": ser.values})
            .pivot_table(index="rot_date", columns="rot_block", values="y", aggfunc="mean")
            .sort_index())
    candidates = piv.loc[piv.index < pd.Timestamp(rot_day)]
    # Pull a larger pool and drop anomalous days so a drastic T-1 doesn't poison the baseline
    pool = candidates.tail(_BASELINE_POOL)
    if len(pool) >= 3:
        day_means = pool.mean(axis=1)
        mu, sigma = day_means.mean(), day_means.std()
        if sigma > 0:
            clean = pool.loc[(day_means - mu).abs() <= _OUTLIER_SIGMA * sigma]
            pool = clean if len(clean) >= 2 else pool
    hist = pool.tail(K) if len(pool) >= K else pool
    if hist.empty:
        hist = piv.tail(1)
    vec       = hist.mean(axis=0)
    vec.index = pd.RangeIndex(1, 97)
    start_t   = _rot_anchor(rot_day, sb)
    return pd.Series(vec.values, index=pd.date_range(start_t, periods=96, freq="15T"), name="baseline")


# ── Full feature stream ────────────────────────────────────────────────────────

def _build_state_series(df_load: pd.DataFrame, state_exog: pd.DataFrame,
                         sb: int = START_BLOCK) -> pd.DataFrame:
    df = df_load.copy()
    df["Datetime"] = df["Date"] + pd.to_timedelta((df["block"] - 1) * 15, unit="m")
    df = df.set_index("Datetime").sort_index()
    first_day = df["Date"].min()
    df = df.loc[~((df["Date"] == first_day) & (df["block"] <= sb - 1))].copy()

    # Align all weather onto the load timeline
    wx = state_exog.reindex(df.index).ffill().bfill()
    for c in _WX_RAW:
        if c not in wx.columns:
            wx[c] = 0.0

    # Derived weather
    deltas  = _weather_deltas(wx)
    thermal = _thermal_solar_features(wx)

    # Load lags
    lags = _load_lags(df["total_drawal_adj"])

    # Causal baseline
    base_rot = _causal_baseline(
        df[["Date", "block", "total_drawal_adj"]].reset_index(drop=False),
        K=BASELINE_K, sb=sb
    ).reindex(df.index).ffill().bfill()
    resid = df["total_drawal_adj"] - base_rot

    # Time + holiday features
    tfeat = _time_features(df.index, sb)
    hfeat = _holiday_features(df.index, sb)

    full = pd.concat([
        df[["total_drawal_adj"]].rename(columns={"total_drawal_adj": "y"}),
        base_rot.rename("baseline"),
        resid.rename("residual"),
        lags,
        wx[_WX_RAW],
        deltas,
        thermal,
        tfeat,
        hfeat,
    ], axis=1)

    return full


# ── Windowing ─────────────────────────────────────────────────────────────────

def _make_windows(full: pd.DataFrame, sb: int = START_BLOCK):
    full = full.copy().sort_index()
    full["rot_date"] = _assign_rot_date(full.index, sb).to_numpy()
    ok_dates = set(full["rot_date"].value_counts()[lambda s: s >= 92].index)
    full = full[full["rot_date"].isin(ok_dates)].sort_index()
    Xh, Xf, Y, Yin, stamps = [], [], [], [], []
    total_steps = HORIZON * FORECAST_DAYS  # 192 for K=2
    for d in sorted(pd.unique(full["rot_date"])):
        t0         = _rot_anchor(d, sb)
        hist_start = t0 - pd.to_timedelta(SEQ_LEN * 15, unit="m")
        hist_sl    = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_sl     = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15 * total_steps))]
        if len(hist_sl) < int(0.9 * SEQ_LEN) or len(fut_sl) < int(0.95 * total_steps):
            continue
        hist_sl = hist_sl.tail(SEQ_LEN)
        fut_sl  = fut_sl.head(total_steps)
        if len(hist_sl) != SEQ_LEN or len(fut_sl) != total_steps:
            continue
        y_tgt       = fut_sl[["residual"]].values.astype(np.float32)
        y_in        = np.empty_like(y_tgt)
        y_in[0, 0]  = hist_sl["residual"].iloc[-1]
        y_in[1:, 0] = y_tgt[:-1, 0]
        Xh.append(hist_sl[_HIST_COLS].values.astype(np.float32))
        Xf.append(fut_sl[_FUT_COLS].values.astype(np.float32))
        Y.append(y_tgt)
        Yin.append(y_in)
        stamps.append(t0)
    return np.asarray(Xh), np.asarray(Xf), np.asarray(Y), np.asarray(Yin), pd.to_datetime(stamps)


def _sample_weights(full: pd.DataFrame, n_samples: int, sb: int = START_BLOCK) -> np.ndarray:
    full = full.copy().sort_index()
    full["rot_date"] = _assign_rot_date(full.index, sb).to_numpy()
    ok_dates = set(full["rot_date"].value_counts()[lambda s: s == 96].index)
    full = full[full["rot_date"].isin(ok_dates)].sort_index()
    W = []
    total_steps = HORIZON * FORECAST_DAYS
    for d in sorted(pd.unique(full["rot_date"])):
        t0         = _rot_anchor(d, sb)
        hist_start = t0 - pd.to_timedelta(SEQ_LEN * 15, unit="m")
        hist_sl    = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_sl     = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15 * total_steps))]
        if len(hist_sl) != SEQ_LEN or len(fut_sl) != total_steps:
            continue
        T    = fut_sl["temperature"].to_numpy()
        RAIN = fut_sl.get("rain",     pd.Series(0.0, index=fut_sl.index)).to_numpy()
        SHOW = fut_sl.get("showers",  pd.Series(0.0, index=fut_sl.index)).to_numpy()
        SNOW = fut_sl.get("snowfall", pd.Series(0.0, index=fut_sl.index)).to_numpy()
        WIND = fut_sl.get("wind_speed_10m", pd.Series(0.0, index=fut_sl.index)).to_numpy()
        P    = np.maximum(0.0, RAIN + SHOW * 1.5)    # rain composite
        dT   = np.r_[0.0, np.diff(T)]
        wind_excess = np.maximum(0.0, WIND - 15.0) / 30.0   # 0–1 at 45 km/h
        w  = (1.0
              + WT_TEMP    * np.abs(dT)
              + WT_PRECIP  * np.log1p(P)              # log-saturate precip
              + WT_WIND    * wind_excess
              + WT_SNOW    * np.minimum(1.0, SNOW / 2.0)  # 1 at 2 mm snowfall
              + WT_SUNDAY  * fut_sl["is_sunday"].to_numpy()
              + WT_HOLIDAY * fut_sl["is_holiday"].to_numpy())
        W.append(np.clip(w, 1.0, W_MAX).astype("float32"))
    W_arr = np.asarray(W, dtype="float32")
    if W_arr.shape[0] != n_samples:
        return np.ones((n_samples, HORIZON * FORECAST_DAYS), dtype="float32")
    return W_arr


# ── Diurnal day-type adjustment ───────────────────────────────────────────────

def _diurnal_shape_correction(df_load: pd.DataFrame, t0_day: pd.Timestamp,
                               yhat_mw: np.ndarray, n_act: int,
                               n_weeks: int = 8) -> np.ndarray:
    """Return a 96-element shape-correction multiplier for blocks n_act..95.

    Instead of applying the absolute historical shape (which would over-correct
    because Haryana mornings are high due to agricultural night load), this
    compares the model's own afternoon-to-morning ratio against the historical
    same-day-type ratio and only corrects the *difference*.

    correction[b] = historical_ratio_daytype[b] / model_ratio[b]

    where ratio[b] = block_b / mean(blocks 0..n_act-1).

    If the model already captures the shape correctly, correction ≈ 1.0.
    On weekdays the commercial afternoon uplift is added; on weekends it is
    not, so the forecast naturally stays flatter.

    Falls back to ones if there is insufficient history.
    """
    is_weekend = t0_day.dayofweek >= 5

    df = df_load.copy()
    df["Datetime"] = df["Date"] + pd.to_timedelta((df["block"] - 1) * 15, unit="m")
    piv = (df.set_index("Datetime")["total_drawal_adj"]
           .reset_index()
           .assign(date=lambda x: x["Datetime"].dt.normalize(),
                   blk=lambda x: (x["Datetime"].dt.hour * 4
                                  + x["Datetime"].dt.minute // 15))
           .pivot_table(index="date", columns="blk",
                        values="total_drawal_adj", aggfunc="mean")
           .sort_index())

    prior = piv.loc[piv.index < pd.Timestamp(t0_day)]
    mask  = (prior.index.dayofweek >= 5) == is_weekend
    same  = prior.loc[mask].tail(n_weeks)

    if len(same) < 3 or n_act < 4:
        return np.ones(96)

    morning_cols = [c for c in same.columns if c < n_act]
    if not morning_cols:
        return np.ones(96)

    # Historical shape: each block normalised by the morning mean of that day
    hist_morning_mean = same[morning_cols].mean(axis=1).replace(0, np.nan)
    hist_ratios  = same.div(hist_morning_mean, axis=0)
    hist_shape   = hist_ratios.mean(axis=0).reindex(range(96)).ffill().bfill().to_numpy()

    # Model shape: same normalisation applied to the raw forecast
    model_morning_mean = float(np.mean(yhat_mw[:n_act])) or 1.0
    model_shape  = yhat_mw / model_morning_mean   # shape (96,)

    # Shape correction: where model under/over-represents the afternoon relative
    # to historical same-day-type pattern
    shape_corr = np.ones(96)
    for b in range(n_act, 96):
        ms = model_shape[b]
        hs = hist_shape[b]
        if ms > 0:
            shape_corr[b] = hs / ms
        # else leave as 1.0

    # Clamp: don't allow the shape correction to swing more than ±20%
    shape_corr = np.clip(shape_corr, 0.80, 1.20)
    shape_corr[:n_act] = 1.0
    return shape_corr


# ── Post-inference sanity clamp & drastic-day utilities ──────────────────────

def _kday_block_stats(df_load: pd.DataFrame, t0_day: pd.Timestamp,
                      n_days: int = 14, sb: int = START_BLOCK):
    """Return (k_mean_96, k_std_96) in calendar-block order for the K prior rotated days."""
    df = df_load.copy()
    df["Datetime"] = df["Date"] + pd.to_timedelta((df["block"] - 1) * 15, unit="m")
    ser = df.set_index("Datetime")["total_drawal_adj"].sort_index()
    idx = ser.index
    rdat = _assign_rot_date(idx, sb).values
    rblk = _rot_block_from_idx(idx, sb)
    piv = (pd.DataFrame({"rot_date": rdat, "rot_block": rblk, "y": ser.values})
           .pivot_table(index="rot_date", columns="rot_block", values="y", aggfunc="mean")
           .sort_index())
    hist = piv.loc[piv.index < pd.Timestamp(t0_day)].tail(n_days)
    if len(hist) < 3:
        return None, None

    # Convert rotated-frame stats to calendar-block order
    start_rot = _rot_anchor(t0_day, sb)
    rot_idx   = pd.date_range(start=start_rot, periods=HORIZON, freq="15T")
    cal_blks  = ((rot_idx.hour * 60 + rot_idx.minute) // 15 + 1).to_numpy()

    k_mean_cal = np.full(96, np.nan)
    k_std_cal  = np.full(96, np.nan)
    for i, cb in enumerate(cal_blks):
        rb = i + 1   # rotated block index (1-based)
        if 1 <= cb <= 96 and rb in hist.columns:
            k_mean_cal[cb - 1] = float(hist[rb].mean())
            k_std_cal[cb - 1]  = max(float(hist[rb].std(ddof=0)), 50.0)

    # Fill any NaN with global stats
    gm = np.nanmean(k_mean_cal)
    gs = max(np.nanmean(k_std_cal), 50.0)
    k_mean_cal = np.where(np.isfinite(k_mean_cal), k_mean_cal, gm)
    k_std_cal  = np.where(np.isfinite(k_std_cal),  k_std_cal,  gs)
    return k_mean_cal, k_std_cal


def _post_inference_clamp(yhat_mw: np.ndarray, df_load: pd.DataFrame,
                           t0_day: pd.Timestamp, n_days: int = 14,
                           sigma_start: float = _CLAMP_SIGMA,
                           max_pull: float = _CLAMP_MAX_PULL,
                           exempt_blocks: set = None) -> np.ndarray:
    """
    Progressive pull-back for hallucinated peaks.
    At sigma_start deviation: 0% correction.
    At 2×sigma_start deviation: max_pull×100% correction toward K-day mean.

    exempt_blocks: 1-based block numbers to skip clamping entirely (weather-flagged
    blocks where a genuine load shift should not be reversed).
    sigma_start: pass WeatherAnomalyResult.clamp_sigma_override for weather-aware
    tolerance widening (default 2.5, up to 5.0 on CRITICAL days).
    """
    k_mean, k_std = _kday_block_stats(df_load, t0_day, n_days)
    if k_mean is None:
        return yhat_mw

    _exempt = exempt_blocks or set()
    out = yhat_mw.copy()
    for b in range(96):
        if (b + 1) in _exempt:
            continue   # genuine weather event — trust the model
        mu    = k_mean[b]
        sigma = k_std[b]
        dev   = out[b] - mu
        n_sig = abs(dev) / sigma
        if n_sig > sigma_start:
            blend = min(max_pull, (n_sig - sigma_start) / sigma_start * max_pull)
            out[b] = out[b] * (1.0 - blend) + mu * blend
    return np.maximum(out, 0.0)


def _drastic_score(yhat_mw: np.ndarray, df_load: pd.DataFrame,
                   t0_day: pd.Timestamp, n_days: int = 14) -> float:
    """Mean per-block σ-deviation vs K-day baseline. >0.55 → drastic."""
    k_mean, k_std = _kday_block_stats(df_load, t0_day, n_days)
    if k_mean is None:
        return 0.0
    valid = k_std > 0
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(yhat_mw[valid] - k_mean[valid]) / k_std[valid]))


def _mark_drastic(art_dir: Path, date_str: str, score: float) -> None:
    """Persist a drastic-day flag so the next T+2 forecast can restabilize."""
    try:
        flag = art_dir / "drastic_flags.json"
        data: dict = {}
        if flag.exists():
            data = json.loads(flag.read_text())
        data[date_str] = {"score": round(score, 4), "ts": time.time()}
        # Keep only last 14 entries
        if len(data) > 14:
            oldest = sorted(data, key=lambda k: data[k]["ts"])[:len(data) - 14]
            for k in oldest:
                del data[k]
        flag.write_text(json.dumps(data))
    except Exception:
        pass


def _check_prev_drastic(art_dir: Path, prev_date_str: str,
                         threshold: float = _DRASTIC_SCORE_THR) -> bool:
    """Return True if the previous forecast day was marked drastic."""
    try:
        flag = art_dir / "drastic_flags.json"
        if not flag.exists():
            return False
        data = json.loads(flag.read_text())
        entry = data.get(prev_date_str, {})
        return float(entry.get("score", 0.0)) >= threshold
    except Exception:
        return False


# ── Model ─────────────────────────────────────────────────────────────────────

class _BahdanauAttention(tf.keras.layers.Layer if _TF_AVAILABLE else object):
    def __init__(self, units, **kw):
        super().__init__(**kw)
        self.W1 = L.Dense(units, use_bias=False)
        self.W2 = L.Dense(units, use_bias=False)
        self.V  = L.Dense(1,    use_bias=False)

    def call(self, query, values):
        score   = tf.nn.tanh(self.W1(values) + self.W2(tf.expand_dims(query, 1)))
        weights = tf.nn.softmax(self.V(score), axis=1)
        return tf.reduce_sum(weights * values, axis=1), tf.squeeze(weights, -1)


class _ARDecoder(tf.keras.layers.Layer if _TF_AVAILABLE else object):
    def __init__(self, dec_units, exg_embed_dim, attn_units, dropout=0.0,
                 exog_activation="relu", dec_activation="tanh",
                 dec_recurrent_activation="sigmoid", **kw):
        super().__init__(**kw)
        self.dropout_rate = float(dropout)
        self.attn   = _BahdanauAttention(attn_units)
        self.ex_emb = L.Dense(exg_embed_dim, activation=exog_activation)
        self.cell   = L.LSTMCell(dec_units, dropout=0.0, recurrent_dropout=0.0,
                                  activation=dec_activation,
                                  recurrent_activation=dec_recurrent_activation)
        self.out    = L.Dense(1)

    def call(self, inputs, training=None):
        fut, y_teacher, enc_outs, s_h, s_c, hist_in, train_flag = inputs
        horizon = tf.shape(fut)[1]
        last_y  = tf.ensure_shape(hist_in[:, -1, 0:1], [None, 1])
        if isinstance(train_flag, bool):
            train_flag = tf.constant(train_flag, dtype=tf.bool)
        tgate = tf.reduce_all(tf.cast(train_flag, tf.bool))

        outputs = tf.TensorArray(tf.float32, size=horizon)
        t = tf.constant(0)

        def _cond(t, *_): return tf.less(t, horizon)

        def _body(t, last_y, s_h, s_c, outputs):
            ex_t      = self.ex_emb(fut[:, t, :])
            ctx_t, _  = self.attn(s_h, enc_outs)
            step_in   = tf.concat([last_y, ex_t, ctx_t], axis=-1)
            if self.dropout_rate > 0.0:
                step_in = tf.cond(tgate,
                                  lambda: tf.nn.dropout(step_in, rate=self.dropout_rate),
                                  lambda: step_in)
            out_t, [s_h, s_c] = self.cell(step_in, states=[s_h, s_c])
            y_t = tf.ensure_shape(self.out(out_t), [None, 1])
            next_y = tf.ensure_shape(
                tf.where(tgate, tf.ensure_shape(y_teacher[:, t, :], [None, 1]), y_t), [None, 1])
            return t + 1, next_y, s_h, s_c, outputs.write(t, y_t)

        _, _, _, _, outputs = tf.while_loop(_cond, _body, [t, last_y, s_h, s_c, outputs],
                                            parallel_iterations=1)
        return tf.ensure_shape(tf.transpose(outputs.stack(), [1, 0, 2]), [None, None, 1])

    def compute_output_shape(self, in_shape):
        return (in_shape[0][0], in_shape[0][1], 1)


def _build_model(hp: dict, Fh: int, Ff: int) -> "Model":
    eu  = int(hp.get("enc_units", 256))
    el  = int(hp.get("enc_layers", 2))
    du  = int(hp.get("dec_units", 256))
    au  = int(hp.get("attn_units", eu))
    ee  = int(hp.get("exog_embed", 64))
    dr  = float(hp.get("dropout", 0.1))
    lr  = float(hp.get("lr", 3e-4))
    eac = hp.get("enc_activation", "tanh")
    erc = hp.get("enc_recurrent_activation", "sigmoid")
    dac = hp.get("dec_activation", "tanh")
    drc = hp.get("dec_recurrent_activation", "sigmoid")
    xac = hp.get("exog_activation", "relu")

    dec_steps = HORIZON * FORECAST_DAYS   # 192 for K=2
    h_in  = L.Input(shape=(SEQ_LEN,    Fh), name="hist")
    f_in  = L.Input(shape=(dec_steps,  Ff), name="future_exog")
    y_in  = L.Input(shape=(dec_steps,   1), name="y_in")
    tr_in = L.Input(shape=(),  dtype=tf.bool, name="is_training")

    x = h_in
    for _ in range(el):
        x = L.LSTM(eu, return_sequences=True, activation=eac, recurrent_activation=erc)(x)
        x = L.Dropout(dr)(x)
    enc_out = x
    _, eh, ec = L.LSTM(eu, return_sequences=False, return_state=True,
                       activation=eac, recurrent_activation=erc)(enc_out)
    ih = L.Dense(du, activation="tanh")(eh)
    ic = L.Dense(du, activation="tanh")(ec)

    y_hat = _ARDecoder(dec_units=du, exg_embed_dim=ee, attn_units=au, dropout=dr,
                       exog_activation=xac, dec_activation=dac,
                       dec_recurrent_activation=drc)(
        [f_in, y_in, enc_out, ih, ic, h_in, tr_in]
    )

    model = Model(inputs=[h_in, f_in, y_in, tr_in], outputs=y_hat)
    huber = tf.keras.losses.Huber(delta=0.02, reduction=tf.keras.losses.Reduction.NONE)

    def _dloss(y_true, y_pred):
        d = tf.reduce_mean(tf.abs((y_true[:,1:,:] - y_true[:,:-1,:])
                                   - (y_pred[:,1:,:] - y_pred[:,:-1,:])), axis=-1)
        return tf.pad(d, [[0,0],[0,1]])

    model.compile(optimizer=O.Adam(lr),
                  loss=lambda yt, yp: huber(yt, yp) + DERIV_LAMBDA * _dloss(yt, yp),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


# ── Future exog window (target day) ───────────────────────────────────────────

def _future_exog_window(state_exog: pd.DataFrame, start_time,
                         last_hist_day_exog: pd.DataFrame,
                         steps: int = 96, sb: int = START_BLOCK) -> pd.DataFrame:
    """Build future exog for the target rotated window, including deltas vs last hist day."""
    idx = pd.date_range(start=pd.Timestamp(start_time), periods=steps, freq="15T")
    win = state_exog.reindex(idx)

    for c in _WX_RAW:
        if c not in win.columns:
            win[c] = 0.0
        win[c] = pd.to_numeric(win[c], errors="coerce")
    # interpolate gaps in weather forecast
    win[_WX_RAW] = win[_WX_RAW].interpolate(limit_direction="both")
    for c in ["rain", "showers", "snowfall"]:
        win[c] = win[c].fillna(0.0)

    # Deltas: target day wx minus last historical day wx (same block positions)
    src_map = {
        "temp_delta":     "temperature",
        "humidity_delta": "humidity",
        "rain_delta":     "rain",
        "apparent_delta": "apparent_temperature",
        "cloud_delta":    "cloud_cover",
        "sun_delta":      "sunshine_duration",
        "rad_delta":      "direct_radiation",
        "wind_delta":     "wind_speed_10m",
    }
    delta_vals = {}
    for dst, col in src_map.items():
        if col in win.columns and col in last_hist_day_exog.columns:
            ref = last_hist_day_exog[col].values
            cur = win[col].values
            n   = min(len(ref), len(cur))
            d   = np.zeros(len(cur))
            d[:n] = cur[:n] - ref[:n]
            delta_vals[dst] = d
        else:
            delta_vals[dst] = np.zeros(len(idx))
    deltas = pd.DataFrame(delta_vals, index=idx)

    # Thermal / solar
    thermal_full = _thermal_solar_features(win)
    thermal_fut  = thermal_full[["CDD", "HDD", "wbgt", "solar_index",
                                   "is_extreme_heat", "is_extreme_cold"]]

    # Time + holiday
    tfeat = _time_features(idx, sb)
    hfeat = _holiday_features(idx, sb)[_HOL_FUT]

    return pd.concat([win[_WX_RAW], deltas, thermal_fut, tfeat, hfeat], axis=1)


# ── Scale helpers ─────────────────────────────────────────────────────────────

def _apply_load_scale(full: pd.DataFrame, load_scaler: MinMaxScaler, fit: bool) -> pd.DataFrame:
    """Scale load MW columns in-place; fill NaN lag values before scaling."""
    cols = [c for c in _LOAD_SCALE_COLS if c in full.columns]
    if fit:
        # fit on y only (lag cols have NaN at start)
        load_scaler.fit(full["y"].to_numpy().reshape(-1, 1))
    for c in cols:
        vals = full[c].bfill().fillna(0.0).to_numpy().reshape(-1, 1)
        full[c] = load_scaler.transform(vals)[:, 0]
    full["residual"] = full["y"] - full["baseline"]
    return full


def _apply_exog_scale(full: pd.DataFrame, exog_scaler: StandardScaler, fit: bool) -> pd.DataFrame:
    cols = [c for c in _EXOG_SCALE_COLS if c in full.columns]
    sub  = full[cols].fillna(0.0)
    if fit:
        exog_scaler.fit(sub)
        full[cols] = exog_scaler.transform(sub)
        return full

    scaler_cols = list(getattr(exog_scaler, "feature_names_in_", cols))
    scaled = pd.DataFrame(
        exog_scaler.transform(full.reindex(columns=scaler_cols, fill_value=0.0).fillna(0.0)),
        index=full.index,
        columns=scaler_cols,
    )
    for col in [c for c in scaler_cols if c in full.columns]:
        full[col] = scaled[col]
    return full


# ── Train ─────────────────────────────────────────────────────────────────────

def _train(df_load: pd.DataFrame, state_exog: pd.DataFrame,
           artifacts_dir: Path, hp: dict = None):
    if hp is None:
        hp = HP
    if _TF_AVAILABLE:
        tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    full = _build_state_series(df_load, state_exog)

    load_scaler = MinMaxScaler()
    exog_scaler = StandardScaler()
    full = _apply_load_scale(full, load_scaler, fit=True)
    full = _apply_exog_scale(full, exog_scaler, fit=True)

    Xh, Xf, Y, Yin, _ = _make_windows(full)
    if len(Xh) < 10:
        raise ValueError(f"Only {len(Xh)} rotated samples — need ≥10 to train.")

    W  = _sample_weights(full, len(Xh))
    Fh = len(_HIST_COLS)
    Ff = len(_FUT_COLS)
    model = _build_model(hp, Fh=Fh, Ff=Ff)

    es = C.EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True)
    rl = C.ReduceLROnPlateau(monitor="loss", factor=0.5,
                              patience=max(2, PATIENCE // 2), min_lr=1e-5)
    model.fit(
        {"hist": Xh, "future_exog": Xf, "y_in": Yin,
         "is_training": np.ones(len(Xh), bool)},
        Y, sample_weight=W,
        epochs=EPOCHS, batch_size=max(1, BATCH_SIZE),
        callbacks=[es, rl], verbose=0,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(artifacts_dir / "model.weights.h5"))
    with open(artifacts_dir / "scalers.pkl", "wb") as f:
        pickle.dump(_Artifacts(load_scaler, exog_scaler, schema_version=_SCHEMA_VERSION), f)
    logger.info("Helper model trained → %s  (Fh=%d, Ff=%d, schema=%s)",
                artifacts_dir, Fh, Ff, _SCHEMA_VERSION)
    return model, load_scaler, exog_scaler


# ── Infer ─────────────────────────────────────────────────────────────────────

def _infer(df_load: pd.DataFrame, state_exog: pd.DataFrame, target_date,
           actual_blocks: int, actual_partial: np.ndarray,
           artifacts_dir: Path, prev_was_drastic: bool = False,
           clamp_sigma_override: float = None,
           correction_decay_override: float = None,
           exempt_blocks: set = None):

    with open(artifacts_dir / "scalers.pkl", "rb") as f:
        arts: _Artifacts = pickle.load(f)

    full2 = _build_state_series(df_load, state_exog)
    full2 = _apply_load_scale(full2, arts.load_scaler, fit=False)
    full2 = _apply_exog_scale(full2, arts.exog_scaler, fit=False)

    # Build encoder history slice
    eff_last   = max(1, min(int(actual_blocks) if actual_blocks > 0 else 32, 32))
    t0_day     = pd.Timestamp(str(target_date)[:10]).normalize()
    desired    = t0_day + pd.Timedelta(minutes=(eff_last - 1) * 15)
    hist_end   = full2.index[full2.index <= desired].max()
    if pd.isna(hist_end):
        raise ValueError("No history before target rotated window.")
    hist_start = hist_end - pd.to_timedelta(SEQ_LEN * 15 - 15, unit="m")
    hist_sl    = full2.loc[(full2.index >= hist_start) & (full2.index <= hist_end)]
    if len(hist_sl) != SEQ_LEN:
        hist_sl = hist_sl.reindex(pd.date_range(hist_start, hist_end, freq="15min")).ffill().bfill()
    if len(hist_sl) != SEQ_LEN:
        raise ValueError(f"History window {len(hist_sl)} != {SEQ_LEN}")
    hist_sl = hist_sl.reindex(columns=_HIST_COLS)

    # Last historical day's raw weather (for delta computation in future window)
    last_hist_exog = state_exog.loc[state_exog.index < pd.Timestamp(str(target_date)[:10])].tail(96)

    # Future exog for K=2 days: T+1 and T+2 rotated windows concatenated
    dec_steps    = HORIZON * FORECAST_DAYS    # 192
    start_rot    = _rot_anchor(t0_day, START_BLOCK)
    fut_exog_raw = _future_exog_window(state_exog, start_rot, last_hist_exog, steps=dec_steps)

    # Baseline for T+1 and T+2 concatenated (MW), then scale
    base_mw_t1 = _baseline_for_future(df_load, t0_day)
    base_mw_t2 = _baseline_for_future(df_load, t0_day + pd.Timedelta(days=1))
    base_mw    = pd.concat([base_mw_t1, base_mw_t2], ignore_index=True)

    fut_scaled = fut_exog_raw.copy()
    fut_scaled["baseline"] = arts.load_scaler.transform(
        base_mw.values.reshape(-1, 1))[:, 0]
    # Scale continuous exog features using the exact schema captured by
    # StandardScaler during training. Some fitted columns are historical-only
    # diagnostics, so they are supplied as neutral zeros for future inference.
    scaler_cols = list(getattr(arts.exog_scaler, "feature_names_in_", _EXOG_SCALE_COLS))
    scaled_exog = pd.DataFrame(
        arts.exog_scaler.transform(fut_scaled.reindex(columns=scaler_cols, fill_value=0.0).fillna(0.0)),
        index=fut_scaled.index,
        columns=scaler_cols,
    )
    for col in [c for c in scaler_cols if c in fut_scaled.columns]:
        fut_scaled[col] = scaled_exog[col]
    fut_scaled = fut_scaled.reindex(columns=_FUT_COLS).fillna(0.0)

    # Rebuild model and load weights
    model = _build_model(HP, Fh=len(_HIST_COLS), Ff=len(_FUT_COLS))
    model.load_weights(str(artifacts_dir / "model.weights.h5"))

    Xh_pred  = hist_sl.to_numpy(dtype=np.float32)
    Xf_pred  = fut_scaled.to_numpy(dtype=np.float32)
    y_in_res = np.zeros((dec_steps, 1), dtype=np.float32)
    y_in_res[0, 0] = Xh_pred[-1, 0]

    yhat_res_scaled = model.predict(
        {"hist": Xh_pred[None], "future_exog": Xf_pred[None],
         "y_in": y_in_res[None], "is_training": np.array([False], dtype=bool)},
        verbose=0,
    )[0, :, 0]   # shape (192,)

    # ── T+1: first 96 rotated blocks ─────────────────────────────────────────
    yhat_t1_scaled = yhat_res_scaled[:HORIZON]
    baseline_t1    = fut_scaled["baseline"].to_numpy()[:HORIZON]
    yhat_t1_scaled = yhat_t1_scaled + baseline_t1
    yhat_mw        = arts.load_scaler.inverse_transform(yhat_t1_scaled.reshape(-1, 1))[:, 0]

    # ── T+2: next 96 rotated blocks ──────────────────────────────────────────
    yhat_t2_scaled = yhat_res_scaled[HORIZON:2 * HORIZON]
    baseline_t2    = fut_scaled["baseline"].to_numpy()[HORIZON:2 * HORIZON]
    yhat_t2_scaled = yhat_t2_scaled + baseline_t2
    yhat_mw_t2     = arts.load_scaler.inverse_transform(yhat_t2_scaled.reshape(-1, 1))[:, 0]

    # ── Post-inference sanity clamp (T+1) ─────────────────────────────────────
    # Use caller-supplied sigma override (from WeatherAnomalyResult) when available;
    # fall back to drastic-day widening, then the module default.
    if clamp_sigma_override is not None:
        clamp_sigma = float(clamp_sigma_override)
    else:
        clamp_sigma = _CLAMP_SIGMA * (1.5 if prev_was_drastic else 1.0)

    yhat_mw = _post_inference_clamp(
        yhat_mw, df_load, t0_day,
        sigma_start=clamp_sigma,
        exempt_blocks=exempt_blocks,
    )

    # ── Post-inference sanity clamp (T+2) ─────────────────────────────────────
    t2_day = t0_day + pd.Timedelta(days=1)
    yhat_mw_t2 = _post_inference_clamp(
        yhat_mw_t2, df_load, t2_day,
        sigma_start=clamp_sigma,      # same weather context applies
        exempt_blocks=exempt_blocks,
    )

    # ── Score and persist drastic flag ────────────────────────────────────────
    score = _drastic_score(yhat_mw, df_load, t0_day)
    if score >= _DRASTIC_SCORE_THR:
        logger.warning("Helper AI: drastic forecast detected (score=%.2f) for %s — "
                       "clamped; T+2 will restabilize.", score, str(target_date)[:10])
        _mark_drastic(artifacts_dir, str(target_date)[:10], score)

    # ── Block-level error correction (T+1) ───────────────────────────────────
    _DECAY    = float(correction_decay_override) if correction_decay_override is not None else 0.997
    _LOOKBACK = 28
    t1_end_correction = 0.0   # carried into T+2 correction below

    if actual_blocks > 0 and len(actual_partial) > 0:
        n_act  = min(int(actual_blocks), HORIZON, len(actual_partial))
        errors = actual_partial[:n_act] - yhat_mw[:n_act]

        recent_n   = min(n_act, _LOOKBACK)
        weights    = np.array([_DECAY ** (recent_n - 1 - i) for i in range(recent_n)])
        weights   /= weights.sum()
        seed_error = float(np.dot(weights, errors[n_act - recent_n:]))

        correction = seed_error
        for i in range(n_act, HORIZON):
            correction *= _DECAY
            yhat_mw[i] += correction
        t1_end_correction = correction   # value at block 96, seed for T+2

        yhat_mw[:n_act] = actual_partial[:n_act]

        logger.info(
            "Helper AI: block adjustment applied — "
            "seed_error=%.1f MW, actual_blocks=%d, decay=%.3f",
            seed_error, n_act, _DECAY,
        )

    # ── T+2 error correction: bridge from T+1 end-of-day correction ──────────
    # No actuals for T+2 yet; seed from the residual correction still active at
    # the end of T+1 so the day boundary is smooth.
    if abs(t1_end_correction) > 1.0:
        corr_t2 = t1_end_correction
        for i in range(HORIZON):
            corr_t2 *= _DECAY
            yhat_mw_t2[i] += corr_t2
        yhat_mw_t2 = np.maximum(yhat_mw_t2, 0.0)

    # ── Map rotated blocks → calendar order 1..96 (T+1) ─────────────────────
    rot_idx = pd.date_range(start=start_rot, periods=HORIZON, freq="15T")
    cal_blk = ((rot_idx.hour * 60 + rot_idx.minute) // 15 + 1).to_numpy()
    out     = np.full(96, np.nan)
    for i, b in enumerate(cal_blk):
        if 1 <= b <= 96:
            out[b - 1] = yhat_mw[i]
    out = pd.Series(out).interpolate(limit_direction="both").to_numpy()
    t1_cal = np.maximum(out, 0.0).astype(float)

    # ── Map rotated blocks → calendar order 1..96 (T+2) ─────────────────────
    rot_idx_t2 = pd.date_range(
        start=_rot_anchor(t2_day, START_BLOCK), periods=HORIZON, freq="15T"
    )
    cal_blk_t2 = ((rot_idx_t2.hour * 60 + rot_idx_t2.minute) // 15 + 1).to_numpy()
    out_t2     = np.full(96, np.nan)
    for i, b in enumerate(cal_blk_t2):
        if 1 <= b <= 96:
            out_t2[b - 1] = yhat_mw_t2[i]
    out_t2 = pd.Series(out_t2).interpolate(limit_direction="both").to_numpy()
    t2_cal = np.maximum(out_t2, 0.0).astype(float)

    return t1_cal, t2_cal


# ── Data conversion (pipeline df → helper format) ─────────────────────────────

def _make_df_load(df: pd.DataFrame, before_date: str) -> pd.DataFrame:
    d = df.copy()
    d["_d"] = d["date"].astype(str).str[:10]
    d = d[d["_d"] < before_date]
    d["Date"]             = pd.to_datetime(d["_d"])
    d["block"]            = pd.to_numeric(d["time_block"], errors="coerce")
    d["total_drawal_adj"] = pd.to_numeric(d["total_drawal"], errors="coerce")
    d = d[["Date", "block", "total_drawal_adj"]].dropna()
    d["block"] = d["block"].astype(int)
    return d[d["block"].between(1, 96)].sort_values(["Date", "block"]).reset_index(drop=True)


def _make_state_exog(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all weather columns (including rain/showers/snowfall) indexed by Datetime."""
    d = df.copy()
    d["_d"]      = d["date"].astype(str).str[:10]
    d["_blk"]    = pd.to_numeric(d["time_block"], errors="coerce").fillna(1).astype(int)
    d["Datetime"]= pd.to_datetime(d["_d"]) + pd.to_timedelta((d["_blk"] - 1) * 15, unit="m")
    d = d.set_index("Datetime").sort_index()

    exog = pd.DataFrame(index=d.index)
    defaults = {"temperature": 25.0, "humidity": 50.0, "rain": 0.0,
                "showers": 0.0, "snowfall": 0.0, "apparent_temperature": 25.0,
                "cloud_cover": 0.0, "sunshine_duration": 0.0,
                "direct_radiation": 0.0, "wind_speed_10m": 0.0}
    for c, default in defaults.items():
        if c in d.columns:
            exog[c] = pd.to_numeric(d[c], errors="coerce").fillna(default)
        else:
            exog[c] = default

    exog = exog[~exog.index.duplicated(keep="last")]
    exog = exog.interpolate(method="time").ffill().bfill()
    return exog


# ── Public entry point ────────────────────────────────────────────────────────

def run_helper_forecast(
    df: pd.DataFrame,
    target_date,
    region: str,
    actual_blocks: int,
    actual_partial: np.ndarray,
    artifacts_dir,
    retrain_every_hours: float = 24.0,
    blend_weight: float = 0.40,
    return_t2: bool = False,
    clamp_sigma_override: float = None,
    correction_decay_override: float = None,
    exempt_blocks: set = None,
):
    """
    Train (if stale) + run encoder-decoder LSTM on pipeline data.

    Returns
    -------
    return_t2=False (default): 96-element ndarray (MW) for T+1, or None on failure.
    return_t2=True           : (t1_vec, t2_vec) tuple of two 96-element ndarrays,
                               or (None, None) on failure.

    Parameters
    ----------
    clamp_sigma_override      : pass WeatherAnomalyResult.clamp_sigma_override
    correction_decay_override : pass WeatherAnomalyResult.correction_decay_override
    exempt_blocks             : set of 1-based block numbers exempt from clamping
                                (pass union of all weather-flagged block lists)
    """
    if not _TF_AVAILABLE:
        logger.warning("TensorFlow not available — helper model skipped.")
        return (None, None) if return_t2 else None

    target_str = str(target_date)[:10]
    art_dir    = Path(artifacts_dir) / region.replace(" ", "_").lower()

    try:
        df_load    = _make_df_load(df, before_date=target_str)
        state_exog = _make_state_exog(df)

        if len(df_load) < SEQ_LEN + 10:
            logger.info("Helper AI: insufficient history (%d rows) — skipping.", len(df_load))
            return (None, None) if return_t2 else None

        w_path = art_dir / "model.weights.h5"
        s_path = art_dir / "scalers.pkl"
        need_train = not (w_path.exists() and s_path.exists())
        if not need_train:
            age_h = (time.time() - w_path.stat().st_mtime) / 3600.0
            need_train = age_h >= retrain_every_hours
        # Schema-version check: retrain if features changed since last save
        if not need_train and s_path.exists():
            try:
                with open(s_path, "rb") as _sf:
                    _art = pickle.load(_sf)
                if getattr(_art, "schema_version", "") != _SCHEMA_VERSION:
                    logger.info("Helper AI: schema changed (%s → %s) — forcing retrain.",
                                getattr(_art, "schema_version", "?"), _SCHEMA_VERSION)
                    need_train = True
            except Exception:
                need_train = True

        if need_train:
            logger.info("Helper AI: training for region=%s …", region)
            _train(df_load, state_exog, art_dir)

        prev_date_str = str(
            (pd.Timestamp(target_str) - pd.Timedelta(days=1)).date()
        )
        prev_drastic = _check_prev_drastic(art_dir, prev_date_str)
        if prev_drastic:
            logger.info("Helper AI: previous day %s was drastic — widening clamp tolerance.",
                        prev_date_str)

        t1_vec, t2_vec = _infer(
            df_load, state_exog, target_date, actual_blocks, actual_partial,
            art_dir,
            prev_was_drastic=prev_drastic,
            clamp_sigma_override=clamp_sigma_override,
            correction_decay_override=correction_decay_override,
            exempt_blocks=exempt_blocks,
        )
        logger.info("Helper AI: inference complete for %s", target_str)
        return (t1_vec, t2_vec) if return_t2 else t1_vec

    except Exception as exc:
        logger.warning("Helper AI failed: %s", exc, exc_info=True)
        return (None, None) if return_t2 else None
