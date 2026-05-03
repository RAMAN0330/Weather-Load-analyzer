import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Model, callbacks as C, layers as L, optimizers as O

warnings.filterwarnings("ignore")

# ------------------
# GLOBAL CONFIG
# ------------------
SEQ_LEN       = 1*96                  # history @15-min
HORIZON       = 96                  # rotated future length
EPOCHS        = 40
PATIENCE      = 10
BATCH_SIZE    = 16
SEED          = 42
BASELINE_K    = 2                # number of past *rotated* days in causal baseline
DERIV_LAMBDA  = 0.3               # derivative loss weight

WT_TEMP    = 0.8
WT_PRECIP  = 1.2
WT_SUNDAY  = 0.8
WT_HOLIDAY = 1.2
W_MAX      = 5.0

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

HP = dict(
    enc_units=256,
    enc_layers=2,
    enc_activation='tanh',
    enc_recurrent_activation='sigmoid',
    dec_units=256,
    attn_units=256,
    exog_embed=64,
    exog_activation='relu',
    dec_activation='tanh',
    dec_recurrent_activation='sigmoid',
    dropout=0.1,
    lr=3e-4
)

@dataclass
class Artifacts:
    load_scaler: MinMaxScaler
    exog_scaler: StandardScaler
    history_cols: list
    future_cols: list

# ------------------
# Static holidays (example; extend as needed)
# ------------------
INDIAN_HOLIDAYS = {
    'republic_day': [(1, 26)],
    'independence_day': [(8, 15)],
    'gandhi_jayanti': [(10, 2)],
    2023: [(1,26),(3,8),(3,30),(4,4),(4,6),(4,14),(4,21),(5,1),(6,29),(8,15),(8,31),(9,19),(10,2),(10,24),(11,12),(11,13),(12,25)],
    2024: [(1,26),(3,8),(3,25),(3,29),(4,11),(4,14),(4,17),(5,1),(6,17),(8,15),(8,19),(9,7),(10,2),(10,31),(11,1),(11,15),(12,25)],
    2025: [(1,26),(3,8),(3,14),(4,10),(4,13),(4,18),(5,1),(6,6),(8,15),(8,27),(9,27),(10,2),(10,20),(11,4),(12,25)],
}

def get_indian_holidays_set():
    holidays = set()
    for year in range(2020, 2031):
        holidays.add(pd.Timestamp(year, 1, 26))
        holidays.add(pd.Timestamp(year, 8, 15))
        holidays.add(pd.Timestamp(year, 10, 2))
        holidays.add(pd.Timestamp(year, 12, 25))
    for year, dates in INDIAN_HOLIDAYS.items():
        if isinstance(year, int):
            for month, day in dates:
                holidays.add(pd.Timestamp(year, month, day))
    return holidays

def _rot_anchor_time(rot_date, start_block: int) -> pd.Timestamp:
    """Return the true timestamp of rotated day start for given rot_date."""
    return pd.Timestamp(rot_date) + pd.Timedelta(minutes=(start_block - 1) * 15)

# ------------------
# IO & cleaning
# ------------------
#####################################################################################

def to_create_input_file(merit_path, nreb_path):

    df= pd.read_csv(merit_path)
    NREB_PATH = Path(nreb_path)

    for col in ['demand_met', 'field_4_', 'own_generation']:
        df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Step 3: Convert 'date' column to datetime ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # --- Step 4: Filter data for 6th June 2025 ---
    target_states = ['Rajasthan']        # <-- Modify as needed
    # target_date = pd.to_datetime("2025-11-26")         # <-- Modify as needed
    filtered_df = df[(df['state'].isin(target_states)) ]

    # --- Step 5: Group by time_slot and calculate mean ---
    agg_df = (
        filtered_df
        .groupby(['date','time_slot'], as_index=False)
        .agg({
            'demand_met': 'mean',
            # 'field_4_': 'mean',
            # 'own_generation': 'mean'
        })
        .round(2)
    )

    # --- Step 6: Create all 96 time slots (15-min blocks) ---
    all_slots = pd.DataFrame({
        'time_slot': pd.date_range("00:00", "23:45", freq="15min").strftime("%H:%M")
    })

    # --- Step 7: Merge to include missing slots as NaN ---
    complete_df = pd.merge(all_slots, agg_df, on='time_slot', how='left')

    # --- Step 9: Interpolate missing values ---
    complete_df['demand_met'] = complete_df['demand_met'].interpolate(
        method='linear',  # you can also try 'time' or 'spline', see notes below
        limit_direction='both'  # fills NaNs at the beginning and end too
    )

    nreb = pd.read_csv(NREB_PATH)
    # nreb['date'] = pd.to_datetime(nreb['date'], dayfirst=True, errors='coerce')
    nreb['date'] = pd.to_datetime(nreb['date'], dayfirst=False, errors='coerce')
    
    # Keep only required columns & rename
    nreb_small = nreb[['date', 'time', 'time_block', 'rvpn_drawl_mw']].copy()
    nreb_small = nreb_small.rename(columns={'rvpn_drawl_mw': 'load_mw'})
    
    merit = complete_df.copy()
    merit['date'] = pd.to_datetime(merit['date'])

    # --------- 3. FILTER MERIT TO ONLY NEWER DATES ----------
    def time_to_block(t):
        if pd.isna(t):
            return pd.NA
        h, m = map(int, str(t).split(':'))
        return h * 4 + m // 15 + 1
 
    merit['time_block'] = merit['time_slot'].apply(time_to_block)
    merit = merit.rename(columns={'demand_met': 'load_mw'})
    merit = merit[['date', 'time_block', 'load_mw']].copy()
 
    # ================================
    # ✅ ZERO VALUE REPLACEMENT LOGIC
    # ================================
    merged = nreb_small.merge(
        merit,
        on=['date', 'time_block'],
        how='left',
        suffixes=('_nreb', '_merit')
    )
 
    merged['load_mw'] = np.where(
        merged['load_mw_nreb'] == 0,
        merged['load_mw_merit'],
        merged['load_mw_nreb']
    )
 
    nreb_fixed = merged[['date', 'time_block', 'load_mw']].copy()
 
    # ---------- FILTER ONLY NEWER MERIT DATES ----------
    last_nreb_date = nreb_fixed['date'].max()
    merit_new = merit[merit['date'] > last_nreb_date].copy()
 
    # ---------- FINAL CONCAT ----------
    final_df = pd.concat(
        [nreb_fixed, merit_new],
        ignore_index=True
    )
 
    final_df = final_df.sort_values(['date', 'time_block']).reset_index(drop=True)
 
    return final_df

############################################################################################

def bound(df, method='const', upper_value=50000.0, lower_value=0.0):
    if method == 'const':
        ub = upper_value; lb = lower_value
    elif method == 'perc':
        ub = np.percentile(df['total_drawal_adj'], upper_value)
        lb = np.percentile(df['total_drawal_adj'], lower_value)
    else:
        raise ValueError("method should be 'const' or 'perc'")
    return ub, lb

def clean_load_df(df_raw, ubp=99, lbp=0.1):
    """
    Expect ['date','time_block' or 'block','total_drawal' or 'load_mw', ...]
    Output: ['Date','block','total_drawal_adj']
    """
    df = df_raw.copy()
    df = df.drop(columns=['datetime','time','state','od_ud_mw','update_date','id','year','month'], errors='ignore')
    df.columns = df.columns.str.strip()
    if 'time_block' in df.columns: df = df.rename(columns={'time_block':'block'})
    if 'load_mw'   in df.columns: df = df.rename(columns={'load_mw':'total_drawal'})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # df['date'] = pd.to_datetime(df['date'],dayfirst=True, errors='coerce')
    df['Date'] = df['date'].dt.normalize()
    df['total_drawal_adj'] = df['total_drawal'].astype(float)

    UB, LB = bound(df, method='const', upper_value=ubp, lower_value=lbp)
    df.loc[df['total_drawal_adj'] > UB, 'total_drawal_adj'] = np.nan
    df.loc[df['total_drawal_adj'] < LB, 'total_drawal_adj'] = np.nan
    df['total_drawal_adj'] = df['total_drawal_adj'].ffill().bfill()

    out = df[['Date','block','total_drawal_adj']].dropna().copy()
    out['block'] = out['block'].astype(int)
    return out

# ------------------
# Weather → state exog (corr-weighted to state)
# ------------------
# def _calc_weights(corr_series, expected_sign=1, method="abs"):
#     if method == "clip":
#         valid = corr_series.clip(lower=0) if expected_sign==1 else (-corr_series).clip(lower=0)
#     elif method == "abs":
#         valid = corr_series.abs()
#     elif method == "hybrid":
#         if expected_sign == 1:
#             valid = corr_series.copy()
#             valid[valid<0] = valid[valid<0].abs()*0.2
#         else:
#             valid = -corr_series.copy()
#             valid[valid<0] = valid[valid<0].abs()*0.2
#     else:
#         raise ValueError
#     if valid.sum() > 0:
#         return valid/valid.sum()
#     return pd.Series(1/len(valid), index=valid.index)

def _calc_weights(corr_series, expected_sign=1, method="abs"):
    """
    Return weights over *all* locations in corr_series.index.
    NaN correlations are handled gracefully so that .dot() alignment never breaks.
    """
    # Handle the simple 'mean' case separately
    if method == "mean":
        # use only non-NaN correlations to decide which locations are "valid"
        valid = corr_series.dropna()
 
        if len(valid) == 0:
            # no valid correlations at all -> equal weights over ALL locations
            base = pd.Series(1.0, index=corr_series.index)
        else:
            # give equal weight only to locations that had valid correlations,
            # but keep all locations in the index with 0 weight
            base = pd.Series(0.0, index=corr_series.index)
            base.loc[valid.index] = 1.0
 
        # normalise
        return base / base.sum()
 
    # ---- the other methods as before, but keep full index ----
    if method == "clip":
        if expected_sign == 1:
            valid = corr_series.copy()
        else:
            valid = -corr_series.copy()
        valid = valid.clip(lower=0)
    elif method == "abs":
        valid = corr_series.abs()
    elif method == "hybrid":
        if expected_sign == 1:
            valid = corr_series.copy()
            valid[valid < 0] = valid[valid < 0].abs() * 0.2
        else:
            valid = -corr_series.copy()
            valid[valid < 0] = valid[valid < 0].abs() * 0.2
    else:
        raise ValueError("Unknown method in _calc_weights")
 
    # keep NaNs but treat them as 0 contribution
    valid = valid.fillna(0.0)
    s = valid.sum()
    if s > 0:
        return valid / s
 
    # all zeros -> fallback to equal weights over all locations
    equal = pd.Series(1.0, index=corr_series.index)
    return equal / equal.sum()

def build_state_exog_from_weather(weather_df, load_df_for_weights):
    """
    weather_df: ['date','time_block','location','apparent_temperature'/'temperature',
                 'relative_humidity_2m'/'humidity','precipitation']
    return state_exog @15min index with columns ['temperature','humidity','precipitation']
    """
    w = weather_df.copy()
    w['date'] = pd.to_datetime(w['date'], errors='coerce')
    w['Datetime'] = w['date'] + pd.to_timedelta((w['time_block']-1)*15, unit='m')
    w = w.rename(columns={'apparent_temperature':'temperature',
                          'relative_humidity_2m':'humidity'})
    needed = ['Datetime','location','temperature','humidity','precipitation']
    miss = [c for c in needed if c not in w.columns]
    if miss:
        raise ValueError(f"Weather missing columns: {miss}")
    for c in ['temperature','humidity','precipitation']:
        w[c] = pd.to_numeric(w[c], errors='coerce')
    w['precipitation'] = w['precipitation'].fillna(0.0)

    ld = load_df_for_weights.copy()
    ld['Datetime'] = ld['Date'] + pd.to_timedelta((ld['block']-1)*15, unit='m')
    load_ser = ld.set_index('Datetime')['total_drawal_adj'].astype(float)

    temp_df   = w.pivot(index='Datetime', columns='location', values='temperature')
    humid_df  = w.pivot(index='Datetime', columns='location', values='humidity')
    precip_df = w.pivot(index='Datetime', columns='location', values='precipitation')

    common_idx = load_ser.index
    for _df in (temp_df, humid_df, precip_df):
        common_idx = common_idx.intersection(_df.index)

    temp_al, humid_al, precip_al = temp_df.loc[common_idx], humid_df.loc[common_idx], precip_df.loc[common_idx]
    load_al = load_ser.loc[common_idx]

    # w_temp   = _calc_weights(temp_al.corrwith(load_al),   expected_sign=+1, method="abs")
    # w_humid  = _calc_weights(humid_al.corrwith(load_al),  expected_sign=-1, method="abs")
    # w_precip = _calc_weights(precip_al.corrwith(load_al), expected_sign=-1, method="abs")

    w_temp   = _calc_weights(temp_al.corrwith(load_al),   expected_sign=+1, method="mean")
    w_humid  = _calc_weights(humid_al.corrwith(load_al),  expected_sign=-1, method="mean")
    w_precip = _calc_weights(precip_al.corrwith(load_al), expected_sign=-1, method="mean")

    full_temp   = w.pivot(index='Datetime', columns='location', values='temperature').dot(w_temp).rename('temperature')
    full_humid  = w.pivot(index='Datetime', columns='location', values='humidity').dot(w_humid).rename('humidity')
    full_precip = w.pivot(index='Datetime', columns='location', values='precipitation').dot(w_precip).rename('precipitation')

    state_exog = pd.concat([full_temp, full_humid, full_precip], axis=1).sort_index()
    return state_exog

# ------------------
# Rotated-day helpers
# ------------------

def make_weather_action_weights_rotated(
    full: pd.DataFrame,
    *,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    start_block: int = 33,
    wt_alpha_temp: float = WT_TEMP,
    wt_alpha_precip: float = WT_PRECIP,
    wt_alpha_sunday: float = WT_SUNDAY,
    wt_alpha_holiday: float = WT_HOLIDAY,
    w_max: float = W_MAX,
) -> np.ndarray:
    """
    Returns sample_weight array with shape (num_rotated_samples, horizon).
    Larger where |ΔT|, precip>0, Sunday, or holiday in the rotated future slice.
    NOTE: Assumes `full` already has standardized temperature/precipitation.
    """
    full = full.copy().sort_index()
    # tag each timestamp with its rotated "day"
    full['rot_date'] = assign_rotated_date(full.index, start_block=start_block).to_numpy()

    # keep only complete rotated days (96 points)
    counts = full['rot_date'].value_counts()
    ok_dates = set(counts[counts == 96].index)
    full = full[full['rot_date'].isin(ok_dates)].sort_values(['rot_date']).sort_index()

    W = []
    for d in sorted(pd.unique(full['rot_date'])):
        # t0 = pd.Timestamp(d) + pd.Timedelta(hours=8)  # 08:00 anchor
        t0 = _rot_anchor_time(d, start_block)
        hist_start = t0 - pd.to_timedelta(seq_len * 15, unit='m')
        hist_slice = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_slice  = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15 * horizon))]

        if len(hist_slice) != seq_len or len(fut_slice) != horizon:
            continue

        T  = fut_slice['temperature'].to_numpy()
        P  = fut_slice['precipitation'].to_numpy()
        dT = np.r_[0.0, np.diff(T)]

        is_sun = fut_slice['is_sunday'].to_numpy().astype(float)
        is_hol = fut_slice['is_holiday'].to_numpy().astype(float)

        w = (
            1.0
            + wt_alpha_temp   * np.abs(dT)
            + wt_alpha_precip * np.maximum(0.0, P)
            + wt_alpha_sunday * is_sun
            + wt_alpha_holiday* is_hol
        )
        W.append(np.clip(w, 1.0, w_max).astype('float32'))

    return np.asarray(W, dtype='float32')


def rot_block_from_index(idx: pd.DatetimeIndex, start_block:int=33) -> np.ndarray:
    """1..96 where 1 corresponds to calendar block start_block (33=08:00)."""
    block = ((idx.hour*60 + idx.minute)//15 + 1).astype(int)
    return ((block - start_block) % 96) + 1

# ------------------
# Rotated causal baseline
# ------------------
# --------- Rotated-day helpers (33..96 + 1..32) ---------

def assign_rotated_date(index_like: pd.DatetimeIndex, start_block: int = 33) -> pd.Series:
    """
    For each timestamp, assign the 'rotated date' so that a 'day' runs from
    start_block (e.g., 33 ≡ 08:00) ... 96, then 1 ... (start_block-1).
    All timestamps with block < start_block belong to the *previous* rotated day.
    """
    idx = pd.DatetimeIndex(index_like)
    block = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    base = idx.normalize()
    # shift back 1 day for blocks before the 33-start seam
    shift = pd.to_timedelta((block < start_block).astype(int), unit="D")
    rot_date = base - shift
    return pd.Series(rot_date, index=idx)


def rotated_block_number(index_like: pd.DatetimeIndex, start_block: int = 33) -> np.ndarray:
    """
    Map true block (1..96) -> rotated block (1..96) such that rotated block 1 == start_block.
    e.g., with start_block=33, true block 33 -> rot 1, true 34 -> rot 2, ..., true 32 -> rot 96.
    """
    idx = pd.DatetimeIndex(index_like)
    true_block = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    rot_block = ((true_block - start_block) % 96) + 1
    return rot_block


def add_time_features_rotated(index_like: pd.DatetimeIndex, start_block: int = 33) -> pd.DataFrame:
    """
    Time features in the rotated frame:
      - rot_block_sin/cos: position of the 15-min slot within the rotated day
      - dow_sin/cos, is_holiday/is_sunday/is_weekend computed from the *rotated* date
    """
    idx = pd.DatetimeIndex(index_like)
    rot_date = assign_rotated_date(idx, start_block=start_block)
    rot_block = rotated_block_number(idx, start_block=start_block)

    theta_b = 2 * np.pi * (rot_block - 1) / 96.0
    # compute DOW on rotated date, not the true calendar date
    dow = pd.DatetimeIndex(rot_date).dayofweek.astype(int)
    theta_d = 2 * np.pi * dow / 7.0

    # holidays based on rotated date
    indian_holidays = get_indian_holidays_set()
    rot_date_norm = pd.DatetimeIndex(rot_date).normalize()
    is_holiday = rot_date_norm.isin(indian_holidays).astype(int)
    is_sunday = (dow == 6).astype(int)
    is_weekend = (dow >= 5).astype(int)

    return pd.DataFrame({
        "rot_block_sin": np.sin(theta_b),
        "rot_block_cos": np.cos(theta_b),
        "dow_sin":       np.sin(theta_d),
        "dow_cos":       np.cos(theta_d),
        "is_holiday":    is_holiday,
        "is_sunday":     is_sunday,
        "is_weekend":    is_weekend,
    }, index=idx)


def causal_baseline_series_rotated(df_load: pd.DataFrame, *, K: int, start_block: int = 33) -> pd.Series:
    """
    Build a *causal* per-rotated-block baseline in MW:
      - Treat each rotated day as 96 blocks (33..96,1..32)
      - For each rotated block b, baseline_b(D) = mean of the last K rotated days
        at the same rotated block, *excluding* current day (shift by 1).
      - Returns a Series indexed by true timestamps (DatetimeIndex) so it
        can align with your main frame.
    
    Expects columns: ['Date','block','total_drawal_adj'] in df_load.
    """
    # 1) Create a Datetime index and drop the very first calendar day's 1..(start_block-1) to avoid the initial seam
    df = df_load.copy()
    df['Datetime'] = df['Date'] + pd.to_timedelta((df['block'] - 1) * 15, unit='m')
    df = df.sort_values(['Date', 'block'])
    first_day = df['Date'].min()
    df = df.loc[~((df['Date'] == first_day) & (df['block'] <= (start_block - 1)))].copy()

    # 2) Compute rotated date & rotated block for each row
    idx = pd.DatetimeIndex(df['Datetime'])
    df['rot_date']  = assign_rotated_date(idx, start_block=start_block).to_numpy()
    df['rot_block'] = rotated_block_number(idx, start_block=start_block)

    # 3) Keep only complete rotated days if you want strict 96-length; otherwise we can tolerate partials
    #    Here we tolerate partials: the rolling mean will just have fewer contributors.
    #    But it's often better to keep only full rotated days:
    # counts = df.groupby('rot_date')['rot_block'].nunique()
    # full_dates = counts[counts == 96].index
    # df = df[df['rot_date'].isin(full_dates)].copy()

    # 4) Pivot to [rot_date x rot_block]; compute rolling mean K over previous rotated days
    piv = df.pivot_table(index='rot_date', columns='rot_block', values='total_drawal_adj', aggfunc='mean').sort_index()
    base_rot = piv.rolling(window=K, min_periods=1).mean().shift(1)  # causal shift

    # 5) Unstack back to long on (rot_date, rot_block)
    base_long = base_rot.stack(dropna=False).rename('baseline').reset_index()

    # 6) Map (rot_date, rot_block) back to the *true* timestamp:
    #    For a rotated day starting at 08:00 (block 33), the k-th rotated block occurs at:
    #      ts = rot_date @ 08:00 + (rot_block-1) * 15 min

    # base_long['Datetime'] = pd.to_datetime(base_long['rot_date']) + pd.to_timedelta(8, unit='h') \
    #                         + pd.to_timedelta((base_long['rot_block'] - 1) * 15, unit='m')

    base_long['Datetime'] = pd.to_datetime(base_long['rot_date']).apply(
        lambda d: _rot_anchor_time(d, start_block)
    ) + pd.to_timedelta((base_long['rot_block'] - 1) * 15, unit='m')

    # 7) Build a Series indexed by true timestamps
    base_series = base_long.set_index('Datetime')['baseline'].sort_index()

    return base_series


def baseline_for_future_rotated_day(df_hist_rot: pd.DataFrame, rot_day: pd.Timestamp, *,
                                    K:int=BASELINE_K, start_block:int=33) -> pd.Series:
    """
    Build a 96-length baseline vector (MW) for the rotated day anchored at 'rot_day @ 08:00'
    using the last K rotated days from df_hist_rot.
    """
    df = df_hist_rot.copy()

    # Build rotated pivot from the *history* (same as builder above but in one shot)
    # Make Datetime index to compute rot mapping:
    # However df may not have every timestamp → reconstruct by what exists:
    df['Datetime'] = df['Date'] + pd.to_timedelta((df['block']-1)*15, unit='m')
    ser = df.set_index('Datetime')['total_drawal_adj'].sort_index()

    idx = ser.index
    rdate = assign_rotated_date(idx, start_block=start_block).values
    rblk  = rot_block_from_index(idx, start_block=start_block)
    tmp = pd.DataFrame({"rot_date": rdate, "rot_block": rblk, "y":ser.values})
    piv_rot = tmp.pivot_table(index='rot_date', columns='rot_block', values='y', aggfunc='mean').sort_index()

    # last K rotated days before rot_day
    hist = piv_rot.loc[piv_rot.index < pd.Timestamp(rot_day)].tail(K)
    if hist.empty:
        # fallback: simple last available rotated day
        hist = piv_rot.tail(1)
    vec = hist.mean(axis=0)  # 96-length Series indexed by rot_block
    vec.index = pd.RangeIndex(1,97)

    # Build target datetime index for this rotated day:
    # start_t = pd.Timestamp(rot_day) + pd.Timedelta(hours=8)  # 08:00 anchor
    start_t = _rot_anchor_time(rot_day, start_block)
    idx_out = pd.date_range(start=start_t, periods=96, freq='15T')
    return pd.Series(vec.values, index=idx_out, name='baseline')

# ------------------
# Build full stream (rotated baseline + residuals + exog)
# ------------------
def build_state_series_rotated(df_load: pd.DataFrame, state_exog: pd.DataFrame, *, start_block: int = 33) -> pd.DataFrame:
    """
    Build the training/eval stream while *treating 33..32 as a single day*.
    - Drops the first calendar day's 1..32 to avoid an initial seam
    - Adds rotated causal baseline (MW), residual, and rotated time features
    - Keeps exogenous features aligned on the true timeline
    """
    df = df_load.copy()
    # True timestamp index
    df['Datetime'] = df['Date'] + pd.to_timedelta((df['block'] - 1) * 15, unit='m')
    df = df.set_index('Datetime').sort_index()

    # Drop the first calendar day's 1..32 blocks to avoid the initial seam
    first_day = df['Date'].min()
    mask_head = (df['Date'] == first_day) & (df['block'] <= (start_block - 1))
    df = df.loc[~mask_head].copy()

    # Exogenous features aligned by true timestamps
    exog = state_exog.reindex(df.index).ffill().bfill()

    # Rotated causal baseline (MW), aligned to true timestamps
    base_rot = causal_baseline_series_rotated(
        df[['Date', 'block', 'total_drawal_adj']].reset_index(drop=False),
        K=BASELINE_K, start_block=start_block
    ).reindex(df.index).ffill().bfill()

    # Residual in MW domain
    resid = df['total_drawal_adj'] - base_rot

    # Rotated time features (rotated day / block, rotated DOW & holiday)
    tfeat = add_time_features_rotated(df.index, start_block=start_block)

    full = pd.concat([
        df[['total_drawal_adj']].rename(columns={'total_drawal_adj': 'y'}),
        base_rot.rename('baseline'),
        resid.rename('residual'),
        exog[['temperature', 'humidity', 'precipitation']],
        tfeat[['rot_block_sin', 'rot_block_cos', 'dow_sin', 'dow_cos', 'is_holiday', 'is_sunday', 'is_weekend']]
    ], axis=1)

    return full



# ------------------
# Windowing on rotated days (exactly 96 future steps)
# ------------------
def make_windows_rotated(full: pd.DataFrame, seq_len=SEQ_LEN, horizon=HORIZON, start_block:int=33):
    full = full.copy().sort_index()
    full['rot_date'] = assign_rotated_date(full.index, start_block=start_block).to_numpy()

    counts = full['rot_date'].value_counts()
    ok_dates = set(counts[counts >= 92].index)  # was ==96; allow small gaps
    full = full[full['rot_date'].isin(ok_dates)].sort_values(['rot_date']).sort_index()

    hist_cols = ['residual','baseline','y','temperature','humidity','precipitation',
                 'rot_block_sin','rot_block_cos','dow_sin','dow_cos',
                 'is_holiday','is_sunday','is_weekend']
    fut_cols  = ['baseline','temperature','humidity','precipitation',
                 'rot_block_sin','rot_block_cos','dow_sin','dow_cos',
                 'is_holiday','is_sunday','is_weekend']

    dates_sorted = sorted(pd.unique(full['rot_date']))
    Xh, Xf, Y, Yin, stamps = [], [], [], [], []

    for d in dates_sorted:
        t0 = _rot_anchor_time(d, start_block)
        hist_start = t0 - pd.to_timedelta(seq_len*15, unit='m')

        hist_slice = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_slice  = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15*horizon))]

        # require *at least* some minimal coverage
        if len(hist_slice) < int(0.90*seq_len) or len(fut_slice) < int(0.95*horizon):
            continue

        # take tail/head to enforce exact sizes
        hist_slice = hist_slice.tail(seq_len)
        fut_slice  = fut_slice.head(horizon)

        # skip if still short after trimming (coverage filter is approximate)
        if len(hist_slice) != seq_len or len(fut_slice) != horizon:
            continue

        y_tgt = fut_slice[['residual']].values.astype(np.float32)
        y_in  = np.empty_like(y_tgt, dtype=np.float32)
        last_hist_resid = hist_slice['residual'].iloc[-1] if len(hist_slice) else 0.0
        y_in[0,0]  = last_hist_resid
        y_in[1:,0] = y_tgt[:-1,0]

        Xh.append(hist_slice[hist_cols].values.astype(np.float32))
        Xf.append(fut_slice[fut_cols].values.astype(np.float32))
        Y.append(y_tgt)
        Yin.append(y_in)
        stamps.append(t0)

    return (np.asarray(Xh), np.asarray(Xf), np.asarray(Y), np.asarray(Yin),
            pd.to_datetime(stamps), hist_cols, fut_cols)

# ------------------
# Attention decoder
# ------------------
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W1 = L.Dense(units, use_bias=False)
        self.W2 = L.Dense(units, use_bias=False)
        self.V  = L.Dense(1,    use_bias=False)
    def call(self, query, values):
        q = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(q))
        weights = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(weights*values, axis=1)
        return context, tf.squeeze(weights, -1)

class ARDecoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, exg_embed_dim, attn_units, dropout=0.0,
                 exog_activation='relu', dec_activation='tanh',
                 dec_recurrent_activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.dec_units = int(dec_units)
        self.exg_embed_dim = int(exg_embed_dim)
        self.attn_units = int(attn_units)
        self.dropout_rate = float(dropout)  # use manual dropout, not cell dropout

        self.attn = BahdanauAttention(attn_units)
        self.ex_emb = tf.keras.layers.Dense(exg_embed_dim, activation=exog_activation)

        # IMPORTANT: disable Keras cell-level dropout inside while_loop
        self.cell = tf.keras.layers.LSTMCell(
            dec_units,
            dropout=0.0,                # was: dropout
            recurrent_dropout=0.0,      # ensure off
            activation=dec_activation,
            recurrent_activation=dec_recurrent_activation,
        )
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        # Unpack; explicit boolean training flag is the last input
        fut, y_teacher, enc_outs, s_h, s_c, hist_in, training_flag = inputs
        # fut: (B,H,Ff), y_teacher: (B,H,1), enc_outs: (B,T,E), s_h/s_c: (B,U), hist_in: (B,T,Fh)
        horizon = tf.shape(fut)[1]

        # start from last observed residual (B,1)
        last_y = hist_in[:, -1, 0:1]
        last_y = tf.ensure_shape(last_y, [None, 1])

        # make scalar training gate
        if isinstance(training_flag, bool):
            training_flag = tf.constant(training_flag, dtype=tf.bool)
        else:
            training_flag = tf.cast(training_flag, tf.bool)
        training_gate = tf.reduce_all(training_flag)  # scalar bool

        outputs = tf.TensorArray(tf.float32, size=horizon)
        t = tf.constant(0)

        def cond(t, *_):
            return tf.less(t, horizon)

        def body(t, last_y, s_h, s_c, outputs):
            # exogenous embed at step t
            ex_t = self.ex_emb(fut[:, t, :])            # (B, exg_embed_dim)
            # attention context from encoder
            ctx_t, _ = self.attn(s_h, enc_outs)         # (B, E)
            # step input
            step_in = tf.concat([last_y, ex_t, ctx_t], axis=-1)  # (B, 1+exg+E)

            # manual dropout on input (safe in while_loop)
            if self.dropout_rate > 0.0:
                step_in = tf.cond(
                    training_gate,
                    lambda: tf.nn.dropout(step_in, rate=self.dropout_rate),
                    lambda: step_in,
                )

            # LSTMCell step (no internal dropout)
            out_t, [s_h, s_c] = self.cell(step_in, states=[s_h, s_c])  # out_t: (B,U)
            y_t = self.out(out_t)                                      # (B,1)
            y_t = tf.ensure_shape(y_t, [None, 1])

            # teacher forcing vs AR (scalar predicate)
            y_teacher_t = y_teacher[:, t, :]
            y_teacher_t = tf.ensure_shape(y_teacher_t, [None, 1])
            next_last_y = tf.where(training_gate, y_teacher_t, y_t)
            next_last_y = tf.ensure_shape(next_last_y, [None, 1])

            outputs = outputs.write(t, y_t)
            return t + 1, next_last_y, s_h, s_c, outputs

        _, _, _, _, outputs = tf.while_loop(
            cond,
            body,
            loop_vars=[t, last_y, s_h, s_c, outputs],
            parallel_iterations=1,
        )

        Y = tf.transpose(outputs.stack(), [1, 0, 2])  # (B,H,1)
        Y = tf.ensure_shape(Y, [None, None, 1])
        return Y

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]   # fut: (B,H,Ff)
        horizon = input_shape[0][1]
        return (batch, horizon, 1)


# ------------------
# Model & loss
# ------------------
def build_model(hp: dict, seq_len=SEQ_LEN, Fh=13, Ff=11):
    enc_units  = int(hp.get('enc_units', 256))
    enc_layers = int(hp.get('enc_layers', 2))
    dec_units  = int(hp.get('dec_units', 256))
    attn_units = int(hp.get('attn_units', enc_units))
    exg_embed  = int(hp.get('exog_embed', 64))
    dropout    = float(hp.get('dropout', 0.1))
    lr         = float(hp.get('lr', 3e-4))

    enc_activation           = hp.get('enc_activation', 'tanh')
    enc_recurrent_activation = hp.get('enc_recurrent_activation', 'sigmoid')
    dec_activation           = hp.get('dec_activation', 'tanh')
    dec_recurrent_activation = hp.get('dec_recurrent_activation', 'sigmoid')
    exog_activation          = hp.get('exog_activation', 'relu')

    hist     = L.Input(shape=(seq_len, Fh), name='hist')
    fut_exog = L.Input(shape=(HORIZON, Ff), name='future_exog')
    y_in     = L.Input(shape=(HORIZON, 1),  name='y_in')
    is_training = L.Input(shape=(), dtype=tf.bool, name='is_training')

    x = hist
    for _ in range(enc_layers):
        x = L.LSTM(enc_units, return_sequences=True,
                   activation=enc_activation,
                   recurrent_activation=enc_recurrent_activation)(x)
        x = L.Dropout(dropout)(x)
    enc_outputs = x
    _, enc_h, enc_c = L.LSTM(enc_units, return_sequences=False, return_state=True,
                             activation=enc_activation,
                             recurrent_activation=enc_recurrent_activation)(enc_outputs)
    init_h = L.Dense(dec_units, activation='tanh')(enc_h)
    init_c = L.Dense(dec_units, activation='tanh')(enc_c)

    y_hat_resid = ARDecoder(dec_units=dec_units,
                            exg_embed_dim=exg_embed,
                            attn_units=attn_units,
                            dropout=dropout,
                            exog_activation=exog_activation,
                            dec_activation=dec_activation,
                            dec_recurrent_activation=dec_recurrent_activation)(
        ([fut_exog, y_in, enc_outputs, init_h, init_c, hist, is_training])
    )

    model = Model(inputs=[hist, fut_exog, y_in, is_training], outputs=y_hat_resid)

    huber = tf.keras.losses.Huber(delta=0.02, reduction=tf.keras.losses.Reduction.NONE)

    def deriv_loss_per_timestep(y_true, y_pred):
        dt_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dt_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        d = tf.reduce_mean(tf.abs(dt_true - dt_pred), axis=-1)  # (B, H-1)
        d = tf.pad(d, paddings=[[0, 0], [0, 1]], mode='CONSTANT', constant_values=0.0)
        return d

    def total_loss(y_true, y_pred):
        base = huber(y_true, y_pred)            # (B, H)
        deriv = deriv_loss_per_timestep(y_true, y_pred)  # (B, H)
        return base + DERIV_LAMBDA * deriv

    model.compile(
        optimizer=O.Adam(learning_rate=lr),
        loss=total_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                 tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model

# ------------------
# Future exog window in rotated space
# ------------------
def future_exog_rotated(state_exog: pd.DataFrame, start_time, steps=96, start_block:int=33):
    idx = pd.date_range(start=pd.Timestamp(start_time), periods=int(steps), freq='15T')
    win = state_exog.reindex(idx)
    needed = ['temperature','humidity','precipitation']
    ex = win[needed].copy()
    for c in needed:
        ex[c] = pd.to_numeric(ex[c], errors='coerce')
        ex[c] = ex[c].interpolate(limit_direction='both')
    ex['precipitation'] = ex['precipitation'].fillna(0.0)

    tfeat = add_time_features_rotated(idx, start_block=start_block)
    out = pd.concat([ex, tfeat], axis=1)
    out.index = idx
    return out

# ------------------
# Train & Forecast K rotated days (each 96 steps)
# ------------------
def train_and_forecast_rotated(
    df_load_clean: pd.DataFrame,
    state_exog: pd.DataFrame, *,
    forecast_start_date,            # calendar date D; first window begins at D@08:00
    forecast_days: int,             # K rotated windows
    artifacts_dir: Path,
    project_name: str,
    hp: dict = None,
    start_block:int = 33,
    last_observed_block:int=32,
):
    if hp is None: hp = HP
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build rotated state stream
    full = build_state_series_rotated(df_load_clean, state_exog, start_block=start_block)
    # full = build_state_series(df_load_clean, state_exog)

    # Scalers
    load_scaler = MinMaxScaler()
    full['y'] = load_scaler.fit_transform(full['y'].to_numpy().reshape(-1,1))[:,0]
    full['baseline'] = load_scaler.transform(full['baseline'].to_numpy().reshape(-1,1))[:,0]
    full['residual'] = full['y'] - full['baseline']

    ex_cols_std = ['temperature','humidity','precipitation']
    exog_scaler = StandardScaler()
    full[ex_cols_std] = exog_scaler.fit_transform(full[ex_cols_std])

    # Rotated windows (96 future steps)
    Xh, Xf, Y, Yin, stamps, hist_cols, fut_cols = make_windows_rotated(
        full, seq_len=SEQ_LEN, horizon=HORIZON, start_block=start_block
    )
    if len(Xh) < 12:
        raise ValueError(f"Not enough rotated samples ({len(Xh)}).")

    W = make_weather_action_weights_rotated(
        full,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        start_block=start_block,
        wt_alpha_temp=WT_TEMP,
        wt_alpha_precip=WT_PRECIP,
        wt_alpha_sunday=WT_SUNDAY,
        wt_alpha_holiday=WT_HOLIDAY,
        w_max=W_MAX,
    )
    # Keep shapes aligned (defensive)
    if W.shape[0] != Xh.shape[0] or W.shape[1] != Y.shape[1]:
        # fall back to uniform weights if misalignment happens
        W = np.ones((Xh.shape[0], Y.shape[1]), dtype='float32')

    Fh, Ff = Xh.shape[2], Xf.shape[2]
    model = build_model(hp, Fh=Fh, Ff=Ff)

    es = C.EarlyStopping(monitor='loss', mode='min', patience=PATIENCE, restore_best_weights=True)
    rl = C.ReduceLROnPlateau(monitor='loss', mode='min', factor=0.5, patience=max(2,PATIENCE//2), min_lr=1e-5)

    N = Xh.shape[0]
    eff_bs = max(1, BATCH_SIZE)
    model.fit(
        {"hist": Xh, "future_exog": Xf, "y_in": Yin, "is_training": np.ones((N,), bool)},
        Y,
        sample_weight=W,
        epochs=EPOCHS,
        batch_size=eff_bs,          # <= this makes steps/epoch = ceil(N/eff_bs)
        callbacks=[es, rl],
        verbose=1
    )

    # save only weights (H5), avoids serializing custom layers/losses
    model.save_weights(str(artifacts_dir / f"{project_name}.weights.h5"))

    # keep your scalers/cols as you already do
    with open(artifacts_dir / "scalers.pkl", "wb") as f:
        pickle.dump(Artifacts(load_scaler, exog_scaler, hist_cols, fut_cols), f)


    # Prepare a scaled stream for rolling encoder history
    full2 = build_state_series_rotated(df_load_clean, state_exog, start_block=start_block)
    full2['y'] = load_scaler.transform(full2['y'].to_numpy().reshape(-1,1))[:,0]
    full2['baseline'] = load_scaler.transform(full2['baseline'].to_numpy().reshape(-1,1))[:,0]
    full2['residual'] = full2['y'] - full2['baseline']
    full2[ex_cols_std] = exog_scaler.transform(full2[ex_cols_std])

    hist_cols_order = hist_cols
    fut_cols_order  = fut_cols

    # History ends at 07:45 of the first output rotated day (block 32)
    t0_day = pd.Timestamp(forecast_start_date).normalize()
    # desired_hist_end = t0_day + pd.Timedelta(hours=7, minutes=45)
    effective_last_block = max(1, min(int(last_observed_block), 32))
    desired_hist_end = t0_day + pd.Timedelta(minutes=(effective_last_block - 1) * 15)
    hist_end_time = full2.index[full2.index <= desired_hist_end].max()
    if pd.isna(hist_end_time):
        raise ValueError("No history available before the first rotated window (need data up to ~07:45 of start day).")

    hist_start = hist_end_time - pd.to_timedelta(SEQ_LEN*15 - 15, unit='m')
    hist_slice = full2.loc[(full2.index >= hist_start) & (full2.index <= hist_end_time)]
    # Fill small gaps by reindexing onto the expected 15-min grid
    if len(hist_slice) != SEQ_LEN:
        expected_idx = pd.date_range(hist_start, hist_end_time, freq='15min')
        hist_slice = hist_slice.reindex(expected_idx).ffill().bfill()
    if len(hist_slice) != SEQ_LEN:
        raise ValueError(f"History window length {len(hist_slice)} != {SEQ_LEN}.")
    hist_slice = hist_slice.reindex(columns=hist_cols_order)

    # Rolling baseline history in MW domain (for generating future baseline if you extend to RT)
    df_hist_for_baseline = df_load_clean.copy()

    outputs = []

    for m in range(int(forecast_days)):
        # start_rot = t0_day + pd.Timedelta(days=m) + pd.Timedelta(hours=8)  # D+m @ 08:00
        start_rot = _rot_anchor_time(t0_day + pd.Timedelta(days=m), start_block)
        fut_exog_raw = future_exog_rotated(state_exog, start_rot, steps=96, start_block=start_block)

        # Build *rotated* baseline (MW) for this rotated day from historical rotated days
        base_mw = baseline_for_future_rotated_day(df_hist_for_baseline, t0_day + pd.Timedelta(days=m),
                                                  K=BASELINE_K, start_block=start_block)
        # Scale for NN
        fut_scaled = fut_exog_raw.copy()
        fut_scaled[ex_cols_std] = exog_scaler.transform(fut_scaled[ex_cols_std].to_numpy())
        fut_scaled['baseline'] = load_scaler.transform(base_mw.values.reshape(-1,1))[:,0]

        Xh_pred  = hist_slice.to_numpy(dtype=np.float32)
        Xf_pred  = fut_scaled.reindex(columns=fut_cols_order).to_numpy(dtype=np.float32)
        y_in_res = np.zeros((HORIZON,1), dtype=np.float32)
        y_in_res[0,0] = Xh_pred[-1, 0]

        yhat_res_scaled = model.predict(
            {"hist": Xh_pred[None,...],
             "future_exog": Xf_pred[None,...],
             "y_in": y_in_res[None,...],
             "is_training": np.array([False], dtype=bool)}, verbose=0
        )[0,:,0]
        yhat_scaled = yhat_res_scaled + fut_scaled['baseline'].to_numpy()
        yhat_mw = load_scaler.inverse_transform(yhat_scaled.reshape(-1,1))[:,0]

        # Roll encoder history with synthesized (scaled) stream
        synth = fut_scaled.copy()
        synth['y'] = yhat_scaled
        synth['residual'] = yhat_scaled - fut_scaled['baseline'].values
        hist_slice = pd.concat([hist_slice, synth.reindex(columns=hist_cols_order)], axis=0).iloc[-SEQ_LEN:]

        # Output the full 96-step window as Date/block rows
        idx = fut_exog_raw.index
        out_rot = pd.DataFrame({
            "Date": idx.normalize(),
            "block": ((idx.hour*60 + idx.minute)//15 + 1).astype(int),
            "forecast_drawal": yhat_mw
        })
        outputs.append(out_rot)

        # (Optional) if you want baseline to see your own forecasts day by day:
        # append only the calendar day D+m portion to df_hist_for_baseline
        dm = (t0_day + pd.Timedelta(days=m)).normalize()
        mask_dm = (out_rot["Date"] == dm)
        if mask_dm.any():
            add = pd.DataFrame({
                "Date": out_rot.loc[mask_dm, "Date"].values,
                "block": out_rot.loc[mask_dm, "block"].values,
                "total_drawal_adj": out_rot.loc[mask_dm, "forecast_drawal"].values
            })
            df_hist_for_baseline = pd.concat([df_hist_for_baseline, add], axis=0).sort_values(["Date","block"])

    long_df = pd.concat(outputs, axis=0).reset_index(drop=True)
    return long_df

###############################################################################

def adjust_forecast_multiday_rt(
    hist_df, forecast_df,
    *,
    first_forecast_date,          # Timestamp/str of the first date in forecast_df
    last_observed_block,          # e.g., 32 (user-provided)
    date_col='Date', block_col='block',
    hist_val_col='total_drawal_adj', fc_val_col='forecast_drawal',
    window=7, z=2.0, blend=False, C=None,
):
    """
    Real-time aware version:
    • anchors day-1 at (first_forecast_date, last_observed_block) actual value
    • rotates the slope order so first predicted step is (B -> B+1) with wrap
    """
    # ---- 1) prep history & mean/std slopes on calendar 1..96 ----
    h = hist_df.copy()
    h['date'] = pd.to_datetime(h[date_col]).dt.date
    days = sorted(h['date'].unique())
    last_days = days[-window:] if len(days) >= window else days

    by_day = {
        d: grp.set_index(block_col)[hist_val_col]
        for d, grp in h[h['date'].isin(last_days)].groupby('date')
    }
    D = len(last_days)
    slopes = np.full((D, 97), np.nan)   # 1..96

    for i, d in enumerate(last_days):
        prev = last_days[i-1] if i > 0 else None
        for k in range(1, 97):
            if k == 1:
                # cross-day slope (96 -> 1)
                if prev is not None and prev in by_day and 96 in by_day[prev] and 1 in by_day[d]:
                    slopes[i, k] = float(by_day[d].loc[1] - by_day[prev].loc[96])
                else:
                    slopes[i, k] = np.nan
            else:
                if (k in by_day[d]) and (k-1 in by_day[d]):
                    slopes[i, k] = float(by_day[d].loc[k] - by_day[d].loc[k-1])
                else:
                    slopes[i, k] = np.nan

    s_mean = np.nanmean(slopes, axis=0)  # index 1..96
    s_std  = np.nanstd( slopes, axis=0)

    # helper: build the rotated order of next 96 block labels starting *after* base_block
    def next_96_blocks_after(base_block):
        # first step is to block (base_block+1), then wrap through 96, then 1..base_block
        seq = list(range(base_block+1, 97)) + list(range(1, base_block+1))
        return np.asarray(seq, dtype=int)

    # ---- 2) loop over forecast days, chaining end-of-day ----
    out_list = []
    last_y_at_96 = None

    # normalize to dates
    f_dates_sorted = sorted(pd.to_datetime(forecast_df[date_col]).dt.date.unique())
    first_day = pd.to_datetime(first_forecast_date).date()

    # anchor for the first forecast date
    hist_first_day = h[h['date'] == first_day]
    if not hist_first_day.empty and (last_observed_block in hist_first_day[block_col].values):
        base_anchor = float(
            hist_first_day.loc[hist_first_day[block_col] == last_observed_block, hist_val_col].iloc[0]
        )
        base_block  = int(last_observed_block)
    else:
        # fallback to "y[prev day, 96]"
        prev = last_days[-1] if last_days else None
        if prev is None or (prev not in by_day) or (96 not in by_day[prev]):
            # ultimate fallback: use first available value in history
            base_anchor = float(h[hist_val_col].dropna().iloc[-1])
            base_block  = 96
        else:
            base_anchor = float(by_day[prev].loc[96])
            base_block  = 96

    for f_date in f_dates_sorted:
        f_sub = forecast_df[pd.to_datetime(forecast_df[date_col]).dt.date == f_date].copy()
        f_sub_blocks = f_sub[block_col].astype(int).tolist()

        # 2a) choose the starting base and the slope order for this day
        if f_date == first_day:
            # real-time day: start from (first_day, last_observed_block) and move forward
            start_base_val   = base_anchor
            slope_block_order = next_96_blocks_after(base_block)   # e.g., 33..96,1..32 if base=32
        else:
            # subsequent days: base is prior adjusted block-96, slopes in natural 1..96 order
            start_base_val = last_y_at_96 if last_y_at_96 is not None else base_anchor
            slope_block_order = np.arange(1, 97, dtype=int)

        # 2b) build trend T per *block label* (1..96)
        # t_prog has 97 values: t_prog[0]=base, then add slope for each next block in slope_block_order
        t_prog = np.zeros(97, dtype=float)
        t_prog[0] = start_base_val
        for i, blk in enumerate(slope_block_order, start=1):
            t_prog[i] = t_prog[i-1] + (s_mean[blk] if np.isfinite(s_mean[blk]) else 0.0)

        # map that progression back to a per-block trend array
        trend = np.full(97, np.nan, dtype=float)  # index 1..96 valid
        for i, blk in enumerate(slope_block_order, start=1):
            trend[blk] = t_prog[i]

        # 2c) pull today’s raw forecast by block
        f_arr = np.full(97, np.nan, dtype=float)
        for _, row in f_sub.iterrows():
            b = int(row[block_col])
            f_arr[b] = float(row[fc_val_col])

        # 2d) anomaly detection on the blocks we actually have
        #     fallback sigma per block: if s_std[blk] is NaN/0, use global mean
        global_sigma = np.nanmean(s_std) if np.isfinite(np.nanmean(s_std)) else 1.0
        sigma = np.where(np.isfinite(s_std), s_std, global_sigma)

        mask = np.zeros_like(f_arr, dtype=bool)
        for b in f_sub_blocks:
            if np.isfinite(trend[b]) and np.isfinite(f_arr[b]) and sigma[b] > 0:
                mask[b] = (abs(f_arr[b] - trend[b]) > z * sigma[b])

        bad = np.where(mask)[0]

        # 2e) contiguous ranges among the available blocks only
        ranges = []
        start = prev = None
        for b in bad:
            if b not in f_sub_blocks:  # skip blocks we didn't forecast for this day
                continue
            if start is None:
                start = prev = b
            elif b == prev + 1:
                prev = b
            else:
                ranges.append((start, prev))
                start = prev = b
        if start is not None:
            ranges.append((start, prev))

        # 2f) patch anomalies for those blocks
        y_adj = f_arr.copy()
        if blend and C is None:
            C = float(global_sigma)
        for (s_, e_) in ranges:
            if blend:
                for k in range(s_, e_ + 1):
                    if k not in f_sub_blocks:
                        continue
                    if not (np.isfinite(trend[k]) and np.isfinite(y_adj[k])):
                        continue
                    alpha = sigma[k] / (sigma[k] + C)
                    y_adj[k] = alpha * y_adj[k] + (1 - alpha) * trend[k]
            else:
                y_adj[s_:e_ + 1] = trend[s_:e_ + 1]

        last_y_at_96 = y_adj[96] if 96 in f_sub_blocks and np.isfinite(y_adj[96]) else last_y_at_96
        f_sub['adjusted_drawal'] = f_sub[block_col].map(lambda k: float(y_adj[int(k)]) if np.isfinite(y_adj[int(k)]) else np.nan)
        out_list.append(f_sub)

    return pd.concat(out_list, ignore_index=True).sort_values([date_col, block_col])


import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt


def smooth_butter_per_day_rt(
    df,
    *,
    date_col='Date',
    block_col='block',
    val_col='adjusted_drawal',
    out_col='final_drawal',
    median_win=3,
    cutoff_period_blocks=8,
    order=3,
    anchor_endpoints=True,
    min_points=12,        # need enough points to filter stably
):
    """
    Median(3) -> zero-phase Butterworth low-pass on the AVAILABLE blocks per day.
    No NaNs are passed to filtfilt. Days with too-few points are copied as-is.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    outs = []
    for d, g in df.groupby(df[date_col].dt.date):
        g = g.sort_values(block_col).copy()
        y = g[val_col].to_numpy(float)

        # contiguous segment among available blocks
        # (if it's split, we still filter the whole set as one; for split days you can segment further)
        present_mask = np.isfinite(y)
        if present_mask.sum() < min_points:
            gg = g.copy()
            gg[out_col] = y   # leave as-is
            outs.append(gg)
            continue

        y_in = y[present_mask]

        # median spike killer (only on present segment)
        if median_win and median_win > 1:
            if (median_win % 2) == 0:
                median_win += 1
            y_med = medfilt(y_in, kernel_size=median_win)
        else:
            y_med = y_in

        # zero-phase low-pass; sampling is 1 sample per block.
        fc = 1.0 / float(cutoff_period_blocks)   # cycles per block
        Wn = min(0.99, 2.0 * fc)                 # normalized to Nyquist (0..1)
        b, a = butter(order, Wn, btype='low', analog=False)

        try:
            z = filtfilt(b, a, y_med, method='pad')
        except Exception:
            # fallback: no filtering if numerics are unhappy
            z = y_med.copy()

        # preserve mean on the present segment
        z += (y_in.mean() - z.mean())

        # optional: anchor ends to original present segment
        if anchor_endpoints and len(z) >= 2:
            z[0]  = y_in[0]
            z[-1] = y_in[-1]

        # write back only for present blocks; leave others untouched (they weren't there anyway)
        gg = g.copy()
        gg[out_col] = np.nan
        gg.loc[present_mask, out_col] = z
        outs.append(gg)

    sm = pd.concat(outs, ignore_index=True)
    return sm

############################################   Weather  ###############################################################

def interpolate_weather(input_csv,state):

    print("=============== Starting Weather Processing ================")

    df = pd.read_csv(input_csv, sep=',')
    def keep_hist_then_forecast(df, keys=('state','location'), date_col='date', type_col='type'):
        # normalize the date column to date (not datetime)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        hist_dates = (
            df.loc[df[type_col] == 'HISTORICAL', list(keys) + [date_col]]
            .drop_duplicates()
        )
        hist_idx = pd.MultiIndex.from_frame(hist_dates)
        is_conflicting_forecast = (
            (df[type_col] == 'FORECAST') &
            df.set_index(list(keys) + [date_col]).index.isin(hist_idx)
        )
        out = df.loc[~is_conflicting_forecast].reset_index(drop=True)
        return out
    df = keep_hist_then_forecast(df, keys=('state','location'), date_col='date', type_col='type')
    df['time'] = df['time'].astype(str).str.strip()
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'])
    def distribute_hourly_prec_to_quarterly(s_hourly: pd.Series, full_range: pd.DatetimeIndex) -> pd.Series:
        if s_hourly is None or s_hourly.empty:
            return pd.Series(0.0, index=full_range)
        s = s_hourly.dropna().astype(float).copy()
        # snap to exact hour (handles slight drifts like 18:59:30)
        s.index = pd.to_datetime(s.index).round('H')
        s = s.groupby(s.index).mean()  # collapse duplicates post-rounding
        # four quarter-ends: T, T-15m, T-30m, T-45m
        s0 = s.copy()
        s1 = s.copy(); s1.index = s1.index - pd.Timedelta(minutes=15)
        s2 = s.copy(); s2.index = s2.index - pd.Timedelta(minutes=30)
        s3 = s.copy(); s3.index = s3.index - pd.Timedelta(minutes=45)
        out = pd.concat([s0, s1, s2, s3]).groupby(level=0).sum() / 4.0
        out = out.reindex(full_range).fillna(0.0)  # align to 15-min grid
        return out

    # 3) Weather columns to interpolate
    weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
        'precipitation', 'rain', 'showers', 'snowfall', 'snow_depth',
        'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
        'wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m',
        'sunshine_duration', 'direct_radiation', 'direct_radiation_instant'
    ]

    # 5) Process each state
    SPECIAL_STATE = "DADRA AND NAGAR HAVELI AND DAMAN AND DIU"

    def process_and_save(df_state_in, out_basename, allowed_locations=None):
        df_state = df_state_in.copy()
        if allowed_locations is not None:
            df_state = df_state[df_state['location'].isin(allowed_locations)]
        if df_state.empty:
            print(f"[skip] No rows for {out_basename} with locations={allowed_locations}")
            return

        df_state = df_state.set_index('datetime').sort_index()

        # a) full 15-min grid for every day in this (sub)state
        start = df_state.index.min().normalize()
        end   = df_state.index.max().normalize() + pd.Timedelta(hours=23, minutes=45)
        full_range = pd.date_range(start=start, end=end, freq='15T')

        # b) for each location, de-dup per timestamp -> reindex -> interpolate
        loc_frames = []
        agg_map = {c: 'mean' for c in weather_cols}

        for loc, df_loc in df_state.groupby('location'):
            # keep only the weather columns; dedupe identical timestamps (mean)
            df_loc = df_loc[weather_cols].copy()
            df_loc = df_loc.groupby(df_loc.index).agg(agg_map)

            # ---- PRECIP: distribute hourly to 15-min quarters (right-aligned) ----
            precip15 = None
            if 'precipitation' in df_loc.columns:
                precip15 = distribute_hourly_prec_to_quarterly(df_loc['precipitation'], full_range)

            # align to full 15-min grid
            df_loc = df_loc.reindex(full_range)

            # interpolate other weather columns (NOT precipitation)
            non_precip_cols = [c for c in weather_cols if c != 'precipitation' and c in df_loc.columns]
            if non_precip_cols:
                df_loc[non_precip_cols] = (
                    df_loc[non_precip_cols]
                    .interpolate(method='time')
                    .ffill()
                    .bfill()
                )

            # set the quartered precipitation (no interpolation)
            if precip15 is not None:
                df_loc['precipitation'] = precip15.values

            # add location tag and collect
            df_loc['location'] = loc
            loc_frames.append(df_loc)

        if not loc_frames:
            print(f"[skip] No locations produced for {out_basename}")
            return

        # c) combine back
        df_interp = pd.concat(loc_frames)

        # d) add date and time_block
        df_interp['date'] = df_interp.index.date
        df_interp['time_block'] = (df_interp.index.hour * 60 + df_interp.index.minute)//15 + 1

        # e) add state column (optional)
        df_interp['state'] = out_basename.replace('_', ' ')

        # f) reset index and select columns
        cols = ['date', 'time_block', 'location'] + weather_cols
        df_interp = df_interp[cols]
        return df_interp


    # main loop with special split
    states = [state]
    ALL_STATES_NORM = [s.upper() for s in states]
    ALL_STATES = ["Rajasthan"]

    # main loop with special split
    for state_name, df_state in df.groupby('state', sort=False):
        # --- skip states not requested ---
        if ALL_STATES and state_name.upper() not in ALL_STATES_NORM:
            continue
        if state_name == SPECIAL_STATE:
            # 1) Dadra & Nagar Haveli → only that location
            weath = process_and_save(
                df_state,
                out_basename="Dadra_and_Nagar_Haveli",
                allowed_locations=["DADRA AND NAGAR HAVELI"]
            )
            # 2) Daman & Diu → both locations
            weath = process_and_save(
                df_state,
                out_basename="Daman_and_Diu",
                allowed_locations=["DAMAN", "DIU"]
            )
        else:
            # default: one file per state (spaces → underscores)
            weath = process_and_save(
                df_state,
                out_basename=state_name.replace(' ', '_'),
                allowed_locations=None
            )
    print("=============== Weather Processing Completed ================")
    return weath

#######################################################################################################################
