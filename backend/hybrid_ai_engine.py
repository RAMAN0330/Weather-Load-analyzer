import os
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .hybrid_ai_model import build_model, SEQ_LEN, HORIZON, DERIV_LAMBDA

logger = logging.getLogger(__name__)

@dataclass
class Artifacts:
    load_scaler: MinMaxScaler
    exog_scaler: StandardScaler
    history_cols: list
    future_cols: list

# ------------------
# Constants
# ------------------
BASELINE_K     = 2
_BASELINE_POOL = 7      # candidate window for anomaly-aware baseline
_OUTLIER_SIGMA = 1.8    # days whose daily mean deviates > 1.8σ are excluded
WT_TEMP = 0.8
WT_PRECIP = 1.2
WT_SUNDAY = 0.8
WT_HOLIDAY = 1.2
W_MAX = 5.0

# ------------------
# Holiday Helpers (Simplified from helper.py)
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
    return pd.Timestamp(rot_date) + pd.Timedelta(minutes=(start_block - 1) * 15)

# ------------------
# Rotated Helpers
# ------------------

def assign_rotated_date(index_like, start_block: int = 33):
    idx = pd.DatetimeIndex(index_like)
    block = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    base = idx.normalize()
    shift = pd.to_timedelta((block < start_block).astype(int), unit="D")
    rot_date = base - shift
    return pd.Series(rot_date, index=idx)

def rotated_block_number(index_like, start_block: int = 33):
    idx = pd.DatetimeIndex(index_like)
    true_block = ((idx.hour * 60 + idx.minute) // 15 + 1).astype(int)
    rot_block = ((true_block - start_block) % 96) + 1
    return rot_block

def add_time_features_rotated(index_like, start_block: int = 33):
    idx = pd.DatetimeIndex(index_like)
    rot_date_series = assign_rotated_date(idx, start_block=start_block)
    rot_block = rotated_block_number(idx, start_block=start_block)

    theta_b = 2 * np.pi * (rot_block - 1) / 96.0
    # DOW from rotated dates
    dow = pd.DatetimeIndex(rot_date_series).dayofweek.astype(int)
    theta_d = 2 * np.pi * dow / 7.0

    holidays = get_indian_holidays_set()
    rot_date_norm = pd.DatetimeIndex(rot_date_series).normalize()
    is_holiday = rot_date_norm.isin(holidays).astype(int)
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

def causal_baseline_series_rotated(df_load: pd.DataFrame, K: int, start_block: int = 33) -> pd.Series:
    """Causal K-day rolling baseline with anomalous days excluded — consistent with inference."""
    df = df_load.copy()
    if 'Datetime' not in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta((df['block']-1)*15, unit='m')

    df = df.sort_values('Datetime')
    idx = df['Datetime']
    df['rot_date']  = assign_rotated_date(idx, start_block=start_block).to_numpy()
    df['rot_block'] = rotated_block_number(idx, start_block=start_block)

    piv = df.pivot_table(index='rot_date', columns='rot_block',
                         values='total_drawal_adj', aggfunc='mean').sort_index()
    day_means = piv.mean(axis=1)

    baselines = []
    for i in range(len(piv)):
        pool_slice = piv.iloc[max(0, i - _BASELINE_POOL):i]
        if len(pool_slice) == 0:
            baselines.append(pd.Series([float('nan')] * piv.shape[1], index=piv.columns))
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
    base_long = base_rot.stack(dropna=False).rename('baseline').reset_index()
    base_long['Datetime'] = pd.to_datetime(base_long['rot_date']).apply(
        lambda d: _rot_anchor_time(d, start_block)
    ) + pd.to_timedelta((base_long['rot_block'] - 1) * 15, unit='m')

    return base_long.set_index('Datetime')['baseline'].sort_index()

def build_state_series_rotated(df_load: pd.DataFrame, state_exog: pd.DataFrame, start_block: int = 33) -> pd.DataFrame:
    df = df_load.copy()
    df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta((df['block'] - 1) * 15, unit='m')
    df = df.set_index('Datetime').sort_index()

    first_day = df['Date'].min()
    mask_head = (df['Date'] == first_day) & (df['block'] < start_block)
    df = df.loc[~mask_head].copy()

    exog = state_exog.reindex(df.index).ffill().bfill()
    base_rot = causal_baseline_series_rotated(
        df[['Date', 'block', 'total_drawal_adj']].reset_index(drop=False),
        K=BASELINE_K, start_block=start_block
    ).reindex(df.index).ffill().bfill()

    resid = df['total_drawal_adj'] - base_rot
    tfeat = add_time_features_rotated(df.index, start_block=start_block)

    full = pd.concat([
        df[['total_drawal_adj']].rename(columns={'total_drawal_adj': 'y'}),
        base_rot.rename('baseline'),
        resid.rename('residual'),
        exog,
        tfeat
    ], axis=1).dropna(subset=['y', 'baseline'])
    return full

def make_windows_rotated(full: pd.DataFrame, seq_len: int = 96, horizon: int = 96, start_block: int = 33):
    full = full.copy().sort_index()
    full['rot_date'] = assign_rotated_date(full.index, start_block=start_block).to_numpy()
    
    hist_cols = ['residual', 'temperature', 'humidity', 'precipitation', 
                 'rot_block_sin', 'rot_block_cos', 'dow_sin', 'dow_cos', 
                 'is_sunday', 'is_holiday', 'baseline']
    fut_cols  = ['temperature', 'humidity', 'precipitation', 
                 'rot_block_sin', 'rot_block_cos', 'dow_sin', 'dow_cos', 
                 'is_sunday', 'is_holiday', 'baseline']

    Xh, Xf, Y, Yin, stamps = [], [], [], [], []
    dates_sorted = sorted(pd.unique(full['rot_date']))
    
    for d in dates_sorted:
        t0 = _rot_anchor_time(d, start_block)
        hist_start = t0 - pd.to_timedelta(seq_len*15, unit='m')
        
        hist_slice = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_slice  = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15*horizon))]
        
        if len(hist_slice) != seq_len or len(fut_slice) != horizon:
            continue

        y_tgt = fut_slice[['residual']].values.astype(np.float32)
        y_in  = np.empty_like(y_tgt, dtype=np.float32)
        y_in[0,0]  = hist_slice['residual'].iloc[-1]
        y_in[1:,0] = y_tgt[:-1,0]

        Xh.append(hist_slice[hist_cols].values.astype(np.float32))
        Xf.append(fut_slice[fut_cols].values.astype(np.float32))
        Y.append(y_tgt)
        Yin.append(y_in)
        stamps.append(t0)

    return (np.asarray(Xh), np.asarray(Xf), np.asarray(Y), np.asarray(Yin),
            pd.to_datetime(stamps), hist_cols, fut_cols)

def make_weather_action_weights_rotated(full, seq_len=96, horizon=96, start_block=33):
    full = full.copy().sort_index()
    full['rot_date'] = assign_rotated_date(full.index, start_block=start_block).to_numpy()
    W = []
    for d in sorted(pd.unique(full['rot_date'])):
        t0 = _rot_anchor_time(d, start_block)
        hist_start = t0 - pd.to_timedelta(seq_len*15, unit='m')
        hist_slice = full.loc[(full.index >= hist_start) & (full.index < t0)]
        fut_slice  = full.loc[(full.index >= t0) & (full.index < t0 + pd.Timedelta(minutes=15*horizon))]
        if len(hist_slice) != seq_len or len(fut_slice) != horizon:
            continue
        T = fut_slice['temperature'].to_numpy()
        P = fut_slice['precipitation'].to_numpy()
        dT = np.r_[0.0, np.diff(T)]
        is_sun = fut_slice['is_sunday'].to_numpy().astype(float)
        is_hol = fut_slice['is_holiday'].to_numpy().astype(float)
        w = 1.0 + WT_TEMP * np.abs(dT) + WT_PRECIP * np.maximum(0.0, P) + WT_SUNDAY * is_sun + WT_HOLIDAY * is_hol
        W.append(np.clip(w, 1.0, W_MAX).astype('float32'))
    return np.asarray(W, dtype='float32')

def baseline_for_future_rotated_day(df_hist_rot: pd.DataFrame, rot_day: pd.Timestamp, K:int=BASELINE_K, start_block:int=33) -> pd.Series:
    df = df_hist_rot.copy()
    df['Datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta((df['block']-1)*15, unit='m')
    ser = df.set_index('Datetime')['total_drawal_adj'].sort_index()
    idx = ser.index
    rdate = assign_rotated_date(idx, start_block=start_block).values
    rblk = rotated_block_number(idx, start_block=start_block)
    tmp = pd.DataFrame({"rot_date": rdate, "rot_block": rblk, "y": ser.values})
    piv_rot = tmp.pivot_table(index='rot_date', columns='rot_block', values='y', aggfunc='mean').sort_index()
    candidates = piv_rot.loc[piv_rot.index < pd.Timestamp(rot_day)]
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
        hist = piv_rot.tail(1)
    vec = hist.mean(axis=0)
    vec.index = pd.RangeIndex(1, 97)
    start_t = _rot_anchor_time(rot_day, start_block)
    idx_out = pd.date_range(start=start_t, periods=96, freq='15T')
    return pd.Series(vec.values, index=idx_out, name='baseline')

def run_attention_hybrid_ai(df_load_clean, state_exog, forecast_start_date, forecast_days=1, start_block=33, last_observed_block=32, artifacts_dir=None):
    if artifacts_dir is None:
        artifacts_dir = Path("artifacts/hybrid_ai")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    project_name = "hybrid_attention"
    
    # 1. Build stream
    full = build_state_series_rotated(df_load_clean, state_exog, start_block=start_block)
    
    # 2. Scalers
    load_scaler = MinMaxScaler()
    full['y'] = load_scaler.fit_transform(full['y'].to_numpy().reshape(-1,1))[:,0]
    full['baseline'] = load_scaler.transform(full['baseline'].to_numpy().reshape(-1,1))[:,0]
    full['residual'] = full['y'] - full['baseline']
    
    ex_cols_std = ['temperature','humidity','precipitation']
    exog_scaler = StandardScaler()
    full[ex_cols_std] = exog_scaler.fit_transform(full[ex_cols_std])
    
    # 3. Windows
    Xh, Xf, Y, Yin, stamps, hist_cols, fut_cols = make_windows_rotated(full, seq_len=SEQ_LEN, horizon=HORIZON, start_block=start_block)
    if len(Xh) < 12:
         logger.warning(f"Not enough samples for hybrid AI: {len(Xh)}")
         return None
    
    W = make_weather_action_weights_rotated(full, seq_len=SEQ_LEN, horizon=HORIZON, start_block=start_block)
    
    # 4. Build/Train Model
    hp = {'lr': 1e-3, 'epochs': 30} # Fast training for integration
    model = build_model(hp, Fh=len(hist_cols), Ff=len(fut_cols))
    
    # Training
    N = Xh.shape[0]
    model.fit(
        {"hist": Xh, "future_exog": Xf, "y_in": Yin, "is_training": np.ones((N,), bool)},
        Y,
        sample_weight=W,
        epochs=hp['epochs'],
        batch_size=16,
        verbose=0
    )
    
    # 5. Forecast
    t0_day = pd.Timestamp(forecast_start_date).normalize()
    effective_last_block = max(1, min(int(last_observed_block), 32))
    desired_hist_end = t0_day + pd.Timedelta(minutes=(effective_last_block - 1) * 15)
    
    # History for encoder
    hist_end_time = full.index[full.index <= desired_hist_end].max()
    hist_start = hist_end_time - pd.to_timedelta(SEQ_LEN*15 - 15, unit='m')
    hist_slice = full.loc[(full.index >= hist_start) & (full.index <= hist_end_time)]
    if len(hist_slice) != SEQ_LEN:
         expected_idx = pd.date_range(hist_start, hist_end_time, freq='15min')
         hist_slice = hist_slice.reindex(expected_idx).ffill().bfill()
    hist_slice = hist_slice.reindex(columns=hist_cols)
    
    df_hist_for_baseline = df_load_clean.copy()
    outputs = []
    
    for m in range(int(forecast_days)):
        start_rot = _rot_anchor_time(t0_day + pd.Timedelta(days=m), start_block)
        
        # Future exog
        idx_fut = pd.date_range(start=start_rot, periods=HORIZON, freq='15T')
        fut_exog_raw = state_exog.reindex(idx_fut).ffill().bfill()
        tfeat_fut = add_time_features_rotated(idx_fut, start_block=start_block)
        base_mw = baseline_for_future_rotated_day(df_hist_for_baseline, t0_day + pd.Timedelta(days=m), K=BASELINE_K, start_block=start_block)
        
        fut_scaled = fut_exog_raw.copy()
        fut_scaled[ex_cols_std] = exog_scaler.transform(fut_scaled[ex_cols_std].to_numpy())
        fut_scaled['baseline'] = load_scaler.transform(base_mw.values.reshape(-1,1))[:,0]
        fut_scaled = pd.concat([fut_scaled, tfeat_fut], axis=1)
        
        Xh_pred = hist_slice.to_numpy(dtype=np.float32)
        Xf_pred = fut_scaled.reindex(columns=fut_cols).to_numpy(dtype=np.float32)
        y_in_res = np.zeros((HORIZON, 1), dtype=np.float32)
        y_in_res[0,0] = Xh_pred[-1, 0]
        
        yhat_res_scaled = model.predict(
            {"hist": Xh_pred[None,...], "future_exog": Xf_pred[None,...], "y_in": y_in_res[None,...], "is_training": np.array([False], dtype=bool)}, 
            verbose=0
        )[0,:,0]
        
        yhat_scaled = yhat_res_scaled + fut_scaled['baseline'].to_numpy()
        yhat_mw = load_scaler.inverse_transform(yhat_scaled.reshape(-1,1))[:,0]
        
        # Output
        idx = fut_exog_raw.index
        out_rot = pd.DataFrame({
            "Date": idx.normalize(),
            "block": ((idx.hour*60 + idx.minute)//15 + 1).astype(int),
            "forecast_drawal": yhat_mw
        })
        outputs.append(out_rot)
        
        # Roll history
        synth = fut_scaled.copy()
        synth['y'] = yhat_scaled
        synth['residual'] = yhat_res_scaled
        hist_slice = pd.concat([hist_slice, synth.reindex(columns=hist_cols)], axis=0).iloc[-SEQ_LEN:]

    return pd.concat(outputs, axis=0).reset_index(drop=True)
