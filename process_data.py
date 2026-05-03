import warnings
from datetime import date

import holidays
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

from weatherloadIntegrator import WeatherLoadIntegrator


class LoadProcessor:
    """
    Handles loading, cleaning, and preprocessing of load data.
    """
    def __init__(self, 
                 diff_threshold: float = 600.0,
                 strict_spike: bool = True,
                 zero_end_date: str = "2025-10-08",
                 zero_end_block: int = 96,
                 drop_win: int = 13,
                 drop_delta_mw: float = 800.0,
                 drop_factor: float = 1.00,
                 max_consec_n: int = 4):
        self.diff_threshold = diff_threshold
        self.strict_spike = strict_spike
        self.zero_end_date = zero_end_date
        self.zero_end_block = zero_end_block
        self.drop_win = drop_win
        self.drop_delta_mw = drop_delta_mw
        self.drop_factor = drop_factor
        self.max_consec_n = max_consec_n
        
    def process(self, input_csv: str, start_date: str = '2025-07-03') -> pd.DataFrame:
        """
        Main method to process load data from CSV.
        """
        print("Reading load data...")
        df = pd.read_csv(input_csv)

        for col in ['demand_met', 'field_4_', 'own_generation', 'isgs', 'importdata']:
            if col in df.columns:
                df[col] = (df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True))
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        df = df[df['date'] >= start_dt]

        times = pd.to_datetime(df['time_slot'].astype(str).str.strip(), format='%H:%M', errors='coerce')
        df['time_block'] = (times.dt.hour * 4 + (times.dt.minute // 15) + 1).astype('Int64')
        df = df.dropna(subset=['time_block'])
        df = self._aggregate_and_align(df)
        df = self._apply_overrides(df)
        df = self._clean_data(df)

        df["Datetime"] = df["date"] + pd.to_timedelta((df["time_block"] - 1) * 15, unit="m")
        
        return df

    def _aggregate_and_align(self, df: pd.DataFrame) -> pd.DataFrame:
        if "state" in df.columns:
            df = df[df["state"]=="Odisha"]
        
        agg_input_cols = [c for c in ['demand_met', 'isgs', 'importdata'] if c in df.columns]
        
        agg = (
            df
            .groupby(['date', 'time_block'], as_index=False)[agg_input_cols]
            .mean()
            .round(2)
        )

        all_blocks = pd.DataFrame({'time_block': np.arange(1, 97)})
        dates_df = pd.DataFrame({'date': sorted(agg['date'].unique())})
        dates_df['key'] = 1
        blocks_df = all_blocks.copy(); blocks_df['key'] = 1
        full = dates_df.merge(blocks_df, on='key').drop(columns='key')

        merged = full.merge(agg, on=['date', 'time_block'], how='left')

        rename_map = {}
        if 'demand_met' in merged.columns:
            rename_map['demand_met'] = 'total_drawal'
        out = merged.rename(columns=rename_map)

        if 'total_drawal' not in out.columns: out['total_drawal'] = 0.0
        for c in ['total_drawal', 'isgs', 'importdata']:
            if c in out.columns: out[c] = out[c].fillna(0.0)

        out = out.sort_values(['date', 'time_block']).reset_index(drop=True)
        return out

    def _apply_overrides(self, out: pd.DataFrame) -> pd.DataFrame:
        july30 = pd.Timestamp(2025, 7, 30); july31 = pd.Timestamp(2025, 7, 31); aug1 = pd.Timestamp(2025, 8, 1)
        cols_to_override = [c for c in ['total_drawal', 'isgs', 'importdata'] if c in out.columns]
        
        if cols_to_override:
            # July 30 83-96 -> July 31
            src_30 = out.loc[(out['date'] == july30) & (out['time_block'].between(83, 96)),
                             ['time_block'] + cols_to_override]
            if not src_30.empty:
                src_map = src_30.set_index('time_block')[cols_to_override]
                for tb, vals in src_map.iterrows():
                    tgt = (out['date'] == july31) & (out['time_block'] == tb)
                    if tgt.any():
                        out.loc[tgt, cols_to_override] = vals.values
            
            # July 31 1-31 -> Aug 1
            src_31 = out.loc[(out['date'] == july31) & (out['time_block'].between(1, 31)),
                             ['time_block'] + cols_to_override]
            if not src_31.empty:
                src_map = src_31.set_index('time_block')[cols_to_override]
                for tb, vals in src_map.iterrows():
                    tgt = (out['date'] == aug1) & (out['time_block'] == tb)
                    if tgt.any():
                        out.loc[tgt, cols_to_override] = vals.values
        return out

    def _clean_data(self, out: pd.DataFrame) -> pd.DataFrame:
        check_cols = ['total_drawal', 'isgs', 'importdata']
        cols_to_fix = [c for c in check_cols if c in out.columns]
        
        _end_date_val = pd.to_datetime(self.zero_end_date) if self.zero_end_date else None

        if cols_to_fix:
            out, _ = self.fix_continuity_spikes(out, cols_to_fix, self.diff_threshold, self.strict_spike)
            out, _ = self.interpolate_zeros_upto(out, cols_to_fix, _end_date_val, self.zero_end_block)
            out, _ = self.smooth_local_drops(out, cols_to_fix, win=self.drop_win, delta=self.drop_delta_mw, factor=self.drop_factor, max_consec=self.max_consec_n)
        return out

    @staticmethod
    def _smooth_spikes_one_day(s: pd.Series, thr: float, strict: bool) -> tuple[pd.Series, int]:
        vals = s.values.astype(float)
        n = len(vals)
        if n < 2: return s, 0
        bad = np.zeros(n, dtype=bool)
        if n >= 3:
            left  = np.abs(vals[1:-1] - vals[0:-2])
            right = np.abs(vals[2:]   - vals[1:-1])
            avg2  = (left + right) / 2.0
            bad[1:-1] = (avg2 > thr) & (left > thr) & (right > thr) if strict else (avg2 > thr)
        if n >= 2:
            if np.abs(vals[1] - vals[0]) > thr: bad[0]  = True
            if np.abs(vals[-1] - vals[-2]) > thr: bad[-1] = True
        n_flagged = int(bad.sum())
        if n_flagged == 0: return s, 0
        fixed = vals.copy()
        fixed[bad] = np.nan
        fixed = pd.Series(fixed).interpolate(method='linear', limit_direction='both').values
        return pd.Series(fixed, index=s.index), n_flagged

    def fix_continuity_spikes(self, df: pd.DataFrame, cols: list[str], thr: float, strict: bool,
                              date_col='date', block_col='time_block') -> tuple[pd.DataFrame, dict]:
        out = df.copy().sort_values([date_col, block_col]).reset_index(drop=True)
        counts = {c: 0 for c in cols}
        for c in cols: out[c] = out[c].astype(float)
        for c in cols:
            def _apply(g):
                fixed, n_bad = self._smooth_spikes_one_day(g[c], thr, strict)
                counts[c] += n_bad
                g[c] = fixed.values
                return g
            out = out.groupby(date_col, group_keys=False).apply(_apply)
        return out, counts

    @staticmethod
    def _smooth_local_drops_one_day(s: pd.Series, win: int, delta: float, factor: float, max_consec: int) -> tuple[pd.Series, int]:
        x = s.astype(float).copy()
        med = x.rolling(win, center=True, min_periods=1).median()
        gap = med - x
        cand = (gap > delta) & (x < factor * med)
        if cand.any():
            idx = np.where(cand.values)[0]
            fix_mask = np.zeros_like(cand.values, dtype=bool)
            if idx.size:
                start = idx[0]; prev = idx[0]
                for i in idx[1:]+1:
                    if i-1 != prev:
                        run = np.arange(start, prev+1)
                        if len(run) <= max_consec: fix_mask[run] = True
                        start = i-1
                    prev = i-1
                run = np.arange(start, prev+1)
                if len(run) <= max_consec: fix_mask[run] = True
            n_fixed = int(fix_mask.sum())
            if n_fixed:
                x.values[fix_mask] = np.nan
                x = x.interpolate(method='linear', limit_direction='both')
                x = x.fillna(method='bfill').fillna(method='ffill')
                return x, n_fixed
        return x, 0

    def smooth_local_drops(self, df: pd.DataFrame, cols: list[str], win: int, delta: float, factor: float,
                           max_consec: int, date_col='date', block_col='time_block') -> tuple[pd.DataFrame, dict]:
        out = df.copy().sort_values([date_col, block_col]).reset_index(drop=True)
        counts = {c: 0 for c in cols}
        for c in cols: out[c] = out[c].astype(float)
        for c in cols:
            def _apply(g):
                y, k = self._smooth_local_drops_one_day(g[c], win, delta, factor, max_consec)
                counts[c] += k
                g[c] = y.values
                return g
            out = out.groupby(date_col, group_keys=False).apply(_apply)
        return out, counts

    @staticmethod
    def interpolate_zeros_upto(df: pd.DataFrame, cols: list[str], end_date: date | None, end_block: int | None,
                               date_col='date', block_col='time_block') -> tuple[pd.DataFrame, dict]:
        out = df.copy().sort_values([date_col, block_col]).reset_index(drop=True)
        counts = {c: 0 for c in cols}
        if (end_date is None) or (end_block is None): return out, counts
        for c in cols: out[c] = out[c].astype(float)
        mask_before = out[date_col] < end_date
        mask_on     = (out[date_col] == end_date) & (out[block_col] <= end_block)
        eligible    = mask_before | mask_on
        for d, g in out.groupby(date_col, group_keys=False):
            day_idx = g.index
            for c in cols:
                s = out.loc[day_idx, c]
                elig_zero = eligible.loc[day_idx] & (s == 0.0)
                n_zero = int(elig_zero.sum())
                if n_zero == 0: continue
                nonzero_mask = (s != 0.0) & (~s.isna())
                if nonzero_mask.sum() == 0: continue
                s2 = s.copy()
                s2[elig_zero] = np.nan
                s2 = s2.interpolate(method='linear', limit_direction='both')
                s2 = s2.fillna(s)
                out.loc[day_idx, c] = s2.values
                counts[c] += n_zero
        return out, counts


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, cyclical, and weather-interaction features."""
    df = df.copy()
    dt = pd.to_datetime(df["Datetime"])
    in_holidays = holidays.India()

    # ── base temporal ──
    df["date_only"] = dt.dt.date
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["time_block"] = df["hour"] * 4 + (df["minute"] // 15)

    # ── categorical flags ──
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.map(lambda x: int(x in in_holidays)).values

    # ── cyclical encodings ──
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["block_sin"] = np.sin(2 * np.pi * df["time_block"] / 96)
    df["block_cos"] = np.cos(2 * np.pi * df["time_block"] / 96)

    # ── period flags ──
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 18, 19, 20]).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
    df["is_business_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

    # ── weather interactions (only if temperature available) ──
    if "temperature" in df.columns:
        t = df["temperature"]
        df["temp_squared"] = t ** 2
        df["temp_cubed"] = t ** 3
        df["temperature_sin"] = t * np.sin(2 * np.pi * df["hour"] / 24)

    # cleanup helper column
    df.drop(columns=["date_only"], inplace=True, errors="ignore")
    return df


class WeatherProcessor:
    """
    Handles loading and preprocessing of weather data.
    """
    def __init__(self):
        pass

    def process(self, parquet_path: str) -> pd.DataFrame:
        """
        Reads, filters and prepares weather data.
        """
        print("Reading weather data...")
        weather = pd.read_parquet(parquet_path)
        
        # Construct Datetime
        weather['Datetime'] = pd.to_datetime(
            weather['date'].astype(str) + ' ' + weather['time'].astype(str), 
            errors='coerce'
        )
        
        # Filter duplicates: Prefer HISTORICAL over FORECAST
        if 'type' in weather.columns:
            weather = weather.sort_values('type')
            weather = weather.drop_duplicates(subset=['Datetime', 'location'], keep='last')
        else:
            weather = weather.drop_duplicates(subset=['Datetime', 'location'], keep='last')
            
        return weather


class DataPipeline:
    """
    Orchestrates the data processing pipeline.
    """
    def __init__(self, load_path: str, weather_path: str, output_path: str):
        self.load_path = load_path
        self.weather_path = weather_path
        self.output_path = output_path
        self.load_processor = LoadProcessor()
        self.weather_processor = WeatherProcessor()

    def run(self):
        # 1. Process Load
        load_df = self.load_processor.process(self.load_path)
        print(f"Load data processed: {load_df.shape}")
        
        # 2. Process Weather
        weather_df = self.weather_processor.process(self.weather_path)
        print(f"Weather data processed: {weather_df.shape}")

        # 3. Compute State Specific Weather
        print("Computing state-level weather features...")
        load_series = load_df.set_index("Datetime")["total_drawal"]
        
        builder = WeatherLoadIntegrator(
            weather_df=weather_df,
            load_series=load_series
        )

        state_weather = builder.compute_state_level_features()
        print(f"State weather features: {list(state_weather.columns)} — {state_weather.shape}")

        # 4. Merge
        print("Merging data...")
        final_df = pd.merge(load_df, state_weather, left_on="Datetime", right_index=True, how="left")

        # Interpolate all weather columns that were produced
        for col in state_weather.columns:
            if col in final_df.columns:
                final_df[col] = final_df[col].interpolate(
                    method="linear",
                    limit_direction="both"
                )
        
        # 5. Add temporal & interaction features
        print("Adding time features...")
        final_df = add_time_features(final_df)
        print(f"Final columns ({len(final_df.columns)}): {list(final_df.columns)}")

        # 6. Save
        final_df.to_csv(self.output_path, index=False)
        print(f"Final data saved to: {self.output_path}")
        print(final_df.head())

def main():
    # Paths
    load_csv_path = "/home/raman/insights/Forecast/Rajasthan/merit/statewise_merit_generation.csv"
    weather_parquet_path = "/mnt/c/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/data/weather_ODISHA.parquet"
    output_path = "/mnt/c/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD/final_data.csv"

    pipeline = DataPipeline(load_csv_path, weather_parquet_path, output_path)
    pipeline.run()

if __name__ == "__main__":
    main()
