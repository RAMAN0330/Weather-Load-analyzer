from typing import Optional, List, Tuple, Dict
from datetime import date
import pandas as pd

class WeatherDataProcessor:
    """Processes weather data with interpolation and quartile distribution."""
    
    WEATHER_COLS = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
        'precipitation', 'rain', 'showers', 'snowfall', 'snow_depth',
        'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
        'wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m',
        'sunshine_duration', 'direct_radiation', 'direct_radiation_instant'
    ]
    
    def __init__(self, weather_path: str, state: str = "HARYANA"):
        """
        Initialize the weather data processor.
        
        Args:
            weather_path: Path to the weather parquet file
            state: State name to filter data
        """
        self.weather_path = weather_path
        self.state = state
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and filter weather data."""
        weather = pd.read_parquet(self.weather_path, engine='pyarrow')
        weather = weather[weather["state"] == self.state]
        self.df = weather.copy()
        return self.df
    
    def prepare_datetime(self) -> None:
        """Prepare datetime columns from date and time."""
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce').dt.date
        self.df['time'] = self.df['time'].astype(str).str.strip()
        self.df['datetime'] = pd.to_datetime(
            self.df['date'].astype(str) + ' ' + self.df['time'], 
            errors='coerce'
        )
    
    @staticmethod
    def keep_hist_then_forecast(
        df: pd.DataFrame, 
        keys: Tuple[str, ...] = ('state', 'location'), 
        date_col: str = 'date', 
        type_col: str = 'type'
    ) -> pd.DataFrame:
        """
        Keep historical data and remove forecast data for overlapping dates.
        
        Args:
            df: Input dataframe
            keys: Tuple of key columns for grouping
            date_col: Name of date column
            type_col: Name of type column
            
        Returns:
            Filtered dataframe with historical data prioritized
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        
        # Get all dates with historical data
        hist_dates = (
            df.loc[df[type_col] == 'HISTORICAL', list(keys) + [date_col]]
            .drop_duplicates()
        )
        
        # Build MultiIndex for fast lookup
        hist_idx = pd.MultiIndex.from_frame(hist_dates)
        
        # Mark conflicting forecast rows
        is_conflicting_forecast = (
            (df[type_col] == 'FORECAST') &
            df.set_index(list(keys) + [date_col]).index.isin(hist_idx)
        )
        
        return df.loc[~is_conflicting_forecast].reset_index(drop=True)
    
    @staticmethod
    def distribute_hourly_prec_to_quarterly(
        s_hourly: pd.Series, 
        full_range: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Distribute hourly precipitation into four equal 15-minute blocks.
        
        Args:
            s_hourly: Series with hourly precipitation values
            full_range: Full datetime range at 15-minute frequency
            
        Returns:
            Series with precipitation distributed to 15-minute intervals
        """
        if s_hourly is None or s_hourly.empty:
            return pd.Series(0.0, index=full_range)
        
        s = s_hourly.dropna().astype(float).copy()
        
        # Snap to exact hour
        s.index = pd.to_datetime(s.index).round('H')
        s = s.groupby(s.index).mean()
        
        # Create four quarter-ends
        s0 = s.copy()
        s1 = s.copy(); s1.index = s1.index - pd.Timedelta(minutes=15)
        s2 = s.copy(); s2.index = s2.index - pd.Timedelta(minutes=30)
        s3 = s.copy(); s3.index = s3.index - pd.Timedelta(minutes=45)
        
        out = pd.concat([s0, s1, s2, s3]).groupby(level=0).sum() / 4.0
        out = out.reindex(full_range).fillna(0.0)
        return out
    
    def process_state_data(
        self, 
        df_state: pd.DataFrame, 
        allowed_locations: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process state weather data with interpolation.
        
        Args:
            df_state: State-level weather dataframe
            allowed_locations: Optional list of locations to filter
            
        Returns:
            Processed dataframe with interpolated 15-minute intervals
        """
        df_state = df_state.copy()
        
        if allowed_locations is not None:
            df_state = df_state[df_state['location'].isin(allowed_locations)]
            if df_state.empty:
                print(f"[skip] No rows for locations={allowed_locations}")
                return pd.DataFrame()
        
        df_state = df_state.set_index('datetime').sort_index()
        
        # Create full 15-minute grid
        start = df_state.index.min().normalize()
        end = df_state.index.max().normalize() + pd.Timedelta(hours=23, minutes=45)
        full_range = pd.date_range(start=start, end=end, freq='15T')
        
        # Process each location
        loc_frames = []
        agg_map = {c: 'mean' for c in self.WEATHER_COLS}
        
        for loc, df_loc in df_state.groupby('location'):
            df_loc = df_loc[self.WEATHER_COLS].copy()
            df_loc = df_loc.groupby(df_loc.index).agg(agg_map)
            
            # Handle precipitation separately
            precip15 = None
            if 'precipitation' in df_loc.columns:
                precip15 = self.distribute_hourly_prec_to_quarterly(
                    df_loc['precipitation'], 
                    full_range
                )
            
            # Align to 15-minute grid
            df_loc = df_loc.reindex(full_range)
            
            # Interpolate non-precipitation columns
            non_precip_cols = [
                c for c in self.WEATHER_COLS 
                if c != 'precipitation' and c in df_loc.columns
            ]
            if non_precip_cols:
                df_loc[non_precip_cols] = (
                    df_loc[non_precip_cols]
                    .interpolate(method='time')
                    .ffill()
                    .bfill()
                )
            
            # Set quartered precipitation
            if precip15 is not None:
                df_loc['precipitation'] = precip15.values
            
            df_loc['location'] = loc
            loc_frames.append(df_loc)
        
        if not loc_frames:
            return pd.DataFrame()
        
        # Combine and add metadata
        df_interp = pd.concat(loc_frames)
        df_interp['date'] = df_interp.index.date
        df_interp['time_block'] = (
            df_interp.index.hour * 60 + df_interp.index.minute
        ) // 15 + 1
        df_interp['state'] = self.state
        
        # Select and reorder columns
        cols = ['date', 'time_block', 'location'] + self.WEATHER_COLS
        df_out = df_interp.reset_index(drop=True)[cols]
        
        # Add datetime column
        df_out['date'] = pd.to_datetime(df_out['date'], errors='coerce').dt.date
        df_out['time_block'] = pd.to_numeric(df_out['time_block'], errors='coerce')
        df_out['datetime'] = (
            pd.to_datetime(df_out['date'].astype(str)) + 
            pd.to_timedelta((df_out['time_block'] - 1) * 15, unit='m')
        )
        
        return df_out
    
    def process(self) -> pd.DataFrame:
        """Execute full weather data processing pipeline."""
        self.load_data()
        self.prepare_datetime()
        self.df = self.keep_hist_then_forecast(
            self.df, 
            keys=('state', 'location'), 
            date_col='date', 
            type_col='type'
        )
        
        # Fix datetime index
        self.df['time'] = self.df['time'].astype(str).str.strip()
        self.df['datetime'] = (
            pd.to_datetime(self.df['date']) + pd.to_timedelta(self.df['time'])
        )
        
        # Process state data
        weather_new = self.process_state_data(self.df)
        return weather_new