import json
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ── Indian State → holidays library subdivision code ─────────────
INDIAN_STATE_HOLIDAY_SUBDIV = {
    "punjab": "PB", "haryana": "HR", "delhi": "DL",
    "uttar pradesh": "UP", "uttarakhand": "UK",
    "himachal pradesh": "HP", "jammu & kashmir": "JK",
    "rajasthan": "RJ", "gujarat": "GJ", "maharashtra": "MH", "goa": "GA",
    "tamil nadu": "TN", "kerala": "KL", "karnataka": "KA",
    "andhra pradesh": "AP", "telangana": "TS",
    "west bengal": "WB", "odisha": "OD", "bihar": "BR",
    "jharkhand": "JH", "assam": "AS",
    "madhya pradesh": "MP", "chhattisgarh": "CG",
}

try:
    import polars as pl
    _HAS_POLARS = True
except Exception:
    pl = None
    _HAS_POLARS = False

# ==========================================
# ENHANCED FEATURE REGISTRY
# ==========================================
class FeatureRegistry:
    CATALOG = {
        "total_drawal": {
            "label": "Load (MW)",
            "type": "numeric",
            "role": "primary",
            "domain": "load",
            "allowed_edas": ["Temporal", "Distributional", "Anomaly", "Comparative", "Pattern", "Segmentation"],
            "aggregations": ["mean", "max", "min", "sum", "std", "p95", "p99", "median"]
        },
        "temperature": {
            "label": "Temperature (°C)",
            "type": "numeric",
            "role": "context",
            "domain": "weather",
            "allowed_edas": ["Temporal", "Distributional", "Comparative", "Impact", "Pattern"],
            "aggregations": ["mean", "max", "min", "median"]
        },
        "humidity": {
            "label": "Humidity (%)",
            "type": "numeric",
            "role": "context",
            "domain": "weather",
            "allowed_edas": ["Temporal", "Distributional", "Comparative", "Impact"],
            "aggregations": ["mean", "median"]
        },
        "precipitation": {
            "label": "Rainfall (mm)",
            "type": "numeric",
            "role": "context",
            "domain": "weather",
            "allowed_edas": ["Temporal", "Distributional", "Comparative", "Impact"],
            "aggregations": ["sum", "max"]
        }
    }

    EDA_TYPES = {
        "Temporal": [
            "Rolling Stats",
            "Overlay Comparison",
            "Multi-Date Overlay",
            "Days T>T-1",
            "Trend Decomposition",
            "YoY Comparison"
        ],
        "Distributional": ["Histogram", "Boxplot"],
        "Comparative": ["Correlation Heatmap"],
        "Anomaly": ["Statistical Outliers", "Isolation Forest", "Peak Detection", "TDLLS (Level Shift)", "Absolute Jump"],
        "Pattern": ["Load Duration Curve", "Ramp Rate", "Weekly Patterns", "Cyclic Patterns"],
        "Impact": ["Temperature-Load Curve", "Weather Dashboard", "Full Weather Impact", "Calendar Effects"]
    }

    COLORS = {
        "primary": "#00F0FF",
        "secondary": "#06B6D4",
        "context": "#FF00AA",
        "baseline": "#4A4A4A",
        "up_shift": "#39FF14",
        "down_shift": "#FF3131",
        "highlight": "#FFE800",
        "warning": "#FF6B35",
        "critical": "#FF0000",
        "cluster_1": "#00F0FF",
        "cluster_2": "#FF00AA",
        "cluster_3": "#39FF14",
        "cluster_4": "#FFE800",
        "cluster_5": "#9D4EDD"
    }

    @classmethod
    def get_features(cls):
        return list(cls.CATALOG.keys())

    @classmethod
    def get_meta(cls, feature):
        return cls.CATALOG.get(feature, {})

    @classmethod
    def get_allowed_categories(cls, feature):
        meta = cls.get_meta(feature)
        return meta.get("allowed_edas", [])

    @classmethod
    def get_color(cls, feature):
        role = cls.CATALOG.get(feature, {}).get("role", "context")
        return cls.COLORS.get(role, cls.COLORS["context"])


# ==========================================
# ENHANCED EDA ENGINE
# ==========================================
class EDAEngine:
    def __init__(self, data_path, region="haryana"):
        self.data_path = data_path
        self._region = str(region or "haryana")
        self._df = self._load_data_static(data_path, region=self._region)

    def reload_data(self):
        """Reload data from disk and re-run preprocessing."""
        self._df = self._load_data_static(self.data_path, region=self._region)
        self._cached_stats = {}
        return self._df

    @staticmethod
    def _safe_corr(a, b):
        """Spearman rank correlation (robust to outliers and non-linear relationships)."""
        try:
            v = pd.Series(a).corr(pd.Series(b), method="spearman")
            if pd.isna(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_div(a, b):
        return float(a / b) if b not in (0, 0.0, None) else 0.0

    @staticmethod
    def _pct(a, b):
        return float((a - b) / b * 100.0) if b not in (0, 0.0, None) else 0.0

    @staticmethod
    def _load_data_static(data_path, region="haryana"):
        """Loads and preprocesses data with comprehensive feature engineering."""
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Could not find file at {data_path}")
            df = pd.read_csv(data_path)
            
            if "Datetime" in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
            elif "date" in df.columns and "time_block" in df.columns:
                df['Datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta((df['time_block'] - 1) * 15, unit="m")
            
            df = df.sort_values("Datetime")
            
            # Temporal Features
            df['Date'] = df['Datetime'].dt.date
            df['DayOfWeek'] = df['Datetime'].dt.day_name()
            df['DayOfWeekNum'] = df['Datetime'].dt.dayofweek
            df['Month'] = df['Datetime'].dt.month_name()
            df['MonthNum'] = df['Datetime'].dt.month
            df['Hour'] = df['Datetime'].dt.hour
            df['Quarter'] = df['Datetime'].dt.quarter
            df['WeekOfYear'] = df['Datetime'].dt.isocalendar().week
            df['DayOfYear'] = df['Datetime'].dt.dayofyear
            df['Year'] = df['Datetime'].dt.year
            
            # Categorical Time Features
            df['IsWeekend'] = df['DayOfWeekNum'].isin([5, 6]).astype(int)
            df['TimeOfDay'] = pd.cut(df['Hour'], 
                                     bins=[-1, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            
            # Phase 1: Custom Day Flags
            df['IsMonday'] = (df['DayOfWeekNum'] == 0).astype(int)
            df['IsFriday'] = (df['DayOfWeekNum'] == 4).astype(int)
            
            # Season mapping
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall'}
            df['Season'] = df['MonthNum'].map(season_map)

            # --------------------------------------------------
            # Feature Engineering for Behavioral Analysis
            # --------------------------------------------------
            # Time encoding
            blocks_per_day = 96
            df['block_sin'] = np.sin(2 * np.pi * (df['time_block'] - 1) / blocks_per_day)
            df['block_cos'] = np.cos(2 * np.pi * (df['time_block'] - 1) / blocks_per_day)
            df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
            df['is_morning_peak_block'] = df['Hour'].between(7, 10).astype(int)
            df['is_evening_peak_block'] = df['Hour'].between(18, 22).astype(int)
            df['is_night_valley_block'] = df['Hour'].between(1, 5).astype(int)

            # Calendar & holiday features (national + state-specific, deduplicated)
            # Vectorized for speed — avoid per-row lambdas
            df['is_holiday'] = 0
            hol_lookup = {}
            try:
                import holidays
                yrs = pd.to_datetime(df['Date']).dt.year.unique().tolist()
                national = holidays.India(years=yrs)
                _subdiv = INDIAN_STATE_HOLIDAY_SUBDIV.get(
                    str(region or "haryana").lower().strip(), 'HR'
                )
                state = holidays.India(subdiv=_subdiv, years=yrs)
                _combined_dates = {
                    pd.Timestamp(d).normalize()
                    for d in (set(national.keys()) | set(state.keys()))
                }
                # Vectorized: check unique dates only, then map back
                unique_dates = pd.to_datetime(df['Date']).dt.normalize().unique()
                hol_lookup = {d: int(d in _combined_dates) for d in unique_dates}
                df['is_holiday'] = pd.to_datetime(df['Date']).dt.normalize().map(hol_lookup).fillna(0).astype(int)
            except Exception:
                df['is_holiday'] = 0

            # Holiday proximity flags — vectorized via date-level computation
            date_series = pd.to_datetime(df['Date']).dt.normalize()
            holiday_dates_arr = (
                np.array(sorted({d for d, is_holiday in hol_lookup.items() if is_holiday}), dtype='datetime64[ns]')
                if df['is_holiday'].any()
                else np.array([], dtype='datetime64[ns]')
            )

            if len(holiday_dates_arr) > 0:
                holiday_set = set(pd.to_datetime(holiday_dates_arr))
                dow = date_series.dt.weekday
                df['holiday_before_weekend'] = ((df['is_holiday'] == 1) & (dow == 4)).astype(int)
                df['holiday_after_weekend'] = ((df['is_holiday'] == 1) & (dow == 0)).astype(int)

                # Bridge day: vectorized via shift on unique-date level
                date_plus_1 = date_series + pd.Timedelta(days=1)
                date_minus_1 = date_series - pd.Timedelta(days=1)
                is_adj_holiday = date_plus_1.isin(holiday_set) | date_minus_1.isin(holiday_set)
                df['sandwich_day'] = ((df['is_holiday'] == 0) & (dow < 5) & is_adj_holiday).astype(int)
                df['bridge_day_flag'] = df['sandwich_day']

                df['long_weekend_flag'] = ((dow >= 5) & is_adj_holiday).astype(int)

                # Days to/since holiday — vectorized with searchsorted
                holidays_sorted = np.sort(np.array(list(holiday_set), dtype='datetime64[ns]'))
                date_vals = date_series.values.astype('datetime64[ns]')
                idx_right = np.searchsorted(holidays_sorted, date_vals, side='left')
                idx_left = idx_right - 1

                days_to = np.full(len(df), 999, dtype=int)
                valid_r = idx_right < len(holidays_sorted)
                days_to[valid_r] = ((holidays_sorted[idx_right[valid_r]] - date_vals[valid_r]) / np.timedelta64(1, 'D')).astype(int)

                days_since = np.full(len(df), 999, dtype=int)
                valid_l = idx_left >= 0
                days_since[valid_l] = ((date_vals[valid_l] - holidays_sorted[idx_left[valid_l]]) / np.timedelta64(1, 'D')).astype(int)

                df['days_to_holiday'] = days_to
                df['days_since_holiday'] = days_since
            else:
                df['holiday_before_weekend'] = 0
                df['holiday_after_weekend'] = 0
                df['sandwich_day'] = 0
                df['bridge_day_flag'] = 0
                df['long_weekend_flag'] = 0
                df['days_to_holiday'] = 999
                df['days_since_holiday'] = 999

            # Weather proxy features
            if 'temperature' in df.columns:
                df['temp_D_same_block'] = df['temperature'].shift(blocks_per_day)
                df['temp_D_minus_7_same_block'] = df['temperature'].shift(blocks_per_day * 7)
                df['rolling_temp_7day_mean'] = df['temperature'].rolling(blocks_per_day * 7, min_periods=1).mean()
                df['cooling_degree'] = (df['temperature'] - 24.0).clip(lower=0)
                # Temperature features
                df['temperature_sq'] = df['temperature'] ** 2
                df['temperature_cb'] = df['temperature'] ** 3
                df['temperature_x_block_sin'] = df['temperature'] * df['block_sin']
                df['temperature_x_block_cos'] = df['temperature'] * df['block_cos']
                df['temperature_x_hour_sin'] = df['temperature'] * df['hour_sin']
                df['temperature_x_hour_cos'] = df['temperature'] * df['hour_cos']
                if 'humidity' in df.columns:
                    df['heat_index'] = 0.5 * (df['temperature'] + 61.0 + (df['temperature'] - 68.0) * 1.2 + df['humidity'] * 0.094)
                else:
                    df['heat_index'] = df['temperature']

            if 'humidity' in df.columns:
                df['humidity_7day_mean'] = df['humidity'].rolling(blocks_per_day * 7, min_periods=1).mean()
                # Humidity features
                df['humidity_sq'] = df['humidity'] ** 2
                df['humidity_cb'] = df['humidity'] ** 3
                df['humidity_x_block_sin'] = df['humidity'] * df['block_sin']
                df['humidity_x_block_cos'] = df['humidity'] * df['block_cos']
                df['humidity_x_hour_sin'] = df['humidity'] * df['hour_sin']
                df['humidity_x_hour_cos'] = df['humidity'] * df['hour_cos']

            if 'precipitation' in df.columns:
                df['rain_proxy'] = (df['precipitation'] > 0.1).astype(int)
                df['rain_intensity'] = pd.cut(df['precipitation'], 
                                            bins=[-1, 0.01, 1, 5, 20, 1000], 
                                            labels=['None', 'Light', 'Moderate', 'Heavy', 'Storm'])
                df['is_heavy_rain'] = (df['precipitation'] > 5).astype(int)
                # Precipitation features
                df['precipitation_sq'] = df['precipitation'] ** 2
                df['precipitation_cb'] = df['precipitation'] ** 3
                df['precipitation_x_block_sin'] = df['precipitation'] * df['block_sin']
                df['precipitation_x_block_cos'] = df['precipitation'] * df['block_cos']
                df['precipitation_x_hour_sin'] = df['precipitation'] * df['hour_sin']
                df['precipitation_x_hour_cos'] = df['precipitation'] * df['hour_cos']

            # Load memory features
            if 'total_drawal' in df.columns:
                df['load_D_same_block'] = df['total_drawal'].shift(blocks_per_day)
                df['load_D_minus_2'] = df['total_drawal'].shift(blocks_per_day * 2)
                df['load_D_minus_7'] = df['total_drawal'].shift(blocks_per_day * 7)
                df['rolling_7day_mean_block'] = (
                    df.groupby('time_block')['total_drawal']
                    .transform(lambda s: s.rolling(blocks_per_day * 7, min_periods=1).mean())
                )
                df['load_last_24_blocks'] = df['total_drawal'].rolling(24, min_periods=1).mean()

                # Period change features
                df['dod_block_change'] = df['total_drawal'] - df['load_D_same_block']
                df['wow_block_change'] = df['total_drawal'] - df['load_D_minus_7']
                df['rolling_avg_trend'] = df['rolling_7day_mean_block'] - df['rolling_7day_mean_block'].shift(blocks_per_day * 7)

                # Interaction features
                if 'temperature' in df.columns:
                    df['temp_x_peak_flag'] = df['temperature'] * df['is_evening_peak_block']
                    df['weather_x_load'] = df['temperature'] * df['total_drawal']
                df['block_x_load_t96'] = df['time_block'] * df['load_D_same_block']
                df['weekend_load_shift'] = df['total_drawal'] * df['IsWeekend']
                df['holiday_load_suppression'] = df['total_drawal'] * df['is_holiday']

                # Baseline curve features
                baseline_by_block = df.groupby('time_block')['total_drawal'].transform('mean')
                df['baseline_block'] = baseline_by_block
                df['seasonal_profile'] = df.groupby('Season')['total_drawal'].transform('mean')
                df['weekly_profile'] = df.groupby('DayOfWeek')['total_drawal'].transform('mean')
                df['baseline_deviation'] = df['total_drawal'] - df['baseline_block']

                # Risk & stability indicators
                block_vol = df.groupby('time_block')['total_drawal'].transform('std')
                df['block_volatility'] = block_vol
                df['historical_variance_score'] = (block_vol - block_vol.mean()) / (block_vol.std() + 1e-6)
                df['curve_deviation_score'] = df['baseline_deviation'].abs() / (block_vol + 1e-6)
                df['peak_volatility_flag'] = (df['curve_deviation_score'] > df['curve_deviation_score'].quantile(0.95)).astype(int)

            # --- V2 UPGRADE: Lag and Thermal Inertia Modeling ---
            if 'temperature' in df.columns:
                # 3-hour and 6-hour rolling averages (12 and 24 blocks of 15-min each)
                df['temp_roll_3h'] = df['temperature'].rolling(window=12, min_periods=1).mean()
                df['temp_roll_6h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
            
            return df

        except Exception as e:
            logging.getLogger(__name__).warning("Error loading data: %s", e)
            return pd.DataFrame()

    def get_date_range(self):
        if self._df.empty:
            return None, None
        if "Date" in self._df.columns:
            return str(self._df["Date"].min()), str(self._df["Date"].max())
        if "Datetime" in self._df.columns:
            dt = pd.to_datetime(self._df["Datetime"]).dt.date
            return str(dt.min()), str(dt.max())
        if "date" in self._df.columns:
            d = pd.to_datetime(self._df["date"]).dt.date
            return str(d.min()), str(d.max())
        return None, None

    def get_filtered_data(self, start_date, end_date, filters=None):
        """Applies time range and structural filters."""
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        
        mask = (self._df['Date'] >= start_date) & (self._df['Date'] <= end_date)
        
        if filters:
            if 'weekdays' in filters and filters['weekdays']:
                mask &= self._df['DayOfWeek'].isin(filters['weekdays'])
            if 'months' in filters and filters['months']:
                mask &= self._df['Month'].isin(filters['months'])
            if 'blocks' in filters and len(filters['blocks']) == 2:
                mask &= (self._df['time_block'] >= filters['blocks'][0]) & (self._df['time_block'] <= filters['blocks'][1])
            if 'weekend_only' in filters and filters['weekend_only']:
                mask &= self._df['IsWeekend'] == 1
        
        return self._df.loc[mask].copy()

    def _fig_to_json(self, fig):
        """Convert Plotly figure to JSON with NaN handling."""
        import math
        
        def sanitize_value(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        
        def sanitize_dict(d):
            if isinstance(d, dict):
                return {k: sanitize_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [sanitize_dict(i) for i in d]
            else:
                return sanitize_value(d)
        
        raw = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        return sanitize_dict(raw)

    def _sanitize_kpis(self, kpis):
        """Ensure KPIs are JSON-safe (no NaN/Inf)."""
        import math
        safe_kpis = {}
        for k, v in kpis.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    safe_kpis[k] = 0
                else:
                    safe_kpis[k] = v
            else:
                safe_kpis[k] = v
        return safe_kpis

    def _get_standard_kpis(self, df, feature):
        """Get standard KPIs for any dataset - used for consistent display across all views."""
        if df.empty or feature not in df.columns:
            return {"average": 0, "peak": 0, "volatility": 0, "points": 0}
        
        data = df[feature].dropna()
        return {
            "average": float(data.mean()),
            "peak": float(data.max()),
            "volatility": float(data.std()),
            "points": len(data)
        }

    def _get_hourly_kpis(self, df, feature):
        """Get KPIs computed on hourly aggregated data."""
        if df.empty or feature not in df.columns or "Datetime" not in df.columns:
            return {"average": 0, "peak": 0, "volatility": 0, "points": 0}

        hourly = df.set_index('Datetime')[feature].resample('1h').mean()
        data = hourly.dropna()
        if data.empty:
            return {"average": 0, "peak": 0, "volatility": 0, "points": 0}

        return {
            "average": float(data.mean()),
            "peak": float(data.max()),
            "volatility": float(data.std()),
            "points": int(len(data))
        }

    def _get_hourly_kpi_series(self, df, feature):
        """Return KPI series per hour of day (0-23)."""
        if df.empty or feature not in df.columns or "Hour" not in df.columns:
            return []

        grouped = df.groupby('Hour')[feature].agg(['mean', 'max', 'std', 'count']).reset_index()
        grouped = grouped.sort_values('Hour')
        results = []
        for _, row in grouped.iterrows():
            results.append({
                "hour": int(row['Hour']),
                "average": float(row['mean']) if pd.notna(row['mean']) else 0.0,
                "peak": float(row['max']) if pd.notna(row['max']) else 0.0,
                "volatility": float(row['std']) if pd.notna(row['std']) else 0.0,
                "points": int(row['count']) if pd.notna(row['count']) else 0
            })
        return results

    # --- V2 UPGRADE: Localized & Regression Based Logic ---

    def _compute_base_load(self, region=None):
        """
        Isolates Fair Weather Days (15-22°C) and computes median load 
        per (Season, DayType, Block).
        """
        if self._df.empty or 'temperature' not in self._df.columns:
            return pd.DataFrame()
        
        # 1. Identify Fair Weather Days
        # Typical range: 15°C to 22°C, low rain (< 0.1mm), moderate humidity (30-60%)
        fair_mask = (
            (self._df['temperature'] >= 15) & (self._df['temperature'] <= 22) &
            (self._df['precipitation'] <= 0.1) &
            (self._df['humidity'] >= 30) & (self._df['humidity'] <= 70)
        )
        
        fair_df = self._df[fair_mask].copy()
        if fair_df.empty:
            # Fallback: less restrictive fair weather
            fair_mask = (self._df['temperature'] >= 15) & (self._df['temperature'] <= 25)
            fair_df = self._df[fair_mask].copy()
            
        if fair_df.empty:
            return pd.DataFrame()

        # 2. Compute median load per Block, Season, DayType
        # Note: DayType is derived from DayOfWeekNum
        fair_df['DayType'] = fair_df['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        base_load_map = fair_df.groupby(['Season', 'DayType', 'time_block'])['total_drawal'].median().reset_index()
        base_load_map.rename(columns={'total_drawal': 'base_load_mw'}, inplace=True)
        
        return base_load_map

    def _compute_block_sensitivity(self):
        """
        Runs linear regression per block to derive localized beta (Temp), 
        gamma (Hum), and delta (Rain) coefficients.
        """
        if self._df.empty or 'total_drawal' not in self._df.columns:
            return {}

        results = {}
        blocks = sorted(self._df['time_block'].unique())
        
        for b in blocks:
            b_df = self._df[self._df['time_block'] == b].dropna(subset=['total_drawal', 'temperature', 'humidity', 'precipitation'])
            if len(b_df) < 10: 
                continue
                
            X = b_df[['temperature', 'humidity', 'precipitation']]
            y = b_df['total_drawal']
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                results[int(b)] = {
                    "alpha": float(model.intercept_),
                    "beta_temp": float(model.coef_[0]),
                    "gamma_hum": float(model.coef_[1]),
                    "delta_rain": float(model.coef_[2]),
                    "r2": float(model.score(X, y))
                }
            except Exception:
                continue
                
        return results

    def _compute_lag_features(self, df):
        """
        Capture building thermal inertia and delayed cooling impact.
        Already integrated into _load_data_static via temp_roll_3h/6h.
        This provides a standalone helper for runtime dataframes.
        """
        if df.empty or 'temperature' not in df.columns:
            return df
            
        df = df.copy()
        df['temp_roll_3h'] = df['temperature'].rolling(window=12, min_periods=1).mean()
        df['temp_roll_6h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        return df

    def run_temperature_load_curve(self, df):
        """Analyze temperature-load relationship with decision insights (V2 HDD/CDD)."""
        if 'temperature' not in df.columns or 'total_drawal' not in df.columns:
            return {"error": "Required columns not found"}
        
        df_copy = df.copy()
        
        # --- V2 UPGRADE: Nonlinear Modeling ---
        # 1. Define Degree Days
        df_copy['CDD'] = (df_copy['temperature'] - 22.0).clip(lower=0)
        df_copy['HDD'] = (15.0 - df_copy['temperature']).clip(lower=0)
        
        # Create temperature bins for visualization
        df_copy['temp_bin'] = pd.cut(df_copy['temperature'], bins=25)
        
        temp_load = df_copy.groupby('temp_bin', observed=True).agg({
            'total_drawal': ['mean', 'std', 'count'],
            'temperature': 'mean'
        }).reset_index()
        
        temp_load.columns = ['temp_bin', 'load_mean', 'load_std', 'count', 'temp_mean']
        temp_load = temp_load.sort_values('temp_mean')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=temp_load['temp_mean'],
            y=temp_load['load_mean'],
            mode='markers+lines',
            name='Temp-Load Curve',
            marker=dict(
                size=temp_load['count']/20,
                color=temp_load['temp_mean'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Temp °C"),
                line=dict(width=1, color='white')
            ),
            line=dict(color=FeatureRegistry.COLORS['primary'], width=2)
        ))
        
        # Add Vertical Lines for Zones
        fig.add_vline(x=15, line_dash="dash", line_color="#FF3131", annotation_text="Heating Zone")
        fig.add_vline(x=22, line_dash="dash", line_color="#00F0FF", annotation_text="Cooling Zone")
        
        fig.update_layout(
            template="plotly_dark",
            title="Nonlinear Temperature-Load Profile (HDD/CDD Aware)",
            xaxis_title="Temperature (°C)",
            yaxis_title="Average Load (MW)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Calculate sensitivity metrics using piecewise linear regression
        # Cooling Elasticity (CDD)
        cooling_df = df_copy[df_copy['temperature'] > 22].dropna(subset=['CDD', 'total_drawal'])
        if not cooling_df.empty and len(cooling_df) > 5:
            model_c = LinearRegression()
            model_c.fit(cooling_df[['CDD']], cooling_df['total_drawal'])
            elasticity_c = float(model_c.coef_[0])
        else:
            elasticity_c = 0.0
            
        # Heating Elasticity (HDD)
        heating_df = df_copy[df_copy['temperature'] < 15].dropna(subset=['HDD', 'total_drawal'])
        if not heating_df.empty and len(heating_df) > 5:
            model_h = LinearRegression()
            model_h.fit(heating_df[['HDD']], heating_df['total_drawal'])
            elasticity_h = float(model_h.coef_[0])
        else:
            elasticity_h = 0.0

        correlation = df[['temperature', 'total_drawal']].corr(method="spearman").iloc[0, 1]
        
        # Determine Impact
        insight = f"Nonlinear Sensitivity: Cooling impacts load by {elasticity_c:.1f} MW/°C (above 22°C), " \
                  f"Heating by {elasticity_h:.1f} MW/°C (below 15°C). " \
                  f"Efficiency zone exists between 15-22°C."
        
        return {
            "plot": self._fig_to_json(fig),
            "kpis": self._sanitize_kpis({
                "correlation": correlation,
                "cooling_elasticity_mw_c": elasticity_c,
                "heating_elasticity_mw_c": elasticity_h,
                "max_load_mw": float(temp_load['load_mean'].max()),
                "optimal_temp_c": float(temp_load.loc[temp_load['load_mean'].idxmin(), 'temp_mean']),
                "deadband": "15°C - 22°C"
            }),
            "insight": insight
        }

    def run_weather_dashboard(self, start_date, end_date):
        """Enhanced weather analytics dashboard."""
        df = self.get_filtered_data(start_date, end_date)
        
        # KPIs
        kpis = {
            "max_temp": float(df['temperature'].max()) if 'temperature' in df else 0,
            "min_temp": float(df['temperature'].min()) if 'temperature' in df else 0,
            "total_precip": float(df['precipitation'].sum()) if 'precipitation' in df else 0,
            "avg_humidity": float(df['humidity'].mean()) if 'humidity' in df else 0,
        }
        
        fig = go.Figure()
        if 'temperature' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['temperature'],
                mode='lines', name='Temperature (°C)',
                line=dict(color=FeatureRegistry.COLORS['primary'], width=2)
            ))
        if 'humidity' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['humidity'],
                mode='lines', name='Humidity (%)',
                line=dict(color=FeatureRegistry.COLORS['context'], width=2),
                yaxis='y2'
            ))
        if 'precipitation' in df.columns:
            fig.add_trace(go.Bar(
                x=df['Datetime'], y=df['precipitation'],
                name='Precipitation (mm)',
                marker_color=FeatureRegistry.COLORS['secondary'],
                opacity=0.4,
                yaxis='y3'
            ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Weather Overview",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            yaxis2=dict(title="Humidity (%)", overlaying='y', side='right', showgrid=False),
            yaxis3=dict(title="Precipitation (mm)", overlaying='y', side='right', position=0.95, showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return {"plot": self._fig_to_json(fig), "kpis": self._sanitize_kpis(kpis)}

    # ==========================================
    # LOAD BEHAVIOR MORPHOLOGY (Pattern Tab)
    # ==========================================
    
    def run_distribution_boxplot(self, df, feature, group_by="DayOfWeek"):
        """Load Shape Intelligence: Analyze Load distribution by Day-Type."""
        meta = FeatureRegistry.get_meta(feature)
        fig = px.box(
            df, x=group_by, y=feature, color=group_by,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title=f"Load Behavior Morphology: {meta.get('label', feature)} by {group_by}"
        )
        
        # Calculate Peak Day
        avg_by_day = df.groupby(group_by, observed=True)[feature].mean()
        peak_day = avg_by_day.idxmax()
        min_day = avg_by_day.idxmin()
        peak_gap_pct = ((avg_by_day[peak_day] - avg_by_day[min_day]) / avg_by_day[min_day]) * 100
        
        insight = f"Load consumption peaks on {peak_day}s, averaging {peak_gap_pct:.1f}% higher than {min_day}s (lowest usage day)."
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({"peak_day_gap_pct": peak_gap_pct})
        }

    def run_distribution_histogram(self, df, feature):
        """Load Probability Distribution."""
        fig = px.histogram(
            df, x=feature, nbins=50,
            title=f"Load Probability Distribution: {feature}",
            template="plotly_dark",
            color_discrete_sequence=[FeatureRegistry.COLORS['primary']]
        )
        
        # Add mean and median lines
        mean_val = df[feature].mean()
        median_val = df[feature].median()
        skewness = df[feature].skew()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color=FeatureRegistry.COLORS['up_shift'], annotation_text=f"Mean: {mean_val:.1f}")
        fig.add_vline(
            x=median_val, line_dash="dot",
            line_color=FeatureRegistry.COLORS['highlight'],
            annotation_text=f"Median: {median_val:.1f}"
        )
        
        # Determine Skewness Insight
        skew_desc = "symmetrical"
        if skewness > 0.5: skew_desc = "positively skewed (frequent low Load)"
        elif skewness < -0.5: skew_desc = "negatively skewed (frequent high Load)"
        
        insight = f"Load distribution is {skew_desc}. The most common Load level (mode) is around {df[feature].mode()[0]:.1f} MW."

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "skewness": skewness,
                "kurtosis": df[feature].kurt(),
                "p95_load": df[feature].quantile(0.95),
                "p05_load": df[feature].quantile(0.05)
            })
        }

    # ==========================================
    # RESTORED ANALYTICS METHODS
    # ==========================================

    def run_rolling_stats(self, df, feature, window=7, agg='mean'):
        """Load Command Center: Analysis of Load deviations from moving baseline."""
        if df.empty or feature not in df.columns: return {"error": "No data"}
        
        # 1. Calculate Load Baseline (Moving Average)
        resampled = df.set_index('Datetime')[feature].resample('1h').mean().interpolate()
        rolled_mean = resampled.rolling(window=window*24).mean()
        rolled_std = resampled.rolling(window=window*24).std()
        
        upper_band = rolled_mean + 2*rolled_std
        lower_band = rolled_mean - 2*rolled_std
        
        # 2. Calculate Deviations
        deviation = resampled - rolled_mean
        avg_deviation = deviation.abs().mean()
        max_deviation = deviation.abs().max()
        
        # 3. Generate Load-Centric Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resampled.index, y=resampled, mode='lines', name='Actual Load (MW)', 
                                line=dict(color='rgba(255,255,255,0.7)', width=1.5)))
        fig.add_trace(go.Scatter(x=rolled_mean.index, y=upper_band, mode='lines', name='Upper Band', 
                                line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=rolled_mean.index, y=lower_band, mode='lines', name='Normal Range (±2σ)', 
                                fill='tonexty', fillcolor='rgba(0,240,255,0.1)', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=rolled_mean.index, y=rolled_mean, mode='lines', name='Load Baseline (Moving Avg)', 
                                line=dict(color=FeatureRegistry.COLORS['primary'], width=3)))
        
        # 4. Generate Insight
        trend_direction = "increasing" if rolled_mean.iloc[-1] > rolled_mean.iloc[0] else "decreasing"
        insight = f"Load is {trend_direction} with an average deviation of {avg_deviation:.1f} MW from baseline. Peak deviation hit {max_deviation:.1f} MW."

        fig.update_layout(template="plotly_dark", title=f"Load Command Center: {feature} vs Baseline", 
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
        
        return {
            "plot": self._fig_to_json(fig), 
            "kpis": self._sanitize_kpis({
                "avg_load_mw": float(rolled_mean.iloc[-1]) if len(rolled_mean) > 0 else 0,
                "avg_deviation_mw": float(avg_deviation),
                "peak_deviation_mw": float(max_deviation),
                "volatility_mw": float(rolled_std.iloc[-1]) if len(rolled_std) > 0 else 0
            }),
            "insight": insight
        }

    def run_yoy_comparison(self, df, feature):
        """Year-over-year comparison analysis."""
        if 'Year' not in df.columns or df['Year'].nunique() < 2:
            return {"error": "Insufficient data for YoY"}
        yoy_data = df.groupby(['Year', 'DayOfYear'])[feature].mean().reset_index()
        fig = go.Figure()
        years = sorted(yoy_data['Year'].unique())
        colors = [FeatureRegistry.COLORS['primary'], FeatureRegistry.COLORS['context'], FeatureRegistry.COLORS['up_shift']]
        for i, year in enumerate(years):
            year_data = yoy_data[yoy_data['Year'] == year]
            fig.add_trace(go.Scatter(x=year_data['DayOfYear'], y=year_data[feature], mode='lines', name=str(year), line=dict(color=colors[i % len(colors)])))
        fig.update_layout(template="plotly_dark", title="YoY Comparison", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {"plot": self._fig_to_json(fig)}

    def run_overlay_comparison(self, feature, target_date):
        """Standard overlay with statistical context."""
        target_date_obj = pd.to_datetime(target_date).date()
        target_df = self._df[self._df['Date'] == target_date_obj]
        if target_df.empty: return {"error": "No data"}
        prev_day = self._df[self._df['Date'] == target_date_obj - timedelta(days=1)]
        prev_week = self._df[self._df['Date'] == target_date_obj - timedelta(days=7)]
        fig = go.Figure()
        if not prev_week.empty: fig.add_trace(go.Scatter(x=prev_week['time_block'], y=prev_week[feature], mode='lines', name='T-7 Days', line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
        if not prev_day.empty: fig.add_trace(go.Scatter(x=prev_day['time_block'], y=prev_day[feature], mode='lines', name='T-1 Day', line=dict(color='rgba(255,255,255,0.5)', dash='dot')))
        fig.add_trace(go.Scatter(x=target_df['time_block'], y=target_df[feature], mode='lines', name=f'Target: {target_date}', line=dict(color=FeatureRegistry.COLORS['primary'], width=3)))
        fig.update_layout(template="plotly_dark", title="Overlay Comparison", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {"plot": self._fig_to_json(fig)}

    def run_seasonality_decomposition(self, df, feature):
        """Decompose time series into trend, seasonal, and residual components."""
        daily_avg = df.groupby('Date')[feature].mean()
        trend = daily_avg.rolling(window=7, center=True).mean()
        detrended = daily_avg - trend
        seasonal_df = pd.DataFrame({'value': detrended, 'dow': pd.to_datetime(detrended.index).dayofweek})
        seasonal = seasonal_df.groupby('dow')['value'].transform('mean')
        residual = detrended - seasonal
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Original", "Trend", "Seasonal", "Residual"), shared_xaxes=True)
        fig.add_trace(go.Scatter(x=daily_avg.index, y=daily_avg, name="Original"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend.index, y=trend, name="Trend"), row=2, col=1)
        fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, name="Seasonal"), row=3, col=1)
        fig.add_trace(go.Scatter(x=residual.index, y=residual, name="Residual"), row=4, col=1)
        fig.update_layout(template="plotly_dark", height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {"plot": self._fig_to_json(fig)}

    def run_load_duration_curve(self, df, feature):
        sorted_load = np.sort(df[feature].dropna().values)[::-1]
        percentiles = np.linspace(0, 100, len(sorted_load))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=percentiles, y=sorted_load, mode='lines', name='LDC', fill='tozeroy'))
        fig.update_layout(template="plotly_dark", title="Load Duration Curve", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        kpis = {"peak_load": float(sorted_load[0]) if len(sorted_load) > 0 else 0, "p95_load": float(sorted_load[int(len(sorted_load)*0.05)]) if len(sorted_load) > 20 else 0}
        return {"plot": self._fig_to_json(fig), "kpis": self._sanitize_kpis(kpis)}

    def run_ramp_rate_analysis(self, df, feature):
        df_copy = df.copy()
        df_copy['ramp_rate'] = df_copy[feature].diff() / 0.25
        fig = px.histogram(df_copy, x='ramp_rate', nbins=100, title="Ramp Rate Distribution", template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        kpis = {"max_up_ramp": float(df_copy['ramp_rate'].max()), "max_down_ramp": float(df_copy['ramp_rate'].min()), "avg_abs_ramp": float(df_copy['ramp_rate'].abs().mean())}
        return {"plot": self._fig_to_json(fig), "kpis": self._sanitize_kpis(kpis)}

    def run_weekly_pattern_analysis(self, df, feature):
        weekly_profile = df.groupby(['DayOfWeek', 'time_block'])[feature].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_profile['DayOfWeek'] = pd.Categorical(weekly_profile['DayOfWeek'], categories=day_order, ordered=True)
        pivot_data = weekly_profile.pivot(index='DayOfWeek', columns='time_block', values=feature)
        fig = go.Figure(data=go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index, colorscale='Turbo'))
        fig.update_layout(template="plotly_dark", title="Weekly Pattern Heatmap", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {"plot": self._fig_to_json(fig)}

    def run_cyclic_pattern_detection(self, df, feature):
        hourly_pattern = df.groupby('Hour')[feature].mean()
        fig = go.Figure(go.Scatter(x=hourly_pattern.index, y=hourly_pattern.values, mode='lines+markers'))
        fig.update_layout(template="plotly_dark", title="Daily Cycle (Hourly)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return {"plot": self._fig_to_json(fig), "peak_hour": int(hourly_pattern.idxmax()), "valley_hour": int(hourly_pattern.idxmin())}

    def run_statistical_outliers(self, df, feature, method='iqr', threshold=1.5):
        df_copy = df.copy()
        if method == 'iqr':
            Q1, Q3 = df_copy[feature].quantile(0.25), df_copy[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            df_copy['is_outlier'] = (df_copy[feature] < lower) | (df_copy[feature] > upper)
        elif method == 'zscore':
            z = stats.zscore(df_copy[feature].fillna(0))
            df_copy['is_outlier'] = np.abs(z) > threshold
        
        outliers = df_copy[df_copy['is_outlier']].copy()
        
        # Insights
        outlier_count = len(outliers)
        total_count = len(df_copy)
        distortion_rate = (outlier_count / total_count) * 100 if total_count > 0 else 0
        
        insight = f"Detected {outlier_count} Load Distortions ({distortion_rate:.1f}% of samples) using {method.upper()} method."
        if outlier_count > 0:
            avg_outlier_load = outliers[feature].mean()
            avg_normal_load = df_copy[~df_copy['is_outlier']][feature].mean()
            deviation_magnitude = ((avg_outlier_load - avg_normal_load) / avg_normal_load) * 100
            insight += f" Distortions deviate by {deviation_magnitude:+.1f}% from normal Load levels."

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_copy['Datetime'], y=df_copy[feature], mode='markers', name='Normal Load', marker=dict(size=3, opacity=0.5, color=FeatureRegistry.COLORS['primary'])))
        fig.add_trace(go.Scatter(x=outliers['Datetime'], y=outliers[feature], mode='markers', name='Load Distortion', marker=dict(color=FeatureRegistry.COLORS['critical'], size=8, symbol='x')))
        
        fig.update_layout(template="plotly_dark", title=f"Load Distortion Detector ({method.upper()})", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "timeline_plot": self._fig_to_json(fig), 
            "outlier_count": outlier_count,
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "distortion_rate_pct": distortion_rate,
                "distortion_count": outlier_count
            })
        }

    def run_isolation_forest_anomaly(self, df, feature, contamination=0.01):
        """ML-based anomaly detection using Isolation Forest."""
        df_model = df.copy()
        df_model['hour'] = df_model['Datetime'].dt.hour
        df_model['dow'] = df_model['Datetime'].dt.dayofweek
        X = df_model[[feature, 'hour', 'dow']].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        df_model['anomaly'] = predictions
        # Insights
        anomalies = df_model[df_model['anomaly'] == -1]
        anomaly_count = len(anomalies)
        processed_count = len(df_model)
        anomaly_rate = (anomaly_count / processed_count * 100) if processed_count > 0 else 0
        
        insight = f"Isolation Forest identified {anomaly_count} complex Load Distortions ({anomaly_rate:.1f}% rate). These events defy normal multi-variable patterns."
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_model['Datetime'], y=df_model[feature], mode='markers', name='Normal Load', marker=dict(color=FeatureRegistry.COLORS['primary'], size=3, opacity=0.5)))
        fig.add_trace(go.Scatter(x=anomalies['Datetime'], y=anomalies[feature], mode='markers', name='Load Distortion', marker=dict(color=FeatureRegistry.COLORS['critical'], size=8, symbol='x')))
        
        fig.update_layout(template="plotly_dark", title="Load Distortion Detector (Isolation Forest)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "timeline_plot": self._fig_to_json(fig),
            "anomaly_count": anomaly_count,
            "insight": insight,
            "kpis": self._sanitize_kpis({"anomaly_rate_pct": anomaly_rate})
        }

    def run_peak_detection(self, df, feature, prominence=100):
        """Detect peak loads using signal processing."""
        values = df[feature].fillna(0).values
        peaks, properties = find_peaks(values, prominence=prominence)
        peak_data = df.iloc[peaks].copy()
        
        # Insights
        peak_count = len(peak_data)
        avg_peak = peak_data[feature].mean() if peak_count > 0 else 0
        max_peak = peak_data[feature].max() if peak_count > 0 else 0
        
        insight = f"Detected {peak_count} Peak Load Events. Average peak magnitude is {avg_peak:.1f} MW, with a maximum of {max_peak:.1f} MW."

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df[feature], mode='lines', name='Load Profile', line=dict(color=FeatureRegistry.COLORS['primary'], width=1.5)))
        fig.add_trace(go.Scatter(x=peak_data['Datetime'], y=peak_data[feature], mode='markers', name='Peak Events', marker=dict(color=FeatureRegistry.COLORS['highlight'], size=8, symbol='triangle-up')))
        
        fig.update_layout(template="plotly_dark", title=f"Load Peak Distortion Detector (Prominence {prominence})", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "timeline_plot": self._fig_to_json(fig),
            "peak_count": peak_count,
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "peak_event_count": peak_count,
                "avg_peak_mw": avg_peak,
                "max_peak_mw": max_peak
            })
        }

    # ==========================================
    # PHASE 1 UPGRADES: ANALYTICAL DEPTH
    # ==========================================

    def run_advanced_decomposition(self, df, feature):
        """Break the single trend into interpretable layers: Seasonal, Weather, and Residual."""
        if df.empty or feature not in df.columns: return {"error": "Insufficient data"}
        # Ensure time drivers exist
        if 'time_block' not in df.columns or 'DayOfWeek' not in df.columns:
            return {"error": "Missing time drivers for decomposition"}
            
        df_copy = df.copy().sort_values('Datetime')
        seasonal_map = df_copy.groupby(['DayOfWeek', 'time_block'])[feature].transform('mean')
        df_copy['seasonal_component'] = seasonal_map
        df_copy['weather_component'] = 0
        if 'temperature' in df_copy.columns:
            valid = df_copy[[feature, 'temperature']].dropna()
            if len(valid) > 20:
                model = LinearRegression()
                X = valid[['temperature']]
                y = valid[feature] - seasonal_map.loc[valid.index]
                model.fit(X, y)
                df_copy['weather_component'] = model.predict(df_copy[['temperature']].fillna(df_copy['temperature'].mean()))
        df_copy['residual_component'] = df_copy[feature] - df_copy['seasonal_component'] - df_copy['weather_component']
        
        # Insights
        seasonal_contrib = df_copy['seasonal_component'].abs().sum()
        weather_contrib = df_copy['weather_component'].abs().sum()
        residual_contrib = df_copy['residual_component'].abs().sum()
        total_contrib = seasonal_contrib + weather_contrib + residual_contrib
        
        seasonal_pct = (seasonal_contrib / total_contrib * 100) if total_contrib > 0 else 0
        weather_pct = (weather_contrib / total_contrib * 100) if total_contrib > 0 else 0
        residual_pct = (residual_contrib / total_contrib * 100) if total_contrib > 0 else 0
        
        insight = f"Load Decomposition: Seasonality drives {seasonal_pct:.0f}% of Load structure, Weather {weather_pct:.0f}%, and {residual_pct:.0f}% is granular/residual stress."

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_copy['Datetime'], y=df_copy['seasonal_component'], mode='lines', name='Seasonal Base', stackgroup='one', line=dict(width=0.5, color='#4A4A4A')))
        fig.add_trace(go.Scatter(x=df_copy['Datetime'], y=df_copy['weather_component'], mode='lines', name='Weather Impact', stackgroup='one', line=dict(width=0.5, color=FeatureRegistry.COLORS['highlight'])))
        fig.add_trace(go.Scatter(x=df_copy['Datetime'], y=df_copy['residual_component'], mode='lines', name='Residual Stress', stackgroup='one', line=dict(width=0.5, color=FeatureRegistry.COLORS['context'])))
        fig.add_trace(go.Scatter(x=df_copy['Datetime'], y=df_copy[feature], mode='lines', name='Actual Load', line=dict(color='white', width=1.5, dash='dot')))
        
        fig.update_layout(template="plotly_dark", title="Load Decomposition (Granular Stress)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "seasonal_strength_pct": seasonal_pct,
                "weather_strength_pct": weather_pct,
                "residual_stress_pct": residual_pct
            })
        }

    def run_block_intelligence(self, df, feature):
        """Block-Level Intelligence (96 block brain)."""
        if df.empty or 'time_block' not in df.columns: return {"error": "Missing blocks"}
        block_stats = df.groupby('time_block')[feature].agg(['mean', 'std']).reset_index()
        # Insights
        peak_block = block_stats.loc[block_stats['mean'].idxmax()]
        volatile_block = block_stats.loc[block_stats['std'].idxmax()]
        
        insight = f"Block-Level Load Intelligence: Peak Load stress occurs at Block {int(peak_block['time_block'])}. Volatility is highest at Block {int(volatile_block['time_block'])}."

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Average Load Profile per Block", "Block-Level Load Volatility"))
        fig.add_trace(go.Scatter(x=block_stats['time_block'], y=block_stats['mean'], mode='lines', name='Avg Load', line=dict(color=FeatureRegistry.COLORS['primary'])), row=1, col=1)
        fig.add_trace(go.Bar(x=block_stats['time_block'], y=block_stats['std'], name='Volatility (Std Dev)', marker_color=FeatureRegistry.COLORS['warning']), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Block-Level Load Intelligence")
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "peak_load_block": int(peak_block['time_block']),
                "most_volatile_block": int(volatile_block['time_block']),
                "avg_volatility_mw": float(block_stats['std'].mean())
            })
        }

    def run_load_shape_clustering(self, df, feature):
        """Cluster days into behavioral archetypes using K-Means."""
        if df['Date'].nunique() < 5: return {"error": "Insufficient data for clustering"}
        pivot = df.pivot(index='Date', columns='time_block', values=feature).dropna()
        if len(pivot) < 3: return {"error": "Too many missing values"}
        normalized = pivot.div(pivot.max(axis=1), axis=0)
        kmeans = KMeans(n_clusters=min(4, len(pivot)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(normalized)
        centroids = pivot.groupby(clusters).mean()
        # Insights
        n_clusters = len(centroids)
        insight = f"Load Profile Clustering: Identified {n_clusters} distinct Load archetypes explaining day-to-day behavior variations."

        fig = go.Figure()
        for i in range(len(centroids)):
            fig.add_trace(go.Scatter(x=centroids.columns, y=centroids.iloc[i], mode='lines', name=f"Archetype {i+1}"))
            
        fig.update_layout(template="plotly_dark", title="Load Profile Clustering (Normalized Archetypes)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({"detected_archetypes": n_clusters}),
            "clusters": pd.Series(clusters, index=pivot.index).to_dict()
        }

    # ==========================================
    # TDLLS DETECTION
    # ==========================================
    
    def run_tdlls_detection(self, feature, tolerance=80, shift_min=300, shift_max=500, 
                           reversion_thresh=100, start_date=None, end_date=None):
        """TDLLS Detection logic."""
        df = self._df.copy()
        blocks_per_day = 96
        
        shifts = [blocks_per_day * 7 * i for i in range(1, 5)]
        shifted_cols = []
        for s in shifts:
            col_name = f'shift_{s}'
            df[col_name] = df[feature].shift(s)
            shifted_cols.append(col_name)
        
        df['Baseline'] = df[shifted_cols].median(axis=1)
        df['Delta'] = df[feature] - df['Baseline']
        
        valid_df = df.dropna(subset=['Baseline'])
        daily_stats = valid_df.groupby('Date').apply(lambda x: pd.Series({
            'daily_shift': x['Delta'].median(),
            'consistency': ((x['Delta'] - x['Delta'].median()).abs() < tolerance).mean(),
        })).reset_index()
        
        candidates = daily_stats[
            (daily_stats['daily_shift'].abs().between(shift_min, shift_max)) &
            (daily_stats['consistency'] >= 0.8)
        ].copy()
        
        daily_stats['next_day_shift'] = daily_stats['daily_shift'].shift(-1)
        candidates = candidates.merge(daily_stats[['Date', 'next_day_shift']], on='Date', how='left')
        tdlls_days = candidates[candidates['next_day_shift'].abs() < reversion_thresh]
        
        if start_date and end_date:
            s_date = pd.to_datetime(start_date).date()
            e_date = pd.to_datetime(end_date).date()
            tdlls_days = tdlls_days[(tdlls_days['Date'] >= s_date) & (tdlls_days['Date'] <= e_date)]
        
        events_list = tdlls_days.copy()
        events_list['Date'] = events_list['Date'].astype(str)
        
        return {
            "events": events_list.to_dict(orient='records'),
            "kpis": self._sanitize_kpis({
                "count": len(tdlls_days),
                "avg_magnitude": float(tdlls_days['daily_shift'].abs().mean()) if not tdlls_days.empty else 0
            })
        }

    def get_tdlls_xray(self, feature, event_date_str):
        """Generate X-ray plot for TDLLS event."""
        df = self._df.copy()
        blocks_per_day = 96
        
        shifts = [blocks_per_day * 7 * i for i in range(1, 5)]
        shifted_cols = []
        for s in shifts:
            col_name = f'shift_{s}'
            df[col_name] = df[feature].shift(s)
            shifted_cols.append(col_name)
        
        df['Baseline'] = df[shifted_cols].median(axis=1)
        
        event_date = pd.to_datetime(event_date_str).date()
        next_day = event_date + timedelta(days=1)
        
        event_data = df[df['Date'] == event_date]
        next_day_data = df[df['Date'] == next_day]
        
        fig = go.Figure()
        
        if not event_data.empty:
            fig.add_trace(go.Scatter(
                x=event_data['time_block'], y=event_data['Baseline'],
                name="Baseline",
                line=dict(color=FeatureRegistry.COLORS['baseline'], width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=event_data['time_block'], y=event_data[feature],
                name=f"Shift Day: {event_date}",
                line=dict(color=FeatureRegistry.COLORS['primary'], width=4)
            ))
        
        if not next_day_data.empty:
            fig.add_trace(go.Scatter(
                x=next_day_data['time_block'], y=next_day_data[feature],
                name=f"Reversion: {next_day}",
                line=dict(color=FeatureRegistry.COLORS['context'], width=2, dash='dot')
            ))
        
        fig.update_layout(
            template="plotly_dark",
            title=f"TDLLS Shift Anatomy: {event_date}",
            xaxis_title="Time Block",
            yaxis_title="Load (MW)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return self._fig_to_json(fig)

    def run_anomaly_detection(self, df, feature, rule="Absolute Jump", threshold=400):
        """Legacy anomaly detection method."""
        analysis_df = df.copy()
        
        if rule == "Absolute Jump":
            analysis_df['Delta'] = analysis_df[feature].diff().fillna(0)
            analysis_df['Is_Anomaly'] = analysis_df['Delta'].abs() > threshold
            metric_col = 'Delta'
        elif rule == "Z-Score Spike":
            feat_std = analysis_df[feature].std()
            if feat_std == 0:
                feat_std = 1.0
            analysis_df['Z'] = (analysis_df[feature] - analysis_df[feature].mean()) / feat_std
            analysis_df['Is_Anomaly'] = analysis_df['Z'].abs() > threshold
            metric_col = 'Z'
        else:
            analysis_df['Delta'] = analysis_df[feature].diff().fillna(0)
            analysis_df['Is_Anomaly'] = analysis_df['Delta'].abs() > threshold
            metric_col = 'Delta'
        
        anomalies = analysis_df[analysis_df['Is_Anomaly']].copy()
        anomalies['Size'] = anomalies[metric_col].abs()
        
        fig_timeline = px.scatter(
            anomalies, x="Datetime", y=feature,
            color=metric_col, size='Size' if rule=="Absolute Jump" and len(anomalies) > 0 else None,
            title=f"Detected Anomalies ({rule})",
            template="plotly_dark",
            color_continuous_scale="Viridis"
        )
        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return {
            "timeline_plot": self._fig_to_json(fig_timeline),
            "anomaly_count": len(anomalies)
        }

    # ==========================================
    # LOAD BEHAVIOR ANALYSIS
    # ==========================================

    def run_behavior_analysis(self, df, feature):
        """Load Response Intelligence: Driver efficiency and response analysis."""
        if df.empty or feature not in df.columns:
            return {"error": "No data available for the selected feature."}

        # Block profile
        block_stats = df.groupby('time_block')[feature].agg(['mean', 'std', 'max', 'count']).reset_index()
        block_stats.rename(columns={'mean': 'avg_load', 'std': 'std_load', 'max': 'peak_load'}, inplace=True)
        block_stats['peak_frequency'] = block_stats['peak_load'] / (block_stats['avg_load'] + 1e-6)

        # Weekend distortion
        weekend_avg = df[df['IsWeekend'] == 1].groupby('time_block')[feature].mean()
        weekday_avg = df[df['IsWeekend'] == 0].groupby('time_block')[feature].mean()
        weekend_delta = (weekend_avg - weekday_avg).fillna(0)

        # Weather sensitivity (temperature bins)
        temp_bins = None
        temp_curve = None
        if 'temperature' in df.columns:
            temp_bins = pd.qcut(df['temperature'].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
            temp_curve = df.assign(temp_bin=temp_bins).groupby(['temp_bin', 'time_block'])[feature].mean().reset_index()

        # Volatility by block
        block_vol = df.groupby('time_block')[feature].std().fillna(0)
        
        # Calculate Peak Probability Evening
        peak_block_idx = df.groupby('Date')[feature].idxmax()
        peak_blocks = df.loc[peak_block_idx, 'time_block'] if len(peak_block_idx) else pd.Series(dtype=float)
        peak_prob_evening = float((peak_blocks.between(18, 22)).mean()) if len(peak_blocks) else 0.0

        # KPIs & Insight
        temp_corr = float(df[[feature, 'temperature']].corr(method="spearman").iloc[0, 1]) if 'temperature' in df.columns else 0
        weekend_impact = float(weekend_delta.abs().mean())
        
        impact_driver = "Temperature" if abs(temp_corr) > 0.5 else "Weekend/Calendar"
        impact_desc = "strongly drives" if abs(temp_corr) > 0.7 else "moderately influences"
        insight = f"{impact_driver} {impact_desc} Load behavior. Evening peak probability is {peak_prob_evening*100:.1f}%."

        kpis = {
            "peak_block_probability": peak_prob_evening,
            "average_ramp_rate": float(df[feature].diff().abs().mean()),
            "curve_flatness_index": float(block_stats['std_load'].mean()),
            "weekend_flattening_coefficient": weekend_impact,
            "holiday_suppression_factor": float(df.loc[df['is_holiday'] == 1, feature].mean() / (df.loc[df['is_holiday'] == 0, feature].mean() + 1e-6)) if 'is_holiday' in df.columns else 0,
            "cooling_sensitivity": temp_corr,
            "block_volatility_index": float(block_vol.mean()),
            "daily_curve_deviation": float(df['baseline_deviation'].abs().mean()) if 'baseline_deviation' in df.columns else 0,
            "peak_risk_score": float(df['curve_deviation_score'].mean()) if 'curve_deviation_score' in df.columns else 0
        }

        # Plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Load Shape (Avg by Block)", "Weekend Load Destruction",
                            "Temperature → Load Response", "Block-Level Volatility"),
            vertical_spacing=0.12, horizontal_spacing=0.08
        )

        fig.add_trace(go.Scatter(
            x=block_stats['time_block'], y=block_stats['avg_load'],
            mode='lines', name='Avg Load',
            line=dict(color=FeatureRegistry.COLORS['primary'], width=2.5)
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=weekend_delta.index, y=weekend_delta.values,
            name='Weekend Delta',
            marker_color=FeatureRegistry.COLORS['context']
        ), row=1, col=2)

        if temp_curve is not None:
            for label, color in zip(['Low', 'Medium', 'High'],
                                    [FeatureRegistry.COLORS['secondary'], FeatureRegistry.COLORS['primary'], FeatureRegistry.COLORS['highlight']]):
                subset = temp_curve[temp_curve['temp_bin'] == label]
                fig.add_trace(go.Scatter(
                    x=subset['time_block'], y=subset[feature],
                    mode='lines', name=f'Temp {label}',
                    line=dict(color=color, width=2)
                ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=block_vol.index, y=block_vol.values,
            name='Block Volatility',
            marker_color=FeatureRegistry.COLORS['warning']
        ), row=2, col=2)

        fig.update_layout(
            template="plotly_dark",
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            title="Load Response Intelligence Dashboard"
        )

        return {
            "plot": self._fig_to_json(fig),
            "kpis": self._sanitize_kpis(kpis),
            "insight": insight,
            "block_profile": block_stats.to_dict(orient='records')
        }

    def run_behavior_tab(self, df, feature, tab_name):
        """Compute KPIs for a specific behavior tab."""
        if df.empty or feature not in df.columns:
            return {"error": "No data available for the selected feature."}

        tab_key = (tab_name or "").strip().lower()
        kpis = {}

        # Common helpers
        daily_avg = df.groupby('Date')[feature].mean()
        daily_max = df.groupby('Date')[feature].max()
        daily_min = df.groupby('Date')[feature].min()
        baseline_by_block = df.groupby('time_block')[feature].mean()
        baseline_peak_block = int(baseline_by_block.idxmax())

        # TAB 1: Time & Block Pattern Analysis
        if tab_key.startswith("time"):
            peak_block = df.groupby('Date')[feature].idxmax()
            peak_blocks = df.loc[peak_block, 'time_block']
            peak_prob = float((peak_blocks.between(18, 22)).mean()) if len(peak_blocks) else 0.0

            morning = df[df['Hour'].between(7, 10)]
            evening = df[df['Hour'].between(18, 22)]
            ramp_morning = float(morning[feature].diff().abs().mean())
            ramp_evening = float(evening[feature].diff().abs().mean())
            base_load = float(df[df['Hour'].between(1, 5)][feature].mean())
            peak_load = float(daily_max.mean())
            curve_flatness = float(df.groupby('Date')[feature].std().mean())
            intraday_vol = float(df[feature].diff().abs().mean())
            peak_timing_stability = float(peak_blocks.std()) if len(peak_blocks) > 1 else 0.0
            valley_depth = float((daily_max - daily_min).mean() / (daily_max.mean() + 1e-6))

            kpis = {
                "avg_load_per_block": float(df.groupby('time_block')[feature].mean().mean()),
                "peak_block_prob_pct": peak_prob * 100.0,
                "morning_ramp_rate": ramp_morning,
                "evening_ramp_rate": ramp_evening,
                "base_load_night": base_load,
                "peak_load_daily": peak_load,
                "curve_flatness": curve_flatness,
                "intraday_volatility": intraday_vol,
                "peak_timing_stability": peak_timing_stability,
                "valley_depth_index": valley_depth
            }

        # TAB 2: Calendar & Holiday Impact
        elif tab_key.startswith("calendar"):
            # Day-level aggregation to avoid block-level bias
            daily = df.groupby('Date')[feature].mean()
            daily_is_holiday = df.groupby('Date')['is_holiday'].max() if 'is_holiday' in df.columns else None
            daily_is_weekend = df.groupby('Date')['IsWeekend'].max()
            daily_days_to = df.groupby('Date')['days_to_holiday'].min() if 'days_to_holiday' in df.columns else None
            daily_bridge = df.groupby('Date')['bridge_day_flag'].max() if 'bridge_day_flag' in df.columns else None
            daily_longwk = df.groupby('Date')['long_weekend_flag'].max() if 'long_weekend_flag' in df.columns else None

            weekend_avg = daily[daily_is_weekend == 1].mean()
            weekday_avg = daily[daily_is_weekend == 0].mean()
            holiday_avg = daily[daily_is_holiday == 1].mean() if daily_is_holiday is not None else 0
            nonholiday_avg = daily[daily_is_holiday == 0].mean() if daily_is_holiday is not None else 0

            pre_holiday = daily[(daily_days_to == 1) & (daily_is_holiday == 0)].mean() if daily_days_to is not None else 0
            bridge = daily[daily_bridge == 1].mean() if daily_bridge is not None else 0
            long_wk = daily[daily_longwk == 1].mean() if daily_longwk is not None else 0

            holiday_morning = df[(df['is_holiday'] == 1) & (df['Hour'].between(7, 10))][feature].mean() if 'is_holiday' in df.columns else 0
            nonholiday_morning = df[(df['is_holiday'] == 0) & (df['Hour'].between(7, 10))][feature].mean() if 'is_holiday' in df.columns else 0
            holiday_evening = df[(df['is_holiday'] == 1) & (df['Hour'].between(18, 22))][feature].mean() if 'is_holiday' in df.columns else 0
            nonholiday_evening = df[(df['is_holiday'] == 0) & (df['Hour'].between(18, 22))][feature].mean() if 'is_holiday' in df.columns else 0

            weekend_peak_block = int(df[df['IsWeekend'] == 1].groupby('Date')[feature].idxmax().map(lambda i: df.loc[i, 'time_block']).mean()) if len(df) else 0
            weekday_peak_block = int(df[df['IsWeekend'] == 0].groupby('Date')[feature].idxmax().map(lambda i: df.loc[i, 'time_block']).mean()) if len(df) else 0

            # Post-holiday recovery lag (days to recover to 95% of non-holiday avg)
            recovery_lags = []
            if daily_is_holiday is not None and nonholiday_avg and not np.isnan(nonholiday_avg):
                holiday_dates = daily[daily_is_holiday == 1].index
                daily_sorted = daily.sort_index()
                for hdate in holiday_dates:
                    future = daily_sorted.loc[daily_sorted.index > hdate]
                    recovered = future[future >= 0.95 * nonholiday_avg]
                    if len(recovered):
                        lag = (pd.to_datetime(recovered.index[0]) - pd.to_datetime(hdate)).days
                        recovery_lags.append(lag)
            recovery_lag_days = float(np.mean(recovery_lags)) if recovery_lags else 0.0

            kpis = {
                "weekend_load_change_pct": self._pct(weekend_avg, weekday_avg),
                "holiday_load_suppression": self._safe_div(holiday_avg, nonholiday_avg),
                "pre_holiday_uplift_pct": self._pct(pre_holiday, nonholiday_avg),
                "post_holiday_recovery_days": recovery_lag_days,
                "bridge_day_deviation_pct": self._pct(bridge, nonholiday_avg),
                "long_weekend_impact_pct": self._pct(long_wk, weekend_avg if weekend_avg else nonholiday_avg),
                "weekday_weekend_peak_shift": float(weekend_peak_block - weekday_peak_block),
                "holiday_morning_dip_pct": self._pct(holiday_morning, nonholiday_morning),
                "holiday_evening_peak_reduction_pct": self._pct(holiday_evening, nonholiday_evening)
            }

        # TAB 3: Weather Sensitivity
        elif tab_key.startswith("weather"):
            if 'temperature' in df.columns:
                temp = df['temperature']
                load = df[feature]
                slope = np.polyfit(temp.fillna(0), load.fillna(0), 1)[0] if len(temp) > 1 else 0
                cooling = df[df.get('cooling_degree', 0) > 0][feature].mean()
                heating = df[df.get('cooling_degree', 0) == 0][feature].mean()
                humidity_hi = df[df.get('humidity', 0) >= 75][feature].mean()
                humidity_lo = df[df.get('humidity', 0) < 75][feature].mean()
                rain_day = df[(df.get('rain_proxy', 0) == 1) & (df['Hour'].between(9, 17))][feature].mean()
                rain_eve = df[(df.get('rain_proxy', 0) == 1) & (df['Hour'].between(18, 22))][feature].mean()
                no_rain_day = df[(df.get('rain_proxy', 0) == 0) & (df['Hour'].between(9, 17))][feature].mean()
                no_rain_eve = df[(df.get('rain_proxy', 0) == 0) & (df['Hour'].between(18, 22))][feature].mean()
                heat_stress = self._safe_corr(df.get('heat_index', temp), load)
                hot_days = df[df['temperature'] >= df['temperature'].quantile(0.8)]
                normal_days = df[df['temperature'] <= df['temperature'].quantile(0.5)]
                peak_expand = self._pct(hot_days[feature].max(), normal_days[feature].max())
                season_elastic = df.groupby('Season').apply(lambda g: self._safe_corr(g['temperature'], g[feature])).abs().mean()

                kpis = {
                    "temp_sensitivity": float(slope),
                    "cooling_load_pct": self._pct(cooling, heating if heating else cooling),
                    "heating_load_pct": self._pct(heating, cooling if cooling else heating),
                    "humidity_amplification": self._safe_div(humidity_hi, humidity_lo),
                    "rain_impact_day_pct": self._pct(rain_day, no_rain_day),
                    "rain_impact_eve_pct": self._pct(rain_eve, no_rain_eve),
                    "heat_stress_index": float(heat_stress),
                    "weather_peak_expansion_pct": float(peak_expand),
                    "seasonal_temp_elasticity": float(season_elastic)
                }
            else:
                kpis = {"temp_sensitivity": 0.0}

        # TAB 4: Load Memory & Persistence
        elif tab_key.startswith("load memory"):
            kpis = {
                "same_block_persistence": self._safe_corr(df.get('load_D_same_block', 0), df[feature]),
                "weekly_recurrence": self._safe_corr(df.get('load_D_minus_7', 0), df[feature]),
                "rolling_mean_dependency": self._safe_corr(df.get('rolling_7day_mean_block', 0), df[feature]),
                "load_inertia": self._safe_div(df[feature].std(), df[feature].diff().abs().mean() + 1e-6),
                "base_load_stability": float(df[df['Hour'].between(1, 5)][feature].std()),
                "shape_similarity": self._safe_corr(df.groupby('time_block')[feature].mean(), baseline_by_block),
                "yesterday_influence": self._safe_corr(df.get('load_D_same_block', 0), df[feature]),
                "weekly_reliability": float(df.groupby('DayOfWeek')[feature].std().mean())
            }

        # TAB 5: Trend & Momentum Behavior
        elif tab_key.startswith("trend"):
            dod = daily_avg.pct_change() * 100.0
            wow = daily_avg.pct_change(7) * 100.0
            peak_growth = np.polyfit(range(len(daily_max)), daily_max.values, 1)[0] if len(daily_max) > 1 else 0
            rolling_trend = daily_avg.rolling(7, min_periods=1).mean()
            trend_strength = self._safe_corr(range(len(rolling_trend)), rolling_trend)
            streaks = (dod > 0).astype(int)
            trend_persistence = float(streaks.groupby((streaks != streaks.shift()).cumsum()).sum().mean())
            accel = float(dod.diff().abs().mean())
            regime_shift = float((dod.abs() > 2 * dod.std()).mean())
            momentum_stability = self._safe_div(1.0, 1.0 + dod.std())

            kpis = {
                "dod_change_pct": float(dod.mean()),
                "wow_change_pct": float(wow.mean()),
                "peak_growth_rate_pct": float(peak_growth),
                "trend_strength_7d": float(trend_strength),
                "trend_persistence_days": trend_persistence,
                "demand_acceleration": accel,
                "regime_shift_indicator": regime_shift,
                "momentum_stability": momentum_stability
            }

        # TAB 6: Interaction Effects Analysis
        elif tab_key.startswith("interaction"):
            hot = df['temperature'] >= df['temperature'].quantile(0.8) if 'temperature' in df.columns else pd.Series([False] * len(df))
            hot_evening = df[hot & (df['is_evening_peak_block'] == 1)][feature].mean()
            overall = df[feature].mean()
            hot_weekend = df[hot & (df['IsWeekend'] == 1)][feature].mean()
            hot_weekday = df[hot & (df['IsWeekend'] == 0)][feature].mean()
            holiday_hot = df[hot & (df.get('is_holiday', 0) == 1)][feature].mean() if 'is_holiday' in df.columns else 0
            nonholiday_hot = df[hot & (df.get('is_holiday', 0) == 0)][feature].mean() if 'is_holiday' in df.columns else 0
            memory_weather = self._safe_corr(df.get('load_D_same_block', 0) * df.get('temperature', 0), df[feature])
            weather_peak = float((hot & (df['is_evening_peak_block'] == 1)).mean())
            interaction_strength = np.mean([
                abs(self._safe_corr(df.get('temp_x_peak_flag', 0), df[feature])),
                abs(self._safe_corr(df.get('weather_x_load', 0), df[feature]))
            ])

            kpis = {
                "peak_temp_amplification": self._safe_div(hot_evening, overall),
                "weekend_heat_impact": self._safe_div(hot_weekend, hot_weekday),
                "holiday_heat_stress": self._safe_div(holiday_hot, nonholiday_hot),
                "load_memory_weather": float(memory_weather),
                "weather_peak_coincidence": weather_peak,
                "interaction_strength": float(interaction_strength),
                "combined_impact_score": float(interaction_strength)
            }

        # TAB 7: Baseline Curve Performance
        elif tab_key.startswith("baseline"):
            deviation = df.get('baseline_deviation', df[feature] - baseline_by_block)
            baseline_error = float(deviation.abs().mean() / (df[feature].mean() + 1e-6) * 100.0)
            daily_peak_block = df.groupby('Date')[feature].idxmax().map(lambda i: df.loc[i, 'time_block'])
            peak_timing_offset = float((daily_peak_block - baseline_peak_block).abs().mean())
            shape_similarity = self._safe_corr(df.groupby('time_block')[feature].mean(), baseline_by_block)
            daily_energy_dev = self._pct(daily_avg.mean(), baseline_by_block.mean())
            seasonal_drift = float(df.get('seasonal_profile', df[feature]).std())
            baseline_stability = float(deviation.std())

            kpis = {
                "baseline_error_pct": baseline_error,
                "baseline_peak_deviation": float((daily_max - baseline_by_block.max()).mean()),
                "shape_similarity": float(shape_similarity),
                "daily_energy_deviation_pct": float(daily_energy_dev),
                "peak_timing_offset": peak_timing_offset,
                "seasonal_drift": seasonal_drift,
                "baseline_stability": baseline_stability
            }

        # TAB 8: Stability & Risk Monitoring
        elif tab_key.startswith("stability"):
            ramp_stress = float(df[feature].diff().abs().quantile(0.95))
            extreme_freq = float((df[feature] > df[feature].quantile(0.99)).mean())
            anomaly_day_count = float((daily_avg.pct_change().abs() > 2 * daily_avg.pct_change().std()).sum())

            kpis = {
                "peak_volatility_index": float(df.get('block_volatility', df[feature]).mean()),
                "block_risk_heat": float(df.get('curve_deviation_score', 0).mean()),
                "ramp_stress_index": ramp_stress,
                "curve_distortion": float(df.get('baseline_deviation', 0).abs().mean()),
                "extreme_event_freq": extreme_freq,
                "anomaly_day_count": anomaly_day_count,
                "dispatch_risk": float(df.get('peak_volatility_flag', 0).mean()),
                "load_uncertainty": float(df[feature].rolling(7 * 96, min_periods=1).std().mean())
            }

        # TAB 9: Executive Summary
        elif tab_key.startswith("executive"):
            peak_range = (float(df[feature].quantile(0.9)), float(df[feature].quantile(0.99)))
            monthly = df.groupby(df['Datetime'].dt.to_period('M'))[feature].mean()
            growth_rate = np.polyfit(range(len(monthly)), monthly.values, 1)[0] if len(monthly) > 1 else 0
            stability_index = self._safe_div(1.0, 1.0 + df[feature].std())
            forecast_ready = self._safe_div(1.0, 1.0 + df.get('baseline_deviation', 0).abs().mean())
            risk_level = float(df.get('peak_volatility_flag', 0).mean())

            kpis = {
                "expected_peak_range": f"{peak_range[0]:.0f}-{peak_range[1]:.0f}",
                "demand_growth_rate": float(growth_rate),
                "weather_sensitivity": float(self._safe_corr(df.get('temperature', 0), df[feature])),
                "holiday_impact": float(self._pct(df[df.get('is_holiday', 0) == 1][feature].mean(), df[df.get('is_holiday', 0) == 0][feature].mean())),
                "system_stability": stability_index,
                "forecast_readiness": forecast_ready,
                "operational_risk": risk_level
            }

        else:
            kpis = {"Info": "Unknown tab"}

        return {
            "kpis": self._sanitize_kpis(kpis)
        }

    # ==========================================
    # MULTI-DATE OVERLAY COMPARISON
    # ==========================================
    
    def run_multi_date_overlay(self, feature, dates_list):
        """Compare multiple dates on a single overlay chart."""
        colors = [
            FeatureRegistry.COLORS['primary'],
            FeatureRegistry.COLORS['context'],
            FeatureRegistry.COLORS['up_shift'],
            FeatureRegistry.COLORS['highlight'],
            FeatureRegistry.COLORS['warning'],
            '#9D4EDD', '#FF6B6B', '#4ECDC4'
        ]
        
        fig = go.Figure()
        
        for i, date_str in enumerate(dates_list):
            date_obj = pd.to_datetime(date_str).date()
            day_data = self._df[self._df['Date'] == date_obj]
            
            if not day_data.empty:
                dow = day_data.iloc[0]['DayOfWeek']
                fig.add_trace(go.Scatter(
                    x=day_data['time_block'], 
                    y=day_data[feature],
                    mode='lines',
                    name=f"{date_str} ({dow[:3]})",
                    line=dict(color=colors[i % len(colors)], width=2.5)
                ))
        
        fig.update_layout(
            template="plotly_dark",
            title=f"Multi-Date Comparison: {feature}",
            xaxis_title="Time Block",
            yaxis_title=feature,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Calculate stats for each date
        date_stats = []
        for date_str in dates_list:
            date_obj = pd.to_datetime(date_str).date()
            day_data = self._df[self._df['Date'] == date_obj]
            if not day_data.empty:
                date_stats.append({
                    "date": date_str,
                    "avg": float(day_data[feature].mean()),
                    "max": float(day_data[feature].max()),
                    "min": float(day_data[feature].min())
                })
        
        return {"plot": self._fig_to_json(fig), "date_stats": date_stats}

    # ==========================================
    # DAYS WHERE T > T-1 (WHOLE DAY)
    # ==========================================
    
    def run_days_exceeding_previous(self, df, feature):
        """Find days where load was consistently higher than previous day across all blocks."""
        # Aggregate daily
        daily_avg = df.groupby('Date')[feature].agg(['mean', 'max', 'min', 'sum']).reset_index()
        daily_avg.columns = ['Date', 'avg', 'max', 'min', 'total']
        daily_avg = daily_avg.sort_values('Date')
        
        # Compare with previous day
        daily_avg['prev_avg'] = daily_avg['avg'].shift(1)
        daily_avg['prev_max'] = daily_avg['max'].shift(1)
        daily_avg['delta'] = daily_avg['avg'] - daily_avg['prev_avg']
        daily_avg['delta_pct'] = (daily_avg['delta'] / daily_avg['prev_avg']) * 100
        
        # Days where T > T-1 (average load higher)
        exceeding_days = daily_avg[daily_avg['delta'] > 0].copy()
        
        # Plot
        fig = go.Figure()
        
        # All days as bars
        fig.add_trace(go.Bar(
            x=daily_avg['Date'],
            y=daily_avg['delta'],
            name='Daily Change (T - T-1)',
            marker_color=[
                FeatureRegistry.COLORS['up_shift'] if d > 0 else FeatureRegistry.COLORS['down_shift'] 
                for d in daily_avg['delta'].fillna(0)
            ]
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
        
        fig.update_layout(
            template="plotly_dark",
            title=f"Daily Load Change (T vs T-1): {feature}",
            xaxis_title="Date",
            yaxis_title="Change in Load (MW)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Summary stats
        kpis = {
            "days_exceeding": len(exceeding_days),
            "days_below": len(daily_avg[daily_avg['delta'] < 0]),
            "total_days": len(daily_avg.dropna()),
            "avg_increase_when_exceeding": float(exceeding_days['delta'].mean()) if not exceeding_days.empty else 0,
            "max_increase": float(daily_avg['delta'].max()) if not daily_avg.empty else 0,
            "max_decrease": float(daily_avg['delta'].min()) if not daily_avg.empty else 0
        }
        
        # List of exceeding days
        exceeding_list = exceeding_days[['Date', 'avg', 'delta', 'delta_pct']].copy()
        exceeding_list['Date'] = exceeding_list['Date'].astype(str)
        exceeding_list = exceeding_list.sort_values('delta', ascending=False).head(20)
        
        return {
            "plot": self._fig_to_json(fig), 
            "kpis": self._sanitize_kpis(kpis),
            "top_exceeding_days": exceeding_list.to_dict(orient='records')
        }

    # ==========================================
    # BASELINE OPTIMIZER FOR FORECASTING
    # ==========================================
    
    def run_baseline_optimizer(self, df, feature, window_override=None):
        """Determine optimal lookback window for baseline forecast based on autocorrelation and stability."""
        daily_avg = df.groupby('Date')[feature].mean()
        
        # Test lookback windows from 1 to 15 days
        windows = list(range(1, 16))
        results = []

        for w in windows:
            if len(daily_avg) < w + 7:
                continue

            # Rolling prediction: use past w days to predict next day
            errors = []
            for i in range(w, len(daily_avg) - 1):
                baseline = daily_avg.iloc[i-w:i].mean()
                actual = daily_avg.iloc[i]
                errors.append(abs(actual - baseline) / actual * 100)  # MAPE

            if errors:
                mape = float(np.mean(errors))
                stability = float(np.std(errors))
                results.append({
                    'window': w,
                    'mape': mape,
                    'stability': stability,
                    'score': float(mape + stability * 0.5)
                })

        if not results:
            return {"error": "Insufficient data for baseline analysis"}

        results_df = pd.DataFrame(results)
        best_window = int(results_df.loc[results_df['score'].idxmin(), 'window'])
        best_row = results_df.loc[results_df['window'] == best_window].iloc[0]
        mape = float(best_row['mape'])
        stability = float(best_row['stability'])

        selected_window = int(window_override) if window_override and 1 <= int(window_override) <= 15 else best_window
        effective_window = selected_window or best_window
        
        # Calculate unexplained deviation (residual)
        best_baseline = daily_avg.rolling(int(effective_window)).mean()
        residuals = (daily_avg - best_baseline).abs()
        unexplained_dev_pct = (residuals / daily_avg * 100).mean()
        
        insight = f"Expected Load Physics: A {effective_window}-day baseline best explains Load behavior (Error: {mape:.1f}%). Unexplained Load deviation is {unexplained_dev_pct:.1f}%."

        # Create visualization
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Forecast Accuracy by Window (Lower is Better)', 'Expected vs Actual Load Physics'])
        
        fig.add_trace(go.Bar(
            x=[f"{w}d" for w in results_df['window']],
            y=results_df['mape'],
            name='Forecast Error (MAPE %)',
            marker_color=[FeatureRegistry.COLORS['primary'] if w == effective_window else FeatureRegistry.COLORS['baseline']
                         for w in results_df['window']]
        ), row=1, col=1)
        
        # Show baseline forecast
        fig.add_trace(go.Scatter(x=daily_avg.index, y=daily_avg.values, name='Actual Load', 
                                  line=dict(color=FeatureRegistry.COLORS['primary'])), row=2, col=1)
        fig.add_trace(go.Scatter(x=best_baseline.index, y=best_baseline.values, name=f'Expected Baseline ({effective_window}d)',
                                  line=dict(color=FeatureRegistry.COLORS['context'], dash='dash', width=2)), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Expected Load Physics Engine")
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "best_window_days": int(best_window),
                "baseline_error_mape": mape,
                "baseline_stability": stability,
                "unexplained_deviation_pct": float(unexplained_dev_pct),
                "used_window": int(effective_window),
                "selected_window": int(selected_window)
            }),
            "all_results": results_df.to_dict(orient='records')
        }

    # ==========================================
    # HOLIDAY/WEEKEND EFFECT ANALYSIS
    # ==========================================
    
    def run_calendar_effect_analysis(self, df, feature):
        """Analyze holiday, weekend, and seasonal effects on load with weather context."""
        # Group by day type
        weekday_avg = df[df['IsWeekend'] == 0].groupby('time_block')[feature].mean()
        weekend_avg = df[df['IsWeekend'] == 1].groupby('time_block')[feature].mean()
        
        # Season breakdown
        season_stats = df.groupby(['Season', 'IsWeekend'])[feature].agg(['mean', 'std']).reset_index()
        
        # Weather correlation by day type
        weather_cols = ['temperature', 'humidity', 'precipitation']
        weather_corr = {}
        for col in weather_cols:
            if col in df.columns:
                weekday_corr = df[df['IsWeekend'] == 0][[feature, col]].corr(method="spearman").iloc[0, 1]
                weekend_corr = df[df['IsWeekend'] == 1][[feature, col]].corr(method="spearman").iloc[0, 1]
                weather_corr[col] = {'weekday': float(weekday_corr), 'weekend': float(weekend_corr)}
        
        # Create visualization
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=['Weekday vs Weekend Profile', 'Season Effect', 
                                          'Load by Day of Week', 'Weather Sensitivity'])
        
        # Profile comparison
        fig.add_trace(go.Scatter(x=weekday_avg.index, y=weekday_avg.values, name='Weekday',
                                  line=dict(color=FeatureRegistry.COLORS['primary'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=weekend_avg.index, y=weekend_avg.values, name='Weekend',
                                  line=dict(color=FeatureRegistry.COLORS['context'])), row=1, col=1)
        
        # Season bars
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = season_stats[season_stats['Season'] == season]
            if not season_data.empty:
                fig.add_trace(go.Bar(name=season, x=['Weekday', 'Weekend'], 
                                     y=season_data['mean'].values), row=1, col=2)
        
        # Day of week
        dow_avg = df.groupby('DayOfWeek')[feature].mean()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = dow_avg.reindex(days_order)
        fig.add_trace(go.Bar(x=dow_avg.index, y=dow_avg.values, name='Avg Load',
                             marker_color=[FeatureRegistry.COLORS['up_shift'] if d in ['Saturday', 'Sunday'] 
                                          else FeatureRegistry.COLORS['primary'] for d in dow_avg.index]), row=2, col=1)
        
        # Weather sensitivity heatmap data
        if weather_corr:
            corr_data = []
            for col, vals in weather_corr.items():
                corr_data.append([vals['weekday'], vals['weekend']])
            fig.add_trace(go.Heatmap(z=corr_data, x=['Weekday', 'Weekend'], 
                                      y=list(weather_corr.keys()),
                                      colorscale='RdBu', zmid=0), row=2, col=2)
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         height=600, showlegend=True)
        
        # KPIs
        weekday_mean = float(weekday_avg.mean())
        weekend_mean = float(weekend_avg.mean())
        effect_pct = ((weekend_mean - weekday_mean) / weekday_mean) * 100
        
        return {
            "plot": self._fig_to_json(fig),
            "kpis": {
                "weekday_avg": weekday_mean,
                "weekend_avg": weekend_mean,
                "weekend_effect_pct": float(effect_pct),
                "strongest_season": season_stats.groupby('Season')['mean'].mean().idxmax()
            },
            "weather_correlations": weather_corr
        }

    # ==========================================
    # ENHANCED WEATHER IMPACT WITH ALL FACTORS
    # ==========================================
    
    def run_full_weather_impact(self, df):
        """Comprehensive weather impact analysis including temperature, humidity, and precipitation."""
        weather_cols = []
        load_col = 'total_drawal' if 'total_drawal' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        
        for col in ['temperature', 'humidity', 'precipitation']:
            if col in df.columns:
                weather_cols.append(col)
        
        if not weather_cols:
            return {"error": "No weather columns found in data"}
        
        # Create subplots
        n_cols = len(weather_cols)
        fig = make_subplots(rows=2, cols=n_cols, 
                           subplot_titles=[f'{c.title()} vs Load' for c in weather_cols] + 
                                          [f'{c.title()} Distribution' for c in weather_cols])
        
        correlations = {}
        for i, col in enumerate(weather_cols):
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=df[col], y=df[load_col], mode='markers',
                marker=dict(size=3, opacity=0.3, color=FeatureRegistry.COLORS['primary']),
                name=f'{col.title()}'
            ), row=1, col=i+1)
            
            # Add trendline
            z = np.polyfit(df[col].dropna(), df.loc[df[col].notna(), load_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df[col].min(), df[col].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_trend, y=p(x_trend), mode='lines',
                line=dict(color=FeatureRegistry.COLORS['context'], width=2),
                name=f'{col.title()} Trend'
            ), row=1, col=i+1)
            
            # Distribution
            fig.add_trace(go.Histogram(x=df[col], nbinsx=30, name=col.title(),
                                       marker_color=FeatureRegistry.COLORS['secondary']), row=2, col=i+1)
            
            # Correlation
            correlations[col] = float(df[[load_col, col]].corr(method="spearman").iloc[0, 1])
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         height=600, showlegend=False)
        
        # Find strongest impact
        if correlations:
            strongest = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        else:
            strongest = "N/A"
        
        return {
            "plot": self._fig_to_json(fig),
            "kpis": {
                "temp_correlation": correlations.get('temperature', 0),
                "humidity_correlation": correlations.get('humidity', 0),
                "precip_correlation": correlations.get('precipitation', 0),
                "strongest_factor": strongest
            },
            "correlations": correlations
        }

    # ==========================================
    # DAYS T>T-1 WITH PRECIPITATION
    # ==========================================
    
    def run_days_exceeding_with_weather(self, df, feature):
        """Days where load exceeded previous day, with precipitation context."""
        # Daily aggregation
        agg_cols = {feature: 'mean'}
        if 'precipitation' in df.columns:
            agg_cols['precipitation'] = 'sum'
        if 'temperature' in df.columns:
            agg_cols['temperature'] = 'mean'
        
        daily = df.groupby('Date').agg(agg_cols).reset_index()
        daily = daily.sort_values('Date')
        
        # Calculate changes
        daily['prev_load'] = daily[feature].shift(1)
        daily['delta'] = daily[feature] - daily['prev_load']
        daily['exceeds'] = daily['delta'] > 0
        
        if 'precipitation' in daily.columns:
            daily['precip_day'] = daily['precipitation'] > 0.1
        
        # Create visualization
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=['Daily Load Change (T - T-1)', 'Precipitation'])
        
        # Load change bars
        fig.add_trace(go.Bar(
            x=daily['Date'], y=daily['delta'],
            name='Load Change',
            marker_color=[FeatureRegistry.COLORS['up_shift'] if d > 0 else FeatureRegistry.COLORS['down_shift'] 
                         for d in daily['delta'].fillna(0)]
        ), row=1, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=1, col=1)
        
        # Precipitation bars
        if 'precipitation' in daily.columns:
            fig.add_trace(go.Bar(
                x=daily['Date'], y=daily['precipitation'],
                name='Precipitation',
                marker_color=FeatureRegistry.COLORS['secondary']
            ), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         height=500)
        
        # Analysis: Does precipitation affect load increase?
        if 'precipitation' in daily.columns:
            precip_days = daily[daily['precip_day'] == True]
            no_precip_days = daily[daily['precip_day'] == False]
            precip_exceed_rate = precip_days['exceeds'].mean() * 100 if not precip_days.empty else 0
            no_precip_exceed_rate = no_precip_days['exceeds'].mean() * 100 if not no_precip_days.empty else 0
        else:
            precip_exceed_rate = 0
            no_precip_exceed_rate = 0
        
        exceeding_days = daily[daily['delta'] > 0]
        
        kpis = {
            "days_exceeding": len(exceeding_days),
            "days_below": len(daily[daily['delta'] < 0]),
            "total_days": len(daily.dropna()),
            "precip_exceed_rate": float(precip_exceed_rate),
            "no_precip_exceed_rate": float(no_precip_exceed_rate),
            "avg_increase": float(exceeding_days['delta'].mean()) if not exceeding_days.empty else 0
        }
        
        return {
            "plot": self._fig_to_json(fig),
            "kpis": self._sanitize_kpis(kpis)
        }

    # ==========================================
    # PHASE 2: FORECAST & SCENARIO INTELLIGENCE
    # ==========================================

    def run_multi_horizon_forecast(self, df, feature, window=7):
        """Generates a 48-hour multi-horizon forecast using hybrid persistence-regression."""
        # Preparation: Get the last `window` days profile
        last_date = df['Date'].max()
        last_hist_days = df[df['Date'] > (last_date - pd.Timedelta(days=window))]
        Profile_Avg = last_hist_days.groupby('time_block')[feature].mean()
        Profile_Std = last_hist_days.groupby('time_block')[feature].std()

        # Projection
        future_dates = [last_date + pd.Timedelta(days=1), last_date + pd.Timedelta(days=2)]
        forecast_series = []
        for d in future_dates:
            for b in range(1, 97):
                forecast_series.append({
                    'Date': d,
                    'time_block': b,
                    'forecast': Profile_Avg.get(b, 0),
                    'upper': Profile_Avg.get(b, 0) + 1.96 * Profile_Std.get(b, 0),
                    'lower': max(0, Profile_Avg.get(b, 0) - 1.96 * Profile_Std.get(b, 0))
                })
        
        forecast_df = pd.DataFrame(forecast_series)
        forecast_df['Timestamp'] = forecast_df.apply(lambda row: pd.to_datetime(row['Date']) + pd.Timedelta(minutes=(row['time_block']-1)*15), axis=1)

        # Historical context for plotting (last 3 days)
        hist_context = df[df['Date'] > (last_date - pd.Timedelta(days=3))].copy()
        hist_context['Timestamp'] = hist_context.apply(lambda row: pd.to_datetime(row['Date']) + pd.Timedelta(minutes=(row['time_block']-1)*15), axis=1)

        # Insights
        avg_forecast = float(forecast_df['forecast'].mean())
        peak_forecast = float(forecast_df['forecast'].max())
        peak_time = forecast_df.loc[forecast_df['forecast'].idxmax(), 'Timestamp']
        uncertainty_range = float((forecast_df['upper'] - forecast_df['lower']).mean())
        
        insight = f"Load Forecast: Projected Load averages {avg_forecast:.0f} MW for next 48h. Peak of {peak_forecast:.0f} MW expected around {peak_time.strftime('%H:%M')}. Uncertainty band ±{uncertainty_range/2:.0f} MW."

        fig = go.Figure()
        # Historical
        fig.add_trace(go.Scatter(x=hist_context['Timestamp'], y=hist_context[feature], name='Historical Load', line=dict(color=FeatureRegistry.COLORS['primary'], width=2)))
        # Forecast
        fig.add_trace(go.Scatter(x=forecast_df['Timestamp'], y=forecast_df['forecast'], name='Load Forecast', line=dict(color=FeatureRegistry.COLORS['highlight'], dash='dash', width=2)))
        # Confidence Intervals
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['Timestamp'], forecast_df['Timestamp'][::-1]]),
            y=pd.concat([forecast_df['upper'], forecast_df['lower'][::-1]]),
            fill='toself', fillcolor='rgba(0, 240, 255, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='95% Confidence Band'
        ))

        fig.update_layout(title="Load Forecast (Multi-Horizon)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')

        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "forecast_horizon_hours": 48,
                "model_type": f"Seasonal Persistence ({window}D)",
                "avg_expected_load_mw": avg_forecast,
                "peak_expected_load_mw": peak_forecast,
                "uncertainty_band_mw": uncertainty_range,
                "window_used": int(window)
            })
        }

    def run_residual_analysis(self, df, feature):
        """Analyzes forecast residuals and distribution."""
        # Simple backtest: Compare T vs T-7 as a proxy for "naive forecast"
        df_sorted = df.sort_values(['time_block', 'Date'])
        df_sorted['naive_forecast'] = df_sorted.groupby('time_block')[feature].shift(7)
        valid = df_sorted.dropna(subset=['naive_forecast']).copy()
        
        if valid.empty:
            return {"error": "Insufficient history for residual analysis (Need >7 days)."}

        valid['residual'] = valid[feature] - valid['naive_forecast']
        valid['abs_error'] = valid['residual'].abs()
        valid['mape'] = (valid['abs_error'] / valid[feature].replace(0, np.nan)) * 100

        # Insights
        mape = float(valid['mape'].mean())
        bias = float(valid['residual'].mean())
        bias_desc = "under-forecasting" if bias > 0 else "over-forecasting"
        
        insight = f"Forecast Error Analysis: Historical forecast error averages {mape:.1f}% MAPE. Bias indicates slight {bias_desc} ({abs(bias):.1f} MW)."

        # Plots
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Error Distribution (Residuals)", "Hourly Forecast Bias Profile"])
        
        # Error Dist
        fig.add_trace(go.Histogram(x=valid['residual'], nbinsx=50, name='Residual Deviations', marker_color=FeatureRegistry.COLORS['down_shift']), row=1, col=1)
        
        # Hourly Bias
        hourly_bias = valid.groupby('time_block')['residual'].mean().reset_index()
        fig.add_trace(go.Bar(x=hourly_bias['time_block'], y=hourly_bias['residual'], name='Avg Bias per Block', marker_color=FeatureRegistry.COLORS['primary']), row=2, col=1)

        fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Forecast Error Analysis")

        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "historical_mape_pct": mape,
                "rmse_mw": float(np.sqrt((valid['residual']**2).mean())),
                "forecast_bias_mw": bias,
                "max_under_forecast_mw": float(valid['residual'].min()),
                "max_over_forecast_mw": float(valid['residual'].max())
            })
        }

    def run_scenario_stress_test(self, df, feature, temp_delta=2.0, hum_delta=5.0, is_raining=False):
        """
        Expanded Grid Stress Scenario Framework.
        Fuses Weather, Calendar, and Hybrid stresses into a Composite Grid Stress Index.
        """
        if df.empty: return {"error": "No data for stress testing."}
        
        # 1. Weather Sensitivity Engine
        # --------------------------------------------------
        # Multi-variable regression: Load ~ Temperature + Humidity
        # We use residuals to isolate effects
        # Temperature & Humidity Elasticity
        if 'temperature' in df.columns and 'humidity' in df.columns:
            X = df[['temperature', 'humidity']].fillna(df[['temperature', 'humidity']].mean())
            y = df[feature]
            reg = LinearRegression().fit(X, y)
            temp_beta = reg.coef_[0] # MW per Degree
            hum_beta = reg.coef_[1]  # MW per % Humidity
        else:
            temp_beta = 0
            hum_beta = 0

        # Rain Sensitivity (Load Dip)
        rain_dip_mw = 0
        if 'precipitation' in df.columns:
            rain_days = df[df['precipitation'] > 0.5][feature].mean()
            dry_days = df[df['precipitation'] <= 0.5][feature].mean()
            if not np.isnan(rain_days) and not np.isnan(dry_days):
                rain_dip_mw = dry_days - rain_days # MW dropped during rain

        # 2. Stress Dimension Calculations
        # --------------------------------------------------
        # A) Weather Stress
        weather_stress_mw = (temp_beta * temp_delta) + (hum_beta * hum_delta) - (rain_dip_mw if is_raining else 0)
        
        # B) Calendar Stress (Baseline deviations)
        weekday_avg = df[df['IsWeekend'] == 0][feature].mean()
        weekend_avg = df[df['IsWeekend'] == 1][feature].mean()
        weekend_drop_pct = self._pct(weekday_avg - weekend_avg, weekday_avg)
        
        holiday_avg = df[df['is_holiday'] == 1][feature].mean()
        holiday_suppression_mw = (weekday_avg - holiday_avg) if not np.isnan(holiday_avg) else 0

        # C) Hybrid Stress (Contextual Amplifiers)
        friday_amp = self._pct(df[df['IsFriday'] == 1][feature].mean() - weekday_avg, weekday_avg)
        monday_ramp = self._pct(df[df['IsMonday'] == 1][feature].mean() - weekday_avg, weekday_avg)
        sandwich_distort = self._pct(df[df['sandwich_day'] == 1][feature].mean() - weekday_avg, weekday_avg)

        # 3. Composite Grid Stress Index (0-100)
        # --------------------------------------------------
        # Normalize MW impacts into a score
        base_load = df[feature].mean()
        
        w_score = abs(weather_stress_mw / base_load * 100) * 1.5 # Weight 1.5
        c_score = abs(weekend_drop_pct) + abs(holiday_suppression_mw / base_load * 100)
        h_score = (abs(friday_amp) + abs(monday_ramp) + abs(sandwich_distort)) * 2.0 # Weight 2.0
        
        composite_index = min(100, (w_score * 0.4 + c_score * 0.3 + h_score * 0.3))

        # 4. Scenario Simulation Trace
        # --------------------------------------------------
        latest_avg = df.groupby('time_block')[feature].mean()
        simulated = latest_avg + weather_stress_mw
        
        # Apply hybrid multipliers to the curve shape
        if friday_amp > 0: simulated = simulated * (1 + friday_amp/100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=latest_avg.index, y=latest_avg.values, name='Base Profile', line=dict(color=FeatureRegistry.COLORS['baseline'], dash='dot')))
        fig.add_trace(go.Scatter(x=latest_avg.index, y=simulated.values, name='Stressed Scenario', 
                                 line=dict(color=FeatureRegistry.COLORS['critical'] if composite_index > 50 else FeatureRegistry.COLORS['primary'])))

        fig.update_layout(title="Composite Grid Stress Simulation", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # 5. UI Metadata & Chips
        # --------------------------------------------------
        chips = []
        if temp_delta > 3: chips.append("🌡️ Heat Wave")
        if hum_delta > 10: chips.append("💧 High Humidity")
        if is_raining: chips.append("🌧️ Rain Active")
        if friday_amp > 2: chips.append("📅 Pre-Weekend Build")
        if abs(sandwich_distort) > 5: chips.append("🥪 Sandwich Day")
        
        # Risk Banner
        banner = "Normal Grid State"
        if composite_index > 70: banner = "🚨 CRITICAL LOAD STRESS"
        elif composite_index > 40: banner = "⚠️ EVALUATED PEAK RISK"

        return {
            "plot": self._fig_to_json(fig),
            "stress_index": round(composite_index, 1),
            "breakdown": {
                "Weather": round(w_score, 1),
                "Calendar": round(c_score, 1),
                "Hybrid": round(h_score, 1)
            },
            "chips": chips,
            "banner": banner,
            "kpis": {
                "Grid Stress Index": f"{composite_index:.1f}/100",
                "Temp Sensitivity": f"{temp_beta:.2f} MW/°C",
                "Humidity Impact": f"{(hum_beta * hum_delta):.2f} MW",
                "Rain Load Dip": f"{rain_dip_mw:.2f} MW",
                "Peak Expected": float(simulated.max())
            }
        }


    def run_scenario_heatmap(self, df, feature, mode='scenario'):
        """
        Phase 3: Decision-Grade Scenario Heatmap.
        Converts raw temperature bins into semantic Risk Scenarios.
        """
        if 'temperature' not in df.columns:
            return {"error": "Temperature required for heatmap."}
            
        data = df.copy()
        base_load = data[feature].mean()
        
        # Calculate sensitivites
        X = data[['temperature', 'humidity']].fillna(data[['temperature', 'humidity']].mean())
        reg = LinearRegression().fit(X, data[feature])
        t_beta, h_beta = reg.coef_[0], reg.coef_[1]

        # 1. Define Grouping Dimension
        if mode == 'temp':
            # Raw Bins (Original Logic)
            min_t, max_t = data['temperature'].min(), data['temperature'].max()
            data['Scenario'] = pd.cut(data['temperature'], 
                                    bins=np.arange(np.floor(min_t), np.ceil(max_t)+5, 5)).astype(str)
        else:
            # Semantic Scenarios
            data['temp_cat'] = pd.cut(data['temperature'], bins=[-10, 15, 25, 35, 100], labels=['Cool', 'Mild', 'Hot', 'Extreme'])
            data['hum_cat'] = pd.cut(data['humidity'], bins=[-1, 40, 70, 100], labels=['Dry', 'Normal', 'Humid'])
            data['Scenario'] = data['temp_cat'].astype(str) + "-" + data['hum_cat'].astype(str)

        # 2. Aggregation Logic
        pivot_data = []
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days:
            day_df = data[data['DayOfWeek'] == day]
            for scen in data['Scenario'].unique():
                seg = day_df[day_df['Scenario'] == scen]
                if seg.empty: continue
                
                # Composite Calculation
                avg_t = seg['temperature'].mean()
                avg_h = seg['humidity'].mean()
                
                w_stress = abs(((avg_t - data['temperature'].mean()) * t_beta + (avg_h - data['humidity'].mean()) * h_beta) / base_load * 100)
                
                day_num = seg['DayOfWeekNum'].iloc[0]
                # High-weight Friday/Monday ramp and Weekend variance
                c_stress = 15 if day_num in [5, 6] else 5
                if day_num in [0, 4]: c_stress += 10
                
                composite = min(100, (w_stress * 0.7 + c_stress * 0.3))
                
                # Semantic Risk Mapping
                risk_level = "Green" # Low
                if composite > 8: risk_level = "Red" # Critical
                elif composite > 5: risk_level = "Orange" # Elevated
                elif composite > 2: risk_level = "Amber" # Watch
                
                # Annotations
                note = ""
                if seg[feature].max() > data[feature].quantile(0.95): note += "▲ " # Peak
                if day_num == 4: note += "⚡ " # Friday Ramp
                if avg_h > 75: note += "💧 " # Humidity driver
                
                pivot_data.append({
                    "Day": day,
                    "Scenario": scen,
                    "Stress Index": round(composite, 2),
                    "Risk Level": risk_level,
                    "Label": f"{round(composite, 1)} {note}",
                    "Drivers": f"Temp: {avg_t:.1f}°C, Hum: {avg_h:.1f}%",
                    "Action": "🚨 Critical Monitoring" if composite > 8 else "⚠️ Enhanced Vigilance" if composite > 5 else "Normal Ops"
                })

        pdf = pd.DataFrame(pivot_data)

        # Create Heatmap
        # We use a custom order for scenarios to ensure mental continuity
        scen_order = []
        for t in ['Cool', 'Mild', 'Hot', 'Extreme']:
            for h in ['Dry', 'Normal', 'Humid']:
                s = f"{t}-{h}"
                if s in pdf['Scenario'].unique(): scen_order.append(s)

        fig = px.imshow(
            pdf.pivot(index="Day", columns="Scenario", values="Stress Index").reindex(index=days, columns=scen_order),
            labels=dict(x="Scenario", y="Grid Load Patterns", color="Stress Index"),
            color_continuous_scale=[[0, "#2f8f82"], [0.2, "#2f8f82"], [0.2, "#d4a373"], [0.5, "#b97833"], [0.8, "#b35454"], [1, "#b35454"]],
            title="Where Grid Stress Concentrates (Weather × Calendar Interaction)"
        )
        
        # Add labels inside cells
        pivot_labels = pdf.pivot(index="Day", columns="Scenario", values="Label").reindex(index=days, columns=scen_order)
        for i, day in enumerate(days):
            for j, scen in enumerate(scen_order):
                val = pivot_labels.iloc[i, j]
                if pd.notna(val):
                    fig.add_annotation(x=scen, y=day, text=str(val), showarrow=False, font=dict(color="white", size=10))

        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Identify Highest Risk for Summary Strip
        top_node = pdf.loc[pdf['Stress Index'].idxmax()]

        return {
            "plot": self._fig_to_json(fig),
            "summary": f"{top_node['Risk Level'].upper()} RISK: {top_node['Scenario']} on {top_node['Day']} (Index {top_node['Stress Index']})",
            "kpis": {
                "Max Stress Node": f"{top_node['Stress Index']}%",
                "Primary Driver": f"{top_node['Scenario']} Scenarios",
                "Action Logic": top_node['Action']
            }
        }

    # ==========================================
    # SCENARIO DEVIATION SIMULATOR
    # ==========================================

    def run_scenario_deviation_simulator(self, df, temp_delta=0.0, hum_delta=0.0,
                                          is_raining=False, is_weekend=False,
                                          is_holiday=False, is_pre_weekend=False,
                                          granularity="block", adjustments=None):
        """
        Core Scenario Deviation Simulator.
        Calculates: ΔDriver → ΔLoad
        Returns: MW deviation, % deviation, peak shift, driver contributions, insight.
        """
        feature = 'total_drawal'
        if df.empty or feature not in df.columns:
            return {"error": "Load data required for deviation simulation."}
        
        base_load = df[feature].mean()

        # 1. Calculate Driver Sensitivities
        # --------------------------------------------------
        temp_beta, hum_beta = 0.0, 0.0
        if 'temperature' in df.columns and 'humidity' in df.columns:
            X = df[['temperature', 'humidity']].fillna(df[['temperature', 'humidity']].mean())
            reg = LinearRegression().fit(X, df[feature])
            temp_beta = reg.coef_[0]  # MW per °C
            hum_beta = reg.coef_[1]   # MW per % Humidity

        # Rain Sensitivity
        rain_dip_mw = 0.0
        if 'precipitation' in df.columns:
            rain_days = df[df['precipitation'] > 0.5][feature].mean()
            dry_days = df[df['precipitation'] <= 0.5][feature].mean()
            if not np.isnan(rain_days) and not np.isnan(dry_days):
                rain_dip_mw = dry_days - rain_days

        # Calendar Sensitivities
        weekday_avg = df[df['IsWeekend'] == 0][feature].mean()
        weekend_avg = df[df['IsWeekend'] == 1][feature].mean()
        weekend_drop_mw = weekday_avg - weekend_avg if not np.isnan(weekend_avg) else 0

        holiday_avg = df[df['is_holiday'] == 1][feature].mean()
        holiday_drop_mw = (weekday_avg - holiday_avg) if not np.isnan(holiday_avg) else 0

        friday_avg = df[df['IsFriday'] == 1][feature].mean()
        friday_surge_mw = (friday_avg - weekday_avg) if not np.isnan(friday_avg) else 0

        # 2. Calculate Total Load Deviation
        # --------------------------------------------------
        weather_deviation = (temp_beta * temp_delta) + (hum_beta * hum_delta)
        if is_raining:
            weather_deviation -= rain_dip_mw

        calendar_deviation = 0.0
        if is_weekend:
            calendar_deviation -= weekend_drop_mw
        if is_holiday:
            calendar_deviation -= holiday_drop_mw
        if is_pre_weekend:
            calendar_deviation += friday_surge_mw

        total_deviation_mw = weather_deviation + calendar_deviation

        # 3. Peak Shift & Ramp Amplification
        # --------------------------------------------------
        # Estimate peak shift based on humidity (high hum = later peak)
        peak_shift_blocks = int(hum_delta / 10) if hum_delta > 0 else 0
        
        # Ramp amplification on hot + humid days
        ramp_amplification = abs(temp_delta * 2 + hum_delta * 0.5) if temp_delta > 0 else 0

        # 4. Driver Contribution Breakdown
        # --------------------------------------------------
        total_abs = abs(weather_deviation) + abs(calendar_deviation) + 0.01
        driver_contributions = {
            "Temperature": round(abs(temp_beta * temp_delta) / total_abs * 100, 1),
            "Humidity": round(abs(hum_beta * hum_delta) / total_abs * 100, 1),
            "Rain": round(abs(rain_dip_mw if is_raining else 0) / total_abs * 100, 1),
            "Calendar": round(abs(calendar_deviation) / total_abs * 100, 1)
        }

        # 5. Simulated Load Curve (Base)
        # --------------------------------------------------
        granularity_key = (granularity or "block").strip().lower()
        peak_shift_units = peak_shift_blocks
        peak_shift_label = "blocks"
        if granularity_key == "hourly":
            peak_shift_units = int(round(peak_shift_blocks / 4))
            peak_shift_label = "hours"
            base_curve = df.groupby('Hour')[feature].mean()
            x_axis_title = "Hour"
            shift_units = peak_shift_units
        else:
            base_curve = df.groupby('time_block')[feature].mean()
            x_axis_title = "Time Block"
            shift_units = peak_shift_blocks

        simulated_curve = base_curve + total_deviation_mw
        if shift_units != 0:
            simulated_curve = simulated_curve.shift(shift_units).fillna(method='bfill')

        # 6. Apply manual per-hour/block adjustments (MW)
        # --------------------------------------------------
        adjustments_list = adjustments if isinstance(adjustments, list) else []
        adjustment_map = {}
        for item in adjustments_list:
            if not isinstance(item, dict):
                continue
            if granularity_key == "hourly":
                key = item.get("hour")
            else:
                key = item.get("block")
            delta = item.get("delta_mw")
            try:
                key = int(key)
                delta_val = float(delta)
            except (TypeError, ValueError):
                continue
            adjustment_map[key] = delta_val

        manual_adjust_avg = 0.0
        if adjustment_map:
            simulated_curve = simulated_curve.copy()
            for idx in simulated_curve.index:
                adj = adjustment_map.get(int(idx), 0.0)
                if adj:
                    simulated_curve.loc[idx] = simulated_curve.loc[idx] + adj
            manual_adjust_avg = sum(adjustment_map.values()) / float(len(base_curve))

        # 7. Stress Classification (post-adjustments)
        # --------------------------------------------------
        adjusted_deviation_mw = total_deviation_mw + manual_adjust_avg
        adjusted_deviation_pct = (adjusted_deviation_mw / base_load) * 100

        stress_class = "Low"
        if abs(adjusted_deviation_pct) > 10:
            stress_class = "Critical"
        elif abs(adjusted_deviation_pct) > 6:
            stress_class = "Elevated"
        elif abs(adjusted_deviation_pct) > 3:
            stress_class = "Moderate"

        # 8. Generate Insight
        # --------------------------------------------------
        drivers = []
        if temp_delta != 0: drivers.append(f"{'+' if temp_delta > 0 else ''}{temp_delta} degC temperature")
        if hum_delta != 0: drivers.append(f"{'+' if hum_delta > 0 else ''}{hum_delta}% humidity")
        if is_raining: drivers.append("rain event")
        if is_weekend: drivers.append("weekend")
        if is_holiday: drivers.append("holiday")
        if is_pre_weekend: drivers.append("pre-weekend Friday")
        driver_str = " + ".join(drivers) if drivers else "baseline conditions"
        direction = "increased" if adjusted_deviation_mw > 0 else "decreased"
        insight = f"{driver_str.capitalize()} {direction} Load by {abs(adjusted_deviation_mw):.1f} MW ({abs(adjusted_deviation_pct):.1f}%), classifying grid stress as {stress_class}."
        if adjustment_map:
            insight += f" Manual adjustments applied to {len(adjustment_map)} {('hours' if granularity_key == 'hourly' else 'blocks')}."

        # 9. Load Sensitivity KPIs
        # --------------------------------------------------
        kpis = {
            "Load MW Deviation": f"{adjusted_deviation_mw:+.1f} MW",
            "Load % Deviation": f"{adjusted_deviation_pct:+.1f}%",
            "Temp Elasticity": f"{temp_beta:.2f} MW/degC",
            "Humidity Factor": f"{hum_beta:.2f} MW/%",
            "Rain Suppression": f"{rain_dip_mw:.1f} MW",
            "Weekend Drop": f"{weekend_drop_mw:.1f} MW",
            "Holiday Loss": f"{holiday_drop_mw:.1f} MW",
            "Friday Surge": f"{friday_surge_mw:.1f} MW",
            "Peak Shift": f"{peak_shift_units} {peak_shift_label}",
            "Manual Adjust Avg": f"{manual_adjust_avg:+.2f} MW",
            "Ramp Amplification": f"{ramp_amplification:.1f}%",
            "Stress Classification": stress_class
        }

        # 10. Build per-hour/block series table
        series_rows = []
        for idx in base_curve.index:
            base_val = float(base_curve.loc[idx])
            sim_val = float(simulated_curve.loc[idx])
            row = {
                "baseline_mw": round(base_val, 2),
                "simulated_mw": round(sim_val, 2),
                "deviation_mw": round(sim_val - base_val, 2)
            }
            if granularity_key == "hourly":
                row["hour"] = int(idx)
            else:
                row["block"] = int(idx)
            series_rows.append(row)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=base_curve.index, y=base_curve.values, 
                                  name='Baseline Load', line=dict(color='#6b9080', dash='dot')))
        fig.add_trace(go.Scatter(x=simulated_curve.index, y=simulated_curve.values, 
                                  name='Simulated Load', 
                                  line=dict(color='#b35454' if stress_class == 'Critical' else '#d4a373')))
        
        # Add deviation band
        fig.add_trace(go.Scatter(
            x=list(base_curve.index) + list(base_curve.index[::-1]),
            y=list(base_curve.values) + list(simulated_curve.values[::-1]),
            fill='toself',
            fillcolor='rgba(179, 84, 84, 0.2)' if total_deviation_mw > 0 else 'rgba(45, 143, 130, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Deviation Band'
        ))

        fig.update_layout(
            title=f"Load Deviation Simulation: {stress_class} Stress",
            xaxis_title=x_axis_title,
            yaxis_title="Load (MW)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return {
            "plot": self._fig_to_json(fig),
            "load_mw_deviation": round(adjusted_deviation_mw, 2),
            "load_pct_deviation": round(adjusted_deviation_pct, 2),
            "peak_shift_blocks": peak_shift_blocks,
            "ramp_amplification_pct": round(ramp_amplification, 1),
            "stress_classification": stress_class,
            "driver_contributions": driver_contributions,
            "insight": insight,
            "kpis": kpis,
            "granularity": granularity_key,
            "sim_series": series_rows
        }

    # ==========================================
    # PHASE 3: ANOMALY & RISK LAYER
    # ==========================================

    def run_smart_anomaly_detection(self, df, feature):
        """Weather-decoupled anomaly detection using residuals."""
        if 'temperature' not in df.columns:
            return {"error": "Temperature data required for weather-decoupled anomalies."}
            
        # 1. Calculate Residuals (Load - Seasonal - Weather)
        # Clean data for regression to prevent 500 errors
        clean_df = df.copy().dropna(subset=[feature, 'temperature'])
        if len(clean_df) < 10:
             return {"error": "Insufficient data (after cleaning NaN/Null) for smart anomaly detection."}

        clean_df['Weekday'] = pd.to_datetime(clean_df['Date']).dt.weekday
        medians = clean_df.groupby(['Weekday', 'time_block'])[feature].median().to_dict()
        clean_df['seasonal_comp'] = clean_df.apply(lambda x: medians.get((pd.to_datetime(x['Date']).weekday(), x['time_block']), 0), axis=1)
        
        X = clean_df[['temperature']].values
        y = clean_df[feature].values - clean_df['seasonal_comp'].values
        
        try:
            lr = LinearRegression().fit(X, y)
            clean_df['weather_comp'] = lr.predict(X)
            clean_df['residual'] = clean_df[feature] - clean_df['seasonal_comp'] - clean_df['weather_comp']
            
            # 2. Detect Outliers in Residuals using nan-safe zscore logic
            res_std = clean_df['residual'].std()
            if res_std == 0:
                clean_df['is_anomaly'] = False
            else:
                z_scores = (clean_df['residual'] - clean_df['residual'].mean()) / res_std
                clean_df['is_anomaly'] = np.abs(z_scores) > 2.5
        except Exception as e:
            return {"error": f"Statistical outlier calculation failed: {str(e)}"}
            
        anomalies = clean_df[clean_df['is_anomaly']].copy()
        
        # Insights
        n_anomalies = int(clean_df['is_anomaly'].sum())
        rate = (clean_df['is_anomaly'].mean()*100)
        max_dev = float(clean_df['residual'].abs().max())
        
        insight = f"Smart Load Distortion Detector: Found {n_anomalies} weather-decoupled anomalies ({rate:.1f}% rate). Max unexplained deviation is {max_dev:.0f} MW."

        # 3. Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=clean_df.index, y=clean_df['residual'], name='Weather-Decoupled Residual', line=dict(color=FeatureRegistry.COLORS['context'], width=1)))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['residual'], mode='markers', name='Smart Load Distortion', 
                                 marker=dict(color=FeatureRegistry.COLORS['critical'], size=8, symbol='x')))
        
        fig.update_layout(title="Smart Load Distortion Detector (Weather-Decoupled)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "smart_anomalies_detected": n_anomalies,
                "anomaly_rate_pct": rate,
                "max_unexplained_deviation_mw": max_dev,
                "root_cause_hint": "Non-weather volatility (Process/Shift)"
            })
        }

    def run_risk_monitor(self, df, feature):
        """Monitors volatility and trend risks."""
        # 1. Volatility tracking
        df_sorted = df.sort_values(['Date', 'time_block'])
        df_sorted['ramp'] = df_sorted[feature].diff()
        volatility = df_sorted['ramp'].rolling(window=96).std() # Rolling 24h volatility
        
        # 2. Trend direction
        daily_avg = df.groupby('Date')[feature].mean()
        momentum = daily_avg.diff().rolling(window=7).mean()
        
        # Insights
        current_vol = float(volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0)
        risk_level = "Elevated" if current_vol > volatility.mean() * 1.5 else "Stable"
        momentum_dir = "Increasing" if momentum.iloc[-1] > 0 else "Decreasing"
        
        insight = f"Load Risk Monitor: Current Grid Stress is {risk_level}. Volatility index is {current_vol:.1f}. Load momentum is {momentum_dir}."

        # 3. Create Monitoring Plot
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Intraday Ramping Risk (Volatility)", "Momentum Tracker (7D)"])
        fig.add_trace(go.Scatter(x=df_sorted.index, y=volatility, name='Ramp Volatility', fill='tozeroy', line=dict(color=FeatureRegistry.COLORS['warning'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=momentum.index, y=momentum.values, name='Load Momentum', line=dict(color=FeatureRegistry.COLORS['primary'])), row=2, col=1)
        
        fig.update_layout(height=600, title="Load Risk Monitor", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return {
            "plot": self._fig_to_json(fig),
            "insight": insight,
            "kpis": self._sanitize_kpis({
                "current_risk_level": risk_level,
                "max_ramp_mw": float(df_sorted['ramp'].abs().max()),
                "momentum_direction": momentum_dir,
                "volatility_index": current_vol
            })
        }

    def generate_alerts(self, df, feature):
        """Generates prioritized alert list based on multiple risk factors."""
        alerts = []
        
        # A. Absolute Peak Alert
        peak = df[feature].max()
        mean = df[feature].mean()
        if peak > mean * 1.4:
            alerts.append({
                "id": 1, "priority": "CRITICAL", "type": "Peak Threshold", 
                "message": f"Extreme peak detected at {peak:.1f} MW ({((peak/mean)-1)*100:.1f}% above avg).",
                "confidence": 0.95
            })
            
        # B. Level Shift Alert (TDLLS)
        daily_avg = df.groupby('Date')[feature].mean()
        shifts = daily_avg.diff()
        max_shift = shifts.abs().max()
        if max_shift > mean * 0.15:
            alerts.append({
                "id": 2, "priority": "WARNING", "type": "Base Shift", 
                "message": f"Significant base load shift of {max_shift:.1f} MW detected.",
                "confidence": 0.88
            })
            
        # C. Volatility Alert
        intraday_std = df.groupby('Date')[feature].std().mean()
        if intraday_std > mean * 0.2:
             alerts.append({
                "id": 3, "priority": "INFO", "type": "Unstable Profile", 
                "message": "High intraday volatility observed. Review block-level intelligence.",
                "confidence": 0.75
            })
            
        sorted_alerts = sorted(alerts, key=lambda x: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[x["priority"]])
        
        # Insights
        n_alerts = len(alerts)
        top_alert = sorted_alerts[0]['message'] if n_alerts > 0 else "System stable."
        insight = f"Load Consequence Alerts: Generated {n_alerts} priority alerts. Top Concern: {top_alert}"

        return {
            "alerts": sorted_alerts, 
            "alert_count": len(alerts),
            "insight": insight
        }

    # ==========================================
    # EXPORT CHART TO FILE
    # ==========================================
    
    def export_chart_to_file(self, fig_json, filename="chart", format="png"):
        """Export chart to image file."""
        from plotly.graph_objects import Figure
        
        # Reconstruct figure from JSON
        fig = Figure(data=fig_json.get('data', []), layout=fig_json.get('layout', {}))
        
        # Set up export path
        export_dir = os.path.join(os.path.dirname(self.data_path), "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(export_dir, f"{filename}_{timestamp}.{format}")
        
        # Export
        try:
            if format == "html":
                fig.write_html(filepath)
            else:
                fig.write_image(filepath, format=format, width=1200, height=800, scale=2)
            return {"success": True, "filepath": filepath}
        except Exception as e:
            # Fallback to HTML if static export dependencies are missing
            fallback_path = os.path.join(export_dir, f"{filename}_{timestamp}.html")
            try:
                fig.write_html(fallback_path)
                return {"success": False, "error": str(e), "fallback": fallback_path}
            except Exception:
                return {"success": False, "error": str(e)}

    # ==========================================
    # GENERATE REPORT
    # ==========================================
    
    def generate_report(self, df, feature, start_date, end_date):
        """Generate comprehensive analysis report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "date_range": {"start": start_date, "end": end_date},
            "feature": feature,
            "summary": {},
            "sections": []
        }
        
        # Basic statistics
        data = df[feature].dropna()
        report["summary"] = {
            "total_records": len(df),
            "date_range_days": (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days,
            "average": float(data.mean()),
            "peak": float(data.max()),
            "minimum": float(data.min()),
            "std_deviation": float(data.std()),
            "coefficient_of_variation": float(data.std() / data.mean() * 100) if data.mean() != 0 else 0
        }
        
        # Trend analysis
        daily_avg = df.groupby('Date')[feature].mean()
        if len(daily_avg) > 7:
            trend_start = daily_avg.iloc[:7].mean()
            trend_end = daily_avg.iloc[-7:].mean()
            trend_change = ((trend_end - trend_start) / trend_start) * 100
            report["sections"].append({
                "name": "Trend Analysis",
                "findings": [
                    f"7-day average at start: {trend_start:.0f}",
                    f"7-day average at end: {trend_end:.0f}",
                    f"Overall trend: {'+' if trend_change > 0 else ''}{trend_change:.1f}%"
                ]
            })
        
        # Weekend effect
        weekday_avg = df[df['IsWeekend'] == 0][feature].mean()
        weekend_avg = df[df['IsWeekend'] == 1][feature].mean()
        weekend_effect = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        report["sections"].append({
            "name": "Calendar Effects",
            "findings": [
                f"Weekday average: {weekday_avg:.0f}",
                f"Weekend average: {weekend_avg:.0f}",
                f"Weekend effect: {'+' if weekend_effect > 0 else ''}{weekend_effect:.1f}%"
            ]
        })
        
        # Weather impact
        if 'temperature' in df.columns:
            temp_corr = df[[feature, 'temperature']].corr(method="spearman").iloc[0, 1]
            report["sections"].append({
                "name": "Weather Impact",
                "findings": [
                    f"Temperature correlation: {temp_corr:.3f}",
                    f"Impact strength: {'Strong' if abs(temp_corr) > 0.5 else 'Moderate' if abs(temp_corr) > 0.3 else 'Weak'}"
                ]
            })
        
        # Anomalies
        z_scores = stats.zscore(data)
        anomaly_count = (np.abs(z_scores) > 3).sum()
        report["sections"].append({
            "name": "Anomaly Summary",
            "findings": [
                f"Outliers detected (|Z| > 3): {anomaly_count}",
                f"Anomaly rate: {anomaly_count / len(data) * 100:.2f}%"
            ]
        })
        
        return report
