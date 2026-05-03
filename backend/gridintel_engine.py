import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .engine import EDAEngine
except Exception:
    from engine import EDAEngine


@dataclass
class DataLayerConfig:
    data_path: str
    timezone: Optional[str] = None
    default_granularity: str = "15min"


class DataLayer:
    def __init__(self, config: DataLayerConfig):
        self.config = config
        self._raw_df = pd.DataFrame()
        self._df = pd.DataFrame()

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        elif "date" in df.columns and "time_block" in df.columns:
            df["Datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta((df["time_block"] - 1) * 15, unit="m")
        df = df.sort_values("Datetime")
        return df

    def get_data(self, start_date: str, end_date: str, granularity: str = "15min") -> pd.DataFrame:
        if self._df.empty:
            return self._df
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = self._df[(self._df["Datetime"] >= start) & (self._df["Datetime"] <= end)].copy()
        if df.empty:
            return df
        if granularity == "hourly":
            df = df.set_index("Datetime").resample("1h").mean(numeric_only=True).reset_index()
        elif granularity == "daily":
            df = df.set_index("Datetime").resample("1d").mean(numeric_only=True).reset_index()
        return df


class FeatureEngineering:
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Datetime" not in df.columns:
            return df
        df = df.copy()
        df["Date"] = df["Datetime"].dt.date
        df["Hour"] = df["Datetime"].dt.hour
        df["DayOfWeek"] = df["Datetime"].dt.day_name()
        df["DayOfWeekNum"] = df["Datetime"].dt.dayofweek
        df["Month"] = df["Datetime"].dt.month_name()
        df["MonthNum"] = df["Datetime"].dt.month
        df["IsWeekend"] = df["DayOfWeekNum"].isin([5, 6]).astype(int)
        df["Season"] = df["MonthNum"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })

        if "total_drawal" in df.columns:
            blocks_per_day = 96
            if "time_block" not in df.columns:
                df["time_block"] = (df["Datetime"].dt.hour * 4) + (df["Datetime"].dt.minute // 15) + 1
            df["load_D_same_block"] = df["total_drawal"].shift(blocks_per_day)
            df["load_D_minus_7"] = df["total_drawal"].shift(blocks_per_day * 7)
            df["rolling_mean_baseline"] = df["total_drawal"].rolling(blocks_per_day * 7, min_periods=1).mean()
            df["rolling_volatility"] = df["total_drawal"].rolling(blocks_per_day * 7, min_periods=1).std()

        if "temperature" in df.columns:
            df["cooling_degree_days"] = (df["temperature"] - 24.0).clip(lower=0)
            df["heating_degree_days"] = (18.0 - df["temperature"]).clip(lower=0)
            if "humidity" in df.columns:
                df["heat_index"] = 0.5 * (
                    df["temperature"] + 61.0 +
                    (df["temperature"] - 68.0) * 1.2 +
                    df["humidity"] * 0.094
                )

        if "precipitation" in df.columns:
            df["rain_flag"] = (df["precipitation"] > 0.1).astype(int)
            if "humidity" in df.columns:
                df["humidity_index"] = df["humidity"] / 100.0

        # New weather features
        if "apparent_temperature" in df.columns:
            df["apparent_cdd"] = (df["apparent_temperature"] - 24.0).clip(lower=0)
            df["apparent_hdd"] = (18.0 - df["apparent_temperature"]).clip(lower=0)
        
        if "cloud_cover" in df.columns:
            df["cloud_impact_flag"] = (df["cloud_cover"] > 0.5).astype(int)
            
        if "direct_radiation" in df.columns and "sunshine_duration" in df.columns:
            # Simple solar index: normalized radiation * sunshine fraction
            df["solar_index"] = (df["direct_radiation"] / 1000.0).clip(upper=1.0) * (df["sunshine_duration"] / 3600.0).clip(upper=1.0)

        return df


class KPIEngine:
    def __init__(self):
        pass

    def compute_load_state(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        series = df[feature].dropna()
        if series.empty:
            return {}
        peak = series.max()
        avg = series.mean()
        p90 = series.quantile(0.9)
        time_above = float((series >= p90).mean() * 100)
        return {
            "average_load": float(avg),
            "peak_load": float(peak),
            "minimum_load": float(series.min()),
            "energy_consumption": float(series.sum()),
            "load_factor": float(avg / peak) if peak else 0.0,
            "peak_duration_points": int((series >= p90).sum()),
            "time_above_90pct_peak": float(time_above)
        }

    def compute_shape_curve(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        series = df[feature].dropna()
        if series.empty:
            return {}
        avg = series.mean()
        p95 = series.quantile(0.95)
        p05 = series.quantile(0.05)
        peak = series.max()
        curve_flatness = float(series.std() / avg) if avg else 0.0
        valley_depth = float((peak - series.min()) / peak) if peak else 0.0
        peak_sharpness = float(peak / avg) if avg else 0.0
        duck_severity = 0.0
        shoulder_strength = 0.0
        curve_symmetry = 0.0
        if "Hour" in df.columns:
            hourly = df.groupby("Hour")[feature].mean()
            if not hourly.empty:
                midday_valley = hourly.loc[hourly.index.isin(range(11, 16))].min() if len(hourly) else np.nan
                evening_peak = hourly.loc[hourly.index.isin(range(18, 23))].max() if len(hourly) else np.nan
                if np.isfinite(midday_valley) and np.isfinite(evening_peak) and avg:
                    duck_severity = float((evening_peak - midday_valley) / avg)
                morning_shoulder = hourly.loc[hourly.index.isin(range(7, 10))].mean()
                evening_shoulder = hourly.loc[hourly.index.isin(range(16, 19))].mean()
                if np.isfinite(morning_shoulder) and np.isfinite(evening_shoulder) and avg:
                    shoulder_strength = float(((morning_shoulder + evening_shoulder) / 2) / avg)
                morning_slope = hourly.loc[hourly.index.isin(range(6, 10))].diff().mean()
                evening_slope = hourly.loc[hourly.index.isin(range(16, 20))].diff().mean()
                if np.isfinite(morning_slope) and np.isfinite(evening_slope):
                    denom = abs(morning_slope) + abs(evening_slope) + 1e-6
                    curve_symmetry = float(1.0 - (abs(morning_slope - evening_slope) / denom))
        return {
            "curve_flatness_index": curve_flatness,
            "peak_sharpness": peak_sharpness,
            "valley_depth": valley_depth,
            "load_spread_p95_p05": float(p95 - p05),
            "shoulder_strength": shoulder_strength,
            "duck_curve_severity": duck_severity,
            "curve_symmetry_score": curve_symmetry
        }

    def compute_ramp(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        series = df[feature].dropna()
        ramps = series.diff()
        if ramps.empty:
            return {}
        return {
            "morning_ramp_rate": float(ramps.median()),
            "evening_ramp_rate": float(ramps.median()),
            "max_ramp_spike": float(ramps.max()),
            "ramp_volatility": float(ramps.std()),
            "negative_ramp_rate": float(ramps.min()),
            "sustained_ramp_duration": int((ramps.abs() > ramps.abs().quantile(0.9)).sum())
        }

    def compute_temporal(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns or "time_block" not in df.columns:
            return {}
        daily_peak_blocks = df.groupby("Date")[feature].idxmax()
        peak_blocks = df.loc[daily_peak_blocks, "time_block"] if len(daily_peak_blocks) else pd.Series(dtype=float)
        return {
            "peak_block_frequency": float(peak_blocks.value_counts().head(1).sum()) if len(peak_blocks) else 0.0,
            "peak_hour_drift": float(peak_blocks.std()) if len(peak_blocks) else 0.0,
            "weekend_peak_shift": float(df[df["IsWeekend"] == 1][feature].mean() - df[df["IsWeekend"] == 0][feature].mean())
        }

    def compute_stability(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        series = df[feature].dropna()
        if series.empty:
            return {}
        relative_vol = float(series.std() / series.mean()) if series.mean() else 0.0
        return {
            "relative_volatility": relative_vol,
            "load_persistence_d1": float(df.get("load_D_same_block", series).corr(series, method="spearman")) if "load_D_same_block" in df else 0.0,
            "load_persistence_d7": float(df.get("load_D_minus_7", series).corr(series, method="spearman")) if "load_D_minus_7" in df else 0.0,
            "stability_bandwidth": float(series.quantile(0.95) - series.quantile(0.05))
        }

    def compute_weather(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        kpis = {}
        
        weather_cols = [
            "temperature", "humidity", "precipitation", 
            "apparent_temperature", "cloud_cover", "cloud_cover_low",
            "sunshine_duration", "direct_radiation", "wind_speed_10m"
        ]
        
        for w_col in weather_cols:
            if w_col in df.columns:
                corr = df[[feature, w_col]].corr(method="spearman").iloc[0, 1]
                kpis[f"{w_col}_sensitivity"] = float(corr)
                
        if "temperature" in df.columns and "heat_index" in df.columns:
            kpis["heat_stress_index"] = float(df[[feature, "heat_index"]].corr(method="spearman").iloc[0, 1])
            
        if "humidity" in df.columns:
            hi = df[df["humidity"] >= df["humidity"].quantile(0.75)][feature].mean()
            lo = df[df["humidity"] <= df["humidity"].quantile(0.25)][feature].mean()
            kpis["humidity_amplification"] = float(hi / lo) if lo and not np.isnan(lo) else 0.0
            
        if "precipitation" in df.columns:
            rain = df[df["precipitation"] > 0.1][feature].mean()
            dry = df[df["precipitation"] <= 0.1][feature].mean()
            kpis["rain_suppression_mw"] = float(dry - rain) if not np.isnan(rain) and not np.isnan(dry) else 0.0
            
        return kpis

    def compute_risk(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        if df.empty or feature not in df.columns:
            return {}
        series = df[feature].dropna()
        if series.empty:
            return {}
        capacity_mw = None
        if "capacity_mw" in df.columns:
            capacity_mw = float(df["capacity_mw"].dropna().iloc[-1]) if not df["capacity_mw"].dropna().empty else None
        return {
            "peak_probability": float((series > series.quantile(0.9)).mean()),
            "extreme_event_frequency": float((series > series.quantile(0.99)).mean()),
            "ramp_stress_score": float(series.diff().abs().quantile(0.95)),
            "tail_risk_load_p99": float(series.quantile(0.99)),
            "capacity_breach_risk": float((series > capacity_mw).mean()) if capacity_mw else 0.0
        }

    def compute_composites(self, kpis: Dict[str, Any]) -> Dict[str, Any]:
        rel_vol = float(kpis.get("relative_volatility", 0.0))
        risk = float(kpis.get("extreme_event_frequency", 0.0))
        temp_sens = float(kpis.get("temp_sensitivity_mw_c", 0.0))
        ramp = float(kpis.get("ramp_stress_score", 0.0))
        mape = float(kpis.get("forecast_mape", 0.0))
        forecast_conf = 0.0 if mape <= 0 else float(max(0.0, 1.0 - (mape / 100.0)))
        return {
            "grid_stress_index": float(min(100.0, (rel_vol * 100.0) + (risk * 100.0))),
            "weather_dependency_index": float(min(100.0, abs(temp_sens) * 100.0)),
            "forecast_confidence_index": float(min(100.0, forecast_conf * 100.0)),
            "dispatch_difficulty_index": float(min(100.0, (rel_vol * 50.0) + (ramp * 2.0)))
        }


class GridIntelControlDesk:
    def __init__(self, data_path: str = ""):
        self.data_layer = DataLayer(DataLayerConfig(data_path=data_path or ""))
        self.fe = FeatureEngineering()
        self.kpis = KPIEngine()
        self.eda = EDAEngine(data_path or "")

    def _wrap(self, charts: Dict[str, Any], kpis: Dict[str, Any], insights: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
        return {"charts": charts, "kpis": kpis, "insights": insights, "metadata": meta}

    def get_summary(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        kpi = self.kpis.compute_load_state(df, feature)
        return self._wrap({}, kpi, [], {"module": "load_summary"})

    def get_shape(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_load_duration_curve(df, feature)
        kpi = self.kpis.compute_shape_curve(df, feature)
        return self._wrap({"load_duration_curve": chart.get("plot")}, kpi, [chart.get("insight", "")], {"module": "shape"})

    def get_calendar(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_calendar_effect_analysis(df, feature)
        return self._wrap({"calendar_effects": chart.get("plot")}, chart.get("kpis", {}), [], {"module": "calendar"})

    def get_decomposition(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_advanced_decomposition(df, feature)
        return self._wrap({"decomposition": chart.get("plot")}, chart.get("kpis", {}), [chart.get("insight", "")], {"module": "decomposition"})

    def get_weather(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_temperature_load_curve(df)
        kpi = self.kpis.compute_weather(df, feature)
        return self._wrap({"temp_load_curve": chart.get("plot")}, {**chart.get("kpis", {}), **kpi}, [chart.get("insight", "")], {"module": "weather"})

    def get_stability(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_risk_monitor(df, feature)
        kpi = self.kpis.compute_stability(df, feature)
        risk = self.kpis.compute_risk(df, feature)
        comp = self.kpis.compute_composites({**kpi, **risk})
        return self._wrap({"risk_monitor": chart.get("plot")}, {**kpi, **risk, **comp}, [chart.get("insight", "")], {"module": "stability"})

    def get_risk_anomalies(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_smart_anomaly_detection(df, feature)
        return self._wrap({"smart_anomalies": chart.get("plot")}, chart.get("kpis", {}), [chart.get("insight", "")], {"module": "risk"})

    def get_forecast(self, start: str, end: str, feature: str) -> Dict[str, Any]:
        try:
            from short_term_pipeline import run_short_term_pipeline
        except ImportError:
            try:
                from .short_term_pipeline import run_short_term_pipeline
            except ImportError:
                # Fallback if relative import fails and not in path
                from backend.short_term_pipeline import run_short_term_pipeline

        # Load ample history for the pipeline; if a training range is set (via /api/switch-region),
        # ensure the DF covers it plus the same-window last year.
        train_range = getattr(self, "train_range", None)
        req_start = pd.to_datetime(start)
        req_end = pd.to_datetime(end)
        if isinstance(train_range, dict) and train_range.get("from") and train_range.get("to"):
            tr_from = pd.to_datetime(train_range["from"])
            tr_to = pd.to_datetime(train_range["to"])
            needed_start = min(req_start - pd.Timedelta(days=60), tr_from - pd.Timedelta(days=430))
            needed_end = max(req_end, tr_to)
            df = self.fe.add_features(self.data_layer.get_data(needed_start.strftime("%Y-%m-%d"), needed_end.strftime("%Y-%m-%d")))
        else:
            freq_extended_start = req_start - pd.Timedelta(days=60)
            df = self.fe.add_features(self.data_layer.get_data(freq_extended_start.strftime("%Y-%m-%d"), end))
        
        # Determine dates to forecast
        date_range = pd.date_range(req_start, req_end, freq='D')
        
        all_forecasts = []
        insights = []
        kpis = {}
        
        first_run_done = False
        
        for dt in date_range:
            d_str = dt.strftime("%Y-%m-%d")
            # Logic to determine actual_blocks (assuming forecast is for future, so 0 actuals know)
            # In a real scenario, this might depend on 'today'. For now, we assume 0 for full forecast.
            # If date is in the past/today, we might have actuals.
            
            # Check if we have actual data for this day to set actual_blocks
            # Run pipeline
            try:
                # If target date is today/future, actual_blocks might be 0 or partial. 
                # For this "Forecast" endpoint, we typically want the Model's view, so we force actual_blocks=0 
                # to get a full forecast curve, OR we use actuals where available.
                # Let's align with the previous behavior: Forecast for the future.
                # If start date > max data date, actual_blocks = 0.
                
                if isinstance(train_range, dict) and train_range.get("from") and train_range.get("to"):
                    pipeline_res = run_short_term_pipeline(
                        df,
                        d_str,
                        actual_blocks=0,
                        config={
                            "human_behaviour_weight": 0.0,
                            "weather_gating": {"similar_day": 0.45, "ml_model": 0.35, "elasticity": 0.20},
                            "trend_weight": 0.1,
                            "pattern_weight": 0.3,
                            "weather_tune": True,
                            "weather_tune_iters": 30,
                            "auto_similarity_from_data": True,
                            "correction_smoothing": True,
                            "training_range": train_range,
                        },
                    )
                else:
                    pipeline_res = run_short_term_pipeline(
                        df,
                        d_str,
                        actual_blocks=0,
                        config={
                            "human_behaviour_weight": 0.0,
                            "weather_gating": {"similar_day": 0.45, "ml_model": 0.35, "elasticity": 0.20},
                            "trend_weight": 0.1,
                            "pattern_weight": 0.3,
                            "weather_tune": True,
                            "weather_tune_iters": 30,
                            "auto_similarity_from_data": True,
                            "correction_smoothing": True,
                        },
                    )
                
                # Extract forecast curve
                f_df = pd.DataFrame(pipeline_res['forecast_df'])
                f_df['Timestamp'] = f_df.apply(lambda row: pd.to_datetime(row['date']) + pd.Timedelta(minutes=(row['time_block']-1)*15), axis=1)
                all_forecasts.append(f_df)
                
                # Extract insights/KPIs from the first day (usually the target day)
                if not first_run_done:
                    kpis["forecast_confidence"] = pipeline_res.get("metadata", {}).get("forecast_confidence", 0.85) * 100
                    kpis["weather_sensitivity"] = pipeline_res.get("peak_impact", {}).get("forecast_peak", {}).get("weather_effect_mw", 0)
                    if "driver_contributions" in pipeline_res:
                        # Add top drivers to insights
                        drivers = sorted(pipeline_res["driver_contributions"], key=lambda x: abs(x["mw"]), reverse=True)[:2]
                        driver_text = ", ".join([f"{d['factor']} ({d['mw']} MW)" for d in drivers])
                        insights.append(f"Primary Drivers: {driver_text}")
                    first_run_done = True
                    
                    # Store decision signals or other metadata if needed
                    kpis["driver_contributions"] = pipeline_res.get("driver_contributions")

            except Exception as e:
                print(f"Pipeline failed for {d_str}: {e}")
                continue

        if not all_forecasts:
            # Fallback to old method if pipeline fails completely
            return self._fallback_forecast(df, feature, start)

        full_forecast_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Prepare Chart Data
        fig = self.eda._fig_to_json(self._create_forecast_plot(df, full_forecast_df, feature))
        
        # Calculate residuals if actuals exist
        # ... (simplified residual calc)
        
        return self._wrap(
            {"forecast": fig},
            kpis,
            insights,
            {"module": "forecast_advanced", "driver_contributions": kpis.get("driver_contributions")}
        )

    def _create_forecast_plot(self, hist_df, forecast_df, feature):
        import plotly.graph_objects as go
        from engine import FeatureRegistry
        
        fig = go.Figure()
        
        # Plot some history
        last_hist_date = pd.to_datetime(forecast_df['Timestamp'].min())
        hist_view = hist_df[(hist_df['Datetime'] < last_hist_date) & 
                           (hist_df['Datetime'] > (last_hist_date - pd.Timedelta(days=3)))]
        
        fig.add_trace(go.Scatter(x=hist_view['Datetime'], y=hist_view[feature], 
                                name='Historical', line=dict(color='gray', width=1)))
        
        # Plot Forecast
        fig.add_trace(go.Scatter(x=forecast_df['Timestamp'], y=forecast_df['forecast'], 
                                name='Forecast', line=dict(color=FeatureRegistry.COLORS['highlight'], width=2)))
        
        fig.update_layout(title="Short Term Forecast (Advanced)", template="plotly_dark", 
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    def _fallback_forecast(self, df, feature, start):
        # Original simple logic
        return self.eda.run_multi_horizon_forecast(df, feature, window=7)

    def get_scenario(self, start: str, end: str, feature: str, params: Dict[str, Any]) -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_scenario_deviation_simulator(
            df,
            temp_delta=float(params.get("temp_delta", 0.0)),
            hum_delta=float(params.get("hum_delta", 0.0)),
            is_raining=bool(params.get("is_raining", False)),
            is_weekend=bool(params.get("is_weekend", False)),
            is_holiday=bool(params.get("is_holiday", False)),
            is_pre_weekend=bool(params.get("is_pre_weekend", False)),
            granularity=params.get("sim_granularity", "block"),
            adjustments=params.get("sim_adjustments", None),
        )
        return self._wrap(
            {"scenario_curve": chart.get("plot")},
            chart.get("kpis", {}),
            [chart.get("insight", "")],
            {"module": "scenario", "simulator_data": chart}
        )

    def get_heatmap(self, start: str, end: str, feature: str, mode: str = "scenario") -> Dict[str, Any]:
        df = self.fe.add_features(self.data_layer.get_data(start, end))
        chart = self.eda.run_scenario_heatmap(df, feature, mode=mode)
        return self._wrap({"scenario_heatmap": chart.get("plot")}, chart.get("kpis", {}), [chart.get("summary", "")], {"module": "heatmap"})
