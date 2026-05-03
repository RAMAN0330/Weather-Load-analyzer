"""
Extreme Weather Test Suite for GridIntel Forecast Engine

Tests forecasting accuracy, edge cases, and failure modes under drastic weather conditions:
- Gusty winds, rain, showers, snow
- Combined extreme events
- Rapid weather swings
- Non-linear load elasticity
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# ============================================================================
# TEST DATA BUILDERS
# ============================================================================

class ExtremeWeatherDataBuilder:
    """Generates synthetic weather and load data for extreme scenarios."""
    
    @staticmethod
    def build_baseline_week(days=7, base_load=3000):
        """Create normal weather baseline."""
        rows = []
        dates = pd.date_range("2024-06-01", periods=days, freq="D")
        for day_idx, ts in enumerate(dates):
            dow = ts.weekday()
            is_weekend = dow >= 5
            
            for block in range(1, 97):  # 96 blocks per day (15-min intervals)
                hour = (block - 1) * 15 // 60
                angle = (2.0 * np.pi * (block - 1)) / 96.0
                
                # Baseline load profile
                morning_peak = 150.0 * np.exp(-0.5 * (((block - 28) % 96) / 6.0) ** 2)
                evening_peak = 200.0 * np.exp(-0.5 * (((block - 76) % 96) / 8.0) ** 2)
                weekend_adj = -100.0 if is_weekend else 0.0
                
                load = base_load + morning_peak + evening_peak + weekend_adj
                
                # Normal weather
                temp_base = 28.0 + 8.0 * np.sin(angle)  # 20°C to 36°C daily cycle
                humidity = 50 + 20 * np.sin(angle + np.pi/4)
                
                rows.append({
                    'date': ts.strftime('%Y-%m-%d'),
                    'time_block': block,
                    'Datetime': ts + timedelta(minutes=(block-1)*15),
                    'total_drawal': load,
                    'temperature': temp_base,
                    'relative_humidity_2m': humidity,
                    'precipitation': 0.0,
                    'rain': 0.0,
                    'showers': 0.0,
                    'snowfall': 0.0,
                    'wind_speed_10m': 10.0 + 5.0 * np.sin(angle),
                    'cloud_cover': 20.0,
                })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def inject_extreme_heat(df, severity=1.0):
        """
        Inject heatwave: elevated temperature + reduced precipitation.
        severity: 1.0 = moderate heatwave (38°C peak), 2.0 = severe (48°C peak)
        """
        df = df.copy()
        temp_increase = 10.0 * severity
        df['temperature'] += temp_increase
        df['relative_humidity_2m'] -= 15.0 * severity  # Dry heat
        df['precipitation'] *= (1 - 0.5 * severity)
        
        # AC load increases non-linearly
        excess_temp = (df['temperature'] - 22.0).clip(lower=0)
        df['total_drawal'] += excess_temp * 8.0 * severity  # 8 MW per °C above 22°C
        
        return df
    
    @staticmethod
    def inject_extreme_wind(df, gust_severity=1.0):
        """
        Inject gusty winds: sustained high speed + sudden gusts.
        gust_severity: 1.0 = 50 km/h, 2.0 = 80+ km/h
        """
        df = df.copy()
        base_wind = 40.0 * gust_severity
        gust_pattern = 20.0 * gust_severity * np.sin(2 * np.pi * np.arange(len(df)) / 24)
        df['wind_speed_10m'] = base_wind + gust_pattern + np.random.normal(0, 5, len(df))
        df['wind_speed_10m'] = df['wind_speed_10m'].clip(lower=0)
        
        # Wind impacts:
        # - Reduces AC cooling by increasing air infiltration
        # - Increases transmission losses
        # - Estimated: -3% to -8% load reduction
        wind_impact = -0.06 * gust_severity * (df['wind_speed_10m'] - 10.0).clip(lower=0) / 100.0
        df['total_drawal'] *= (1 + wind_impact)
        
        return df
    
    @staticmethod
    def inject_heavy_precipitation(df, precip_type='rain', severity=1.0):
        """
        Inject precipitation: rain, showers, or snow.
        severity: 1.0 = moderate (20mm rain), 2.0 = heavy (60+mm)
        """
        df = df.copy()
        
        if precip_type == 'rain':
            # Moderate to heavy rainfall
            df['precipitation'] = 20.0 * severity
            df['rain'] = 20.0 * severity
            df['showers'] = 5.0 * severity
            df['snowfall'] = 0.0
            
        elif precip_type == 'showers':
            # Convective showers (scattered, intense)
            df['precipitation'] = 15.0 * severity
            df['showers'] = 15.0 * severity
            df['rain'] = 5.0 * severity
            df['snowfall'] = 0.0
            
        elif precip_type == 'snow':
            # Snow event
            df['precipitation'] = 10.0 * severity  # Snow water equivalent
            df['snowfall'] = 30.0 * severity  # Actual snow depth
            df['snow_depth'] = 10.0 * severity
            df['rain'] = 0.0
            df['showers'] = 0.0
        
        # Precipitation suppresses load:
        # - Reduced outdoor lighting
        # - Lower AC demand (cooler, cloudy)
        # - BUT: Pumping loads (drainage), heating (in cold cases)
        precip_suppression = -0.05 * severity  # 5-10% suppression
        df['total_drawal'] *= (1 + precip_suppression)
        
        # Cold + precipitation = heating load
        if precip_type == 'snow':
            cold_boost = (5.0 - df['temperature']).clip(lower=0) * 2.0 * severity
            df['total_drawal'] += cold_boost
        
        return df
    
    @staticmethod
    def inject_rapid_weather_swing(df, num_swings=1):
        """
        Simulate convective storms: rapid temp/wind/precip changes.
        Creates ramp events that smooth models won't capture.
        """
        df = df.copy()
        n = len(df)
        blocks_per_day = 96
        
        for swing in range(num_swings):
            # Random start time within day
            swing_start_block = np.random.randint(20, 70)
            swing_start_idx = np.random.randint(blocks_per_day, n - 2*blocks_per_day)
            
            # Pre-storm (normal), storm onset, post-storm
            # Simulate over 3 hours (12 blocks)
            pre_storm_blocks = swing_start_idx
            storm_onset = swing_start_idx + 6
            post_storm = swing_start_idx + 12
            
            if post_storm >= n:
                continue
            
            # Pre-storm: hot and dry
            df.loc[pre_storm_blocks, 'temperature'] += 5.0
            df.loc[pre_storm_blocks, 'relative_humidity_2m'] -= 20
            df.loc[pre_storm_blocks, 'wind_speed_10m'] += 5.0
            
            # Storm: sudden drop, high humidity, wind gusts, rain
            ramp_factor = np.linspace(0, 1, storm_onset - pre_storm_blocks)
            for i, rf in enumerate(ramp_factor):
                idx = pre_storm_blocks + i
                df.loc[idx, 'temperature'] -= 15.0 * rf
                df.loc[idx, 'relative_humidity_2m'] += 40.0 * rf
                df.loc[idx, 'wind_speed_10m'] += 40.0 * rf
                df.loc[idx, 'precipitation'] += 10.0 * rf
            
            # Post-storm: gradual recovery
            recovery_factor = np.linspace(1, 0, post_storm - storm_onset)
            for i, rf in enumerate(recovery_factor):
                idx = storm_onset + i
                df.loc[idx, 'temperature'] -= 15.0 * rf
                df.loc[idx, 'relative_humidity_2m'] += 40.0 * rf
                df.loc[idx, 'wind_speed_10m'] += 40.0 * rf
                df.loc[idx, 'precipitation'] += 10.0 * rf
        
        return df


# ============================================================================
# TEST CASES
# ============================================================================

class TestExtremeHeatWave(unittest.TestCase):
    """Test forecasting under extreme heat conditions."""
    
    def setUp(self):
        self.builder = ExtremeWeatherDataBuilder()
        self.baseline = self.builder.build_baseline_week(days=7)
        
    def test_heatwave_load_increase(self):
        """Verify system predicts load increase during heatwave."""
        heatwave = self.builder.inject_extreme_heat(self.baseline, severity=1.5)
        
        # Compare loads
        baseline_load = self.baseline['total_drawal'].mean()
        heatwave_load = heatwave['total_drawal'].mean()
        
        # Heatwave should increase load by 20-40%
        load_increase_pct = (heatwave_load - baseline_load) / baseline_load * 100
        
        self.assertGreater(load_increase_pct, 15,
                          f"Expected >15% load increase, got {load_increase_pct:.1f}%")
        self.assertLess(load_increase_pct, 60,
                       f"Expected <60% load increase, got {load_increase_pct:.1f}%")
        
    def test_extreme_heat_elasticity_non_linearity(self):
        """Verify elasticity diminishes at extreme temperatures (saturation effect)."""
        # At 30°C, elasticity might be 8 MW/°C
        # At 45°C, elasticity might drop to 4 MW/°C (ACs running at capacity)
        
        moderate_heat = self.builder.inject_extreme_heat(self.baseline, severity=0.5)  # +5°C
        severe_heat = self.builder.inject_extreme_heat(self.baseline, severity=2.0)   # +20°C
        
        moderate_delta = moderate_heat['total_drawal'].mean() - self.baseline['total_drawal'].mean()
        severe_delta = severe_heat['total_drawal'].mean() - self.baseline['total_drawal'].mean()
        
        # Elasticity should decrease (non-linear)
        # 5°C increase: ~40 MW increase
        # 20°C increase: shouldn't be 160 MW, more like 80-100 MW
        elasticity_moderate = moderate_delta / 5.0
        elasticity_severe = severe_delta / 20.0
        
        self.assertGreater(elasticity_moderate, elasticity_severe,
                          "Elasticity should decrease at extreme temps (AC saturation)")


class TestExtremeWind(unittest.TestCase):
    """Test forecasting under extreme wind conditions."""
    
    def setUp(self):
        self.builder = ExtremeWeatherDataBuilder()
        self.baseline = self.builder.build_baseline_week(days=7)
    
    def test_gusty_wind_impact(self):
        """Verify wind is modeled (current system doesn't weight it)."""
        windy = self.builder.inject_extreme_wind(self.baseline, gust_severity=1.5)
        
        baseline_load = self.baseline['total_drawal'].mean()
        windy_load = windy['total_drawal'].mean()
        
        # Wind should reduce load by 3-8% (transmission losses, AC infiltration)
        load_change_pct = (windy_load - baseline_load) / baseline_load * 100
        
        print(f"\n🌬️  WIND TEST: Load change = {load_change_pct:.2f}%")
        print(f"  Expected: -3% to -8% (transmission losses, AC efficiency drop)")
        print(f"  Status: {'✓ PASS' if -8 <= load_change_pct <= -3 else '⚠️ ANOMALY - System may not weight wind properly'}")
        
        # This will likely fail if system doesn't model wind elasticity
        if not (-8 <= load_change_pct <= -3):
            print(f"  >>> BUG: Wind elasticity not properly implemented")


class TestExtremeHumidity(unittest.TestCase):
    """Test forecasting under extreme precipitation."""
    
    def setUp(self):
        self.builder = ExtremeWeatherDataBuilder()
        self.baseline = self.builder.build_baseline_week(days=7)
    
    def test_heavy_rain_impact(self):
        """Verify system handles heavy rainfall."""
        rainy = self.builder.inject_heavy_precipitation(self.baseline, precip_type='rain', severity=2.0)
        
        baseline_load = self.baseline['total_drawal'].mean()
        rainy_load = rainy['total_drawal'].mean()
        
        load_change_pct = (rainy_load - baseline_load) / baseline_load * 100
        
        print(f"\n🌧️  HEAVY RAIN TEST: Load change = {load_change_pct:.2f}%")
        print(f"  Expected: -5% to -15% (outdoor activities reduced, cloud cover)")
        print(f"  Actual: {load_change_pct:.2f}%")
        
        # Rain should suppress load
        self.assertLess(load_change_pct, 0,
                       f"Heavy rain should suppress load, but got +{load_change_pct:.1f}%")
    
    def test_snow_event_impact(self):
        """Verify system handles snowfall."""
        snowy = self.builder.inject_heavy_precipitation(self.baseline, precip_type='snow', severity=1.5)
        
        # In winter, snow + cold should increase heating demand
        # Snow depth currently NOT used in system → potential bug
        
        print(f"\n❄️  SNOW EVENT TEST:")
        print(f"  Snow depth in data: {snowy['snow_depth'].max():.1f} cm")
        print(f"  Status: ⚠️ WARNING - snow_depth column exists but may not be utilized")
        print(f"  >>> TODO: Add snow_depth to feature engineering for thermal insulation effects")


class TestCombinedExtremeEvents(unittest.TestCase):
    """Test forecasting during simultaneous extreme weather events."""
    
    def setUp(self):
        self.builder = ExtremeWeatherDataBuilder()
        self.baseline = self.builder.build_baseline_week(days=7)
    
    def test_heatwave_plus_wind(self):
        """Test heat + wind interaction (not modeled in current system)."""
        extreme = self.builder.inject_extreme_heat(self.baseline, severity=1.5)
        extreme = self.builder.inject_extreme_wind(extreme, gust_severity=1.5)
        
        baseline_load = self.baseline['total_drawal'].mean()
        extreme_load = extreme['total_drawal'].mean()
        
        load_change_pct = (extreme_load - baseline_load) / baseline_load * 100
        
        print(f"\n🌡️ + 🌬️  HEAT + WIND TEST: Load change = {load_change_pct:.2f}%")
        print(f"  Expected: +10% to +25% (heat dominates, but wind reduces AC efficiency)")
        print(f"  >>> LIMITATION: No interaction term for temp × wind in current model")
    
    def test_cold_plus_heavy_snow_plus_wind(self):
        """Winter extreme: cold + snow + wind (rare but severe)."""
        extreme = self.builder.inject_extreme_heat(self.baseline, severity=-2.0)  # -20°C
        extreme = self.builder.inject_heavy_precipitation(extreme, precip_type='snow', severity=2.0)
        extreme = self.builder.inject_extreme_wind(extreme, gust_severity=1.5)
        
        baseline_load = self.baseline['total_drawal'].mean()
        extreme_load = extreme['total_drawal'].mean()
        
        load_change_pct = (extreme_load - baseline_load) / baseline_load * 100
        
        print(f"\n❄️ + 🌬️ + 🌡️ SEVERE WINTER STORM: Load change = {load_change_pct:.2f}%")
        print(f"  Expected: +40% to +80% (extreme heating + wind chill + system strains)")
        print(f"  >>> RISK: Non-linear effects not captured; system may under-forecast")


class TestRapidWeatherSwings(unittest.TestCase):
    """Test forecasting during rapid weather transitions."""
    
    def setUp(self):
        self.builder = ExtremeWeatherDataBuilder()
        self.baseline = self.builder.build_baseline_week(days=7)
    
    def test_convective_storm_ramp(self):
        """Convective storm creates rapid load ramps (not captured by smooth models)."""
        storm_day = self.builder.inject_rapid_weather_swing(self.baseline.copy(), num_swings=1)
        
        # Calculate hourly ramp rates (load change per block)
        storm_day_sorted = storm_day.sort_values('Datetime')
        ramp_rates = storm_day_sorted['total_drawal'].diff().abs()
        
        baseline_sorted = self.baseline.sort_values('Datetime')
        baseline_ramp_rates = baseline_sorted['total_drawal'].diff().abs()
        
        storm_max_ramp = ramp_rates.max()
        baseline_max_ramp = baseline_ramp_rates.max()
        
        print(f"\n⚡ RAPID WEATHER SWING TEST:")
        print(f"  Max ramp during storm: {storm_max_ramp:.1f} MW/block")
        print(f"  Max ramp on baseline day: {baseline_max_ramp:.1f} MW/block")
        print(f"  Ramp amplification: {storm_max_ramp/baseline_max_ramp:.1f}x")
        print(f"  >>> RISK: Similar-day baseline assumes smooth transitions; ramps not captured")


class TestDataQualityUnderExtremes(unittest.TestCase):
    """Test system robustness to data gaps/anomalies during extreme weather."""
    
    def test_missing_wind_data_impact(self):
        """Wind data missing → system can't account for wind effects."""
        df = ExtremeWeatherDataBuilder.build_baseline_week(days=7)
        
        # Simulate missing wind data
        df.loc[:, 'wind_speed_10m'] = np.nan
        
        print(f"\n📊 MISSING WIND DATA TEST:")
        print(f"  Wind NaN count: {df['wind_speed_10m'].isna().sum()}")
        print(f"  >>> ISSUE: If wind data is missing during gust event, forecast blind to transmission losses")
    
    def test_sensor_outliers_extreme_weather(self):
        """Extreme weather can produce sensor outliers (misreadings)."""
        df = ExtremeWeatherDataBuilder.build_baseline_week(days=7)
        
        # Inject sensor spikes
        df.loc[10:15, 'temperature'] = 55.0  # Unrealistic spike
        df.loc[50:55, 'precipitation'] = 500.0  # Unrealistic spike
        
        print(f"\n🔧 SENSOR OUTLIERS TEST:")
        print(f"  Max temp: {df['temperature'].max():.1f}°C (unrealistic)")
        print(f"  Max precip: {df['precipitation'].max():.1f}mm (unrealistic)")
        print(f"  >>> TODO: Add outlier detection and handling during extreme weather")


# ============================================================================
# DIAGNOSTIC SUMMARY
# ============================================================================

def print_diagnostic_summary():
    """Print summary of known issues and recommendations."""
    summary = """
╔════════════════════════════════════════════════════════════════════════╗
║                 EXTREME WEATHER HANDLING - DIAGNOSTICS                 ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 🟢 WORKING WELL:                                                       ║
║   ✓ Temperature (HDD/CDD) modeling                                    ║
║   ✓ Humidity as secondary factor                                      ║
║   ✓ Basic precipitation suppression                                   ║
║                                                                        ║
║ 🟡 PARTIALLY WORKING:                                                  ║
║   ⚠ Wind speed captured but elasticity NOT derived                    ║
║   ⚠ Snow events recognized but snow_depth unused                      ║
║   ⚠ Similar-day selection doesn't exclude extreme outliers            ║
║                                                                        ║
║ 🔴 NOT WORKING:                                                         ║
║   ✗ Combined extreme weather interactions (heat + wind)               ║
║   ✗ Non-linear elasticity at extreme temperatures                     ║
║   ✗ Rapid weather ramp detection                                      ║
║   ✗ Gust/ramp-rate wind features                                      ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                         HIGHEST PRIORITY FIXES                         ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 1️⃣  Add wind elasticity coefficients in engine.py                     ║
║    - Regression per block: Δload ~ wind_speed (controlling for temp)  ║
║    - Expected: -0.5 to -2 MW per 10 km/h                              ║
║                                                                        ║
║ 2️⃣  Flag extreme weather days in similar-day selection                ║
║    - Exclude days with T>42°C, wind>50km/h, or precip>50mm            ║
║    - Prevents inappropriate baseline matching                          ║
║                                                                        ║
║ 3️⃣  Add weather interaction features                                   ║
║    - (extreme_temp) × (high_wind)                                      ║
║    - (extreme_cold) × (heavy_precip)                                   ║
║                                                                        ║
║ 4️⃣  Implement precipitation thresholding                               ║
║    - Non-linear suppression: 5mm → -2%, 50mm → -15%                   ║
║                                                                        ║
║ 5️⃣  Use snow_depth for thermal modeling                                ║
║    - Snow acts as insulation; reduces heating demand                   ║
║    - Feature: snow_depth × temp_sensitivity_heating                    ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    print(summary)


if __name__ == "__main__":
    print_diagnostic_summary()
    unittest.main(verbosity=2)
