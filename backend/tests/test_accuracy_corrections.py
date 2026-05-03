import unittest

import numpy as np
import pandas as pd


class TestAccuracyCorrections(unittest.TestCase):
    def test_bias_table_sign_and_actual_block_preservation(self):
        from backend.accuracy.bias_correction import apply_bias_correction, compute_bias_table

        rows = []
        for day in pd.date_range("2024-01-01", periods=10, freq="D"):
            for block in range(1, 97):
                rows.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "time_block": block,
                        "forecast_t1": 1000.0,
                        "actual": 1100.0,
                    }
                )
        df = pd.DataFrame(rows)
        table, meta = compute_bias_table(df, window_days=10, day_type="all", horizon="t1")
        self.assertTrue(meta["applied"])
        self.assertEqual(len(table), 96)
        self.assertLess(float(np.mean(table)), 0.0)

        forecast = np.full(96, 1000.0)
        result = apply_bias_correction(forecast, table, actual_blocks=10)
        np.testing.assert_allclose(result.forecast[:10], forecast[:10])
        self.assertGreater(float(result.forecast[10]), 1000.0)
        self.assertTrue(result.metadata["applied"])

    def test_t2_night_correction_tapers_and_stops(self):
        from backend.accuracy.t2_bias_correction import apply_t2_night_correction, build_t2_bias_correction

        errors = {b: [-300.0] * 8 for b in range(1, 97)}
        correction, meta = build_t2_bias_correction(errors, fallback_night_mw=0.0, min_samples=7)
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(float(correction[0]), 300.0, places=6)
        self.assertGreater(float(correction[40]), 0.0)
        self.assertAlmostEqual(float(correction[48]), 0.0, places=6)

        forecast = np.full(96, 1000.0)
        result = apply_t2_night_correction(forecast, correction)
        self.assertAlmostEqual(float(result.forecast[0]), 1300.0, places=6)
        self.assertAlmostEqual(float(result.forecast[48]), 1000.0, places=6)

    def test_afternoon_ramp_only_boosts_hot_haryana_afternoon(self):
        from backend.accuracy.ramp_weather import build_afternoon_ramp_adjustment

        rows = []
        for block in range(1, 97):
            hour = ((block - 1) * 15) // 60
            rows.append(
                {
                    "time_block": block,
                    "temperature": 43.0 if 13 <= hour <= 17 else 34.0,
                    "humidity": 25.0,
                    "apparent_temperature": 44.0 if 13 <= hour <= 17 else 35.0,
                    "wind_speed_10m": 10.0,
                    "rain": 0.0,
                }
            )
        target = pd.DataFrame(rows)
        result = build_afternoon_ramp_adjustment(np.full(96, 5000.0), target, season="summer", region="haryana")
        self.assertTrue(result.metadata["applied"])
        self.assertGreater(float(np.max(result.adjustment_mw[52:68])), 0.0)
        self.assertAlmostEqual(float(result.adjustment_mw[0]), 0.0, places=6)

    def test_sldc_validator_flags_and_enforces_bounds(self):
        from backend.accuracy.sldc_validator import validate_sldc_schedule

        forecast = np.full(96, 3000.0)
        forecast[10] = 15000.0
        baseline = np.full(96, 5000.0)
        result = validate_sldc_schedule(forecast, baseline, season="summer")
        self.assertTrue(result.flags)
        self.assertGreaterEqual(float(result.forecast[0]), 3500.0)
        self.assertLessEqual(float(result.forecast[10]), 14000.0)
        self.assertEqual(result.metadata["status"], "flagged")

    def test_forecast_ramp_limit_clips_future_jumps_only(self):
        from backend.short_term_pipeline import _apply_ramp_limit

        forecast = np.full(96, 5000.0)
        forecast[9] = 5100.0
        forecast[10] = 5900.0
        forecast[11] = 4500.0

        limited, applied, meta = _apply_ramp_limit(forecast, actual_blocks=10, max_ramp_mw=300.0)

        self.assertAlmostEqual(float(limited[9]), 5100.0, places=6)
        self.assertAlmostEqual(float(limited[10]), 5400.0, places=6)
        self.assertAlmostEqual(float(limited[11]), 5100.0, places=6)
        self.assertAlmostEqual(float(applied[9]), 0.0, places=6)
        self.assertTrue(meta["applied"])


if __name__ == "__main__":
    unittest.main()
