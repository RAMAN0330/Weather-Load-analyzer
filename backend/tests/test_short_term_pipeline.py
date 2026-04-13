import unittest
from unittest.mock import patch
import types

import numpy as np
import pandas as pd


def _build_history(days: int = 12) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    for day_idx, ts in enumerate(dates):
        for block in range(1, 97):
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "time_block": block,
                    "total_drawal": 1000.0 + (day_idx * 10.0) + (block * 1.5),
                    "temperature": 20.0 + ((block % 24) * 0.3),
                    "humidity": 55.0 + (block % 8),
                    "precipitation": 0.0 if block % 16 else 1.0,
                    "apparent_temperature": 21.0 + ((block % 24) * 0.25),
                    "cloud_cover": 35.0 + (block % 12),
                    "sunshine_duration": 10.0 + (block % 6),
                    "direct_radiation": 120.0 + (block * 2.0),
                    "wind_speed_10m": 3.0 + ((block % 5) * 0.2),
                }
            )
    return pd.DataFrame(rows)


def _build_behaviour_history(days: int = 35) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    for day_idx, ts in enumerate(dates):
        dow = ts.weekday()
        weekend = dow >= 5
        for block in range(1, 97):
            angle = (2.0 * np.pi * (block - 1)) / 96.0
            cyclic_shape = (90.0 * np.sin(angle)) + (35.0 * np.cos(2.0 * angle))
            weekend_effect = cyclic_shape if weekend else 0.0
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "time_block": block,
                    "total_drawal": 2500.0 + (day_idx * 2.0) + (block * 0.25) + weekend_effect,
                    "temperature": 18.0 + (4.0 * np.sin(angle)),
                    "humidity": 58.0 + (6.0 * np.cos(angle)),
                    "precipitation": 0.0,
                    "apparent_temperature": 19.0 + (3.0 * np.sin(angle)),
                    "cloud_cover": 42.0 + (4.0 * np.cos(angle)),
                    "sunshine_duration": 9.0 + max(0.0, 2.5 * np.sin(angle)),
                    "direct_radiation": 140.0 + max(0.0, 80.0 * np.sin(angle)),
                    "wind_speed_10m": 3.5 + (0.6 * np.cos(angle)),
                }
            )
    return pd.DataFrame(rows)


class TestShortTermFeatureEngineering(unittest.TestCase):
    def test_prepare_training_adds_required_96_block_features(self):
        from backend.short_term_pipeline import _prepare_training

        prepared = _prepare_training(_build_history())
        required = {
            "lag_1",
            "lag_7",
            "lag_block_1",
            "lag_block_4",
            "rolling_4",
            "rolling_12",
            "block_sin",
            "block_cos",
        }

        self.assertTrue(required.issubset(set(prepared.columns)))
        latest_day = prepared[prepared["date"] == "2024-01-12"].sort_values("time_block").reset_index(drop=True)
        self.assertFalse(latest_day[list(required)].isna().any().any())

    def test_prepare_inference_frame_uses_anchor_for_intraday_features(self):
        from backend.short_term_pipeline import _prepare_inference_frame

        history = _build_history(days=11)
        target = _build_history(days=12)
        target = target[target["date"] == "2024-01-12"].copy()
        anchor = np.linspace(1200.0, 1295.0, 96)

        inference = _prepare_inference_frame(
            history_df=history,
            target_df=target,
            load_anchor=anchor,
        ).sort_values("time_block")

        self.assertEqual(len(inference), 96)
        self.assertAlmostEqual(float(inference.loc[inference["time_block"] == 2, "lag_block_1"].iloc[0]), float(anchor[0]), places=6)
        self.assertAlmostEqual(float(inference.loc[inference["time_block"] == 5, "lag_block_4"].iloc[0]), float(anchor[0]), places=6)
        self.assertAlmostEqual(float(inference.loc[inference["time_block"] == 6, "rolling_4"].iloc[0]), float(np.mean(anchor[1:5])), places=6)
        self.assertFalse(inference[["lag_block_1", "lag_block_4", "rolling_4", "rolling_12"]].isna().any().any())

    def test_run_short_term_pipeline_supports_pure_dayahead_mode(self):
        from backend.short_term_pipeline import run_short_term_pipeline

        df = _build_history(days=14)
        result = run_short_term_pipeline(
            df,
            target_date="2024-01-14",
            actual_blocks=0,
            config={"weather_tune": False, "region": "punjab"},
        )

        self.assertEqual(len(result["series"]["forecast"]), 96)
        self.assertEqual(result["metadata"]["model_scope"], "global")
        self.assertIn("forecast_shrinkage", result["metadata"])
        self.assertEqual(len(result["series"]["forecast_raw_before_shrinkage"]), 96)
        self.assertIn("hybrid_ai_engine", result["metadata"])

    def test_behaviour_calibration_learns_cyclic_profile_from_data(self):
        import backend.short_term_pipeline as stp

        df = _build_behaviour_history(days=42)
        profiles = stp._calibrate_behaviour_profiles(
            df=df,
            region="punjab",
            cfg={
                "behaviour_calibration_days": 42,
                "behaviour_cyclic_harmonics": 4,
                "behaviour_cyclic_blend": 0.65,
                "behaviour_cyclic_smoothing": 5,
            },
        )

        key = ("north", "winter", "Weekend")
        self.assertIn(key, profiles)
        learned = np.asarray(profiles[key], dtype=float)
        self.assertEqual(learned.size, 96)
        self.assertGreater(float(np.std(learned)), 10.0)
        self.assertAlmostEqual(float(learned[0]), float(learned[-1]), delta=25.0)

        stp._CALIBRATED_BEHAVIOUR = profiles
        adj, label = stp._human_behaviour_adjustment("winter", "punjab", "Weekend", weight=1.0)
        self.assertIn("CalibratedCyclic", label)
        self.assertEqual(len(adj), 96)


class TestDayAheadMetadata(unittest.TestCase):
    def test_v2_dayahead_merges_api_and_pipeline_metadata(self):
        from backend.main import v2_dayahead

        df = _build_history(days=10)
        fake_result = {
            "metadata": {"model_scope": "global", "feature_count": 12},
            "series": {
                "blocks": list(range(1, 97)),
                "baseline": [1000.0] * 96,
                "forecast": [1010.0] * 96,
                "actual": [1005.0] * 96,
                "weather_engine_mode": "elastic_net_weather_lag_dod_momentum",
            },
            "driver_contributions": [],
            "decision_signals": [],
            "forecast_uncertainty": [],
            "slot_sensitivity_profile": [],
            "similar_days": [],
        }

        with patch("backend.main._get_df", return_value=df), patch(
            "backend.main._resolve_date", return_value=("2024-01-10", True, sorted(df["date"].unique().tolist()))
        ), patch(
            "backend.main._find_best_baseline_window",
            return_value={"best_baseline_window": 7, "best_mape": 2.5, "window_mapes": [{"window_days": 7, "baseline_mape": 2.5}]},
        ), patch(
            "backend.main._compute_dayahead", return_value=fake_result
        ), patch(
            "backend.main._baseline_window", return_value=df[df["date"] < "2024-01-10"]
        ), patch(
            "backend.main._compute_kpis_full", return_value={}
        ), patch(
            "backend.main._compute_daytype_metrics", return_value={}
        ), patch(
            "backend.main._compute_baseline_quality", return_value={}
        ), patch(
            "backend.main._compute_weather_sensitivity", return_value={}
        ), patch(
            "backend.main._compute_peak_time_accuracy", return_value=100.0
        ), patch(
            "backend.main._compute_interactions", return_value={}
        ), patch(
            "backend.main._compute_dr_accuracy", return_value={"dr_accuracy_pct": None}
        ):
            response = v2_dayahead({"date": "2024-01-10", "baseline_days": 7})

        metadata = response["metadata"]
        self.assertEqual(metadata["requested_date"], "2024-01-10")
        self.assertEqual(metadata["effective_date"], "2024-01-10")
        self.assertEqual(metadata["model_scope"], "global")
        self.assertIn("weather_engine", metadata)
        self.assertIn("features", metadata["weather_engine"])


class TestEngineHolidayFeatures(unittest.TestCase):
    def test_engine_marks_known_holiday_for_default_region(self):
        from backend.engine import EDAEngine

        class FakeIndia(dict):
            def __init__(self, *args, **kwargs):
                super().__init__({pd.Timestamp("2024-01-26").date(): "Republic Day"})

        fake_holidays = types.SimpleNamespace(India=FakeIndia)

        df = pd.DataFrame(
            [
                {
                    "date": "2024-01-26",
                    "time_block": block,
                    "total_drawal": 1000.0 + block,
                    "temperature": 18.0,
                    "humidity": 55.0,
                    "precipitation": 0.0,
                }
                for block in range(1, 97)
            ]
        )

        with patch.dict("sys.modules", {"holidays": fake_holidays}), patch.object(
            EDAEngine, "_load_data_static", wraps=EDAEngine._load_data_static
        ) as wrapped_loader, patch("backend.engine.os.path.exists", return_value=True), patch(
            "backend.engine.pd.read_csv", return_value=df
        ):
            engine = EDAEngine("dummy.csv")

        self.assertEqual(wrapped_loader.call_args.kwargs.get("region"), "haryana")
        self.assertTrue((engine._df["is_holiday"] == 1).all())


class TestT2Pipeline(unittest.TestCase):
    def test_run_t2_pipeline_uses_t1_forecast_as_sequential_history(self):
        import backend.short_term_pipeline as stp

        df = _build_history(days=14)
        t1_date = "2024-01-14"
        t2_date = "2024-01-15"
        t1_forecast = np.linspace(2100.0, 2195.0, 96)
        t2_forecast = np.linspace(1200.0, 1295.0, 96)
        call_log = []

        def fake_run_short_term_pipeline(df, target_date, actual_blocks=40, config=None):
            call_log.append(
                {
                    "df": df.copy(),
                    "target_date": target_date,
                    "actual_blocks": actual_blocks,
                }
            )
            if target_date == t1_date:
                return {
                    "series": {"forecast": t1_forecast.tolist()},
                }
            if target_date == t2_date:
                return {
                    "series": {
                        "forecast": t2_forecast.tolist(),
                        "final_load": t2_forecast.tolist(),
                        "hybrid_ai_forecast": t2_forecast.tolist(),
                    },
                    "metadata": {},
                    "forecast_df": [
                        {"time_block": idx + 1, "forecast": float(val)}
                        for idx, val in enumerate(t2_forecast)
                    ],
                }
            raise AssertionError(f"Unexpected target_date {target_date}")

        with patch.object(stp, "run_short_term_pipeline", side_effect=fake_run_short_term_pipeline):
            result = stp.run_t2_pipeline(
                df=df,
                t1_date=t1_date,
                config={"t2_seam_bridge": {"window_blocks": 4}},
            )

        self.assertEqual(len(call_log), 2)
        self.assertEqual(call_log[0]["target_date"], t1_date)
        self.assertEqual(call_log[1]["target_date"], t2_date)
        self.assertEqual(call_log[0]["actual_blocks"], 0)
        self.assertEqual(call_log[1]["actual_blocks"], 0)

        second_df = call_log[1]["df"]
        seeded_t1 = second_df[second_df["date"].astype(str) == t1_date].sort_values("time_block")
        seeded_t2 = second_df[second_df["date"].astype(str) == t2_date].sort_values("time_block")

        self.assertEqual(len(seeded_t1), 96)
        self.assertEqual(len(seeded_t2), 96)
        np.testing.assert_allclose(seeded_t1["total_drawal"].to_numpy(dtype=float), t1_forecast, atol=1e-6)

        self.assertAlmostEqual(result["series"]["forecast"][0], float(t1_forecast[-1]), places=6)
        self.assertAlmostEqual(result["forecast_df"][0]["forecast"], float(t1_forecast[-1]), places=6)
        self.assertEqual(result["t1_date"], t1_date)
        self.assertEqual(result["t2_date"], t2_date)
        self.assertTrue(result["metadata"]["sequential_forecast"]["enabled"])
        self.assertGreater(abs(result["metadata"]["sequential_forecast"]["seam_gap_before_mw"]), 100.0)
        self.assertEqual(result["metadata"]["sequential_forecast"]["seam_gap_after_mw"], 0.0)


if __name__ == "__main__":
    unittest.main()
