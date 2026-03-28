import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _build_hybrid_history(days: int = 48) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    for day_idx, ts in enumerate(dates):
        dow = ts.weekday()
        is_weekend = dow >= 5
        vol_boost = 120.0 if day_idx % 9 == 0 else 0.0
        for block in range(1, 97):
            angle = (2.0 * np.pi * (block - 1)) / 96.0
            morning = 80.0 * np.exp(-0.5 * (((block - 28.0) % 96) / 6.0) ** 2)
            evening = 140.0 * np.exp(-0.5 * (((block - 76.0) % 96) / 8.0) ** 2)
            weekend_adj = -60.0 * np.sin(angle) if is_weekend else 0.0
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "time_block": block,
                    "total_drawal": 3200.0 + (day_idx * 3.0) + morning + evening + weekend_adj + vol_boost,
                }
            )
    return pd.DataFrame(rows)


class TestHybridForecastEngine(unittest.TestCase):
    def test_engine_returns_96_block_forecast_with_uncertainty(self):
        from backend.hybrid import HybridForecastConfig, HybridForecastingEngine

        df = _build_hybrid_history(48)
        engine = HybridForecastingEngine(
            HybridForecastConfig(
                lookback_days=7,
                min_history_days=21,
                training_days=36,
                blend_weight=0.3,
                sequence_epochs=5,
            )
        )
        target_date = sorted(df["date"].astype(str).unique().tolist())[-1]
        fallback = (
            df[df["date"] == target_date]
            .sort_values("time_block")["total_drawal"]
            .to_numpy(dtype=float)
        )
        result = engine.run(df, target_date=target_date, fallback_forecast=fallback)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.forecast), 96)
        self.assertEqual(len(result.p10), 96)
        self.assertEqual(len(result.p90), 96)
        self.assertEqual(len(result.confidence), 96)
        self.assertIn("resnet_lstm_moe_gpr", result.metadata.get("engine", ""))

    def test_forecast_aliases_delegate_to_v2_dayahead(self):
        from backend.main import forecast_get, forecast_run

        fake = {"metadata": {"effective_date": "2024-02-01"}, "series": {"forecast": [1.0] * 96}}
        with patch("backend.main.v2_dayahead", return_value=fake) as mocked:
            res_get = forecast_get(date="2024-02-01", baseline_days=7, region="punjab")
            res_post = forecast_run({"date": "2024-02-01"})

        self.assertEqual(res_get["metadata"]["effective_date"], "2024-02-01")
        self.assertEqual(res_post["metadata"]["effective_date"], "2024-02-01")
        self.assertGreaterEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
