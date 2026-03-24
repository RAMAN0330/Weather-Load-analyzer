import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _build_df(dates=("2024-01-01", "2024-01-02")) -> pd.DataFrame:
    rows = []
    for d in dates:
        for block in range(1, 97):
            rows.append(
                {
                    "date": d,
                    "time_block": block,
                    "total_drawal": 1000.0 + (block * 1.0),
                    "temperature": 25.0,
                    "humidity": 60.0,
                    "precipitation": 0.0,
                    "wind_speed_10m": 3.0,
                }
            )
    return pd.DataFrame(rows)


class TestSimulatorBlocksEndpoint(unittest.TestCase):
    def test_baseline_and_final_nonzero(self):
        from backend.main import SimulatorBlocksRequest, api_simulator_blocks

        df = _build_df()
        baseline = np.full(96, 1000.0, dtype=float).tolist()
        deltas = {
            "temperature": [1.0] * 96,
            "humidity": [0.0] * 96,
            "precipitation": [0.0] * 96,
            "wind_speed_10m": [0.0] * 96,
        }
        fake_dayahead = {"series": {"historical_baseline": baseline, "weather_feature_deltas": deltas}}

        weights_df = pd.DataFrame(
            {
                "block": list(range(1, 97)),
                "temp_weight": [0.001] * 96,
                "humidity_weight": [0.0] * 96,
                "rain_weight": [0.0] * 96,
                "wind_weight": [0.0] * 96,
                "daytype_weight": [0.0] * 96,
                "holiday_weight": [0.0] * 96,
                "weight_confidence": [0.9] * 96,
            }
        )

        with patch("backend.main._get_df", return_value=df), patch(
            "backend.main._compute_dayahead", return_value=fake_dayahead
        ), patch("backend.main.compute_block_driver_weights", return_value=weights_df):
            res = api_simulator_blocks(SimulatorBlocksRequest(date="2024-01-02", baseline_days=7))

        self.assertIn("blocks", res)
        self.assertEqual(len(res["blocks"]), 96)
        first = res["blocks"][0]
        self.assertGreater(first["baseline_mw"], 0.0)
        self.assertGreater(first["final_mw"], 0.0)
        self.assertAlmostEqual(first["baseline_mw"], 1000.0, places=6)
        self.assertAlmostEqual(first["final_mw"], 1000.0 * (1.0 + 0.001), places=6)


if __name__ == "__main__":
    unittest.main()

