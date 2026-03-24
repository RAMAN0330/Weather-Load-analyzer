import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gridintel_engine import KPIEngine, FeatureEngineering  # noqa: E402


def _build_sample_df():
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(0, 96 * 2):
        ts = base + timedelta(minutes=15 * i)
        hour = ts.hour
        load = 1000 + 50 * np.sin(i / 10.0) + (200 if 18 <= hour <= 21 else 0)
        rows.append({
            "Datetime": ts,
            "total_drawal": load,
            "temperature": 25 + 5 * np.sin(i / 30.0),
            "humidity": 60 + 10 * np.cos(i / 25.0),
            "precipitation": 0.0 if i % 30 else 1.0,
        })
    df = pd.DataFrame(rows)
    df["time_block"] = (df["Datetime"].dt.hour * 4) + (df["Datetime"].dt.minute // 15) + 1
    return df


class TestKPIEngine(unittest.TestCase):
    def setUp(self):
        self.df = _build_sample_df()
        self.fe = FeatureEngineering()
        self.kpi = KPIEngine()
        self.df = self.fe.add_features(self.df)

    def test_shape_curve_kpis(self):
        kpis = self.kpi.compute_shape_curve(self.df, "total_drawal")
        self.assertIn("duck_curve_severity", kpis)
        self.assertIn("shoulder_strength", kpis)
        self.assertIn("curve_symmetry_score", kpis)

    def test_risk_capacity_breach(self):
        df = self.df.copy()
        df["capacity_mw"] = 1050.0
        kpis = self.kpi.compute_risk(df, "total_drawal")
        self.assertIn("capacity_breach_risk", kpis)
        self.assertGreaterEqual(kpis["capacity_breach_risk"], 0.0)

    def test_composite_indices(self):
        comp = self.kpi.compute_composites({
            "relative_volatility": 0.1,
            "extreme_event_frequency": 0.02,
            "ramp_stress_score": 25.0,
            "forecast_mape": 8.0,
            "temp_sensitivity_mw_c": 0.3
        })
        self.assertIn("forecast_confidence_index", comp)
        self.assertIn("dispatch_difficulty_index", comp)


if __name__ == "__main__":
    unittest.main()
