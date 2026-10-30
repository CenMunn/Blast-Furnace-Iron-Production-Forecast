from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.burden_features import BurdenFeatureBuilder
from tests.test_hourly_features import one_sample


def eleven_visible_events_plus_future() -> pd.DataFrame:
    times = list(pd.date_range("2024-07-01 02:00:00", periods=11, freq="h"))
    times.append(pd.Timestamp("2024-07-01 13:00:00"))
    values = np.arange(1.0, 13.0)
    return pd.DataFrame(
        {
            "cal_time": pd.to_datetime(times).astype("datetime64[ns]"),
            "pig": values,
            "all_quality": values + 50.0,
            "consumption": values / 10.0,
            "fuel_rate": values + 500.0,
            "coke_rate": values + 300.0,
        }
    )


class BurdenFeatureTests(unittest.TestCase):
    def test_recent_event_statistics_include_equal_time_and_exclude_future(self):
        features = BurdenFeatureBuilder().build(one_sample(), eleven_visible_events_plus_future())

        self.assertEqual(11.0, features.loc[0, "g4_event_history_count"])
        self.assertEqual(0.0, features.loc[0, "g4_hours_since_last_event"])
        self.assertEqual(11.0, features.loc[0, "g4_pig_recent_1_last"])
        self.assertEqual(10.0, features.loc[0, "g4_pig_recent_3_mean"])
        self.assertAlmostEqual(0.816496580927726, features.loc[0, "g4_pig_recent_3_std"])
        self.assertEqual(10.0, features.loc[0, "g4_pig_recent_3_median"])
        self.assertEqual(9.0, features.loc[0, "g4_pig_recent_3_min"])
        self.assertEqual(11.0, features.loc[0, "g4_pig_recent_3_max"])
        self.assertEqual(30.0, features.loc[0, "g4_pig_recent_3_sum"])
        self.assertAlmostEqual(1.0, features.loc[0, "g4_pig_recent_3_trend"])
        self.assertEqual(10.0, features.loc[0, "g4_effective_length_10"])
        self.assertEqual(0.0, features.loc[0, "g4_insufficient_10"])
        self.assertEqual(1.0, features.loc[0, "g4_event_gap_mean_hours"])
        self.assertEqual(0.0, features.loc[0, "g4_event_gap_std_hours"])

    def test_no_visible_event_returns_missing_values_and_flags(self):
        early = one_sample()
        early.loc[:, "reference_time"] = pd.Timestamp("2024-07-01 00:00:00")

        features = BurdenFeatureBuilder().build(early, eleven_visible_events_plus_future())

        self.assertEqual(0.0, features.loc[0, "g4_event_history_count"])
        self.assertEqual(1.0, features.loc[0, "g4_insufficient_10"])
        self.assertTrue(np.isnan(features.loc[0, "g4_pig_recent_1_last"]))


if __name__ == "__main__":
    unittest.main()
