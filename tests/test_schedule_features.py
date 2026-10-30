from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.schedule_features import ScheduleFeatureBuilder
from src.timeaware_contract import G7_FEATURE_NAMES
from tests.fixtures import timeaware_schedule_samples


class ScheduleFeatureBuilderTests(unittest.TestCase):
    def test_hand_calculated_gap_rolling_count_and_spout_features(self) -> None:
        samples = timeaware_schedule_samples()
        features = ScheduleFeatureBuilder().build(samples)

        self.assertEqual(list(G7_FEATURE_NAMES), list(features.columns))
        self.assertEqual(4.0, features.loc[3, "g7_start_gap_lag_1"])
        self.assertEqual(3.0, features.loc[3, "g7_start_gap_lag_2"])
        self.assertEqual(2.0, features.loc[3, "g7_start_gap_lag_3"])
        self.assertEqual(2.5, features.loc[3, "g7_start_gap_mean_3"])
        self.assertEqual(0.5, features.loc[3, "g7_start_gap_std_3"])
        self.assertEqual(1.0, features.loc[3, "g7_tap_count_6h"])
        self.assertEqual(4.0, features.loc[3, "g7_hours_since_same_spout_start"])
        self.assertEqual(4.0, features.loc[3, "g7_same_spout_start_gap_lag_1"])
        self.assertEqual(5.0, features.loc[3, "g7_same_spout_start_gap_lag_2"])

    def test_equal_time_samples_share_strictly_earlier_history_only(self) -> None:
        samples = timeaware_schedule_samples()
        equal = samples.iloc[[2]].copy()
        equal.loc[:, "sample_id"] = "S2B"
        equal.loc[:, "spout_no"] = 2
        combined = pd.concat([samples, equal], ignore_index=True)

        features = ScheduleFeatureBuilder().build(combined)

        first = features.loc[2]
        second = features.loc[4]
        self.assertEqual(3.0, first["g7_start_gap_lag_1"])
        self.assertEqual(3.0, second["g7_start_gap_lag_1"])
        self.assertEqual(2.0, first["g7_start_gap_lag_2"])
        self.assertEqual(2.0, second["g7_start_gap_lag_2"])
        self.assertEqual(2.0, first["g7_tap_count_6h"])
        self.assertEqual(2.0, second["g7_tap_count_6h"])

    def test_minute_fraction_uses_minute_and_second_fraction(self) -> None:
        samples = timeaware_schedule_samples().iloc[[0]].copy()
        samples.loc[:, "reference_time"] = pd.Timestamp("2024-07-01 09:30:30")

        features = ScheduleFeatureBuilder().build(samples)

        self.assertAlmostEqual(30.5 / 60.0, features.iloc[0]["g7_reference_minute_fraction"])

    def test_output_preserves_input_order_and_does_not_mutate_input(self) -> None:
        samples = timeaware_schedule_samples().iloc[[3, 0, 2, 1]].copy()
        samples.index = pd.Index([40, 10, 30, 20])
        before = samples.copy(deep=True)

        features = ScheduleFeatureBuilder().build(samples)

        pd.testing.assert_frame_equal(before, samples)
        self.assertEqual(list(samples.index), list(features.index))
        self.assertEqual(list(G7_FEATURE_NAMES), list(features.columns))
        self.assertEqual(4.0, features.loc[40, "g7_start_gap_lag_1"])

    def test_missing_history_remains_nan_and_long_gap_flags_use_current_gap(self) -> None:
        samples = timeaware_schedule_samples().iloc[[0, 1]].copy()
        samples.loc[samples.index[1], "reference_time"] = pd.Timestamp("2024-07-01 04:01:00")

        features = ScheduleFeatureBuilder().build(samples)

        self.assertTrue(np.isnan(features.loc[0, "g7_start_gap_lag_1"]))
        self.assertEqual(1.0, features.loc[1, "g7_long_gap_flag_180m"])
        self.assertEqual(1.0, features.loc[1, "g7_long_gap_flag_240m"])


if __name__ == "__main__":
    unittest.main()
