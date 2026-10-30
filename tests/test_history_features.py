from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.history_features import CausalBaselineBuilder, CausalHistoryFeatureBuilder


def literal_samples(current_iron: float = 999999.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["CUR"],
            "tap_no": [200],
            "spout_no": [1],
            "reference_time": pd.to_datetime(["2024-07-01 12:00:00"]).astype("datetime64[ns]"),
            "tap_iron": [current_iron],
            "tap_time_len": [999.0],
        }
    )


def literal_tap_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["H1", "H2", "H3", "H4", "CUR", "EQ", "FUTURE"],
            "tap_no": [101, 102, 103, 104, 200, 105, 106],
            "spout_no": [1, 2, 1, 1, 1, 2, 1],
            "reference_time": pd.to_datetime(
                [
                    "2024-07-01 06:00:00",
                    "2024-07-01 07:00:00",
                    "2024-07-01 08:00:00",
                    "2024-07-01 09:00:00",
                    "2024-07-01 10:00:00",
                    "2024-07-01 10:00:00",
                    "2024-07-01 11:00:00",
                ]
            ).astype("datetime64[ns]"),
            "tap_end_time": pd.to_datetime(
                [
                    "2024-07-01 08:00:00",
                    "2024-07-01 09:00:00",
                    "2024-07-01 10:00:00",
                    "2024-07-01 11:00:00",
                    "2024-07-01 11:30:00",
                    "2024-07-01 12:00:00",
                    "2024-07-01 13:00:00",
                ]
            ).astype("datetime64[ns]"),
            "tap_iron": [800.0, 900.0, 1000.0, 1100.0, 999999.0, 1200.0, 500000.0],
            "tap_time_len": [100.0, 100.0, 100.0, 100.0, 999.0, 100.0, 100.0],
            "slag_time_len": [60.0, 61.0, 62.0, 63.0, 99.0, 64.0, 65.0],
            "break_slag_time_len": [3.0, 4.0, 5.0, 6.0, 99.0, 7.0, 8.0],
            "ladle_count": [3.0, 3.0, 4.0, 4.0, 9.0, 5.0, 6.0],
        }
    ).sort_values(["tap_end_time", "sample_id"], kind="mergesort").reset_index(drop=True)


class HistoryFeatureTests(unittest.TestCase):
    def test_history_includes_equal_end_time_but_excludes_self_and_future(self):
        features = CausalHistoryFeatureBuilder().build(literal_samples(), literal_tap_history())

        self.assertEqual(5.0, features.loc[0, "g1_visible_history_count"])
        self.assertEqual(1200.0, features.loc[0, "g1_iron_lag_1"])
        self.assertEqual(1100.0, features.loc[0, "g1_iron_lag_2"])
        self.assertEqual(1000.0, features.loc[0, "g1_iron_lag_3"])
        self.assertEqual(800.0, features.loc[0, "g1_iron_lag_5"])
        self.assertTrue(np.isnan(features.loc[0, "g1_iron_lag_10"]))
        self.assertEqual(0.0, features.loc[0, "g1_hours_since_last_tap_end"])

    def test_global_statistics_and_physical_rate_use_literal_values(self):
        features = CausalHistoryFeatureBuilder().build(literal_samples(), literal_tap_history())

        self.assertEqual(1100.0, features.loc[0, "g1_iron_roll_3_mean"])
        self.assertAlmostEqual(81.64965809277261, features.loc[0, "g1_iron_roll_3_std"], places=10)
        self.assertEqual(1100.0, features.loc[0, "g1_iron_roll_3_median"])
        self.assertEqual(1000.0, features.loc[0, "g1_iron_roll_3_min"])
        self.assertEqual(1200.0, features.loc[0, "g1_iron_roll_3_max"])
        self.assertAlmostEqual(1018.9339828220179, features.loc[0, "g1_iron_ewma_hl2"], places=10)
        self.assertEqual(12.0, features.loc[0, "g1_rate_lag_1"])
        self.assertEqual(12.0, features.loc[0, "g5_rate_lag_1"])
        self.assertEqual(11.0, features.loc[0, "g5_rate_roll_3_median"])
        self.assertEqual(0.0, features.loc[0, "g5_iron_identity_residual_lag1"])
        self.assertEqual(100.0, features.loc[0, "g1_iron_diff_lag1_lag2"])
        self.assertAlmostEqual(100.0 / 1100.0, features.loc[0, "g1_iron_change_rate_lag1_lag2"])

    def test_same_spout_history_is_separate_from_global_history(self):
        features = CausalHistoryFeatureBuilder().build(literal_samples(), literal_tap_history())

        self.assertEqual(3.0, features.loc[0, "g2_visible_history_count"])
        self.assertEqual(1100.0, features.loc[0, "g2_iron_lag_1"])
        self.assertEqual(1000.0, features.loc[0, "g2_iron_lag_2"])
        self.assertEqual(800.0, features.loc[0, "g2_iron_lag_3"])
        self.assertEqual(1.0, features.loc[0, "g2_hours_since_same_spout"])
        self.assertAlmostEqual(966.6666666666666, features.loc[0, "g2_iron_roll_3_mean"])
        self.assertAlmostEqual(
            -133.33333333333337,
            features.loc[0, "g2_iron_roll_3_mean_diff_global"],
        )

    def test_extreme_current_target_never_changes_own_features(self):
        builder = CausalHistoryFeatureBuilder()

        ordinary = builder.build(literal_samples(current_iron=1000.0), literal_tap_history())
        extreme = builder.build(literal_samples(current_iron=1e30), literal_tap_history())

        pd.testing.assert_frame_equal(ordinary, extreme)


class BaselineTests(unittest.TestCase):
    def test_baselines_use_visible_history_in_physical_units(self):
        train = pd.DataFrame(
            {
                "sample_id": ["T1", "T2", "T3"],
                "tap_iron": [800.0, 1000.0, 1200.0],
                "tap_time_len": [100.0, 100.0, 100.0],
            }
        )
        builder = CausalBaselineBuilder().fit(train, literal_tap_history())

        baselines = builder.transform(literal_samples(), literal_tap_history(), fold=1, split="valid")

        self.assertEqual("CUR", baselines.loc[0, "sample_id"])
        self.assertEqual(1, baselines.loc[0, "fold"])
        self.assertEqual("valid", baselines.loc[0, "split"])
        self.assertEqual(1200.0, baselines.loc[0, "baseline_prev_tap_iron"])
        self.assertEqual(1100.0, baselines.loc[0, "baseline_same_spout_prev_tap_iron"])
        self.assertEqual(1100.0, baselines.loc[0, "baseline_median_3_tap_iron"])
        self.assertEqual(1000.0, baselines.loc[0, "baseline_median_5_tap_iron"])
        self.assertEqual(0, baselines.loc[0, "baseline_prev_fallback"])

    def test_no_history_uses_only_fitted_train_medians_and_marks_fallback(self):
        train = pd.DataFrame(
            {
                "sample_id": ["T1", "T2", "T3"],
                "tap_iron": [800.0, 1000.0, 1200.0],
                "tap_time_len": [90.0, 100.0, 110.0],
            }
        )
        early = pd.DataFrame(
            {
                "sample_id": ["EARLY"],
                "tap_no": [1],
                "spout_no": [4],
                "reference_time": pd.to_datetime(["2024-06-01 00:00:00"]).astype("datetime64[ns]"),
            }
        )
        builder = CausalBaselineBuilder().fit(train, literal_tap_history())

        baselines = builder.transform(early, literal_tap_history(), fold=1, split="train")

        self.assertEqual(1000.0, baselines.loc[0, "baseline_prev_tap_iron"])
        self.assertEqual(100.0, baselines.loc[0, "baseline_prev_tap_time_len"])
        self.assertEqual(10.0, baselines.loc[0, "baseline_prev_rate"])
        self.assertEqual(1, baselines.loc[0, "baseline_prev_fallback"])
        self.assertEqual(1, baselines.loc[0, "baseline_same_spout_prev_fallback"])

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            CausalBaselineBuilder().transform(
                literal_samples(), literal_tap_history(), fold=1, split="valid"
            )


if __name__ == "__main__":
    unittest.main()
