from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_io import RawTables
from src.feature_assembler import FeatureAssembler
from src.hourly_features import HourlyFeatureBuilder
from src.legacy_features import LegacyFeatureBuilder
from tests.test_history_features import literal_tap_history


def one_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["CUR"],
            "tap_no": [200],
            "spout_no": [1],
            "reference_time": pd.to_datetime(["2024-07-01 12:00:00"]).astype("datetime64[ns]"),
            "tap_iron": [1300.0],
            "tap_time_len": [100.0],
        }
    )


def literal_hourly() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2024-07-01 09:00:00",
            "2024-07-01 10:00:00",
            "2024-07-01 11:00:00",
            "2024-07-01 12:00:00",
            "2024-07-01 13:00:00",
        ]
    ).astype("datetime64[ns]")
    values = [100.0, 1.0, 3.0, 5.0, 999.0]
    channels = (
        "air_volume",
        "cold_air_press",
        "hot_air_press",
        "oxygen",
        "hot_air_temp",
        "coal_rate",
        "humidity",
        "gas_rate",
        "furnace_top_press",
        "upper_press_diff",
        "lower_press_diff",
        "total_press_diff",
        "air_press_ratio",
        "furnace_top_temp_avg",
        "air_speed",
        "furnace_throat_temp",
    )
    frame = pd.DataFrame({"clock": times})
    for offset, channel in enumerate(channels):
        frame[channel] = np.asarray(values) + float(offset)
    return frame


def literal_burden_for_assembler() -> pd.DataFrame:
    times = pd.date_range("2024-07-01 08:00:00", periods=5, freq="h")
    return pd.DataFrame(
        {
            "cal_time": times.astype("datetime64[ns]"),
            "pig": [1.0, 2.0, 3.0, 4.0, 5.0],
            "all_quality": [50.0, 51.0, 52.0, 53.0, 54.0],
            "consumption": [1.0, 1.1, 1.2, 1.3, 1.4],
            "fuel_rate": [500.0, 501.0, 502.0, 503.0, 504.0],
            "coke_rate": [300.0, 301.0, 302.0, 303.0, 304.0],
        }
    )


class HourlyFeatureTests(unittest.TestCase):
    def test_three_hour_window_excludes_left_boundary_includes_reference_and_future_never_enters(self):
        features = HourlyFeatureBuilder().build(one_sample(), literal_hourly())

        self.assertEqual(3.0, features.loc[0, "g3_air_volume_w3h_mean"])
        self.assertAlmostEqual(1.632993161855452, features.loc[0, "g3_air_volume_w3h_std"])
        self.assertEqual(3.0, features.loc[0, "g3_air_volume_w3h_median"])
        self.assertEqual(1.0, features.loc[0, "g3_air_volume_w3h_min"])
        self.assertEqual(5.0, features.loc[0, "g3_air_volume_w3h_max"])
        self.assertEqual(5.0, features.loc[0, "g3_air_volume_w3h_last"])
        self.assertAlmostEqual(2.0, features.loc[0, "g3_air_volume_w3h_slope"])
        self.assertEqual(4.0, features.loc[0, "g3_air_volume_w3h_first_last_diff"])
        self.assertEqual(2.0, features.loc[0, "g3_air_volume_w3h_max_abs_jump"])
        self.assertEqual(2.0, features.loc[0, "g3_air_volume_w3h_diff_mean"])
        self.assertEqual(3.0, features.loc[0, "g3_air_volume_w3h_valid_count"])
        self.assertEqual(0.0, features.loc[0, "g3_air_volume_w3h_missing_rate"])
        self.assertEqual(0.0, features.loc[0, "g3_air_volume_w3h_longest_missing_run"])
        self.assertEqual(5.0, features.loc[0, "g3_air_volume_w1h_mean"])

    def test_missing_hour_is_counted_on_expected_grid(self):
        hourly = literal_hourly()
        hourly.loc[hourly["clock"].eq(pd.Timestamp("2024-07-01 11:00:00")), "air_volume"] = np.nan

        features = HourlyFeatureBuilder().build(one_sample(), hourly)

        self.assertEqual(2.0, features.loc[0, "g3_air_volume_w3h_valid_count"])
        self.assertAlmostEqual(1.0 / 3.0, features.loc[0, "g3_air_volume_w3h_missing_rate"])
        self.assertEqual(1.0, features.loc[0, "g3_air_volume_w3h_longest_missing_run"])
        self.assertEqual(4.0, features.loc[0, "g3_air_volume_w3h_diff_mean"])

    def test_legacy_schema_is_exactly_288_unique_features(self):
        builder = LegacyFeatureBuilder()

        features = builder.build(one_sample(), literal_hourly(), literal_burden_for_assembler())

        self.assertEqual(288, len(builder.feature_names_))
        self.assertEqual(288, features.shape[1])
        self.assertEqual(288, len(set(features.columns)))
        self.assertEqual(5.0, features.loc[0, "g0_legacy_air_volume_w1h_mean"])

    def test_feature_assembler_has_fixed_groups_and_never_contains_current_targets(self):
        samples = one_sample()
        empty_test = samples.loc[:, ["sample_id", "tap_no", "spout_no", "reference_time"]].copy()
        tables = RawTables(
            train_samples=samples,
            test_a_samples=empty_test,
            test_b_samples=empty_test,
            operation_hourly=literal_hourly(),
            burden_change=literal_burden_for_assembler(),
            tap_history_train=literal_tap_history(),
            deduplicated_rows={},
        )

        raw = FeatureAssembler().build_raw(samples, tables)

        self.assertEqual(samples.index.tolist(), raw.index.tolist())
        self.assertEqual(len(raw.columns), len(set(raw.columns)))
        self.assertTrue(all(column.startswith(tuple(f"g{i}_" for i in range(7))) for column in raw))
        self.assertNotIn("tap_iron", raw.columns)
        self.assertNotIn("tap_time_len", raw.columns)
        self.assertEqual(200.0, raw.loc[0, "g0_tap_no"])


if __name__ == "__main__":
    unittest.main()
