from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.burden_state_features import BurdenStateFeatureBuilder
from src.constants import BURDEN_CHANNELS
from src.timeaware_contract import G8_FEATURE_NAMES
from tests.fixtures import timeaware_burden_events


def sample_at(reference_time: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["Q0"],
            "tap_no": [1],
            "spout_no": [1],
            "reference_time": pd.to_datetime([reference_time]),
        }
    )


class BurdenStateFeatureBuilderTests(unittest.TestCase):
    def test_duplicate_timestamp_keeps_last_and_same_state_is_not_a_change(self) -> None:
        batch = BurdenStateFeatureBuilder().build(
            sample_at("2024-07-01 09:00:00"), timeaware_burden_events()
        )

        self.assertEqual(list(G8_FEATURE_NAMES), list(batch.features.columns))
        self.assertEqual(3.0, batch.features.loc[0, "g8_pig_current_value"])
        self.assertEqual(1, batch.diagnostics["duplicate_timestamp_count"])
        self.assertEqual(1, batch.diagnostics["duplicate_row_count"])
        self.assertEqual(
            0.0, batch.features.loc[0, "g8_pig_effective_change_count_6h"]
        )
        self.assertEqual(2.0, batch.features.loc[0, "g8_raw_event_count_12h"])

    def test_grid_is_old_to_new_forward_filled_and_tracks_age_and_missingness(self) -> None:
        reference = pd.Timestamp("2024-07-01 09:00:00")
        batch = BurdenStateFeatureBuilder().build(
            sample_at(str(reference)), timeaware_burden_events()
        )

        expected_times = pd.date_range(
            reference - pd.Timedelta(hours=48), reference, freq="2h"
        ).to_numpy(dtype="datetime64[ns]")
        np.testing.assert_array_equal(expected_times, batch.grid.grid_times[0])
        self.assertEqual((1, 25, 5), batch.grid.state_values.shape)
        self.assertEqual((1, 25, 5), batch.grid.state_age_hours.shape)
        self.assertEqual((1, 25, 5), batch.grid.effective_change_mask.shape)
        self.assertEqual((1, 25), batch.grid.raw_event_count.shape)
        self.assertEqual((1, 25, 5), batch.grid.missing_mask.shape)
        self.assertTrue(np.all(np.diff(batch.grid.grid_times[0]) > np.timedelta64(0, "ns")))
        self.assertEqual(3.0, batch.grid.state_values[0, -1, 0])
        self.assertEqual(9.0, batch.grid.state_age_hours[0, -1, 0])
        self.assertFalse(batch.grid.missing_mask[0, -1, 0])
        self.assertTrue(batch.grid.missing_mask[0, 0, 0])
        self.assertNotIn(9.0, batch.grid.state_values[0, :, 0])

    def test_event_exactly_at_reference_time_is_visible(self) -> None:
        batch = BurdenStateFeatureBuilder().build(
            sample_at("2024-07-01 00:00:00"), timeaware_burden_events()
        )

        self.assertEqual(3.0, batch.features.loc[0, "g8_pig_current_value"])
        self.assertEqual(0.0, batch.features.loc[0, "g8_pig_hours_since_last_real_change"])

    def test_real_time_weighting_lags_slopes_changes_and_event_gaps(self) -> None:
        times = pd.to_datetime(
            ["2024-07-01 00:00:00", "2024-07-01 01:00:00", "2024-07-01 09:00:00"]
        )
        events = pd.DataFrame({"cal_time": times})
        for channel in BURDEN_CHANNELS:
            events[channel] = [0.0, 10.0, 20.0]

        batch = BurdenStateFeatureBuilder().build(
            sample_at("2024-07-01 12:00:00"), events
        )
        row = batch.features.iloc[0]

        self.assertAlmostEqual((0.0 * 1.0 + 10.0 * 8.0 + 20.0 * 3.0) / 12.0, row["g8_pig_time_weighted_mean_12h"])
        self.assertEqual(20.0, row["g8_pig_value_at_2h"])
        self.assertEqual(10.0, row["g8_pig_value_at_8h"])
        self.assertTrue(np.isnan(row["g8_pig_value_at_24h"]))
        self.assertEqual(10.0, row["g8_pig_current_minus_lag_8h"])
        expected_slope = float(np.polyfit([0.0, 1.0, 9.0], [0.0, 10.0, 20.0], 1)[0])
        self.assertAlmostEqual(expected_slope, row["g8_pig_real_time_slope_12h"])
        self.assertEqual(10.0, row["g8_pig_max_abs_change_12h"])
        self.assertEqual(2.0, row["g8_pig_effective_change_count_12h"])
        self.assertEqual(4.5, row["g8_event_gap_median_24h"])
        self.assertEqual(3.5, row["g8_event_gap_std_24h"])

    def test_output_preserves_sample_order_and_inputs(self) -> None:
        samples = pd.concat(
            [sample_at("2024-07-01 09:00:00"), sample_at("2024-07-01 00:00:00")],
            ignore_index=True,
        )
        samples.loc[:, "sample_id"] = ["LATE", "EARLY"]
        samples.index = pd.Index([9, 2])
        events = timeaware_burden_events()
        samples_before = samples.copy(deep=True)
        events_before = events.copy(deep=True)

        batch = BurdenStateFeatureBuilder().build(samples, events)

        pd.testing.assert_frame_equal(samples_before, samples)
        pd.testing.assert_frame_equal(events_before, events)
        self.assertEqual([9, 2], list(batch.features.index))
        self.assertEqual(3.0, batch.features.loc[9, "g8_pig_current_value"])
        self.assertEqual(3.0, batch.features.loc[2, "g8_pig_current_value"])


if __name__ == "__main__":
    unittest.main()
