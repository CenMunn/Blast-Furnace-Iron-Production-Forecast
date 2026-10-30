from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.burden_state_features import BurdenStateFeatureBuilder
from src.schedule_features import ScheduleFeatureBuilder
from src.timeaware_contract import PUBLIC_SAMPLE_COLUMNS
from tests.fixtures import timeaware_burden_events, timeaware_schedule_samples


class TimeawareNoFutureTests(unittest.TestCase):
    def test_future_sample_time_mutation_cannot_change_current_g7(self) -> None:
        samples = timeaware_schedule_samples()
        future = samples.iloc[[0]].copy()
        future.loc[:, "sample_id"] = "FUTURE"
        future.loc[:, "reference_time"] = pd.Timestamp("2024-07-01 20:00:00")
        context = pd.concat([samples, future], ignore_index=True)
        before = ScheduleFeatureBuilder().build(context).iloc[3].to_numpy()

        mutated = context.copy()
        mutated.loc[4, "reference_time"] = pd.Timestamp("2024-07-03 20:00:00")
        after = ScheduleFeatureBuilder().build(mutated).iloc[3].to_numpy()

        np.testing.assert_allclose(before, after, equal_nan=True)

    def test_future_burden_delete_or_extreme_mutation_cannot_change_current_g8(self) -> None:
        samples = timeaware_schedule_samples().iloc[[3]].copy()
        events = timeaware_burden_events()
        builder = BurdenStateFeatureBuilder()
        before = builder.build(samples, events)

        without_future = builder.build(
            samples, events.loc[events["cal_time"] <= samples.iloc[0]["reference_time"]]
        )
        mutated_events = events.copy()
        future = mutated_events["cal_time"] > samples.iloc[0]["reference_time"]
        mutated_events.loc[future, ["pig", "all_quality", "consumption", "fuel_rate", "coke_rate"]] = 1e30
        with_extreme_future = builder.build(samples, mutated_events)

        for candidate in (without_future, with_extreme_future):
            pd.testing.assert_frame_equal(before.features, candidate.features)
            np.testing.assert_array_equal(before.grid.grid_times, candidate.grid.grid_times)
            np.testing.assert_allclose(
                before.grid.state_values, candidate.grid.state_values, equal_nan=True
            )
            np.testing.assert_allclose(
                before.grid.state_age_hours,
                candidate.grid.state_age_hours,
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                before.grid.effective_change_mask,
                candidate.grid.effective_change_mask,
            )
            np.testing.assert_array_equal(
                before.grid.raw_event_count, candidate.grid.raw_event_count
            )
            np.testing.assert_array_equal(
                before.grid.missing_mask, candidate.grid.missing_mask
            )

    def test_target_permutation_and_extremes_cannot_change_g7_or_g8(self) -> None:
        samples = timeaware_schedule_samples().assign(
            tap_iron=[1400.0, 1500.0, 1600.0, 1700.0],
            tap_time_len=[100.0, 110.0, 120.0, 130.0],
        )
        public = samples.loc[:, PUBLIC_SAMPLE_COLUMNS]
        schedule_before = ScheduleFeatureBuilder().build(public)
        burden_before = BurdenStateFeatureBuilder().build(
            public, timeaware_burden_events()
        ).features

        mutated = samples.copy()
        mutated.loc[:, "tap_iron"] = [1e30, -1e30, 7.0, 8.0]
        mutated.loc[:, "tap_time_len"] = [1e-30, 9e29, -4.0, 5.0]
        mutated_public = mutated.loc[:, PUBLIC_SAMPLE_COLUMNS]

        pd.testing.assert_frame_equal(
            schedule_before, ScheduleFeatureBuilder().build(mutated_public)
        )
        pd.testing.assert_frame_equal(
            burden_before,
            BurdenStateFeatureBuilder()
            .build(mutated_public, timeaware_burden_events())
            .features,
        )

    def test_earlier_validation_public_time_updates_later_validation_g7(self) -> None:
        samples = timeaware_schedule_samples()
        with_earlier_valid = ScheduleFeatureBuilder().build(samples)
        without_earlier_valid = ScheduleFeatureBuilder().build(samples.drop(index=2))

        self.assertEqual(4.0, with_earlier_valid.loc[3, "g7_start_gap_lag_1"])
        self.assertEqual(7.0, without_earlier_valid.loc[3, "g7_start_gap_lag_1"])

    def test_equal_time_g7_samples_are_not_mutual_history(self) -> None:
        samples = timeaware_schedule_samples().iloc[:3].copy()
        duplicate_time = samples.iloc[[2]].copy()
        duplicate_time.loc[:, "sample_id"] = "S2X"
        context = pd.concat([samples, duplicate_time], ignore_index=True)

        features = ScheduleFeatureBuilder().build(context)

        self.assertEqual(3.0, features.loc[2, "g7_start_gap_lag_1"])
        self.assertEqual(3.0, features.loc[3, "g7_start_gap_lag_1"])

    def test_equal_time_burden_event_is_visible_but_later_event_is_not(self) -> None:
        samples = timeaware_schedule_samples().iloc[[0]].copy()
        events = timeaware_burden_events()

        features = BurdenStateFeatureBuilder().build(samples, events).features

        self.assertEqual(3.0, features.iloc[0]["g8_pig_current_value"])
        self.assertNotEqual(9.0, features.iloc[0]["g8_pig_current_value"])


if __name__ == "__main__":
    unittest.main()
