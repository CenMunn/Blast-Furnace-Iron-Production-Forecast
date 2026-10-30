from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data_io import RawTables
from src.sequences import ThreeStreamSequenceBuilder
from tests.test_burden_features import eleven_visible_events_plus_future
from tests.test_history_features import literal_tap_history
from tests.test_hourly_features import literal_hourly, one_sample


def sequence_tables() -> RawTables:
    sample = one_sample()
    test = sample.loc[:, ["sample_id", "tap_no", "spout_no", "reference_time"]]
    return RawTables(
        train_samples=sample,
        test_a_samples=test,
        test_b_samples=test,
        operation_hourly=literal_hourly(),
        burden_change=eleven_visible_events_plus_future(),
        tap_history_train=literal_tap_history(),
        deduplicated_rows={},
    )


def inverse_channel(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values.astype(np.float64) * scale + mean


class ThreeStreamSequenceTests(unittest.TestCase):
    def test_shapes_masks_and_padding_are_exact(self):
        samples = one_sample()
        tables = sequence_tables()
        builder = ThreeStreamSequenceBuilder().fit(samples, tables)

        batch = builder.transform(samples, tables)

        self.assertEqual((1, 24, 16), batch.hourly_values.shape)
        self.assertEqual((1, 24), batch.hourly_mask.shape)
        self.assertEqual((1, 10, 8), batch.tap_values.shape)
        self.assertEqual((1, 10), batch.tap_mask.shape)
        self.assertEqual((1, 10, 6), batch.burden_values.shape)
        self.assertEqual((1, 10), batch.burden_mask.shape)
        self.assertEqual(4, int(batch.hourly_mask.sum()))
        self.assertEqual(5, int(batch.tap_mask.sum()))
        self.assertEqual(10, int(batch.burden_mask.sum()))
        self.assertTrue(np.all(batch.hourly_values[0, ~batch.hourly_mask[0], :] == 0.0))
        self.assertTrue(np.all(batch.tap_values[0, ~batch.tap_mask[0], :] == 0.0))

    def test_equal_time_rows_are_included_future_rows_excluded_and_order_is_old_to_new(self):
        samples = one_sample()
        tables = sequence_tables()
        builder = ThreeStreamSequenceBuilder().fit(samples, tables)
        batch = builder.transform(samples, tables)
        state = builder.state_dict()

        hourly_raw = inverse_channel(
            batch.hourly_values[0, -4:, 0],
            state["hourly"]["means"][0],
            state["hourly"]["scales"][0],
        )
        tap_raw = inverse_channel(
            batch.tap_values[0, -5:, 0],
            state["tap"]["means"][0],
            state["tap"]["scales"][0],
        )
        burden_raw = inverse_channel(
            batch.burden_values[0, :, 0],
            state["burden"]["means"][0],
            state["burden"]["scales"][0],
        )

        np.testing.assert_allclose([100.0, 1.0, 3.0, 5.0], hourly_raw, atol=1e-5)
        np.testing.assert_allclose([800.0, 900.0, 1000.0, 1100.0, 1200.0], tap_raw, atol=1e-4)
        np.testing.assert_allclose(np.arange(2.0, 12.0), burden_raw, atol=1e-5)
        tap_age = inverse_channel(
            batch.tap_values[0, -5:, 7],
            state["tap"]["means"][7],
            state["tap"]["scales"][7],
        )
        burden_age = inverse_channel(
            batch.burden_values[0, :, 5],
            state["burden"]["means"][5],
            state["burden"]["scales"][5],
        )
        np.testing.assert_allclose([4.0, 3.0, 2.0, 1.0, 0.0], tap_age, atol=1e-5)
        self.assertAlmostEqual(0.0, burden_age[-1], places=5)

    def test_validation_extreme_does_not_refit_sequence_state(self):
        samples = one_sample()
        tables = sequence_tables()
        builder = ThreeStreamSequenceBuilder().fit(samples, tables)
        before = builder.state_dict()
        extreme_hourly = tables.operation_hourly.copy()
        extreme_hourly.loc[extreme_hourly["clock"].eq(pd.Timestamp("2024-07-01 13:00:00")), "air_volume"] = 1e30
        extreme_tables = RawTables(
            train_samples=tables.train_samples,
            test_a_samples=tables.test_a_samples,
            test_b_samples=tables.test_b_samples,
            operation_hourly=extreme_hourly,
            burden_change=tables.burden_change,
            tap_history_train=tables.tap_history_train,
            deduplicated_rows=tables.deduplicated_rows,
        )
        validation = samples.copy()
        validation.loc[:, "sample_id"] = "VALID"
        validation.loc[:, "reference_time"] = pd.Timestamp("2024-07-01 13:00:00")

        result = builder.transform(validation, extreme_tables)

        self.assertEqual(before, builder.state_dict())
        self.assertTrue(np.isfinite(result.hourly_values).all())

    def test_null_reference_time_is_rejected(self):
        samples = one_sample()
        samples.loc[:, "reference_time"] = pd.NaT

        with self.assertRaisesRegex(ValueError, "null reference_time"):
            ThreeStreamSequenceBuilder().fit(samples, sequence_tables())


if __name__ == "__main__":
    unittest.main()
