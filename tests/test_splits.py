from __future__ import annotations

import unittest

import pandas as pd

from src.data_io import RawTables
from src.splits import StrictExpandingMonthSplit, build_context_index


def samples_from_march_through_november() -> pd.DataFrame:
    times = pd.date_range("2024-03-01", periods=9, freq="MS") + pd.Timedelta(hours=2)
    return pd.DataFrame(
        {
            "sample_id": [f"S{month:02d}" for month in range(3, 12)],
            "tap_no": list(range(103, 112)),
            "spout_no": [1, 2, 3, 4, 1, 2, 3, 4, 1],
            "reference_time": times.astype("datetime64[ns]"),
            "tap_iron": [1000.0 + month for month in range(3, 12)],
            "tap_time_len": [120.0 + month for month in range(3, 12)],
        }
    )


def tables_for_context_index() -> RawTables:
    empty = pd.DataFrame()
    return RawTables(
        train_samples=samples_from_march_through_november(),
        test_a_samples=pd.DataFrame(
            {
                "sample_id": ["A01"],
                "tap_no": [201],
                "spout_no": [1],
                "reference_time": pd.to_datetime(["2024-12-01 02:00:00"]).astype("datetime64[ns]"),
            }
        ),
        test_b_samples=pd.DataFrame(
            {
                "sample_id": ["B01"],
                "tap_no": [301],
                "spout_no": [2],
                "reference_time": pd.to_datetime(["2025-02-01 02:00:00"]).astype("datetime64[ns]"),
            }
        ),
        operation_hourly=empty,
        burden_change=empty,
        tap_history_train=empty,
        deduplicated_rows={},
    )


class StrictExpandingSplitTests(unittest.TestCase):
    def test_strict_expanding_months_are_literal_contract(self):
        folds = StrictExpandingMonthSplit().split(samples_from_march_through_november())

        self.assertEqual(
            [
                (["2024-03", "2024-04", "2024-05", "2024-06"], "2024-07"),
                (["2024-03", "2024-04", "2024-05", "2024-06", "2024-07"], "2024-08"),
                (["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"], "2024-09"),
                (["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"], "2024-10"),
                (["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10"], "2024-11"),
            ],
            [(fold.train_months, fold.valid_month) for fold in folds],
        )
        for fold in folds:
            self.assertLess(fold.train_end, fold.valid_start)
            self.assertTrue(set(fold.train_ids).isdisjoint(fold.valid_ids))

    def test_missing_required_month_is_rejected(self):
        samples = samples_from_march_through_november()
        samples = samples[samples["reference_time"].dt.month != 9]

        with self.assertRaisesRegex(ValueError, "missing required training months: 2024-09"):
            StrictExpandingMonthSplit().split(samples)

    def test_training_month_outside_contract_is_rejected(self):
        samples = samples_from_march_through_november()
        outside = samples.iloc[[0]].copy()
        outside.loc[:, "sample_id"] = "S02"
        outside.loc[:, "reference_time"] = pd.Timestamp("2024-02-01 02:00:00")
        samples = pd.concat([samples, outside], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "unexpected training months: 2024-02"):
            StrictExpandingMonthSplit().split(samples)

    def test_context_index_order_and_array_rows_are_deterministic(self):
        tables = tables_for_context_index()
        folds = StrictExpandingMonthSplit().split(tables.train_samples)

        first = build_context_index(tables, folds)
        shuffled_tables = RawTables(
            train_samples=tables.train_samples.sample(frac=1.0, random_state=8),
            test_a_samples=tables.test_a_samples,
            test_b_samples=tables.test_b_samples,
            operation_hourly=tables.operation_hourly,
            burden_change=tables.burden_change,
            tap_history_train=tables.tap_history_train,
            deduplicated_rows=tables.deduplicated_rows,
        )
        second = build_context_index(shuffled_tables, StrictExpandingMonthSplit().split(shuffled_tables.train_samples))

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(list(range(46)), first["array_row"].tolist())
        self.assertEqual(
            [
                "fold_1_train",
                "fold_1_valid",
                "fold_2_train",
                "fold_2_valid",
                "fold_3_train",
                "fold_3_valid",
                "fold_4_train",
                "fold_4_valid",
                "fold_5_train",
                "fold_5_valid",
                "full_train",
                "test_a",
                "test_b",
            ],
            first["data_context"].drop_duplicates().tolist(),
        )
        self.assertEqual(46, len(first))


if __name__ == "__main__":
    unittest.main()
