from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_io import build_source_manifest, load_raw_tables, resolve_source_root
from tests.fixtures import (
    append_conflicting_hourly_row,
    append_exact_burden_duplicate,
    duplicate_first_train_row,
    write_source_tree,
)


class DataIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source_root = write_source_tree(Path(self.temp.name) / "official")

    def test_loader_parses_types_and_keeps_official_sample_ids(self):
        tables = load_raw_tables(self.source_root)

        self.assertEqual(["S0001", "S0002"], tables.train_samples["sample_id"].tolist())
        self.assertEqual("datetime64[ns]", str(tables.train_samples["reference_time"].dtype))
        self.assertEqual("datetime64[ns]", str(tables.tap_history_train["tap_end_time"].dtype))
        self.assertEqual("datetime64[ns]", str(tables.operation_hourly["clock"].dtype))
        self.assertEqual("datetime64[ns]", str(tables.burden_change["cal_time"].dtype))

    def test_resolve_source_root_accepts_parent_of_nested_official_root(self):
        parent = Path(self.temp.name) / "container"
        nested = write_source_tree(parent / "初赛数据集" / "初赛数据集")

        self.assertEqual(nested.resolve(), resolve_source_root(parent))

    def test_loader_rejects_duplicate_sample_id(self):
        duplicate_first_train_row(self.source_root)

        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            load_raw_tables(self.source_root)

    def test_loader_rejects_conflicting_hourly_timestamp(self):
        append_conflicting_hourly_row(self.source_root)

        with self.assertRaisesRegex(ValueError, "conflicting duplicate clock"):
            load_raw_tables(self.source_root)

    def test_loader_rejects_null_critical_timestamp(self):
        path = self.source_root / "train" / "train_samples.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "reference_time"] = None
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "null or invalid reference_time"):
            load_raw_tables(self.source_root)

    def test_loader_rejects_missing_published_column(self):
        path = self.source_root / "train" / "operation_hourly.csv"
        frame = pd.read_csv(path).drop(columns=["air_speed"])
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "missing required columns.*air_speed"):
            load_raw_tables(self.source_root)

    def test_loader_rejects_empty_training_target(self):
        path = self.source_root / "train" / "train_samples.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "tap_iron"] = None
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "null or non-finite training target tap_iron"):
            load_raw_tables(self.source_root)

    def test_loader_drops_only_exact_auxiliary_duplicates(self):
        append_exact_burden_duplicate(self.source_root)

        tables = load_raw_tables(self.source_root)

        self.assertEqual(2, len(tables.burden_change))
        self.assertEqual(1, tables.deduplicated_rows["burden_change"])

    def test_source_manifest_has_exact_six_sha256_keys(self):
        tables = load_raw_tables(self.source_root)

        manifest = build_source_manifest(self.source_root, tables)

        self.assertEqual(
            {
                "train/train_samples.csv",
                "train/operation_hourly.csv",
                "train/burden_change.csv",
                "train/tap_history_train.csv",
                "test/test_a_samples.csv",
                "test/test_b_samples.csv",
            },
            set(manifest["source_file_hashes"]),
        )
        for digest in manifest["source_file_hashes"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
