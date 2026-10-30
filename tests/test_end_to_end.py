from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_data_v3 import build_data_v3
from tests.fixtures import write_monthly_source_tree


class EndToEndBuildTests(unittest.TestCase):
    def test_build_generates_core_delivery_with_aligned_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_monthly_source_tree(root / "source")
            output = root / "delivery"

            summary = build_data_v3(source, output)

            self.assertEqual("ok", summary["status"])
            expected_fold_files = {
                *(f"fold_{fold}_{split}.parquet" for fold in range(1, 6) for split in ("train", "valid")),
                "full_train.parquet",
                "test_a.parquet",
                "test_b.parquet",
            }
            self.assertEqual(
                expected_fold_files,
                {path.name for path in (output / "folds").glob("*.parquet")},
            )
            for name in (
                "hourly_values.npy",
                "hourly_mask.npy",
                "tap_values.npy",
                "tap_mask.npy",
                "burden_values.npy",
                "burden_mask.npy",
            ):
                self.assertTrue((output / "sequences" / name).is_file(), name)
            for name in (
                "source_manifest.json",
                "fold_definition.json",
                "preprocessing_state.json",
                "feature_dictionary.md",
                "development_log.md",
                "leakage_audit.md",
            ):
                self.assertTrue((output / "manifests" / name).is_file(), name)
            self.assertTrue((output / "baselines" / "causal_baselines.parquet").is_file())
            self.assertTrue((output / "sample_index.parquet").is_file())
            self.assertTrue((output / "README.md").is_file())
            self.assertTrue((output / "requirements.txt").is_file())
            self.assertTrue((output / "reports" / "drift_report.md").is_file())
            self.assertTrue((output / "reports" / "adversarial_validation.json").is_file())
            self.assertTrue((output / "reports" / "teacher_script_audit.md").is_file())

            sample_index = pd.read_parquet(output / "sample_index.parquet")
            self.assertEqual(46, len(sample_index))
            self.assertEqual(list(range(46)), sample_index["array_row"].tolist())
            self.assertEqual(
                46,
                np.load(output / "sequences" / "hourly_values.npy", allow_pickle=False).shape[0],
            )
            self.assertEqual(
                46,
                np.load(output / "sequences" / "tap_values.npy", allow_pickle=False).shape[0],
            )
            self.assertEqual(
                46,
                np.load(output / "sequences" / "burden_values.npy", allow_pickle=False).shape[0],
            )

    def test_fold_membership_targets_and_feature_order_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_monthly_source_tree(root / "source")
            output = root / "delivery"
            build_data_v3(source, output)

            fold_1_train = pd.read_parquet(output / "folds" / "fold_1_train.parquet")
            fold_1_valid = pd.read_parquet(output / "folds" / "fold_1_valid.parquet")
            fold_5_train = pd.read_parquet(output / "folds" / "fold_5_train.parquet")
            fold_5_valid = pd.read_parquet(output / "folds" / "fold_5_valid.parquet")
            self.assertEqual(
                {"2024-03", "2024-04", "2024-05", "2024-06"},
                set(fold_1_train["reference_time"].dt.strftime("%Y-%m")),
            )
            self.assertEqual({"2024-07"}, set(fold_1_valid["reference_time"].dt.strftime("%Y-%m")))
            self.assertEqual(
                {f"2024-{month:02d}" for month in range(3, 11)},
                set(fold_5_train["reference_time"].dt.strftime("%Y-%m")),
            )
            self.assertEqual({"2024-11"}, set(fold_5_valid["reference_time"].dt.strftime("%Y-%m")))
            self.assertLess(fold_1_train["reference_time"].max(), fold_1_valid["reference_time"].min())
            feature_columns = [column for column in fold_1_train if column.startswith(tuple(f"g{i}_" for i in range(7)))]
            self.assertGreater(len(feature_columns), 288)
            self.assertEqual(feature_columns, [column for column in fold_5_train if column.startswith(tuple(f"g{i}_" for i in range(7)))])
            self.assertTrue(np.isfinite(fold_1_train[feature_columns].to_numpy(dtype=np.float32)).all())
            self.assertTrue({"tap_iron", "tap_time_len", "target_rate"}.issubset(fold_1_train.columns))
            for file_name in ("test_a.parquet", "test_b.parquet"):
                test_frame = pd.read_parquet(output / "folds" / file_name)
                self.assertTrue({"tap_iron", "tap_time_len", "target_rate"}.isdisjoint(test_frame.columns))

            definition = json.loads((output / "manifests" / "fold_definition.json").read_text(encoding="utf-8"))
            self.assertEqual("2024-07", definition["folds"][0]["valid_month"])
            self.assertEqual("2024-11", definition["folds"][4]["valid_month"])
            states = json.loads((output / "manifests" / "preprocessing_state.json").read_text(encoding="utf-8"))
            self.assertEqual(
                ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "full_train"],
                list(states["contexts"]),
            )


if __name__ == "__main__":
    unittest.main()
