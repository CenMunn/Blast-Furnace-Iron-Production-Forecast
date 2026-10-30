from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_data_v3 import build_data_v3
from src.reproducibility_audit import compare_builds
from src.validate_data_v3 import validate_data_v3
from src.writers import write_snappy_parquet
from tests.fixtures import write_monthly_source_tree


class DeliveryValidationTests(unittest.TestCase):
    def _build(self, root: Path) -> tuple[Path, Path]:
        source = write_monthly_source_tree(root / "source")
        output = root / "delivery"
        build_data_v3(source, output)
        return source, output

    def test_valid_delivery_passes_and_writes_leakage_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            result = validate_data_v3(source, output)
            self.assertEqual("ok", result["status"])
            self.assertGreaterEqual(result["check_count"], 10)
            self.assertTrue((output / "manifests" / "leakage_audit.md").is_file())

    def test_fold_month_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            train_path = output / "folds" / "fold_1_train.parquet"
            train = pd.read_parquet(train_path)
            july = pd.read_parquet(output / "folds" / "fold_1_valid.parquet")
            write_snappy_parquet(pd.concat([train, july], ignore_index=True), train_path)
            with self.assertRaisesRegex(ValueError, "fold membership"):
                validate_data_v3(source, output)

    def test_feature_order_and_nonfinite_corruptions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "folds" / "fold_2_train.parquet"
            frame = pd.read_parquet(path)
            features = [name for name in frame if name.startswith("g")]
            columns = frame.columns.tolist()
            left, right = columns.index(features[0]), columns.index(features[1])
            columns[left], columns[right] = columns[right], columns[left]
            write_snappy_parquet(frame.loc[:, columns], path)
            with self.assertRaisesRegex(ValueError, "feature order"):
                validate_data_v3(source, output)

        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "folds" / "fold_2_train.parquet"
            frame = pd.read_parquet(path)
            feature = next(name for name in frame if name.startswith("g"))
            frame.loc[0, feature] = np.nan
            write_snappy_parquet(frame, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_data_v3(source, output)

    def test_index_mask_and_value_corruptions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "sample_index.parquet"
            index = pd.read_parquet(path)
            index.loc[1, "sample_id"] = index.loc[0, "sample_id"]
            index.loc[1, "data_context"] = index.loc[0, "data_context"]
            write_snappy_parquet(index, path)
            with self.assertRaisesRegex(ValueError, "duplicate sample-index"):
                validate_data_v3(source, output)

        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "sequences" / "hourly_mask.npy"
            mask = np.load(path, allow_pickle=False)
            np.save(path, mask[:, :-1], allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hourly_mask shape"):
                validate_data_v3(source, output)

        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "sequences" / "tap_values.npy"
            values = np.load(path, allow_pickle=False)
            values[0, -1, 0] += np.float32(0.25)
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "tap_values content"):
                validate_data_v3(source, output)

    def test_source_state_and_baseline_corruptions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "manifests" / "source_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            key = next(iter(manifest["source_file_hashes"]))
            manifest["source_file_hashes"][key] = "0" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash"):
                validate_data_v3(source, output)

        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "manifests" / "preprocessing_state.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["contexts"]["fold_1"]["tabular"]["medians"][0] += 1.0
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preprocessor state"):
                validate_data_v3(source, output)

        with tempfile.TemporaryDirectory() as temporary:
            source, output = self._build(Path(temporary))
            path = output / "baselines" / "causal_baselines.parquet"
            baselines = pd.read_parquet(path)
            baselines.loc[0, "baseline_prev_tap_iron"] = 999999.0
            write_snappy_parquet(baselines, path)
            with self.assertRaisesRegex(ValueError, "baseline causality"):
                validate_data_v3(source, output)

    def test_build_comparison_detects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_monthly_source_tree(root / "source")
            first = root / "first"
            second = root / "second"
            build_data_v3(source, first)
            build_data_v3(source, second)
            self.assertEqual("ok", compare_builds(first, second)["status"])
            path = second / "sequences" / "burden_values.npy"
            values = np.load(path, allow_pickle=False)
            values[0, -1, 0] += np.float32(1.0)
            np.save(path, values, allow_pickle=False)
            self.assertEqual("mismatch", compare_builds(first, second)["status"])


if __name__ == "__main__":
    unittest.main()
