from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_data_v4_timeaware import build_data_v4_timeaware
from src.reproducibility_timeaware import compare_timeaware_builds
from src.timeaware_preprocessing import (
    FoldTimeawareImputer,
    assemble_timeaware_raw,
)
from src.timeaware_contract import (
    G7_FEATURE_NAMES,
    G8_FEATURE_NAMES,
    load_timeaware_sources,
)
from src.validate_data_v4_timeaware import validate_data_v4_timeaware
from src.writers import write_snappy_parquet
from tests.fixtures import write_timeaware_monthly_source_tree


class TimeawareContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = write_timeaware_monthly_source_tree(self.root / "source")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_feature_registries_drive_exact_unique_model_schema(self):
        self.assertEqual(40, len(G7_FEATURE_NAMES))
        self.assertEqual(133, len(G8_FEATURE_NAMES))
        self.assertEqual(173, len(set((*G7_FEATURE_NAMES, *G8_FEATURE_NAMES))))
        self.assertTrue(all(name.startswith("g7_") for name in G7_FEATURE_NAMES))
        self.assertTrue(all(name.startswith("g8_") for name in G8_FEATURE_NAMES))

    def test_loader_preserves_raw_duplicate_burden_timestamps_for_last_row_rule(self):
        sources = load_timeaware_sources(self.source)

        duplicated = sources.burden_change_raw["cal_time"].duplicated(keep=False)

        self.assertEqual(2, int(duplicated.sum()))
        self.assertEqual(
            list(range(len(sources.burden_change_raw))),
            sources.burden_change_raw["_original_row"].tolist(),
        )

    def test_loader_rejects_invalid_sample_time_before_feature_generation(self):
        path = self.source / "train" / "train_samples.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "reference_time"] = "not-a-time"
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "invalid reference_time"):
            load_timeaware_sources(self.source)

    def test_loader_rejects_duplicate_sample_id_across_test_sets(self):
        test_a_path = self.source / "test" / "test_a_samples.csv"
        test_b_path = self.source / "test" / "test_b_samples.csv"
        test_a = pd.read_csv(test_a_path)
        test_b = pd.read_csv(test_b_path)
        test_b.loc[0, "sample_id"] = test_a.loc[0, "sample_id"]
        test_b.to_csv(test_b_path, index=False)

        with self.assertRaisesRegex(ValueError, "sample_id overlap"):
            load_timeaware_sources(self.source)

    def test_loader_rejects_nonfinite_training_target(self):
        path = self.source / "train" / "train_samples.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "tap_iron"] = float("nan")
        frame.to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "non-finite training target tap_iron"):
            load_timeaware_sources(self.source)

    def test_fold_imputer_uses_only_fit_data_and_returns_finite_float32(self):
        raw_train = pd.DataFrame(
            {
                "varying": [1.0, np.nan, 5.0],
                "constant": [7.0, 7.0, 7.0],
                "all_missing": [np.nan, np.inf, -np.inf],
            },
            index=[3, 4, 5],
        )
        imputer = FoldTimeawareImputer().fit(raw_train)
        before = imputer.state_dict()
        raw_valid = pd.DataFrame(
            {
                "varying": [1e30, np.nan],
                "constant": [np.nan, 7.0],
                "all_missing": [np.nan, 9.0],
            },
            index=[8, 9],
        )

        transformed = imputer.transform(raw_valid)

        self.assertEqual(before, imputer.state_dict())
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in transformed.dtypes))
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertEqual(3.0, transformed.loc[9, "varying"])
        self.assertEqual(7.0, transformed.loc[8, "constant"])
        self.assertEqual(0.0, transformed.loc[8, "all_missing"])
        self.assertIn("all_missing", before["all_missing_columns"])
        self.assertIn("constant", before["constant_columns"])

    def test_fold_imputer_rejects_unfitted_use_and_any_column_drift(self):
        raw = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            FoldTimeawareImputer().transform(raw)

        imputer = FoldTimeawareImputer().fit(raw)
        for drifted in (
            raw[["b", "a"]],
            raw[["a"]],
            raw.assign(c=5.0),
        ):
            with self.assertRaisesRegex(
                ValueError, "raw feature columns do not match fitted order"
            ):
                imputer.transform(drifted)

    def test_raw_assembly_is_exact_target_free_173_column_registry(self):
        sources = load_timeaware_sources(self.source)
        samples = sources.train_samples.iloc[[2, 4]].copy()

        raw = assemble_timeaware_raw(
            samples=samples,
            schedule_context=sources.train_samples,
            burden_events=sources.burden_change_raw,
        )

        self.assertEqual([*G7_FEATURE_NAMES, *G8_FEATURE_NAMES], list(raw.columns))
        self.assertEqual(list(samples.index), list(raw.index))
        self.assertFalse(any(name in raw.columns for name in ("tap_iron", "tap_time_len")))
        self.assertEqual(173, raw.shape[1])


class TimeawareBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = write_timeaware_monthly_source_tree(self.root / "source")
        self.output = self.root / "data_v4_timeaware"
        self.summary = build_data_v4_timeaware(self.source, self.output)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_has_exact_delivery_file_inventory_without_readme(self) -> None:
        expected = {
            *(f"folds/fold_{fold}_{split}.parquet" for fold in range(1, 6) for split in ("train", "valid")),
            "folds/full_train.parquet",
            "test_a.parquet",
            "test_b.parquet",
            "manifests/feature_dictionary.md",
            "manifests/build_config.json",
            "manifests/source_manifest.json",
            "manifests/leakage_audit.md",
            "reports/schedule_feature_stats.csv",
            "reports/burden_time_feature_stats.csv",
        }
        actual = {
            path.relative_to(self.output).as_posix()
            for path in self.output.rglob("*")
            if path.is_file()
        }

        self.assertEqual(expected, actual)
        self.assertFalse(any("readme" in path.lower() for path in actual))
        self.assertEqual(19, self.summary["file_count"])

    def test_folds_targets_schema_dtype_and_time_order_are_literal(self) -> None:
        fold_1_train = pd.read_parquet(self.output / "folds/fold_1_train.parquet")
        fold_1_valid = pd.read_parquet(self.output / "folds/fold_1_valid.parquet")
        fold_5_train = pd.read_parquet(self.output / "folds/fold_5_train.parquet")
        fold_5_valid = pd.read_parquet(self.output / "folds/fold_5_valid.parquet")
        test_a = pd.read_parquet(self.output / "test_a.parquet")

        self.assertEqual(
            {"2024-03", "2024-04", "2024-05", "2024-06"},
            set(fold_1_train["reference_time"].dt.strftime("%Y-%m")),
        )
        self.assertEqual({"2024-07"}, set(fold_1_valid["reference_time"].dt.strftime("%Y-%m")))
        self.assertEqual(set(f"2024-{month:02d}" for month in range(3, 11)), set(fold_5_train["reference_time"].dt.strftime("%Y-%m")))
        self.assertEqual({"2024-11"}, set(fold_5_valid["reference_time"].dt.strftime("%Y-%m")))
        self.assertLess(fold_1_train["reference_time"].max(), fold_1_valid["reference_time"].min())
        self.assertLess(fold_5_train["reference_time"].max(), fold_5_valid["reference_time"].min())

        front = ["sample_id", "reference_time", "tap_no", "spout_no", "fold", "split"]
        targets = ["tap_iron", "tap_time_len", "target_rate"]
        expected_train = [*front, *targets, *G7_FEATURE_NAMES, *G8_FEATURE_NAMES]
        expected_test = [*front, *G7_FEATURE_NAMES, *G8_FEATURE_NAMES]
        self.assertEqual(expected_train, list(fold_1_train.columns))
        self.assertEqual(expected_test, list(test_a.columns))
        self.assertFalse(any(target in test_a.columns for target in targets))
        for frame in (fold_1_train, fold_1_valid, fold_5_train, fold_5_valid, test_a):
            feature_frame = frame.loc[:, [*G7_FEATURE_NAMES, *G8_FEATURE_NAMES]]
            self.assertTrue(all(dtype == np.dtype("float32") for dtype in feature_frame.dtypes))
            self.assertTrue(np.isfinite(feature_frame.to_numpy()).all())

    def test_manifests_and_three_context_statistics_are_complete(self) -> None:
        config = json.loads(
            (self.output / "manifests/build_config.json").read_text("utf-8")
        )
        source_manifest = json.loads(
            (self.output / "manifests/source_manifest.json").read_text("utf-8")
        )
        schedule_stats = pd.read_csv(
            self.output / "reports/schedule_feature_stats.csv"
        )
        burden_stats = pd.read_csv(
            self.output / "reports/burden_time_feature_stats.csv"
        )

        self.assertEqual(40, config["feature_groups"]["G7_schedule"]["count"])
        self.assertEqual(133, config["feature_groups"]["G8_burden_time"]["count"])
        self.assertEqual([48, 2, 25], config["burden_grid"]["hours_step_points"])
        self.assertEqual(6, len(config["imputation_states"]))
        self.assertEqual(6, len(source_manifest["sources"]))
        self.assertEqual(4, len(source_manifest["legacy_data_v3_sentinels"]))
        self.assertEqual(3 * 40, len(schedule_stats))
        self.assertEqual(3 * 133, len(burden_stats))
        self.assertEqual({"full_train", "test_a", "test_b"}, set(schedule_stats["context"]))
        self.assertEqual({"full_train", "test_a", "test_b"}, set(burden_stats["context"]))
        self.assertIn(
            "pending independent validation",
            (self.output / "manifests/leakage_audit.md").read_text("utf-8"),
        )

    def test_independent_validator_passes_and_finalizes_leakage_audit(self) -> None:
        result = validate_data_v4_timeaware(
            self.source, self.output, enforce_official_counts=False
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual(0, result["g7_boundary_violations"])
        self.assertEqual(0, result["g8_boundary_violations"])
        self.assertGreater(result["independent_samples_checked"], 0)
        self.assertIn(
            "结论：通过",
            (self.output / "manifests/leakage_audit.md").read_text("utf-8"),
        )

    def test_validator_rejects_six_delivery_corruptions_by_named_check(self) -> None:
        feature_names = [*G7_FEATURE_NAMES, *G8_FEATURE_NAMES]
        corruptions: list[tuple[str, str, object]] = []

        schema_root = self._copy_output("bad_schema")
        path = schema_root / "folds/fold_1_train.parquet"
        frame = pd.read_parquet(path)
        columns = list(frame.columns)
        left = columns.index(feature_names[0])
        right = columns.index(feature_names[1])
        columns[left], columns[right] = columns[right], columns[left]
        write_snappy_parquet(frame.loc[:, columns], path)
        corruptions.append(("schema_order", "schema_order", schema_root))

        finite_root = self._copy_output("bad_finite")
        path = finite_root / "folds/fold_1_train.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, feature_names[0]] = np.nan
        write_snappy_parquet(frame, path)
        corruptions.append(("finite_features", "finite_features", finite_root))

        fold_root = self._copy_output("bad_fold")
        path = fold_root / "folds/fold_1_train.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "reference_time"] = pd.Timestamp("2024-07-20 00:00:00")
        write_snappy_parquet(frame, path)
        corruptions.append(("fold_membership", "fold_membership", fold_root))

        state_root = self._copy_output("bad_state")
        path = state_root / "manifests/build_config.json"
        config = json.loads(path.read_text("utf-8"))
        first_feature = feature_names[0]
        config["imputation_states"]["fold_1"]["medians"][first_feature] += 123.0
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", "utf-8")
        corruptions.append(("imputation_state", "imputation_state", state_root))

        source_root = self._copy_output("bad_source")
        path = source_root / "manifests/source_manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["sources"]["train_samples"]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
        corruptions.append(("source_hash", "source_hash", source_root))

        target_root = self._copy_output("bad_test_target")
        path = target_root / "test_a.parquet"
        frame = pd.read_parquet(path)
        frame.insert(6, "tap_iron", np.zeros(len(frame), dtype=np.float32))
        write_snappy_parquet(frame, path)
        corruptions.append(("test_target_rule", "test_target_rule", target_root))

        for label, expected_check, corrupt_root in corruptions:
            with self.subTest(corruption=label):
                with self.assertRaisesRegex(ValueError, expected_check):
                    validate_data_v4_timeaware(
                        self.source, corrupt_root, enforce_official_counts=False
                    )

    def test_reproducibility_comparison_detects_one_changed_parquet(self) -> None:
        second = self.root / "second_build"
        build_data_v4_timeaware(self.source, second)

        same = compare_timeaware_builds(self.output, second)
        self.assertEqual("ok", same["status"])
        self.assertEqual(19, same["file_count"])

        path = second / "test_b.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, G7_FEATURE_NAMES[-1]] += np.float32(1.0)
        write_snappy_parquet(frame, path)
        changed = compare_timeaware_builds(self.output, second)

        self.assertEqual("mismatch", changed["status"])
        self.assertIn("test_b.parquet", changed["mismatched_files"])

    def _copy_output(self, name: str) -> Path:
        destination = self.root / name
        shutil.copytree(self.output, destination)
        return destination


if __name__ == "__main__":
    unittest.main()
