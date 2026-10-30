from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_io import build_source_manifest, load_raw_tables
from src.feature_assembler import FeatureAssembler
from src.history_features import CausalBaselineBuilder
from src.preprocessing import FoldPreprocessor
from src.sequences import ThreeStreamSequenceBuilder
from src.splits import StrictExpandingMonthSplit, build_context_index


_FEATURE_PREFIXES = tuple(f"g{group}_" for group in range(7))


def validate_data_v3(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Independently rebuild critical contracts and reject a corrupt delivery."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    tables = load_raw_tables(source_root)
    folds = StrictExpandingMonthSplit().split(tables.train_samples)
    checks: list[str] = []

    source_manifest = _read_json(output_root / "manifests" / "source_manifest.json")
    expected_source = build_source_manifest(source_root, tables)
    if source_manifest.get("source_file_hashes") != expected_source["source_file_hashes"]:
        raise ValueError("source hash mismatch")
    checks.append("source_hashes_match_six_official_files")

    expected_index = build_context_index(tables, folds)
    actual_index = pd.read_parquet(output_root / "sample_index.parquet")
    if actual_index.duplicated(["data_context", "sample_id"]).any():
        raise ValueError("duplicate sample-index key")
    if actual_index["array_row"].tolist() != list(range(len(actual_index))):
        raise ValueError("sample-index array_row is not contiguous")
    _assert_frame_values_equal(
        actual_index,
        expected_index,
        [
            "array_row",
            "data_context",
            "fold",
            "split",
            "sample_id",
            "reference_time",
            "parquet_file",
            "parquet_row",
        ],
        "sample-index mapping",
    )
    checks.append("sample_index_exact_mapping")

    fold_manifest = _read_json(output_root / "manifests" / "fold_definition.json")
    if len(fold_manifest.get("folds", [])) != 5:
        raise ValueError("fold definition must contain five folds")
    feature_order = _validate_fold_parquets(output_root, tables, folds)
    checks.extend(
        [
            "fold_membership_and_strict_time_order",
            "parquet_snappy_compression",
            "fixed_feature_order",
            "finite_float32_features",
            "test_targets_absent",
        ]
    )

    all_samples = pd.concat(
        [tables.train_samples, tables.test_a_samples, tables.test_b_samples],
        ignore_index=True,
        sort=False,
    ).sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    raw = FeatureAssembler().build_raw(all_samples, tables)
    raw.index = all_samples["sample_id"].astype(str)
    states = _read_json(output_root / "manifests" / "preprocessing_state.json")["contexts"]
    train_lookup = tables.train_samples.set_index("sample_id", drop=False)
    for fold in folds:
        samples = _ordered(train_lookup.loc[list(fold.train_ids)].reset_index(drop=True))
        expected_state = FoldPreprocessor().fit(_raw_for(samples, raw)).state_dict()
        _assert_tabular_state(states[f"fold_{fold.fold}"]["tabular"], expected_state)
    full_train = _ordered(tables.train_samples)
    full_state = FoldPreprocessor().fit(_raw_for(full_train, raw)).state_dict()
    _assert_tabular_state(states["full_train"]["tabular"], full_state)
    if full_state["feature_names"] != feature_order:
        raise ValueError("feature order differs from full-train preprocessor state")
    checks.append("fold_local_preprocessor_states_refitted")

    _validate_sequences(output_root, tables, folds, actual_index)
    checks.extend(["sequence_shapes_and_dtypes", "sequence_content_recomputed"])
    _validate_baselines(output_root, tables, folds)
    checks.append("baseline_causality_recomputed")

    report_path = output_root / "manifests" / "leakage_audit.md"
    report_path.write_text(_render_leakage_report(folds, checks, feature_order), encoding="utf-8")
    return {
        "status": "ok",
        "check_count": len(checks),
        "checks": checks,
        "feature_dim": len(feature_order),
        "sample_index_rows": len(actual_index),
        "leakage_audit": str(report_path),
    }


def _validate_fold_parquets(output_root: Path, tables: Any, folds: list[Any]) -> list[str]:
    train_lookup = tables.train_samples.set_index("sample_id", drop=False)
    expected_feature_order: list[str] | None = None
    for fold in folds:
        expected_train = set(fold.train_ids)
        expected_valid = set(fold.valid_ids)
        for split, expected_ids in (("train", expected_train), ("valid", expected_valid)):
            path = output_root / "folds" / f"fold_{fold.fold}_{split}.parquet"
            frame = pd.read_parquet(path)
            if set(frame["sample_id"].astype(str)) != expected_ids:
                raise ValueError(f"fold membership mismatch in fold {fold.fold} {split}")
            _validate_parquet_file(path)
            current = [column for column in frame if column.startswith(_FEATURE_PREFIXES)]
            if expected_feature_order is None:
                expected_feature_order = current
            elif current != expected_feature_order:
                raise ValueError(f"feature order mismatch in {path.name}")
            if not np.isfinite(frame[current].to_numpy(dtype=np.float32)).all():
                raise ValueError(f"non-finite feature in {path.name}")
            if any(frame[column].dtype != np.float32 for column in current):
                raise ValueError(f"feature dtype is not float32 in {path.name}")
        train = train_lookup.loc[list(fold.train_ids)]
        valid = train_lookup.loc[list(fold.valid_ids)]
        if not pd.Timestamp(train["reference_time"].max()) < pd.Timestamp(valid["reference_time"].min()):
            raise ValueError(f"fold {fold.fold} violates strict time order")
    for name, samples, has_targets in (
        ("full_train", tables.train_samples, True),
        ("test_a", tables.test_a_samples, False),
        ("test_b", tables.test_b_samples, False),
    ):
        path = output_root / "folds" / f"{name}.parquet"
        frame = pd.read_parquet(path)
        if frame["sample_id"].astype(str).tolist() != _ordered(samples)["sample_id"].astype(str).tolist():
            raise ValueError(f"fold membership mismatch in {name}")
        _validate_parquet_file(path)
        current = [column for column in frame if column.startswith(_FEATURE_PREFIXES)]
        if current != expected_feature_order:
            raise ValueError(f"feature order mismatch in {path.name}")
        if not np.isfinite(frame[current].to_numpy(dtype=np.float32)).all():
            raise ValueError(f"non-finite feature in {path.name}")
        target_columns = {"tap_iron", "tap_time_len", "target_rate"}
        if has_targets and not target_columns.issubset(frame.columns):
            raise ValueError("full_train targets are missing")
        if not has_targets and not target_columns.isdisjoint(frame.columns):
            raise ValueError("test targets must be absent")
    return expected_feature_order or []


def _validate_parquet_file(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    try:
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                if group.column(column_index).compression != "SNAPPY":
                    raise ValueError(f"Parquet compression is not Snappy: {path.name}")
    finally:
        parquet.close()


def _assert_tabular_state(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in (
        "feature_names",
        "raw_feature_names",
        "indicator_columns",
        "all_missing_columns",
        "zero_variance_columns",
        "distance_columns",
    ):
        if actual.get(key) != expected[key]:
            raise ValueError(f"preprocessor state mismatch: {key}")
    for key in ("medians", "means", "scales"):
        if not np.allclose(actual.get(key), expected[key], rtol=0.0, atol=1e-12):
            raise ValueError(f"preprocessor state mismatch: {key}")
    for key in ("distance_mean", "distance_scale"):
        if not np.isclose(actual.get(key), expected[key], rtol=0.0, atol=1e-12):
            raise ValueError(f"preprocessor state mismatch: {key}")


def _validate_sequences(
    output_root: Path, tables: Any, folds: list[Any], actual_index: pd.DataFrame
) -> None:
    expected_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "hourly_values",
            "hourly_mask",
            "tap_values",
            "tap_mask",
            "burden_values",
            "burden_mask",
        )
    }
    lookup = tables.train_samples.set_index("sample_id", drop=False)
    for fold in folds:
        train = _ordered(lookup.loc[list(fold.train_ids)].reset_index(drop=True))
        valid = _ordered(lookup.loc[list(fold.valid_ids)].reset_index(drop=True))
        builder = ThreeStreamSequenceBuilder().fit(train, tables)
        for samples in (train, valid):
            _append_expected(expected_parts, builder.transform(samples, tables))
    full_train = _ordered(tables.train_samples)
    builder = ThreeStreamSequenceBuilder().fit(full_train, tables)
    for samples in (full_train, _ordered(tables.test_a_samples), _ordered(tables.test_b_samples)):
        _append_expected(expected_parts, builder.transform(samples, tables))
    expected_shapes = {
        "hourly_values": (len(actual_index), 24, 16),
        "hourly_mask": (len(actual_index), 24),
        "tap_values": (len(actual_index), 10, 8),
        "tap_mask": (len(actual_index), 10),
        "burden_values": (len(actual_index), 10, 6),
        "burden_mask": (len(actual_index), 10),
    }
    for name, parts in expected_parts.items():
        actual = np.load(output_root / "sequences" / f"{name}.npy", allow_pickle=False)
        expected = np.concatenate(parts, axis=0)
        if actual.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape mismatch")
        required_dtype = bool if name.endswith("_mask") else np.float32
        if actual.dtype != required_dtype:
            raise ValueError(f"{name} dtype mismatch")
        if not np.array_equal(actual, expected):
            raise ValueError(f"{name} content mismatch")


def _append_expected(parts: dict[str, list[np.ndarray]], batch: Any) -> None:
    for name in parts:
        parts[name].append(getattr(batch, name))


def _validate_baselines(output_root: Path, tables: Any, folds: list[Any]) -> None:
    expected_parts: list[pd.DataFrame] = []
    lookup = tables.train_samples.set_index("sample_id", drop=False)
    for fold in folds:
        train = _ordered(lookup.loc[list(fold.train_ids)].reset_index(drop=True))
        valid = _ordered(lookup.loc[list(fold.valid_ids)].reset_index(drop=True))
        builder = CausalBaselineBuilder().fit(train, tables.tap_history_train)
        for split, samples in (("train", train), ("valid", valid)):
            part = builder.transform(samples, tables.tap_history_train, fold.fold, split)
            part.insert(3, "reference_time", samples["reference_time"].to_numpy())
            expected_parts.append(part.reset_index(drop=True))
    full_train = _ordered(tables.train_samples)
    builder = CausalBaselineBuilder().fit(full_train, tables.tap_history_train)
    for split, samples in (
        ("train", full_train),
        ("test_a", _ordered(tables.test_a_samples)),
        ("test_b", _ordered(tables.test_b_samples)),
    ):
        part = builder.transform(samples, tables.tap_history_train, 0, split)
        part.insert(3, "reference_time", samples["reference_time"].to_numpy())
        expected_parts.append(part.reset_index(drop=True))
    expected = pd.concat(expected_parts, ignore_index=True)
    actual = pd.read_parquet(output_root / "baselines" / "causal_baselines.parquet")
    if actual.columns.tolist() != expected.columns.tolist() or len(actual) != len(expected):
        raise ValueError("baseline causality schema mismatch")
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(actual[column], expected[column], rtol=0.0, atol=1e-9):
                raise ValueError(f"baseline causality mismatch: {column}")
        elif actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"baseline causality mismatch: {column}")


def _render_leakage_report(folds: list[Any], checks: list[str], feature_order: list[str]) -> str:
    lines = [
        "# 防泄露独立验收报告",
        "",
        "结论：通过。验证器重新读取官方原始表并重算关键状态、基线与三路序列。",
        "",
        "| Fold | 训练结束 | 验证开始 | 严格早于 |",
        "|---:|---|---|---|",
    ]
    for fold in folds:
        lines.append(
            f"| {fold.fold} | {fold.train_end.isoformat()} | {fold.valid_start.isoformat()} | 是 |"
        )
    lines.extend(["", f"固定模型特征维度：{len(feature_order)}。", "", "已通过检查：", ""])
    lines.extend(f"- `{check}`" for check in checks)
    lines.extend(
        [
            "",
            "因果边界：历史铁次 `tap_end_time <= reference_time` 且排除当前 sample_id；",
            "小时表 `clock <= reference_time`；变料表 `cal_time <= reference_time`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)


def _raw_for(samples: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    return raw.loc[samples["sample_id"].astype(str).tolist()].reset_index(drop=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_frame_values_equal(
    actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str], label: str
) -> None:
    if actual.columns.tolist() != expected.columns.tolist() or len(actual) != len(expected):
        raise ValueError(f"{label} schema mismatch")
    for column in columns:
        if column == "reference_time":
            left = pd.to_datetime(actual[column]).to_numpy(dtype="datetime64[ns]")
            right = pd.to_datetime(expected[column]).to_numpy(dtype="datetime64[ns]")
            equal = np.array_equal(left, right)
        else:
            equal = actual[column].astype(str).tolist() == expected[column].astype(str).tolist()
        if not equal:
            raise ValueError(f"{label} mismatch: {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description="独立验证 data_v3 交付")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate_data_v3(**vars(arguments)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
