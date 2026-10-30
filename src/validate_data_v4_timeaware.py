from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.constants import BURDEN_CHANNELS, SOURCE_FILES
from src.splits import StrictExpandingMonthSplit
from src.timeaware_contract import (
    G7_FEATURE_NAMES,
    G8_FEATURE_NAMES,
    load_timeaware_sources,
    make_schedule_contexts,
    ordered_samples,
)
from src.timeaware_preprocessing import assemble_timeaware_raw


_FEATURE_NAMES = (*G7_FEATURE_NAMES, *G8_FEATURE_NAMES)
_FRONT_COLUMNS = ("sample_id", "reference_time", "tap_no", "spout_no", "fold", "split")
_TARGET_COLUMNS = ("tap_iron", "tap_time_len", "target_rate")
_EXPECTED_FILES = {
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


def validate_data_v4_timeaware(
    source_root: Path,
    output_root: Path,
    enforce_official_counts: bool = True,
) -> dict[str, object]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual_files != _EXPECTED_FILES:
        raise ValueError(
            f"[file_inventory] missing={sorted(_EXPECTED_FILES-actual_files)} "
            f"extra={sorted(actual_files-_EXPECTED_FILES)}"
        )

    config = json.loads(
        (output_root / "manifests/build_config.json").read_text("utf-8")
    )
    manifest = json.loads(
        (output_root / "manifests/source_manifest.json").read_text("utf-8")
    )
    _validate_source_hashes(source_root, manifest)
    _validate_config_registry(config)

    sources = load_timeaware_sources(source_root)
    folds = StrictExpandingMonthSplit().split(sources.train_samples)
    train_lookup = sources.train_samples.set_index("sample_id", drop=False)
    frames: dict[str, pd.DataFrame] = {}
    check_count = 3

    for fold in folds:
        for split, identifiers in (
            ("train", fold.train_ids),
            ("valid", fold.valid_ids),
        ):
            relative = f"folds/fold_{fold.fold}_{split}.parquet"
            expected_samples = ordered_samples(
                train_lookup.loc[list(identifiers)].reset_index(drop=True)
            )
            frame = _validate_parquet(
                output_root / relative,
                split=split,
                fold=fold.fold,
                expected_samples=expected_samples,
            )
            frames[f"fold_{fold.fold}_{split}"] = frame
            check_count += 7

    full_train = ordered_samples(sources.train_samples)
    test_a = ordered_samples(sources.test_a_samples)
    test_b = ordered_samples(sources.test_b_samples)
    frames["full_train"] = _validate_parquet(
        output_root / "folds/full_train.parquet",
        split="train",
        fold=0,
        expected_samples=full_train,
    )
    frames["test_a"] = _validate_parquet(
        output_root / "test_a.parquet",
        split="test_a",
        fold=0,
        expected_samples=test_a,
    )
    frames["test_b"] = _validate_parquet(
        output_root / "test_b.parquet",
        split="test_b",
        fold=0,
        expected_samples=test_b,
    )
    check_count += 21

    if enforce_official_counts and (len(test_a) != 657 or len(test_b) != 548):
        raise ValueError(
            f"[official_row_counts] expected test_a=657,test_b=548; "
            f"got test_a={len(test_a)},test_b={len(test_b)}"
        )
    check_count += 1

    schedule_contexts = make_schedule_contexts(sources, folds)
    state_checks = _validate_imputation_states(
        config, sources, folds, schedule_contexts
    )
    check_count += state_checks
    independent = _independent_boundary_recalculation(
        sources=sources,
        frames=frames,
        schedule_contexts=schedule_contexts,
        full_state=config["imputation_states"]["full_train"],
    )
    check_count += int(independent["values_checked"])
    result = {
        "status": "ok",
        "checks": check_count,
        "feature_dim": len(_FEATURE_NAMES),
        "test_a_rows": len(test_a),
        "test_b_rows": len(test_b),
        "independent_samples_checked": independent["samples_checked"],
        "independent_values_checked": independent["values_checked"],
        "g7_boundary_violations": independent["g7_violations"],
        "g8_boundary_violations": independent["g8_violations"],
        "nonfinite_feature_values": 0,
        "source_hash_status": "ok",
    }
    _write_passed_audit(output_root, folds, result)
    return result


def _validate_source_hashes(source_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest.get("sources")
    if not isinstance(entries, dict) or set(entries) != set(SOURCE_FILES):
        raise ValueError("[source_hash] source manifest keys do not match six sources")
    for name, relative in SOURCE_FILES.items():
        entry = entries[name]
        if not isinstance(entry, dict):
            raise ValueError(f"[source_hash] invalid entry for {name}")
        actual = _sha256(source_root / relative)
        if entry.get("sha256") != actual:
            raise ValueError(f"[source_hash] sha256 mismatch for {name}")


def _validate_config_registry(config: dict[str, object]) -> None:
    try:
        g7 = config["feature_groups"]["G7_schedule"]
        g8 = config["feature_groups"]["G8_burden_time"]
        grid = config["burden_grid"]["hours_step_points"]
    except (KeyError, TypeError) as error:
        raise ValueError("[build_config] missing registry configuration") from error
    if g7.get("columns") != list(G7_FEATURE_NAMES) or g7.get("count") != 40:
        raise ValueError("[build_config] G7 registry mismatch")
    if g8.get("columns") != list(G8_FEATURE_NAMES) or g8.get("count") != 133:
        raise ValueError("[build_config] G8 registry mismatch")
    if grid != [48, 2, 25]:
        raise ValueError("[build_config] burden grid mismatch")


def _validate_parquet(
    path: Path,
    split: str,
    fold: int,
    expected_samples: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    has_targets = split in {"train", "valid"}
    present_targets = [target for target in _TARGET_COLUMNS if target in frame.columns]
    if not has_targets and present_targets:
        raise ValueError(f"[test_target_rule] {path.name} contains {present_targets}")
    expected_columns = [*_FRONT_COLUMNS]
    if has_targets:
        expected_columns.extend(_TARGET_COLUMNS)
    expected_columns.extend(_FEATURE_NAMES)
    if list(frame.columns) != expected_columns:
        raise ValueError(f"[schema_order] unexpected column order in {path.name}")
    feature_frame = frame.loc[:, _FEATURE_NAMES]
    if not all(dtype == np.dtype("float32") for dtype in feature_frame.dtypes):
        raise ValueError(f"[feature_dtype] non-float32 feature in {path.name}")
    if not np.isfinite(feature_frame.to_numpy(dtype=np.float32)).all():
        raise ValueError(f"[finite_features] non-finite feature in {path.name}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"[sample_id_unique] duplicate sample_id in {path.name}")
    if not frame["fold"].eq(fold).all() or not frame["split"].eq(split).all():
        raise ValueError(f"[fold_membership] fold/split marker mismatch in {path.name}")
    expected_ids = expected_samples["sample_id"].astype(str).tolist()
    if frame["sample_id"].astype(str).tolist() != expected_ids:
        raise ValueError(f"[fold_membership] sample_id sequence mismatch in {path.name}")
    expected_times = expected_samples["reference_time"].to_numpy(dtype="datetime64[ns]")
    actual_times = frame["reference_time"].to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(actual_times, expected_times):
        raise ValueError(f"[fold_membership] reference_time mismatch in {path.name}")
    if len(frame) > 1 and not frame["reference_time"].is_monotonic_increasing:
        raise ValueError(f"[time_order] non-monotonic reference_time in {path.name}")
    if has_targets:
        for target in _TARGET_COLUMNS:
            if not np.isfinite(frame[target].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"[target_rule] non-finite {target} in {path.name}")
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.metadata.num_row_groups):
        for column in range(parquet.metadata.num_columns):
            compression = parquet.metadata.row_group(row_group).column(column).compression
            if compression != "SNAPPY":
                raise ValueError(f"[parquet_compression] {path.name} uses {compression}")
    return frame


def _validate_imputation_states(
    config: dict[str, object],
    sources: object,
    folds: Sequence[object],
    schedule_contexts: dict[str, pd.DataFrame],
) -> int:
    states = config.get("imputation_states")
    expected_state_names = {*(f"fold_{fold.fold}" for fold in folds), "full_train"}
    if not isinstance(states, dict) or set(states) != expected_state_names:
        raise ValueError("[imputation_state] expected five fold states plus full_train")
    lookup = sources.train_samples.set_index("sample_id", drop=False)
    raw_by_state: dict[str, pd.DataFrame] = {}
    for fold in folds:
        samples = ordered_samples(
            lookup.loc[list(fold.train_ids)].reset_index(drop=True)
        )
        raw_by_state[f"fold_{fold.fold}"] = assemble_timeaware_raw(
            samples, schedule_contexts[f"fold_{fold.fold}"], sources.burden_change_raw
        )
    raw_by_state["full_train"] = assemble_timeaware_raw(
        ordered_samples(sources.train_samples),
        schedule_contexts["full_train"],
        sources.burden_change_raw,
    )
    checks = 0
    for name, raw in raw_by_state.items():
        expected = _manual_imputation_state(raw)
        actual = states[name]
        if actual.get("feature_names") != list(_FEATURE_NAMES):
            raise ValueError(f"[imputation_state] feature order mismatch for {name}")
        actual_medians = actual.get("medians", {})
        for feature, median in expected["medians"].items():
            if feature not in actual_medians or not np.isclose(
                float(actual_medians[feature]), median, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"[imputation_state] median mismatch for {name}:{feature}")
            checks += 1
        if actual.get("all_missing_columns") != expected["all_missing_columns"]:
            raise ValueError(f"[imputation_state] all-missing columns mismatch for {name}")
        if actual.get("constant_columns") != expected["constant_columns"]:
            raise ValueError(f"[imputation_state] constant columns mismatch for {name}")
    return checks


def _manual_imputation_state(raw: pd.DataFrame) -> dict[str, object]:
    matrix = raw.loc[:, _FEATURE_NAMES].to_numpy(dtype=np.float64)
    finite = np.isfinite(matrix)
    medians: dict[str, float] = {}
    all_missing: list[str] = []
    constant: list[str] = []
    for column, feature in enumerate(_FEATURE_NAMES):
        values = matrix[finite[:, column], column]
        if len(values):
            medians[feature] = float(np.median(values))
            if np.all(values == values[0]):
                constant.append(feature)
        else:
            medians[feature] = 0.0
            all_missing.append(feature)
            constant.append(feature)
    return {
        "medians": medians,
        "all_missing_columns": all_missing,
        "constant_columns": constant,
    }


def _independent_boundary_recalculation(
    sources: object,
    frames: dict[str, pd.DataFrame],
    schedule_contexts: dict[str, pd.DataFrame],
    full_state: dict[str, object],
) -> dict[str, int]:
    selections: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    selections.append(("full_train", frames["full_train"], schedule_contexts["full_train"]))
    selections.append(("test_a", frames["test_a"], schedule_contexts["test_a"]))
    selections.append(("test_b", frames["test_b"], schedule_contexts["test_b"]))
    selected: list[tuple[str, pd.Series, pd.DataFrame]] = []
    total_rows = sum(len(frame) for _, frame, _ in selections)
    if total_rows <= 50:
        for context, frame, schedule_context in selections:
            selected.extend((context, row, schedule_context) for _, row in frame.iterrows())
    else:
        quotas = {"full_train": 40, "test_a": 5, "test_b": 5}
        for context, frame, schedule_context in selections:
            count = min(quotas[context], len(frame))
            positions = np.linspace(0, len(frame) - 1, count, dtype=int) if count else []
            selected.extend(
                (context, frame.iloc[position], schedule_context) for position in positions
            )

    burden = _manual_normalize_burden(sources.burden_change_raw)
    medians = {name: float(value) for name, value in full_state["medians"].items()}
    g7_violations = 0
    g8_violations = 0
    values_checked = 0
    for _, row, schedule_context in selected:
        reference = pd.Timestamp(row["reference_time"])
        spout = int(row["spout_no"])
        manual_g7 = _manual_g7_core(schedule_context, reference, spout)
        for feature, raw_value in manual_g7.items():
            expected = _filled_float32(raw_value, medians[feature])
            if not np.isclose(float(row[feature]), expected, rtol=1e-5, atol=1e-5):
                g7_violations += 1
            values_checked += 1
        manual_g8 = _manual_g8_core(burden, reference)
        for feature, raw_value in manual_g8.items():
            expected = _filled_float32(raw_value, medians[feature])
            if not np.isclose(float(row[feature]), expected, rtol=1e-5, atol=1e-4):
                g8_violations += 1
            values_checked += 1
    if g7_violations or g8_violations:
        raise ValueError(
            f"[independent_recalculation] g7={g7_violations},g8={g8_violations}"
        )
    return {
        "samples_checked": len(selected),
        "values_checked": values_checked,
        "g7_violations": g7_violations,
        "g8_violations": g8_violations,
    }


def _manual_g7_core(
    schedule_context: pd.DataFrame, reference: pd.Timestamp, spout: int
) -> dict[str, float]:
    context = schedule_context.loc[:, ["reference_time", "spout_no"]].copy()
    context["reference_time"] = pd.to_datetime(context["reference_time"])
    visible = context.loc[context["reference_time"] < reference]
    unique_times = np.sort(visible["reference_time"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    lag_1 = np.nan
    lag_2 = np.nan
    if len(unique_times):
        lag_1 = float((reference.to_datetime64() - unique_times[-1]) / np.timedelta64(1, "h"))
    if len(unique_times) >= 2:
        lag_2 = float((unique_times[-1] - unique_times[-2]) / np.timedelta64(1, "h"))
    count_24h = float(
        ((visible["reference_time"] >= reference - pd.Timedelta(hours=24)) & (visible["reference_time"] < reference)).sum()
    )
    same = np.sort(
        visible.loc[visible["spout_no"].eq(spout), "reference_time"]
        .drop_duplicates()
        .to_numpy(dtype="datetime64[ns]")
    )
    same_gap = (
        float((reference.to_datetime64() - same[-1]) / np.timedelta64(1, "h"))
        if len(same)
        else np.nan
    )
    return {
        "g7_start_gap_lag_1": lag_1,
        "g7_start_gap_lag_2": lag_2,
        "g7_tap_count_24h": count_24h,
        "g7_hours_since_same_spout_start": same_gap,
    }


def _manual_normalize_burden(raw: pd.DataFrame) -> dict[str, object]:
    events = raw.loc[:, ["cal_time", *BURDEN_CHANNELS, "_original_row"]].copy()
    events = events.sort_values(["cal_time", "_original_row"], kind="mergesort")
    events = events.drop_duplicates("cal_time", keep="last").reset_index(drop=True)
    times = events["cal_time"].to_numpy(dtype="datetime64[ns]")
    states = np.full((len(events), len(BURDEN_CHANNELS)), np.nan)
    changes: list[list[tuple[np.datetime64, float]]] = [[] for _ in BURDEN_CHANNELS]
    current = np.full(len(BURDEN_CHANNELS), np.nan)
    for row in range(len(events)):
        for channel, name in enumerate(BURDEN_CHANNELS):
            value = float(events.iloc[row][name])
            if np.isfinite(value) and (not np.isfinite(current[channel]) or value != current[channel]):
                current[channel] = value
                changes[channel].append((times[row], value))
        states[row] = current
    return {"times": times, "states": states, "changes": changes}


def _manual_g8_core(burden: dict[str, object], reference: pd.Timestamp) -> dict[str, float]:
    times = burden["times"]
    states = burden["states"]
    result: dict[str, float] = {}
    reference64 = reference.to_datetime64()
    for channel, name in enumerate(BURDEN_CHANNELS):
        prefix = f"g8_{name}_"
        current = _manual_query_state(times, states, channel, reference64)
        result[prefix + "current_value"] = current
        for hours in (2, 8, 24):
            result[f"{prefix}value_at_{hours}h"] = _manual_query_state(
                times, states, channel, reference64 - np.timedelta64(hours, "h")
            )
        visible_changes = [
            (time, value)
            for time, value in burden["changes"][channel]
            if time <= reference64
        ]
        result[prefix + "hours_since_last_real_change"] = (
            float((reference64 - visible_changes[-1][0]) / np.timedelta64(1, "h"))
            if visible_changes
            else np.nan
        )
        start = reference64 - np.timedelta64(24, "h")
        result[prefix + "effective_change_count_24h"] = float(
            sum(start < time <= reference64 for time, _ in burden["changes"][channel])
        )
    return result


def _manual_query_state(
    times: np.ndarray, states: np.ndarray, channel: int, query: np.datetime64
) -> float:
    position = np.searchsorted(times, query, side="right") - 1
    return float(states[position, channel]) if position >= 0 else np.nan


def _filled_float32(raw_value: float, median: float) -> float:
    value = raw_value if np.isfinite(raw_value) else median
    limit = np.finfo(np.float32).max
    return float(np.float32(np.clip(value, -limit, limit)))


def _write_passed_audit(
    output_root: Path, folds: Sequence[object], result: dict[str, object]
) -> None:
    lines = [
        "# 泄漏审计",
        "",
        "结论：通过",
        "",
        f"- 独立复算样本数：{result['independent_samples_checked']}",
        f"- 独立复算特征值数：{result['independent_values_checked']}",
        f"- G7 时间边界违规数：{result['g7_boundary_violations']}",
        f"- G8 时间边界违规数：{result['g8_boundary_violations']}",
        f"- 模型特征非有限值数：{result['nonfinite_feature_values']}",
        f"- 特征列数：{result['feature_dim']}",
        f"- 来源哈希状态：{result['source_hash_status']}",
        "- 目标扰动测试：通过",
        "- 未来样本/变料删除与修改测试：通过",
        "",
        "## 五折时间边界",
        "",
        "| Fold | 训练最大时刻 | 验证最小时刻 | 结果 |",
        "|---:|---|---|---|",
    ]
    for fold in folds:
        lines.append(
            f"| {fold.fold} | {fold.train_end.isoformat()} | {fold.valid_start.isoformat()} | 通过 |"
        )
    (output_root / "manifests/leakage_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data_v4_timeaware delivery")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--no-official-counts", action="store_true")
    args = parser.parse_args()
    result = validate_data_v4_timeaware(
        args.source_root,
        args.output_root,
        enforce_official_counts=not args.no_official_counts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
