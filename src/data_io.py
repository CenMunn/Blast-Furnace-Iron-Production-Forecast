from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.constants import (
    BURDEN_CHANNELS,
    HOURLY_CHANNELS,
    REQUIRED_COLUMNS,
    SOURCE_FILES,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMNS,
)


@dataclass(frozen=True)
class RawTables:
    train_samples: pd.DataFrame
    test_a_samples: pd.DataFrame
    test_b_samples: pd.DataFrame
    operation_hourly: pd.DataFrame
    burden_change: pd.DataFrame
    tap_history_train: pd.DataFrame
    deduplicated_rows: Mapping[str, int]


def resolve_source_root(path: Path) -> Path:
    base = Path(path).expanduser().resolve()
    candidates = (
        base,
        base / "初赛数据集",
        base / "初赛数据集" / "初赛数据集",
    )
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in SOURCE_FILES.values()):
            return candidate
    required = ", ".join(str(value) for value in SOURCE_FILES.values())
    raise FileNotFoundError(f"could not resolve official source root under {base}; required: {required}")


def load_raw_tables(path: Path) -> RawTables:
    root = resolve_source_root(path)
    frames: dict[str, pd.DataFrame] = {}
    deduplicated_rows: dict[str, int] = {}

    for name, relative in SOURCE_FILES.items():
        frame = pd.read_csv(root / relative, dtype={"sample_id": "string"})
        _validate_required_columns(name, frame)
        frame = frame.loc[:, list(REQUIRED_COLUMNS[name])].copy()
        _parse_timestamps(name, frame)
        _normalize_identifier_columns(name, frame)
        _normalize_numeric_columns(name, frame)
        before = len(frame)
        if name in {"operation_hourly", "burden_change", "tap_history_train"}:
            frame = frame.drop_duplicates(keep="first").copy()
        deduplicated_rows[name] = before - len(frame)
        _validate_business_keys(name, frame)
        frames[name] = _stable_sort(name, frame)

    _validate_global_sample_ids(frames)
    _validate_training_targets(frames["train_samples"])
    return RawTables(
        train_samples=frames["train_samples"],
        test_a_samples=frames["test_a_samples"],
        test_b_samples=frames["test_b_samples"],
        operation_hourly=frames["operation_hourly"],
        burden_change=frames["burden_change"],
        tap_history_train=frames["tap_history_train"],
        deduplicated_rows=deduplicated_rows,
    )


def build_source_manifest(root: Path, tables: RawTables) -> dict[str, object]:
    source_root = resolve_source_root(root)
    hashes = {
        relative.as_posix(): _sha256(source_root / relative)
        for relative in SOURCE_FILES.values()
    }
    row_counts = {
        name: int(len(getattr(tables, name)))
        for name in SOURCE_FILES
    }
    time_ranges: dict[str, dict[str, str]] = {}
    for name, columns in TIMESTAMP_COLUMNS.items():
        frame = getattr(tables, name)
        for column in columns:
            values = frame[column]
            time_ranges[f"{name}.{column}"] = {
                "min": pd.Timestamp(values.min()).isoformat(),
                "max": pd.Timestamp(values.max()).isoformat(),
            }
    return {
        "schema_version": "data_v3.source_manifest.v1",
        "source_root": str(source_root),
        "source_file_hashes": hashes,
        "row_counts_after_exact_deduplication": row_counts,
        "deduplicated_rows": dict(tables.deduplicated_rows),
        "time_ranges": time_ranges,
    }


def _validate_required_columns(name: str, frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS[name] if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _parse_timestamps(name: str, frame: pd.DataFrame) -> None:
    for column in TIMESTAMP_COLUMNS[name]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").astype(
            "datetime64[ns]"
        )
        if frame[column].isna().any():
            raise ValueError(f"{name} contains null or invalid {column}")


def _normalize_identifier_columns(name: str, frame: pd.DataFrame) -> None:
    if "sample_id" in frame:
        frame["sample_id"] = frame["sample_id"].astype("string").str.strip()
        if frame["sample_id"].isna().any() or frame["sample_id"].eq("").any():
            raise ValueError(f"{name} contains null or empty sample_id")
    for column in ("tap_no", "spout_no"):
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError(f"{name} contains invalid integer {column}")
            frame[column] = numeric.astype("int64")


def _normalize_numeric_columns(name: str, frame: pd.DataFrame) -> None:
    excluded = {"sample_id", *TIMESTAMP_COLUMNS[name], "tap_no", "spout_no"}
    for column in frame.columns:
        if column not in excluded:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _validate_business_keys(name: str, frame: pd.DataFrame) -> None:
    key_by_table = {
        "train_samples": "sample_id",
        "test_a_samples": "sample_id",
        "test_b_samples": "sample_id",
        "tap_history_train": "sample_id",
        "operation_hourly": "clock",
        "burden_change": "cal_time",
    }
    key = key_by_table[name]
    duplicated = frame[key].duplicated(keep=False)
    if duplicated.any():
        if key in {"clock", "cal_time"}:
            raise ValueError(f"{name} contains conflicting duplicate {key}")
        raise ValueError(f"{name} contains duplicate sample_id")


def _validate_global_sample_ids(frames: Mapping[str, pd.DataFrame]) -> None:
    sample_sets = {
        name: set(frames[name]["sample_id"].astype(str))
        for name in ("train_samples", "test_a_samples", "test_b_samples")
    }
    names = tuple(sample_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sample_sets[left] & sample_sets[right]
            if overlap:
                raise ValueError(f"sample_id overlap between {left} and {right}: {sorted(overlap)[:3]}")


def _validate_training_targets(frame: pd.DataFrame) -> None:
    for column in TARGET_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"null or non-finite training target {column}")


def _stable_sort(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = {
        "train_samples": ["reference_time", "sample_id"],
        "test_a_samples": ["reference_time", "sample_id"],
        "test_b_samples": ["reference_time", "sample_id"],
        "operation_hourly": ["clock"],
        "burden_change": ["cal_time"],
        "tap_history_train": ["tap_end_time", "sample_id"],
    }[name]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
