from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.constants import BURDEN_CHANNELS, SOURCE_FILES
from src.data_io import resolve_source_root
from src.splits import FoldDefinition


PUBLIC_SAMPLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "tap_no",
    "spout_no",
    "reference_time",
)
TIMEAWARE_TARGET_COLUMNS: tuple[str, ...] = ("tap_iron", "tap_time_len")

_G7_NAMES: list[str] = []
_G7_NAMES.extend(f"g7_start_gap_lag_{lag}" for lag in (1, 2, 3, 4, 5, 6, 8, 10))
for statistic in ("mean", "median", "std"):
    _G7_NAMES.extend(f"g7_start_gap_{statistic}_{window}" for window in (3, 5, 10, 20))
for statistic in ("min", "max"):
    _G7_NAMES.extend(f"g7_start_gap_{statistic}_{window}" for window in (5, 10, 20))
_G7_NAMES.extend(f"g7_start_gap_trend_{window}" for window in (5, 10))
_G7_NAMES.append("g7_start_gap_current_minus_median_10")
_G7_NAMES.extend(f"g7_tap_count_{hours}h" for hours in (6, 12, 24, 48))
_G7_NAMES.append("g7_hours_since_same_spout_start")
_G7_NAMES.extend(f"g7_same_spout_start_gap_lag_{lag}" for lag in (1, 2, 3))
_G7_NAMES.extend(("g7_long_gap_flag_180m", "g7_long_gap_flag_240m"))
_G7_NAMES.append("g7_reference_minute_fraction")
G7_FEATURE_NAMES: tuple[str, ...] = tuple(_G7_NAMES)

_G8_NAMES: list[str] = []
for field in BURDEN_CHANNELS:
    prefix = f"g8_{field}_"
    _G8_NAMES.extend((f"{prefix}current_value", f"{prefix}hours_since_last_real_change"))
    _G8_NAMES.extend(f"{prefix}value_at_{hours}h" for hours in (2, 4, 8, 12, 24, 48))
    _G8_NAMES.extend(f"{prefix}current_minus_lag_{hours}h" for hours in (2, 8, 24))
    _G8_NAMES.extend(f"{prefix}time_weighted_mean_{hours}h" for hours in (6, 12, 24, 48))
    _G8_NAMES.extend(f"{prefix}real_time_slope_{hours}h" for hours in (12, 24, 48))
    _G8_NAMES.extend(f"{prefix}max_abs_change_{hours}h" for hours in (12, 24, 48))
    _G8_NAMES.extend(
        f"{prefix}effective_change_count_{hours}h" for hours in (6, 12, 24, 48)
    )
_G8_NAMES.extend(f"g8_raw_event_count_{hours}h" for hours in (6, 12, 24, 48))
for statistic in ("median", "std"):
    _G8_NAMES.extend(f"g8_event_gap_{statistic}_{hours}h" for hours in (24, 48))
G8_FEATURE_NAMES: tuple[str, ...] = tuple(_G8_NAMES)

assert len(G7_FEATURE_NAMES) == 40
assert len(G8_FEATURE_NAMES) == 133
assert len(set((*G7_FEATURE_NAMES, *G8_FEATURE_NAMES))) == 173


@dataclass(frozen=True)
class TimeawareSources:
    train_samples: pd.DataFrame
    test_a_samples: pd.DataFrame
    test_b_samples: pd.DataFrame
    burden_change_raw: pd.DataFrame
    source_root: Path


def load_timeaware_sources(path: Path) -> TimeawareSources:
    root = resolve_source_root(path)
    train = _read_samples(root / SOURCE_FILES["train_samples"], training=True)
    test_a = _read_samples(root / SOURCE_FILES["test_a_samples"], training=False)
    test_b = _read_samples(root / SOURCE_FILES["test_b_samples"], training=False)
    _validate_global_ids(train, test_a, test_b)
    burden = _read_raw_burden(root / SOURCE_FILES["burden_change"])
    return TimeawareSources(
        train_samples=ordered_samples(train),
        test_a_samples=ordered_samples(test_a),
        test_b_samples=ordered_samples(test_b),
        burden_change_raw=burden,
        source_root=root,
    )


def ordered_samples(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["reference_time", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)


def make_schedule_contexts(
    sources: TimeawareSources, folds: Sequence[FoldDefinition]
) -> dict[str, pd.DataFrame]:
    lookup = sources.train_samples.set_index("sample_id", drop=False)
    contexts: dict[str, pd.DataFrame] = {}
    for fold in folds:
        identifiers = [*fold.train_ids, *fold.valid_ids]
        contexts[f"fold_{fold.fold}"] = ordered_samples(
            lookup.loc[identifiers].reset_index(drop=True)
        )
    contexts["full_train"] = ordered_samples(sources.train_samples)
    contexts["test_a"] = ordered_samples(
        pd.concat(
            [sources.train_samples, sources.test_a_samples], ignore_index=True, sort=False
        )
    )
    contexts["test_b"] = ordered_samples(
        pd.concat(
            [sources.train_samples, sources.test_a_samples, sources.test_b_samples],
            ignore_index=True,
            sort=False,
        )
    )
    return contexts


def _read_samples(path: Path, training: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": "string"})
    required = [*PUBLIC_SAMPLE_COLUMNS]
    if training:
        required.extend(TIMEAWARE_TARGET_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, required].copy()
    frame["reference_time"] = pd.to_datetime(
        frame["reference_time"], errors="coerce", format="mixed"
    ).astype("datetime64[ns]")
    if frame["reference_time"].isna().any():
        raise ValueError(f"{path.name} contains invalid reference_time")
    frame["sample_id"] = frame["sample_id"].astype("string").str.strip()
    if frame["sample_id"].isna().any() or frame["sample_id"].eq("").any():
        raise ValueError(f"{path.name} contains null or empty sample_id")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate sample_id")
    for column in ("tap_no", "spout_no"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{path.name} contains invalid integer {column}")
        frame[column] = numeric.astype("int64")
    if training:
        for column in TIMEAWARE_TARGET_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
            if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"non-finite training target {column}")
    return frame


def _read_raw_burden(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["cal_time", *BURDEN_CHANNELS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, required].copy()
    frame["cal_time"] = pd.to_datetime(
        frame["cal_time"], errors="coerce", format="mixed"
    ).astype(
        "datetime64[ns]"
    )
    if frame["cal_time"].isna().any():
        raise ValueError(f"{path.name} contains invalid cal_time")
    for column in BURDEN_CHANNELS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["_original_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _validate_global_ids(
    train: pd.DataFrame, test_a: pd.DataFrame, test_b: pd.DataFrame
) -> None:
    sets = {
        "train": set(train["sample_id"].astype(str)),
        "test_a": set(test_a["sample_id"].astype(str)),
        "test_b": set(test_b["sample_id"].astype(str)),
    }
    names = tuple(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(
                    f"sample_id overlap between {left} and {right}: {sorted(overlap)[:3]}"
                )
