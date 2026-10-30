from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow

from src.constants import RANDOM_SEED, SOURCE_FILES, TIMESTAMP_COLUMNS
from src.splits import StrictExpandingMonthSplit
from src.timeaware_contract import (
    G7_FEATURE_NAMES,
    G8_FEATURE_NAMES,
    TimeawareSources,
    load_timeaware_sources,
    make_schedule_contexts,
    ordered_samples,
)
from src.timeaware_preprocessing import FoldTimeawareImputer, assemble_timeaware_raw
from src.timeaware_reports import build_feature_stats, write_feature_dictionary
from src.writers import write_snappy_parquet


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FRONT_COLUMNS = ("sample_id", "reference_time", "tap_no", "spout_no", "fold", "split")
_TARGET_COLUMNS = ("tap_iron", "tap_time_len", "target_rate")
_LEGACY_SENTINELS = (
    Path("folds/full_train.parquet"),
    Path("folds/test_a.parquet"),
    Path("sample_index.parquet"),
    Path("manifests/source_manifest.json"),
)


def build_data_v4_timeaware(
    source_root: Path, output_root: Path
) -> dict[str, object]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    for directory in ("folds", "manifests", "reports"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    sources = load_timeaware_sources(source_root)
    folds = StrictExpandingMonthSplit().split(sources.train_samples)
    schedule_contexts = make_schedule_contexts(sources, folds)
    train_lookup = sources.train_samples.set_index("sample_id", drop=False)
    imputation_states: dict[str, dict[str, object]] = {}

    for fold in folds:
        train_samples = ordered_samples(
            train_lookup.loc[list(fold.train_ids)].reset_index(drop=True)
        )
        valid_samples = ordered_samples(
            train_lookup.loc[list(fold.valid_ids)].reset_index(drop=True)
        )
        context = schedule_contexts[f"fold_{fold.fold}"]
        raw_train = assemble_timeaware_raw(
            train_samples, context, sources.burden_change_raw
        )
        raw_valid = assemble_timeaware_raw(
            valid_samples, context, sources.burden_change_raw
        )
        imputer = FoldTimeawareImputer().fit(raw_train)
        imputation_states[f"fold_{fold.fold}"] = imputer.state_dict()
        for split, samples, raw in (
            ("train", train_samples, raw_train),
            ("valid", valid_samples, raw_valid),
        ):
            delivery = _make_delivery_frame(
                samples, imputer.transform(raw), fold.fold, split
            )
            write_snappy_parquet(
                delivery,
                output_root / "folds" / f"fold_{fold.fold}_{split}.parquet",
            )

    full_train = ordered_samples(sources.train_samples)
    test_a = ordered_samples(sources.test_a_samples)
    test_b = ordered_samples(sources.test_b_samples)
    raw_contexts = {
        "full_train": assemble_timeaware_raw(
            full_train, schedule_contexts["full_train"], sources.burden_change_raw
        ),
        "test_a": assemble_timeaware_raw(
            test_a, schedule_contexts["test_a"], sources.burden_change_raw
        ),
        "test_b": assemble_timeaware_raw(
            test_b, schedule_contexts["test_b"], sources.burden_change_raw
        ),
    }
    full_imputer = FoldTimeawareImputer().fit(raw_contexts["full_train"])
    imputation_states["full_train"] = full_imputer.state_dict()
    filled_contexts = {
        context: full_imputer.transform(raw)
        for context, raw in raw_contexts.items()
    }
    write_snappy_parquet(
        _make_delivery_frame(full_train, filled_contexts["full_train"], 0, "train"),
        output_root / "folds" / "full_train.parquet",
    )
    write_snappy_parquet(
        _make_delivery_frame(test_a, filled_contexts["test_a"], 0, "test_a"),
        output_root / "test_a.parquet",
    )
    write_snappy_parquet(
        _make_delivery_frame(test_b, filled_contexts["test_b"], 0, "test_b"),
        output_root / "test_b.parquet",
    )

    _write_manifests(
        source_root=source_root,
        output_root=output_root,
        sources=sources,
        folds=folds,
        imputation_states=imputation_states,
    )
    _write_reports(output_root, raw_contexts, filled_contexts)
    file_count = sum(path.is_file() for path in output_root.rglob("*"))
    return {
        "status": "ok",
        "output_root": str(output_root),
        "folds": len(folds),
        "feature_dim": len(G7_FEATURE_NAMES) + len(G8_FEATURE_NAMES),
        "train_rows": len(full_train),
        "test_a_rows": len(test_a),
        "test_b_rows": len(test_b),
        "file_count": file_count,
    }


def _make_delivery_frame(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    fold: int,
    split: str,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].astype(str).to_numpy(),
            "reference_time": samples["reference_time"].to_numpy(
                dtype="datetime64[ns]"
            ),
            "tap_no": samples["tap_no"].to_numpy(dtype=np.int64),
            "spout_no": samples["spout_no"].to_numpy(dtype=np.int64),
            "fold": np.full(len(samples), fold, dtype=np.int8),
            "split": np.full(len(samples), split, dtype=object),
        }
    )
    if split in {"train", "valid"}:
        iron = samples["tap_iron"].to_numpy(dtype=np.float32)
        duration = samples["tap_time_len"].to_numpy(dtype=np.float32)
        rate = np.divide(
            iron,
            duration,
            out=np.full(len(samples), np.nan, dtype=np.float32),
            where=np.abs(duration) >= 1e-6,
        )
        if not np.isfinite(rate).all():
            raise ValueError("target_rate contains non-finite values")
        base["tap_iron"] = iron
        base["tap_time_len"] = duration
        base["target_rate"] = rate.astype(np.float32)
    delivery = pd.concat([base, features.reset_index(drop=True)], axis=1)
    expected = [*_FRONT_COLUMNS]
    if split in {"train", "valid"}:
        expected.extend(_TARGET_COLUMNS)
    expected.extend((*G7_FEATURE_NAMES, *G8_FEATURE_NAMES))
    if list(delivery.columns) != expected:
        raise RuntimeError("delivery column order mismatch")
    return delivery


def _write_manifests(
    source_root: Path,
    output_root: Path,
    sources: TimeawareSources,
    folds: list[Any],
    imputation_states: dict[str, dict[str, object]],
) -> None:
    manifests = output_root / "manifests"
    write_feature_dictionary(
        manifests / "feature_dictionary.md", G7_FEATURE_NAMES, G8_FEATURE_NAMES
    )
    burden = sources.burden_change_raw
    duplicate_timestamp_count = int(
        (burden.groupby("cal_time", sort=False).size() > 1).sum()
    )
    duplicate_row_count = int(
        len(burden) - burden["cal_time"].nunique(dropna=False)
    )
    config = {
        "schema_version": "data_v4_timeaware.g7_g8.v1",
        "random_seed": RANDOM_SEED,
        "build_command": "py -3.14 -B -m src.build_data_v4_timeaware SOURCE_ROOT OUTPUT_ROOT",
        "feature_dim": len(G7_FEATURE_NAMES) + len(G8_FEATURE_NAMES),
        "feature_groups": {
            "G7_schedule": {"count": len(G7_FEATURE_NAMES), "columns": list(G7_FEATURE_NAMES)},
            "G8_burden_time": {"count": len(G8_FEATURE_NAMES), "columns": list(G8_FEATURE_NAMES)},
        },
        "schedule_boundaries": {
            "history": "reference_time strictly less than current reference_time",
            "count_windows": "[T-window, T)",
            "equal_time_samples_visible": False,
            "validation_public_times_can_update_later_validation": True,
            "test_a_context": "train + test_a",
            "test_b_context": "train + test_a + test_b",
        },
        "burden_grid": {
            "hours_step_points": [48, 2, 25],
            "event_visibility": "cal_time <= query_time",
            "ordering": "oldest_to_newest",
            "interpolation": "forward_hold",
            "duplicate_timestamp_rule": "stable sort by cal_time,_original_row; keep last",
            "duplicate_timestamp_count": duplicate_timestamp_count,
            "duplicate_row_count": duplicate_row_count,
        },
        "folds": [
            {
                "fold": fold.fold,
                "train_months": fold.train_months,
                "valid_month": fold.valid_month,
                "train_rows": len(fold.train_ids),
                "valid_rows": len(fold.valid_ids),
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "valid_start": fold.valid_start.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
            }
            for fold in folds
        ],
        "imputation_states": imputation_states,
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    _write_json(manifests / "build_config.json", config)
    _write_json(
        manifests / "source_manifest.json",
        _build_source_manifest(source_root),
    )
    (manifests / "leakage_audit.md").write_text(
        "# 泄漏审计\n\n状态：pending independent validation\n\n"
        "构建阶段已执行目标隔离和时间边界单元测试；独立交付验证将在下一阶段写入最终结论。\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_reports(
    output_root: Path,
    raw_contexts: dict[str, pd.DataFrame],
    filled_contexts: dict[str, pd.DataFrame],
) -> None:
    reports = output_root / "reports"
    build_feature_stats(
        raw_contexts, filled_contexts, G7_FEATURE_NAMES
    ).to_csv(reports / "schedule_feature_stats.csv", index=False, encoding="utf-8")
    build_feature_stats(
        raw_contexts, filled_contexts, G8_FEATURE_NAMES
    ).to_csv(reports / "burden_time_feature_stats.csv", index=False, encoding="utf-8")


def _build_source_manifest(source_root: Path) -> dict[str, object]:
    sources: dict[str, dict[str, object]] = {}
    for name, relative_path in SOURCE_FILES.items():
        path = source_root / relative_path
        frame = pd.read_csv(path)
        entry: dict[str, object] = {
            "relative_path": relative_path.as_posix(),
            "sha256": _sha256(path),
            "row_count": int(len(frame)),
        }
        ranges: dict[str, dict[str, str | None]] = {}
        for timestamp_column in TIMESTAMP_COLUMNS.get(name, ()):
            parsed = pd.to_datetime(
                frame[timestamp_column], errors="coerce", format="mixed"
            )
            ranges[timestamp_column] = {
                "min": parsed.min().isoformat() if parsed.notna().any() else None,
                "max": parsed.max().isoformat() if parsed.notna().any() else None,
            }
        entry["time_ranges"] = ranges
        sources[name] = entry
    sentinels: dict[str, dict[str, object]] = {}
    for relative_path in _LEGACY_SENTINELS:
        path = _PROJECT_ROOT / relative_path
        sentinels[relative_path.as_posix()] = {
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return {
        "schema_version": "data_v4_timeaware.source_manifest.v1",
        "sources": sources,
        "legacy_data_v3_sentinels": sentinels,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe G7/G8 data_v4_timeaware")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            build_data_v4_timeaware(args.source_root, args.output_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
