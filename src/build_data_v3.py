from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_io import build_source_manifest, load_raw_tables
from src.drift import compute_drift_report
from src.feature_assembler import FeatureAssembler
from src.history_features import CausalBaselineBuilder
from src.preprocessing import FoldPreprocessor
from src.sequences import SequenceBatch, ThreeStreamSequenceBuilder
from src.splits import FoldDefinition, StrictExpandingMonthSplit, build_context_index
from src.teacher_script_audit import audit_teacher_script, render_teacher_audit_markdown
from src.writers import write_snappy_parquet


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ID_COLUMNS = ("sample_id", "fold", "split", "reference_time", "tap_no", "spout_no")
_TARGET_COLUMNS = ("tap_iron", "tap_time_len", "target_rate")


def build_data_v3(
    source_root: Path,
    output_root: Path,
    teacher_script: Path | None = None,
) -> dict[str, Any]:
    """Build the strict expanding-window, leakage-safe data_v3 delivery."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    for directory in ("folds", "sequences", "baselines", "manifests", "reports"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(source_root)
    folds = StrictExpandingMonthSplit().split(tables.train_samples)
    all_samples = pd.concat(
        [tables.train_samples, tables.test_a_samples, tables.test_b_samples],
        ignore_index=True,
        sort=False,
    )
    all_samples = _ordered(all_samples)
    raw_features = FeatureAssembler().build_raw(all_samples, tables)
    raw_features.index = all_samples["sample_id"].astype(str)
    feature_order: list[str] | None = None

    preprocessing_contexts: dict[str, Any] = {}
    baseline_parts: list[pd.DataFrame] = []
    sequence_parts: dict[str, list[np.ndarray]] = {
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

    train_lookup = tables.train_samples.set_index("sample_id", drop=False)
    for fold in folds:
        train_samples = _ordered(train_lookup.loc[list(fold.train_ids)].reset_index(drop=True))
        valid_samples = _ordered(train_lookup.loc[list(fold.valid_ids)].reset_index(drop=True))
        preprocessor = FoldPreprocessor().fit(_raw_for(train_samples, raw_features))
        sequence_builder = ThreeStreamSequenceBuilder().fit(train_samples, tables)
        baseline_builder = CausalBaselineBuilder().fit(train_samples, tables.tap_history_train)
        context = f"fold_{fold.fold}"
        preprocessing_contexts[context] = {
            "tabular": preprocessor.state_dict(),
            "sequences": sequence_builder.state_dict(),
            "baseline_fallback": dict(baseline_builder.fallback_),
        }
        for split, samples in (("train", train_samples), ("valid", valid_samples)):
            transformed = preprocessor.transform(_raw_for(samples, raw_features))
            feature_order = _check_feature_order(feature_order, transformed.columns.tolist())
            delivery = _make_delivery_frame(samples, transformed, fold.fold, split)
            write_snappy_parquet(
                delivery, output_root / "folds" / f"fold_{fold.fold}_{split}.parquet"
            )
            baselines = baseline_builder.transform(
                samples, tables.tap_history_train, fold.fold, split
            )
            baselines.insert(3, "reference_time", samples["reference_time"].to_numpy())
            baseline_parts.append(baselines.reset_index(drop=True))
            _append_batch(sequence_parts, sequence_builder.transform(samples, tables))

    full_train = _ordered(tables.train_samples)
    full_preprocessor = FoldPreprocessor().fit(_raw_for(full_train, raw_features))
    full_sequence_builder = ThreeStreamSequenceBuilder().fit(full_train, tables)
    full_baseline_builder = CausalBaselineBuilder().fit(full_train, tables.tap_history_train)
    preprocessing_contexts["full_train"] = {
        "tabular": full_preprocessor.state_dict(),
        "sequences": full_sequence_builder.state_dict(),
        "baseline_fallback": dict(full_baseline_builder.fallback_),
    }
    for context, split, samples in (
        ("full_train", "train", full_train),
        ("test_a", "test_a", _ordered(tables.test_a_samples)),
        ("test_b", "test_b", _ordered(tables.test_b_samples)),
    ):
        transformed = full_preprocessor.transform(_raw_for(samples, raw_features))
        feature_order = _check_feature_order(feature_order, transformed.columns.tolist())
        delivery = _make_delivery_frame(samples, transformed, 0, split)
        write_snappy_parquet(delivery, output_root / "folds" / f"{context}.parquet")
        baselines = full_baseline_builder.transform(samples, tables.tap_history_train, 0, split)
        baselines.insert(3, "reference_time", samples["reference_time"].to_numpy())
        baseline_parts.append(baselines.reset_index(drop=True))
        _append_batch(sequence_parts, full_sequence_builder.transform(samples, tables))

    sample_index = build_context_index(tables, folds)
    write_snappy_parquet(sample_index, output_root / "sample_index.parquet")
    _write_sequence_arrays(sequence_parts, sample_index, output_root / "sequences")
    write_snappy_parquet(
        pd.concat(baseline_parts, ignore_index=True),
        output_root / "baselines" / "causal_baselines.parquet",
    )
    _write_core_manifests(
        source_root,
        output_root,
        tables,
        folds,
        preprocessing_contexts,
        feature_order or [],
        teacher_script,
    )
    _write_diagnostics(output_root, feature_order or [], teacher_script)
    from src.validate_data_v3 import validate_data_v3

    validation = validate_data_v3(source_root, output_root)
    return {
        "status": "ok",
        "output_root": str(output_root),
        "folds": len(folds),
        "feature_dim": len(feature_order or []),
        "sequence_rows": len(sample_index),
        "validation": validation,
    }


def _ordered(samples: pd.DataFrame) -> pd.DataFrame:
    ordered = samples.sort_values(
        ["reference_time", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["reference_time"] = pd.to_datetime(ordered["reference_time"]).astype(
        "datetime64[ns]"
    )
    return ordered


def _raw_for(samples: pd.DataFrame, raw_features: pd.DataFrame) -> pd.DataFrame:
    identifiers = samples["sample_id"].astype(str).tolist()
    return raw_features.loc[identifiers].reset_index(drop=True)


def _make_delivery_frame(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    fold: int,
    split: str,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].astype(str).to_numpy(),
            "fold": np.full(len(samples), fold, dtype=np.int8),
            "split": np.full(len(samples), split, dtype=object),
            "reference_time": samples["reference_time"].to_numpy(dtype="datetime64[ns]"),
            "tap_no": samples["tap_no"].to_numpy(dtype=np.int64),
            "spout_no": samples["spout_no"].to_numpy(dtype=np.int64),
        }
    )
    if all(column in samples.columns for column in ("tap_iron", "tap_time_len")):
        iron = samples["tap_iron"].to_numpy(dtype=np.float32)
        duration = samples["tap_time_len"].to_numpy(dtype=np.float32)
        rate = np.divide(
            iron,
            duration,
            out=np.full(len(samples), np.nan, dtype=np.float32),
            where=np.abs(duration) >= 1e-6,
        )
        base["tap_iron"] = iron
        base["tap_time_len"] = duration
        base["target_rate"] = rate.astype(np.float32)
    return pd.concat([base, features.reset_index(drop=True)], axis=1)


def _check_feature_order(
    expected: list[str] | None, current: list[str]
) -> list[str]:
    if expected is None:
        return list(current)
    if expected != current:
        raise ValueError("feature order differs across delivery contexts")
    return expected


def _append_batch(parts: dict[str, list[np.ndarray]], batch: SequenceBatch) -> None:
    for name in parts:
        parts[name].append(getattr(batch, name))


def _write_sequence_arrays(
    parts: dict[str, list[np.ndarray]], sample_index: pd.DataFrame, destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, arrays in parts.items():
        combined = np.ascontiguousarray(np.concatenate(arrays, axis=0))
        if combined.shape[0] != len(sample_index):
            raise ValueError(f"{name} rows do not align with sample_index")
        if name.endswith("_values") and (
            combined.dtype != np.float32 or not np.isfinite(combined).all()
        ):
            raise ValueError(f"{name} must be finite float32")
        if name.endswith("_mask") and combined.dtype != bool:
            raise ValueError(f"{name} must use bool dtype")
        np.save(destination / f"{name}.npy", combined, allow_pickle=False)


def _write_core_manifests(
    source_root: Path,
    output_root: Path,
    tables: Any,
    folds: list[FoldDefinition],
    contexts: dict[str, Any],
    feature_order: list[str],
    teacher_script: Path | None,
) -> None:
    manifests = output_root / "manifests"
    _write_json(manifests / "source_manifest.json", build_source_manifest(source_root, tables))
    _write_json(
        manifests / "fold_definition.json",
        {
            "schema_version": "data_v3.strict_expanding_month.v1",
            "rule": "max(train_time) < min(valid_time)",
            "folds": [_fold_to_dict(fold) for fold in folds],
        },
    )
    _write_json(
        manifests / "preprocessing_state.json",
        {
            "schema_version": "data_v3.preprocessing_state.v1",
            "contexts": contexts,
        },
    )
    feature_lines = [
        "# 特征字典",
        "",
        f"固定特征维度：{len(feature_order)}。所有数值均按对应训练上下文独立拟合、填补并标准化。",
        "",
        "| 序号 | 字段 | 特征组 | 类型 |",
        "|---:|---|---|---|",
    ]
    for index, name in enumerate(feature_order, start=1):
        feature_lines.append(f"| {index} | `{name}` | {name.split('_', 1)[0].upper()} | float32 |")
    (manifests / "feature_dictionary.md").write_text(
        "\n".join(feature_lines) + "\n", encoding="utf-8"
    )
    development_log = manifests / "development_log.md"
    existing_log = _PROJECT_ROOT / "manifests" / "development_log.md"
    if existing_log.is_file() and existing_log.resolve() != development_log.resolve():
        shutil.copyfile(existing_log, development_log)
    elif not development_log.exists():
        development_log.write_text("# 开发记录\n\n构建流程已完成。\n", encoding="utf-8")
    requirements = _PROJECT_ROOT / "requirements.txt"
    if requirements.is_file() and requirements.resolve() != (output_root / "requirements.txt").resolve():
        shutil.copyfile(requirements, output_root / "requirements.txt")
    elif not (output_root / "requirements.txt").exists():
        (output_root / "requirements.txt").write_text("numpy\npandas\npyarrow\n", encoding="utf-8")
    teacher_note = str(Path(teacher_script).resolve()) if teacher_script else "未提供（not_provided）"
    (output_root / "README.md").write_text(
        "# 王玮 → 石朔冰：data_v3 数据交付\n\n"
        "本目录提供严格按月扩展窗口的 5 折训练/验证数据、防泄露特征、因果基线和三路时序张量。\n\n"
        "- `folds/`：5 折 train/valid、full_train、test_a、test_b 的 Snappy Parquet。\n"
        "- `sequences/`：小时、历史铁次、变料三路 float32 张量及 bool mask。\n"
        "- `sample_index.parquet`：张量行与 Parquet 样本的唯一映射。\n"
        "- `baselines/`：仅使用参考时刻可见历史计算的因果基线。\n"
        "- `manifests/`：数据来源、折定义、预处理状态、特征字典和开发记录。\n"
        "- `reports/`：漂移诊断、稳定性分析及教师脚本审计。\n\n"
        f"教师脚本：{teacher_note}。\n",
        encoding="utf-8",
    )


def _write_diagnostics(
    output_root: Path, feature_order: list[str], teacher_script: Path | None
) -> None:
    reports = output_root / "reports"
    full_train = pd.read_parquet(output_root / "folds" / "full_train.parquet")
    results = []
    for context in ("test_a", "test_b"):
        later = pd.read_parquet(output_root / "folds" / f"{context}.parquet")
        result = compute_drift_report(
            full_train.loc[:, feature_order],
            later.loc[:, feature_order],
            f"full_train_vs_{context}",
        )
        results.append(result)
    _write_json(
        reports / "adversarial_validation.json",
        {
            "schema_version": "data_v3.drift_bundle.v1",
            "comparisons": [result.to_dict() for result in results],
        },
    )
    lines = [
        "# 训练/测试漂移报告",
        "",
        "诊断只读取已经由全训练上下文拟合并转换的特征，不改变模型交付数据。",
        "",
        "| 比较 | 早期行数 | 后期行数 | OOF ROC AUC | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        auc = "不可计算" if result.adversarial_auc is None else f"{result.adversarial_auc:.6f}"
        lines.append(
            f"| {result.context} | {result.earlier_rows} | {result.later_rows} | {auc} | {result.adversarial['status']} |"
        )
        lines.extend(["", f"## {result.context} 重点漂移特征", ""])
        lines.extend(["| 特征 | 重要度 | PSI | 稳健标准化差异 |", "|---|---:|---:|---:|"])
        for item in result.feature_metrics[:20]:
            lines.append(
                f"| `{item['feature']}` | {item['adversarial_importance']:.6f} | "
                f"{item['psi']:.6f} | {item['robust_standardized_mean_difference']:.6f} |"
            )
    (reports / "drift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = audit_teacher_script(output_root.parent, teacher_script)
    (reports / "teacher_script_audit.md").write_text(
        render_teacher_audit_markdown(audit), encoding="utf-8"
    )


def _fold_to_dict(fold: FoldDefinition) -> dict[str, Any]:
    payload = asdict(fold)
    for name in ("train_start", "train_end", "valid_start", "valid_end"):
        payload[name] = pd.Timestamp(payload[name]).isoformat()
    payload["train_ids"] = list(payload["train_ids"])
    payload["valid_ids"] = list(payload["valid_ids"])
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建严格时序 data_v3 交付")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--teacher-script", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(build_data_v3(**vars(arguments)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
