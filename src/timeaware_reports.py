from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def build_feature_stats(
    raw_by_context: dict[str, pd.DataFrame],
    filled_by_context: dict[str, pd.DataFrame],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    train = filled_by_context["full_train"].loc[:, feature_names]
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0, ddof=0)
    rows: list[dict[str, object]] = []
    for context in ("full_train", "test_a", "test_b"):
        raw = raw_by_context[context].loc[:, feature_names]
        filled = filled_by_context[context].loc[:, feature_names]
        for feature in feature_names:
            values = filled[feature].to_numpy(dtype=np.float64)
            standard_deviation = float(train_std[feature])
            mean_shift = (
                float((np.mean(values) - train_mean[feature]) / standard_deviation)
                if standard_deviation > 0.0
                else 0.0
            )
            rows.append(
                {
                    "context": context,
                    "feature": feature,
                    "row_count": int(len(filled)),
                    "raw_missing_rate": float(
                        (~np.isfinite(raw[feature].to_numpy(dtype=np.float64))).mean()
                    ),
                    "filled_mean": float(np.mean(values)),
                    "filled_std": float(np.std(values, ddof=0)),
                    "filled_min": float(np.min(values)),
                    "filled_p05": float(np.quantile(values, 0.05)),
                    "filled_median": float(np.median(values)),
                    "filled_p95": float(np.quantile(values, 0.95)),
                    "filled_max": float(np.max(values)),
                    "standardized_mean_shift_vs_full_train": mean_shift,
                    "constant_in_full_train": bool(standard_deviation == 0.0),
                }
            )
    return pd.DataFrame(rows)


def write_feature_dictionary(
    path: Path,
    g7_names: Sequence[str],
    g8_names: Sequence[str],
) -> None:
    lines = [
        "# G7/G8 特征字典",
        "",
        "| 特征名 | 分组 | 输出类型 | 单位 | 因果口径 | 缺失处理 |",
        "|---|---|---|---|---|---|",
    ]
    for name in g7_names:
        lines.append(
            f"| `{name}` | G7_schedule | float32 | {_unit_for(name)} | "
            "仅使用严格早于样本 reference_time 的公开铁次时刻 | 当前折训练中位数；训练全缺失时填0 |"
        )
    for name in g8_names:
        lines.append(
            f"| `{name}` | G8_burden_time | float32 | {_unit_for(name)} | "
            "仅使用 cal_time 小于等于查询时刻的变料记录 | 当前折训练中位数；训练全缺失时填0 |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _unit_for(name: str) -> str:
    if "count" in name:
        return "次"
    if "flag" in name or "minute_fraction" in name:
        return "无量纲"
    if "hours_since" in name or "gap" in name:
        return "小时"
    if "slope" in name:
        return "原字段单位/小时"
    return "原字段单位"
