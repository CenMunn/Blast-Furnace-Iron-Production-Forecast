from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.constants import RANDOM_SEED


@dataclass(frozen=True)
class DriftResult:
    context: str
    earlier_rows: int
    later_rows: int
    adversarial_auc: float | None
    feature_metrics: tuple[dict[str, Any], ...]
    adversarial: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "earlier_rows": self.earlier_rows,
            "later_rows": self.later_rows,
            "adversarial_auc": self.adversarial_auc,
            "feature_metrics": list(self.feature_metrics),
            "adversarial": self.adversarial,
        }


def compute_drift_report(
    train_features: pd.DataFrame,
    later_features: pd.DataFrame,
    context: str,
    seed: int = RANDOM_SEED,
) -> DriftResult:
    """Measure univariate drift and deterministic adversarial separability."""

    feature_names = _validate_pair(train_features, later_features)
    earlier = train_features.copy(deep=True)
    later = later_features.copy(deep=True)
    metrics: list[dict[str, Any]] = []
    for feature in feature_names:
        left = pd.to_numeric(earlier[feature], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(later[feature], errors="coerce").to_numpy(dtype=np.float64)
        metrics.append(
            {
                "feature": feature,
                "missing_rate_earlier": float(np.mean(~np.isfinite(left))),
                "missing_rate_later": float(np.mean(~np.isfinite(right))),
                "missing_rate_difference": float(
                    np.mean(~np.isfinite(right)) - np.mean(~np.isfinite(left))
                ),
                "robust_standardized_mean_difference": _robust_difference(left, right),
                "psi": _population_stability_index(left, right),
            }
        )
    adversarial = run_adversarial_validation(
        earlier, later, feature_names, seed=seed
    )
    importance = {
        item["feature"]: item["importance"]
        for item in adversarial["feature_importance"]
    }
    for item in metrics:
        item["adversarial_importance"] = importance[item["feature"]]
    metrics.sort(
        key=lambda item: (
            -item["adversarial_importance"],
            -abs(item["robust_standardized_mean_difference"]),
            item["feature"],
        )
    )
    return DriftResult(
        context=context,
        earlier_rows=len(earlier),
        later_rows=len(later),
        adversarial_auc=adversarial["oof_roc_auc"],
        feature_metrics=tuple(metrics),
        adversarial=adversarial,
    )


def run_adversarial_validation(
    train_features: pd.DataFrame,
    later_features: pd.DataFrame,
    feature_names: list[str],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Classify earlier versus later rows using fixed ExtraTrees OOF settings."""

    _validate_pair(train_features.loc[:, feature_names], later_features.loc[:, feature_names])
    combined = pd.concat(
        [train_features.loc[:, feature_names], later_features.loc[:, feature_names]],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce")
    matrix = combined.to_numpy(dtype=np.float64)
    finite = np.isfinite(matrix)
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    train_matrix = matrix[: len(train_features)]
    for column in range(matrix.shape[1]):
        observed = train_matrix[np.isfinite(train_matrix[:, column]), column]
        medians[column] = float(np.median(observed)) if observed.size else 0.0
    matrix = np.where(finite, matrix, medians)
    labels = np.concatenate(
        [np.zeros(len(train_features), dtype=np.int8), np.ones(len(later_features), dtype=np.int8)]
    )
    configuration = {
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": int(seed),
        "n_jobs": 1,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True)",
    }
    model = ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    if min(len(train_features), len(later_features)) < 5:
        importances = np.zeros(len(feature_names), dtype=np.float64)
        auc: float | None = None
        status = "insufficient_samples"
    else:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probabilities = cross_val_predict(
            model, matrix, labels, cv=folds, method="predict_proba", n_jobs=1
        )[:, 1]
        auc = float(roc_auc_score(labels, probabilities))
        model.fit(matrix, labels)
        importances = np.nan_to_num(model.feature_importances_, nan=0.0)
        status = "ok"
    ranked = sorted(
        (
            {"feature": feature, "importance": float(importances[index])}
            for index, feature in enumerate(feature_names)
        ),
        key=lambda item: (-item["importance"], item["feature"]),
    )
    return {
        "schema_version": "data_v3.adversarial_validation.v1",
        "status": status,
        "oof_roc_auc": auc,
        "configuration": configuration,
        "earlier_rows": int(len(train_features)),
        "later_rows": int(len(later_features)),
        "feature_importance": ranked,
    }


def _validate_pair(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if left.empty or right.empty:
        raise ValueError("drift inputs must both contain rows")
    if left.columns.tolist() != right.columns.tolist():
        raise ValueError("drift inputs must have identical ordered feature columns")
    if left.columns.duplicated().any():
        raise ValueError("drift inputs contain duplicate feature names")
    return left.columns.tolist()


def _robust_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return 0.0
    pooled = np.concatenate([left, right])
    scale = float((np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25)) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(pooled, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 0.0
    return float((np.median(right) - np.median(left)) / scale)


def _population_stability_index(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return 0.0
    edges = np.unique(np.quantile(left, np.linspace(0.0, 1.0, 11)))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    left_counts, _ = np.histogram(left, bins=edges)
    right_counts, _ = np.histogram(right, bins=edges)
    epsilon = 1e-6
    left_rates = np.clip(left_counts / left_counts.sum(), epsilon, None)
    right_rates = np.clip(right_counts / right_counts.sum(), epsilon, None)
    return float(np.sum((right_rates - left_rates) * np.log(right_rates / left_rates)))
