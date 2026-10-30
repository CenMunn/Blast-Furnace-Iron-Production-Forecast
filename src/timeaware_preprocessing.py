from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.burden_state_features import BurdenStateFeatureBuilder
from src.schedule_features import ScheduleFeatureBuilder
from src.timeaware_contract import (
    G7_FEATURE_NAMES,
    G8_FEATURE_NAMES,
    PUBLIC_SAMPLE_COLUMNS,
)


class FoldTimeawareImputer:
    """Fold-local median imputation without scaling or target access."""

    state_version = "timeaware-fold-median-v1"

    def __init__(self) -> None:
        self.feature_names_: tuple[str, ...] | None = None
        self.medians_: np.ndarray | None = None
        self.all_missing_columns_: tuple[str, ...] = ()
        self.constant_columns_: tuple[str, ...] = ()

    def fit(self, raw_train: pd.DataFrame) -> FoldTimeawareImputer:
        if raw_train.columns.duplicated().any():
            raise ValueError("raw feature columns must be unique")
        if len(raw_train.columns) == 0:
            raise ValueError("raw feature table has no columns")
        matrix = raw_train.to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(matrix)
        medians = np.zeros(matrix.shape[1], dtype=np.float64)
        all_missing: list[str] = []
        constant: list[str] = []
        for column_position, name in enumerate(raw_train.columns.astype(str)):
            values = matrix[finite[:, column_position], column_position]
            if len(values):
                medians[column_position] = float(np.median(values))
                if np.all(values == values[0]):
                    constant.append(name)
            else:
                medians[column_position] = 0.0
                all_missing.append(name)
                constant.append(name)
        self.feature_names_ = tuple(raw_train.columns.astype(str))
        self.medians_ = medians
        self.all_missing_columns_ = tuple(all_missing)
        self.constant_columns_ = tuple(constant)
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None or self.medians_ is None:
            raise RuntimeError("FoldTimeawareImputer is not fitted")
        if tuple(raw.columns.astype(str)) != self.feature_names_:
            raise ValueError("raw feature columns do not match fitted order")
        matrix = raw.to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(matrix)
        filled = np.where(finite, matrix, self.medians_[None, :])
        float32_limit = np.finfo(np.float32).max
        filled = np.clip(filled, -float32_limit, float32_limit)
        return pd.DataFrame(
            filled.astype(np.float32),
            columns=self.feature_names_,
            index=raw.index.copy(),
        )

    def fit_transform(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_train).transform(raw_train)

    def state_dict(self) -> dict[str, object]:
        if self.feature_names_ is None or self.medians_ is None:
            raise RuntimeError("FoldTimeawareImputer is not fitted")
        return deepcopy(
            {
                "state_version": self.state_version,
                "feature_names": list(self.feature_names_),
                "medians": {
                    name: float(value)
                    for name, value in zip(self.feature_names_, self.medians_, strict=True)
                },
                "all_missing_columns": list(self.all_missing_columns_),
                "constant_columns": list(self.constant_columns_),
                "output_dtype": "float32",
                "scaling": "none",
            }
        )


def assemble_timeaware_raw(
    samples: pd.DataFrame,
    schedule_context: pd.DataFrame,
    burden_events: pd.DataFrame,
) -> pd.DataFrame:
    missing = [column for column in PUBLIC_SAMPLE_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"samples missing required columns: {', '.join(missing)}")
    context_missing = [
        column for column in PUBLIC_SAMPLE_COLUMNS if column not in schedule_context.columns
    ]
    if context_missing:
        raise ValueError(
            f"schedule_context missing required columns: {', '.join(context_missing)}"
        )
    public_samples = samples.loc[:, PUBLIC_SAMPLE_COLUMNS].copy(deep=True)
    public_context = schedule_context.loc[:, PUBLIC_SAMPLE_COLUMNS].copy(deep=True)
    if public_context["sample_id"].duplicated().any():
        raise ValueError("schedule_context contains duplicate sample_id")

    wanted_ids = public_samples["sample_id"].astype(str).tolist()
    context_ids = set(public_context["sample_id"].astype(str))
    absent = [sample_id for sample_id in wanted_ids if sample_id not in context_ids]
    if absent:
        raise ValueError(f"samples absent from schedule_context: {absent[:3]}")

    schedule_all = ScheduleFeatureBuilder().build(public_context)
    schedule_by_id = schedule_all.copy()
    schedule_by_id.insert(0, "sample_id", public_context["sample_id"].astype(str).to_numpy())
    selected_schedule = schedule_by_id.set_index("sample_id").loc[wanted_ids]
    selected_schedule.index = samples.index.copy()
    selected_schedule = selected_schedule.loc[:, G7_FEATURE_NAMES]

    burden = BurdenStateFeatureBuilder().build(public_samples, burden_events).features
    raw = pd.concat([selected_schedule, burden], axis=1)
    expected = [*G7_FEATURE_NAMES, *G8_FEATURE_NAMES]
    if raw.columns.duplicated().any() or list(raw.columns) != expected:
        raise RuntimeError("assembled timeaware feature schema mismatch")
    if any(name in raw.columns for name in ("tap_iron", "tap_time_len", "target_rate")):
        raise RuntimeError("target column leaked into timeaware features")
    if not all(name.startswith(("g7_", "g8_")) for name in raw.columns):
        raise RuntimeError("unexpected timeaware feature prefix")
    return raw
