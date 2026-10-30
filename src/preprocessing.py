from __future__ import annotations

import numpy as np
import pandas as pd


_DISTANCE_FEATURE = "g6_train_distribution_rms_z"
_FLOAT32_MAX = float(np.finfo(np.float32).max)


class FoldPreprocessor:
    """Median-impute and standardize with state fitted on one training context."""

    def __init__(self) -> None:
        self.feature_names_: list[str] | None = None
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.indicator_columns_: list[str] | None = None
        self.all_missing_columns_: list[str] | None = None
        self.zero_variance_columns_: list[str] | None = None
        self.distance_columns_: list[str] | None = None
        self.distance_mean_: float | None = None
        self.distance_scale_: float | None = None

    def fit(self, raw_train: pd.DataFrame) -> "FoldPreprocessor":
        self._validate_frame(raw_train)
        names = raw_train.columns.tolist()
        matrix = raw_train.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(matrix)
        medians = np.zeros(len(names), dtype=np.float64)
        all_missing: list[str] = []
        indicators: list[str] = []
        for index, name in enumerate(names):
            values = matrix[finite[:, index], index]
            if values.size:
                medians[index] = float(np.median(values))
            else:
                medians[index] = 0.0
                all_missing.append(name)
            if self._is_indicator(name, values):
                indicators.append(name)
        filled = np.where(finite, matrix, medians)
        means = np.zeros(len(names), dtype=np.float64)
        scales = np.ones(len(names), dtype=np.float64)
        zero_variance: list[str] = []
        indicator_set = set(indicators)
        for index, name in enumerate(names):
            if name in indicator_set or name in all_missing:
                if name in all_missing:
                    zero_variance.append(name)
                continue
            means[index] = float(np.mean(filled[:, index]))
            scale = float(np.std(filled[:, index], ddof=0))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
                zero_variance.append(name)
            scales[index] = scale
        standardized = (filled - means) / scales
        for index, name in enumerate(names):
            if name in indicator_set:
                standardized[:, index] = filled[:, index]
        distance_columns = [
            name
            for name in names
            if name.startswith(("g3_", "g4_")) and name not in indicator_set
        ]
        distance_indices = [names.index(name) for name in distance_columns]
        distances = self._stable_row_rms(standardized[:, distance_indices])
        distance_mean = float(np.mean(distances)) if distances.size else 0.0
        distance_scale = float(np.std(distances, ddof=0)) if distances.size else 1.0
        if not np.isfinite(distance_scale) or distance_scale == 0.0:
            distance_scale = 1.0

        self.feature_names_ = names
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.indicator_columns_ = indicators
        self.all_missing_columns_ = all_missing
        self.zero_variance_columns_ = zero_variance
        self.distance_columns_ = distance_columns
        self.distance_mean_ = distance_mean
        self.distance_scale_ = distance_scale
        return self

    def fit_transform(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_train).transform(raw_train)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_frame(raw)
        if raw.columns.tolist() != self.feature_names_:
            raise ValueError("raw feature columns do not match fitted order")
        matrix = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(matrix)
        filled = np.where(finite, matrix, self.medians_)
        standardized = (filled - self.means_) / self.scales_
        indicator_set = set(self.indicator_columns_)
        for index, name in enumerate(self.feature_names_):
            if name in indicator_set:
                standardized[:, index] = filled[:, index]
        distance_indices = [self.feature_names_.index(name) for name in self.distance_columns_]
        distances = self._stable_row_rms(standardized[:, distance_indices])
        distance_z = (distances - self.distance_mean_) / self.distance_scale_
        output = np.column_stack([standardized, distance_z])
        output = np.nan_to_num(
            output,
            nan=0.0,
            posinf=_FLOAT32_MAX,
            neginf=-_FLOAT32_MAX,
        )
        output = np.clip(output, -_FLOAT32_MAX, _FLOAT32_MAX).astype(np.float32)
        return pd.DataFrame(
            output,
            columns=[*self.feature_names_, _DISTANCE_FEATURE],
            index=raw.index,
        )

    def state_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "schema_version": "data_v3.fold_preprocessor.v1",
            "feature_names": [*self.feature_names_, _DISTANCE_FEATURE],
            "raw_feature_names": list(self.feature_names_),
            "medians": self.medians_.tolist(),
            "means": self.means_.tolist(),
            "scales": self.scales_.tolist(),
            "indicator_columns": list(self.indicator_columns_),
            "all_missing_columns": list(self.all_missing_columns_),
            "zero_variance_columns": list(self.zero_variance_columns_),
            "distance_columns": list(self.distance_columns_),
            "distance_mean": self.distance_mean_,
            "distance_scale": self.distance_scale_,
            "output_dtype": "float32",
        }

    @staticmethod
    def _stable_row_rms(values: np.ndarray) -> np.ndarray:
        if values.shape[1] == 0:
            return np.zeros(values.shape[0], dtype=np.float64)
        absolute = np.abs(values)
        maximum = np.max(absolute, axis=1)
        result = np.zeros(values.shape[0], dtype=np.float64)
        nonzero = maximum > 0.0
        scaled = values[nonzero] / maximum[nonzero, None]
        result[nonzero] = maximum[nonzero] * np.sqrt(np.mean(scaled * scaled, axis=1))
        return result

    @staticmethod
    def _is_indicator(name: str, finite_values: np.ndarray) -> bool:
        indicator_name = any(
            token in name
            for token in ("_flag", "_insufficient", "_fallback", "_is_missing")
        )
        return bool(
            indicator_name
            and finite_values.size
            and set(np.unique(finite_values)).issubset({0.0, 1.0})
        )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("raw feature frame is empty")
        if frame.columns.duplicated().any():
            raise ValueError("raw feature frame has duplicate columns")
        if _DISTANCE_FEATURE in frame.columns:
            raise ValueError(f"raw feature frame must not contain {_DISTANCE_FEATURE}")

    def _require_fitted(self) -> None:
        if self.feature_names_ is None:
            raise RuntimeError("FoldPreprocessor is not fitted")
