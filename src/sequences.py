from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import (
    BURDEN_CHANNELS,
    BURDEN_SEQUENCE_LENGTH,
    EPSILON,
    HOURLY_CHANNELS,
    HOURLY_SEQUENCE_LENGTH,
    TAP_SEQUENCE_LENGTH,
)
from src.data_io import RawTables


_FLOAT32_MAX = float(np.finfo(np.float32).max)
_TAP_CHANNELS = (
    "tap_iron",
    "tap_time_len",
    "rate",
    "slag_time_len",
    "break_slag_time_len",
    "ladle_count",
    "is_same_spout",
    "age_hours",
)
_BURDEN_SEQUENCE_CHANNELS = (*BURDEN_CHANNELS, "age_hours")


@dataclass(frozen=True)
class SequenceBatch:
    hourly_values: np.ndarray
    hourly_mask: np.ndarray
    tap_values: np.ndarray
    tap_mask: np.ndarray
    burden_values: np.ndarray
    burden_mask: np.ndarray


@dataclass(frozen=True)
class _StreamState:
    channel_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    all_missing_channels: tuple[str, ...]
    zero_variance_channels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_names": list(self.channel_names),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "all_missing_channels": list(self.all_missing_channels),
            "zero_variance_channels": list(self.zero_variance_channels),
        }


class ThreeStreamSequenceBuilder:
    def __init__(self) -> None:
        self.hourly_state_: _StreamState | None = None
        self.tap_state_: _StreamState | None = None
        self.burden_state_: _StreamState | None = None

    def fit(
        self, train_samples: pd.DataFrame, tables: RawTables
    ) -> "ThreeStreamSequenceBuilder":
        self._validate_samples(train_samples)
        raw = self._extract_raw(train_samples, tables)
        self.hourly_state_ = self._fit_stream(
            raw.hourly_values, raw.hourly_mask, HOURLY_CHANNELS
        )
        self.tap_state_ = self._fit_stream(raw.tap_values, raw.tap_mask, _TAP_CHANNELS)
        self.burden_state_ = self._fit_stream(
            raw.burden_values, raw.burden_mask, _BURDEN_SEQUENCE_CHANNELS
        )
        return self

    def transform(self, samples: pd.DataFrame, tables: RawTables) -> SequenceBatch:
        self._require_fitted()
        self._validate_samples(samples)
        raw = self._extract_raw(samples, tables)
        return SequenceBatch(
            hourly_values=self._apply_state(
                raw.hourly_values, raw.hourly_mask, self.hourly_state_
            ),
            hourly_mask=np.ascontiguousarray(raw.hourly_mask, dtype=bool),
            tap_values=self._apply_state(raw.tap_values, raw.tap_mask, self.tap_state_),
            tap_mask=np.ascontiguousarray(raw.tap_mask, dtype=bool),
            burden_values=self._apply_state(
                raw.burden_values, raw.burden_mask, self.burden_state_
            ),
            burden_mask=np.ascontiguousarray(raw.burden_mask, dtype=bool),
        )

    def state_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "schema_version": "data_v3.three_stream_sequence.v1",
            "hourly": self.hourly_state_.to_dict(),
            "tap": self.tap_state_.to_dict(),
            "burden": self.burden_state_.to_dict(),
            "shapes": {
                "hourly": [HOURLY_SEQUENCE_LENGTH, len(HOURLY_CHANNELS)],
                "tap": [TAP_SEQUENCE_LENGTH, len(_TAP_CHANNELS)],
                "burden": [BURDEN_SEQUENCE_LENGTH, len(_BURDEN_SEQUENCE_CHANNELS)],
            },
            "output_dtype": "float32",
            "mask_dtype": "bool",
        }

    def _extract_raw(self, samples: pd.DataFrame, tables: RawTables) -> SequenceBatch:
        hourly_values, hourly_mask = self._extract_hourly(
            samples, tables.operation_hourly
        )
        tap_values, tap_mask = self._extract_tap(samples, tables.tap_history_train)
        burden_values, burden_mask = self._extract_burden(
            samples, tables.burden_change
        )
        return SequenceBatch(
            hourly_values=hourly_values,
            hourly_mask=hourly_mask,
            tap_values=tap_values,
            tap_mask=tap_mask,
            burden_values=burden_values,
            burden_mask=burden_mask,
        )

    @staticmethod
    def _extract_hourly(
        samples: pd.DataFrame, operation_hourly: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        hourly = operation_hourly.copy()
        hourly["clock"] = pd.to_datetime(hourly["clock"], errors="raise").astype(
            "datetime64[ns]"
        )
        hourly = hourly.sort_values("clock", kind="mergesort").set_index("clock")
        values = np.full(
            (len(samples), HOURLY_SEQUENCE_LENGTH, len(HOURLY_CHANNELS)),
            np.nan,
            dtype=np.float64,
        )
        mask = np.zeros((len(samples), HOURLY_SEQUENCE_LENGTH), dtype=bool)
        for row_index, sample in enumerate(samples.itertuples(index=False)):
            reference = pd.Timestamp(sample.reference_time)
            end = reference.floor("h")
            grid = pd.date_range(
                end=end, periods=HOURLY_SEQUENCE_LENGTH, freq="h"
            ).astype("datetime64[ns]")
            present = grid.isin(hourly.index) & (grid <= reference)
            reindexed = hourly.reindex(grid)
            values[row_index] = reindexed.loc[:, list(HOURLY_CHANNELS)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            mask[row_index] = present
        return values, mask

    @staticmethod
    def _extract_tap(
        samples: pd.DataFrame, tap_history: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        history = tap_history.copy()
        history["tap_end_time"] = pd.to_datetime(
            history["tap_end_time"], errors="raise"
        ).astype("datetime64[ns]")
        history["sample_id"] = history["sample_id"].astype(str)
        history = history.sort_values(
            ["tap_end_time", "sample_id"], kind="mergesort"
        ).reset_index(drop=True)
        end_times = history["tap_end_time"].to_numpy(dtype="datetime64[ns]")
        values = np.full(
            (len(samples), TAP_SEQUENCE_LENGTH, len(_TAP_CHANNELS)),
            np.nan,
            dtype=np.float64,
        )
        mask = np.zeros((len(samples), TAP_SEQUENCE_LENGTH), dtype=bool)
        for row_index, sample in enumerate(samples.itertuples(index=False)):
            reference = pd.Timestamp(sample.reference_time)
            stop = int(
                np.searchsorted(end_times, reference.to_datetime64(), side="right")
            )
            visible = history.iloc[:stop]
            visible = visible[visible["sample_id"].ne(str(sample.sample_id))].iloc[
                -TAP_SEQUENCE_LENGTH:
            ]
            count = len(visible)
            if count == 0:
                continue
            start = TAP_SEQUENCE_LENGTH - count
            iron = pd.to_numeric(visible["tap_iron"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            duration = pd.to_numeric(
                visible["tap_time_len"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            rate = np.full(count, np.nan, dtype=np.float64)
            valid_rate = (
                np.isfinite(iron)
                & np.isfinite(duration)
                & (np.abs(duration) >= EPSILON)
            )
            rate[valid_rate] = iron[valid_rate] / duration[valid_rate]
            matrix = np.column_stack(
                [
                    iron,
                    duration,
                    rate,
                    pd.to_numeric(visible["slag_time_len"], errors="coerce"),
                    pd.to_numeric(
                        visible["break_slag_time_len"], errors="coerce"
                    ),
                    pd.to_numeric(visible["ladle_count"], errors="coerce"),
                    visible["spout_no"].eq(int(sample.spout_no)).astype(float),
                    (
                        reference - visible["tap_end_time"]
                    ).dt.total_seconds().to_numpy(dtype=np.float64)
                    / 3600.0,
                ]
            ).astype(np.float64)
            values[row_index, start:] = matrix
            mask[row_index, start:] = True
        return values, mask

    @staticmethod
    def _extract_burden(
        samples: pd.DataFrame, burden_change: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        events = burden_change.copy()
        events["cal_time"] = pd.to_datetime(events["cal_time"], errors="raise").astype(
            "datetime64[ns]"
        )
        events = events.sort_values("cal_time", kind="mergesort").reset_index(drop=True)
        event_times = events["cal_time"].to_numpy(dtype="datetime64[ns]")
        values = np.full(
            (len(samples), BURDEN_SEQUENCE_LENGTH, len(_BURDEN_SEQUENCE_CHANNELS)),
            np.nan,
            dtype=np.float64,
        )
        mask = np.zeros((len(samples), BURDEN_SEQUENCE_LENGTH), dtype=bool)
        for row_index, sample in enumerate(samples.itertuples(index=False)):
            reference = pd.Timestamp(sample.reference_time)
            stop = int(
                np.searchsorted(event_times, reference.to_datetime64(), side="right")
            )
            visible = events.iloc[:stop].iloc[-BURDEN_SEQUENCE_LENGTH:]
            count = len(visible)
            if count == 0:
                continue
            start = BURDEN_SEQUENCE_LENGTH - count
            matrix = visible.loc[:, list(BURDEN_CHANNELS)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            ages = (
                (reference - visible["cal_time"]).dt.total_seconds().to_numpy(
                    dtype=np.float64
                )
                / 3600.0
            )
            values[row_index, start:] = np.column_stack([matrix, ages])
            mask[row_index, start:] = True
        return values, mask

    @staticmethod
    def _fit_stream(
        values: np.ndarray, mask: np.ndarray, channel_names: tuple[str, ...]
    ) -> _StreamState:
        medians = np.zeros(values.shape[2], dtype=np.float64)
        means = np.zeros(values.shape[2], dtype=np.float64)
        scales = np.ones(values.shape[2], dtype=np.float64)
        all_missing: list[str] = []
        zero_variance: list[str] = []
        present = mask[:, :, None]
        for channel_index, channel in enumerate(channel_names):
            valid = present[:, :, 0] & np.isfinite(values[:, :, channel_index])
            observed = values[:, :, channel_index][valid]
            if observed.size == 0:
                all_missing.append(channel)
                zero_variance.append(channel)
                continue
            medians[channel_index] = float(np.median(observed))
            means[channel_index] = float(np.mean(observed))
            scale = float(np.std(observed, ddof=0))
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
                zero_variance.append(channel)
            scales[channel_index] = scale
        return _StreamState(
            channel_names=channel_names,
            medians=medians,
            means=means,
            scales=scales,
            all_missing_channels=tuple(all_missing),
            zero_variance_channels=tuple(zero_variance),
        )

    @staticmethod
    def _apply_state(
        values: np.ndarray, mask: np.ndarray, state: _StreamState
    ) -> np.ndarray:
        present = mask[:, :, None]
        finite = np.isfinite(values) & present
        standardized = (values - state.means[None, None, :]) / state.scales[
            None, None, :
        ]
        standardized = np.where(finite, standardized, 0.0)
        standardized = np.nan_to_num(
            standardized,
            nan=0.0,
            posinf=_FLOAT32_MAX,
            neginf=-_FLOAT32_MAX,
        )
        standardized = np.clip(
            standardized, -_FLOAT32_MAX, _FLOAT32_MAX
        ).astype(np.float32)
        return np.ascontiguousarray(standardized)

    @staticmethod
    def _validate_samples(samples: pd.DataFrame) -> None:
        required = {"sample_id", "spout_no", "reference_time"}
        missing = sorted(required - set(samples.columns))
        if missing:
            raise ValueError(f"samples missing required columns: {', '.join(missing)}")
        if samples["reference_time"].isna().any():
            raise ValueError("samples contain null reference_time")

    def _require_fitted(self) -> None:
        if self.hourly_state_ is None or self.tap_state_ is None or self.burden_state_ is None:
            raise RuntimeError("ThreeStreamSequenceBuilder is not fitted")
