from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import BURDEN_CHANNELS
from src.timeaware_contract import G8_FEATURE_NAMES, PUBLIC_SAMPLE_COLUMNS


@dataclass(frozen=True)
class BurdenGridBatch:
    grid_times: np.ndarray
    state_values: np.ndarray
    state_age_hours: np.ndarray
    effective_change_mask: np.ndarray
    raw_event_count: np.ndarray
    missing_mask: np.ndarray


@dataclass(frozen=True)
class BurdenFeatureBatch:
    features: pd.DataFrame
    grid: BurdenGridBatch
    diagnostics: dict[str, int]


class BurdenStateFeatureBuilder:
    """Reconstruct causal burden states and summarize them on real time."""

    lookback_hours = 48
    grid_step_hours = 2

    def build(
        self, samples: pd.DataFrame, burden_change: pd.DataFrame
    ) -> BurdenFeatureBatch:
        public = self._validate_samples(samples)
        normalized, diagnostics = self._normalize_events(burden_change)
        event_times = normalized["cal_time"].to_numpy(dtype="datetime64[ns]")
        event_states, last_change_times, change_times, change_values, change_masks = (
            self._reconstruct_states(normalized)
        )

        sample_count = len(public)
        channel_count = len(BURDEN_CHANNELS)
        grid_count = self.lookback_hours // self.grid_step_hours + 1
        grid_times = np.empty((sample_count, grid_count), dtype="datetime64[ns]")
        grid_values = np.full((sample_count, grid_count, channel_count), np.nan)
        grid_ages = np.full_like(grid_values, np.nan)
        grid_changes = np.zeros((sample_count, grid_count, channel_count), dtype=bool)
        grid_raw_counts = np.zeros((sample_count, grid_count), dtype=np.int32)

        feature_matrix = np.full(
            (sample_count, len(G8_FEATURE_NAMES)), np.nan, dtype=np.float64
        )
        feature_position = {name: index for index, name in enumerate(G8_FEATURE_NAMES)}

        for sample_position, reference in enumerate(public["reference_time"]):
            reference_time = pd.Timestamp(reference).to_datetime64()
            offsets = np.arange(
                -self.lookback_hours, 1, self.grid_step_hours, dtype=np.int64
            ).astype("timedelta64[h]")
            current_grid = reference_time + offsets
            grid_times[sample_position] = current_grid
            grid_positions = np.searchsorted(event_times, current_grid, side="right") - 1
            valid_grid = grid_positions >= 0
            if valid_grid.any():
                grid_values[sample_position, valid_grid] = event_states[
                    grid_positions[valid_grid]
                ]
                visible_last_changes = last_change_times[grid_positions[valid_grid]]
                ages = (
                    current_grid[valid_grid, None] - visible_last_changes
                ) / np.timedelta64(1, "h")
                grid_ages[sample_position, valid_grid] = ages.astype(np.float64)

            for grid_position, grid_time in enumerate(current_grid):
                interval_start = grid_time - np.timedelta64(self.grid_step_hours, "h")
                left = np.searchsorted(event_times, interval_start, side="right")
                right = np.searchsorted(event_times, grid_time, side="right")
                grid_raw_counts[sample_position, grid_position] = right - left
                if right > left:
                    grid_changes[sample_position, grid_position] = np.any(
                        change_masks[left:right], axis=0
                    )

            self._summarize_sample(
                sample_position=sample_position,
                reference_time=reference_time,
                event_times=event_times,
                event_states=event_states,
                last_change_times=last_change_times,
                change_times=change_times,
                change_values=change_values,
                feature_matrix=feature_matrix,
                feature_position=feature_position,
            )

        missing_mask = np.isnan(grid_values)
        grid = BurdenGridBatch(
            grid_times=np.ascontiguousarray(grid_times),
            state_values=np.ascontiguousarray(grid_values),
            state_age_hours=np.ascontiguousarray(grid_ages),
            effective_change_mask=np.ascontiguousarray(grid_changes),
            raw_event_count=np.ascontiguousarray(grid_raw_counts),
            missing_mask=np.ascontiguousarray(missing_mask),
        )
        features = pd.DataFrame(
            feature_matrix, index=samples.index.copy(), columns=G8_FEATURE_NAMES
        )
        return BurdenFeatureBatch(features=features, grid=grid, diagnostics=diagnostics)

    @staticmethod
    def _validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in PUBLIC_SAMPLE_COLUMNS if column not in samples.columns]
        if missing:
            raise ValueError(f"samples missing required columns: {', '.join(missing)}")
        public = samples.loc[:, PUBLIC_SAMPLE_COLUMNS].copy(deep=True)
        public["reference_time"] = pd.to_datetime(
            public["reference_time"], errors="coerce"
        ).astype("datetime64[ns]")
        if public["reference_time"].isna().any():
            raise ValueError("samples contain invalid reference_time")
        return public

    @staticmethod
    def _normalize_events(
        burden_change: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        required = ["cal_time", *BURDEN_CHANNELS]
        missing = [column for column in required if column not in burden_change.columns]
        if missing:
            raise ValueError(f"burden_change missing required columns: {', '.join(missing)}")
        columns = [*required]
        if "_original_row" in burden_change.columns:
            columns.append("_original_row")
        events = burden_change.loc[:, columns].copy(deep=True)
        events["cal_time"] = pd.to_datetime(events["cal_time"], errors="coerce").astype(
            "datetime64[ns]"
        )
        if events["cal_time"].isna().any():
            raise ValueError("burden_change contains invalid cal_time")
        for channel in BURDEN_CHANNELS:
            events[channel] = pd.to_numeric(events[channel], errors="coerce").astype(
                "float64"
            )
        if "_original_row" not in events.columns:
            events["_original_row"] = np.arange(len(events), dtype=np.int64)
        else:
            original = pd.to_numeric(events["_original_row"], errors="coerce")
            if original.isna().any() or original.duplicated().any():
                raise ValueError("burden_change contains invalid _original_row")
            events["_original_row"] = original.astype("int64")

        duplicate_timestamp_count = int(
            (events.groupby("cal_time", sort=False).size() > 1).sum()
        )
        events = events.sort_values(
            ["cal_time", "_original_row"], kind="mergesort"
        ).reset_index(drop=True)
        normalized = events.drop_duplicates("cal_time", keep="last").reset_index(drop=True)
        diagnostics = {
            "raw_row_count": int(len(events)),
            "normalized_event_count": int(len(normalized)),
            "duplicate_row_count": int(len(events) - len(normalized)),
            "duplicate_timestamp_count": duplicate_timestamp_count,
        }
        return normalized, diagnostics

    @staticmethod
    def _reconstruct_states(
        events: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        list[np.ndarray],
        np.ndarray,
    ]:
        raw = events.loc[:, BURDEN_CHANNELS].to_numpy(dtype=np.float64)
        times = events["cal_time"].to_numpy(dtype="datetime64[ns]")
        row_count, channel_count = raw.shape
        states = np.full((row_count, channel_count), np.nan, dtype=np.float64)
        last_changes = np.full((row_count, channel_count), np.datetime64("NaT", "ns"))
        masks = np.zeros((row_count, channel_count), dtype=bool)
        current = np.full(channel_count, np.nan, dtype=np.float64)
        last_change = np.full(channel_count, np.datetime64("NaT", "ns"))

        channel_change_times: list[list[np.datetime64]] = [
            [] for _ in range(channel_count)
        ]
        channel_change_values: list[list[float]] = [[] for _ in range(channel_count)]
        for row in range(row_count):
            for channel in range(channel_count):
                value = raw[row, channel]
                if np.isfinite(value) and (
                    not np.isfinite(current[channel]) or value != current[channel]
                ):
                    current[channel] = value
                    last_change[channel] = times[row]
                    masks[row, channel] = True
                    channel_change_times[channel].append(times[row])
                    channel_change_values[channel].append(float(value))
            states[row] = current
            last_changes[row] = last_change

        change_times = [
            np.asarray(values, dtype="datetime64[ns]") for values in channel_change_times
        ]
        change_values = [
            np.asarray(values, dtype=np.float64) for values in channel_change_values
        ]
        return states, last_changes, change_times, change_values, masks

    def _summarize_sample(
        self,
        sample_position: int,
        reference_time: np.datetime64,
        event_times: np.ndarray,
        event_states: np.ndarray,
        last_change_times: np.ndarray,
        change_times: list[np.ndarray],
        change_values: list[np.ndarray],
        feature_matrix: np.ndarray,
        feature_position: dict[str, int],
    ) -> None:
        current_event_position = (
            np.searchsorted(event_times, reference_time, side="right") - 1
        )
        for channel_position, channel in enumerate(BURDEN_CHANNELS):
            prefix = f"g8_{channel}_"
            current_value = np.nan
            current_age = np.nan
            if current_event_position >= 0:
                current_value = event_states[current_event_position, channel_position]
                change_time = last_change_times[current_event_position, channel_position]
                if not np.isnat(change_time):
                    current_age = float(
                        (reference_time - change_time) / np.timedelta64(1, "h")
                    )
            self._set(feature_matrix, sample_position, feature_position, prefix + "current_value", current_value)
            self._set(feature_matrix, sample_position, feature_position, prefix + "hours_since_last_real_change", current_age)

            lag_values: dict[int, float] = {}
            for hours in (2, 4, 8, 12, 24, 48):
                lag_time = reference_time - np.timedelta64(hours, "h")
                lag_values[hours] = self._query_field_state(
                    event_times, event_states, channel_position, lag_time
                )
                self._set(
                    feature_matrix,
                    sample_position,
                    feature_position,
                    f"{prefix}value_at_{hours}h",
                    lag_values[hours],
                )
            for hours in (2, 8, 24):
                difference = (
                    current_value - lag_values[hours]
                    if np.isfinite(current_value) and np.isfinite(lag_values[hours])
                    else np.nan
                )
                self._set(
                    feature_matrix,
                    sample_position,
                    feature_position,
                    f"{prefix}current_minus_lag_{hours}h",
                    difference,
                )

            for hours in (6, 12, 24, 48):
                start = reference_time - np.timedelta64(hours, "h")
                point_times, point_values = self._state_points(
                    start,
                    reference_time,
                    event_times,
                    event_states,
                    channel_position,
                    change_times[channel_position],
                    change_values[channel_position],
                )
                weighted_mean = self._time_weighted_mean(
                    point_times, point_values, reference_time
                )
                self._set(
                    feature_matrix,
                    sample_position,
                    feature_position,
                    f"{prefix}time_weighted_mean_{hours}h",
                    weighted_mean,
                )
                changes_in_window = int(
                    np.searchsorted(change_times[channel_position], reference_time, side="right")
                    - np.searchsorted(change_times[channel_position], start, side="right")
                )
                self._set(
                    feature_matrix,
                    sample_position,
                    feature_position,
                    f"{prefix}effective_change_count_{hours}h",
                    float(changes_in_window),
                )

            for hours in (12, 24, 48):
                start = reference_time - np.timedelta64(hours, "h")
                point_times, point_values = self._state_points(
                    start,
                    reference_time,
                    event_times,
                    event_states,
                    channel_position,
                    change_times[channel_position],
                    change_values[channel_position],
                )
                slope = self._real_time_slope(point_times, point_values)
                max_change = (
                    float(np.max(np.abs(np.diff(point_values))))
                    if len(point_values) >= 2
                    else np.nan
                )
                self._set(feature_matrix, sample_position, feature_position, f"{prefix}real_time_slope_{hours}h", slope)
                self._set(feature_matrix, sample_position, feature_position, f"{prefix}max_abs_change_{hours}h", max_change)

        for hours in (6, 12, 24, 48):
            start = reference_time - np.timedelta64(hours, "h")
            count = int(
                np.searchsorted(event_times, reference_time, side="right")
                - np.searchsorted(event_times, start, side="right")
            )
            self._set(feature_matrix, sample_position, feature_position, f"g8_raw_event_count_{hours}h", float(count))

        for hours in (24, 48):
            start = reference_time - np.timedelta64(hours, "h")
            left = np.searchsorted(event_times, start, side="left")
            right = np.searchsorted(event_times, reference_time, side="right")
            gaps = (
                np.diff(event_times[left:right]) / np.timedelta64(1, "h")
            ).astype(np.float64)
            if len(gaps):
                median = float(np.median(gaps))
                standard_deviation = float(np.std(gaps, ddof=0))
            else:
                median = np.nan
                standard_deviation = np.nan
            self._set(feature_matrix, sample_position, feature_position, f"g8_event_gap_median_{hours}h", median)
            self._set(feature_matrix, sample_position, feature_position, f"g8_event_gap_std_{hours}h", standard_deviation)

    @staticmethod
    def _query_field_state(
        event_times: np.ndarray,
        event_states: np.ndarray,
        channel_position: int,
        query_time: np.datetime64,
    ) -> float:
        position = np.searchsorted(event_times, query_time, side="right") - 1
        if position < 0:
            return np.nan
        return float(event_states[position, channel_position])

    def _state_points(
        self,
        start: np.datetime64,
        end: np.datetime64,
        event_times: np.ndarray,
        event_states: np.ndarray,
        channel_position: int,
        field_change_times: np.ndarray,
        field_change_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        times: list[np.datetime64] = []
        values: list[float] = []
        start_value = self._query_field_state(
            event_times, event_states, channel_position, start
        )
        if np.isfinite(start_value):
            times.append(start)
            values.append(start_value)
        left = np.searchsorted(field_change_times, start, side="right")
        right = np.searchsorted(field_change_times, end, side="right")
        times.extend(field_change_times[left:right].tolist())
        values.extend(field_change_values[left:right].tolist())
        return (
            np.asarray(times, dtype="datetime64[ns]"),
            np.asarray(values, dtype=np.float64),
        )

    @staticmethod
    def _time_weighted_mean(
        point_times: np.ndarray, point_values: np.ndarray, end: np.datetime64
    ) -> float:
        if not len(point_values):
            return np.nan
        segment_ends = np.concatenate(
            [point_times[1:], np.asarray([end], dtype="datetime64[ns]")]
        )
        durations = (
            (segment_ends - point_times) / np.timedelta64(1, "h")
        ).astype(np.float64)
        covered = float(np.sum(durations))
        if covered <= 0.0:
            return np.nan
        return float(np.dot(point_values, durations) / covered)

    @staticmethod
    def _real_time_slope(point_times: np.ndarray, point_values: np.ndarray) -> float:
        if len(point_values) < 2:
            return np.nan
        x = (
            (point_times - point_times[0]) / np.timedelta64(1, "h")
        ).astype(np.float64)
        if np.unique(x).size < 2:
            return np.nan
        return float(np.polyfit(x, point_values, 1)[0])

    @staticmethod
    def _set(
        matrix: np.ndarray,
        row: int,
        positions: dict[str, int],
        name: str,
        value: float,
    ) -> None:
        matrix[row, positions[name]] = float(value)
