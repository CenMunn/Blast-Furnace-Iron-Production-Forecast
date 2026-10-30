from __future__ import annotations

import numpy as np
import pandas as pd

from src.timeaware_contract import G7_FEATURE_NAMES, PUBLIC_SAMPLE_COLUMNS


_HOUR = pd.Timedelta(hours=1)


class ScheduleFeatureBuilder:
    """Build target-free tap-schedule features from strictly earlier samples."""

    def build(self, samples: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in PUBLIC_SAMPLE_COLUMNS if column not in samples.columns]
        if missing:
            raise ValueError(f"samples missing required columns: {', '.join(missing)}")

        public = samples.loc[:, PUBLIC_SAMPLE_COLUMNS].copy(deep=True)
        public["reference_time"] = pd.to_datetime(
            public["reference_time"], errors="coerce"
        ).astype("datetime64[ns]")
        if public["reference_time"].isna().any():
            raise ValueError("samples contain invalid reference_time")
        public["spout_no"] = pd.to_numeric(public["spout_no"], errors="coerce")
        if public["spout_no"].isna().any():
            raise ValueError("samples contain invalid spout_no")
        public["_input_position"] = np.arange(len(public), dtype=np.int64)

        ordered = public.sort_values(
            ["reference_time", "sample_id", "_input_position"], kind="mergesort"
        ).reset_index(drop=True)
        result = np.full((len(public), len(G7_FEATURE_NAMES)), np.nan, dtype=np.float64)
        column_position = {name: index for index, name in enumerate(G7_FEATURE_NAMES)}

        all_times = ordered["reference_time"].to_numpy(dtype="datetime64[ns]")
        for row in ordered.to_dict(orient="records"):
            current = pd.Timestamp(row["reference_time"])
            position = int(row["_input_position"])
            stop = int(np.searchsorted(all_times, current.to_datetime64(), side="left"))
            visible = ordered.iloc[:stop]
            unique_times = visible["reference_time"].drop_duplicates().to_numpy(
                dtype="datetime64[ns]"
            )

            current_gap = np.nan
            historical_gaps = np.empty(0, dtype=np.float64)
            gap_times = np.empty(0, dtype="datetime64[ns]")
            if len(unique_times):
                current_gap = float(
                    (current - pd.Timestamp(unique_times[-1])) / _HOUR
                )
            if len(unique_times) >= 2:
                historical_gaps = (
                    np.diff(unique_times) / np.timedelta64(1, "h")
                ).astype(np.float64)
                gap_times = unique_times[1:]

            lag_values = [current_gap, *historical_gaps[::-1].tolist()]
            for lag in (1, 2, 3, 4, 5, 6, 8, 10):
                if len(lag_values) >= lag:
                    self._set(result, position, column_position, f"g7_start_gap_lag_{lag}", lag_values[lag - 1])

            for statistic in ("mean", "median", "std"):
                for window in (3, 5, 10, 20):
                    values = historical_gaps[-window:]
                    if len(values):
                        value = self._statistic(values, statistic)
                        self._set(result, position, column_position, f"g7_start_gap_{statistic}_{window}", value)

            for statistic in ("min", "max"):
                for window in (5, 10, 20):
                    values = historical_gaps[-window:]
                    if len(values):
                        value = self._statistic(values, statistic)
                        self._set(result, position, column_position, f"g7_start_gap_{statistic}_{window}", value)

            for window in (5, 10):
                values = historical_gaps[-window:]
                times = gap_times[-window:]
                if len(values) >= 2:
                    x = (times - times[-1]) / np.timedelta64(1, "h")
                    slope = float(np.polyfit(x.astype(float), values, 1)[0])
                    self._set(result, position, column_position, f"g7_start_gap_trend_{window}", slope)

            recent_gaps = historical_gaps[-10:]
            if np.isfinite(current_gap) and len(recent_gaps):
                self._set(
                    result,
                    position,
                    column_position,
                    "g7_start_gap_current_minus_median_10",
                    current_gap - float(np.median(recent_gaps)),
                )

            for hours in (6, 12, 24, 48):
                lower = current - pd.Timedelta(hours=hours)
                count = int(
                    ((visible["reference_time"] >= lower) & (visible["reference_time"] < current)).sum()
                )
                self._set(result, position, column_position, f"g7_tap_count_{hours}h", float(count))

            same_spout_times = (
                visible.loc[visible["spout_no"].eq(row["spout_no"]), "reference_time"]
                .drop_duplicates()
                .to_numpy(dtype="datetime64[ns]")
            )
            if len(same_spout_times):
                same_current_gap = float(
                    (current - pd.Timestamp(same_spout_times[-1])) / _HOUR
                )
                self._set(
                    result,
                    position,
                    column_position,
                    "g7_hours_since_same_spout_start",
                    same_current_gap,
                )
                same_lags = [same_current_gap]
                if len(same_spout_times) >= 2:
                    same_lags.extend(
                        (
                            np.diff(same_spout_times) / np.timedelta64(1, "h")
                        ).astype(np.float64)[::-1].tolist()
                    )
                for lag in (1, 2, 3):
                    if len(same_lags) >= lag:
                        self._set(
                            result,
                            position,
                            column_position,
                            f"g7_same_spout_start_gap_lag_{lag}",
                            same_lags[lag - 1],
                        )

            if np.isfinite(current_gap):
                self._set(
                    result,
                    position,
                    column_position,
                    "g7_long_gap_flag_180m",
                    float(current_gap * 60.0 > 180.0),
                )
                self._set(
                    result,
                    position,
                    column_position,
                    "g7_long_gap_flag_240m",
                    float(current_gap * 60.0 > 240.0),
                )

            minute_fraction = (
                current.minute + current.second / 60.0 + current.microsecond / 60_000_000.0
            ) / 60.0
            self._set(
                result,
                position,
                column_position,
                "g7_reference_minute_fraction",
                minute_fraction,
            )

        return pd.DataFrame(result, index=samples.index.copy(), columns=G7_FEATURE_NAMES)

    @staticmethod
    def _statistic(values: np.ndarray, statistic: str) -> float:
        if statistic == "mean":
            return float(np.mean(values))
        if statistic == "median":
            return float(np.median(values))
        if statistic == "std":
            return float(np.std(values, ddof=0))
        if statistic == "min":
            return float(np.min(values))
        if statistic == "max":
            return float(np.max(values))
        raise ValueError(f"unsupported statistic: {statistic}")

    @staticmethod
    def _set(
        result: np.ndarray,
        row: int,
        positions: dict[str, int],
        name: str,
        value: float,
    ) -> None:
        result[row, positions[name]] = float(value)
