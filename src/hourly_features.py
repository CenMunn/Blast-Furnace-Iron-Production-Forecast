from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import HOURLY_CHANNELS, HOURLY_WINDOWS


_STATS = (
    "mean",
    "std",
    "median",
    "min",
    "max",
    "last",
    "slope",
    "first_last_diff",
    "max_abs_jump",
    "diff_mean",
    "valid_count",
    "missing_rate",
    "longest_missing_run",
)


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


class HourlyFeatureBuilder:
    def __init__(self) -> None:
        names: list[str] = []
        for channel in HOURLY_CHANNELS:
            for window in HOURLY_WINDOWS:
                names.extend(f"g3_{channel}_w{window}h_{stat}" for stat in _STATS)
            names.extend(
                f"g3_{channel}_w{window}h_mean_minus_24h"
                for window in (1, 3, 6, 12)
            )
        names.append("g6_hours_since_operation_start")
        for channel in HOURLY_CHANNELS:
            names.extend(
                f"g6_{channel}_recent_{days}d_mean_minus_history"
                for days in (7, 14, 30)
            )
        self.feature_names_ = names

    def build(self, samples: pd.DataFrame, operation_hourly: pd.DataFrame) -> pd.DataFrame:
        hourly = operation_hourly.copy()
        hourly["clock"] = pd.to_datetime(hourly["clock"], errors="raise").astype(
            "datetime64[ns]"
        )
        hourly = hourly.sort_values("clock", kind="mergesort").reset_index(drop=True)
        rows: list[dict[str, float]] = []
        operation_start = pd.Timestamp(hourly["clock"].min()) if len(hourly) else pd.NaT
        for sample in samples.itertuples(index=False):
            reference = pd.Timestamp(sample.reference_time)
            visible = hourly[hourly["clock"].le(reference)]
            row: dict[str, float] = {}
            window_means: dict[tuple[str, int], float] = {}
            for channel in HOURLY_CHANNELS:
                for window in HOURLY_WINDOWS:
                    left = reference - pd.Timedelta(hours=window)
                    window_rows = visible[visible["clock"].gt(left)]
                    grid_start = left.floor("h") + pd.Timedelta(hours=1)
                    grid_end = reference.floor("h")
                    grid = (
                        pd.date_range(grid_start, grid_end, freq="h")
                        if grid_start <= grid_end
                        else pd.DatetimeIndex([])
                    )
                    series = window_rows.set_index("clock")[channel].reindex(grid)
                    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
                    finite_mask = np.isfinite(values)
                    finite = values[finite_mask]
                    prefix = f"g3_{channel}_w{window}h"
                    if finite.size:
                        positions = np.flatnonzero(finite_mask).astype(np.float64)
                        row[f"{prefix}_mean"] = float(np.mean(finite))
                        row[f"{prefix}_std"] = float(np.std(finite, ddof=0))
                        row[f"{prefix}_median"] = float(np.median(finite))
                        row[f"{prefix}_min"] = float(np.min(finite))
                        row[f"{prefix}_max"] = float(np.max(finite))
                        row[f"{prefix}_last"] = float(finite[-1])
                        row[f"{prefix}_slope"] = (
                            float(np.polyfit(positions, finite, 1)[0])
                            if finite.size >= 2
                            else np.nan
                        )
                        row[f"{prefix}_first_last_diff"] = (
                            float(finite[-1] - finite[0]) if finite.size >= 2 else np.nan
                        )
                        differences = np.diff(finite)
                        row[f"{prefix}_max_abs_jump"] = (
                            float(np.max(np.abs(differences))) if differences.size else np.nan
                        )
                        row[f"{prefix}_diff_mean"] = (
                            float(np.mean(differences)) if differences.size else np.nan
                        )
                    else:
                        for statistic in _STATS[:10]:
                            row[f"{prefix}_{statistic}"] = np.nan
                    row[f"{prefix}_valid_count"] = float(finite.size)
                    row[f"{prefix}_missing_rate"] = (
                        float(1.0 - finite.size / len(grid)) if len(grid) else 1.0
                    )
                    row[f"{prefix}_longest_missing_run"] = float(
                        _longest_true_run(~finite_mask)
                    )
                    window_means[(channel, window)] = row[f"{prefix}_mean"]
                mean_24 = window_means[(channel, 24)]
                for window in (1, 3, 6, 12):
                    recent_mean = window_means[(channel, window)]
                    row[f"g3_{channel}_w{window}h_mean_minus_24h"] = (
                        recent_mean - mean_24
                        if np.isfinite(recent_mean) and np.isfinite(mean_24)
                        else np.nan
                    )
            row["g6_hours_since_operation_start"] = (
                float((reference - operation_start) / pd.Timedelta(hours=1))
                if not pd.isna(operation_start)
                else np.nan
            )
            for channel in HOURLY_CHANNELS:
                all_values = pd.to_numeric(visible[channel], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                history_mean = (
                    float(np.nanmean(all_values))
                    if np.isfinite(all_values).any()
                    else np.nan
                )
                for days in (7, 14, 30):
                    recent = visible[
                        visible["clock"].gt(reference - pd.Timedelta(days=days))
                    ]
                    recent_values = pd.to_numeric(recent[channel], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    recent_mean = (
                        float(np.nanmean(recent_values))
                        if np.isfinite(recent_values).any()
                        else np.nan
                    )
                    row[f"g6_{channel}_recent_{days}d_mean_minus_history"] = (
                        recent_mean - history_mean
                        if np.isfinite(recent_mean) and np.isfinite(history_mean)
                        else np.nan
                    )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names_, index=samples.index)
