from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import BURDEN_CHANNELS, HOURLY_CHANNELS, SHUTDOWN_CORE_CHANNELS


class LegacyFeatureBuilder:
    """Rebuild the prior 288-feature contract with the final <= T boundary."""

    def __init__(self) -> None:
        names: list[str] = []
        for channel in HOURLY_CHANNELS:
            for window in (1, 6, 12, 24):
                names.extend(
                    f"g0_legacy_{channel}_w{window}h_{stat}"
                    for stat in ("mean", "std", "trend", "last")
                )
        for window in (1, 6, 12, 24):
            names.extend(
                [
                    f"g0_legacy_coverage_hours_w{window}h",
                    f"g0_legacy_coverage_ratio_w{window}h",
                    f"g0_legacy_shutdown_fraction_w{window}h",
                ]
            )
        names.extend(
            [
                "g0_legacy_spout_no",
                "g0_legacy_hour_sin",
                "g0_legacy_hour_cos",
                "g0_legacy_weekday_sin",
                "g0_legacy_weekday_cos",
                "g0_legacy_month_sin",
                "g0_legacy_month_cos",
                "g0_legacy_event_history_count",
                "g0_legacy_event_lt3_flag",
                "g0_legacy_hours_since_last_event",
                *(f"g0_legacy_event_last_{channel}" for channel in BURDEN_CHANNELS),
                *(f"g0_legacy_event_last3_sum_{channel}" for channel in BURDEN_CHANNELS),
            ]
        )
        if len(names) != 288 or len(set(names)) != 288:
            raise AssertionError("legacy feature registry must contain 288 unique names")
        self.feature_names_ = names

    def build(
        self,
        samples: pd.DataFrame,
        operation_hourly: pd.DataFrame,
        burden_change: pd.DataFrame,
    ) -> pd.DataFrame:
        hourly = operation_hourly.copy().sort_values("clock", kind="mergesort")
        hourly["clock"] = pd.to_datetime(hourly["clock"], errors="raise").astype(
            "datetime64[ns]"
        )
        events = burden_change.copy().sort_values("cal_time", kind="mergesort")
        events["cal_time"] = pd.to_datetime(events["cal_time"], errors="raise").astype(
            "datetime64[ns]"
        )
        event_times = events["cal_time"].to_numpy(dtype="datetime64[ns]")
        rows: list[dict[str, float]] = []
        for sample in samples.itertuples(index=False):
            reference = pd.Timestamp(sample.reference_time)
            row: dict[str, float] = {}
            visible_hourly = hourly[hourly["clock"].le(reference)]
            for window in (1, 6, 12, 24):
                window_rows = visible_hourly[
                    visible_hourly["clock"].gt(reference - pd.Timedelta(hours=window))
                ]
                for channel in HOURLY_CHANNELS:
                    values = pd.to_numeric(window_rows[channel], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    times = (
                        (window_rows["clock"] - reference) / pd.Timedelta(hours=1)
                    ).to_numpy(dtype=np.float64)
                    finite = np.isfinite(values)
                    prefix = f"g0_legacy_{channel}_w{window}h"
                    if finite.any():
                        selected = values[finite]
                        row[f"{prefix}_mean"] = float(np.mean(selected))
                        row[f"{prefix}_std"] = float(np.std(selected, ddof=0))
                        row[f"{prefix}_trend"] = (
                            float(np.polyfit(times[finite], selected, 1)[0])
                            if finite.sum() >= 2
                            else 0.0
                        )
                        row[f"{prefix}_last"] = float(selected[-1])
                    else:
                        row[f"{prefix}_mean"] = np.nan
                        row[f"{prefix}_std"] = np.nan
                        row[f"{prefix}_trend"] = np.nan
                        row[f"{prefix}_last"] = np.nan
                coverage = int(window_rows["clock"].nunique())
                core = window_rows.loc[:, list(SHUTDOWN_CORE_CHANNELS)].apply(
                    pd.to_numeric, errors="coerce"
                )
                shutdown = (
                    (core.notna().all(axis=1) & core.le(0.0).all(axis=1)).mean()
                    if len(core)
                    else np.nan
                )
                row[f"g0_legacy_coverage_hours_w{window}h"] = float(coverage)
                row[f"g0_legacy_coverage_ratio_w{window}h"] = float(
                    min(coverage / window, 1.0)
                )
                row[f"g0_legacy_shutdown_fraction_w{window}h"] = float(shutdown)
            hour = reference.hour + reference.minute / 60.0 + reference.second / 3600.0
            row["g0_legacy_spout_no"] = float(sample.spout_no)
            row["g0_legacy_hour_sin"] = float(np.sin(2.0 * np.pi * hour / 24.0))
            row["g0_legacy_hour_cos"] = float(np.cos(2.0 * np.pi * hour / 24.0))
            row["g0_legacy_weekday_sin"] = float(
                np.sin(2.0 * np.pi * reference.weekday() / 7.0)
            )
            row["g0_legacy_weekday_cos"] = float(
                np.cos(2.0 * np.pi * reference.weekday() / 7.0)
            )
            row["g0_legacy_month_sin"] = float(
                np.sin(2.0 * np.pi * (reference.month - 1) / 12.0)
            )
            row["g0_legacy_month_cos"] = float(
                np.cos(2.0 * np.pi * (reference.month - 1) / 12.0)
            )
            stop = int(
                np.searchsorted(event_times, reference.to_datetime64(), side="right")
            )
            visible_events = events.iloc[:stop]
            row["g0_legacy_event_history_count"] = float(len(visible_events))
            row["g0_legacy_event_lt3_flag"] = float(len(visible_events) < 3)
            row["g0_legacy_hours_since_last_event"] = (
                float((reference - visible_events.iloc[-1]["cal_time"]) / pd.Timedelta(hours=1))
                if len(visible_events)
                else np.nan
            )
            for channel in BURDEN_CHANNELS:
                values = pd.to_numeric(visible_events[channel], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                finite = values[np.isfinite(values)]
                row[f"g0_legacy_event_last_{channel}"] = (
                    float(finite[-1]) if finite.size else np.nan
                )
                row[f"g0_legacy_event_last3_sum_{channel}"] = (
                    float(np.sum(finite[-3:])) if finite.size else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names_, index=samples.index)
