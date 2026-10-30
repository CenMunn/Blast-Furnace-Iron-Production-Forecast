from __future__ import annotations

import numpy as np
import pandas as pd


class StaticFeatureBuilder:
    def __init__(self, dataset_start: pd.Timestamp | None = None) -> None:
        self.dataset_start = pd.Timestamp(dataset_start) if dataset_start is not None else None
        self.feature_names_ = [
            "g0_tap_no",
            "g0_spout_no",
            "g0_hour_sin",
            "g0_hour_cos",
            "g0_weekday_sin",
            "g0_weekday_cos",
            "g0_month_sin",
            "g0_month_cos",
            "g0_days_since_dataset_start",
            "g0_hourly_valid_length_24h",
            "g0_burden_valid_length_10",
            "g0_tap_valid_length_10",
            "g0_total_missing_count",
        ]

    def build(
        self, samples: pd.DataFrame, stream_counts: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        times = pd.to_datetime(samples["reference_time"], errors="raise")
        start = self.dataset_start if self.dataset_start is not None else pd.Timestamp(times.min())
        hour = times.dt.hour + times.dt.minute / 60.0 + times.dt.second / 3600.0
        weekday = times.dt.weekday.astype(float)
        month = times.dt.month.astype(float) - 1.0
        result = pd.DataFrame(
            {
                "g0_tap_no": pd.to_numeric(samples["tap_no"], errors="coerce"),
                "g0_spout_no": pd.to_numeric(samples["spout_no"], errors="coerce"),
                "g0_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
                "g0_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
                "g0_weekday_sin": np.sin(2.0 * np.pi * weekday / 7.0),
                "g0_weekday_cos": np.cos(2.0 * np.pi * weekday / 7.0),
                "g0_month_sin": np.sin(2.0 * np.pi * month / 12.0),
                "g0_month_cos": np.cos(2.0 * np.pi * month / 12.0),
                "g0_days_since_dataset_start": (times - start) / pd.Timedelta(days=1),
            },
            index=samples.index,
        )
        for column in (
            "g0_hourly_valid_length_24h",
            "g0_burden_valid_length_10",
            "g0_tap_valid_length_10",
            "g0_total_missing_count",
        ):
            result[column] = (
                pd.to_numeric(stream_counts[column], errors="coerce")
                if stream_counts is not None and column in stream_counts
                else np.nan
            )
        return result.loc[:, self.feature_names_]
