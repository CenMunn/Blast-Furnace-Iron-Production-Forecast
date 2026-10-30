from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import BURDEN_CHANNELS, BURDEN_RECENT_COUNTS


_EVENT_STATS = ("last", "mean", "std", "median", "min", "max", "sum", "trend")


class BurdenFeatureBuilder:
    def __init__(self) -> None:
        names = [
            "g4_event_history_count",
            "g4_hours_since_last_event",
            "g4_event_gap_mean_hours",
            "g4_event_gap_std_hours",
        ]
        for count in BURDEN_RECENT_COUNTS:
            names.extend([f"g4_effective_length_{count}", f"g4_insufficient_{count}"])
        for channel in BURDEN_CHANNELS:
            for count in BURDEN_RECENT_COUNTS:
                names.extend(
                    f"g4_{channel}_recent_{count}_{statistic}"
                    for statistic in _EVENT_STATS
                )
        self.feature_names_ = names

    def build(self, samples: pd.DataFrame, burden_change: pd.DataFrame) -> pd.DataFrame:
        events = burden_change.copy()
        events["cal_time"] = pd.to_datetime(events["cal_time"], errors="raise").astype(
            "datetime64[ns]"
        )
        events = events.sort_values("cal_time", kind="mergesort").reset_index(drop=True)
        event_times = events["cal_time"].to_numpy(dtype="datetime64[ns]")
        rows: list[dict[str, float]] = []
        for sample in samples.itertuples(index=False):
            reference = pd.Timestamp(sample.reference_time)
            stop = int(
                np.searchsorted(event_times, reference.to_datetime64(), side="right")
            )
            visible = events.iloc[:stop]
            row: dict[str, float] = {
                "g4_event_history_count": float(len(visible)),
                "g4_hours_since_last_event": (
                    float((reference - visible.iloc[-1]["cal_time"]) / pd.Timedelta(hours=1))
                    if len(visible)
                    else np.nan
                ),
            }
            recent_times = visible["cal_time"].iloc[-10:]
            gaps = np.diff(recent_times.to_numpy(dtype="datetime64[ns]")).astype(
                "timedelta64[s]"
            ).astype(np.float64) / 3600.0
            row["g4_event_gap_mean_hours"] = float(np.mean(gaps)) if gaps.size else np.nan
            row["g4_event_gap_std_hours"] = (
                float(np.std(gaps, ddof=0)) if gaps.size else np.nan
            )
            for count in BURDEN_RECENT_COUNTS:
                row[f"g4_effective_length_{count}"] = float(min(len(visible), count))
                row[f"g4_insufficient_{count}"] = float(len(visible) < count)
            for channel in BURDEN_CHANNELS:
                all_values = pd.to_numeric(visible[channel], errors="coerce")
                for count in BURDEN_RECENT_COUNTS:
                    values = all_values.iloc[-count:].to_numpy(dtype=np.float64)
                    finite = values[np.isfinite(values)]
                    prefix = f"g4_{channel}_recent_{count}"
                    if finite.size:
                        row[f"{prefix}_last"] = float(finite[-1])
                        row[f"{prefix}_mean"] = float(np.mean(finite))
                        row[f"{prefix}_std"] = float(np.std(finite, ddof=0))
                        row[f"{prefix}_median"] = float(np.median(finite))
                        row[f"{prefix}_min"] = float(np.min(finite))
                        row[f"{prefix}_max"] = float(np.max(finite))
                        row[f"{prefix}_sum"] = float(np.sum(finite))
                        row[f"{prefix}_trend"] = (
                            float(np.polyfit(np.arange(finite.size), finite, 1)[0])
                            if finite.size >= 2
                            else 0.0
                        )
                    else:
                        for statistic in _EVENT_STATS:
                            row[f"{prefix}_{statistic}"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names_, index=samples.index)
