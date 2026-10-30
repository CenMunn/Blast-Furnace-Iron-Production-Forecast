from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import (
    EPSILON,
    HISTORY_EWMA_HALFLIVES,
    HISTORY_LAGS,
    HISTORY_ROLLING,
)


_METRICS = ("iron", "time", "rate")
_SOURCE_COLUMNS = {"iron": "tap_iron", "time": "tap_time_len", "rate": "_rate"}
_ROLLING_STATS = ("mean", "std", "median", "min", "max")


def _safe_rate(iron: np.ndarray, duration: np.ndarray) -> np.ndarray:
    iron_values = np.asarray(iron, dtype=np.float64)
    time_values = np.asarray(duration, dtype=np.float64)
    result = np.full(iron_values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(iron_values) & np.isfinite(time_values) & (np.abs(time_values) >= EPSILON)
    result[valid] = iron_values[valid] / time_values[valid]
    return result


def _prepare_history(tap_history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "spout_no",
        "tap_end_time",
        "tap_iron",
        "tap_time_len",
    }
    missing = sorted(required - set(tap_history.columns))
    if missing:
        raise ValueError(f"tap history missing required columns: {', '.join(missing)}")
    history = tap_history.copy()
    history["tap_end_time"] = pd.to_datetime(history["tap_end_time"], errors="raise").astype(
        "datetime64[ns]"
    )
    history["sample_id"] = history["sample_id"].astype(str)
    history["_rate"] = _safe_rate(history["tap_iron"], history["tap_time_len"])
    return history.sort_values(["tap_end_time", "sample_id"], kind="mergesort").reset_index(
        drop=True
    )


def _visible_history(
    history: pd.DataFrame,
    end_times: np.ndarray,
    sample_id: str,
    reference_time: pd.Timestamp,
) -> pd.DataFrame:
    stop = int(np.searchsorted(end_times, reference_time.to_datetime64(), side="right"))
    visible = history.iloc[:stop]
    return visible[visible["sample_id"].ne(sample_id)]


def _finite_tail(values: pd.Series, count: int | None = None) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if count is not None:
        array = array[-count:]
    return array


def _population_stat(values: np.ndarray, statistic: str) -> float:
    if values.size == 0:
        return np.nan
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "std":
        return float(np.std(values, ddof=0))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "min":
        return float(np.min(values))
    if statistic == "max":
        return float(np.max(values))
    raise ValueError(f"unsupported statistic: {statistic}")


def _ewma_last(values: np.ndarray, halflife: int) -> float:
    if values.size == 0:
        return np.nan
    alpha = 1.0 - np.exp(np.log(0.5) / float(halflife))
    result = float(values[0])
    for value in values[1:]:
        result = (1.0 - alpha) * result + alpha * float(value)
    return result


class CausalHistoryFeatureBuilder:
    """Build G1/G2/G5 features from labels visible by each reference time."""

    def __init__(self) -> None:
        self.feature_names_ = self._feature_names()

    @staticmethod
    def _feature_names() -> list[str]:
        names: list[str] = [
            "g1_visible_history_count",
            "g1_hours_since_last_tap_end",
        ]
        for metric in _METRICS:
            names.extend(f"g1_{metric}_lag_{lag}" for lag in HISTORY_LAGS)
            for window in HISTORY_ROLLING:
                names.extend(
                    f"g1_{metric}_roll_{window}_{statistic}"
                    for statistic in _ROLLING_STATS
                )
                names.append(f"g1_{metric}_roll_{window}_insufficient")
            names.extend(
                f"g1_{metric}_ewma_hl{halflife}"
                for halflife in HISTORY_EWMA_HALFLIVES
            )
            names.extend(
                [
                    f"g1_{metric}_diff_lag1_lag2",
                    f"g1_{metric}_change_rate_lag1_lag2",
                ]
            )
        names.extend(["g2_visible_history_count", "g2_hours_since_same_spout"])
        for metric in _METRICS:
            names.extend(f"g2_{metric}_lag_{lag}" for lag in (1, 2, 3))
            for window in (3, 5):
                names.extend(
                    [
                        f"g2_{metric}_roll_{window}_mean",
                        f"g2_{metric}_roll_{window}_median",
                        f"g2_{metric}_roll_{window}_std",
                        f"g2_{metric}_roll_{window}_insufficient",
                        f"g2_{metric}_roll_{window}_mean_diff_global",
                    ]
                )
        names.extend(
            [
                *(f"g5_rate_lag_{lag}" for lag in HISTORY_LAGS),
                *(f"g5_rate_roll_{window}_median" for window in HISTORY_ROLLING),
                *(f"g5_rate_ewma_hl{halflife}" for halflife in HISTORY_EWMA_HALFLIVES),
                "g5_iron_identity_residual_lag1",
                "g5_iron_vs_roll3_median",
                "g5_time_vs_roll3_median",
                "g5_rate_vs_roll3_median",
            ]
        )
        return names

    def build(self, samples: pd.DataFrame, tap_history: pd.DataFrame) -> pd.DataFrame:
        required = {"sample_id", "spout_no", "reference_time"}
        missing = sorted(required - set(samples.columns))
        if missing:
            raise ValueError(f"samples missing required columns: {', '.join(missing)}")
        if samples["reference_time"].isna().any():
            raise ValueError("samples contain null reference_time")

        history = _prepare_history(tap_history)
        end_times = history["tap_end_time"].to_numpy(dtype="datetime64[ns]")
        rows: list[dict[str, float]] = []
        for sample in samples.itertuples(index=False):
            sample_id = str(sample.sample_id)
            reference_time = pd.Timestamp(sample.reference_time)
            spout_no = int(sample.spout_no)
            visible = _visible_history(history, end_times, sample_id, reference_time)
            same_spout = visible[visible["spout_no"].eq(spout_no)]
            row: dict[str, float] = {
                "g1_visible_history_count": float(len(visible)),
                "g1_hours_since_last_tap_end": (
                    float((reference_time - visible.iloc[-1]["tap_end_time"]) / pd.Timedelta(hours=1))
                    if len(visible)
                    else np.nan
                ),
                "g2_visible_history_count": float(len(same_spout)),
                "g2_hours_since_same_spout": (
                    float((reference_time - same_spout.iloc[-1]["tap_end_time"]) / pd.Timedelta(hours=1))
                    if len(same_spout)
                    else np.nan
                ),
            }
            for metric in _METRICS:
                source = _SOURCE_COLUMNS[metric]
                values = _finite_tail(visible[source])
                same_values = _finite_tail(same_spout[source])
                for lag in HISTORY_LAGS:
                    row[f"g1_{metric}_lag_{lag}"] = (
                        float(values[-lag]) if values.size >= lag else np.nan
                    )
                for window in HISTORY_ROLLING:
                    recent = values[-window:]
                    for statistic in _ROLLING_STATS:
                        row[f"g1_{metric}_roll_{window}_{statistic}"] = _population_stat(
                            recent, statistic
                        )
                    row[f"g1_{metric}_roll_{window}_insufficient"] = float(
                        values.size < window
                    )
                for halflife in HISTORY_EWMA_HALFLIVES:
                    row[f"g1_{metric}_ewma_hl{halflife}"] = _ewma_last(values, halflife)
                if values.size >= 2:
                    difference = float(values[-1] - values[-2])
                    denominator = float(values[-2])
                    change = difference / denominator if abs(denominator) >= EPSILON else np.nan
                else:
                    difference = np.nan
                    change = np.nan
                row[f"g1_{metric}_diff_lag1_lag2"] = difference
                row[f"g1_{metric}_change_rate_lag1_lag2"] = change

                for lag in (1, 2, 3):
                    row[f"g2_{metric}_lag_{lag}"] = (
                        float(same_values[-lag]) if same_values.size >= lag else np.nan
                    )
                for window in (3, 5):
                    recent_same = same_values[-window:]
                    recent_global = values[-window:]
                    same_mean = _population_stat(recent_same, "mean")
                    global_mean = _population_stat(recent_global, "mean")
                    row[f"g2_{metric}_roll_{window}_mean"] = same_mean
                    row[f"g2_{metric}_roll_{window}_median"] = _population_stat(
                        recent_same, "median"
                    )
                    row[f"g2_{metric}_roll_{window}_std"] = _population_stat(
                        recent_same, "std"
                    )
                    row[f"g2_{metric}_roll_{window}_insufficient"] = float(
                        same_values.size < window
                    )
                    row[f"g2_{metric}_roll_{window}_mean_diff_global"] = (
                        same_mean - global_mean
                        if np.isfinite(same_mean) and np.isfinite(global_mean)
                        else np.nan
                    )

            for lag in HISTORY_LAGS:
                row[f"g5_rate_lag_{lag}"] = row[f"g1_rate_lag_{lag}"]
            for window in HISTORY_ROLLING:
                row[f"g5_rate_roll_{window}_median"] = row[
                    f"g1_rate_roll_{window}_median"
                ]
            for halflife in HISTORY_EWMA_HALFLIVES:
                row[f"g5_rate_ewma_hl{halflife}"] = row[
                    f"g1_rate_ewma_hl{halflife}"
                ]
            iron_lag = row["g1_iron_lag_1"]
            time_lag = row["g1_time_lag_1"]
            rate_lag = row["g1_rate_lag_1"]
            row["g5_iron_identity_residual_lag1"] = (
                iron_lag - time_lag * rate_lag
                if np.isfinite(iron_lag) and np.isfinite(time_lag) and np.isfinite(rate_lag)
                else np.nan
            )
            for metric in _METRICS:
                row[f"g5_{metric}_vs_roll3_median"] = (
                    row[f"g1_{metric}_lag_1"] - row[f"g1_{metric}_roll_3_median"]
                    if np.isfinite(row[f"g1_{metric}_lag_1"])
                    and np.isfinite(row[f"g1_{metric}_roll_3_median"])
                    else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=self.feature_names_, index=samples.index)


@dataclass
class CausalBaselineBuilder:
    fallback_: dict[str, float] | None = None

    def fit(
        self, train_samples: pd.DataFrame, tap_history: pd.DataFrame
    ) -> "CausalBaselineBuilder":
        del tap_history
        iron = pd.to_numeric(train_samples["tap_iron"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        duration = pd.to_numeric(
            train_samples["tap_time_len"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        rate = _safe_rate(iron, duration)
        values = {"tap_iron": iron, "tap_time_len": duration, "rate": rate}
        fallback: dict[str, float] = {}
        for name, array in values.items():
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise ValueError(f"cannot fit baseline fallback for {name}")
            fallback[name] = float(np.median(finite))
        self.fallback_ = fallback
        return self

    def transform(
        self,
        samples: pd.DataFrame,
        tap_history: pd.DataFrame,
        fold: int,
        split: str,
    ) -> pd.DataFrame:
        if self.fallback_ is None:
            raise RuntimeError("CausalBaselineBuilder is not fitted")
        history = _prepare_history(tap_history)
        end_times = history["tap_end_time"].to_numpy(dtype="datetime64[ns]")
        rows: list[dict[str, object]] = []
        for sample in samples.itertuples(index=False):
            sample_id = str(sample.sample_id)
            reference_time = pd.Timestamp(sample.reference_time)
            visible = _visible_history(history, end_times, sample_id, reference_time)
            same_spout = visible[visible["spout_no"].eq(int(sample.spout_no))]
            row: dict[str, object] = {
                "fold": int(fold),
                "split": split,
                "sample_id": sample_id,
            }
            self._write_previous(row, "baseline_prev", visible)
            self._write_previous(row, "baseline_same_spout_prev", same_spout)
            for window in (3, 5):
                self._write_median(row, f"baseline_median_{window}", visible, window)
            for halflife in HISTORY_EWMA_HALFLIVES:
                self._write_ewma(row, f"baseline_ewma_{halflife}", visible, halflife)
            rows.append(row)
        return pd.DataFrame(rows, index=samples.index)

    def _write_previous(
        self, row: dict[str, object], prefix: str, history: pd.DataFrame
    ) -> None:
        available = len(history) > 0
        row[f"{prefix}_fallback"] = int(not available)
        for name, source in (
            ("tap_iron", "tap_iron"),
            ("tap_time_len", "tap_time_len"),
            ("rate", "_rate"),
        ):
            values = _finite_tail(history[source])
            row[f"{prefix}_{name}"] = (
                float(values[-1]) if values.size else self.fallback_[name]
            )

    def _write_median(
        self,
        row: dict[str, object],
        prefix: str,
        history: pd.DataFrame,
        window: int,
    ) -> None:
        row[f"{prefix}_fallback"] = int(len(history) == 0)
        for name, source in (
            ("tap_iron", "tap_iron"),
            ("tap_time_len", "tap_time_len"),
            ("rate", "_rate"),
        ):
            values = _finite_tail(history[source], window)
            row[f"{prefix}_{name}"] = (
                float(np.median(values)) if values.size else self.fallback_[name]
            )

    def _write_ewma(
        self,
        row: dict[str, object],
        prefix: str,
        history: pd.DataFrame,
        halflife: int,
    ) -> None:
        row[f"{prefix}_fallback"] = int(len(history) == 0)
        for name, source in (
            ("tap_iron", "tap_iron"),
            ("tap_time_len", "tap_time_len"),
            ("rate", "_rate"),
        ):
            values = _finite_tail(history[source])
            estimate = _ewma_last(values, halflife)
            row[f"{prefix}_{name}"] = (
                estimate if np.isfinite(estimate) else self.fallback_[name]
            )
