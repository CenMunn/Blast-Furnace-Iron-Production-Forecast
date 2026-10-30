from __future__ import annotations

import numpy as np
import pandas as pd

from src.burden_features import BurdenFeatureBuilder
from src.constants import HOURLY_CHANNELS
from src.data_io import RawTables
from src.history_features import CausalHistoryFeatureBuilder
from src.hourly_features import HourlyFeatureBuilder
from src.legacy_features import LegacyFeatureBuilder
from src.static_features import StaticFeatureBuilder


class FeatureAssembler:
    def __init__(self) -> None:
        self.legacy = LegacyFeatureBuilder()
        self.history = CausalHistoryFeatureBuilder()
        self.hourly = HourlyFeatureBuilder()
        self.burden = BurdenFeatureBuilder()
        self.feature_names_: list[str] | None = None

    def build_raw(self, samples: pd.DataFrame, tables: RawTables) -> pd.DataFrame:
        legacy = self.legacy.build(samples, tables.operation_hourly, tables.burden_change)
        history = self.history.build(samples, tables.tap_history_train)
        hourly = self.hourly.build(samples, tables.operation_hourly)
        burden = self.burden.build(samples, tables.burden_change)
        stream_counts = pd.DataFrame(index=samples.index)
        stream_counts["g0_hourly_valid_length_24h"] = hourly[
            [f"g3_{channel}_w24h_valid_count" for channel in HOURLY_CHANNELS]
        ].max(axis=1)
        stream_counts["g0_burden_valid_length_10"] = burden["g4_effective_length_10"]
        stream_counts["g0_tap_valid_length_10"] = history[
            "g1_visible_history_count"
        ].clip(upper=10)
        static_builder = StaticFeatureBuilder(
            dataset_start=pd.Timestamp(tables.train_samples["reference_time"].min())
        )
        static = static_builder.build(samples, stream_counts)
        interactions = pd.DataFrame(index=samples.index)
        for channel in HOURLY_CHANNELS:
            interactions[f"g5_rate_lag1_x_{channel}_w1h_mean"] = (
                history["g1_rate_lag_1"] * hourly[f"g3_{channel}_w1h_mean"]
            )
        parts = [legacy, static, history, hourly, burden, interactions]
        raw_without_missing_count = pd.concat(parts, axis=1)
        static["g0_total_missing_count"] = raw_without_missing_count.isna().sum(axis=1)
        raw = pd.concat([legacy, static, history, hourly, burden, interactions], axis=1)
        if raw.columns.duplicated().any():
            duplicates = raw.columns[raw.columns.duplicated()].tolist()
            raise ValueError(f"duplicate engineered feature names: {duplicates}")
        invalid_names = [
            column
            for column in raw.columns
            if not column.startswith(tuple(f"g{group}_" for group in range(7)))
        ]
        if invalid_names:
            raise ValueError(f"feature names missing G0-G6 prefix: {invalid_names}")
        if any(target in raw.columns for target in ("tap_iron", "tap_time_len")):
            raise ValueError("current targets entered raw feature matrix")
        self.feature_names_ = raw.columns.tolist()
        return raw
