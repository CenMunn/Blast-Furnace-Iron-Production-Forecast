from __future__ import annotations

from pathlib import Path


SOURCE_FILES: dict[str, Path] = {
    "train_samples": Path("train/train_samples.csv"),
    "operation_hourly": Path("train/operation_hourly.csv"),
    "burden_change": Path("train/burden_change.csv"),
    "tap_history_train": Path("train/tap_history_train.csv"),
    "test_a_samples": Path("test/test_a_samples.csv"),
    "test_b_samples": Path("test/test_b_samples.csv"),
}

HOURLY_CHANNELS: tuple[str, ...] = (
    "air_volume",
    "cold_air_press",
    "hot_air_press",
    "oxygen",
    "hot_air_temp",
    "coal_rate",
    "humidity",
    "gas_rate",
    "furnace_top_press",
    "upper_press_diff",
    "lower_press_diff",
    "total_press_diff",
    "air_press_ratio",
    "furnace_top_temp_avg",
    "air_speed",
    "furnace_throat_temp",
)

SHUTDOWN_CORE_CHANNELS: tuple[str, ...] = (
    "air_volume",
    "cold_air_press",
    "hot_air_press",
    "oxygen",
    "hot_air_temp",
    "gas_rate",
    "furnace_top_press",
    "upper_press_diff",
    "lower_press_diff",
    "total_press_diff",
    "air_press_ratio",
    "furnace_top_temp_avg",
    "air_speed",
    "furnace_throat_temp",
)

BURDEN_CHANNELS: tuple[str, ...] = (
    "pig",
    "all_quality",
    "consumption",
    "fuel_rate",
    "coke_rate",
)

TARGET_COLUMNS: tuple[str, ...] = ("tap_iron", "tap_time_len")
TRAIN_MONTHS: tuple[str, ...] = (
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
)
VALID_MONTHS: tuple[str, ...] = (
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
)
HOURLY_WINDOWS: tuple[int, ...] = (1, 3, 6, 12, 24)
HISTORY_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10)
HISTORY_ROLLING: tuple[int, ...] = (3, 5, 10)
HISTORY_EWMA_HALFLIVES: tuple[int, ...] = (2, 4, 8)
BURDEN_RECENT_COUNTS: tuple[int, ...] = (1, 3, 5, 10)
HOURLY_SEQUENCE_LENGTH = 24
TAP_SEQUENCE_LENGTH = 10
BURDEN_SEQUENCE_LENGTH = 10
EPSILON = 1e-6
RANDOM_SEED = 20260801

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "train_samples": (
        "sample_id",
        "tap_no",
        "spout_no",
        "reference_time",
        "tap_iron",
        "tap_time_len",
    ),
    "test_a_samples": ("sample_id", "tap_no", "spout_no", "reference_time"),
    "test_b_samples": ("sample_id", "tap_no", "spout_no", "reference_time"),
    "operation_hourly": ("clock", *HOURLY_CHANNELS),
    "burden_change": ("cal_time", *BURDEN_CHANNELS),
    "tap_history_train": (
        "sample_id",
        "tap_no",
        "spout_no",
        "reference_time",
        "tap_end_time",
        "tap_iron",
        "tap_time_len",
        "slag_time_len",
        "break_slag_time_len",
        "ladle_count",
    ),
}

TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "train_samples": ("reference_time",),
    "test_a_samples": ("reference_time",),
    "test_b_samples": ("reference_time",),
    "operation_hourly": ("clock",),
    "burden_change": ("cal_time",),
    "tap_history_train": ("reference_time", "tap_end_time"),
}
