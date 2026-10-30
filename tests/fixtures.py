from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_source_tree(root: Path) -> Path:
    train = root / "train"
    test = root / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "sample_id": ["S0001", "S0002"],
            "tap_no": [101, 102],
            "spout_no": [1, 2],
            "reference_time": ["2024-03-01 02:00:00", "2024-03-01 05:00:00"],
            "tap_iron": [980.0, 1020.0],
            "tap_time_len": [120.0, 125.0],
        }
    ).to_csv(train / "train_samples.csv", index=False)

    hourly_rows = {
        "clock": ["2024-03-01 00:00:00", "2024-03-01 01:00:00"],
        "air_volume": [4100.0, 4200.0],
        "cold_air_press": [390.0, 391.0],
        "hot_air_press": [380.0, 381.0],
        "oxygen": [10000.0, 10100.0],
        "hot_air_temp": [1180.0, 1181.0],
        "coal_rate": [40.0, 41.0],
        "humidity": [13.0, 14.0],
        "gas_rate": [48.0, 49.0],
        "furnace_top_press": [220.0, 221.0],
        "upper_press_diff": [50.0, 51.0],
        "lower_press_diff": [110.0, 111.0],
        "total_press_diff": [160.0, 162.0],
        "air_press_ratio": [25.0, 26.0],
        "furnace_top_temp_avg": [155.0, 156.0],
        "air_speed": [285.0, 286.0],
        "furnace_throat_temp": [205.0, 206.0],
    }
    pd.DataFrame(hourly_rows).to_csv(train / "operation_hourly.csv", index=False)

    pd.DataFrame(
        {
            "cal_time": ["2024-02-29 20:00:00", "2024-03-01 01:00:00"],
            "pig": [1.0, 2.0],
            "all_quality": [55.0, 56.0],
            "consumption": [1.5, 1.6],
            "fuel_rate": [500.0, 501.0],
            "coke_rate": [320.0, 321.0],
        }
    ).to_csv(train / "burden_change.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": ["S0001", "S0002"],
            "tap_no": [101, 102],
            "spout_no": [1, 2],
            "reference_time": ["2024-03-01 02:00:00", "2024-03-01 05:00:00"],
            "tap_end_time": ["2024-03-01 04:00:00", "2024-03-01 07:00:00"],
            "tap_iron": [980.0, 1020.0],
            "tap_time_len": [120.0, 125.0],
            "slag_time_len": [80.0, 82.0],
            "break_slag_time_len": [5.0, 6.0],
            "ladle_count": [4, 5],
        }
    ).to_csv(train / "tap_history_train.csv", index=False)

    for name, sample_id, reference_time in (
        ("test_a_samples.csv", "A0001", "2024-12-01 02:00:00"),
        ("test_b_samples.csv", "B0001", "2025-02-01 02:00:00"),
    ):
        pd.DataFrame(
            {
                "sample_id": [sample_id],
                "tap_no": [9001],
                "spout_no": [1],
                "reference_time": [reference_time],
            }
        ).to_csv(test / name, index=False)
    return root


def duplicate_first_train_row(root: Path) -> None:
    path = root / "train" / "train_samples.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)


def append_conflicting_hourly_row(root: Path) -> None:
    path = root / "train" / "operation_hourly.csv"
    frame = pd.read_csv(path)
    conflict = frame.iloc[[0]].copy()
    conflict.loc[:, "air_volume"] = 9999.0
    pd.concat([frame, conflict], ignore_index=True).to_csv(path, index=False)


def append_exact_burden_duplicate(root: Path) -> None:
    path = root / "train" / "burden_change.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)


def write_monthly_source_tree(root: Path) -> Path:
    write_source_tree(root)
    train_root = root / "train"
    test_root = root / "test"
    train_times = [
        pd.Period(f"2024-{month:02d}", freq="M").to_timestamp()
        + pd.Timedelta(days=14, hours=12)
        for month in range(3, 12)
    ]
    train_samples = pd.DataFrame(
        {
            "sample_id": [f"S{month:02d}" for month in range(3, 12)],
            "tap_no": list(range(103, 112)),
            "spout_no": [1, 2, 3, 4, 1, 2, 3, 4, 1],
            "reference_time": train_times,
            "tap_iron": [1000.0 + 10.0 * index for index in range(9)],
            "tap_time_len": [120.0 + index for index in range(9)],
        }
    )
    train_samples.to_csv(train_root / "train_samples.csv", index=False)

    tap_history = train_samples.copy()
    tap_history["tap_end_time"] = tap_history["reference_time"] + pd.Timedelta(hours=1)
    tap_history["slag_time_len"] = 80.0 + np.arange(9, dtype=float)
    tap_history["break_slag_time_len"] = 5.0 + np.arange(9, dtype=float)
    tap_history["ladle_count"] = 4.0 + (np.arange(9) % 2)
    tap_history.to_csv(train_root / "tap_history_train.csv", index=False)

    test_a_time = pd.Timestamp("2024-12-15 12:00:00")
    test_b_time = pd.Timestamp("2025-02-15 12:00:00")
    pd.DataFrame(
        {
            "sample_id": ["A01"],
            "tap_no": [201],
            "spout_no": [2],
            "reference_time": [test_a_time],
        }
    ).to_csv(test_root / "test_a_samples.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["B01"],
            "tap_no": [301],
            "spout_no": [3],
            "reference_time": [test_b_time],
        }
    ).to_csv(test_root / "test_b_samples.csv", index=False)

    all_references = [*train_times, test_a_time, test_b_time]
    hourly_rows: list[dict[str, object]] = []
    burden_rows: list[dict[str, object]] = []
    hourly_channels = [
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
    ]
    for reference_index, reference in enumerate(all_references):
        for hour_offset in (-1, 0):
            row: dict[str, object] = {"clock": reference + pd.Timedelta(hours=hour_offset)}
            for channel_index, channel in enumerate(hourly_channels):
                row[channel] = 100.0 * (channel_index + 1) + reference_index + hour_offset
            hourly_rows.append(row)
        for event_offset in (-2, 0):
            burden_rows.append(
                {
                    "cal_time": reference + pd.Timedelta(hours=event_offset),
                    "pig": 1.0 + reference_index,
                    "all_quality": 50.0 + reference_index,
                    "consumption": 1.5 + 0.1 * reference_index,
                    "fuel_rate": 500.0 + reference_index,
                    "coke_rate": 300.0 + reference_index,
                }
            )
    pd.DataFrame(hourly_rows).to_csv(train_root / "operation_hourly.csv", index=False)
    pd.DataFrame(burden_rows).to_csv(train_root / "burden_change.csv", index=False)
    return root


def timeaware_schedule_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["S0", "S1", "S2", "S3"],
            "tap_no": [10, 11, 12, 13],
            "spout_no": [1, 2, 1, 1],
            "reference_time": pd.to_datetime(
                [
                    "2024-07-01 00:00:00",
                    "2024-07-01 02:00:00",
                    "2024-07-01 05:00:00",
                    "2024-07-01 09:00:00",
                ]
            ).astype("datetime64[ns]"),
        }
    )


def timeaware_burden_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cal_time": pd.to_datetime(
                [
                    "2024-06-30 12:00:00",
                    "2024-06-30 20:00:00",
                    "2024-07-01 00:00:00",
                    "2024-07-01 00:00:00",
                    "2024-07-01 06:00:00",
                    "2024-07-01 10:00:00",
                ]
            ).astype("datetime64[ns]"),
            "pig": [1.0, 1.0, 2.0, 3.0, 3.0, 9.0],
            "all_quality": [50.0, 50.0, 51.0, 52.0, 52.0, 99.0],
            "consumption": [1.0, 1.0, 1.1, 1.2, 1.2, 9.9],
            "fuel_rate": [500.0, 500.0, 501.0, 502.0, 502.0, 999.0],
            "coke_rate": [300.0, 300.0, 301.0, 302.0, 302.0, 999.0],
        }
    )


def write_timeaware_monthly_source_tree(root: Path) -> Path:
    write_monthly_source_tree(root)
    path = root / "train" / "burden_change.csv"
    burden = pd.read_csv(path)
    duplicate = burden.iloc[[0]].copy()
    duplicate.loc[:, "pig"] = float(duplicate.iloc[0]["pig"]) + 100.0
    pd.concat([burden.iloc[[0]], duplicate, burden.iloc[1:]], ignore_index=True).to_csv(
        path, index=False
    )
    return root
