from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.constants import TRAIN_MONTHS, VALID_MONTHS
from src.data_io import RawTables


@dataclass(frozen=True)
class FoldDefinition:
    fold: int
    train_months: list[str]
    valid_month: str
    train_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


class StrictExpandingMonthSplit:
    """Create the fixed March-to-November expanding-window validation folds."""

    def split(self, samples: pd.DataFrame) -> list[FoldDefinition]:
        required = {"sample_id", "reference_time"}
        missing_columns = sorted(required - set(samples.columns))
        if missing_columns:
            raise ValueError(f"samples missing required columns: {', '.join(missing_columns)}")
        if samples["sample_id"].duplicated().any():
            raise ValueError("samples contain duplicate sample_id")
        if samples["reference_time"].isna().any():
            raise ValueError("samples contain null reference_time")

        normalized = samples.loc[:, ["sample_id", "reference_time"]].copy()
        normalized["reference_time"] = pd.to_datetime(
            normalized["reference_time"], errors="raise"
        ).astype("datetime64[ns]")
        normalized["sample_id"] = normalized["sample_id"].astype(str)
        normalized = normalized.sort_values(
            ["reference_time", "sample_id"], kind="mergesort"
        ).reset_index(drop=True)
        normalized["month"] = normalized["reference_time"].dt.strftime("%Y-%m")

        observed_months = set(normalized["month"])
        required_months = set(TRAIN_MONTHS)
        missing_months = [month for month in TRAIN_MONTHS if month not in observed_months]
        unexpected_months = sorted(observed_months - required_months)
        if missing_months:
            raise ValueError(f"missing required training months: {', '.join(missing_months)}")
        if unexpected_months:
            raise ValueError(f"unexpected training months: {', '.join(unexpected_months)}")

        folds: list[FoldDefinition] = []
        for fold_number, valid_month in enumerate(VALID_MONTHS, start=1):
            train_months = [month for month in TRAIN_MONTHS if month < valid_month]
            train = normalized[normalized["month"].isin(train_months)]
            valid = normalized[normalized["month"].eq(valid_month)]
            if train.empty or valid.empty:
                raise ValueError(f"fold {fold_number} has an empty train or validation partition")
            train_end = pd.Timestamp(train["reference_time"].max())
            valid_start = pd.Timestamp(valid["reference_time"].min())
            if not train_end < valid_start:
                raise ValueError(
                    f"fold {fold_number} violates max(train_time) < min(valid_time): "
                    f"{train_end} >= {valid_start}"
                )
            train_ids = tuple(train["sample_id"].tolist())
            valid_ids = tuple(valid["sample_id"].tolist())
            if not set(train_ids).isdisjoint(valid_ids):
                raise ValueError(f"fold {fold_number} train and validation sample_id overlap")
            folds.append(
                FoldDefinition(
                    fold=fold_number,
                    train_months=train_months,
                    valid_month=valid_month,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    train_start=pd.Timestamp(train["reference_time"].min()),
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=pd.Timestamp(valid["reference_time"].max()),
                )
            )
        return folds


def build_context_index(
    tables: RawTables, folds: Sequence[FoldDefinition]
) -> pd.DataFrame:
    train_lookup = tables.train_samples.set_index("sample_id", drop=False)
    contexts: list[tuple[str, int, str, pd.DataFrame, str]] = []
    for fold in folds:
        contexts.append(
            (
                f"fold_{fold.fold}_train",
                fold.fold,
                "train",
                train_lookup.loc[list(fold.train_ids)].reset_index(drop=True),
                f"folds/fold_{fold.fold}_train.parquet",
            )
        )
        contexts.append(
            (
                f"fold_{fold.fold}_valid",
                fold.fold,
                "valid",
                train_lookup.loc[list(fold.valid_ids)].reset_index(drop=True),
                f"folds/fold_{fold.fold}_valid.parquet",
            )
        )
    contexts.extend(
        [
            ("full_train", 0, "train", tables.train_samples, "folds/full_train.parquet"),
            ("test_a", 0, "test_a", tables.test_a_samples, "folds/test_a.parquet"),
            ("test_b", 0, "test_b", tables.test_b_samples, "folds/test_b.parquet"),
        ]
    )

    rows: list[dict[str, object]] = []
    array_row = 0
    for context, fold, split, frame, parquet_file in contexts:
        ordered = frame.sort_values(
            ["reference_time", "sample_id"], kind="mergesort"
        ).reset_index(drop=True)
        for parquet_row, sample in ordered.iterrows():
            rows.append(
                {
                    "array_row": array_row,
                    "data_context": context,
                    "fold": fold,
                    "split": split,
                    "sample_id": str(sample["sample_id"]),
                    "reference_time": pd.Timestamp(sample["reference_time"]),
                    "parquet_file": parquet_file,
                    "parquet_row": int(parquet_row),
                }
            )
            array_row += 1
    index = pd.DataFrame(rows)
    index["array_row"] = index["array_row"].astype("int64")
    index["fold"] = index["fold"].astype("int8")
    index["parquet_row"] = index["parquet_row"].astype("int64")
    index["reference_time"] = pd.to_datetime(index["reference_time"]).astype(
        "datetime64[ns]"
    )
    return index
