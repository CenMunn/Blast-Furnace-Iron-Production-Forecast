from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_snappy_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic Parquet file with Snappy on every column."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        destination,
        compression="snappy",
        use_dictionary=False,
        write_statistics=True,
    )
