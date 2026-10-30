from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.writers import write_snappy_parquet


class ParquetWriterTests(unittest.TestCase):
    def test_writer_preserves_float32_and_uses_snappy_for_every_column(self):
        frame = pd.DataFrame(
            {
                "sample_id": ["S01", "S02"],
                "fold": np.asarray([1, 1], dtype=np.int8),
                "split": ["train", "valid"],
                "reference_time": pd.to_datetime(
                    ["2024-06-01 00:00:00", "2024-07-01 00:00:00"]
                ).astype("datetime64[ns]"),
                "tap_iron": np.asarray([1000.0, 1010.0], dtype=np.float32),
                "g0_value": np.asarray([0.5, -0.5], dtype=np.float32),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold.parquet"

            write_snappy_parquet(frame, path)

            parquet = pq.ParquetFile(path)
            try:
                self.assertEqual("float", str(parquet.schema_arrow.field("g0_value").type))
                for row_group in range(parquet.metadata.num_row_groups):
                    for column in range(parquet.metadata.row_group(row_group).num_columns):
                        self.assertEqual(
                            "SNAPPY",
                            parquet.metadata.row_group(row_group).column(column).compression,
                        )
            finally:
                parquet.close()


if __name__ == "__main__":
    unittest.main()
