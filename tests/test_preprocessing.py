from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import FoldPreprocessor


def raw_train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g0_binary_flag": [0.0, 1.0],
            "g3_value": [1.0, 3.0],
            "g3_all_missing": [np.nan, np.nan],
            "g3_zero_variance": [5.0, 5.0],
            "g4_level": [10.0, 14.0],
        }
    )


class FoldPreprocessorTests(unittest.TestCase):
    def test_validation_extreme_does_not_change_fitted_state(self):
        fitted = FoldPreprocessor().fit(raw_train_frame())
        before = fitted.state_dict()
        validation = raw_train_frame().iloc[[0]].copy()
        validation.loc[:, "g3_value"] = 1e30

        fitted.transform(validation)

        self.assertEqual(before, fitted.state_dict())

    def test_all_missing_zero_variance_and_indicator_have_explicit_behavior(self):
        fitted = FoldPreprocessor().fit(raw_train_frame())

        transformed = fitted.transform(raw_train_frame())
        state = fitted.state_dict()

        self.assertEqual([0.0, 1.0], transformed["g0_binary_flag"].tolist())
        self.assertEqual([-1.0, 1.0], transformed["g3_value"].tolist())
        self.assertEqual([0.0, 0.0], transformed["g3_all_missing"].tolist())
        self.assertEqual([0.0, 0.0], transformed["g3_zero_variance"].tolist())
        self.assertIn("g3_all_missing", state["all_missing_columns"])
        self.assertIn("g3_zero_variance", state["zero_variance_columns"])
        self.assertIn("g0_binary_flag", state["indicator_columns"])

    def test_transform_is_fixed_order_finite_float32_and_appends_distance(self):
        fitted = FoldPreprocessor().fit(raw_train_frame())
        validation = raw_train_frame().iloc[[0]].copy()
        validation.loc[:, "g3_value"] = 1e300

        transformed = fitted.transform(validation)

        self.assertEqual(
            [*raw_train_frame().columns, "g6_train_distribution_rms_z"],
            transformed.columns.tolist(),
        )
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in transformed.dtypes))
        self.assertTrue(np.isfinite(transformed.to_numpy(dtype=np.float32)).all())

    def test_transform_rejects_column_order_drift(self):
        fitted = FoldPreprocessor().fit(raw_train_frame())
        reversed_columns = raw_train_frame().loc[:, list(reversed(raw_train_frame().columns))]

        with self.assertRaisesRegex(ValueError, "raw feature columns do not match fitted order"):
            fitted.transform(reversed_columns)

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            FoldPreprocessor().transform(raw_train_frame())


if __name__ == "__main__":
    unittest.main()
