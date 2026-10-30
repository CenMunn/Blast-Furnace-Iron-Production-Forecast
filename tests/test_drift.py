from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.drift import compute_drift_report, run_adversarial_validation
from src.teacher_script_audit import audit_teacher_script


class DriftTests(unittest.TestCase):
    def test_shifted_feature_is_detected_without_mutating_inputs(self):
        rng = np.random.default_rng(20260801)
        train = pd.DataFrame(
            {
                "g3_shifted": rng.normal(0.0, 0.15, 150),
                "g4_noise": rng.normal(0.0, 1.0, 150),
            }
        ).astype(np.float32)
        later = pd.DataFrame(
            {
                "g3_shifted": rng.normal(3.0, 0.15, 150),
                "g4_noise": rng.normal(0.0, 1.0, 150),
            }
        ).astype(np.float32)
        train_bytes = train.to_numpy().tobytes()
        later_bytes = later.to_numpy().tobytes()

        result = compute_drift_report(train, later, "train_vs_test")
        adversarial = run_adversarial_validation(
            train, later, train.columns.tolist()
        )

        self.assertGreater(result.adversarial_auc, 0.90)
        self.assertGreater(adversarial["oof_roc_auc"], 0.90)
        self.assertEqual("g3_shifted", adversarial["feature_importance"][0]["feature"])
        self.assertEqual(train_bytes, train.to_numpy().tobytes())
        self.assertEqual(later_bytes, later.to_numpy().tobytes())

    def test_identical_distributions_have_near_chance_auc(self):
        rng = np.random.default_rng(88)
        shared = pd.DataFrame(
            rng.normal(size=(400, 3)), columns=["g3_a", "g3_b", "g4_c"]
        ).astype(np.float32)
        train = shared.iloc[:200].reset_index(drop=True)
        later = shared.iloc[200:].reset_index(drop=True)
        result = run_adversarial_validation(
            train, later, shared.columns.tolist()
        )
        self.assertGreaterEqual(result["oof_roc_auc"], 0.35)
        self.assertLessEqual(result["oof_roc_auc"], 0.65)


class TeacherScriptAuditTests(unittest.TestCase):
    def test_audit_names_four_boundary_and_preprocessing_findings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "teacher_transformer.py"
            script.write_text(
                "from sklearn.preprocessing import StandardScaler\n"
                "scaler = StandardScaler().fit(all_data)\n"
                "x = frame.fillna(0)\n"
                "pooled = hidden.mean(dim=1)\n"
                "visible = events[events.event_time <= target_time + horizon]\n",
                encoding="utf-8",
            )
            result = audit_teacher_script(root, script)
            names = {finding["name"] for finding in result["findings"]}
            self.assertEqual("audited", result["status"])
            self.assertTrue(
                {
                    "global_scaler_fit",
                    "global_zero_fill",
                    "unmasked_sequence_pooling",
                    "future_inclusive_event_boundary",
                }.issubset(names)
            )

    def test_missing_teacher_script_is_reported_honestly(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_teacher_script(Path(temporary), None)
            self.assertEqual("not_provided", result["status"])
            self.assertTrue(result["requested_file_names"])


if __name__ == "__main__":
    unittest.main()
