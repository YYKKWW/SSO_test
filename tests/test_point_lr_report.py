"""Validate the released November 4 snapshot without Torch or an H20."""

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report", ROOT / "scripts/manifold/report_point_lr_batch.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class PointLRReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "docs/experiments/frobenius_point_lr_20261104/results.json").read_text())

    def test_scope_and_health(self):
        self.assertEqual(len(self.data["runs"]), 18)
        batch = [r for r in self.data["runs"] if r["batch"] == "point_lr_batch_20261102"]
        self.assertEqual(len(batch), 8)
        self.assertEqual(sum(r["state"] == "COMPLETED" for r in batch), 6)
        self.assertEqual(sum(r["final_3b_val"] is not None for r in batch), 4)
        for run in batch:
            self.assertEqual(run["counters"], {"nan": 0, "skipped": 0})
            self.assertEqual(run["config"]["train_iters"], 11445)
            if run["state"] == "RUNNING":
                self.assertIsNone(run["final_3b_val"])
                self.assertIsNone(run["endpoint_val"])

    def test_stitched_points_and_control(self):
        series = REPORT.point_series(self.data)
        self.assertEqual(len(series), 5)
        self.assertEqual(series[-1][1][3000], 3.788594)
        self.assertEqual(series[0][1][3000], 3.778030)
        reference = next(r for r in self.data["references_lr001"] if r["method"] == "manifold_mcsd")
        old = {p["step"]: p["loss"] for p in reference["val"] if p["step"] <= 3000}
        self.assertEqual(series[0][1], old)

    def test_no_duplicate_resume_points(self):
        corrupted = json.loads(json.dumps(self.data))
        run = next(r for r in corrupted["runs"] if r["job_id"] == "4375560")
        run["val"].insert(0, dict(step=1000, loss=4., final=False))
        with self.assertRaises(AssertionError):
            REPORT.point_series(corrupted)


if __name__ == "__main__":
    unittest.main()
