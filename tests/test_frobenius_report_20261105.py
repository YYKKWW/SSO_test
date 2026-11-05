"""Dated published results and pending-job semantics; CPU and standard library only."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_nov5", ROOT / "scripts/manifold/report_point_lr_batch.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class November5ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "docs/experiments/frobenius_results_20261105/results.json").read_text())
        cls.jobs = {run["job_id"]: run for run in cls.data["runs"]}

    def test_completed_grid_and_unitf(self):
        self.assertEqual(len(self.data["runs"]), 22)
        for job, loss in {"4375562": 3.463020, "4375563": 3.458744,
                          "4375564": 3.458990, "4375565": 3.474075,
                          "4375566": 3.466872, "4375567": 3.465047,
                          "4390468": 3.460665}.items():
            run = self.jobs[job]
            self.assertEqual(run["state"], "COMPLETED")
            self.assertEqual(run["exit_code"], "0:0")
            self.assertEqual(run["completion"]["exit_code"], 0)
            self.assertEqual(run["last_val"]["step"], 11445)
            self.assertEqual(run["final_3b_val"], loss)
            self.assertEqual(run["counters"], {"nan": 0, "skipped": 0})

    def test_running_and_pending_are_not_final_results(self):
        for job in ("4390469", "4390470", "4390471"):
            self.assertIsNone(self.jobs[job]["final_3b_val"])
            self.assertIsNone(self.jobs[job]["endpoint_val"])
        for job in ("4390470", "4390471"):
            self.assertEqual(self.jobs[job]["state"], "PENDING")
            self.assertEqual(self.jobs[job]["val"], [])
            self.assertIsNone(self.jobs[job]["last_train"])
            self.assertIsNone(self.jobs[job]["counters"]["nan"])
        self.assertEqual(self.jobs["4390469"]["last_val"]["step"], 10600)

    def test_comparison_uses_same_step(self):
        differences = REPORT.same_step_differences(
            self.jobs["4390469"]["val"], self.jobs["4390468"]["val"])
        self.assertEqual(differences[-1]["step"], 10600)
        self.assertAlmostEqual(differences[-1]["loss"], .000169, places=6)
        with self.assertRaises(AssertionError):
            REPORT.same_step_differences([dict(step=1, loss=2.)] * 2, [])

    def test_collector_accepts_pending_without_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results/manifold"
            run = base / "pending"
            run.mkdir(parents=True)
            (root / "logs").mkdir()
            for name in ("point_pilot_20261031", "point_followup_20261101", "point_lr_batch_20261102"):
                (base / (name + "_register.json")).write_text('{"runs": []}')
            cfg = dict(self.jobs["4390470"]["config"], train_prefix="/data/train",
                       valid_prefix="/data/valid", source_hashes={})
            (run / "resolved_config.json").write_text(json.dumps(cfg))
            (base / "extra_register.json").write_text(json.dumps({"runs": [
                dict(job_id="9001", run_dir=str(run), method="manifold_muonh")
            ]}))
            with patch.object(REPORT.subprocess, "check_output",
                              return_value="9001|PENDING|0:0|00:00:00|Unknown|Unknown|\n"):
                result = REPORT.collect(root, ["extra"])["runs"][0]
            self.assertIsNone(result["stdout_sha256"])
            self.assertIsNone(result["final_3b_val"])
            self.assertEqual(result["train"], [])
            self.assertEqual(result["config"]["training"]["seed"], 2026)


if __name__ == "__main__":
    unittest.main()
