"""Validate the four authorized follow-ups without submitting GPU jobs."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANS = [
    ROOT / "configs/manifold/frobenius_nesterov095_baselines_20261028.json",
    ROOT / "configs/manifold/frobenius_nesterov095_seed2026_20261028.json",
]


class NesterovPlanTests(unittest.TestCase):
    def setUp(self):
        self.plans = [json.loads(path.read_text()) for path in PLANS]
        self.rows = [row for plan in self.plans for row in plan]

    def test_exact_scope(self):
        self.assertEqual(len(self.rows), 4)
        self.assertEqual(len({row["key"] for row in self.rows}), 4)
        self.assertEqual({(row["method"], row["seed"]) for row in self.rows}, {
            ("manifold_muonh", 1234), ("manifold_normuon_sphere", 1234),
            ("manifold_mcsd", 2026), ("manifold_mcsd_tp", 2026),
        })
        for row in self.rows:
            self.assertEqual((row["lr"], row["beta"], row["tokens"]),
                             (.01, .95, 3000000000))
            self.assertIs(row["nesterov"], True)
        # Each plan is one independent pair; together they use at most four GPUs.
        for plan in self.plans:
            self.assertEqual([row["lane"] for row in plan], [0, 1])

    def test_no_completed_duplicates(self):
        ledger = json.loads((ROOT / "docs/experiments/frobenius_3b_6b/results.json").read_text())
        keys = ("method", "lr", "beta", "nesterov", "seed", "tokens")
        completed = {tuple(row[key] for key in keys) for row in ledger["runs"]
                     if row["direction"] == "none"}
        planned = {tuple(row[key] for key in keys) for row in self.rows}
        self.assertFalse(planned & completed)

    def test_submitted_snapshot_matches_plan(self):
        path = ROOT / "docs/experiments/frobenius_3b_6b/nesterov095_submission_20261028.json"
        snapshot = json.loads(path.read_text())
        self.assertEqual(snapshot["max_concurrency"], 4)
        runs = {row["key"]: row for row in snapshot["runs"]}
        self.assertEqual(set(runs), {row["key"] for row in self.rows})
        self.assertEqual(len({row["job_id"] for row in runs.values()}), 4)
        self.assertEqual(snapshot["git_status"], "")
        for spec in self.rows:
            run = runs[spec["key"]]
            for key, value in spec.items():
                self.assertEqual(run[key], value)
            self.assertTrue(run["source_hashes_match"])
            cfg = run["config"]
            self.assertEqual(cfg["source_commit"], "fc15600a2898d1748c8b34fe8d3945b1036f03ab")
            self.assertEqual((cfg["gpus"], cfg["train_iters"], cfg["warmup_iters"]), (1, 11445, 229))
            self.assertEqual(cfg["actual_tokens"], 3000238080)
            self.assertIsNone(cfg["resume"])
            self.assertEqual(cfg["initialization"], {"radius_policy": "shape", "radius_factor": 1.0})
            self.assertEqual(cfg["training"]["save_interval"], 0)
            args = run["runtime_args"]
            self.assertEqual(args["optimizer"], spec["method"])
            self.assertEqual(args["seed"], str(spec["seed"]))
            self.assertEqual(args["manifold_momentum_beta"], "0.95")
            is_mcsd = spec["method"] in ("manifold_mcsd", "manifold_mcsd_tp")
            flag = "manifold_mcsd_nesterov" if is_mcsd else "manifold_baseline_nesterov"
            self.assertEqual(args[flag], "True")


if __name__ == "__main__":
    unittest.main()
