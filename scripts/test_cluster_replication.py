"""Third seed changes no algorithm setting; offline replay handles overflow."""

import os
import subprocess
import unittest
from pathlib import Path

import torch
from audit_cluster_startup_capacity import cone_direction, replay, summarize

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


class ReplicationTests(unittest.TestCase):
    def test_third_seed_grid_matches_second_seed(self):
        old = ROOT / "slurm/cluster_seed_mechanism_3b_20261028.sbatch"
        new = ROOT / "slurm/cluster_seed3_3b_20261029.sbatch"
        for i in range(4):
            env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}
            env.pop("CLUSTER_EXPERIMENT_SEED", None)
            a = subprocess.check_output(["bash", str(old), "--describe"], env=env, text=True).strip().split("\t")
            b = subprocess.check_output(["bash", str(new), "--describe"], env=env, text=True).strip().split("\t")
            self.assertEqual(a[:-1], b[:-1])
            self.assertEqual((a[-1], b[-1]), ("2345", "3456"))
        result = subprocess.run(["bash", str(new), "--describe"],
                                env={**os.environ, "SLURM_ARRAY_TASK_ID": "4"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_capacity_is_not_silently_clipped(self):
        s = torch.tensor([1., 1., 1., 1., 1., 1., .5, .4], device=DEVICE)
        eye = torch.eye(8, device=DEVICE)
        m = -eye
        q4, rank4, active4 = cone_direction(m, eye, s, eye, 4, .002)
        q6, rank6, active6 = cone_direction(m, eye, s, eye, 6, .002)
        self.assertEqual((rank4, rank6, active4, active6), (4, 6, 6, 6))
        self.assertGreater(float(torch.linalg.vector_norm(q4-q6)), .1)

    def test_paired_replay_and_summary(self):
        s = torch.tensor([1., 1., 1., 1., 1., 1., .5, .4], device=DEVICE)
        eye = torch.eye(8, device=DEVICE)
        sample = dict(current=torch.diag(s), momentum=-eye, current_u=eye, current_s=s,
                      current_v=eye, radius=1., gap_threshold=5e-5, cluster_threshold=.002,
                      effective_lr=.015, cluster_mode="adaptive4", coefficient=8.,
                      direction_mode="momentum_fro", near_projection="cone", current_gap=0.,
                      step=2, selection="test_only")
        before = sample["current"].clone()
        row = replay(sample)
        torch.testing.assert_close(sample["current"], before, rtol=0, atol=0)
        self.assertEqual(len(row["startup"]), 2)
        self.assertEqual(len(row["capacity"]), 5)
        report = summarize([row])
        self.assertEqual(report["true_capacity_over4"], 1)
        self.assertEqual(report["true_capacity_over6"], 0)
        for entry in row["capacity"]:
            self.assertLess(abs(entry["radius_error"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
