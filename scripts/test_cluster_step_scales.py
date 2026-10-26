"""Tests for the branch coefficient and the matched global-LR experiment grid."""
import os
import subprocess
import unittest
from unittest.mock import patch

import torch

from calibrate_cluster_step_scales import infer_effective_lr
from test_cluster_momentum_normalization import ROOT, fixture, host
from emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial import compute_cluster_radial_update


class StepScaleTests(unittest.TestCase):
    def test_coefficient_changes_trial_not_direction_normalization(self):
        x, m, cache = fixture()
        d = host.project_spectral_sphere_cone(
            -m / torch.linalg.vector_norm(m), cache.u_basis[:, :2], cache.v_basis[:, :2])
        saved = m.clone()
        for coefficient in (.5, 1., 2., 4., 8.):
            with self.subTest(coefficient=coefficient):
                with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
                    _, info = compute_cluster_radial_update(
                        x.clone(), m, 1., 10, 8, effective_lr=.015, cluster_mode="adaptive4",
                        cluster_threshold_rel=.002, gap_threshold_rel=5e-5,
                        pgd_lr_scale=coefficient, current_spectral_cache=cache,
                        cluster_direction_mode="momentum_fro",
                    )
                torch.testing.assert_close(svd.call_args.args[0], x + .015*coefficient*d)
                self.assertEqual(info.cluster_step_multiplier, coefficient)
                self.assertEqual(svd.call_count, 1)
        torch.testing.assert_close(m, saved, atol=0., rtol=0.)

    def test_reconstruct_old_snapshot_effective_lr(self):
        x, m, _ = fixture()
        d = -m / torch.linalg.vector_norm(m)
        expected = .0037
        returned = (x + expected*d)*.997
        alpha, fit_error = infer_effective_lr(x, returned, d)
        self.assertAlmostEqual(alpha, expected, places=6)
        self.assertLess(fit_error, 2e-6)

    def test_tuning_grid_and_matched_controls(self):
        launcher = ROOT / "slurm/cluster_momentum_lr_tune_100m.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                                       env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}).strip().split("\t")
                for i in range(8)]
        configs = {(float(r[1]), float(r[3]), r[4]) for r in rows}
        self.assertEqual(len(configs), 8)
        for coefficient in (.5, 2., 4., 8.):
            self.assertIn((.015, coefficient, "adaptive4"), configs)
        for lr in (.01, .02):
            self.assertIn((lr, 1., "control"), configs)
            self.assertIn((lr, 4., "adaptive4"), configs)
        for r in rows:
            self.assertAlmostEqual(float(r[2]), .1*float(r[1]))
        text = launcher.read_text()
        self.assertIn("#SBATCH --array=0-7%1", text)
        self.assertIn("SPEL_PGD_GAP_THRESHOLD_REL=5e-5", text)
        self.assertIn("SPEL_PGD_CLUSTER_ITERATIONS=20", text)
        self.assertIn("SAVE_CHECKPOINT=0", text)


if __name__ == "__main__":
    unittest.main()
