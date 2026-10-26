"""Keep the selected 3B arms paired and the K10 trial literal."""
import os
import subprocess
import unittest
from unittest.mock import patch

import torch

from test_cluster_momentum_normalization import ROOT, fixture, host
from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as cluster
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz


class Selected3BTests(unittest.TestCase):
    def test_six_arms_are_matched(self):
        launcher = ROOT / "slurm/cluster_selected_3b_20261025.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                                       env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}).strip().split("\t")
                for i in range(6)]
        configs = {(r[1], r[2], int(r[3]), float(r[4])) for r in rows}
        self.assertEqual(len(configs), 6)
        for k in (10, 20):
            self.assertIn(("spel_pgd_dist", "control", k, 1.), configs)
            self.assertIn(("spel_pgd_dist", "adaptive4", k, 8.), configs)
        self.assertEqual({r[1] for r in rows if r[2] == "none"},
                         {"muon_ball_dist", "spectral_ball_dist"})
        self.assertTrue(all(float(r[5]) == .015 and float(r[6]) == .0015 for r in rows))
        text = launcher.read_text()
        for required in ("#SBATCH --array=0-5%3", "TRAIN_ITER=5723 TRAIN_TOKENS=3000000000",
                         "LR_WARMUP_ITERS=250 LR_DECAY_ITERS=5723", "SAVE_CHECKPOINT=0",
                         "EVAL_INTERVAL=250 EVAL_ITERS=20", "SPEL_PGD_GAP_THRESHOLD_REL=5e-5",
                         "SPEL_PGD_CLUSTER_DIRECTION_MODE=momentum_fro", "CLUSTER_SNAPSHOT_MAX=56",
                         "CLUSTER_SNAPSHOT_LATE_AFTER=3000", "from_scratch_no_checkpoint_resume"):
            self.assertIn(required, text)
        self.assertEqual(5723 * 128 * 4096, 3000500224)
        self.assertLess((5723 - 1) * 128 * 4096, 3000000000)

    def test_invalid_array_index_fails(self):
        launcher = ROOT / "slurm/cluster_selected_3b_20261025.sbatch"
        for index in ("-1", "6"):
            result = subprocess.run(["bash", str(launcher), "--describe"],
                                    env={**os.environ, "SLURM_ARRAY_TASK_ID": index}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_k10_and_k20_keep_the_same_normalized_trial(self):
        x, m, cache = fixture()
        d = host.project_spectral_sphere_cone(
            -m / torch.linalg.vector_norm(m), cache.u_basis[:, :2], cache.v_basis[:, :2])
        target = "emerging_optimizers.orthogonalized_optimizers.warm_block_ritz.warm_block8_ritz"
        for k in (10, 20):
            with self.subTest(iterations=k), patch(target, wraps=warm_block8_ritz) as solve:
                with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
                    with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                        _, info = cluster.compute_cluster_radial_update(
                            x.clone(), m, 1., 10, 8, effective_lr=.015, cluster_mode="adaptive4",
                            cluster_threshold_rel=.002, gap_threshold_rel=5e-5,
                            pgd_lr_scale=8., current_spectral_cache=cache,
                            cluster_direction_mode="momentum_fro", cluster_spectrum_mode="warm_block8",
                            cluster_iterations=k,
                        )
                torch.testing.assert_close(solve.call_args.args[0], x + .015*8*d)
                self.assertEqual(solve.call_args.kwargs["iterations"], k)
                self.assertEqual(svd.call_count, 1)
                self.assertEqual(svd.call_args.args[0].shape, (8, 8))
                self.assertFalse(info.next_spectral_cache.is_exact)


if __name__ == "__main__":
    unittest.main()
