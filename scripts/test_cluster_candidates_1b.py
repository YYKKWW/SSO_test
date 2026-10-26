"""Matched 1B configurations and bounded late-training observations."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_cluster_momentum_normalization import ROOT, fixture, host
from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as cluster
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz


def load_observer():
    spec = importlib.util.spec_from_file_location("observer_1b", ROOT / "scripts/pretrain_cluster_ritz_observed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateTests(unittest.TestCase):
    def test_grid_has_matched_controls_and_baselines(self):
        launcher = ROOT / "slurm/cluster_candidates_1b_20261024.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                                       env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}).strip().split("\t")
                for i in range(12)]
        configs = {(r[1], r[2], int(r[3]), float(r[4]), float(r[5])) for r in rows}
        self.assertEqual(len(configs), 12)
        for r in rows:
            self.assertAlmostEqual(float(r[6]), .1 * float(r[5]))
            if r[2] == "adaptive4":
                self.assertIn(("spel_pgd_dist", "control", int(r[3]), 1., float(r[5])), configs)
        for lr in (.015, .02):
            for optimizer in ("muon_ball_dist", "spectral_ball_dist"):
                self.assertTrue(any(r[1] == optimizer and float(r[5]) == lr for r in rows))
        self.assertEqual(sum(r[2] == "adaptive4" for r in rows), 5)
        self.assertTrue(any(r[2] == "adaptive4" and int(r[3]) == 40 for r in rows))
        text = launcher.read_text()
        for required in ("#SBATCH --array=0-11%3", "TRAIN_ITER=1908 TRAIN_TOKENS=1000000000",
                         "LR_WARMUP_ITERS=250 LR_DECAY_ITERS=1908", "SAVE_CHECKPOINT=0",
                         "SPEL_PGD_GAP_THRESHOLD_REL=5e-5", "SPEL_PGD_CLUSTER_DIRECTION_MODE=momentum_fro",
                         "CLUSTER_SNAPSHOT_LATE=8", "EVAL_INTERVAL=250"):
            self.assertIn(required, text)
        self.assertEqual(1908 * 128 * 4096, 1000341504)

    def test_invalid_array_index_fails(self):
        result = subprocess.run(["bash", str(ROOT / "slurm/cluster_candidates_1b_20261024.sbatch"), "--describe"],
                                env={**os.environ, "SLURM_ARRAY_TASK_ID": "12"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_c12_k40_keeps_literal_trial_and_small_svd(self):
        x, m, cache = fixture()
        d = host.project_spectral_sphere_cone(
            -m / torch.linalg.vector_norm(m), cache.u_basis[:, :2], cache.v_basis[:, :2])
        target = "emerging_optimizers.orthogonalized_optimizers.warm_block_ritz.warm_block8_ritz"
        with patch(target, wraps=warm_block8_ritz) as solve:
            with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
                with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                    _, info = cluster.compute_cluster_radial_update(
                        x.clone(), m, 1., 10, 8, effective_lr=.02, cluster_mode="adaptive4",
                        cluster_threshold_rel=.002, gap_threshold_rel=5e-5,
                        pgd_lr_scale=12., current_spectral_cache=cache,
                        cluster_direction_mode="momentum_fro", cluster_spectrum_mode="warm_block8",
                        cluster_iterations=40,
                    )
        torch.testing.assert_close(solve.call_args.args[0], x + .02*12*d)
        self.assertEqual(solve.call_args.kwargs["iterations"], 40)
        self.assertEqual(svd.call_count, 1)
        self.assertEqual(svd.call_args.args[0].shape, (8, 8))
        self.assertFalse(info.next_spectral_cache.is_exact)

    def test_default_observer_schedule_is_unchanged(self):
        with patch.dict(os.environ, {"MEGATRON_PATH": str(ROOT / "Megatron-LM")}):
            observer = load_observer()
        self.assertEqual(observer.snapshot_options({}), ({2, 50, 150, 190}, 24, 8, 0, 1000))
        with self.assertRaises(ValueError):
            observer.snapshot_options({"CLUSTER_SNAPSHOT_MAX": "0"})

    def test_late_samples_are_bounded_and_store_actual_lr(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MEGATRON_PATH": str(ROOT / "Megatron-LM"), "CLUSTER_SNAPSHOT_DIR": directory,
            "CLUSTER_SNAPSHOT_STEPS": "2,1900", "CLUSTER_SNAPSHOT_MAX": "4",
            "CLUSTER_SNAPSHOT_EARLY": "1", "CLUSTER_SNAPSHOT_LATE": "1",
            "CLUSTER_SNAPSHOT_LATE_AFTER": "1000",
        }):
            observer = load_observer()

            def synthetic_training(*args, **kwargs):
                x, m, _ = fixture()
                cache = None
                for step in (1, 2, 3, 1001, 1900, 1901):
                    lr = 0. if step == 1 else .02
                    d, info = cluster.compute_cluster_radial_update(
                        W=x, M=m, target_radius=1., power_iteration_steps=10, msign_steps=8,
                        effective_lr=lr, cluster_mode="adaptive4", cluster_threshold_rel=1.,
                        gap_threshold_rel=1., pgd_lr_scale=8., current_spectral_cache=cache,
                        cluster_spectrum_mode="warm_block8", cluster_iterations=40,
                        cluster_direction_mode="momentum_fro", optimizer_step=step,
                    )
                    x.add_(d, alpha=-lr)
                    cache = info.next_spectral_cache

            with patch.object(observer.runpy, "run_path", side_effect=synthetic_training):
                with contextlib.redirect_stdout(io.StringIO()):
                    observer.main()
            counts = json.loads((Path(directory) / "spectral_call_counts.json").read_text())
            self.assertEqual(counts["saved"], 3)
            self.assertEqual(counts["component_updates"], counts["small_svd"])
            self.assertEqual(counts["full_svd"], 0)
            samples = [torch.load(p, map_location="cpu", weights_only=True)
                       for p in sorted(Path(directory).glob("sample_*.pt"))]
            self.assertEqual([s["step"] for s in samples], [2, 1001, 1900])
            self.assertEqual(samples[1]["selection"], "late_cluster_calls")
            self.assertTrue(all(s["effective_lr"] == .02 for s in samples))


if __name__ == "__main__":
    unittest.main()
