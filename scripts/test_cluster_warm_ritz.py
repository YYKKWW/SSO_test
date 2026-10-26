"""CPU/CUDA tests for the no-fallback block-8 cluster-radial path."""
import os
import sys
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial import compute_cluster_radial_update
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


def matrix(shape=(48, 32)):
    generator = torch.Generator(device=DEVICE).manual_seed(19)
    return torch.randn(*shape, generator=generator, device=DEVICE)


def step(x, m, cache=None, *, mode="adaptive4", iterations=10, lr=.015, gap=5e-5):
    work = x.clone()
    update, info = compute_cluster_radial_update(
        work, m, 1., 10, 8, effective_lr=lr, cluster_mode=mode,
        cluster_threshold_rel=max(.002, gap), gap_threshold_rel=gap,
        pgd_lr_scale=1., current_spectral_cache=cache,
        cluster_spectrum_mode="warm_block8", cluster_iterations=iterations,
    )
    return work - lr * update, info


class WarmRitzTests(unittest.TestCase):
    def test_width256_observer_and_offline_audit(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MEGATRON_PATH": str(root / "Megatron-LM"), "CLUSTER_SNAPSHOT_DIR": directory,
        }):
            spec = importlib.util.spec_from_file_location("ritz_observer", root / "scripts/pretrain_cluster_ritz_observed.py")
            observer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(observer)
            from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as cluster

            def synthetic_training(*args, **kwargs):
                x, m, cache = matrix((256, 256)), matrix((256, 256)), None
                for index, lr in ((1, 0.), (2, .015), (50, .015)):
                    d, info = cluster.compute_cluster_radial_update(
                        W=x, M=m, target_radius=1., power_iteration_steps=10, msign_steps=8,
                        effective_lr=lr, cluster_mode="adaptive4", cluster_threshold_rel=.002,
                        gap_threshold_rel=5e-5, pgd_lr_scale=1., current_spectral_cache=cache,
                        cluster_spectrum_mode="warm_block8", cluster_iterations=10,
                        optimizer_step=index, param_name="synthetic.weight", component_label="matrix",
                    )
                    x.add_(d, alpha=-lr)
                    cache = info.next_spectral_cache

            with patch.object(observer.runpy, "run_path", side_effect=synthetic_training):
                with contextlib.redirect_stdout(io.StringIO()):
                    observer.main()
            counts = json.loads((Path(directory) / "spectral_call_counts.json").read_text())
            self.assertEqual(counts["small_svd"], counts["component_updates"])
            self.assertEqual(counts["full_svd"], 0)
            self.assertEqual(counts["svdvals"], 0)
            self.assertEqual(counts["saved"], 2)
            spec = importlib.util.spec_from_file_location("ritz_auditor", root / "scripts/audit_cluster_ritz_snapshots.py")
            auditor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auditor)
            with patch.object(sys, "argv", ["audit", directory, "--device", DEVICE]):
                with contextlib.redirect_stdout(io.StringIO()):
                    auditor.main()
            result = json.loads((Path(directory) / "offline_audit.json").read_text())
            self.assertEqual(result["summary"]["samples"], 2)
            self.assertTrue(all(torch.isfinite(torch.tensor(r["direction_relative_error"])) for r in result["rows"]))

    def test_rectangular_pairing_and_scaling(self):
        for shape in ((48, 32), (32, 48)):
            x = matrix(shape)
            u, s, vh = warm_block8_ritz(x, iterations=10)
            torch.testing.assert_close(u.T @ u, torch.eye(8, device=DEVICE), atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(vh @ vh.T, torch.eye(8, device=DEVICE), atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(x @ vh.T, u * s, atol=2e-5, rtol=3e-5)
            _, s2, _ = warm_block8_ritz(16*x, iterations=10)
            torch.testing.assert_close(s2, 16*s, atol=2e-4, rtol=3e-5)

    def test_fixed_iteration_count_and_only_small_svd(self):
        x = matrix()
        for k in (10, 20):
            with patch.object(torch.linalg, "qr", wraps=torch.linalg.qr) as qr:
                with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                    warm_block8_ritz(x, iterations=k)
            self.assertEqual(qr.call_count, 2*k+2)
            self.assertEqual(svd.call_count, 1)
            self.assertEqual(svd.call_args.args[0].shape, (8, 8))

    def test_no_global_rng_consumption(self):
        before = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state().clone() if DEVICE == "cuda" else None
        warm_block8_ritz(matrix(), iterations=10)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if before_cuda is not None:
            self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))

    def test_cache_is_approximate_and_next_step_uses_it(self):
        x, m = matrix(), matrix()
        with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
            with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("full vals")):
                x, info = step(x, m, lr=0.)
                self.assertFalse(info.next_spectral_cache.is_exact)
                self.assertFalse(info.next_spectral_cache.bounds_certified)
                self.assertEqual((info.next_spectral_cache.gap_lower, info.next_spectral_cache.gap_upper), (0., 1.))
                for mode, gap, branch in (("control", 5e-5, "spel"), ("adaptive4", 1., "pgd")):
                    with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                        _, result = step(x, m, info.next_spectral_cache, mode=mode, gap=gap)
                    self.assertEqual(result.branch, branch)
                    self.assertEqual(svd.call_count, 1)
                    self.assertEqual(svd.call_args.args[0].shape, (8, 8))
                    self.assertTrue(result.spectral_cache_hit)
                    self.assertFalse(result.approximate_exact_fallback)

    def test_returned_cache_matches_scaled_ritz_values_not_exact_claim(self):
        x, info = step(matrix(), matrix(), lr=0.)
        original = warm_block8_ritz
        captured = []
        def record(y, **kwargs):
            result = original(y, **kwargs)
            captured.append((y.clone(), result))
            return result
        target = "emerging_optimizers.orthogonalized_optimizers.warm_block_ritz.warm_block8_ritz"
        with patch(target, side_effect=record):
            returned, result = step(x, matrix(), info.next_spectral_cache)
        y, (u, s, vh) = captured[0]
        torch.testing.assert_close(returned, y/s[0], atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(result.next_spectral_cache.s_basis, s/s[0])
        torch.testing.assert_close(result.next_spectral_cache.v_basis, vh.T)
        self.assertAlmostEqual(result.next_spectral_cache.rel_gap, float(1-s[1]/s[0]), places=6)
        self.assertTrue(torch.isnan(torch.tensor(result.post_retract_bias)))

    def test_missed_subspace_is_not_silently_certified_or_repaired(self):
        x = torch.diag(torch.tensor([1., .9]+[.5]*14, device=DEVICE))
        q = torch.eye(16, device=DEVICE)[:, 2:10]
        _, s, _ = warm_block8_ritz(x, iterations=20, initial_v=q)
        self.assertAlmostEqual(float(s[0]), .5, places=5)
        # This intentional counterexample documents why a fixed warm subspace
        # is an approximation, even with many iterations and perfect arithmetic.

    def test_optimizer_integration(self):
        p = torch.nn.Parameter(matrix())
        opt = host.SpELPGDSameProjection(
            [p], lr=.015, radial_cluster_mode="adaptive4",
            cluster_spectrum_mode="warm_block8", cluster_iterations=20,
            gap_threshold_rel=5e-5, cluster_threshold_rel=.002, pgd_lr_scale=1.,
            pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
            post_return_spectral_cache=True, tangent_project_after_msign=True,
        )
        with torch.no_grad(), patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
            opt._compute_component_update(p, matrix(), None, None, current_lr=0., param_name="test")
            d = opt._compute_component_update(p, matrix(), None, None, current_lr=.015, param_name="test")
            p.add_(d, alpha=-.015)
        stats = opt.get_branch_stats()
        self.assertEqual(stats["cumulative_spectral_cache_bootstrap_count"], 1)
        self.assertEqual(stats["cumulative_spectral_cache_hit_count"], 1)
        self.assertTrue(torch.isfinite(p).all())


if __name__ == "__main__":
    unittest.main()
