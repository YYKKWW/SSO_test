"""Regression tests for normalize-momentum -> cone -> radial updates."""

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial import compute_cluster_radial_update

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


def fixture():
    s = torch.tensor([1., 1., .9, .8, .7, .6, .5, .4], device=DEVICE)
    u = torch.eye(8, device=DEVICE)
    x = torch.diag(s)
    m = torch.diag(torch.tensor([-3., -1., 2., 1., 1., 1., 1., 1.], device=DEVICE))
    return x, m, host._spectral_point_cache_from_decomposition(u, s, u)


def update(x, m, cache, *, mode="momentum_fro", spectrum="exact", cluster="adaptive4"):
    work = x.clone()
    d, info = compute_cluster_radial_update(
        work, m, 1., 10, 8, effective_lr=.015, cluster_mode=cluster,
        cluster_threshold_rel=.002, gap_threshold_rel=5e-5, pgd_lr_scale=1.,
        current_spectral_cache=cache, cluster_direction_mode=mode,
        cluster_spectrum_mode=spectrum, cluster_iterations=10,
    )
    return work - .015*d, info


class MomentumNormalizationTests(unittest.TestCase):
    def test_literal_trial_no_length_matching_or_msign(self):
        x, m, cache = fixture()
        saved = m.clone()
        z = -m / torch.linalg.vector_norm(m)
        d = host.project_spectral_sphere_cone(z, cache.u_basis[:, :2], cache.v_basis[:, :2])
        self.assertLess(float(torch.linalg.vector_norm(d)), .9)
        y = x + .015*d
        expected = y / host.accurate_svdvals_fp32(y)[0]
        with patch.object(host, "msign", side_effect=AssertionError("unneeded MCSD reference")):
            with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as solve:
                returned, info = update(x, m, cache)
        torch.testing.assert_close(solve.call_args.args[0], y)
        torch.testing.assert_close(returned, expected, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(m, saved, atol=0., rtol=0.)
        self.assertEqual(info.cluster_step_multiplier, 1.)
        self.assertEqual(info.cluster_dimension, 2)
        self.assertEqual(solve.call_count, 1)
        self.assertAlmostEqual(float((m*d).sum()),
                               -float(torch.linalg.vector_norm(m)*torch.linalg.vector_norm(d)**2), places=5)

    def test_positive_rescaling_of_momentum_preserves_update(self):
        x, m, cache = fixture()
        expected, _ = update(x, m, cache)
        for scale in (.01, 100.):
            actual, _ = update(x, scale*m, cache)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_zero_momentum_stays_finite_and_stationary(self):
        x, m, cache = fixture()
        actual, info = update(x, torch.zeros_like(m), cache)
        torch.testing.assert_close(actual, x, atol=3e-6, rtol=3e-6)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(info.radial_change_ratio, 0.)

    def test_control_is_bitwise_unchanged(self):
        x, m, cache = fixture()
        previous, _ = update(x, m, cache, mode="matched_fro", cluster="control")
        actual, _ = update(x, m, cache, cluster="control")
        torch.testing.assert_close(actual, previous, atol=0., rtol=0.)

    def test_warm_backend_has_only_one_small_svd(self):
        x, m, cache = fixture()
        with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
            with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("svdvals")):
                with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                    actual, info = update(x, m, cache, spectrum="warm_block8")
        self.assertEqual(svd.call_count, 1)
        self.assertEqual(svd.call_args.args[0].shape, (8, 8))
        self.assertTrue(info.spectral_cache_hit)
        self.assertFalse(info.next_spectral_cache.is_exact)
        self.assertTrue(torch.isfinite(actual).all())

    def test_offline_audit_uses_the_new_direction(self):
        spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_cluster_ritz_snapshots.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        _, m, cache = fixture()
        actual = audit.direction(m, cache.u_basis, cache.s_basis, cache.v_basis,
                                 mode="adaptive4", threshold=5e-5, cluster_threshold=.002,
                                 coefficient=1., direction_mode="momentum_fro")
        expected = host.project_spectral_sphere_cone(
            -m/torch.linalg.vector_norm(m), cache.u_basis[:, :2], cache.v_basis[:, :2])
        torch.testing.assert_close(actual, expected)

    def test_optimizer_dispatch_passes_direction_mode(self):
        x, m, _ = fixture()
        p = torch.nn.Parameter(x.clone())
        opt = host.SpELPGDSameProjection(
            [p], lr=.015, radial_cluster_mode="adaptive4", cluster_direction_mode="momentum_fro",
            cluster_spectrum_mode="warm_block8", cluster_iterations=10,
            gap_threshold_rel=5e-5, cluster_threshold_rel=.002, pgd_lr_scale=1.,
            pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
            post_return_spectral_cache=True, tangent_project_after_msign=True,
        )
        with torch.no_grad():
            opt._compute_component_update(p, m, None, None, current_lr=0., param_name="test")
            target = "emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial.compute_cluster_radial_update"
            with patch(target, wraps=compute_cluster_radial_update) as dispatch:
                opt._compute_component_update(p, m, None, None, current_lr=.015, param_name="test")
        self.assertEqual(dispatch.call_args.kwargs["cluster_direction_mode"], "momentum_fro")

    def test_grid(self):
        launcher = ROOT / "slurm/cluster_momentum_fro_100m.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                                       env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}).strip().split("\t")
                for i in range(4)]
        self.assertEqual({(int(r[1]), float(r[2])) for r in rows},
                         {(10, 5e-5), (10, .002), (20, 5e-5), (20, .002)})
        self.assertTrue(all(r[3] == "momentum_fro" for r in rows))


if __name__ == "__main__":
    unittest.main()
