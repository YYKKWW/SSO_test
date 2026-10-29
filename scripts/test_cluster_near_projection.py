"""Matched near-branch controls: normalize once, change only the correction."""

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
from emerging_optimizers.orthogonalized_optimizers import \
    spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial import \
    compute_cluster_radial_update

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


def fixture():
    s = torch.tensor([1., 1., .999, .9985, .8, .7, .6, .5], device=DEVICE)
    eye = torch.eye(8, device=DEVICE)
    cache = host._spectral_point_cache_from_decomposition(eye, s, eye)
    m = torch.diag(torch.tensor([-3., -1., 2., -2., 1., 1., 1., 1.], device=DEVICE))
    m[0, 2] = .5
    return torch.diag(s), m, cache


def update(x, m, cache, near="cone", mode="adaptive4", spectrum="exact"):
    work = x.clone()
    d, info = compute_cluster_radial_update(
        work, m, 1., 10, 8, effective_lr=.015, cluster_mode=mode,
        cluster_threshold_rel=.002, gap_threshold_rel=5e-5, pgd_lr_scale=8.,
        current_spectral_cache=cache, cluster_direction_mode="momentum_fro",
        cluster_near_projection=near, cluster_spectrum_mode=spectrum,
        cluster_iterations=10,
    )
    return work - .015 * d, info


class NearProjectionTests(unittest.TestCase):
    def test_literal_trials_and_no_second_normalization(self):
        x, m, cache = fixture()
        z = -m / torch.linalg.vector_norm(m)
        expected = {
            "ambient": z,
            "tangent1": host.project_to_tangent_plane(z, cache.u1, cache.v1),
            "cone": host.project_spectral_sphere_cone(z, cache.u_basis[:, :4], cache.v_basis[:, :4]),
        }
        for near, direction in expected.items():
            with self.subTest(near=near), patch.object(host, "msign", side_effect=AssertionError("extra msign")):
                with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
                    returned, info = update(x, m, cache, near)
                trial = x + .015 * 8 * direction
                torch.testing.assert_close(svd.call_args.args[0], trial)
                torch.testing.assert_close(returned, trial / host.accurate_svdvals_fp32(trial)[0])
                self.assertEqual(svd.call_count, 1)
                self.assertEqual(info.cluster_step_multiplier, 8.)
                self.assertEqual(info.cluster_dimension, {"ambient": 0, "tangent1": 1, "cone": 4}[near])

    def test_safe_branch_is_bitwise_unchanged(self):
        x, m, cache = fixture()
        reference, _ = update(x, m, cache, mode="control")
        for near in ("ambient", "tangent1"):
            actual, _ = update(x, m, cache, near, mode="control")
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_momentum_scale_invariance_and_zero(self):
        x, m, cache = fixture()
        for near in ("ambient", "tangent1", "cone"):
            ref, _ = update(x, m, cache, near)
            scaled, _ = update(x, 100 * m, cache, near)
            stationary, _ = update(x, torch.zeros_like(m), cache, near)
            torch.testing.assert_close(scaled, ref, atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(stationary, x, atol=3e-6, rtol=3e-6)

    def test_cached_warm_update_uses_only_one_small_svd(self):
        x, m, cache = fixture()
        for near in ("ambient", "tangent1", "cone"):
            with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
                with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("svdvals")):
                    with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                        returned, info = update(x, m, cache, near, spectrum="warm_block8")
            self.assertEqual(svd.call_count, 1)
            self.assertEqual(tuple(svd.call_args.args[0].shape), (8, 8))
            self.assertTrue(torch.isfinite(returned).all())
            self.assertTrue(info.spectral_cache_hit)
            self.assertFalse(info.next_spectral_cache.is_exact)

    def test_audit_reconstructs_mechanism(self):
        spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_cluster_ritz_snapshots.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        x, m, cache = fixture()
        for near in ("ambient", "tangent1", "cone"):
            direction = audit.direction(m, cache.u_basis, cache.s_basis, cache.v_basis,
                                        mode="adaptive4", threshold=5e-5, cluster_threshold=.002,
                                        coefficient=8., direction_mode="momentum_fro", near_projection=near)
            with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
                update(x, m, cache, near)
            torch.testing.assert_close(svd.call_args.args[0], x + .015 * direction)

    def test_host_dispatch_and_invalid_mode(self):
        x, m, _ = fixture()
        p = torch.nn.Parameter(x.clone())
        options = dict(lr=.015, radial_cluster_mode="adaptive4", cluster_direction_mode="momentum_fro",
                       cluster_spectrum_mode="warm_block8", cluster_iterations=10,
                       gap_threshold_rel=5e-5, cluster_threshold_rel=.002, pgd_lr_scale=8.,
                       pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
                       post_return_spectral_cache=True, tangent_project_after_msign=True)
        for near in ("ambient", "tangent1"):
            opt = host.SpELPGDSameProjection([p], cluster_near_projection=near, **options)
            target = "emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial.compute_cluster_radial_update"
            with torch.no_grad(), patch(target, wraps=compute_cluster_radial_update) as dispatch:
                opt._compute_component_update(p, m, None, None, current_lr=0., param_name="test")
            self.assertEqual(dispatch.call_args.kwargs["cluster_near_projection"], near)
        with self.assertRaises(ValueError):
            host.SpELPGDSameProjection([p], cluster_near_projection="typo", **options)

    def test_six_arm_grid(self):
        launcher = ROOT / "slurm/cluster_seed_mechanism_3b_20261028.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                                       env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i),
                                            "CLUSTER_EXPERIMENT_SEED": "2345"}).strip().split("\t")
                for i in range(6)]
        self.assertEqual([r[1] for r in rows[:2]], ["muon_ball_dist", "spectral_ball_dist"])
        self.assertEqual(rows[2][2], "control")
        self.assertEqual([r[7] for r in rows[3:]], ["cone", "ambient", "tangent1"])
        self.assertTrue(all(r[3] == "10" and r[8] == "2345" and float(r[5]) == .015 for r in rows))
        self.assertTrue(all(float(r[4]) == 8 for r in rows[3:]))


if __name__ == "__main__":
    unittest.main()
