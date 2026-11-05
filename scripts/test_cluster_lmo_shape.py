"""Matched LMO rank/shape controls, including actual dispatch and return."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_cluster_near_projection import ROOT, fixture, host, update
from emerging_optimizers.orthogonalized_optimizers.spectral_cone_lmo import (
    matrix_sign_cone_direction, spectral_cone_lmo_direction,
)

KINDS = ("cone_lmo_top1", "cone_lmo_mix25")


def polar(x):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


class LMOShapeTests(unittest.TestCase):
    def test_blend_cone_descent_and_matched_length(self):
        generator = torch.Generator().manual_seed(927)
        device = os.environ.get("P2GD_TEST_DEVICE", "cpu")
        for m, n in ((8, 6), (6, 8)):
            for q in (1, 2, 4):
                z = torch.randn(m, n, dtype=torch.float64, generator=generator).to(device)
                u = torch.linalg.qr(torch.randn(m, q, dtype=torch.float64, generator=generator).to(device))[0]
                v = torch.linalg.qr(torch.randn(n, q, dtype=torch.float64, generator=generator).to(device))[0]
                reference = spectral_cone_lmo_direction(z, u, v, polar)
                calls = []

                def oracle(x):
                    calls.append(x)
                    return polar(x)

                d = spectral_cone_lmo_direction(z, u, v, oracle, blend=.25, match_reference_fro=True)
                h = u.T @ d @ v
                self.assertEqual(len(calls), 1)
                self.assertLess(abs(float(torch.linalg.eigvalsh(.5 * (h + h.T))[-1])), 2e-10)
                self.assertGreater(float((z * d).sum()), 0.)
                torch.testing.assert_close(torch.linalg.vector_norm(d), torch.linalg.vector_norm(reference))
                self.assertGreater(float(torch.linalg.vector_norm(d-reference)), 1e-3)

    def test_literal_top1_lmo_trial(self):
        x, m, cache = fixture()
        z = -m / torch.linalg.vector_norm(m)
        d = matrix_sign_cone_direction(z, cache.u_basis[:, :1], cache.v_basis[:, :1], msign=host.msign)
        with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
            _, info = update(x, m, cache, "cone_lmo_top1")
        torch.testing.assert_close(svd.call_args.args[0], x + .015 * 8 * d)
        self.assertEqual(info.cluster_dimension, 1)

    def test_production_one_ns_one_small_svd_and_returned_cache(self):
        x, m, cache = fixture()
        for kind in KINDS:
            with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
                with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("svdvals")):
                    with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                        with patch.object(host, "msign", wraps=host.msign) as ns:
                            result, info = update(x, m, cache, kind, spectrum="warm_block8")
            self.assertEqual(ns.call_count, 1)
            self.assertEqual(svd.call_count, 1)
            self.assertEqual(tuple(svd.call_args.args[0].shape), (8, 8))
            self.assertTrue(info.spectral_cache_hit)
            post_s = torch.linalg.svdvals(result)
            torch.testing.assert_close(info.next_spectral_cache.s_basis, post_s, atol=2e-6, rtol=2e-6)
            self.assertEqual(info.cluster_dimension, 1 if kind.endswith("top1") else 4)

    def test_safe_unchanged_zero_and_momentum_scaling(self):
        x, m, cache = fixture()
        reference, _ = update(x, m, cache, mode="control")
        for kind in KINDS:
            safe, _ = update(x, m, cache, kind, mode="control")
            torch.testing.assert_close(safe, reference, atol=0, rtol=0)
            stationary, _ = update(x, torch.zeros_like(m), cache, kind)
            torch.testing.assert_close(stationary, x, atol=2e-6, rtol=2e-6)
            normal, _ = update(x, m, cache, kind)
            scaled, _ = update(x, m * 100., cache, kind)
            torch.testing.assert_close(normal, scaled, atol=3e-5, rtol=3e-5)

    def test_offline_direction_matches_real_trial(self):
        from audit_cluster_ritz_snapshots import direction
        x, m, cache = fixture()
        for kind in KINDS:
            d = direction(m, cache.u_basis, cache.s_basis, cache.v_basis, mode="adaptive4",
                          threshold=5e-5, cluster_threshold=.002, coefficient=8.,
                          direction_mode="momentum_fro", near_projection=kind)
            with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
                update(x, m, cache, kind)
            torch.testing.assert_close(svd.call_args.args[0], x + .015 * d)

    def test_host_accepts_variants(self):
        x, _, _ = fixture()
        for kind in KINDS:
            opt = host.SpELPGDSameProjection(
                [torch.nn.Parameter(x.clone())], lr=.015, radial_cluster_mode="adaptive4",
                cluster_direction_mode="momentum_fro", cluster_near_projection=kind,
                cluster_spectrum_mode="warm_block8", gap_threshold_rel=5e-5,
                cluster_threshold_rel=.0002, pgd_lr_scale=.5,
                pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
                post_return_spectral_cache=True, tangent_project_after_msign=True,
            )
            self.assertEqual(opt.cluster_near_projection, kind)

    def test_grid_and_spooled_wrapper(self):
        launcher = ROOT / "slurm/cluster_lmo_shape_1b_20261104.sbatch"
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "slurm_script"
            shutil.copyfile(launcher, spool)
            for file in (launcher, spool):
                for i, kind in enumerate(KINDS):
                    row = subprocess.check_output(
                        ["bash", str(file), "--describe"], text=True,
                        env={**os.environ, "PROJECT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": str(i),
                             "CLUSTER_ARM_INDEX": "0", "CLUSTER_EXPERIMENT_SEED": "999"},
                    ).strip().split("\t")
                    self.assertEqual(row, [str(i+10), "spel_pgd_dist", "adaptive4", kind,
                                           "2e-4", "0.5", "2345", "1908", "20"])

    def test_invalid_blend(self):
        x, m, cache = fixture()
        with self.assertRaises(ValueError):
            spectral_cone_lmo_direction(m, cache.u_basis, cache.v_basis, polar, blend=1.1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
