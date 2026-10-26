"""CPU/CUDA checks for the opt-in LLM near-cluster / radial pilot."""

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.spectral_cluster_radial import (
    compute_cluster_radial_update,
)

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


def sample(values=(1., 1. - 1e-5, 1. - 2e-5, 1. - 4e-5, .4, .2)):
    s = torch.tensor(values, device=DEVICE)
    u = torch.eye(len(s), device=DEVICE)
    x = torch.diag(s)
    gen = torch.Generator(device=DEVICE).manual_seed(25)
    m = torch.randn(x.shape, generator=gen, device=DEVICE)
    cache = host._spectral_point_cache_from_decomposition(u, s, u)
    return x, m, cache


def update(x, m, cache, mode="adaptive4", coefficient=1., lr=.015):
    work = x.clone()
    d, info = compute_cluster_radial_update(
        work, m, 1., 10, 8, effective_lr=lr, cluster_mode=mode,
        cluster_threshold_rel=5e-5, gap_threshold_rel=5e-5,
        pgd_lr_scale=coefficient, current_spectral_cache=cache,
    )
    return work - lr * d, info


class ClusterRadialTests(unittest.TestCase):
    def test_training_aggregation_keeps_multiplier_mean(self):
        # Exercise the actual collector without importing Megatron's GPU runtime.
        from collections import defaultdict
        from types import SimpleNamespace
        from typing import Dict, Optional
        path = Path(host.__file__).parents[2] / "megatron/training/training.py"
        tree = ast.parse(path.read_text())
        collector = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "_collect_spel_pgd_branch_stats")
        namespace = dict(defaultdict=defaultdict, _HIERARCHICAL_REJECTION_REASONS=(),
                         Optional=Optional, Dict=Dict)
        exec(compile(ast.Module(body=[collector], type_ignores=[]), str(path), "exec"), namespace)
        fake = SimpleNamespace(get_branch_stats=lambda: dict(
            cluster_rank4_count=2, cluster_multiplier_sum=5.,
            total_matrix_updates=2, cumulative_total_matrix_updates=2,
        ))
        stats = namespace["_collect_spel_pgd_branch_stats"](fake)
        self.assertEqual(stats["cluster_multiplier_mean"], 2.5)
        self.assertNotIn("cluster_multiplier_sum", stats)

    def test_rank_choice_does_not_label_near_cluster_exact_multiplicity(self):
        x, m, cache = sample()
        self.assertEqual(cache.top_u.shape[1], 1)
        for mode, rank in (("control", 0), ("fixed2", 2), ("fixed4", 4), ("adaptive4", 4)):
            _, info = update(x, m, cache, mode)
            self.assertEqual(info.cluster_dimension, rank)
            self.assertEqual(info.p2gd_top_multiplicity, 0)
            self.assertEqual(info.projection_mode, "cluster_radial")

    def test_one_trial_svd_and_no_projection(self):
        x, m, cache = sample()
        for mode in ("control", "fixed2", "fixed4", "adaptive4"):
            with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
                with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("extra vals")):
                    with patch.object(host, "_post_project_trial", side_effect=AssertionError("projection")):
                        returned, info = update(x, m, cache, mode)
            self.assertEqual(svd.call_count, 1)
            self.assertTrue(info.spectral_cache_hit)
            s = torch.linalg.svdvals(returned.double())
            self.assertLess(abs(float(s[0]) - 1), 4e-6)
            self.assertLess(abs(float(1 - s[1] / s[0]) - info.next_spectral_cache.rel_gap), 4e-6)
            torch.testing.assert_close(info.next_spectral_cache.s_basis.double(), s[:8],
                                       atol=4e-6, rtol=4e-6)

    def test_control_matches_existing_exact_mc_sd(self):
        x, m, cache = sample((1., .8, .7, .4, .3, .2))
        returned, _ = update(x, m, cache, "control")
        work = x.clone()
        d, _ = host.compute_spel_pgd_same_projection_update(
            work, m, 1., 10, 8, effective_lr=.015, branch_mode="spel",
            projection_mode="tp_radial_pgd_exact", tangent_project_after_msign=True,
            post_return_spectral_cache=True, current_spectral_cache=cache,
            main_power_dtype="fp32",
        )
        torch.testing.assert_close(returned, work - .015 * d, atol=3e-6, rtol=3e-6)

    def test_literal_matched_direction_and_radial_return(self):
        x, m, cache = sample()
        p = host.project_to_tangent_plane(m, cache.u1, cache.v1)
        p = p / torch.linalg.vector_norm(p).clamp_min(1e-7)
        reference = host.project_to_tangent_plane(host.msign(p, steps=8).float(),
                                                 cache.u1, cache.v1)
        d = host.project_spectral_sphere_cone(-m, cache.u_basis[:, :4], cache.v_basis[:, :4])
        y = x + .015 * .5 * torch.linalg.vector_norm(reference) / torch.linalg.vector_norm(d) * d
        expected = y / host.accurate_svdvals_fp32(y)[0]
        returned, info = update(x, m, cache, coefficient=.5)
        torch.testing.assert_close(returned, expected, atol=3e-6, rtol=3e-6)
        self.assertEqual(info.pgd_projection_clipped_count, 0)

    def test_cap_overflow_is_logged_not_repaired(self):
        x, m, cache = sample((1., 1., 1., 1., 1., 1., .5))
        _, info = update(x, m, cache)
        self.assertEqual(info.cluster_dimension, 4)
        self.assertTrue(info.cluster_cap_overflow)

    def test_bootstrap_zero_lr_then_single_trial(self):
        x, m, _ = sample()
        with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
            returned, info = update(x * 1.1, m, None, lr=0.)
            self.assertEqual(svd.call_count, 1)
            self.assertEqual(info.branch, "zero_lr")
            svd.reset_mock()
            _, info = update(returned, m, info.next_spectral_cache)
            self.assertEqual(svd.call_count, 1)
            self.assertTrue(info.spectral_cache_hit)

    def test_optimizer_dispatch_and_cumulative_stats(self):
        x, m, _ = sample()
        p = torch.nn.Parameter(x)
        opt = host.SpELPGDSameProjection(
            [p], lr=.015, radial_cluster_mode="fixed4",
            gap_threshold_rel=5e-5, cluster_threshold_rel=5e-5,
            pgd_lr_scale=1., pgd_direction_normalization="none",
            projection_mode="tp_radial_pgd_exact", post_return_spectral_cache=True,
            tangent_project_after_msign=True,
        )
        with torch.no_grad():
            opt._compute_component_update(p, m, None, None, current_lr=0.,
                                          param_name="test")
            d = opt._compute_component_update(p, m, None, None, current_lr=.015,
                                              param_name="test")
            p.add_(d, alpha=-.015)
        self.assertEqual(opt.get_branch_stats()["cluster_rank4_count"], 1)
        self.assertLess(abs(float(torch.linalg.svdvals(p.detach().double())[0]) - 1), 4e-6)

    def test_safe_branches_are_identical(self):
        x, m, cache = sample((1., .8, .7, .4, .3, .2))
        expected, _ = update(x, m, cache, "control")
        for mode in ("fixed2", "fixed4", "adaptive4"):
            returned, info = update(x, m, cache, mode)
            self.assertEqual(info.branch, "spel")
            torch.testing.assert_close(returned, expected, atol=0., rtol=0.)

    def test_cluster_width_and_route_threshold_are_independent(self):
        for second_gap, expected_branch in ((2e-5, "pgd"), (2e-4, "spel")):
            x, m, cache = sample((1., 1. - second_gap, 1. - 3e-4, 1. - 1e-3, .4, .2))
            for tolerance, expected_rank in ((5e-5, 2), (5e-4, 3), (2e-3, 4)):
                _, info = compute_cluster_radial_update(
                    x.clone(), m, 1., 10, 8, effective_lr=.015,
                    cluster_mode="adaptive4", cluster_threshold_rel=tolerance,
                    gap_threshold_rel=5e-5, pgd_lr_scale=1.,
                    current_spectral_cache=cache,
                )
                self.assertEqual(info.branch, expected_branch)
                self.assertEqual(info.cluster_dimension,
                                 expected_rank if expected_branch == "pgd" else 0)

    def test_followup_grid_is_unique_and_bounded(self):
        import subprocess
        launcher = Path(__file__).resolve().parents[1] / "slurm/cluster_radial_followup_100m.sbatch"
        if not launcher.exists():
            self.skipTest("follow-up launcher not part of this source snapshot")
        rows = []
        for i in range(14):
            row = subprocess.check_output(
                ["bash", str(launcher), "--describe"],
                env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}, text=True,
            ).strip().split("\t")
            self.assertEqual(row[0], str(i))
            self.assertGreaterEqual(float(row[4]), float(row[3]))
            self.assertIn(float(row[5]), (.25, .5, 1.))
            rows.append(tuple(row[1:]))
        self.assertEqual(len(set(rows)), 14)
        self.assertEqual(rows[0][0], "muon_ball_dist")
        self.assertEqual(rows[1][0], "spectral_ball_dist")
        self.assertEqual(rows[12][1], "control")
        self.assertEqual(rows[12][-1], rows[13][-1])
        self.assertEqual(rows[13][1], "fixed4")


if __name__ == "__main__":
    unittest.main()
