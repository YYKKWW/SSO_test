"""Direction, cache, capacity, launch and no-online-SVD regression tests."""
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from kernel_lmo_h20 import host, pilot_dispatch, sym
from direct_top_pgd_pk import DirectPKAdapter
from run_kernel_lmo_pilot import main
from warm_p2gd_transfer import MODES, WarmP2GDAdapter, near_direction

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


class WarmTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self, q=3):
        s = torch.linspace(.7, .1, 16, device=DEVICE)
        s[:q] = 1.
        g = torch.Generator(device=DEVICE).manual_seed(712)
        return torch.diag(s), torch.randn(16, 16, device=DEVICE, generator=g)

    def test_cone_and_top1_formulas_same_raw_scale(self):
        x, m = self.sample()
        uv = torch.eye(16, device=DEVICE)[:, :3]
        d0 = m.clone()
        z = near_direction(m, d0, uv, uv, "ambient", .075)
        torch.testing.assert_close(z.norm(), .075*d0.norm())
        cone = near_direction(m, d0, uv, uv, "cone", .075)
        top = near_direction(m, d0, uv, uv, "top1", .075)
        eigen = torch.linalg.eigvalsh(sym(uv.T@cone@uv))
        self.assertLess(abs(float(eigen[-1])), 1e-6)
        self.assertLess(abs(float(top[0, 0])), 1e-6)
        self.assertLessEqual(float(cone.norm()), float(z.norm())+1e-6)
        torch.testing.assert_close(cone[3:], z[3:])
        for mode in MODES:
            self.assertEqual(float(near_direction(m, torch.zeros_like(m), uv, uv, mode, .075).norm()), 0.)
            self.assertEqual(float(near_direction(torch.zeros_like(m), d0, uv, uv, mode, .075).norm()), 0.)

    def test_safe_updates_equal_completed_warm_control(self):
        x, m = self.sample(1)
        control = DirectPKAdapter("none", "warm_block8", ambient_return="radial")
        _, init = control(x, m, 1., 10, 8, effective_lr=0.)
        expected, _ = control(x.clone(), m, 1., 10, 8, effective_lr=.015,
                              current_spectral_cache=init.next_spectral_cache)
        for mode in MODES:
            a = WarmP2GDAdapter(mode)
            actual, _ = a(x.clone(), m, 1., 10, 8, effective_lr=.015,
                          current_spectral_cache=init.next_spectral_cache)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(a.last["record"]["near"])

    def test_persistent_cache_and_one_8x8_svd_per_trial_in_host(self):
        for mode in MODES:
            x, m = self.sample()
            p = torch.nn.Parameter(x.clone())
            a = WarmP2GDAdapter(mode)
            opt = host.SpELPGDSameProjection([p], lr=.001, radial_cluster_mode="adaptive4",
                cluster_spectrum_mode="warm_block8", cluster_iterations=10,
                gap_threshold_rel=2e-4, cluster_threshold_rel=2e-4, pgd_lr_scale=1.,
                pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
                post_return_spectral_cache=True, tangent_project_after_msign=True)
            original = torch.linalg.svd
            calls = []
            def small_only(t, *args, **kw):
                self.assertEqual(tuple(t.shape), (8, 8))
                self.assertEqual(t.dtype, torch.float32)
                calls.append(1)
                return original(t, *args, **kw)
            with torch.no_grad(), pilot_dispatch(a), \
                 patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")), \
                 patch("torch.linalg.svdvals", side_effect=AssertionError("norm SVD")), \
                 patch("torch.linalg.svd", side_effect=small_only):
                opt._compute_component_update(p, m, None, None, current_lr=0., param_name="test")
                for _ in range(4):
                    update = opt._compute_component_update(p, m, None, None, current_lr=.001, param_name="test")
                    p.add_(update, alpha=-.001)
                    torch.testing.assert_close(p, a.last["returned"])
            self.assertEqual(len(calls), 5)
            self.assertEqual(a.counts["bootstrap_solves"], 1)
            self.assertEqual(a.counts["trial_solves"], 4)
            self.assertGreater(a.counts["near"], 0)
            self.assertFalse(a.report()["actual_elr_matched"])

    def test_capacity_overflow_is_not_silently_certified(self):
        x, m = self.sample(7)
        a = WarmP2GDAdapter("cone")
        _, info = a(x, m, 1., 10, 8, effective_lr=.001)
        self.assertEqual(info.cluster_dimension, 6)
        self.assertTrue(info.cluster_cap_overflow)
        self.assertEqual(a.counts["direction_cap_overflow"], 1)
        self.assertFalse(info.next_spectral_cache.is_exact)

    def test_launches_are_paired_except_near_policy(self):
        configs = []
        with tempfile.TemporaryDirectory() as root:
            for mode in MODES:
                def fake_train(env, folder, smoke):
                    folder.mkdir(parents=True, exist_ok=True)
                    if not smoke:
                        configs.append(dict(env))
                argv = ["run", "--study", "direct_pk", "--arm", "0", "--budget", "3b",
                        "--solver", "warm_block8", "--ambient-return", "radial", "--warm-p2gd-mode", mode]
                with patch.dict(os.environ, dict(PROJECT_DIR=root, SLURM_ARRAY_JOB_ID="test", SLURM_ARRAY_TASK_ID="0")), \
                     patch("sys.argv", argv), patch("run_kernel_lmo_pilot.run_training", side_effect=fake_train), \
                     patch("run_kernel_lmo_pilot.subprocess.run"):
                    main()
        first = configs[0]
        for c in configs:
            changed = {k for k in set(first)|set(c) if first.get(k) != c.get(k)}
            self.assertLessEqual(changed, {"JOB_NAME", "KERNEL_WARM_P2GD_MODE"})
            for key, expected in dict(TRAIN_ITER="5723", SEED="2345", WIDTH="256", SAVE_CHECKPOINT="0",
                                      EVAL_ITERS="20", KERNEL_WARM_P2GD_RATIO="0.075").items():
                self.assertEqual(c[key], expected)


if __name__ == "__main__":
    unittest.main()
