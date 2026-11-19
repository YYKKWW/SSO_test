"""Isolate near-step attenuation without changing the warm spectral backend."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from direct_top_pgd_pk import DirectPKAdapter
from kernel_lmo_h20 import host, pilot_dispatch
from run_kernel_lmo_pilot import main

DEVICE = os.environ.get("P2GD_TEST_DEVICE", "cpu")


class WarmControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self, near):
        s = torch.linspace(.6, .1, 16, device=DEVICE)
        s[:2] = torch.tensor([1., .99999 if near else .75], device=DEVICE)
        x = torch.diag(s)
        rng = torch.Generator(device=DEVICE).manual_seed(947)
        return x, torch.randn(x.shape, generator=rng, device=DEVICE)

    def test_same_cache_d0_and_only_near_trial_is_halved(self):
        for near in (True, False):
            x, m = self.sample(near)
            control = DirectPKAdapter("none", "warm_block8", ambient_return="radial")
            half = DirectPKAdapter("tp_half", "warm_block8", ambient_return="radial")
            _, initial = control(x, m, 1., 10, 8, effective_lr=0.)
            cache = initial.next_spectral_cache
            control(x.clone(), m, 1., 10, 8, effective_lr=.015, current_spectral_cache=cache)
            half(x.clone(), m, 1., 10, 8, effective_lr=.015, current_spectral_cache=cache)
            a, b = control.last, half.last
            self.assertEqual(b["record"]["near"], near)
            torch.testing.assert_close(a["d0"], b["d0"], atol=0, rtol=0)
            ratio = .5 if near else 1.
            torch.testing.assert_close(b["direction"], ratio*a["direction"], atol=0, rtol=0)
            torch.testing.assert_close(b["trial"], x+.015*ratio*a["direction"])
            self.assertEqual(half.counts["tp_near_attenuation_calls"], int(near))
            self.assertEqual(half.counts["ambient_near_calls"], 0)
            self.assertEqual(half.counts["pgd_calls"], 0)
            if not near:
                torch.testing.assert_close(a["returned"], b["returned"], atol=0, rtol=0)

    def test_zero_d0_stays_zero_without_ambient_reference_fallback(self):
        x, _ = self.sample(True)
        adapter = DirectPKAdapter("tp_half", "warm_block8", ambient_return="radial")
        _, initial = adapter(x, torch.zeros_like(x), 1., 10, 8, effective_lr=0.)
        adapter(x, torch.zeros_like(x), 1., 10, 8, effective_lr=.015,
                current_spectral_cache=initial.next_spectral_cache)
        self.assertEqual(float(adapter.last["direction"].norm()), 0.)
        self.assertEqual(adapter.counts["ambient_near_calls"], 0)

    def test_host_cache_and_only_one_small_trial_svd(self):
        x, m = self.sample(True)
        p = torch.nn.Parameter(x.clone())
        adapter = DirectPKAdapter("tp_half", "warm_block8", ambient_return="radial")
        opt = host.SpELPGDSameProjection([p], lr=.001, radial_cluster_mode="adaptive4",
            cluster_spectrum_mode="warm_block8", cluster_iterations=10,
            gap_threshold_rel=2e-4, cluster_threshold_rel=2e-4, pgd_lr_scale=1.,
            pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
            post_return_spectral_cache=True, tangent_project_after_msign=True)
        original = torch.linalg.svd
        calls = []
        def small_only(a, *args, **kwargs):
            self.assertEqual(tuple(a.shape), (8, 8))
            self.assertEqual(a.dtype, torch.float32)
            calls.append(a.shape)
            return original(a, *args, **kwargs)
        with torch.no_grad(), pilot_dispatch(adapter), \
             patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")), \
             patch("torch.linalg.svdvals", side_effect=AssertionError("svdvals")), \
             patch("torch.linalg.svd", side_effect=small_only):
            opt._compute_component_update(p, m, None, None, current_lr=0., param_name="test")
            for _ in range(3):
                update = opt._compute_component_update(p, m, None, None, current_lr=.001, param_name="test")
                p.add_(update, alpha=-.001)
                torch.testing.assert_close(p, adapter.last["returned"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(adapter.counts["bootstrap_solves"], 1)
        self.assertEqual(adapter.counts["trial_solves"], 3)
        self.assertGreater(adapter.counts["tp_near_attenuation_calls"], 0)
        self.assertEqual(adapter.report()["method"], "mcsd_tp_near_half")

    def test_launcher_changes_only_method_and_near_scale_metadata(self):
        configs = []
        with tempfile.TemporaryDirectory() as root:
            for half in (False, True):
                captured = []
                def fake_train(env, folder, smoke):
                    folder.mkdir(parents=True, exist_ok=True)
                    if not smoke:
                        captured.append(dict(env))
                argv = ["run_kernel_lmo_pilot.py", "--study", "direct_pk", "--arm", "0",
                        "--budget", "3b", "--ambient-return", "radial", "--solver", "warm_block8"]
                if half:
                    argv.append("--near-tp-half")
                with patch.dict(os.environ, dict(PROJECT_DIR=root, SLURM_ARRAY_JOB_ID="test", SLURM_ARRAY_TASK_ID="0")), \
                     patch("sys.argv", argv), patch("run_kernel_lmo_pilot.run_training", side_effect=fake_train), \
                     patch("run_kernel_lmo_pilot.subprocess.run"):
                    main()
                configs.append(captured[0])
        a, b = configs
        changed = {k for k in set(a)|set(b) if a.get(k) != b.get(k)}
        self.assertEqual(changed, {"KERNEL_METHOD", "KERNEL_TP_NEAR_RATIO", "JOB_NAME"})
        self.assertEqual(b["TRAIN_ITER"], "5723")
        self.assertEqual(b["SAVE_CHECKPOINT"], "0")
        self.assertEqual(b["KERNEL_METHOD"], "tp_half")
        self.assertEqual(b["KERNEL_TP_NEAR_RATIO"], "0.5")
        self.assertEqual(b["KERNEL_PGD_TRIAL_RATIO"], "0.0")

    def test_half_control_requires_radial_return(self):
        with self.assertRaises(ValueError):
            DirectPKAdapter("tp_half")


if __name__ == "__main__":
    unittest.main()
