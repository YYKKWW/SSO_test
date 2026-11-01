"""Algebra, production routing and the matched six-arm follow-up."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers.spectral_cone_lmo import (
    matrix_sign_cone_direction, spectral_cone_lmo_direction)
from test_cluster_near_projection import ROOT, fixture, host, update


def polar(x):
    u, _, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


class ConeLMOTests(unittest.TestCase):
    def test_exact_cone_and_descent_identity(self):
        generator = torch.Generator().manual_seed(901)
        device = os.environ.get("P2GD_TEST_DEVICE", "cpu")
        for m, n in ((4, 4), (7, 5), (5, 7), (16, 12)):
            for q in range(1, min(m, n) + 1):
                for _ in range(4):
                    z = torch.randn(m, n, generator=generator, dtype=torch.float64).to(device)
                    u = torch.linalg.qr(torch.randn(m, q, generator=generator, dtype=torch.float64).to(device))[0]
                    v = torch.linalg.qr(torch.randn(n, q, generator=generator, dtype=torch.float64).to(device))[0]
                    projected = host.project_spectral_sphere_cone(z, u, v)
                    d = spectral_cone_lmo_direction(z, u, v, polar)
                    h = u.T @ d @ v
                    self.assertLess(abs(float(torch.linalg.eigvalsh(.5 * (h+h.T))[-1])), 2e-10)
                    dual = torch.linalg.svdvals(projected).sum()
                    self.assertGreaterEqual(float((z*d).sum()-dual), -2e-10)
                    self.assertLessEqual(float(torch.linalg.vector_norm(d)), min(m, n)**.5 + 2e-10)

    def test_q1_reduces_to_mc_sd_tp_and_fro_lmo(self):
        torch.manual_seed(15)
        z = torch.randn(8, 6, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(torch.randn_like(z), full_matrices=False)
        for q in (1, 3):
            uq, vq = u[:, :q], vh.T[:, :q]
            projected = host.project_spectral_sphere_cone(z, uq, vq)
            fro = lambda x: x / torch.linalg.vector_norm(x).clamp_min(1e-30)
            actual = spectral_cone_lmo_direction(z, uq, vq, fro)
            torch.testing.assert_close(actual, fro(projected), atol=1e-12, rtol=1e-12)
            if q == 1:
                expected = host.project_spectral_sphere_cone(polar(projected), uq, vq)
                torch.testing.assert_close(spectral_cone_lmo_direction(z, uq, vq, polar), expected)

    def test_production_one_msign_one_small_svd(self):
        x, m, cache = fixture()
        with patch.object(host, "accurate_svd_fp32", side_effect=AssertionError("full SVD")):
            with patch.object(host, "accurate_svdvals_fp32", side_effect=AssertionError("svdvals")):
                with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
                    with patch.object(host, "msign", wraps=host.msign) as sign:
                        result, info = update(x, m, cache, "cone_lmo", spectrum="warm_block8")
        self.assertEqual(sign.call_count, 1)
        self.assertEqual(svd.call_count, 1)
        self.assertEqual(tuple(svd.call_args.args[0].shape), (8, 8))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(info.cluster_dimension, 4)

    def test_zero_and_safe_branch(self):
        x, m, cache = fixture()
        d = matrix_sign_cone_direction(torch.zeros_like(m), cache.u_basis[:, :4],
                                      cache.v_basis[:, :4], msign=host.msign)
        torch.testing.assert_close(d, torch.zeros_like(d))
        reference, _ = update(x, m, cache, mode="control")
        actual, _ = update(x, m, cache, "cone_lmo", mode="control")
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_audit_matches_trial_and_scale_invariance(self):
        from audit_cluster_ritz_snapshots import direction
        x, m, cache = fixture()
        d = direction(m, cache.u_basis, cache.s_basis, cache.v_basis,
                      mode="adaptive4", threshold=5e-5, cluster_threshold=.002,
                      coefficient=8., direction_mode="momentum_fro", near_projection="cone_lmo")
        with patch.object(host, "accurate_svd_fp32", wraps=host.accurate_svd_fp32) as svd:
            result, _ = update(x, m, cache, "cone_lmo")
        torch.testing.assert_close(svd.call_args.args[0], x + .015*d)
        scaled, _ = update(x, 100*m, cache, "cone_lmo")
        torch.testing.assert_close(result, scaled, atol=3e-5, rtol=3e-5)

    def test_grid(self):
        launcher = ROOT / "slurm/cluster_cone_lmo_1b_20261031.sbatch"
        rows = [subprocess.check_output(["bash", str(launcher), "--describe"], text=True,
                env={**os.environ, "SLURM_ARRAY_TASK_ID": str(i)}).strip().split("\t") for i in range(6)]
        self.assertEqual([r[1] for r in rows[:2]], ["muon_ball_dist", "spectral_ball_dist"])
        self.assertEqual(rows[2][2], "control")
        self.assertEqual([r[3] for r in rows[3:]], ["cone", "cone", "cone_lmo"])
        self.assertEqual([float(r[4]) for r in rows[3:]], [.002, .0002, .0002])
        self.assertEqual([float(r[5]) for r in rows[3:]], [8., 8., 1.])
        self.assertTrue(all(r[6:] == ["2345", "1908", "20"] for r in rows))

    def test_replication_from_slurm_spool(self):
        launcher = ROOT / "slurm/cluster_replication_1b_20261101.sbatch"
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "slurm_script"
            spool.write_text(launcher.read_text())
            rows = [subprocess.check_output(["bash", str(spool), "--describe"], text=True,
                    env={**os.environ, "PROJECT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": str(i)}
                    ).strip().split("\t") for i in range(2)]
        self.assertEqual([r[0] for r in rows], ["2", "3"])
        self.assertEqual([r[2] for r in rows], ["control", "adaptive4"])
        self.assertEqual([float(r[5]) for r in rows], [1., 8.])
        self.assertTrue(all(r[6:] == ["1234", "1908", "20"] for r in rows))

    def test_scale_audit_keeps_geometry_fixed(self):
        from audit_cluster_cone_lmo import LMO_SCALES
        lmo = [a for a in LMO_SCALES if a[1] == "cone_lmo"]
        self.assertEqual([a[3] for a in lmo], [.5, 1., 2.])
        self.assertTrue(all(a[2] == .0002 and not a[4] for a in lmo))

    def test_standalone_import_without_training_pythonpath(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run([sys.executable, "-B", "-c", "import test_cluster_cone_lmo"],
                       cwd=ROOT / "scripts", env=env, check=True, capture_output=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
