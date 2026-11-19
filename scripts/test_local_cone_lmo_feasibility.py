"""Regression checks for the distinction between the model cone and true cone."""

import unittest

import torch

from local_cone_lmo_feasibility import (
    cone, counterexamples, l1_lmo, local_return, polar,
    spectral_cone_lmo_direction, sphere_project, unit, violation,
)


class LocalConeGeometryTests(unittest.TestCase):
    def test_omitted_direction_and_near_cluster(self):
        rows = counterexamples()
        missing = rows["omitted_top_direction"]
        self.assertAlmostEqual(missing["model_cone"], 0)
        self.assertAlmostEqual(missing["true_cone"], 1)
        near = rows["near_cluster_not_true_cone"]
        self.assertAlmostEqual(near["model_cone"], 0)
        self.assertAlmostEqual(near["true_cone"], -1)

    def test_fixed_point_vs_nonuniform_returns(self):
        rows = local_return(20261106)
        self.assertLess(rows[-1]["radial_error_over_step"], rows[0]["radial_error_over_step"] / 100)
        for row in counterexamples()["nonuniform_return"]:
            self.assertGreater(row["radial_error_over_step"], .69)
            self.assertAlmostEqual(row["projection_error_over_step"], .5, places=7)
            self.assertGreater(row["linear_objective_change_over_step"], .49)

    def test_project_is_sphere_not_ball(self):
        y = torch.diag(torch.tensor([.6, .3], dtype=torch.float64))
        expected = torch.diag(torch.tensor([1., .3], dtype=torch.float64))
        torch.testing.assert_close(sphere_project(y), expected)
        y = torch.diag(torch.tensor([1.4, 1.2], dtype=torch.float64))
        torch.testing.assert_close(sphere_project(y), torch.eye(2, dtype=torch.float64))

    def test_ascent_direction_is_realizable_by_the_lmo(self):
        z = torch.diag(torch.tensor([2., 1.], dtype=torch.float64))
        u = torch.eye(2, dtype=torch.float64)[:, :1]
        d = spectral_cone_lmo_direction(z, u, u, polar)
        torch.testing.assert_close(d, torch.diag(torch.tensor([0., 1.], dtype=torch.float64)))

    def test_norm_capping_preserves_the_cone(self):
        gen = torch.Generator().manual_seed(84)
        z = torch.randn(8, 6, dtype=torch.float64, generator=gen)
        u = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64, generator=gen))[0]
        v = torch.linalg.qr(torch.randn(6, 4, dtype=torch.float64, generator=gen))[0]
        for oracle, norm in ((polar, lambda x: torch.linalg.matrix_norm(x, 2)),
                             (l1_lmo, lambda x: x.abs().sum())):
            d = spectral_cone_lmo_direction(z, u, v, oracle)
            bounded = d / norm(d).clamp_min(1)
            self.assertLessEqual(float(norm(bounded)), 1+1e-12)
            self.assertLess(abs(violation(bounded, u, v)), 1e-12)
            self.assertGreaterEqual(float((z*bounded).sum()), -1e-12)

    def test_tracking_error_direction_bound(self):
        gen = torch.Generator().manual_seed(85)
        for _ in range(20):
            g = torch.randn(8, 6, dtype=torch.float64, generator=gen)
            m = g+.1*torch.randn(8, 6, dtype=torch.float64, generator=gen)
            u = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64, generator=gen))[0]
            v = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64, generator=gen))[0]
            chi = torch.linalg.vector_norm(cone(-g, u, v))
            error = torch.linalg.vector_norm(g-m)
            for oracle, a, b in ((polar, 6**.5, 1.), (unit, 1., 1.), (l1_lmo, 1., 48**.5)):
                d = spectral_cone_lmo_direction(-m, u, v, oracle)
                bound = -chi/b+(a+1/b)*error
                self.assertLessEqual(float((g*d).sum()-bound), 1e-10)


if __name__ == "__main__":
    unittest.main()
