import json
import os
import unittest

import torch
from audit_cluster_structural_directions import (gap_aware_trial_step, host,
                                                 make_steps)


class StructuralDirectionsTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('P2GD_TEST_DEVICE', 'cpu')
        self.eye = torch.eye(5, device=self.device, dtype=torch.float64)

    def test_exact_plateau_reduces_to_cone_projection(self):
        z = torch.tensor([[-2., .2], [.1, -1.]], device=self.device, dtype=torch.float64)
        eye = self.eye[:2, :2]
        step = .01*z
        actual = gap_aware_trial_step(step, eye, torch.ones(2, device=self.device, dtype=torch.float64), eye, 1.)
        expected = .01*host.project_spectral_sphere_cone(z, eye, eye)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_top1_reduces_to_tangent_projection(self):
        z = torch.arange(25, device=self.device, dtype=torch.float64).reshape(5, 5)/100
        u = self.eye[:, :1]
        actual = gap_aware_trial_step(z, u, torch.ones(1, device=self.device, dtype=torch.float64), u, 1.)
        expected = z - u@(u.T@z@u)@u.T
        torch.testing.assert_close(actual, expected)

    def test_preserves_existing_separation_in_small_step(self):
        z = torch.diag(torch.tensor([-2., -1.], device=self.device, dtype=torch.float64))
        s = torch.tensor([1., .999], device=self.device, dtype=torch.float64)
        eye = self.eye[:2, :2]
        actual = gap_aware_trial_step(1e-4*z, eye, s, eye, 1.)
        expected = torch.diag(torch.tensor([0., -1e-4], device=self.device, dtype=torch.float64))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        flat = 1e-4*host.project_spectral_sphere_cone(z, eye, eye)
        self.assertGreater(float(torch.linalg.vector_norm(actual-flat)), 1e-4)

    def test_small_affine_model_lands_on_boundary(self):
        s = torch.tensor([1., .999, .9985], device=self.device, dtype=torch.float64)
        u = self.eye[:, :3]
        z = torch.arange(25, device=self.device, dtype=torch.float64).reshape(5, 5)/100
        corrected = gap_aware_trial_step(z, u, s, u, 1.)
        a = u.T@corrected@u
        b = torch.diag(s)+(a+a.T)*.5
        self.assertAlmostEqual(float(torch.linalg.eigvalsh(b)[-1]), 1., places=12)


if __name__ == '__main__':
    unittest.main()
