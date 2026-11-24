import unittest

import torch

from local_paper_sphere import (cone_batch, loss, noise_sequence, polar,
    problems, project, radial_batch, spectrum, square_residual, tangent)


class PaperSphereTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

    def test_projection_raises_inside_top_and_clips_all_outside_values(self):
        x = torch.diag_embed(torch.tensor([[.7,.5,.1], [1.4,1.2,.3]]))
        torch.testing.assert_close(project(x), torch.diag_embed(torch.tensor([[1.,.5,.1], [1.,1.,.3]])))

    def test_square_residual_does_not_snap_near_corner(self):
        x = torch.diag_embed(torch.tensor([[1.,1.], [1.,1.-1e-10]]))
        g = torch.diag_embed(torch.tensor([[0.,1.], [0.,1.]]))
        torch.testing.assert_close(square_residual(x,g), torch.tensor([0.,1.]))

    def test_limiting_stationarity_does_not_imply_minimum(self):
        x,a = problems(2,'stationary_corner',1)
        self.assertEqual(float(square_residual(x,x-a)[0]),0.)
        self.assertLess(float(loss(project(x-.1*(x-a)),a)), float(loss(x,a)))

    def test_exact_lmo_feasibility_and_tangency(self):
        x,m = problems(16,'corner',3)
        u,_,v = spectrum(x)
        q = tangent(m,u,v)
        s = -polar(q)
        self.assertLessEqual(float(torch.linalg.svdvals(s)[:,0].max()), 1.+1e-12)
        torch.testing.assert_close((q*s).sum((-2,-1)), -torch.linalg.svdvals(q).sum(-1))
        d = tangent(s,u,v)
        h = u[:,:,:1]@v[:,:,:1].transpose(-2,-1)
        self.assertLess(float((d*h).sum((-2,-1)).abs().max()), 1e-12)
        self.assertLessEqual(float(d.norm(dim=(-2,-1)).max()), 4.+1e-12)

    def test_complete_cone_has_zero_top_directional_derivative(self):
        gen = torch.Generator().manual_seed(42)
        z = torch.randn(3,8,8,generator=gen)
        basis = torch.eye(8).expand(3,-1,-1)
        counts = torch.tensor([1,2,6])
        d = cone_batch(z,basis,basis,counts)
        for i,q in enumerate(counts.tolist()):
            a = d[i,:q,:q]
            self.assertLess(abs(float(torch.linalg.eigvalsh((a+a.T)*.5)[-1])),1e-12)

    def test_pgd_full_update_descent_bound(self):
        x,a = problems(16,'corner',4)
        alpha = .02
        y = project(x-alpha*(x-a))
        rhs = -(1/(2*alpha)-.5)*((y-x)**2).sum((-2,-1))
        self.assertTrue(bool(torch.all(loss(y,a)-loss(x,a) <= rhs+1e-11)))

    def test_radial_is_not_metric_projection(self):
        y = torch.diag_embed(torch.tensor([[2.,1.]]))
        self.assertGreater(float((project(y)-radial_batch(y)).norm()), .49)

    def test_noise_pairing_and_diagonal_structure(self):
        a = noise_sequence(2,10,3)
        torch.testing.assert_close(a,noise_sequence(2,10,3),rtol=0,atol=0)
        self.assertFalse(bool((a[:,:,0,1]!=0).any()))
        self.assertFalse(bool((a[:,:,1,0]!=0).any()))


if __name__=='__main__':
    unittest.main()
