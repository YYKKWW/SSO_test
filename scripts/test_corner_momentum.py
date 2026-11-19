import math
import unittest

import torch

from audit_corner_momentum import audit_state, cone_metrics, directions, radial_ascent_example
from local_cone_lmo_feasibility import cone
from local_step_aware_hybrid import sym


class CornerMomentumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cone_projection_geometry_and_rotation_invariance(self):
        rng=torch.Generator().manual_seed(41)
        u=torch.linalg.qr(torch.randn(10,3,dtype=torch.float64,generator=rng))[0]
        v=torch.linalg.qr(torch.randn(8,3,dtype=torch.float64,generator=rng))[0]
        z=torch.randn(10,8,dtype=torch.float64,generator=rng)
        p=cone(z,u,v)
        self.assertLess(abs(float(torch.linalg.eigvalsh(sym(u.T@p@v))[-1])),1e-12)
        self.assertAlmostEqual(float((z*p).sum()),float(p.square().sum()),places=11)
        rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64,generator=rng))[0]
        torch.testing.assert_close(p,cone(z,u@rotation,v@rotation))
        metrics=cone_metrics(z,u,v)
        self.assertAlmostEqual(math.cos(math.radians(metrics['angle_degrees'])),float(p.norm()/z.norm()),places=11)

    def test_cone_is_not_sign_symmetric(self):
        eye=torch.eye(2,dtype=torch.float64)
        z=torch.diag(torch.tensor([2.,1.],dtype=torch.float64))
        plus,minus=cone_metrics(z,eye,eye),cone_metrics(-z,eye,eye)
        self.assertAlmostEqual(plus['angle_degrees'],90.)
        self.assertLess(minus['angle_degrees'],30.)

    def test_radial_directional_derivative_at_multiple_top(self):
        x=torch.diag(torch.tensor([1.,1.,.5],dtype=torch.float64))
        d=torch.tensor([[.2,.4,.1],[.3,-.2,.3],[.2,.1,-.1]],dtype=torch.float64)
        h=torch.linalg.eigvalsh(sym(d[:2,:2]))[-1]
        first=d-h*x
        self.assertLess(abs(float(torch.linalg.eigvalsh(sym(first[:2,:2]))[-1])),1e-12)
        alpha=1e-7
        y=x+alpha*d
        delta=y/torch.linalg.svdvals(y)[0]-x
        torch.testing.assert_close(delta/alpha,first,atol=1e-7,rtol=1e-7)

    def test_near_cluster_is_not_true_multiplicity(self):
        x=torch.diag(torch.tensor([1.,.9999,.3],dtype=torch.float64))
        m=torch.ones_like(x)
        info,rows=audit_state(x,m,1.,.015,gradient=m)
        self.assertEqual(info['true_q'],1)
        self.assertEqual(info['cluster_q'],2)
        self.assertEqual(len(rows),9)
        self.assertTrue(all(r['quadratic_loss_change'] is not None for r in rows))

    def test_shared_trial_length_and_loss_separation(self):
        x=torch.diag(torch.tensor([1.,1.,.3],dtype=torch.float64))
        m=torch.arange(1.,10.,dtype=torch.float64).reshape(3,3)
        info,rows=audit_state(x,m,1.,.015)
        r={row['method']:row for row in rows}
        for key in ('ambient_050','top1_050','cluster_cone_050'):
            self.assertAlmostEqual(r[key]['trial_fro_length'],r['half_d0']['trial_fro_length'],places=12)
        self.assertTrue(all(row['quadratic_loss_change'] is None for row in rows))
        self.assertEqual(r['ambient_050']['trial_fro_length'],r['ambient_050_project']['trial_fro_length'])

    def test_radial_can_increase_true_loss(self):
        example=radial_ascent_example()
        self.assertLess(example['raw_direction_pairing'],0.)
        self.assertGreater(example['quadratic_loss_change'],0.)
        self.assertEqual(example['cone_projection_norm'],0.)


if __name__=='__main__':
    unittest.main()
