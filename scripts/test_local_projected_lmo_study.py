import unittest

import torch

from local_projected_lmo_study import (
    ALPHA, BETA, KAPPA, candidate_direction, guarded_return, make_problem,
    projection_counterexample, sphere_project,
)
from local_cone_lmo_feasibility import polar, tangent


class ProjectedLMOTests(unittest.TestCase):
    def test_mcsd_and_tp_safe_paths_are_distinct(self):
        x=torch.diag(torch.tensor([1.,.8,.5],dtype=torch.float64))
        m=torch.tensor([[2.,1.,.4],[1.,3.,.2],[.3,.1,1.]],dtype=torch.float64)
        u=torch.eye(3,dtype=torch.float64)[:,0]
        oracle=polar(tangent(-m,u,u))
        raw,near,_=candidate_direction(x,m,"hybrid_mcsd_lmo_project")
        tp,_,_=candidate_direction(x,m,"hybrid_lmo_project")
        self.assertFalse(near)
        torch.testing.assert_close(raw,oracle)
        torch.testing.assert_close(tp,tangent(oracle,u,u))
        self.assertGreater(float(torch.linalg.vector_norm(raw-tp)),1e-3)

    def test_known_sphere_optima(self):
        expected=(.045,.08,.09,.5*(.3**2+.2999**2))
        for case in range(8):
            _,_,a=make_problem(case)
            optimum=sphere_project(a)
            self.assertAlmostEqual(float(.5*((optimum-a)**2).sum()),expected[case%4],places=12)
            self.assertAlmostEqual(float(torch.linalg.svdvals(optimum)[0]),1.,places=12)

    def test_projection_can_reverse_descent(self):
        for row in projection_counterexample():
            self.assertTrue(row['near'])
            self.assertLess(row['derivative'],0)
            self.assertGreater(row['actual_change_over_step'],0)
            self.assertFalse(row['guard_accepted'])
            self.assertLess(row['guard_change_over_step'],0)

    def test_constant_step_meets_sufficient_ema_condition(self):
        c=(1-ALPHA)/(2*ALPHA)
        self.assertGreater(c,KAPPA**2*BETA/(1-BETA))

    def test_model_bound_and_movement_cap(self):
        gen=torch.Generator().manual_seed(88)
        for case in range(8):
            _,x,a=make_problem(case)
            grad=x-a
            error=.03*torch.randn(4,4,dtype=torch.float64,generator=gen)
            m=grad+error
            d=torch.randn(4,4,dtype=torch.float64,generator=gen)
            returned,_,info=guarded_return(x,m,d,ALPHA)
            actual=returned-x
            db=info['baseline_displacement']
            self.assertLessEqual(float(torch.linalg.vector_norm(actual)),KAPPA*db+1e-12)
            c=(1-ALPHA)/(2*ALPHA)
            change=.5*((returned-a)**2).sum()-.5*((x-a)**2).sum()
            bound=-.5*c*db**2+KAPPA**2/(2*c)*(error*error).sum()
            self.assertLessEqual(float(change-bound),1e-12)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
