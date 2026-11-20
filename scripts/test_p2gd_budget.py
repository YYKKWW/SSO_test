import os
import unittest

import torch

from p2gd_budget_geometry import ARMS, candidates, match_returned, norm2, radial
from p2gd_elr_geometry import repair
import test_p2gd_elr_replay


class BudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261120)
        self.device = os.environ.get('ELR_TEST_DEVICE','cpu')
        self.dtype = torch.float32 if self.device=='cuda' else torch.float64
        self.x = torch.diag(torch.tensor([1.,.9999,.6,.2],device=self.device,dtype=self.dtype))
        self.m = torch.randn_like(self.x)
        self.d0 = torch.randn_like(self.x)
        self.d0[0,0] = 0
        self.u = torch.eye(4,device=self.device,dtype=self.dtype)[:,:2]
        self.s = torch.tensor([1.,.9999],device=self.device,dtype=self.dtype)
        self.h = torch.randn(17,4,device=self.device,dtype=self.dtype)*torch.tensor(
            [1.,.2,3.,.7],device=self.device,dtype=self.dtype)

    def test_all_budgets_and_shrink_controls(self):
        before = [x.clone() for x in (self.x,self.m,self.d0,self.h)]
        points, records, targets = candidates(self.x,self.m,self.d0,self.u,self.s,self.u,1.,.015,self.h)
        self.assertEqual(len(records),15)
        torch.testing.assert_close(points['half_d0'],radial(self.x+.0075*self.d0,1.))
        for row in records[1:]:
            expected = targets[row['metric']]*row['budget_fraction']
            value = row['returned_spectral_elr'] if row['metric']=='spectral' else row['local_output_elr']
            self.assertLessEqual(abs(value-expected),row['tolerance'])
            self.assertLess(row['radius_error'],2e-6)
            if row['direction']=='half_d0':
                self.assertLess(row['alpha_ratio'],1.)
        for actual,previous in zip((self.x,self.m,self.d0,self.h),before):
            torch.testing.assert_close(actual,previous,rtol=0,atol=0)

    def test_rebuild_finite_model_at_matched_alpha(self):
        seen=[]
        def fn(alpha):
            seen.append(alpha)
            return repair('finite_d0',self.d0,self.m,self.u,self.s,self.u,1.,alpha)
        base=radial(self.x+.0075*self.d0,1.)
        target=.95*float(norm2(base-self.x))
        d,_,row=match_returned(self.x,fn,1.,.015,target,'spectral')
        self.assertGreater(len(set(seen)),1)
        h=self.u.T@d@self.u
        model=torch.diag(self.s)+row['effective_alpha']*(h+h.T)/2
        self.assertLess(abs(float(torch.linalg.eigvalsh(model)[-1])-1),2e-6)

    def test_invalid_targets_and_zero_output(self):
        with self.assertRaises(ValueError):
            match_returned(self.x,lambda a:self.d0,1.,.015,0.,'spectral')
        with self.assertRaises(ValueError):
            match_returned(self.x,lambda a:self.d0,1.,.015,.01,'output',torch.zeros_like(self.h))
        with self.assertRaises(RuntimeError):
            match_returned(self.x,lambda a:self.x,1.,.015,.01,'spectral')

    def test_restore_fifteen_counterfactuals(self):
        test_p2gd_elr_replay.ReplayTests().check_restore(methods=tuple(a['name'] for a in ARMS))

    def test_restore_fifteen_on_failure(self):
        test_p2gd_elr_replay.ReplayTests().check_restore(failure=True,methods=tuple(a['name'] for a in ARMS))


if __name__=='__main__':
    unittest.main()
