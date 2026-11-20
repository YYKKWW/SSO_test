import unittest
import torch
from returned_step_repair_study import State, METHODS, match, metrics


class ReturnedStepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(127)
        self.x = torch.diag(torch.tensor([1., .9999, .6, .2], dtype=torch.float64))
        self.m = torch.randn(4,4,dtype=torch.float64)
        self.state = State(self.x, self.m, 1., .015)

    def test_all_methods_match_returned_metrics(self):
        for metric in ('spectral','frobenius','aus'):
            target = self.state.evaluate('half_d0', .015, metric)['length']
            for method in METHODS:
                result, info = match(self.state, method, target, metric)
                self.assertTrue(info['matched'], (metric, method, info))
                self.assertLessEqual(abs(result['length']/target-1), 1e-5)
                self.assertLess(metrics(self.state,result)['radius_error'], 1e-12)

    def test_finite_step_uses_matching_alpha(self):
        from local_step_aware_hybrid import sym
        for alpha in (.001,.01,.04):
            d=self.state.direction('finite_momentum', alpha)
            h=sym(self.state.u[:,:2].T@d@self.state.v[:,:2])
            self.assertLess(abs(float(torch.linalg.eigvalsh(torch.diag(self.state.s[:2])+alpha*h)[-1])-1),1e-12)

    def test_anchor_and_unmodified_inputs(self):
        x,m=self.x.clone(),self.m.clone()
        d=self.state.direction('anchored_momentum', .015)
        h=(d[:2,:2]+d[:2,:2].T)/2
        self.assertLess(float(h[:,0].norm()),1e-12)
        self.assertLessEqual(float(torch.linalg.eigvalsh(h)[-1]),1e-12)
        torch.testing.assert_close(self.x,x)
        torch.testing.assert_close(self.m,m)

    def test_zero_reference_is_explicit_failure(self):
        state=State(self.x, torch.zeros_like(self.x), 1., .015, torch.zeros_like(self.x))
        _,info=match(state,'anchored_momentum',0.,'spectral')
        self.assertEqual(info['reason'],'zero_reference')
        self.assertFalse(info['matched'])

    def test_unreachable_ray_is_not_reported_as_success(self):
        state=State(torch.eye(2,dtype=torch.float64),-torch.eye(2,dtype=torch.float64),1.,.015,
                    torch.eye(2,dtype=torch.float64))
        _,info=match(state,'ambient',.01,'frobenius')
        self.assertFalse(info['matched'])

    def test_quadratic_complete_loss_identity(self):
        result=self.state.evaluate('finite_d0',.015,'spectral')
        g=torch.randn_like(self.x)
        target=self.state.x-g
        actual=.5*(result['returned']-target).square().sum()-.5*(self.state.x-target).square().sum()
        self.assertAlmostEqual(float(actual),metrics(self.state,result,g)['loss_change'],places=12)


if __name__=='__main__':
    unittest.main()
