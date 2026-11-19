"""Geometry regression tests independent of the production optimizer."""
import math
import unittest
import torch

from local_step_aware_hybrid import (
    METHODS, amplitude_audit, capacity_audit, counterexamples, direction, exact_metrics, gamma,
    make_case, model_project, polar, radial_cached, spectrum, sym, tangent,
    trajectory, window_q,
)


class StepAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)

    def test_972_model_and_return_cases(self):
        count = 0
        for shape in ((4, 4), (8, 6), (6, 8)):
            r = min(shape)
            for h in (.2, .02, .002):
                for kind in range(3):
                    for seed in range(12):
                        gen = torch.Generator().manual_seed(10000+seed)
                        u = torch.linalg.qr(torch.randn(shape[0],r,generator=gen))[0]
                        v = torch.linalg.qr(torch.randn(shape[1],r,generator=gen))[0]
                        s = torch.cat((torch.tensor([1.,1-.2*h,1-.8*h]),torch.linspace(.5,.2,r-3)))
                        if kind == 1: s[1] = 1
                        if kind == 2: s[:] = 1
                        x = (u*s)@v.T
                        m = torch.randn(*shape,generator=gen)
                        q = window_q(s,1.,h)
                        for factor in (.02,.1,.2):
                            alpha = factor*h/(2*math.sqrt(r))
                            d,info = direction(m,(u,s,v),1.,alpha,q,"A4")
                            df = model_project(-m/torch.linalg.vector_norm(m),u[:,:q],s[:q],v[:,:q],1.,alpha)
                            model = torch.diag(s[:q])+alpha*sym(u[:,:q].T@d@v[:,:q])
                            self.assertLess(abs(float(torch.linalg.eigvalsh(model)[-1])-1),3e-12)
                            self.assertLessEqual(float((m*(d-df)).sum()),1e-11)
                            pl = tangent(m,u[:,:q],v[:,:q])
                            self.assertLessEqual(float((m*df).sum())+float(pl.square().sum())/(2*float(torch.linalg.vector_norm(m))),1e-11)
                            norm=float(torch.linalg.matrix_norm(d,ord=2))
                            updated,cache,trial_norm=radial_cached(x+alpha*d,1.)
                            self.assertGreaterEqual(trial_norm,1-2e-12)
                            self.assertLessEqual(trial_norm-1,2*(alpha*norm)**2/gamma(s,q,1.)+2e-12)
                            self.assertLessEqual(float(torch.linalg.vector_norm(d)),2*math.sqrt(r)+1e-11)
                            torch.testing.assert_close(cache[1],torch.linalg.svdvals(updated),atol=2e-12,rtol=2e-12)
                            count += 1
        self.assertEqual(count,972)

    def test_q1_all_methods_agree(self):
        x,_,_,_=make_case(0,(8,6))
        spec=spectrum(x); m=torch.randn_like(x)
        ds=[direction(m,spec,1.,.01,1,method)[0] for method in METHODS]
        for d in ds[1:]: torch.testing.assert_close(d,ds[0],rtol=0,atol=0)

    def test_original_radial_ascent_and_full_cluster_repair(self):
        for row in counterexamples():
            if row['method']=='MCSD-TP':
                self.assertLess(row['predicted'],0)
                self.assertGreater(row['actual_over_alpha'],0)
            else:
                self.assertLessEqual(row['actual_over_alpha'],1e-9)

    def test_cross_face_and_invalid_post_scaling(self):
        x=torch.diag(torch.tensor([1.,.9999])); u,s,v=spectrum(x)
        d=model_project(torch.diag(torch.tensor([-1.,0.])),u,s,v,1.,.01)
        torch.testing.assert_close(d,torch.diag(torch.tensor([-1.,.01])))
        self.assertAlmostEqual(float(torch.linalg.matrix_norm(x+.01*d,ord=2)),1.)
        self.assertGreater(float(torch.linalg.matrix_norm(x+.01*2*d,ord=2)),1.)

    def test_radius_scale_equivariance(self):
        x,_,_,_=make_case(8,(8,6)); m=torch.randn_like(x)
        spec=spectrum(x); q=window_q(spec[1],1.,.05)
        for method in METHODS:
            d,_=direction(m,spec,1.,.001,q,method)
            y,_,_=radial_cached(x+.001*d,1.)
            for radius in (.1,10.):
                scaled=(spec[0],spec[1]*radius,spec[2])
                other,_=direction(m*radius,scaled,radius,.001*radius,q,method)
                torch.testing.assert_close(d,other,atol=2e-10,rtol=2e-10)
                result,_,_=radial_cached(radius*x+.001*radius*other,radius)
                torch.testing.assert_close(result,radius*y,atol=2e-10,rtol=2e-10)

    def test_zero_and_validation(self):
        spec=spectrum(torch.eye(4))
        for method in METHODS:
            d,_=direction(torch.zeros(4,4),spec,1.,.01,4,method)
            self.assertEqual(float(d.abs().max()),0.)
        with self.assertRaises(ValueError):
            model_project(torch.eye(4),* (spec[0],spec[1],spec[2]),1.,0.)

    def test_clarke_does_not_imply_b_stationary(self):
        metrics=exact_metrics(torch.diag(torch.tensor([1.,0.])),spectrum(torch.eye(2)),1.)
        self.assertEqual(metrics['clarke_residual'],0.)
        self.assertAlmostEqual(metrics['b_residual'],1.)

    def test_paired_repeatability_and_call_accounting(self):
        x,a,w,meta=make_case(8,(4,4))
        noise=torch.ones(4,4,4)/4
        rows=[trajectory(x,a,w,meta,noise,'A4','certified_ema',4) for _ in range(2)]
        self.assertEqual(rows[0]['trace'],rows[1]['trace'])
        self.assertEqual(rows[0]['calls']['trial_weight_svd'],4)
        self.assertEqual(rows[0]['counts']['return_bound_violation'],0)

    def test_top1_at_multiple_root_has_no_external_gap_bound(self):
        x=torch.eye(4); a=.5*x; w=torch.ones_like(x)
        meta=dict(radius=1.,fstar=.125,seed=0,shape=(4,4))
        row=trajectory(x,a,w,meta,torch.zeros(4,4,4),'MCSD-TP','certified_det',4)
        self.assertFalse(row['trace'][0]['bound_ok'])
        self.assertIsNone(row['trace'][0]['condition_ratio'])

    def test_matched_audit_only_changes_baseline_length(self):
        x,a,w,meta=make_case(8,(8,6)); m=torch.randn_like(x); saved=x.clone()
        for row in amplitude_audit(x,m,spectrum(x),a,w,1.,.001,4):
            self.assertAlmostEqual(row['trial_length'],row['matched_top1_trial_length'],places=12)
        torch.testing.assert_close(x,saved,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
