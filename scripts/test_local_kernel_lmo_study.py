"""Local algebra and numerical-counterexample checks, not convergence proofs."""
import unittest
import torch
import local_step_aware_hybrid as base
import local_kernel_lmo_study as study


class KernelLMOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)

    def test_model_descent_budget_and_coupling(self):
        for shape in ((4,4),(8,6),(6,8)):
            for seed in range(16):
                x,_,_,_=base.make_case(seed,shape)
                spec=base.spectrum(x)
                gen=torch.Generator().manual_seed(10200+seed)
                m=torch.randn(*shape,generator=gen)
                for q in (2,min(shape)):
                    for alpha in (.0001,.01,.2):
                        ref=base.model_project(-m/m.norm(),spec[0][:,:q],spec[1][:q],spec[2][:,:q],1.,alpha)
                        for method,budget in (("A4-budget",2.),("K-LMO-0.5",1.5),("K-LMO-1",2.)):
                            d,_=study.direction(m,spec,1.,alpha,q,method)
                            top=torch.diag(spec[1][:q])+alpha*base.sym(spec[0][:,:q].T@d@spec[2][:,:q])
                            self.assertAlmostEqual(float(torch.linalg.eigvalsh(top)[-1]),1.,places=10)
                            self.assertLessEqual(float((m*(d-ref)).sum()),1e-10)
                            self.assertLessEqual(float(d.norm()),budget*float(ref.norm())+1e-10)
                            obstruction=(m*(d-ref)).sum()+.5*m.norm()*(d.square().sum()-ref.square().sum())
                            self.assertGreaterEqual(float(obstruction),-1e-10)
                            if method.startswith("K-LMO"):
                                torch.testing.assert_close(base.sym(spec[0][:,:q].T@(d-ref)@spec[2][:,:q]),
                                                           torch.zeros(q,q),atol=1e-10,rtol=0)
                            b=study.coupling_bound(spec,d,q,1.,alpha)
                            actual=float(torch.linalg.matrix_norm(x+alpha*d,ord=2))-1.
                            self.assertLessEqual(actual,b['overshoot_bound']+1e-10)
                            if b['old_overshoot_bound'] is not None:
                                self.assertLessEqual(b['overshoot_bound'],b['old_overshoot_bound']+1e-10)

    def test_coupling_matches_explicit_dilation(self):
        x,_,_,_=base.make_case(8,(8,6)); u,s,v=base.spectrum(x)
        d=torch.randn_like(x); q=3
        w=torch.cat((u[:,:q],v[:,:q]))/2**.5
        dilation=torch.zeros(14,14); dilation[:8,8:]=d; dilation[8:,:8]=d.T
        cross=dilation@w-w@(w.T@dilation@w)
        got=study.coupling_bound((u,s,v),d,q,1.,.01)
        self.assertAlmostEqual(got['coupling'],float(torch.linalg.matrix_norm(cross,ord=2)),places=11)

    def test_unbounded_momentum_noise_lemma(self):
        # Gradient bound G=2; samples can be arbitrarily larger than G.
        spec=base.spectrum(torch.diag(torch.tensor([1.,.9999,.5,.2])))
        gen=torch.Generator().manual_seed(9000)
        for scale in (0.,1e-10,.1,1.,10.,1e6):
            for _ in range(20):
                g=torch.randn(4,4,generator=gen); g=g/g.norm()*2
                m=g+scale*torch.randn(4,4,generator=gen)
                d,_=study.direction(m,spec,1.,.001,2,"K-LMO-1")
                r=base.tangent(g,spec[0][:,:2],spec[2][:,:2]).norm()
                rhs=-r*r/(8*2)+(4+.25)*(m-g).norm()
                self.assertLessEqual(float((g*d).sum()),float(rhs)+1e-9)

    def test_q1_zero_and_scaling(self):
        x,_,_,_=base.make_case(8,(8,6)); spec=base.spectrum(x); m=torch.randn_like(x)
        for method in study.METHODS:
            d,_=study.direction(m,spec,1.,.001,1,method)
            ref,_=base.direction(m,spec,1.,.001,1,"MCSD-TP")
            torch.testing.assert_close(d,ref,rtol=0,atol=0)
            z,_=study.direction(torch.zeros_like(m),spec,1.,.001,4,method)
            self.assertEqual(float(z.norm()),0.)
            d,_=study.direction(m,spec,1.,.001,4,method)
            for radius in (.1,10.):
                ds,_=study.direction(m*radius,(spec[0],spec[1]*radius,spec[2]),radius,.001*radius,4,method)
                torch.testing.assert_close(d,ds,atol=1e-9,rtol=1e-9)

    def test_profile_control_and_repeatability(self):
        x,a,w,meta=base.make_case(8,(4,4)); noise=torch.ones(8,4,4)/4
        ref=study.run(x,a,w,meta,noise,"A2","practical_nesterov",8)
        rows=[study.run(x,a,w,meta,noise,study.PROFILE_CONTROL,"practical_nesterov",8,ref['_norms']) for _ in range(2)]
        self.assertEqual(rows[0]['trace'],rows[1]['trace'])
        self.assertLess(rows[0]['maximum']['profile_length_error'],1e-12)


if __name__ == '__main__':
    unittest.main()
