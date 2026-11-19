import os
import unittest
from unittest.mock import patch

import torch
import direct_top_repair as module
from direct_top_repair import MODES, DirectTopAdapter, repair, shared_directions
from kernel_lmo_h20 import PilotAdapter, host, pilot_dispatch
from local_step_aware_hybrid import sym, tangent

DEVICE=os.environ.get('P2GD_TEST_DEVICE','cpu')


class DirectRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def sample(self):
        g=torch.Generator(device=DEVICE).manual_seed(813)
        u=torch.linalg.qr(torch.randn(12,8,device=DEVICE,dtype=torch.float64,generator=g))[0]
        v=torch.linalg.qr(torch.randn(8,8,device=DEVICE,dtype=torch.float64,generator=g))[0]
        s=torch.linspace(1.,.2,8,device=DEVICE,dtype=torch.float64)
        s[:5]=torch.tensor([1.,.99999,.99998,.99997,.99996],device=DEVICE,dtype=s.dtype)
        m=torch.randn(12,8,device=DEVICE,dtype=s.dtype,generator=g)
        return u,s,v,m

    def test_same_d0_once_and_no_input_mutation(self):
        u,s,v,m=self.sample()
        with patch.object(module,'mcsd_direction',wraps=module.mcsd_direction) as oracle:
            d0,ds=shared_directions(m,(u,s,v),1.,.015,5)
            self.assertEqual(oracle.call_count,1)
        original=d0.clone()
        self.assertIs(ds['none'],d0)
        for mode in MODES:repair(d0,u[:,:5],s[:5],v[:,:5],1.,.015,mode)
        torch.testing.assert_close(d0,original,atol=0,rtol=0)

    def test_low_rank_repair_preserves_free_block(self):
        u,s,v,d0=self.sample();q=5
        for mode in MODES[1:]:
            d=repair(d0,u[:,:q],s[:q],v[:,:q],1.,.015,mode)
            delta=d-d0
            self.assertLess(float(tangent(delta,u[:,:q],v[:,:q]).norm()),1e-11)
            self.assertLessEqual(int(torch.linalg.matrix_rank(delta,atol=1e-10)),q)
            self.assertLess(abs(float(torch.linalg.eigvalsh(sym(u[:,:q].T@d@v[:,:q]))[-1])) if mode=='multiple'
                else abs(float(torch.linalg.eigvalsh(torch.diag(s[:q])+.015*sym(u[:,:q].T@d@v[:,:q]))[-1])-1),1e-11)

    def test_exact_plateau_repairs_coincide(self):
        u,s,v,d0=self.sample();s=torch.ones_like(s)
        torch.testing.assert_close(repair(d0,u,s,v,1.,.015,'multiple'),
                                   repair(d0,u,s,v,1.,.015,'finite_step'),atol=0,rtol=0)

    def test_finite_step_retains_slack_and_cannot_be_post_scaled(self):
        eye=torch.eye(2,device=DEVICE,dtype=torch.float64)
        s=torch.tensor([1.,.9999],device=DEVICE,dtype=eye.dtype)
        d0=torch.diag(torch.tensor([0.,1.],device=DEVICE,dtype=eye.dtype))
        multiple=repair(d0,eye,s,eye,1.,.01,'multiple')
        finite=repair(d0,eye,s,eye,1.,.01,'finite_step')
        self.assertLess(float(multiple.norm()),1e-12)
        self.assertAlmostEqual(float(finite[1,1]),.01,places=10)
        self.assertGreater(float(torch.linalg.eigvalsh(torch.diag(s)+.01*2*finite)[-1])-1,1e-5)

    def test_q1_is_identical_for_all_modes(self):
        u,s,v,d0=self.sample()
        for mode in MODES:
            self.assertIs(repair(d0,u[:,:1],s[:1],v[:,:1],1.,.015,mode),d0)

    def test_none_matches_existing_control(self):
        u,s,v,m=self.sample();x=((u*s)@v.T).float();m=m.float()
        old=PilotAdapter('control','exact');new=DirectTopAdapter('none')
        a,b=x.clone(),x.clone()
        _,ia=old(a,m,1.,10,8,effective_lr=0.)
        _,ib=new(b,m,1.,10,8,effective_lr=0.)
        for _ in range(3):
            da,ia=old(a,m,1.,10,8,effective_lr=.015,current_spectral_cache=ia.next_spectral_cache)
            db,ib=new(b,m,1.,10,8,effective_lr=.015,current_spectral_cache=ib.next_spectral_cache)
            torch.testing.assert_close(da,db,atol=1e-6,rtol=1e-6)
            a=a-.015*da;b=b-.015*db

    def test_host_q5_and_one_trial_svd(self):
        u,s,v,m=self.sample();x=((u*s)@v.T).float();m=m.float()
        for mode in MODES:
            adapter=DirectTopAdapter(mode)
            p=torch.nn.Parameter(x.clone())
            opt=host.SpELPGDSameProjection([p],lr=.001,radial_cluster_mode='adaptive4',
                gap_threshold_rel=2e-4,cluster_threshold_rel=2e-4,pgd_lr_scale=1.,
                pgd_direction_normalization='none',projection_mode='tp_radial_pgd_exact',
                post_return_spectral_cache=True,tangent_project_after_msign=True)
            with torch.no_grad(),pilot_dispatch(adapter):
                opt._compute_component_update(p,m,None,None,current_lr=0.,param_name='q5')
                with patch.object(host,'accurate_svd_fp32',wraps=host.accurate_svd_fp32) as solve:
                    update=opt._compute_component_update(p,m,None,None,current_lr=.001,param_name='q5')
                    self.assertEqual(solve.call_count,1)
                p.add_(update,alpha=-.001)
            self.assertEqual(opt.cluster_rank_counts[5],1)
            self.assertEqual(adapter.counts['repair_calls'],int(mode!='none'))
            self.assertIn('d0',adapter.last)
            radius=host.compute_target_radius(shape=p.shape,radius_mode=opt.radius_mode)
            self.assertLess(abs(float(torch.linalg.svdvals(p.detach())[0])/radius-1),5e-6)

    def test_zero_momentum_and_invalid_arguments(self):
        u,s,v,m=self.sample()
        d0,ds=shared_directions(torch.zeros_like(m),(u,s,v),1.,.015,5)
        for d in ds.values():self.assertLess(float(d.norm()),1e-11)
        with self.assertRaises(ValueError):repair(d0,u,s,v,1.,0.,'finite_step')
        with self.assertRaises(ValueError):DirectTopAdapter('finite_step','warm_block8')

    def test_pilot_arms_only_change_repair(self):
        from run_kernel_lmo_pilot import arm_spec
        for i,mode in enumerate(MODES):
            self.assertEqual(arm_spec(i,'direct_top'),(mode,'exact','spel_pgd_dist'))
        with self.assertRaises(ValueError):arm_spec(3,'direct_top')


if __name__=='__main__':unittest.main()
