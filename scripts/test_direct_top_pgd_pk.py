import os
import unittest
from unittest.mock import patch

import torch

from direct_top_pgd_pk import ARMS, DirectPKAdapter, matched_pgd_direction, sphere_from_svd
from direct_top_repair import DirectTopAdapter
from kernel_lmo_h20 import host, pilot_dispatch

DEVICE=os.environ.get('P2GD_TEST_DEVICE','cpu')


class DirectPKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def matrix(self,near=True):
        s=torch.linspace(1.,.1,12,device=DEVICE)
        s[:5]=torch.tensor([1.,.99999,.99998,.99997,.99996] if near else [1.,.8,.7,.6,.5],device=DEVICE)
        x=torch.diag(s)
        g=torch.Generator(device=DEVICE).manual_seed(834)
        m=torch.randn(x.shape,device=DEVICE,generator=g)
        return x,m

    def test_momentum_direction_and_length(self):
        x,m=self.matrix()
        for ratio in (.5,1.):
            d=matched_pgd_direction(m,x,ratio)
            torch.testing.assert_close(d.norm(),ratio*x.norm())
            self.assertLess(float((m*d).sum()),0)
            torch.testing.assert_close(d/ d.norm(),-m/m.norm())
        self.assertEqual(float(matched_pgd_direction(torch.zeros_like(m),x,1.).norm()),0)
        self.assertGreater(float(matched_pgd_direction(m,torch.zeros_like(x),1.).norm()),0)

    def test_sphere_projection_inside_and_outside(self):
        u=torch.eye(12,device=DEVICE)
        for s in (torch.linspace(.8,.1,12,device=DEVICE),torch.linspace(2.,.2,12,device=DEVICE)):
            p,values=sphere_from_svd(u,s,u,1.)
            expected=s.clamp(max=1.);expected[0]=1.
            torch.testing.assert_close(p,torch.diag(expected))
            torch.testing.assert_close(values,expected)
        with self.assertRaises(ValueError):sphere_from_svd(u,s,u,0.)

    def test_finite_and_control_unchanged(self):
        x,m=self.matrix()
        for mode in ('none','finite_step'):
            old,new=DirectTopAdapter(mode),DirectPKAdapter(mode)
            a,b=x.clone(),x.clone()
            _,ia=old(a,m,1.,10,8,effective_lr=0.)
            _,ib=new(b,m,1.,10,8,effective_lr=0.)
            for _ in range(3):
                da,ia=old(a,m,1.,10,8,effective_lr=.015,current_spectral_cache=ia.next_spectral_cache)
                db,ib=new(b,m,1.,10,8,effective_lr=.015,current_spectral_cache=ib.next_spectral_cache)
                torch.testing.assert_close(da,db,atol=1e-6,rtol=1e-6)
                a=a-.015*da;b=b-.015*db

    def test_safe_pgd_uses_same_radial_path(self):
        x,m=self.matrix(False)
        for mode in ('pgd_half','pgd_full'):
            a=DirectPKAdapter(mode);b=DirectPKAdapter('none')
            xa,xb=x.clone(),x.clone()
            da,ia=a(xa,m,1.,10,8,effective_lr=.015)
            db,ib=b(xb,m,1.,10,8,effective_lr=.015)
            self.assertEqual(a.counts['pgd_calls'],0)
            torch.testing.assert_close(da,db)

    def test_actual_pgd_return_cache_and_counts(self):
        x,m=self.matrix()
        for mode,ratio in (('pgd_half',.5),('pgd_full',1.)):
            a=DirectPKAdapter(mode);w=x.clone()
            _,init=a(w,m,1.,10,8,effective_lr=0.)
            with patch.object(host,'accurate_svd_fp32',wraps=host.accurate_svd_fp32) as svd:
                update,info=a(w,m,1.,10,8,effective_lr=.015,current_spectral_cache=init.next_spectral_cache)
                self.assertEqual(svd.call_count,1)
            state=a.last
            torch.testing.assert_close(state['direction'].norm(),ratio*state['d0'].norm())
            u,s,vh=torch.linalg.svd(state['trial'],full_matrices=False)
            ref,values=sphere_from_svd(u,s,vh,1.)
            returned=w-.015*update
            torch.testing.assert_close(returned,ref,atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(info.next_spectral_cache.s_basis,values,atol=3e-6,rtol=3e-6)
            self.assertEqual(info.branch,'pgd')
            self.assertEqual(info.projection_mode,'full_svd')
            self.assertEqual(a.counts['repair_calls'],0)
            self.assertEqual(a.counts['pgd_calls'],1)
            self.assertLess(abs(float(torch.linalg.svdvals(returned)[0])-1),5e-6)

    def test_full_projection_clips_beyond_eight(self):
        a=DirectPKAdapter('pgd_full');a._is_pgd=True
        u=torch.eye(12,device=DEVICE);s=torch.full((12,),2.,device=DEVICE)
        p,values=a.trial_return(2*u,u,s,u,1.)
        torch.testing.assert_close(p,u)
        self.assertEqual(a._projection_clipped,12)

    def test_radial_scales_all_singular_values_without_clipping(self):
        a=DirectPKAdapter('pgd_full',ambient_return='radial')
        u=torch.eye(3,device=DEVICE)
        s=torch.tensor([4.,3.,1.],device=DEVICE)
        y=torch.diag(s)
        returned,values=a.trial_return(y,u,s,u,2.)
        torch.testing.assert_close(returned,y/2)
        torch.testing.assert_close(values,torch.tensor([2.,1.5,.5],device=DEVICE))
        projected,_=sphere_from_svd(u,s,u,2.)
        self.assertFalse(torch.allclose(returned,projected))
        self.assertEqual(a._projection_clipped,0)

    def test_radial_same_trial_and_post_return_cache(self):
        x,m=self.matrix()
        radius=2.5
        for mode,ratio in (('pgd_half',.5),('pgd_full',1.)):
            projected=DirectPKAdapter(mode)
            radial=DirectPKAdapter(mode,ambient_return='radial')
            wp,wr=x.clone()*radius,x.clone()*radius
            _,ip=projected(wp,m,radius,10,8,effective_lr=0.)
            _,ir=radial(wr,m,radius,10,8,effective_lr=0.)
            projected(wp,m,radius,10,8,effective_lr=.015,current_spectral_cache=ip.next_spectral_cache)
            with patch.object(host,'accurate_svd_fp32',wraps=host.accurate_svd_fp32) as svd:
                update,info=radial(wr,m,radius,10,8,effective_lr=.015,current_spectral_cache=ir.next_spectral_cache)
                self.assertEqual(svd.call_count,1)
            trial=radial.last['trial']
            torch.testing.assert_close(trial,projected.last['trial'])
            torch.testing.assert_close(radial.last['direction'].norm(),ratio*radial.last['d0'].norm())
            s=torch.linalg.svdvals(trial)
            returned=wr-.015*update
            torch.testing.assert_close(returned,trial*(radius/s[0]),atol=4e-6,rtol=4e-6)
            torch.testing.assert_close(info.next_spectral_cache.s_basis,s*(radius/s[0]),atol=4e-6,rtol=4e-6)
            torch.testing.assert_close(torch.linalg.svdvals(returned),info.next_spectral_cache.s_basis,atol=4e-6,rtol=4e-6)
            self.assertAlmostEqual(info.next_spectral_cache.rel_gap,float(1-s[1]/s[0]),places=5)
            self.assertEqual(info.branch,'spel')
            self.assertEqual(radial.last['record']['return_kind'],'radial')
            self.assertEqual(radial.counts['pgd_calls'],0)
            self.assertEqual(radial.counts['ambient_radial_calls'],1)
            self.assertEqual(radial.counts['projection_clipped_total'],0)
            report=radial.report()
            self.assertEqual(report['method'],mode.replace('pgd_','ambient_radial_'))
            self.assertEqual(report['return_rule'],'radial_both_branches')

    def test_radial_safe_region_matches_control(self):
        x,m=self.matrix(False)
        radial=DirectPKAdapter('pgd_full',ambient_return='radial')
        control=DirectPKAdapter('none')
        dr,_=radial(x.clone(),m,1.,10,8,effective_lr=.015)
        dc,_=control(x.clone(),m,1.,10,8,effective_lr=.015)
        torch.testing.assert_close(dr,dc)
        self.assertEqual(radial.counts['ambient_radial_calls'],0)

    def test_radial_replay_uses_requested_return(self):
        from audit_direct_pk import replay
        x,m=self.matrix()
        rows=replay(x.double(),m.double(),1.,.015,ambient_return='radial')
        self.assertEqual(len(rows),4)
        self.assertEqual(sum(r['ambient_near'] for r in rows),2)
        for row in rows:
            self.assertFalse(row['pgd'])
            self.assertEqual(row['return_kind'],'radial')
            self.assertEqual(row['clipped_count'],0)
            self.assertLess(row['radius_error'],1e-10)
        with self.assertRaises(ValueError):
            DirectPKAdapter('pgd_full',ambient_return='unknown')

    def test_host_integration(self):
        x,m=self.matrix()
        for return_kind in ('sphere_projection','radial'):
            adapter=DirectPKAdapter('pgd_full',ambient_return=return_kind)
            p=torch.nn.Parameter(x.clone())
            opt=host.SpELPGDSameProjection([p],lr=.001,radial_cluster_mode='adaptive4',
                gap_threshold_rel=2e-4,cluster_threshold_rel=2e-4,pgd_lr_scale=1.,
                pgd_direction_normalization='none',projection_mode='tp_radial_pgd_exact',
                post_return_spectral_cache=True,tangent_project_after_msign=True)
            with torch.no_grad(),pilot_dispatch(adapter):
                opt._compute_component_update(p,m,None,None,current_lr=0.,param_name='test')
                update=opt._compute_component_update(p,m,None,None,current_lr=.001,param_name='test')
                p.add_(update,alpha=-.001)
            self.assertEqual(adapter.counts['pgd_calls'],int(return_kind=='sphere_projection'))
            self.assertEqual(adapter.counts['ambient_radial_calls'],int(return_kind=='radial'))
            self.assertEqual(opt.cluster_rank_counts[5],1)
            radius=host.compute_target_radius(shape=p.shape,radius_mode=opt.radius_mode)
            self.assertLess(abs(float(torch.linalg.svdvals(p.detach())[0])/radius-1),5e-6)

    def test_six_arm_mapping_and_validation(self):
        from run_kernel_lmo_pilot import arm_spec
        for i,arm in enumerate(ARMS):self.assertEqual(arm_spec(i,'direct_pk'),arm)
        with self.assertRaises(ValueError):arm_spec(6,'direct_pk')
        with self.assertRaises(ValueError):DirectPKAdapter('pgd_full','warm_block8')


if __name__=='__main__':unittest.main()
