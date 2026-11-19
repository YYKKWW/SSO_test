import os
import unittest
from unittest.mock import patch
import torch
from kernel_lmo_h20 import PilotAdapter, candidate, host, sym, ARMS, pilot_dispatch

DEVICE=os.environ.get("P2GD_TEST_DEVICE","cpu")


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self,plateau=False):
        gen=torch.Generator(device=DEVICE).manual_seed(33)
        u=torch.linalg.qr(torch.randn(20,16,generator=gen,device=DEVICE))[0]
        v=torch.linalg.qr(torch.randn(16,16,generator=gen,device=DEVICE))[0]
        s=torch.linspace(.8,.2,16,device=DEVICE);s[:4]=torch.tensor([1.,.99999,.99998,.99997],device=DEVICE)
        if plateau: s[:]=1.
        return (u*s)@v.T,torch.randn(20,16,generator=gen,device=DEVICE)

    def test_bootstrap_full_capacity_and_one_trial_solve(self):
        for method in ("control","a2","a2_1p5","kernel1"):
            x,m=self.sample(True);adapter=PilotAdapter(method,"exact")
            _,info=adapter(x,m,1.,10,8,effective_lr=0.)
            self.assertEqual(len(info.next_spectral_cache.s_basis),16)
            with patch.object(host,"accurate_svd_fp32",wraps=host.accurate_svd_fp32) as solve:
                update,info=adapter(x,m,1.,10,8,effective_lr=.001,current_spectral_cache=info.next_spectral_cache)
                self.assertEqual(solve.call_count,1)
            returned=x-.001*update
            self.assertLess(abs(float(torch.linalg.svdvals(returned)[0])-1),5e-6)
            if method!="control": self.assertEqual(info.cluster_dimension,16)
            self.assertEqual(adapter.counts["updates"],1)

    def test_model_and_kernel_invariant(self):
        x,m=self.sample(); adapter=PilotAdapter("kernel1","exact")
        _,info=adapter(x,m,1.,10,8,effective_lr=0.)
        cache=info.next_spectral_cache
        f,_=candidate(m,cache,1.,.001,4,"a2")
        d,_=candidate(m,cache,1.,.001,4,"kernel1")
        u,v=cache.u_basis[:,:4],cache.v_basis[:,:4]
        self.assertLess(float(sym(u.T@(d-f)@v).abs().max()),2e-5)
        self.assertLessEqual(float((m*(d-f)).sum()),1e-5)
        self.assertLessEqual(float(d.norm()),2*float(f.norm())+1e-5)
        self.assertLess(abs(float(torch.linalg.eigvalsh(torch.diag(cache.s_basis[:4])+.001*sym(u.T@d@v))[-1])-1),3e-6)

    def test_warm_is_uncertified_and_does_not_call_full_svd(self):
        x,m=self.sample(True);adapter=PilotAdapter("kernel1","warm_block8")
        with patch.object(host,"accurate_svd_fp32",side_effect=AssertionError("full SVD")):
            _,info=adapter(x,m,1.,10,8,effective_lr=0.)
            _,info=adapter(x,m,1.,10,8,effective_lr=.001,current_spectral_cache=info.next_spectral_cache)
        self.assertFalse(info.next_spectral_cache.is_exact)
        self.assertTrue(info.cluster_cap_overflow)
        self.assertEqual(adapter.counts["small_ritz_svd"],2)
        self.assertEqual(adapter.counts["full_svd"],0)

    def test_safe_path_shared_and_host_dispatch(self):
        x,m=self.sample();adapter=PilotAdapter("control","exact")
        _,info=adapter(x,m,1.,10,8,effective_lr=0.)
        ds=[candidate(m,info.next_spectral_cache,1.,.001,1,method)[0]
            for method in ("control","a2","a2_1p5","kernel1")]
        for d in ds[1:]: torch.testing.assert_close(d,ds[0],rtol=0,atol=0)
        from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as module
        p=torch.nn.Parameter(x.clone())
        opt=host.SpELPGDSameProjection([p],lr=.001,radial_cluster_mode="adaptive4",
            gap_threshold_rel=2e-4,cluster_threshold_rel=2e-4,pgd_lr_scale=1.,
            pgd_direction_normalization="none",projection_mode="tp_radial_pgd_exact",
            post_return_spectral_cache=True,tangent_project_after_msign=True)
        with torch.no_grad(),patch.object(module,"compute_cluster_radial_update",adapter):
            opt._compute_component_update(p,m,None,None,current_lr=0.,param_name="test")
            update=opt._compute_component_update(p,m,None,None,current_lr=.001,param_name="test")
            p.add_(update,alpha=-.001)
        radius=host.compute_target_radius(shape=p.shape,radius_mode=opt.radius_mode)
        self.assertLess(abs(float(torch.linalg.svdvals(p.detach())[0])-radius),5e-6)

    def test_grid_is_paired_and_bounded(self):
        self.assertEqual(len(ARMS),8)
        self.assertEqual(len(set(ARMS)),8)
        self.assertEqual(sum(s=="exact" for _,s,_ in ARMS),4)

    def test_rerun_selects_only_interrupted_arms(self):
        import argparse
        from prepare_kernel_lmo_h20 import parse_arms
        self.assertEqual(parse_arms("3,5"), (3,5))
        for invalid in ("", "3,3", "5,3", "-1", "8", "3-5"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_arms(invalid)

    def test_host_records_full_window_above_rank_four(self):
        from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as module
        previous_update = module.compute_cluster_radial_update
        previous_record = host.SpELPGDSameProjection._record_update_info
        for solver, rank in (("exact", 16), ("warm_block8", 8)):
            x, m = self.sample(True)
            p = torch.nn.Parameter(x.clone())
            adapter = PilotAdapter("kernel1", solver)
            opt = host.SpELPGDSameProjection([p], lr=.001, radial_cluster_mode="adaptive4",
                gap_threshold_rel=2e-4, cluster_threshold_rel=2e-4, pgd_lr_scale=1.,
                pgd_direction_normalization="none", projection_mode="tp_radial_pgd_exact",
                post_return_spectral_cache=True, tangent_project_after_msign=True)
            with torch.no_grad(), pilot_dispatch(adapter):
                opt._compute_component_update(p, m, None, None, current_lr=0., param_name="plateau")
                opt._compute_component_update(p, m, None, None, current_lr=.001, param_name="plateau")
            self.assertEqual(opt.cluster_rank_counts[rank], 1)
            self.assertEqual(sum(opt.cluster_rank_counts.values()), 1)
            self.assertEqual(adapter.q_hist[str(rank)], 1)
            self.assertIs(module.compute_cluster_radial_update, previous_update)
            self.assertIs(host.SpELPGDSameProjection._record_update_info, previous_record)

    def test_pilot_dispatch_restores_on_failure(self):
        from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as module
        previous_update = module.compute_cluster_radial_update
        previous_record = host.SpELPGDSameProjection._record_update_info
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with pilot_dispatch(PilotAdapter("kernel1", "exact")):
                raise RuntimeError("test failure")
        self.assertIs(module.compute_cluster_radial_update, previous_update)
        self.assertIs(host.SpELPGDSameProjection._record_update_info, previous_record)

    def test_inherited_medium_precision_is_overridden_and_restored(self):
        import kernel_lmo_h20 as module
        from emerging_optimizers import utils
        x,m=self.sample();adapter=PilotAdapter("a2","exact")
        original=module.candidate
        def checked(*args,**kwargs):
            self.assertEqual(torch.get_float32_matmul_precision(),"highest")
            return original(*args,**kwargs)
        with utils.fp32_matmul_precision("medium"),patch.object(module,"candidate",checked):
            _,info=adapter(x,m,1.,10,8,effective_lr=0.)
            adapter(x,m,1.,10,8,effective_lr=.001,current_spectral_cache=info.next_spectral_cache)
            self.assertEqual(torch.get_float32_matmul_precision(),"medium")


if __name__=="__main__": unittest.main()
