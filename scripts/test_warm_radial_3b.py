"""Matched warm-only spectral substitution, without scheduling real training."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from direct_top_pgd_pk import DirectPKAdapter
from kernel_lmo_h20 import host, pilot_dispatch
from run_kernel_lmo_pilot import main

DEVICE=os.environ.get("P2GD_TEST_DEVICE", "cpu")


class WarmRadialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self, near=True):
        g=torch.Generator(device=DEVICE).manual_seed(947)
        u=torch.linalg.qr(torch.randn(20,16,generator=g,device=DEVICE))[0]
        v=torch.linalg.qr(torch.randn(16,16,generator=g,device=DEVICE))[0]
        s=torch.linspace(.7,.1,16,device=DEVICE)
        s[:2]=torch.tensor([1.,1. if near else .75],device=DEVICE)
        return (u*s)@v.T,torch.randn(20,16,generator=g,device=DEVICE)

    def test_only_small_fp32_ritz_no_full_svd_or_svdvals(self):
        import kernel_lmo_h20 as module
        x,m=self.sample()
        adapter=DirectPKAdapter("pgd_half","warm_block8",ambient_return="radial")
        original=torch.linalg.svd
        calls=[]
        def small_only(a,*args,**kwargs):
            self.assertEqual(tuple(a.shape),(8,8))
            self.assertEqual(a.dtype,torch.float32)
            calls.append(a.shape)
            return original(a,*args,**kwargs)
        with patch.object(host,"accurate_svd_fp32",side_effect=AssertionError("full SVD")), \
             patch("torch.linalg.svd",side_effect=small_only), \
             patch("torch.linalg.svdvals",side_effect=AssertionError("online svdvals")), \
             patch.object(module,"warm_block8_ritz",wraps=module.warm_block8_ritz) as ritz:
            _,info=adapter(x,m,1.,10,8,effective_lr=0.)
            self.assertFalse(info.next_spectral_cache.is_exact)
            for _ in range(4):
                old_cache=info.next_spectral_cache
                update,info=adapter(x,m,1.,10,8,effective_lr=.015,current_spectral_cache=old_cache)
                self.assertTrue(info.spectral_cache_hit)
                state=adapter.last
                ratio=.5 if state["record"]["near"] else 1.
                torch.testing.assert_close(state["direction"].norm(),ratio*state["d0"].norm())
                x=x-.015*update
                torch.testing.assert_close(x,state["returned"])
                self.assertEqual(ritz.call_args.kwargs["iterations"],10)
                self.assertIs(ritz.call_args.kwargs["initial_v"],old_cache.v_basis)
        self.assertEqual(len(calls),5)
        self.assertEqual(adapter.counts["bootstrap_solves"],1)
        self.assertEqual(adapter.counts["trial_solves"],4)
        self.assertEqual(adapter.counts["full_svd"],0)
        self.assertGreater(adapter.counts["ambient_radial_calls"],0)
        self.assertEqual(adapter.counts["pgd_calls"],0)

    def test_returned_cache_contains_scaled_ritz_pairs(self):
        from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import warm_block8_ritz
        x,m=self.sample()
        adapter=DirectPKAdapter("pgd_half","warm_block8",ambient_return="radial")
        _,initial=adapter(x,m,1.,10,8,effective_lr=0.)
        previous=initial.next_spectral_cache
        _,info=adapter(x,m,1.,10,8,effective_lr=.015,current_spectral_cache=previous)
        state=adapter.last
        u,s,vh=warm_block8_ritz(state["trial"],iterations=10,initial_v=previous.v_basis)
        cache=info.next_spectral_cache
        torch.testing.assert_close(state["returned"],state["trial"]/s[0])
        torch.testing.assert_close(cache.s_basis,s/s[0])
        torch.testing.assert_close(state["returned"]@cache.v_basis,cache.u_basis*cache.s_basis,
                                   atol=4e-6,rtol=4e-6)
        self.assertAlmostEqual(cache.rel_gap,float(1-s[1]/s[0]),places=5)
        self.assertFalse(cache.is_exact)

    def test_safe_direction_matches_warm_control(self):
        x,m=self.sample(False)
        hybrid=DirectPKAdapter("pgd_half","warm_block8",ambient_return="radial")
        control=DirectPKAdapter("none","warm_block8",ambient_return="radial")
        a,_=hybrid(x.clone(),m,1.,10,8,effective_lr=.015)
        b,_=control(x.clone(),m,1.,10,8,effective_lr=.015)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(hybrid.counts["ambient_radial_calls"],0)

    def test_host_keeps_approximate_cache(self):
        x,m=self.sample()
        p=torch.nn.Parameter(x.clone())
        adapter=DirectPKAdapter("pgd_half","warm_block8",ambient_return="radial")
        opt=host.SpELPGDSameProjection([p],lr=.001,radial_cluster_mode="adaptive4",
            cluster_spectrum_mode="warm_block8",cluster_iterations=10,
            gap_threshold_rel=2e-4,cluster_threshold_rel=2e-4,pgd_lr_scale=1.,
            pgd_direction_normalization="none",projection_mode="tp_radial_pgd_exact",
            post_return_spectral_cache=True,tangent_project_after_msign=True)
        with torch.no_grad(),pilot_dispatch(adapter), \
             patch.object(host,"accurate_svd_fp32",side_effect=AssertionError("full SVD")):
            opt._compute_component_update(p,m,None,None,current_lr=0.,param_name="test")
            for _ in range(3):
                update=opt._compute_component_update(p,m,None,None,current_lr=.001,param_name="test")
                p.add_(update,alpha=-.001)
        self.assertEqual(adapter.counts["bootstrap_solves"],1)
        self.assertEqual(adapter.counts["small_ritz_svd"],4)
        self.assertEqual(adapter.report()["approximate_iterations"],10)
        self.assertFalse(adapter.report()["actual_elr_matched"])

    def test_projection_and_finite_repair_cannot_silently_become_truncated(self):
        for kwargs in ({},{"ambient_return":"sphere_projection"}):
            with self.assertRaises(ValueError):
                DirectPKAdapter("pgd_half","warm_block8",**kwargs)
        with self.assertRaises(ValueError):
            DirectPKAdapter("finite_step","warm_block8",ambient_return="radial")

    def test_3b_launch_changes_only_solver_fields(self):
        with tempfile.TemporaryDirectory() as root:
            configs=[]
            for solver in (None,"warm_block8"):
                captured=[]
                def fake_train(env,folder,smoke):
                    folder.mkdir(parents=True,exist_ok=True)
                    captured.append((dict(env),smoke))
                env=dict(PROJECT_DIR=root,SLURM_ARRAY_JOB_ID="123",SLURM_ARRAY_TASK_ID="2")
                argv=["run_kernel_lmo_pilot.py","--study","direct_pk","--arm","2", "--budget","3b","--ambient-return","radial"]
                if solver:argv.extend(["--solver",solver])
                with patch.dict(os.environ,env),patch("sys.argv",argv), \
                     patch("run_kernel_lmo_pilot.run_training",side_effect=fake_train), \
                     patch("run_kernel_lmo_pilot.subprocess.run") as audit:
                    main()
                self.assertEqual([s for _,s in captured],[True,False])
                configs.append(captured[-1][0])
                self.assertEqual(audit.call_count,2 if solver else 1)
            old,new=configs
            changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
            self.assertEqual(changed,{"KERNEL_SOLVER","SPEL_PGD_CLUSTER_SPECTRUM_MODE","JOB_NAME"})
            self.assertEqual(new["TRAIN_ITER"],"5723")
            self.assertEqual(new["SEED"],"2345")
            self.assertEqual(new["WIDTH"],"256")
            self.assertEqual(new["KERNEL_PGD_TRIAL_RATIO"],"0.5")
            self.assertEqual(new["SPEL_PGD_GAP_THRESHOLD_REL"],"2e-4")
            self.assertEqual(new["SPEL_PGD_CLUSTER_ITERATIONS"],"10")
            self.assertEqual(new["SAVE_CHECKPOINT"],"0")

    def test_offline_audit_uses_actual_states_without_radius_correction(self):
        from audit_warm_radial import audit_snapshot
        x=torch.diag(torch.tensor([1.02,1.02*.9999,.5],device=DEVICE))
        d0=torch.eye(3,device=DEVICE)
        y=x+.01*d0
        returned=y/torch.linalg.svdvals(y)[0]*1.01
        data=dict(current=x,trial=y,returned=returned,d0=d0,radius=1.,alpha=.01,step=20,
                  record=dict(current_gap=4e-4,returned_gap=3e-4,near=False))
        row=audit_snapshot(data)
        self.assertAlmostEqual(row["current_radius_relative_error"],.02,places=6)
        self.assertAlmostEqual(row["returned_radius_relative_error"],.01,places=6)
        self.assertTrue(row["missed_near"])
        self.assertAlmostEqual(row["actual_elr_frobenius"],float((returned-x).norm()/x.norm()),places=6)


if __name__ == "__main__":
    unittest.main()
