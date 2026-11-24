import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from matched_geometry import return_point
from p2gd_late_geometry import ARMS, candidates, forward_match
from p2gd_late_replay import LateReplay
from p2gd_six_replay import SixReplay
from test_p2gd_elr_replay import ReplayTests


class LateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(20261123)
        self.device=os.environ.get('ELR_TEST_DEVICE','cpu')

    def test_forward_targets_and_correction_audit(self):
        n=8
        x=torch.diag(torch.tensor([1.,.9999,.6,.5,.4,.3,.2,.1],device=self.device))
        m=torch.randn_like(x);d0=torch.randn_like(x);d0[0,0]=0
        u=torch.eye(n,device=self.device)[:,:2];s=torch.tensor([1.,.9999],device=self.device)
        h=torch.randn(64,n,device=self.device)
        before=x.bfloat16().float();baseline=return_point(x+.0075*d0,1.,'radial')
        points,records,_=candidates(x,m,d0,u,s,u,1.,.015,h,baseline,before,torch.bfloat16)
        self.assertEqual([r['method'] for r in records],[a['name'] for a in ARMS])
        by={r['method']:r for r in records}
        for r in records[1:]:
            ref=points[r['target_reference']].bfloat16().float()-before
            delta=points[r['method']].bfloat16().float()-before
            if r['metric']=='spectral':
                measured=float(torch.linalg.matrix_norm(delta,ord=2)/torch.linalg.matrix_norm(before,ord=2))
                target=float(torch.linalg.matrix_norm(ref,ord=2)/torch.linalg.matrix_norm(before,ord=2))
            else:
                measured=float((h@delta.T).norm()/(h@before.T).norm())
                target=float((h@ref.T).norm()/(h@before.T).norm())
            self.assertAlmostEqual(r['measured'],measured,places=5)
            self.assertAlmostEqual(r['target'],target,places=5)
            self.assertEqual(r['matching_satisfied'],abs(r['measured']-r['target'])<=r['tolerance'])
            self.assertLess(r['radius_error'],5e-5)
        self.assertEqual(by['ambient']['correction_rank'],0)
        self.assertLessEqual(by['top1']['correction_rank'],1)
        self.assertLessEqual(by['cluster_cone']['correction_rank'],2)
        self.assertEqual(by['ambient']['bf16_changed_fraction_vs_ambient'],0.)

    def test_unattainable_quantized_target_is_reported(self):
        x=torch.diag(torch.tensor([1.,.9999],device=self.device))
        h=torch.eye(2,device=self.device);d=torch.tensor([[0.,1.],[1.,0.]],device=self.device)
        _,_,r=forward_match(x,lambda a:d,1.,.01,.0031415926535,'output',h,
            x.bfloat16().float(),torch.bfloat16,rtol=0.,atol=0.,iterations=1)
        self.assertFalse(r['matching_satisfied'])
        self.assertFalse(r['matching_enforced'])
        self.assertEqual(r['matching_status'],'best_observed_unmatched')
        self.assertGreater(r['relative_error'],0.)

    def test_audit_window(self):
        owner=LateReplay.__new__(LateReplay);owner.measure_first=513
        for t,expected in ((253,True),(268,True),(269,False),(512,False),(513,True),(528,True)):
            owner.iteration=t;self.assertEqual(owner.auditing(),expected)

    def test_fast_continuation_matches_audited_updates_and_momentum(self):
        mb=importlib.import_module('emerging_optimizers.orthogonalized_optimizers.muon_ball')
        original=mb.MuonBall.step,mb.MuonBall.orthogonalize,mb.compute_muon_ball_update
        paths=[]
        for fast in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                try:
                    torch.manual_seed(71)
                    p=torch.nn.Parameter(return_point(torch.randn(16,8,device=self.device),2**.5,'radial'))
                    p.main_param=p
                    owner=(LateReplay if fast else SixReplay)(folder)
                    owner.install()
                    opt=mb.MuonBall([p],lr=.015,scale_mode='spectral_mup',msign_steps=8,weight_decay=.1)
                    opt.state[p]['momentum_buffer']=torch.randn_like(p)
                    h=torch.randn(32,8,device=self.device)
                    path=[]
                    for k in range(3):
                        owner.iteration=(269 if fast else 253)+k
                        p.grad=torch.randn_like(p)
                        owner.inputs={'decoder.layers.0.mlp.linear_fc2':dict(
                            module=SimpleNamespace(weight=p),calibration=h,heldout=h)}
                        opt.step();owner.finish_update()
                        path.append((p.detach().clone(),opt.state[p]['momentum_buffer'].clone()))
                    paths.append(path)
                finally:
                    mb.MuonBall.step,mb.MuonBall.orthogonalize,mb.compute_muon_ball_update=original
        for a,b in zip(*paths):
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_restore_four_controls(self):
        ReplayTests().check_restore(methods=tuple(a['name'] for a in ARMS))

    def test_restore_when_candidate_evaluation_fails(self):
        ReplayTests().check_restore(failure=True,methods=tuple(a['name'] for a in ARMS))


if __name__=='__main__':
    unittest.main()
