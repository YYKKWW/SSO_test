import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from matched_geometry import return_point
from p2gd_late_geometry import candidates
from p2gd_trajectory import Trajectory, optimizer_signature
from p2gd_trajectory_geometry import POLICIES, selected_return


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2201)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        self.device = os.environ.get('ELR_TEST_DEVICE', 'cpu')

    def test_selected_policies_equal_replay_candidates(self):
        x = torch.diag(torch.tensor([1., .9999, .5, .3, .2, .1], device=self.device))
        m = torch.randn_like(x)
        d0 = torch.randn_like(x); d0[0,0]=0
        u = torch.eye(6, device=self.device)[:,:2]
        s = torch.tensor([1., .9999], device=self.device)
        h = torch.randn(32,6,device=self.device)
        before = x.bfloat16().float()
        baseline = return_point(x + .015*.5*d0, 1., 'radial')
        points, _, _ = candidates(x,m,d0,u,s,u,1.,.015,h,baseline,before,torch.bfloat16)
        for policy in POLICIES:
            got, info = selected_return(policy,x,m,d0,u,s,u,1.,.015,h,before)
            torch.testing.assert_close(got,points[policy],rtol=0,atol=0)
            self.assertLess(info['radius_error'],5e-5)

    def test_zero_direction_does_not_create_a_step(self):
        x = torch.eye(4,device=self.device)
        for policy in POLICIES:
            point, info = selected_return(policy,x,torch.ones_like(x),torch.zeros_like(x),x,
                torch.ones(4,device=self.device),x,1.,.01,x,x)
            torch.testing.assert_close(point,x,rtol=0,atol=0)
            self.assertTrue(info['zero_budget'])

    def test_actual_optimizer_commits_each_selected_policy(self):
        mb = importlib.import_module('emerging_optimizers.orthogonalized_optimizers.muon_ball')
        original = mb.MuonBall.step, mb.MuonBall.orthogonalize, mb.compute_muon_ball_update
        endpoints = []
        for policy in POLICIES:
            with tempfile.TemporaryDirectory() as folder:
                try:
                    torch.manual_seed(73)
                    x = torch.diag(torch.tensor([1.,.9999,.8,.7,.6,.5,.4,.3],device=self.device))
                    p = torch.nn.Parameter(x)
                    forward = torch.nn.Parameter(x.bfloat16())
                    forward.main_param = p
                    module = SimpleNamespace(weight=forward)
                    owner = Trajectory(folder,policy)
                    owner.install()
                    opt = mb.MuonBall([p],lr=.015,scale_mode='spectral_mup',msign_steps=8,weight_decay=.1)
                    opt.state[p]['momentum_buffer']=torch.randn_like(p)
                    h = torch.randn(32,8,device=self.device)
                    for k in range(3):
                        owner.iteration = 513+k
                        owner.near_threshold = 1.
                        p.grad = torch.randn_like(p)
                        owner.inputs = {'decoder.layers.0.mlp.linear_fc2':dict(module=module,calibration=h)}
                        opt.step()
                        forward.data.copy_(p)
                        owner.finish_update()
                    self.assertEqual(opt.param_groups[0]['wd_mult'],0)
                    self.assertTrue((Path(folder)/'trajectory_update.jsonl').exists())
                    self.assertEqual(len(optimizer_signature(SimpleNamespace(optimizer=opt))),1)
                    endpoints.append(p.detach().clone())
                finally:
                    mb.MuonBall.step,mb.MuonBall.orthogonalize,mb.compute_muon_ball_update=original
        self.assertTrue(any(not torch.equal(endpoints[0],p) for p in endpoints[1:]))

    def test_wrong_forward_state_fails(self):
        x=torch.eye(4,device=self.device)
        with self.assertRaises(ValueError):
            selected_return('ambient',x,torch.ones_like(x),torch.ones_like(x),x,
                torch.ones(4,device=self.device),x,1.,.01,x,x*2)


if __name__ == '__main__':
    unittest.main()
