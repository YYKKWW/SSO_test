import os
import unittest
from unittest.mock import patch

import torch

from p2gd_budget_geometry import norm2, radial
from p2gd_six_geometry import ARMS, candidates
from p2gd_six_replay import SixReplay
import test_p2gd_elr_replay


class SixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20261122)
        device=os.environ.get('ELR_TEST_DEVICE','cpu')
        dtype=torch.float32 if device=='cuda' else torch.float64
        self.x=torch.diag(torch.tensor([1.,.9999,.6,.2],device=device,dtype=dtype))
        self.m=torch.randn_like(self.x);self.d0=torch.randn_like(self.x);self.d0[0,0]=0
        self.u=torch.eye(4,device=device,dtype=dtype)[:,:2]
        self.s=torch.tensor([1.,.9999],device=device,dtype=dtype)
        self.h=torch.randn(19,4,device=device,dtype=dtype)

    def test_six_targets_after_return(self):
        before=[t.clone() for t in (self.x,self.m,self.d0,self.h)]
        points,records,targets=candidates(self.x,self.m,self.d0,self.u,self.s,self.u,1.,.015,self.h)
        self.assertEqual([r['method'] for r in records],[a['name'] for a in ARMS])
        torch.testing.assert_close(points['half_d0'],radial(self.x+.0075*self.d0,1.))
        by={r['method']:r for r in records}
        for row in records[1:]:
            target=by['cluster_cone']['local_output_elr'] if row['metric']=='output' else targets['spectral']
            value=row['local_output_elr'] if row['metric']=='output' else row['returned_spectral_elr']
            self.assertLessEqual(abs(value-target),row['tolerance'])
            self.assertLess(row['radius_error'],2e-6)
        for t,b in zip((self.x,self.m,self.d0,self.h),before):
            torch.testing.assert_close(t,b,rtol=0,atol=0)

    def test_finite_repair_rebuilt_and_cone_differs_from_top1(self):
        from p2gd_elr_geometry import repair
        points,records,_=candidates(self.x,self.m,self.d0,self.u,self.s,self.u,1.,.015,self.h)
        row=next(r for r in records if r['method']=='finite_momentum');a=row['effective_alpha']
        d=repair('finite_momentum',self.d0,self.m,self.u,self.s,self.u,1.,a)
        b=self.u.T@d@self.u
        self.assertLess(abs(float(torch.linalg.eigvalsh(torch.diag(self.s)+a*(b+b.T)/2)[-1])-1),2e-6)
        torch.testing.assert_close(points['finite_momentum'],radial(self.x+a*d,1.))
        self.assertGreater(float((points['top1']-points['cluster_cone']).norm()),1e-7)

    def test_prefix_uses_same_baseline_without_counterfactuals(self):
        owner=SixReplay.__new__(SixReplay)
        owner.first=253;owner.measure_first=269;owner.iteration=253;owner.near_threshold=2e-4
        with patch.object(SixReplay,'candidate_factory',side_effect=AssertionError('prefix candidate')):
            point,_=owner.propose_update(self.x,self.d0,1.,.015,None,None)
        torch.testing.assert_close(point,radial(self.x+.0075*self.d0,1.))

    def test_six_restore(self):
        test_p2gd_elr_replay.ReplayTests().check_restore(methods=tuple(a['name'] for a in ARMS))

    def test_six_restore_after_failure(self):
        test_p2gd_elr_replay.ReplayTests().check_restore(failure=True,methods=tuple(a['name'] for a in ARMS))

    def test_bf16_attenuation_target_uses_cluster_return(self):
        self.x,self.m,self.d0,self.h=(t.float() for t in (self.x,self.m,self.d0,self.h))
        owner=SixReplay.__new__(SixReplay)
        owner.first=253;owner.measure_first=269;owner.iteration=269;owner.near_threshold=2e-4
        owner.current_momentum=self.m;owner.near_records=[]
        owner.raw_gradient_view=lambda w:self.m
        module=torch.nn.Linear(4,4,bias=False).to(device=self.x.device,dtype=torch.bfloat16)
        module.weight.data.copy_(self.x)
        forward=module.weight.float().detach().clone()
        owner.contexts={owner.key(self.x):dict(calibration=self.h,heldout=self.h,
            model_before=forward,module=module,index=0,parameter='linear.weight',row_start=0,row_stop=4)}
        emitted=[];owner.emit=lambda name,row:emitted.append(row)
        owner.propose_update(self.x,self.d0,1.,.015,None,self.x)
        points=owner.near_records[0]['candidates']
        target=float((self.h@(points['cluster_cone'].bfloat16().float()-forward).T).norm()/
                     (self.h@forward.T).norm())
        row=next(r for r in emitted[0]['rows'] if r['method']=='half_output_of_cluster')
        self.assertAlmostEqual(row['bf16_budget_target'],target,places=6)
        self.assertEqual(row['target_reference'],'cluster_cone')


if __name__=='__main__':
    unittest.main()
