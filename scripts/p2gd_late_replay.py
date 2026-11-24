"""Continue the same path to step 512, then audit four near-branch alternatives."""
import os
import json
from pathlib import Path

import torch

from adapter import choose_direction
from matched_geometry import geometry_precision, return_point, spectral_norm, svd
from p2gd_budget_replay import BudgetReplay
from p2gd_late_geometry import ARMS, candidates
from verify_p2gd_late import verify_initial_prefix


class LateReplay(BudgetReplay):
    def __init__(self,output,first=253,last=528):
        super().__init__(output,first,last)
        self.methods=tuple(a['name'] for a in ARMS)
        self.measure_first=513

    def auditing(self):
        return self.iteration<=268 or self.iteration>=self.measure_first

    def emit(self,name,record):
        result=super().emit(name,record)
        if name=='model_probe' and record['iteration']==268:
            prior=Path(os.environ['ELR_PRIOR_REPLAY_ROOT'])/'results/budget_replay/metrics'
            verification=verify_initial_prefix(self.output,prior)
            (self.output/'prefix_verified.json').write_text(json.dumps(verification,indent=2)+'\n')
        return result

    def candidate_factory(self,*args):
        c=self.active_context
        return candidates(*args,before=c['model_before'],dtype=c['module'].weight.dtype)

    def propose_update(self,x,d0,radius,alpha,reference,W):
        self.active_context=self.contexts[self.key(W)]
        return super().propose_update(x,d0,radius,alpha,reference,W)

    @torch.no_grad()
    @geometry_precision()
    def component(self,W,M,target_radius,power_iteration_steps,msign_steps,**kwargs):
        if self.auditing():
            return super().component(W,M,target_radius,power_iteration_steps,msign_steps,**kwargs)
        # Same update as the audited path, without diagnostic forward passes.
        index=self.counter;self.counter+=1
        alpha=float(kwargs.get('current_lr') or 0.)*self.helpers.get_spectral_ball_scale_factor(
            *W.shape,mode=self.scale_mode)
        if alpha<=0:
            raise ValueError('Continuation must be after initialization')
        x=W.float().clone();u,s,vh=svd(x)
        error=abs(float(s[0])/target_radius-1)
        if error>5e-5:
            self.fail_radius('pre',index,W,x,None,None,target_radius,error)
        d0,_=choose_direction(M,u[:,:1]@vh[:1],'mcsd_tp_aus',self.helpers)
        near=float(1-s[1]/s[0])<=self.near_threshold
        point=return_point(x+alpha*(.5 if near else 1.)*d0,target_radius,'radial')
        record=dict(effective_alpha=alpha,near_eligible=near,matching_enforced=False,reference_reused=False)
        self.return_checks.append((W,x,point,float(target_radius),index,0.,record))
        return (x-point)/alpha,0.,float(s[0])

    @torch.no_grad()
    @geometry_precision()
    def finish_update(self):
        if self.auditing():
            return super().finish_update()
        expected=252 if os.environ.get('ELR_ASSERT_LAYOUT','0')=='1' else len(self.return_checks)
        if not self.return_checks or len(self.return_checks)!=expected:
            raise RuntimeError('Missing continuation component updates')
        errors=[]
        for w,_,point,radius,*_ in self.return_checks:
            if not torch.equal(w,point):
                raise RuntimeError('Continuation failed exact master commit')
            errors.append(abs(float(spectral_norm(w))/radius-1))
        if max(errors)>5e-5:
            raise RuntimeError('Continuation violated spectral radius')
        self.emit('continuation_update',dict(iteration=self.iteration,components=len(errors),
            max_radius_error=max(errors),near_components=sum(r[-1]['near_eligible'] for r in self.return_checks),
            exact_commit=True,heldout_evaluated=False))
        self.return_checks.clear();self.calibration_step.clear();self.pending.clear()
        self.inputs.clear();self.contexts.clear();self.near_records=[];self.replay_records=None
