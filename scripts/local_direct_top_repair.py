"""Same-tensor D0 counterfactuals and paired small spectral-sphere trajectories.

Quadratic objective changes are actual complete-return changes. Real LLM
snapshots only provide momentum scores, not a minibatch loss oracle.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch

import local_step_aware_hybrid as base
from direct_top_repair import MODES, shared_directions


def measure(x, m, spec, d0, d, radius, alpha, q, target=None, weights=None):
    y = x + alpha*d
    returned, _, trial_sigma = base.radial_cached(y, radius)
    h = base.sym(spec[0][:, :q].T @ d @ spec[2][:, :q])
    model = torch.diag(spec[1][:q]) + alpha*h
    correction = d-d0
    free_error = base.tangent(correction, spec[0][:, :q], spec[2][:, :q])
    displacement = returned-x
    result = dict(q=q, alpha=alpha, radius=radius,
        d0_trial_length=float((alpha*d0).norm())/radius,
        trial_length=float((alpha*d).norm())/radius,
        returned_length=float(displacement.norm())/radius,
        radial_length=float((returned-y).norm())/radius,
        correction_ratio=float(correction.norm())/max(float(d0.norm()),1e-30),
        free_block_error=float(free_error.norm()),
        model_error=abs(float(torch.linalg.eigvalsh(model)[-1])-radius)/radius,
        cone_error=abs(float(torch.linalg.eigvalsh(h)[-1])),
        radius_error=abs(float(torch.linalg.svdvals(returned)[0])/radius-1),
        trial_overshoot=trial_sigma/radius-1,
        momentum_trial=float((m*(alpha*d)).sum()),
        momentum_returned=float((m*displacement).sum()),
        returned_direction_cosine=float((displacement*d0).sum())/
            max(float(displacement.norm()*d0.norm()),1e-30))
    if target is not None:
        g = weights*(x-target)
        result.update(objective_change=base.objective(returned,target,weights)-base.objective(x,target,weights),
                      gradient_trial=float((g*(alpha*d)).sum()),
                      gradient_returned=float((g*displacement).sum()))
    return returned, result


def same_state(x, m, radius, alpha, metadata, target=None, weights=None, exact_lmo=False, fixed_d0=None):
    u,s,v = base.spectrum(x)
    pre_bias = float(s[0])/radius-1
    factor = radius/s[0]
    x, s = x*factor, s*factor
    # The LLM momentum remains that of the recorded state, not the adjusted X.
    spec = u,s,v
    q = max(1,int(((s[0]-s)/s[0]<=2e-4).sum()))
    if fixed_d0 is None:
        d0, directions = shared_directions(m,spec,radius,alpha,q,exact_lmo=exact_lmo)
    else:
        from direct_top_repair import repair
        d0=fixed_d0.to(x)
        if d0.shape!=x.shape:raise ValueError('stored D0 shape mismatch')
        directions={mode:repair(d0,u[:,:q],s[:q],v[:,:q],radius,alpha,mode) for mode in MODES}
    rows=[]
    for mode, d in directions.items():
        _, metrics = measure(x,m,spec,d0,d,radius,alpha,q,target,weights)
        rows.append(dict(metadata, mode=mode, pre_normalization_bias=pre_bias,
                         d0_source='stored' if fixed_d0 is not None else 'computed_once',
                         lmo="exact_polar" if exact_lmo else "bf16_ns8", **metrics))
    return rows


def snapshot_rows(directories, limit, device):
    rows=[]
    for directory in directories:
        files=sorted(directory.glob('sample_*.pt'))
        indices=sorted(set(round(i*(len(files)-1)/max(1,min(limit,len(files))-1))
                           for i in range(min(limit,len(files)))))
        for i in indices:
            data=torch.load(files[i],map_location=device,weights_only=True)
            alpha=float(data.get('alpha',data.get('effective_lr',0)))
            if alpha<=0: continue
            x,m=data['current'].double(),data['momentum'].double()
            rows.extend(same_state(x,m,float(data['radius']),alpha,
                dict(group="real_snapshot",source=str(files[i]),step=data['step'],shape=list(x.shape)),
                fixed_d0=data.get('d0')))
    return rows


def synthetic_rows():
    rows=[]
    for shape in ((256,256),(128,256),(768,256)):
        for seed in (0,4,8,12,9,13):
            x,target,weights,meta=base.make_case(seed,shape)
            m=weights*(x-target)
            for lr in (.015,.0075,.0015):
                rows.extend(same_state(x,m,1.,lr,
                    dict(group="llm_shape_quadratic",**meta),target,weights))
    # Aligned outside targets force competing top directions and nonzero slack.
    for shape in ((16,12),(256,256)):
        for gap in (0.,5e-5,1.5e-4):
            gen=torch.Generator().manual_seed(661+shape[0])
            r=min(shape)
            u=torch.linalg.qr(torch.randn(shape[0],r,generator=gen))[0]
            v=torch.linalg.qr(torch.randn(shape[1],r,generator=gen))[0]
            s=torch.linspace(.8,.2,r);s[0]=1.;s[1]=1.-gap
            target_s=s.clone();target_s[:2]=torch.tensor([1.2,1.4])
            x=(u*s)@v.T;target=(u*target_s)@v.T
            weights=torch.ones_like(x);m=x-target
            for lr in (.015,.0015):
                rows.extend(same_state(x,m,1.,lr,dict(group="aligned_slack_quadratic",
                    shape=list(shape),gap=gap),target,weights,exact_lmo=True))
    return rows


def trajectories(steps):
    rows=[]
    for shape in ((16,12),(12,16)):
        for seed in range(8):
            x0,target,weights,meta=base.make_case(seed,shape)
            rng=torch.Generator().manual_seed(129830+seed+shape[0]*77)
            noise=.05*torch.randn(steps,*shape,generator=rng)/math.sqrt(x0.numel())
            for mode in MODES:
                x=x0.clone(); stored=torch.zeros_like(x); near=uphill=0
                trial_sum=returned_sum=0.; max_radius=0.; traces=[]
                for t in range(steps):
                    alpha,h,_=base.settings("practical_nesterov",t,steps,1.,min(shape))
                    g=weights*(x-target); sample=g+noise[t]
                    stored=.9*stored+.1*sample; m=.9*stored+.1*sample
                    spec=base.spectrum(x)
                    q=max(1,int(((spec[1][0]-spec[1])/spec[1][0]<=2e-4).sum()))
                    d0,ds=shared_directions(m,spec,1.,alpha,q,exact_lmo=True)
                    xnew,metrics=measure(x,m,spec,d0,ds[mode],1.,alpha,q,target,weights)
                    near+=q>1;uphill+=metrics['objective_change']>1e-12
                    trial_sum+=metrics['trial_length']; returned_sum+=metrics['returned_length']
                    max_radius=max(max_radius,metrics['radius_error']);x=xnew
                    if t%50==0 or t==steps-1:
                        traces.append(dict(step=t+1,loss=base.objective(x,target,weights),q=q))
                f=base.objective(x,target,weights)
                rows.append(dict(meta,mode=mode,steps=steps,final_loss=f,
                    excess=None if meta['fstar'] is None else f-meta['fstar'],
                    near_rate=near/steps,uphill_rate=uphill/steps,
                    mean_trial_length=trial_sum/steps,mean_returned_length=returned_sum/steps,
                    max_radius_error=max_radius,trace=traces,lmo="exact_polar"))
    return rows


def summary(counterfactuals, paths):
    groups=defaultdict(list)
    for row in counterfactuals: groups[row['group'],row['mode']].append(row)
    stats=[]
    for (group,mode),rows in groups.items():
        actual=[r['objective_change'] for r in rows if 'objective_change' in r]
        stats.append(dict(group=group,mode=mode,states=len(rows),near_states=sum(r['q']>1 for r in rows),
            max_q=max(r['q'] for r in rows),max_radius_error=max(r['radius_error'] for r in rows),
            max_model_error=max(r['model_error'] for r in rows),max_free_block_error=max(r['free_block_error'] for r in rows),
            mean_trial_length=sum(r['trial_length'] for r in rows)/len(rows),
            mean_returned_length=sum(r['returned_length'] for r in rows)/len(rows),
            mean_correction_ratio=sum(r['correction_ratio'] for r in rows)/len(rows),
            positive_return_momentum=sum(r['momentum_returned']>1e-12 for r in rows),
            objective_uphill=None if not actual else sum(a>1e-12 for a in actual),
            mean_objective_change=None if not actual else sum(actual)/len(actual)))
    pstats=[]
    for mode in MODES:
        rs=[r for r in paths if r['mode']==mode]
        if not rs:continue
        known=[r['excess'] for r in rs if r['excess'] is not None]
        pstats.append(dict(mode=mode,instances=len(rs),known_optima=len(known),mean_excess=sum(known)/len(known),
            mean_near_rate=sum(r['near_rate'] for r in rs)/len(rs),
            mean_returned_length=sum(r['mean_returned_length'] for r in rs)/len(rs)))
    return dict(counterfactuals=stats,trajectories=pstats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',nargs='*',type=Path,default=[])
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--skip-synthetic',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.limit<1 or args.steps<2:p.error('positive limit and at least two steps required')
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64)
    start=time.perf_counter()
    rows=snapshot_rows(args.snapshots,args.limit,args.device)
    paths=[]
    if not args.skip_synthetic:
        if args.device!='cpu':p.error('synthetic reference uses CPU; CUDA audit requires --skip-synthetic')
        rows+=synthetic_rows();paths=trajectories(args.steps)
    if not rows:raise RuntimeError('empty audit')
    report=dict(summary=summary(rows,paths),counterfactuals=rows,trajectories=paths,
        elapsed_seconds=time.perf_counter()-start,
        note='Same-state comparisons reuse the exact same D0 tensor, X, M, spectrum and alpha. Independent trajectories share only rules, initialization and oracle noise. Real snapshots have momentum scores, NOT true LLM objective changes. All references use full-spectrum radial return; no rescaling after top repair.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=report['summary'],elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
