"""Four near-branch controls calibrated on rounded forward-weight movement.

The common half-D0 update is an additional reference, not a fifth alternative.
Search sees geometry only. Quantization can prevent an exact budget match;
in that case the best observed candidate and its nonzero mismatch are retained.
"""
import math

import torch

from p2gd_budget_geometry import norm2, radial
from p2gd_elr_geometry import aus, repair

ARMS = [dict(name='half_d0', direction='half_d0', metric='reference', fraction=1.),
        dict(name='half_output_of_cluster', direction='half_d0', metric='output', fraction=1.)]
ARMS += [dict(name=name, direction=name, metric='spectral', fraction=1.)
         for name in ('ambient','top1','cluster_cone')]


def forward_match(x, direction_at, radius, alpha, target, metric, h, before, dtype,
                  rtol=2e-4, atol=2e-7, iterations=30, max_scale=64.):
    if target<=0 or alpha<=0 or not math.isfinite(target):
        raise ValueError('Positive finite coefficient and forward target required')
    if not torch.equal(x.to(dtype).float(), before):
        raise ValueError('Forward weights do not match rounded current masters')
    denominator = norm2(before) if metric=='spectral' else (h@before.T).norm()
    if metric not in ('spectral','output') or denominator<=0:
        raise ValueError('Invalid forward metric or reference')
    tolerance=atol+rtol*target
    calls=0

    def evaluate(coefficient):
        nonlocal calls
        calls+=1
        d=direction_at(coefficient)
        point=radial(x+coefficient*d,radius)
        delta=point.to(dtype).float()-before
        value=float((norm2(delta) if metric=='spectral' else (h@delta.T).norm())/denominator)
        if not math.isfinite(value):
            raise ValueError('Nonfinite rounded displacement')
        return coefficient,d,point,value

    lo,hi=0.,alpha
    current=best=evaluate(hi)
    while current[3]<target-tolerance and hi<max_scale*alpha:
        lo,hi=hi,min(2*hi,max_scale*alpha)
        current=evaluate(hi)
        if abs(current[3]-target)<abs(best[3]-target):
            best=current
    bracketed=current[3]>=target-tolerance
    if bracketed:
        for _ in range(iterations):
            if abs(best[3]-target)<=tolerance:
                break
            mid=(lo+hi)*.5
            current=evaluate(mid)
            if abs(current[3]-target)<abs(best[3]-target):
                best=current
            if current[3]<target:
                lo=mid
            else:
                hi=mid
    coefficient,d,point,value=best
    satisfied=abs(value-target)<=tolerance
    return d,point,dict(effective_alpha=coefficient,target=target,measured=value,
        tolerance=tolerance,relative_error=abs(value-target)/target,evaluations=calls,
        matching_enforced=satisfied,matching_satisfied=satisfied,matching_space='bf16_forward',
        matching_status='within_tolerance' if satisfied else 'best_observed_unmatched',
        search_bracketed=bracketed,metric=metric,return_kind='radial',reference_reused=False,
        norm_svd_calls=(1 if metric=='spectral' else 0)+calls*(2 if metric=='spectral' else 1))


def angle(a,b):
    denominator=a.norm()*b.norm()
    if denominator<=1e-30:
        return None
    return math.degrees(math.acos(float(((a*b).sum()/denominator).clamp(-1,1))))


@torch.no_grad()
def candidates(x,m,d0,u,s,v,radius,alpha,h,baseline,before,dtype):
    if dtype != torch.bfloat16:
        raise ValueError('This replay requires BF16 forward weights')
    delta0=baseline.to(dtype).float()-before
    spectral_target=float(norm2(delta0)/norm2(before))
    output_norm=(h@before.T).norm()
    points={'half_d0':baseline}
    details={'half_d0':(.5*d0,dict(effective_alpha=alpha,target=0.,measured=0.,tolerance=0.,
        relative_error=0.,evaluations=0,matching_enforced=False,matching_satisfied=True,
        matching_space='reference',matching_status='reference',metric='reference',
        return_kind='radial',reference_reused=True,norm_svd_calls=0))}
    for name in ('ambient','top1','cluster_cone'):
        fn=lambda coefficient:repair(name,d0,m,u,s,v,radius,coefficient)
        d,point,info=forward_match(x,fn,radius,alpha,spectral_target,'spectral',h,before,dtype)
        points[name]=point;details[name]=(d,info)
    cluster_delta=points['cluster_cone'].to(dtype).float()-before
    output_target=float((h@cluster_delta.T).norm()/output_norm)
    d,point,info=forward_match(x,lambda coefficient:.5*d0,radius,alpha,
        output_target,'output',h,before,dtype)
    points['half_output_of_cluster']=point;details['half_output_of_cluster']=(d,info)
    z=-.5*d0.norm()*m/m.norm().clamp_min(1e-30)
    a=u.T@z@v;a=(a+a.T)*.5
    ambient_delta=points['ambient'].to(dtype).float()-before
    ambient_norm=ambient_delta.norm().clamp_min(1e-30)
    records=[]
    for arm in ARMS:
        name=arm['name'];d,info=details[name];point=points[name];delta=point-x
        forward_delta=point.to(dtype).float()-before
        block=u.T@d@v;block=(block+block.T)*.5
        momentum_arm=name in ('ambient','top1','cluster_cone')
        correction=block-a if momentum_arm else torch.zeros_like(a)
        singular=torch.linalg.svdvals(correction)
        rank_tolerance=max(float(z.norm())*1e-7,float(singular.max())*1e-5,1e-12)
        records.append(dict(method=name,direction=arm['direction'],budget_fraction=1.,**info,
            target_reference='cluster_cone' if name=='half_output_of_cluster' else 'half_d0',
            alpha_ratio=info['effective_alpha']/alpha,
            returned_spectral_elr=float(norm2(delta)/norm2(x)),
            local_output_elr=float((h@delta.T).norm()/(h@x.T).norm()),actual_aus=aus(x,point),
            radius_error=abs(float(norm2(point)/radius)-1),returned_frobenius_over_radius=float(delta.norm()/radius),
            momentum_work=float((m*delta).sum()),
            return_correction_fro=float((delta-info['effective_alpha']*d).norm()),
            cluster_normal_max_eigenvalue=float(torch.linalg.eigvalsh(block)[-1]),
            correction_fro_over_momentum_direction=float(correction.norm()/z.norm().clamp_min(1e-30)) if momentum_arm else None,
            correction_rank=int((singular>rank_tolerance).sum()) if momentum_arm else None,
            correction_rank_tolerance=rank_tolerance if momentum_arm else None,
            momentum_direction_angle_degrees=angle(z,d) if momentum_arm else None,
            momentum_top_block_angle_degrees=angle(a,block) if momentum_arm else None,
            raw_momentum_top_eigenvalues=torch.linalg.eigvalsh(a).tolist(),
            corrected_top_eigenvalues=torch.linalg.eigvalsh(block).tolist(),
            bf16_delta_vs_ambient_fro_fraction=float((forward_delta-ambient_delta).norm()/ambient_norm),
            bf16_changed_fraction_vs_ambient=float((forward_delta!=ambient_delta).float().mean()),
            bf16_rounding_displacement_fraction=float((forward_delta-delta).norm()/delta.norm().clamp_min(1e-30))))
    return points,records,dict(spectral=spectral_target,output=output_target,
        space='bf16_forward',output_reference='cluster_cone')
