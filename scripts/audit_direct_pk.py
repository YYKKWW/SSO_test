"""Same-state D0 replay with actual radial or spectral-sphere returns.

Real snapshots expose a momentum surrogate, not a minibatch loss oracle.
Exact FP64 auditing is separate from production optimizer SVD counts.
"""
import argparse
import json
from pathlib import Path

import torch

from direct_top_repair import shared_directions
from direct_top_pgd_pk import MODES, RATIOS, shared_candidates, sphere_from_svd


@torch.no_grad()
def replay(x, m, radius, alpha, stored_d0=None, *, ambient_return="sphere_projection"):
    if ambient_return not in ("sphere_projection", "radial"):
        raise ValueError("invalid ambient return")
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    pre_bias=float(s[0]/radius-1)
    factor=radius/s[0]
    x,s=x*factor,s*factor
    q=max(1,int(((s[0]-s)/s[0]<=2e-4).sum()))
    d0=stored_d0.to(x) if stored_d0 is not None else shared_directions(
        m,(u,s,vh.T),radius,alpha,q)[0]
    rows=[]
    for mode,d in shared_candidates(m,d0,u,s,vh.T,radius,alpha,q).items():
        y=x+alpha*d
        uy,sy,vhy=torch.linalg.svd(y,full_matrices=False)
        is_ambient=mode in RATIOS and q>1
        is_pgd=is_ambient and ambient_return == "sphere_projection"
        if is_pgd:
            returned,values=sphere_from_svd(uy,sy,vhy,radius)
        else:
            returned,values=y*(radius/sy[0]),sy*(radius/sy[0])
        displacement=returned-x
        rows.append(dict(mode=mode,q=q,pgd=is_pgd,pre_radius_bias=pre_bias,
            ambient_near=is_ambient,return_kind="sphere_projection" if is_pgd else "radial",
            d0_source="stored" if stored_d0 is not None else "computed_once",
            d0_trial_length=float((alpha*d0).norm())/radius,
            trial_length=float((alpha*d).norm())/radius,
            returned_length=float(displacement.norm())/radius,
            return_correction_length=float((returned-y).norm())/radius,
            returned_gap=float(1-values[1]/values[0]),
            clipped_count=int((sy>radius).sum()) if is_pgd else 0,
            radius_error=abs(float(torch.linalg.svdvals(returned)[0]/radius-1)),
            momentum_trial=float((m*(alpha*d)).sum()),
            momentum_returned=float((m*displacement).sum())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',nargs='+',type=Path,required=True)
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ambient-return',choices=('sphere_projection','radial'),default='sphere_projection')
    args=p.parse_args()
    if args.limit<1:p.error('limit must be positive')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    rows=[]
    for folder in args.snapshots:
        files=sorted(folder.glob('sample_*.pt'));n=min(len(files),args.limit)
        for i in sorted({round(k*(len(files)-1)/max(1,n-1)) for k in range(n)}):
            data=torch.load(files[i],map_location=args.device,weights_only=True)
            alpha=float(data['alpha'])
            if alpha<=0:continue
            for row in replay(data['current'].double(),data['momentum'].double(),
                              float(data['radius']),alpha,data.get('d0'),ambient_return=args.ambient_return):
                rows.append(dict(source=str(files[i]),step=data['step'],**row))
    if not rows:raise RuntimeError('empty snapshot audit')
    summary=[]
    for mode in MODES:
        r=[row for row in rows if row['mode']==mode]
        near=[row for row in r if row['q']>1]
        summary.append(dict(mode=mode,states=len(r),near_states=len(near),
            mean_trial_length=sum(row['trial_length'] for row in r)/len(r),
            mean_returned_length=sum(row['returned_length'] for row in r)/len(r),
            near_mean_returned_length=sum(row['returned_length'] for row in near)/len(near) if near else None,
            non_descent_momentum=sum(row['momentum_returned']>1e-12 for row in r),
            near_returned_plateau=sum(row['returned_gap']<=1e-12 for row in near),
            max_clipped=max(row['clipped_count'] for row in r),
            max_radius_error=max(row['radius_error'] for row in r)))
    report=dict(summary=summary,rows=rows,ambient_return=args.ambient_return,
        note='Same X, M and D0 for each counterfactual. FP64 full-return audit; not a true LLM objective evaluation. Source X is commonly normalized to R first. The selected ambient_return determines the near-branch return; radial is not PGD. No hyperparameter is selected by these momentum scores alone.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
