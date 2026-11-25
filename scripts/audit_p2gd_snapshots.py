"""Offline capacity/return audit; never changes training or asserts LLM loss gain."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from p2gd_budget_geometry import norm2, radial
from p2gd_elr_geometry import repair
from run_p2gd_trajectories import write
from verify_p2gd_six import read


def block_ritz(y, v, iterations):
    v=torch.linalg.qr(v,mode='reduced').Q
    for _ in range(iterations):
        u=torch.linalg.qr(y@v,mode='reduced').Q
        v=torch.linalg.qr(y.T@u,mode='reduced').Q
    u,h=torch.linalg.qr(y@v,mode='reduced')
    a,s,bh=torch.linalg.svd(h,full_matrices=False)
    return u@a,s,v@bh.T


def timed(fn):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start=time.perf_counter();out=fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return out,time.perf_counter()-start


@torch.no_grad()
def main():
    root=Path(os.environ['PROJECT_DIR'])
    if not (root/'results/REPLICATION_COMPLETED.json').exists():
        raise RuntimeError('Branches not fully verified')
    folder=root/'results/branch_cluster_cone/metrics'
    index=read(folder/'snapshot_index.jsonl') if (folder/'snapshot_index.jsonl').exists() else []
    if len(index)>24 or sum(r['bytes'] for r in index)>256*1024**2:
        raise RuntimeError('Snapshot cap violated')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    results=[]
    for record in index:
        path=folder/'matrix_snapshots'/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError('Snapshot changed')
        data=torch.load(path,map_location=device,weights_only=True)
        x,m,d0,h=(data[k].float() for k in ('x','momentum','d0','h'))
        u,s,v=(data[k].float() for k in ('u','s','v'))
        q,R,a=data['q'],data['radius'],data['selected_info']['effective_alpha']
        if q>u.shape[1] or min(x.shape)>v.shape[1] and v.shape[1]<12:
            u,s,vh=torch.linalg.svd(x,full_matrices=False,**({'driver':'gesvd'} if device=='cuda' else {}))
            v=vh.T
        cases=[];full_trial=None
        for label,cap in (('top4',4),('top6',6),('full',q)):
            k=min(cap,q)
            d=repair('cluster_cone',d0,m,u[:,:k],s[:k],v[:,:k],R,a)
            y=x+a*d
            point=radial(y,R)
            ref=data['selected_point']
            cases.append(dict(name=label,capacity=cap,used=k,omitted=max(0,q-k),
                point_relative_error=float((point-ref).norm()/ref.norm()),
                master_output_elr=float((h@(point-x).T).norm()/(h@x.T).norm()),
                bf16_output_elr=float((h@(point.bfloat16().float()-data['before']).T).norm()/(h@data['before'].T).norm()),
                spectral_elr=float(norm2(point-x)/norm2(x)),
                momentum_work=float((m*(point-x)).sum()),radius_error=abs(float(norm2(point)/R)-1)))
            if label=='full': full_trial=y
        exact,t_svd=timed(lambda: torch.linalg.svdvals(full_trial,**({'driver':'gesvd'} if device=='cuda' else {})))
        exact_point=full_trial*(R/exact[0])
        gap=float(1-exact[1]/exact[0])
        approximations=[]
        for b in (8,12):
            for iterations in (10,20):
                (_,sh,_),seconds=timed(lambda: block_ritz(full_trial,v[:,:min(b,v.shape[1])],iterations))
                point=full_trial*(R/sh[0])
                approximations.append(dict(block=b,iterations=iterations,seconds=seconds,
                    radius_error=abs(float(exact[0]/sh[0])-1),
                    gap_error=abs(float(1-sh[1]/sh[0])-gap),
                    point_relative_error=float((point-exact_point).norm()/exact_point.norm())))
        results.append(dict(snapshot=record['path'],category=record['category'],q=q,
            capacities=cases,approximate_returns=approximations,svdvals_seconds=t_svd))
    write(root/'results/SNAPSHOT_AUDIT_COMPLETED.json',dict(snapshots=len(results),results=results,
        initialization='Oracle-warm: true current-X right subspace, not persistent approximate training cache',
        timing='Single-call synchronized diagnostic, not end-to-end training speed',
        scope='Geometric full-return comparisons on frozen inputs; momentum work is not LLM loss descent'))


if __name__=='__main__':
    main()
