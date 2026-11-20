"""Real weight/momentum snapshots with synthetic calibration inputs; no LLM loss."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from p2gd_budget_geometry import candidates, norm2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    torch.manual_seed(20261120)
    start=time.perf_counter();result=[]
    for name in ('sample_004.pt','sample_009.pt','sample_024.pt'):
        path=args.snapshots/name
        sample=torch.load(path,map_location='cpu',weights_only=True)
        x=sample['current'].double();m=sample['momentum'].double();d0=sample['d0'].double()
        radius,alpha=float(sample['radius']),float(sample['alpha'])
        x=x*(radius/norm2(x));u,s,vh=torch.linalg.svd(x,full_matrices=False)
        q=int(((1-s/s[0])<=2e-4).sum())
        h=torch.randn(256,x.shape[1],dtype=torch.float64)*torch.logspace(-.6,.6,x.shape[1],dtype=torch.float64)
        _,records,targets=candidates(x,m,d0,u[:,:q],s[:q],vh[:q].T,radius,alpha,h)
        result.append(dict(source=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shape=list(x.shape),q=q,targets=targets,records=records))
        print(name,'passed',len(records),'candidates',flush=True)
    report=dict(seconds=time.perf_counter()-start,cases=result,
        scope='Actual saved X/M/D0; synthetic anisotropic H; geometry only, not LLM loss')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
