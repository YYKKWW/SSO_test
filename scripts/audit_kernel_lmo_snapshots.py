"""Same-matrix counterfactuals; momentum-return scores are NOT loss changes."""
import argparse,json,math
from pathlib import Path
import torch
from kernel_lmo_h20 import PilotAdapter,candidate,host,sym
from local_kernel_lmo_study import coupling_bound


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--per-directory",type=int,default=6)
    args=parser.parse_args(); torch.set_num_threads(1); torch.set_grad_enabled(False)
    rows=[]
    for directory in args.directories:
        files=sorted(directory.glob("sample_*.pt"))
        chosen=sorted(set(round(i*(len(files)-1)/max(1,args.per_directory-1))
                          for i in range(min(args.per_directory,len(files)))))
        for index in chosen:
            path=files[index]; data=torch.load(path,map_location=args.device,weights_only=True)
            x,m=data["current"].float(),data["momentum"].float()
            radius=float(data["radius"]); alpha=float(data.get("alpha",data.get("effective_lr",0.)))
            if alpha<=0: continue
            u,s,vh=host.accurate_svd_fp32(x)
            pre_bias=float(s[0])/radius-1
            x=x*(radius/s[0]); s=s*(radius/s[0])
            exact=PilotAdapter("a2","exact"); cache=exact.cache(u,s,vh,radius)
            q=max(1,int(((cache.s_basis[0]-cache.s_basis)/cache.s_basis[0]<=2e-4).sum()))
            for method in ("control","a2","a2_1p5","kernel1"):
                used=1 if method=="control" else q
                d,_=candidate(m,cache,radius,alpha,used,method)
                y=x+alpha*d; uy,sy,vhy=host.accurate_svd_fp32(y); returned=y*(radius/sy[0])
                b=coupling_bound((cache.u_basis,cache.s_basis,cache.v_basis),d,used,radius,alpha)
                capacity=None
                if method!="control":
                    for j in sorted(set([q,min(x.shape)]+[n for n in (4,8,16,32,64,128) if q<=n<=min(x.shape)])):
                        dj,_=candidate(m,cache,radius,alpha,j,method)
                        dj_norm=float(host.accurate_svdvals_fp32(dj)[0])
                        gamma=radius-float(cache.s_basis[j]) if j<len(cache.s_basis) else radius
                        if 2*alpha*dj_norm<=gamma:
                            capacity=j; break
                rows.append(dict(source=str(path),step=data["step"],shape=list(x.shape),method=method,
                    radius=radius,alpha=alpha,pre_normalization_bias=pre_bias,q=used,window_q=q,
                    rebuilt_capacity_on_grid=capacity,trial_fro_rel=float((alpha*d).norm())/radius,
                    returned_fro_rel=float((returned-x).norm())/radius,
                    radial_fro_rel=float((returned-y).norm())/radius,
                    momentum_trial=float((m*(alpha*d)).sum()),momentum_returned=float((m*(returned-x)).sum()),
                    trial_overshoot=float(sy[0])-radius,post_gap=float(1-sy[1]/sy[0]),**b))
    result=dict(rows=rows,note="Same feasible-normalized real X and original M. Scores use momentum, NOT actual objective change. Exact FP32 SVD reference; rebuilt capacity is on a grid, not a minimal-rank proof. Offline only.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(rows=len(rows),output=str(args.output))))


if __name__=="__main__": main()
