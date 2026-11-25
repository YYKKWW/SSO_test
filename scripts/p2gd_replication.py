"""Bounded snapshot collection; no effect on the selected training update."""
from collections import Counter
import hashlib
import os
from pathlib import Path

import torch

from adapter import tensor_sha
from p2gd_trajectory import Trajectory, optimizer_signature


class SnapshotSelector:
    def __init__(self, first, budget=256*1024**2):
        self.first, self.budget = first, budget
        self.last, self.counts, self.seen = {}, Counter(), set()
        self.bytes = 0

    def choose(self, step, component, q, size):
        repeated = self.last.get(component) == step-1
        self.last[component] = step
        category = 'large_cluster' if q>4 else ('repeat' if repeated else 'entry')
        window = (step-self.first)//64
        key = window, category
        if window not in range(4) or self.counts[key]>=2 or (window,component) in self.seen:
            return None
        if self.bytes+size>self.budget:
            return None
        self.counts[key]+=1
        self.seen.add((window,component))
        self.bytes+=size
        return dict(window=window,category=category,repeated=repeated)


class ReplicationTrajectory(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_root=Path(os.environ['ELR_OUTPUT'])/'matrix_snapshots'
        self.selector=SnapshotSelector(self.first)

    @torch.no_grad()
    def observe_near_component(self, context, x, momentum, d0, u, s, v,
                               radius, alpha, before, point, info):
        if self.policy!='cluster_cone' or os.environ.get('P2GD_SNAPSHOTS')!='1':
            return
        q=int(((1-s/s[0])<=self.near_threshold).sum())
        tensors=dict(x=x,momentum=momentum,d0=d0,s=s,u=u[:,:16],v=v[:,:16],
                     h=context['calibration'],before=before,selected_point=point)
        size=sum(t.numel()*t.element_size() for t in tensors.values())+65536
        selection=self.selector.choose(self.iteration,context['index'],q,size)
        if selection is None:
            return
        self.snapshot_root.mkdir(parents=True,exist_ok=True)
        path=self.snapshot_root/f"step{self.iteration}_component{context['index']}.pt"
        meta=dict(iteration=self.iteration,component=context['index'],parameter=context['name'],
                  q=q,radius=radius,alpha=alpha,gap=float(1-s[1]/s[0]),
                  near_threshold=self.near_threshold,selected_info=info,**selection)
        torch.save(dict(**{k:t.detach().cpu().clone() for k,t in tensors.items()},**meta),path)
        self.emit('snapshot_index',dict(**meta,path=path.name,bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    @torch.no_grad()
    def after_optimizer_update(self, model, optimizer):
        if self.iteration!=int(os.environ.get('P2GD_RECORD_POST_STATE','-1')):
            return
        from megatron.training.utils import unwrap_model
        core=unwrap_model(model)[0]
        self.emit('checkpoint_post_state',dict(iteration=self.iteration,
            model=[(name,tensor_sha(p)) for name,p in core.named_parameters()],
            optimizer=optimizer_signature(optimizer)))
