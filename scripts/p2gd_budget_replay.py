"""Near-only spectral/output ELR counterfactuals on a shared half-D0 path."""
import torch

from matched_geometry import geometry_precision, return_point, svd
from p2gd_elr_geometry import aus
from p2gd_elr_replay import RepairReplay
from p2gd_budget_geometry import ARMS, candidates, norm2


class BudgetReplay(RepairReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods = tuple(a['name'] for a in ARMS)

    @torch.no_grad()
    @geometry_precision()
    def propose_update(self, x, d0, radius, alpha, reference, W):
        u, s, vh = svd(x)
        gap = float(1-s[1]/s[0])
        near = gap <= self.near_threshold
        baseline = return_point(x+alpha*(.5 if near else 1.)*d0, radius, 'radial')
        self.own_reference = baseline
        target = aus(x,baseline)
        info = dict(effective_alpha=alpha, target=target, measured=target,
            tolerance=2e-6+2e-3*target, relative_error=0., evaluations=1, metric='aus',
            matching_enforced=False, reference_reused=False, return_kind='radial')
        if not near:
            return baseline, info
        q = int(((1-s/s[0])<=self.near_threshold).sum())
        context = self.contexts[self.key(W)]
        points, records, targets = candidates(x,self.current_momentum,d0,u[:,:q],s[:q],vh[:q].T,
            radius,alpha,context['calibration'],baseline)
        g = self.raw_gradient_view(W).float()
        h_independent = context['heldout']
        output_denominator = (h_independent@x.T).norm().clamp_min(1e-30)
        forward_x = context['model_before']
        forward_dtype = context['module'].weight.dtype
        forward_norm = norm2(forward_x)
        forward_output_norm = (context['calibration']@forward_x.T).norm().clamp_min(1e-30)
        base_forward_delta = baseline.to(forward_dtype).float()-forward_x
        forward_targets = dict(spectral=float(norm2(base_forward_delta)/forward_norm),
            output=float((context['calibration']@base_forward_delta.T).norm()/forward_output_norm))
        for record in records:
            delta = points[record['method']]-x
            forward_delta = points[record['method']].to(forward_dtype).float()-forward_x
            record.update(gradient_work=float((g*delta).sum()),
                independent_output_elr=float((h_independent@delta.T).norm()/output_denominator),
                bf16_spectral_elr=float(norm2(forward_delta)/forward_norm),
                bf16_output_elr=float((context['calibration']@forward_delta.T).norm()/forward_output_norm),
                bf16_independent_output_elr=float((h_independent@forward_delta.T).norm()/
                    (h_independent@forward_x.T).norm().clamp_min(1e-30)))
            metric = record['metric']
            if metric!='reference':
                value = record['bf16_spectral_elr'] if metric=='spectral' else record['bf16_output_elr']
                expected = forward_targets[metric]*record['budget_fraction']
                record['bf16_budget_relative_error'] = abs(value-expected)/max(expected,1e-30)
        self.emit('repair_geometry',dict(iteration=self.iteration,component=context['index'],
            parameter=context['parameter'],shape=list(x.shape),gap=gap,cluster_dimension=q,
            exceeds_top4=q>4,cluster_truncated=False,rows=records,baseline_targets=targets,
            bf16_baseline_targets=forward_targets,
            scope='Each policy matches only its named complete-return metric; never loss'))
        self.near_records.append(dict(W=W,module=context['module'],row_start=context['row_start'],
            row_stop=context['row_stop'],candidates=points))
        return baseline, info
