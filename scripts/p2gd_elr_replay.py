"""Same-state LLM counterfactuals on a common MCSD-TP/half-D0 trajectory.

Requires the frozen spectral-elr-confirm-w256-20261118-v1 diagnostic host.
Only this opt-in adapter changes; no production optimizer or old run is edited.
All non-matrix updates are common. Alternate candidates never enter momentum.
"""
import torch

from adapter import Intervention, tensor_sha
from confirm_adapter import ConfirmationIntervention
from matched_geometry import geometry_precision, return_point, spectral_norm, svd
from p2gd_elr_geometry import METHODS, aus, match_aus, repair


class RepairReplay(ConfirmationIntervention):
    def __init__(self, output, first=253, last=268):
        super().__init__(2, output, first, last)
        self.method = 'mcsd_tp_half_near_common_trajectory'
        self.near_records = []
        self.replay_records = None
        self.last_heldout = None
        self.methods = METHODS

    def emit(self, name, record):
        if name == 'heldout_probe':
            self.last_heldout = record
        return super().emit(name, record)

    def observe_direction(self, *args):
        # The inherited direction is D0, not the repaired direction. Log below.
        pass

    def observe_component(self, W, x, updated, reference, calibration):
        super().observe_component(W, x, updated, self.own_reference, calibration)

    @torch.no_grad()
    @geometry_precision()
    def component(self, W, M, *args, **kwargs):
        self.current_momentum = M.float().clone()
        result = super().component(W, M, *args, **kwargs)
        self.calibration_step[-1]['near_changed'] = self.calibration_step[-1]['near_eligible']
        # Do not publish inherited D0 diagnostics as repaired-direction metrics.
        self.pending.clear()
        return result

    @torch.no_grad()
    @geometry_precision()
    def propose_update(self, x, d0, radius, alpha, reference, W):
        u, s, vh = svd(x)
        gap = float(1-s[1]/s[0])
        near = gap <= self.near_threshold
        d = (.5 if near else 1.)*d0
        baseline = return_point(x+alpha*d, radius, 'radial')
        self.own_reference = baseline
        target = aus(x, baseline)
        record = dict(effective_alpha=alpha, target=target, measured=target,
            tolerance=2e-6+2e-3*target, relative_error=0., evaluations=1,
            metric='aus', matching_enforced=False, reference_reused=False,
            return_kind='radial', baseline='own-state half-D0 if near, otherwise D0')
        if not near:
            return baseline, record

        q = int(((1-s/s[0]) <= self.near_threshold).sum())
        uq, sq, vq = u[:, :q], s[:q], vh[:q].T
        context = self.contexts[self.key(W)]
        m = self.current_momentum
        g = self.raw_gradient_view(W).float()
        candidates, geometry = {}, []
        for method in METHODS:
            if method == 'half_d0':
                direction, candidate, info = .5*d0, baseline, dict(record)
            else:
                fn = lambda coefficient: repair(method, d0, m, uq, sq, vq, radius, coefficient)
                direction, trial, info = match_aus(x, fn, alpha, target)
                candidate = return_point(trial, radius, 'radial')
                if abs(aus(x, candidate)-target) > info['tolerance']+2e-6:
                    raise RuntimeError('Actual radial return broke AUS matching')
            delta = candidate-x
            h = uq.T@direction@vq
            h = (h+h.T)*.5
            h_input = uq.T@(-m/m.norm().clamp_min(1e-30))@vq
            candidates[method] = candidate
            hcal = context['calibration']
            geometry.append(dict(method=method, **info,
                alpha_ratio=info['effective_alpha']/alpha,
                returned_spectral_elr=float(spectral_norm(delta)/spectral_norm(x)),
                returned_frobenius_over_radius=float(delta.norm()/radius),
                actual_aus=aus(x, candidate),
                local_output_elr=float((hcal@delta.T).norm()/(hcal@x.T).norm().clamp_min(1e-30)),
                gradient_work=float((g*delta).sum()), momentum_work=float((m*delta).sum()),
                trial_gradient_work=float((g*(info['effective_alpha']*direction)).sum()),
                return_correction_fro=float((delta-info['effective_alpha']*direction).norm()),
                cluster_normal_max_eigenvalue=float(torch.linalg.eigvalsh(h)[-1]),
                raw_momentum_cluster_max_eigenvalue=float(torch.linalg.eigvalsh((h_input+h_input.T)*.5)[-1]),
                radius_error=abs(float(spectral_norm(candidate)/radius)-1)))
        self.emit('repair_geometry', dict(iteration=self.iteration, component=context['index'],
            parameter=context['parameter'], shape=list(x.shape), gap=gap,
            cluster_window=self.near_threshold, cluster_dimension=q, exceeds_top4=q>4,
            cluster_truncated=False, rows=geometry,
            scope='FP32 full-spectrum diagnostic; a numerical cluster is not an exact multiplicity'))
        self.near_records.append(dict(W=W, module=context['module'],
            row_start=context['row_start'], row_stop=context['row_stop'], candidates=candidates))
        return baseline, record

    @torch.no_grad()
    def finish_update(self):
        super().finish_update()
        self.replay_records = self.near_records
        self.near_records = []

    @torch.no_grad()
    def evaluate_probe(self, model):
        replay = self.replay_records
        baseline_training = super().evaluate_probe(model)
        if replay is None:
            return baseline_training
        if self.last_heldout is None or self.last_heldout['iteration'] != self.iteration:
            raise RuntimeError('Missing matched before/after held-out values')
        self.replay_records = None
        baseline_heldout = self.last_heldout
        saved_batch = self.batch
        backups = [(r, r['W'].clone(),
                    r['module'].weight[r['row_start']:r['row_stop']].clone()) for r in replay]
        results = []
        try:
            for method in getattr(self, 'methods', METHODS):
                changed = 0
                for r, master_before, model_before in backups:
                    candidate = r['candidates'][method]
                    model_slice = r['module'].weight[r['row_start']:r['row_stop']]
                    r['W'].copy_(candidate)
                    model_slice.copy_(candidate)
                    changed += int((model_slice != model_before).sum())
                if method == 'half_d0' or changed == 0:
                    training_loss = baseline_training[0]
                    heldout = baseline_heldout['after']
                else:
                    self.batch = saved_batch
                    training_loss = Intervention.evaluate_probe(self, model)[0]
                    heldout = []
                    for batch in self.fixed_batches:
                        self.batch = batch
                        loss, _ = Intervention.evaluate_probe(self, model)
                        heldout.append(dict(loss=loss, tokens=tensor_sha(batch[0]),
                            labels=tensor_sha(batch[1]), valid_tokens=float(batch[2].sum())))
                total = sum(r['valid_tokens'] for r in heldout)
                value = sum(r['loss']*r['valid_tokens'] for r in heldout)/total
                results.append(dict(method=method, heldout_loss=value,
                    heldout_delta=value-baseline_heldout['loss_before'],
                    heldout_difference_vs_half=value-baseline_heldout['loss_after'],
                    heldout_batches=heldout, training_microbatch_loss=training_loss,
                    training_difference_vs_half=training_loss-baseline_training[0],
                    changed_bf16_elements_vs_half=changed))
        finally:
            self.batch = saved_batch
            for r, master_before, model_before in backups:
                r['W'].copy_(master_before)
                model_slice = r['module'].weight[r['row_start']:r['row_stop']]
                model_slice.copy_(model_before)
                if not torch.equal(r['W'], master_before) or not torch.equal(model_slice, model_before):
                    raise RuntimeError('Counterfactual restoration failed')
        self.emit('repair_loss_replay', dict(iteration=self.iteration, near_components=len(replay),
            before_heldout=baseline_heldout['loss_before'], results=results,
            restore_asserted=True, scope='Same X/G/Nesterov input and non-matrix update for all methods; '
            'only half-D0 is committed to the common trajectory; not independent optimizer training'))
        return baseline_training
