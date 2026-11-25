"""Independent 256-update paths using the frozen H20 diagnostic host.

All directions are committed, including the near branch. Only input probes
and periodic held-out evaluations are replayed; weights are never rolled back.
"""
import hashlib
import os
import time

import torch

from activation_adapter import ActivationIntervention
from adapter import Intervention, choose_direction, tensor_sha
from confirm_adapter import ConfirmationIntervention
from matched_geometry import geometry_precision, spectral_norm, svd, return_point
from output_geometry import component_rows
from p2gd_trajectory_geometry import POLICIES, selected_return


def optimizer_signature(optimizer):
    def encode(value):
        if torch.is_tensor(value):
            return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError('Unrecorded optimizer state type: '+str(type(value)))
    return [dict(optimizer=type(w.optimizer).__name__,
                 states=[encode(s) for s in w.optimizer.state.values()])
            for w in getattr(optimizer, 'chained_optimizers', [optimizer])]


class Trajectory(ConfirmationIntervention):
    def __init__(self, output, policy, first=513, last=768):
        if policy not in POLICIES:
            raise ValueError(policy)
        super().__init__(2, output, first, last)
        self.policy = policy
        self.method = 'bf16_budget_trajectory_' + policy
        self.interval = 32
        self.evaluation_records = []
        self.step_rows = []

    def observe_near_component(self, context, x, momentum, d0, u, s, v,
                               radius, alpha, before, point, info):
        pass

    def after_optimizer_update(self, model, optimizer):
        pass

    def auditing(self):
        return self.iteration == self.first or self.iteration == self.last or self.iteration % self.interval == 0

    @torch.no_grad()
    def prepare_inputs(self, model):
        if self.calibration_batch is None:
            if len(self.fixed_batches) != self.heldout_batch_count + 1:
                raise RuntimeError('Require separate calibration and 20 held-out batches')
            self.calibration_batch = self.fixed_batches.pop(0)
            if tensor_sha(self.calibration_batch[0]) in {tensor_sha(b[0]) for b in self.fixed_batches}:
                raise RuntimeError('Calibration/evaluation overlap')
        cal = self.capture_inputs(model, self.calibration_batch)
        self.inputs = {name: dict(module=module, calibration=h) for name,(module,h) in cal.items()}
        if self.iteration == self.first:
            self.emit('evaluation_data', dict(calibration_tokens=tensor_sha(self.calibration_batch[0]),
                heldout_tokens=[tensor_sha(b[0]) for b in self.fixed_batches],
                heldout_labels=[tensor_sha(b[1]) for b in self.fixed_batches]))

    @torch.no_grad()
    def evaluate_heldout(self, model, completed_iteration):
        saved = self.batch
        rows = []
        try:
            for batch in self.fixed_batches:
                self.batch = batch
                loss, _ = Intervention.evaluate_probe(self, model)
                rows.append(dict(loss=loss, valid_tokens=float(batch[2].sum()), tokens=tensor_sha(batch[0])))
        finally:
            self.batch = saved
        total = sum(r['valid_tokens'] for r in rows)
        self.emit('trajectory_validation', dict(iteration=completed_iteration, policy=self.policy,
            loss=sum(r['loss']*r['valid_tokens'] for r in rows)/total, batches=rows,
            scope='20 fixed held-out microbatches; separate from calibration; not a full validation benchmark'))

    @torch.no_grad()
    @geometry_precision()
    def component(self, W, M, target_radius, power_iteration_steps, msign_steps, **kwargs):
        index = self.counter
        self.counter += 1
        alpha = float(kwargs.get('current_lr') or 0.) * self.helpers.get_spectral_ball_scale_factor(
            *W.shape, mode=self.scale_mode)
        if alpha <= 0:
            raise ValueError('Expected positive-LR restored update')
        entry = self.find_input(self.parent)
        start, stop = component_rows(self.parent, W)
        context = dict(**entry, row_start=start, row_stop=stop, index=index)
        self.contexts[self.key(W)] = context
        before = entry['module'].weight[start:stop].detach().float().clone()
        x = W.float().clone()
        if not torch.equal(x.bfloat16().float(), before):
            raise RuntimeError('Master/forward weights disagree before update')
        u, s, vh = svd(x)
        if abs(float(s[0])/target_radius-1) > 5e-5:
            self.fail_radius('pre', index, W, x, None, None, target_radius, abs(float(s[0])/target_radius-1))
        d0, _ = choose_direction(M, u[:,:1]@vh[:1], 'mcsd_tp_aus', self.helpers)
        gap = float(1-s[1]/s[0])
        near = gap <= self.near_threshold
        q = int(((1-s/s[0]) <= self.near_threshold).sum())
        if near:
            point, info = selected_return(self.policy, x, M.float(), d0, u[:,:q], s[:q], vh[:q].T,
                float(target_radius), alpha, entry['calibration'], before, entry['module'].weight.dtype)
            self.observe_near_component(context, x, M.float(), d0, u, s, vh.T,
                float(target_radius), alpha, before, point, info)
            self.emit('trajectory_geometry', dict(iteration=self.iteration, component=index,
                parameter=entry['name'], gap=gap, cluster_dimension=q, cluster_truncated=False, **info))
        else:
            point = return_point(x+alpha*d0, target_radius, 'radial')
            info = dict(effective_alpha=alpha, relative_error=0., matching_satisfied=True, evaluations=0)
        info.update(near_eligible=near, component=index)
        self.return_checks.append((W, x, point, float(target_radius), index, 0., info))
        self.step_rows.append(info)
        return (x-point)/alpha, 0., float(s[0])

    @torch.no_grad()
    @geometry_precision()
    def finish_update(self):
        if os.environ.get('ELR_ASSERT_LAYOUT', '0') == '1':
            fc1 = [c for c in self.contexts.values() if c['name'].endswith('linear_fc1')]
            if len(self.contexts)!=252 or len(fc1)!=28 or any((c['row_start'],c['row_stop'])!=(0,1536) for c in fc1):
                raise RuntimeError('Component layout changed')
        if not self.return_checks:
            raise RuntimeError('No component updates')
        errors = []
        for W, _, point, radius, *_ in self.return_checks:
            c = self.contexts[self.key(W)]
            forward = c['module'].weight[c['row_start']:c['row_stop']]
            if not torch.equal(W, point) or not torch.equal(forward.float(), point.to(forward.dtype).float()):
                raise RuntimeError('Selected return not committed to master and forward weights')
            errors.append(abs(float(spectral_norm(W))/radius-1))
        if max(errors)>5e-5:
            raise RuntimeError('Radial return failed radius guard')
        near = [r for r in self.step_rows if r['near_eligible']]
        self.emit('trajectory_update', dict(iteration=self.iteration, policy=self.policy,
            components=len(errors), max_radius_error=max(errors), exact_commit=True,
            near_components=len(near), unmatched_components=sum(not r['matching_satisfied'] for r in near),
            max_budget_error=max((r['relative_error'] for r in near), default=0.),
            mean_search_calls=sum(r['evaluations'] for r in near)/len(near) if near else 0.))
        self.return_checks.clear()
        self.step_rows.clear()
        self.contexts.clear()
        self.inputs.clear()

    def install_training_hooks(self):
        import megatron.training.training as training
        owner = self
        original_train, original_step = training.train, training.train_step

        def train(forward, model, optimizer, scheduler, train_iterator, valid_iterator, *args, **kwargs):
            iterator = valid_iterator[0] if isinstance(valid_iterator, list) else valid_iterator
            if isinstance(valid_iterator, list) and len(valid_iterator)!=1:
                raise ValueError('Expected a single validation iterator')
            get_batch = forward.__globals__['get_batch']
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for _ in range(int(os.environ.get('ELR_VALID_SKIP_BATCHES', '128'))):
                    get_batch(iterator)
                for _ in range(owner.heldout_batch_count+1):
                    batch = get_batch(iterator)
                    owner.fixed_batches.append(tuple(v.detach().clone() if torch.is_tensor(v) else v for v in batch))
            return original_train(forward, model, optimizer, scheduler, train_iterator, valid_iterator, *args, **kwargs)

        def train_step(forward, iterator, model, optimizer, *args, **kwargs):
            from megatron.training import get_args
            from megatron.training.utils import unwrap_model
            owner.iteration = int(get_args().curr_iteration)+1
            owner.batch = None
            stream = []
            namespace = forward.__globals__
            old_batch, old_optimizer_step = namespace['get_batch'], optimizer.step
            begin = time.perf_counter()

            def get_batch(*a, **kw):
                batch = old_batch(*a, **kw)
                if owner.batch is None:
                    owner.batch = tuple(v.detach().clone() if torch.is_tensor(v) else v for v in batch)
                stream.append((tensor_sha(batch[0]), tensor_sha(batch[1])))
                return batch

            def update(*a, **kw):
                owner.prepare_inputs(model)
                if owner.iteration == owner.first:
                    core = unwrap_model(model)[0]
                    owner.emit('trajectory_initial', dict(iteration=owner.iteration,
                        optimizer=optimizer_signature(optimizer),
                        model=[(name,tensor_sha(p)) for name,p in core.named_parameters()],
                        data_first_tokens=tensor_sha(owner.batch[0]), data_first_labels=tensor_sha(owner.batch[1])))
                    owner.evaluate_heldout(model, owner.first-1)
                result = old_optimizer_step(*a, **kw)
                if not bool(result[0]):
                    raise RuntimeError('Skipped optimizer update')
                owner.finish_update()
                owner.after_optimizer_update(model, optimizer)
                if owner.iteration % owner.interval == 0 or owner.iteration == owner.last:
                    owner.evaluate_heldout(model, owner.iteration)
                return result

            namespace['get_batch'], optimizer.step = get_batch, update
            try:
                result = original_step(forward, iterator, model, optimizer, *args, **kwargs)
                owner.emit('training_stream', dict(iteration=owner.iteration, batches=stream,
                    seconds=time.perf_counter()-begin,
                    digest=hashlib.sha256(repr(stream).encode()).hexdigest()))
                return result
            finally:
                namespace['get_batch'], optimizer.step = old_batch, old_optimizer_step
        training.train, training.train_step = train, train_step
