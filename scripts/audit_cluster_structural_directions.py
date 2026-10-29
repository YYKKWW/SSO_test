"""Offline structural ablation with frozen parameters and recorded LLM matrices.

The candidate preserves the cached spectral separations in a small symmetric
trial model. It is neither a full matrix projection nor a convergence theorem.
No code in this file is imported by training.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Megatron-LM"))
from emerging_optimizers.orthogonalized_optimizers import \
    spel_pgd_same_projection as host
from emerging_optimizers.orthogonalized_optimizers.warm_block_ritz import \
    warm_block8_ritz


def gap(s):
    return float(1 - s[1] / s[0])


def rel_error(actual, reference):
    return float(torch.linalg.vector_norm(actual - reference) /
                 torch.linalg.vector_norm(reference).clamp_min(1e-30))


@torch.no_grad()
def gap_aware_trial_step(raw_step, u, s, v, radius):
    """Frobenius-correct the small affine model to lambda_max(B)=R.

    B = diag(s) + sym(U.T @ raw_step @ V). Clamping its eigenvalues to R,
    with the largest set to R, preserves existing gaps instead of treating
    every selected singular value as exactly R. The skew block and other
    directions are unchanged; their effect still requires a radial return.
    """
    if raw_step.ndim != 2 or len(s) < 1 or radius <= 0:
        raise ValueError("expected a matrix, nonempty spectrum and positive radius")
    a = u.T @ raw_step @ v
    a = (a + a.T) * .5
    b = torch.diag(s) + a
    values, vectors = torch.linalg.eigh(b)
    corrected = values.clamp(max=radius)
    corrected[-1] = radius
    projected = (vectors * corrected) @ vectors.T
    return raw_step + u @ (projected - b) @ v.T


def make_steps(m, u, s, v, radius, alpha, cluster_tolerance):
    active = int(((s[0] - s) / s[0] <= cluster_tolerance).sum())
    q = max(1, min(4, len(s), active))
    z = -m / torch.linalg.vector_norm(m).clamp_min(1e-12)
    uq, vq = u[:, :q], v[:, :q]
    return dict(
        flat_cluster=alpha * host.project_spectral_sphere_cone(z, uq, vq),
        gap_aware=gap_aware_trial_step(alpha * z, uq, s[:q], vq, radius),
        tangent1=alpha * host.project_to_tangent_plane(z, u[:, :1], v[:, :1]),
        ambient=alpha * z,
    ), q, active


@torch.no_grad()
def audit_sample(sample):
    x, m = sample["current"].float(), sample["momentum"].float()
    u, s, v = (sample[k].float() for k in ("current_u", "current_s", "current_v"))
    radius = float(sample["radius"])
    alpha = float(sample["effective_lr"]) * 8.
    threshold = float(sample["gap_threshold"])
    oracle_u, oracle_s, oracle_vh = host.accurate_svd_fp32(x)
    estimated_near = sample["current_gap"] <= threshold
    true_near = gap(oracle_s) <= threshold
    base = dict(step=sample["step"], shape=list(x.shape), selection=sample["selection"],
                source_mode=sample["cluster_mode"], source_branch=sample["branch"],
                estimated_gap=float(sample["current_gap"]), true_gap=gap(oracle_s),
                estimated_near=estimated_near, true_near=true_near,
                current_radius_rel_error=float(oracle_s[0] / radius - 1),
                effective_lr=float(sample["effective_lr"]), coefficient=8., arms=[])
    if not estimated_near:
        return base
    steps, q, active = make_steps(m, u, s, v, radius, alpha, sample["cluster_threshold"])
    base.update(q=q, estimated_cluster_size=active,
                true_cluster_size=int(((oracle_s[0]-oracle_s)/oracle_s[0] <= sample["cluster_threshold"]).sum()))
    returns = {}
    for name, step in steps.items():
        y = x + step
        _, estimated_s, _ = warm_block8_ritz(y, iterations=10, initial_v=v)
        true_s = host.accurate_svdvals_fp32(y)
        returned = y * (radius / estimated_s[0])
        exact_returned = y * (radius / true_s[0])
        displacement, exact_displacement = returned-x, exact_returned-x
        returns[name] = displacement
        inner = float((m * displacement).sum())
        exact_inner = float((m * exact_displacement).sum())
        base["arms"].append(dict(
            method=name,
            predicted_momentum_change=float((m * step).sum()),
            returned_momentum_change=inner, exact_return_momentum_change=exact_inner,
            momentum_cosine=float((m * displacement).sum() /
                (torch.linalg.vector_norm(m)*torch.linalg.vector_norm(displacement)).clamp_min(1e-30)),
            relative_displacement=float(torch.linalg.vector_norm(displacement) /
                                        torch.linalg.vector_norm(x).clamp_min(1e-30)),
            radial_correction_ratio=float(torch.linalg.vector_norm(returned-y) /
                                         torch.linalg.vector_norm(step).clamp_min(1e-30)),
            radius_rel_error=float(true_s[0]/estimated_s[0]-1),
            true_post_gap=gap(true_s), estimated_post_gap=gap(estimated_s),
            exits_near=gap(true_s) > threshold,
            gap_abs_error=abs(gap(true_s)-gap(estimated_s)),
        ))
    base['candidate_vs_flat_displacement_relative_change'] = rel_error(returns['gap_aware'], returns['flat_cluster'])
    base['candidate_vs_top1_displacement_relative_change'] = rel_error(returns['gap_aware'], returns['tangent1'])
    # Same spectrum, momentum, and alpha: measure only the local direction construction.
    timings = []
    for _ in range(3):
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        make_steps(m, u, s, v, radius, alpha, sample["cluster_threshold"])
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        timings.append((time.perf_counter()-start)*1000)
    base['four_directions_median_ms'] = statistics.median(timings)
    return base


def summarize(rows):
    near = [r for r in rows if r['arms']]
    out = dict(samples=len(rows), evaluated_near_samples=len(near),
               false_near=sum(r['estimated_near'] and not r['true_near'] for r in rows),
               false_safe=sum(r['true_near'] and not r['estimated_near'] for r in rows), arms={})
    if not near:
        return out
    out['true_cluster_exceeds4'] = sum(r['true_cluster_size'] > 4 for r in near)
    out['true_cluster_exceeds8'] = sum(r['true_cluster_size'] > 8 for r in near)
    out['mean_candidate_vs_flat_displacement_change'] = statistics.mean(r['candidate_vs_flat_displacement_relative_change'] for r in near)
    out['mean_candidate_vs_top1_displacement_change'] = statistics.mean(r['candidate_vs_top1_displacement_relative_change'] for r in near)
    for name in ('flat_cluster', 'gap_aware', 'tangent1', 'ambient'):
        arms = [next(a for a in r['arms'] if a['method'] == name) for r in near]
        out['arms'][name] = dict(
            exit_count=sum(a['exits_near'] for a in arms),
            positive_returned_momentum_change=sum(a['returned_momentum_change'] > 0 for a in arms),
            mean_returned_momentum_change=statistics.mean(a['returned_momentum_change'] for a in arms),
            mean_exact_return_momentum_change=statistics.mean(a['exact_return_momentum_change'] for a in arms),
            mean_momentum_cosine=statistics.mean(a['momentum_cosine'] for a in arms),
            median_relative_displacement=statistics.median(a['relative_displacement'] for a in arms),
            median_radial_correction_ratio=statistics.median(a['radial_correction_ratio'] for a in arms),
            max_abs_radius_error=max(abs(a['radius_rel_error']) for a in arms),
        )
    changes = []
    for r in near:
        by = {a['method']: a for a in r['arms']}
        changes.append(by['gap_aware']['returned_momentum_change'] - by['flat_cluster']['returned_momentum_change'])
    out['candidate_better_momentum_linear_model'] = sum(d < -1e-12 for d in changes)
    out['candidate_worse_momentum_linear_model'] = sum(d > 1e-12 for d in changes)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob('sample_*.pt')):
            sample = torch.load(path, map_location=args.device, weights_only=True)
            row = audit_sample(sample)
            row.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
    if not rows:
        raise RuntimeError('No recorded matrices found')
    report = dict(summary=summarize(rows), rows=rows, torch_version=torch.__version__,
                  note='Offline selected matrices, not population rates. K10/block8/q<=4/c8/gap5e-5/cluster2e-3 are fixed. Exact SVD is an offline oracle only. Momentum inner products are not measured loss changes or convergence guarantees. Gap-aware correction projects a small symmetric affine model, not the full trial matrix.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
