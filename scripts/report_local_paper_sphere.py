"""Summarize exact local paired tests without treating replay states as seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize_local(folder):
    trajectories = json.loads((folder/'trajectories.json').read_text())
    replay = json.loads((folder/'matched.json').read_text())
    states = defaultdict(dict)
    for r in replay['rows']:
        key = tuple(r[k] for k in ('n','q','exact','seed','cosine','alpha'))
        states[key][r['method']] = r
    pairs = [('cluster_cone', other) for other in ('half_d0','half_output_of_cluster','ambient','top1')]
    pairs += [('finite_momentum','cluster_cone')]
    comparisons = []
    rng = np.random.default_rng(20261122)
    for cosine in (1., .5):
        for region in ('all_near', 'true_corner', 'near_simple', 'smooth_q1'):
            chosen = [(key, by) for key, by in states.items() if key[4] == cosine
                and ((key[1]>=2) if region=='all_near' else
                     (key[2] and key[1]>=2) if region=='true_corner' else
                     (not key[2] and key[1]>=2) if region=='near_simple' else key[1]==1)]
            for candidate, control in pairs:
                values, by_seed = [], defaultdict(list)
                for key, by in chosen:
                    difference = by[candidate]['loss_change']-by[control]['loss_change']
                    values.append(difference)
                    by_seed[key[3]].append(difference)
                seed_means = np.array([np.mean(v) for v in by_seed.values()])
                boot = rng.choice(seed_means, (4000,len(seed_means)), replace=True).mean(1)
                lo, hi = np.quantile(boot,[.025,.975])
                comparisons.append(dict(cosine=cosine, region=region, candidate=candidate, control=control,
                    count=len(values), mean_difference=float(np.mean(values)),
                    seed_bootstrap_95=[float(lo),float(hi)],
                    wins=sum(v < -1e-10 for v in values), losses=sum(v>1e-10 for v in values)))
    descent = []
    for cosine in (1., .5):
        for method in next(iter(states.values())):
            group = [r for r in replay['rows'] if r['cosine']==cosine and r['q']>=2 and r['method']==method]
            descent.append(dict(cosine=cosine, method=method, count=len(group),
                loss_decreases=sum(r['loss_change'] < -1e-10 for r in group),
                loss_increases=sum(r['loss_change'] > 1e-10 for r in group),
                mean_change=float(np.mean([r['loss_change'] for r in group]))))
    groups = defaultdict(list)
    for r in trajectories:
        groups[(r['n'],r['horizon'],r['noise_rms'],r['method'])].append(r)
    aggregate = [dict(n=n,horizon=t,noise=noise,method=method,count=len(rows),
        mean_final_excess=float(np.mean([r['final_excess'] for r in rows])),
        mean_final_residual=float(np.mean([r['final_residual'] for r in rows])),
        mean_best_limiting_residual_squared=float(np.mean([r['best_residual_squared'] for r in rows])) if n==2 else None)
        for (n,t,noise,method), rows in groups.items()]
    integrity = dict(trajectories=len(trajectories), matched_states=len(states),
        matching_failures=len(replay['failures']),
        max_radius_error=max(r['max_radius_error'] for r in trajectories),
        max_safe_perturbation_over_delta=max(r['max_safe_perturbation_over_delta'] for r in trajectories),
        max_matching_radius_error=max(r['radius_error'] for r in replay['rows']),
        max_loss_decomposition_error=max(abs(r['loss_change']-r['true_gradient_work']-r['curvature_cost']) for r in replay['rows']),
        max_matching_relative_error=max(r['relative_error'] for r in replay['rows']))
    result = dict(integrity=integrity, comparisons=comparisons, descent=descent, trajectories=aggregate,
        inference='Bootstrap resamples complete seed groups across related local problems; synthetic, not LLM evidence.',
        deterministic_square_note='Zero-noise square seeds are identical replicas, not independent problem instances.',
        rate_note='Two horizons and conservative steps diagnose implementation, not establish the asymptotic rate.')
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    summary=summarize_local(args.directory)
    print(json.dumps(summary,indent=2))
