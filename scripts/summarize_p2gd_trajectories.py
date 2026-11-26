"""Audit completed short branches and publish compact descriptive statistics."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

POLICIES = ('half_d0', 'ambient', 'top1', 'cluster_cone', 'half_output_of_cluster')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def geometry_statistics(rows, first=513, last=768):
    keys = [(r['iteration'], r['component']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate component update')
    last_seen, runs, streak = {}, [], {}
    repeated = 0
    for r in sorted(rows, key=lambda r: (r['iteration'], r['component'])):
        key, step = r['component'], r['iteration']
        if last_seen.get(key) == step - 1:
            repeated += 1
            streak[key] += 1
        else:
            if key in streak:
                runs.append(streak[key])
            streak[key] = 1
        last_seen[key] = step
    runs.extend(streak.values())
    fields = ('bf16_spectral_elr', 'bf16_output_elr', 'alpha_ratio',
              'correction_fraction', 'momentum_angle', 'evaluations', 'norm_svd_calls')
    means = {k: mean(r[k] for r in rows if r.get(k) is not None)
             for k in fields if any(r.get(k) is not None for r in rows)}
    return dict(near_components=len(rows), repeated_near_events=repeated,
        repeat_fraction=repeated/len(rows) if rows else 0.,
        mean_observed_streak=mean(runs) if runs else 0, max_observed_streak=max(runs, default=0),
        streak_scope=f'Truncated to steps {first}--{last}; not complete lifetime residence times',
        cluster_dimensions=dict(Counter(r['cluster_dimension'] for r in rows)),
        exceeds_top4=sum(r['cluster_dimension'] > 4 for r in rows),
        correction_ranks=dict(Counter(r['correction_rank'] for r in rows if r.get('correction_rank') is not None)),
        over_one_percent_budget=sum(r['relative_error'] > .01 for r in rows),
        reference_cluster_unmatched=sum(not r.get('reference_cluster_match', {}).get('matching_satisfied', True) for r in rows),
        means=means)


def summarize(root, *, first=513, last=768, policies=POLICIES,
              receipt='TRAJECTORIES_COMPLETED.json'):
    source = json.loads((root/receipt).read_text())
    if not source['paired_data'] or not source['paired_initial_state']:
        raise ValueError('Remote pairing gate failed')
    published = {r['policy']: r for r in source['results']}
    if set(published)!=set(policies) or len(source['results'])!=len(policies):
        raise ValueError('Missing or duplicate policy in completion receipt')
    if ('first' in source and source['first']!=first) or ('last' in source and source['last']!=last):
        raise ValueError('Completion receipt has a different iteration window')
    results, hashes, reference = [], {}, None
    for policy in policies:
        folder = root/('branch_'+policy)/'metrics'
        names = ('trajectory_validation', 'trajectory_update', 'trajectory_geometry',
                 'training_stream', 'trajectory_initial', 'evaluation_data')
        data = {name: read(folder/(name+'.jsonl')) for name in names}
        for name in names:
            path = folder/(name+'.jsonl')
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        updates = data['trajectory_update']
        curves = data['trajectory_validation']
        geometry = data['trajectory_geometry']
        if [r['iteration'] for r in updates] != list(range(first,last+1)):
            raise ValueError('Incomplete branch')
        if not all(r['components']==252 and r['exact_commit'] and r['max_radius_error']<=5e-5 for r in updates):
            raise ValueError('Invalid full returns')
        expected=sorted({first-1,last}|set(range(((first+31)//32)*32,last+1,32)))
        if [r['iteration'] for r in curves] != expected:
            raise ValueError('Incomplete evaluation curve')
        pairing = dict(initial=data['trajectory_initial'], evaluation=data['evaluation_data'],
            stream=[(r['iteration'],r['batches']) for r in data['training_stream']], initial_loss=curves[0]['loss'])
        if reference is None:
            reference = pairing
        if pairing != reference:
            raise ValueError('Unpaired initial state, data or initial loss')
        if [step for step,_ in pairing['stream']] != list(range(first,last+1)) or any(len(b)!=32 for _,b in pairing['stream']):
            raise ValueError('Incomplete microbatch stream')
        if len(data['trajectory_initial']) != 1 or len(data['evaluation_data']) != 1:
            raise ValueError('Missing or duplicate initial audit')
        tokens = data['evaluation_data'][0]['heldout_tokens']
        if len(set(tokens)) != 20 or data['evaluation_data'][0]['calibration_tokens'] in tokens:
            raise ValueError('Invalid calibration/evaluation separation')
        for r in curves:
            if [b['tokens'] for b in r['batches']] != tokens:
                raise ValueError('Changed held-out batches')
            weighted = sum(b['loss']*b['valid_tokens'] for b in r['batches'])/sum(b['valid_tokens'] for b in r['batches'])
            if not math.isfinite(weighted) or not math.isfinite(r['loss']) or abs(weighted-r['loss'])>1e-12:
                raise ValueError('Incorrect weighted loss')
        stats = geometry_statistics(geometry,first,last)
        if stats['near_components'] != sum(r['near_components'] for r in updates):
            raise ValueError('Missing geometry rows')
        if Counter(r['iteration'] for r in geometry) != Counter({r['iteration']:r['near_components'] for r in updates if r['near_components']}):
            raise ValueError('Misassigned per-step geometry')
        final, tail = curves[-1]['loss'], mean(r['loss'] for r in curves[-4:])
        old = published[policy]
        if abs(final-old['final_loss'])>1e-12 or abs(tail-old['last_four_mean_loss'])>1e-12:
            raise ValueError('Published endpoint differs from raw evaluations')
        results.append(dict(policy=policy, final_loss=final, last_four_mean_loss=tail,
            near_rate=stats['near_components']/(252*(last-first+1)), geometry=stats,
            unmatched_components=old['unmatched_components'], max_budget_error=old['max_budget_error'],
            max_radius_error=old['max_radius_error'], wall_seconds=old['wall_seconds']['seconds'],
            mean_step_seconds=old['mean_step_seconds'], first32_mean_step_seconds=old['first32_mean_step_seconds'],
            curve=[dict(iteration=r['iteration'],loss=r['loss']) for r in curves]))
    cluster = next(r for r in results if r['policy']=='cluster_cone')
    contrasts = [dict(control=r['policy'], final_cluster_minus_control=cluster['final_loss']-r['final_loss'],
        tail_cluster_minus_control=cluster['last_four_mean_loss']-r['last_four_mean_loss'],
        training_wall_ratio=cluster['wall_seconds']/r['wall_seconds']) for r in results if r!=cluster]
    return dict(verified=True, first=first,last=last,updates_per_arm=last-first+1, initial_loss=reference['initial_loss'],
        results=results, cluster_contrasts=contrasts, source_sha256=hashes,
        scope='One shared checkpoint and seed; independently evolving optimizer states, not independent seeds',
        budget_scope='Same own-state matching rule; not identical absolute displacements after divergence')


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = dict(zip(POLICIES,zip(
        ('Half-D0','Ambient','Top-1','Cluster cone','Output-matched half-D0'),
        ('#555555','#dc8b16','#298f89','#bc374d','#466aba'))))
    fig, axes = plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
    baseline = [p['loss'] for p in result['results'][0]['curve']]
    for row in result['results']:
        label,color=styles[row['policy']]
        x = [p['iteration'] for p in row['curve']]
        y = [p['loss'] for p in row['curve']]
        axes[0].plot(x,y,'o-',ms=3,lw=1.5,label=label,color=color)
        axes[1].plot(x,[a-b for a,b in zip(y,baseline)],'o-',ms=3,lw=1.5,color=color)
    axes[0].set_ylabel('Fixed held-out loss')
    axes[1].set_ylabel('Loss difference vs half-D0')
    axes[1].axhline(0,color='#999999',lw=.7)
    for ax in axes:
        ax.set_xlabel('Training iteration')
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,frameon=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    result=summarize(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.figure:
        plot(result,args.figure)
    print(json.dumps([{k:r[k] for k in ('policy','final_loss','last_four_mean_loss','near_rate')} for r in result['results']],indent=2))
