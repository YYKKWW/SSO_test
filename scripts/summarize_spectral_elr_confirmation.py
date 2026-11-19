"""Audit downloaded array 4464129 logs and emit a portable result summary.

Usage: python scripts/summarize_spectral_elr_confirmation.py RAW_RESULTS OUTPUT
Requires only Python's standard library. Raw checkpoints are not needed.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean


def rows(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Empty evidence: {path}")
    return values


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = sorted(values)
    require(bool(values) and all(math.isfinite(v) for v in values), "Invalid measurements")
    def quantile(q):
        x = q * (len(values) - 1)
        i = int(x)
        return values[i] + (x-i) * (values[min(i+1, len(values)-1)]-values[i])
    return dict(n=len(values), mean=mean(values), p05=quantile(.05),
                median=quantile(.5), p95=quantile(.95), max=values[-1], min=values[0])


def step_summary(step, probes, components, outputs, theory):
    p = next(p for p in probes if p['iteration'] == step)
    cs = [c for c in components if c['iteration'] == step]
    os = [c for c in outputs if c['iteration'] == step]
    ts = [c for c in theory if c['iteration'] == step]
    require(len(cs) == len(ts) == 12 and len(os) == 252, 'Wrong component counts')
    result = dict(iteration=step, loss_before=p['loss_before'], loss_after=p['loss_after'],
                  batch_losses=[b['loss'] for b in p['after']],
                  tokens=[b['valid_tokens'] for b in p['after']])
    for key in ('aus', 'actual_elr_spectral', 'actual_elr_frobenius',
                'displacement_stable_rank', 'weight_stable_rank', 'gradient_work',
                'momentum_work', 'return_correction_fro', 'direction_angle_to_muon'):
        result[key] = mean(c[key] for c in cs)
    for key in ('aus', 'actual_elr_spectral'):
        result[key+'_ratio_to_own_muon'] = mean(c[key]/c['muon_reference'][key] for c in cs)
    for source in ('gradient', 'nesterov_input', 'direction'):
        result[source+'_angle_to_tangent_degrees'] = mean(
            math.degrees(math.asin(min(1., abs(c[source]['spectral_normal_cosine'])))) for c in cs)
    for split in ('calibration', 'heldout'):
        result['output_'+split+'_ratio_to_own_muon'] = mean(
            c[split]['master_output_elr']/c[split]['reference_output_elr'] for c in os)
        result['output_'+split+'_mean_abs_mismatch'] = mean(c[split]['relative_mismatch'] for c in os)
    for key in ('first_order_return_residual', 'trial_vs_return_fraction',
                'first_order_output_residual', 'gap'):
        result[key] = mean(t[key] for t in ts)
    return result


def summarize(root):
    folders = sorted(root.glob('confirm_[0-9]*'), key=lambda p: int(p.name.split('_')[1]))
    require([int(p.name.split('_')[1]) for p in folders] == list(range(13)), 'Need all 13 main arms')
    lanes = [json.loads((root/f'CONFIRM_LANE_{i}_COMPLETED.json').read_text()) for i in range(2)]
    negative = {}
    for lane in range(2):
        for key in ('negative_control', 'heldout_probe'):
            paths = [root/f'lane{lane}_negative_{i}/metrics/{key}.jsonl' for i in (0, 13)]
            records = [[{k: v for k, v in r.items() if k != 'method'} for r in rows(p)] for p in paths]
            require(len(records[0]) == 3 and records[0] == records[1], 'Negative control mismatch')
        negative[str(lane)] = 'Three-step component hashes and heldout results identical'
    first = {}
    records = []
    manifests = {}
    for folder in folders:
        require((folder/'COMPLETED').exists(), f'Incomplete: {folder.name}')
        m = folder/'metrics'
        data = {p.stem: rows(p) for p in m.glob('*.jsonl')}
        for path in sorted([*m.glob('*.jsonl'), folder/'train.log', folder/'launch_config.json', folder/'wall_seconds.json']):
            manifests[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        probes = data['heldout_probe']
        require([p['iteration'] for p in probes] == list(range(253, 285)), 'Wrong update sequence')
        require(all(len(p['after']) == len(p['before']) == 20 for p in probes), 'Wrong evaluation size')
        for probe in probes:
            for phase in ('before', 'after'):
                batches = probe[phase]
                weighted = sum(b['loss']*b['valid_tokens'] for b in batches)/sum(b['valid_tokens'] for b in batches)
                require(math.isfinite(weighted) and abs(weighted-probe['loss_'+phase]) < 1e-10,
                        'Evaluation loss does not match token-weighted microbatches')
        # Match only the common starting state; later optimization states intentionally diverge.
        for key in ('state', 'optimizer_before', 'heldout_probe', 'model_probe', 'activation_inputs'):
            selected = [r for r in data[key] if r['iteration'] == 253]
            if key == 'heldout_probe': selected = [r['before'] for r in selected]
            if key == 'model_probe': selected = [(r['token_sha256'], r['label_sha256'], r['loss_before']) for r in selected]
            require(bool(selected), 'Missing initial pairing evidence')
            first.setdefault(key, selected)
            require(selected == first[key], f'Initial pairing failed: {key}, {folder.name}')
        streams = [(r['iteration'], r['token_sha256'], r['label_sha256']) for r in data['model_probe']]
        first.setdefault('training_stream', streams)
        require(streams == first['training_stream'], 'Training batches differ')
        evaluation = [[(b['tokens'], b['labels'], b['valid_tokens']) for b in p['after']] for p in probes]
        first.setdefault('evaluation_stream', evaluation)
        require(evaluation == first['evaluation_stream'], 'Evaluation batches differ')
        control_rows = data['confirmation_controls']
        require(len(control_rows) == 32 and all(r['exact_commit'] and len(r['controls']) == 252 for r in control_rows), 'Commit check missing')
        require(all(sorted(c['component_index'] for c in r['controls']) == list(range(252))
                    for r in control_rows), 'Duplicate or missing components')
        controls = [c for r in control_rows for c in r['controls']]
        require(sum(r['components'] for r in data['all_component_returns']) == 8064, 'Incomplete returns')
        require(all(r['optimizer_update_successful'] for r in data['model_probe']), 'Skipped update')
        timing = json.loads((folder/'wall_seconds.json').read_text())
        require(timing['exit_code'] == 0, 'Nonzero exit')
        log = (folder/'train.log').read_text()
        for kind in ('skipped', 'nan'):
            counts = [int(v) for v in re.findall(r'number of '+kind+r' iterations:\s*(\d+)', log)]
            require(bool(counts) and max(counts) == 0, f'{kind} log check failed')
        theory = data['theory_geometry']
        record = dict(name=folder.name.split('_', 2)[2], wall_seconds=timing['seconds'],
            initial_loss=probes[0]['loss_before'], final_loss=probes[-1]['loss_after'],
            return_count=8064, max_radius_error=max(r['max_radius_error'] for r in data['all_component_returns']),
            cap_active_fraction=mean(c['effective_alpha'] < c['uncapped_alpha']*(1-1e-8) for c in controls) if controls[0]['cap'] else None,
            coefficient_ratio=stats([c['effective_alpha']/c['uncapped_alpha'] for c in controls]),
            mean_calibration_evaluations=mean(c['mean_evaluations'] for c in data['calibration']),
            max_enforced_matching_error=max((r['max_match_relative_error'] or 0) for r in data['all_component_returns']),
            theory={key: stats([r[key] for r in theory]) for key in ('first_order_return_residual',
                'first_order_output_residual', 'trial_vs_return_fraction', 'gap', 'alpha_over_absolute_gap')},
            oblique_identity_max_absolute_error=max(abs(r['oblique_orthogonal_difference']-
                r['predicted_oblique_orthogonal_difference']) for r in theory),
            by_step=[step_summary(s, probes, data['components'], data['output_components'], theory) for s in range(253, 285)])
        records.append(record)
    reference = records[0]
    for r in records:
        differences = [s['loss_after']-b['loss_after'] for s, b in zip(r['by_step'], reference['by_step'])]
        r['final_delta_vs_muon'] = differences[-1]
        r['mean_trajectory_delta_vs_muon'] = mean(differences)
        r['first_delta_vs_muon'] = differences[0]
        r['last_per_batch_delta_vs_muon'] = [a-b for a, b in zip(r['by_step'][-1]['batch_losses'], reference['by_step'][-1]['batch_losses'])]
        r['better_final_batches'] = sum(v < 0 for v in r['last_per_batch_delta_vs_muon'])
        r['better_steps'] = sum(v < 0 for v in differences)
    return dict(date='2026-11-19', array='4464129', lanes=lanes, negative_controls=negative,
        initial_pairing={k: True for k in first}, width=256, seed=1234,
        checkpoint_step=252, updates_per_arm=32, tokens_per_arm=16777216,
        heldout_microbatches=20, heldout_tokens=327680,
        scope='Mechanism interventions with common exact radial return, not native optimizer rankings',
        component_sampling='12 for spectral/AUS/angles/theory; all 252 for output/caps/returns',
        arms=records, source_sha256=manifests,
        limitations=['One checkpoint and seed; fixed microbatches are not independent training replicates',
            'Fresh relative to the earlier diagnostic; now used for analysis, so not an untouched final test set',
            'Own-state Muon targets after the common starting step; momentum trajectories diverge',
            'Local linear output matching is not full-network output matching',
            'Effective FC1 remains unsplit (252 matrix components)',
            'Timings include diagnostic exact SVD, searches, probes and repeated evaluation',
            'No independent later-checkpoint replication; no 1B/3B endpoint claim'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw_results', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = summarize(args.raw_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for r in result['arms']:
        print(f"{r['name']:32} {r['final_loss']:.9f} {r['final_delta_vs_muon']:+.9f} {r['better_final_batches']}/20 {r['wall_seconds']:.1f}s")


if __name__ == '__main__':
    main()
