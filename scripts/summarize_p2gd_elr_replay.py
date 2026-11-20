"""Summarize completed counterfactual records without treating steps as seeds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(directory):
    loss_file = directory/'repair_loss_replay.jsonl'
    geometry_file = directory/'repair_geometry.jsonl'
    rows, geometry = read(loss_file), read(geometry_file)
    if not rows or any(not r['restore_asserted'] for r in rows):
        raise ValueError('Missing records or failed restoration')
    completed = {r['iteration'] for r in rows}
    pending_geometry = sum(r['iteration'] not in completed for r in geometry)
    geometry = [r for r in geometry if r['iteration'] in completed]
    if len(geometry) != sum(r['near_components'] for r in rows):
        raise ValueError('Geometry and loss files are not a consistent completed-step snapshot')
    informative = [r for r in rows if r['near_components']]
    methods = [a['method'] for a in rows[0]['results']]
    for r in rows:
        if [a['method'] for a in r['results']] != methods:
            raise ValueError('Incomplete counterfactual set')
        baseline_tokens = [(b['tokens'],b['labels'],b['valid_tokens']) for b in r['results'][0]['heldout_batches']]
        if any([(b['tokens'],b['labels'],b['valid_tokens']) for b in a['heldout_batches']] != baseline_tokens
               for a in r['results']):
            raise ValueError('Evaluation batches differ between counterfactuals')
    results = []
    for method in methods:
        selected = [next(a for a in r['results'] if a['method']==method) for r in informative]
        comp = [next(a for a in r['rows'] if a['method']==method) for r in geometry]
        deltas = [a['heldout_difference_vs_half'] for a in selected]
        results.append(dict(method=method, informative_steps=len(selected),
            mean_loss_difference_vs_half=sum(deltas)/len(deltas) if deltas else None,
            minimum_difference=min(deltas,default=None), maximum_difference=max(deltas,default=None),
            better_steps=sum(d < -1e-7 for d in deltas),
            worse_steps=sum(d > 1e-7 for d in deltas),
            loss_increase_steps=sum(a['heldout_delta'] > 1e-7 for a in selected),
            mean_training_microbatch_difference=sum(a['training_difference_vs_half'] for a in selected)/len(selected)
                if selected else None,
            maximum_actual_aus_relative_error=max((abs(a['actual_aus']-a['target'])/a['target'] for a in comp), default=None),
            matching_violations=sum(abs(a['actual_aus']-a['target']) > a['tolerance']+2e-6 for a in comp),
            mean_returned_spectral_elr=sum(a['returned_spectral_elr'] for a in comp)/len(comp) if comp else None,
            mean_output_elr=sum(a['local_output_elr'] for a in comp)/len(comp) if comp else None,
            mean_gradient_work=sum(a['gradient_work'] for a in comp)/len(comp) if comp else None,
            max_radius_error=max((a['radius_error'] for a in comp),default=None)))
    return dict(completed_replay_steps=[r['iteration'] for r in rows], informative_steps=len(informative),
        near_components=sum(r['near_components'] for r in rows),
        cluster_dimensions=dict(Counter(r['cluster_dimension'] for r in geometry)),
        inflight_geometry_records_excluded=pending_geometry,
        summaries=results, last_common_heldout=rows[-1]['results'][0]['heldout_loss'],
        checkpoint_scope='Same-state counterfactuals on one common trajectory; not independent training runs',
        inference_scope='Repeated evaluation batches and correlated steps; no seed-level confidence interval',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (loss_file,geometry_file) if p.exists()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = summarize(args.metrics)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
