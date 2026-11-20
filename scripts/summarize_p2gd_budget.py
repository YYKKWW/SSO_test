"""Paired budget comparisons; counterfactuals are not independent trajectories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from p2gd_budget_geometry import ARMS


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(directory):
    loss_path, geometry_path = directory/'repair_loss_replay.jsonl',directory/'repair_geometry.jsonl'
    loss, geometry = read(loss_path), read(geometry_path)
    completed = {r['iteration'] for r in loss}
    geometry = [r for r in geometry if r['iteration'] in completed]
    if not loss or not all(r['restore_asserted'] for r in loss):
        raise ValueError('Missing or unrestored replay')
    if len(geometry)!=sum(r['near_components'] for r in loss):
        raise ValueError('Inconsistent geometry/loss snapshot')
    informative = [r for r in loss if r['near_components']]
    names = [a['name'] for a in ARMS]
    for row in loss:
        if [a['method'] for a in row['results']]!=names:
            raise ValueError('Missing method')
        tokens = lambda a:[(b['tokens'],b['labels'],b['valid_tokens']) for b in a['heldout_batches']]
        if any(tokens(a)!=tokens(row['results'][0]) for a in row['results']):
            raise ValueError('Unpaired held-out data')
    results=[]
    for arm in ARMS:
        name,metric,fraction=(arm[k] for k in ('name','metric','fraction'))
        control = 'half_d0' if fraction==1. else metric+'_095_half_d0'
        deltas=[];absolute=[];training=[]
        for row in informative:
            by = {a['method']:a for a in row['results']}
            deltas.append(by[name]['heldout_loss']-by[control]['heldout_loss'])
            absolute.append(by[name]['heldout_difference_vs_half'])
            training.append(by[name]['training_microbatch_loss']-by[control]['training_microbatch_loss'])
        comp=[next(a for a in row['rows'] if a['method']==name) for row in geometry]
        def mean(values):
            return sum(values)/len(values) if values else None
        errors=[abs((r['returned_spectral_elr'] if metric=='spectral' else r['local_output_elr'])-r['target'])
                for r in comp] if metric!='reference' else []
        violations=sum(e>r['tolerance']+1e-8 for e,r in zip(errors,comp))
        results.append(dict(**arm,matched_control=control,informative_steps=len(deltas),
            mean_difference_vs_matched_control=mean(deltas),mean_difference_vs_original_half=mean(absolute),
            better_steps=sum(d < -1e-7 for d in deltas),worse_steps=sum(d > 1e-7 for d in deltas),
            mean_training_difference_vs_matched_control=mean(training),matching_violations=violations,
            mean_spectral_elr=mean([r['returned_spectral_elr'] for r in comp]),
            mean_calibration_output_elr=mean([r['local_output_elr'] for r in comp]),
            mean_independent_output_elr=mean([r['independent_output_elr'] for r in comp]),
            mean_bf16_spectral_elr=mean([r['bf16_spectral_elr'] for r in comp]),
            mean_bf16_output_elr=mean([r['bf16_output_elr'] for r in comp]),
            max_bf16_budget_relative_error=max((r.get('bf16_budget_relative_error',0.) for r in comp),default=None),
            mean_aus=mean([r['actual_aus'] for r in comp]),
            mean_alpha_ratio=mean([r['alpha_ratio'] for r in comp]),
            mean_search_evaluations=mean([r['evaluations'] for r in comp]),
            max_radius_error=max((r['radius_error'] for r in comp),default=None)))
    return dict(completed_steps=sorted(completed),informative_steps=len(informative),
        near_components=len(geometry),cluster_dimensions=dict(Counter(r['cluster_dimension'] for r in geometry)),
        results=results,scope='Same-state counterfactuals on one common trajectory; correlated observations',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (loss_path,geometry_path)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=summarize(args.metrics)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
