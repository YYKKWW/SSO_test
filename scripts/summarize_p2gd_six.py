"""Predeclared six-way comparisons on steps 269--300; no seed-level inference."""
import argparse
import json
from pathlib import Path

from p2gd_six_geometry import ARMS
from summarize_p2gd_budget import read, summarize

PAIRS = [(candidate, control) for candidate in ('cluster_cone', 'finite_momentum')
         for control in ('ambient', 'top1', 'half_d0')]
PAIRS += [('cluster_cone', 'half_output_of_cluster'), ('finite_momentum', 'cluster_cone')]


def six_summary(directory):
    result = summarize(directory, arms=ARMS, analysis_first=269)
    loss = [r for r in read(directory/'repair_loss_replay.jsonl') if r['iteration'] >= 269]
    comparisons = []
    for candidate, control in PAIRS:
        all_delta, active_delta, train_delta = [], [], []
        for row in loss:
            by = {r['method']:r for r in row['results']}
            value = by[candidate]['heldout_loss']-by[control]['heldout_loss']
            all_delta.append(value)
            if row['near_components']:
                active_delta.append(value)
                train_delta.append(by[candidate]['training_microbatch_loss']-by[control]['training_microbatch_loss'])
        mean = lambda xs: sum(xs)/len(xs) if xs else None
        comparisons.append(dict(candidate=candidate, control=control,
            mean_informative_loss_difference=mean(active_delta),
            mean_all_step_loss_difference=mean(all_delta),
            mean_training_microbatch_difference=mean(train_delta),
            better_steps=sum(v < -1e-7 for v in active_delta),
            worse_steps=sum(v > 1e-7 for v in active_delta),
            scope='output-matched' if control=='half_output_of_cluster' else 'spectral-matched'))
    result.update(comparisons=comparisons, prefix_steps=list(range(253,269)),
        primary_window='269--300 fixed before seeing loss',
        caveat='Common trajectory, not independent runs; numerical clusters, not exact multiplicity')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args(); result=six_summary(args.metrics)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
