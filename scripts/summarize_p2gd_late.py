"""Later-window comparisons, including rounded-budget errors and repair strength."""
from collections import Counter

from p2gd_late_geometry import ARMS
from summarize_p2gd_budget import read, summarize


def late_summary(metrics):
    result=summarize(metrics,arms=ARMS,analysis_first=513)
    loss=[r for r in read(metrics/'repair_loss_replay.jsonl') if r['iteration']>=513]
    geometry=[r for r in read(metrics/'repair_geometry.jsonl') if r['iteration']>=513]
    accurate={r['iteration'] for r in loss}
    for r in geometry:
        if any(row['metric']!='reference' and row['relative_error']>.01 for row in r['rows']):
            accurate.discard(r['iteration'])
    comparisons=[]
    for control in ('ambient','top1','half_output_of_cluster','half_d0'):
        for subset in ('all_informative','forward_budget_within_1pct'):
            chosen=[r for r in loss if r['near_components'] and
                (subset=='all_informative' or r['iteration'] in accurate)]
            changes=[];training=[]
            for r in chosen:
                by={a['method']:a for a in r['results']}
                changes.append(by['cluster_cone']['heldout_loss']-by[control]['heldout_loss'])
                training.append(by['cluster_cone']['training_microbatch_loss']-by[control]['training_microbatch_loss'])
            comparisons.append(dict(candidate='cluster_cone',control=control,subset=subset,steps=len(changes),
                mean_heldout_difference=sum(changes)/len(changes) if changes else None,
                mean_training_difference=sum(training)/len(training) if training else None,
                better_steps=sum(v < -1e-7 for v in changes),worse_steps=sum(v > 1e-7 for v in changes)))
    mechanisms=[]
    for name in ('ambient','top1','cluster_cone'):
        rows=[a for r in geometry for a in r['rows'] if a['method']==name]
        def avg(key):
            values=[r[key] for r in rows if r[key] is not None]
            return sum(values)/len(values) if values else None
        mechanisms.append(dict(method=name,components=len(rows),rank_counts=dict(Counter(r['correction_rank'] for r in rows)),
            mean_correction_fraction=avg('correction_fro_over_momentum_direction'),
            mean_momentum_angle=avg('momentum_direction_angle_degrees'),
            mean_top_block_angle=avg('momentum_top_block_angle_degrees'),
            mean_bf16_difference_from_ambient=avg('bf16_delta_vs_ambient_fro_fraction'),
            mean_bf16_changed_fraction_from_ambient=avg('bf16_changed_fraction_vs_ambient'),
            mean_rounding_fraction=avg('bf16_rounding_displacement_fraction')))
    result.update(comparisons=comparisons,mechanisms=mechanisms,matching_space='BF16 forward weights',
        matching_note='Unattainable matches are reported, never silently accepted; 1% subset fixed before results',
        scope='16 counterfactual states after one shared continuation; not independent trajectories')
    return result
