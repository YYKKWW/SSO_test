"""Verify later-start paired paths and describe bounded snapshot diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_p2gd_trajectories import read, summarize, plot

POLICIES=('half_d0','top1','cluster_cone','half_output_of_cluster')


def describe(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Missing or nonfinite audit values')
    return dict(mean=statistics.mean(values),median=statistics.median(values),max=max(values))


def snapshot_summary(path):
    source=json.loads(path.read_text());rows=source['results']
    if not rows or len(rows)!=source['snapshots'] or len(rows)>24:
        raise ValueError('Missing or inconsistent snapshot count')
    if len({r['snapshot'] for r in rows})!=len(rows):
        raise ValueError('Duplicate snapshot')
    capacities={}
    for name in ('top4','top6','full'):
        selected=[]
        for r in rows:
            matches=[c for c in r['capacities'] if c['name']==name]
            if len(matches)!=1: raise ValueError('Invalid capacity cases')
            c=matches[0]
            if c['used']!=min(c['capacity'],r['q']) or c['omitted']!=r['q']-c['used']:
                raise ValueError('Invalid capacity/omission count')
            if name=='full' and c['omitted']!=0: raise ValueError('Incomplete full cluster')
            selected.append(c)
        capacities[name]=dict(omitted_snapshots=sum(c['omitted']>0 for c in selected),
            point_relative_error=describe([c['point_relative_error'] for c in selected]),
            radius_error=describe([c['radius_error'] for c in selected]))
    if capacities['full']['point_relative_error']['max']>1e-6:
        raise ValueError('Full recomputation differs from committed return')
    approximations=[]
    for block in (8,12):
        for k in (10,20):
            selected=[]
            for r in rows:
                candidates=[c for c in r['approximate_returns'] if c['block']==block and c['iterations']==k]
                if len(candidates)!=1: raise ValueError('Missing approximation case')
                selected.append(candidates[0])
            seconds=[c['seconds'] for c in selected]
            if min(seconds)<=0: raise ValueError('Nonpositive timing')
            approximations.append(dict(block=block,iterations=k,
                radius_error=describe([c['radius_error'] for c in selected]),
                gap_error=describe([c['gap_error'] for c in selected]),
                point_relative_error=describe([c['point_relative_error'] for c in selected]),
                seconds=describe(seconds),sum_svd_time_over_sum_approx_time=
                    sum(r['svdvals_seconds'] for r in rows)/sum(seconds)))
    return dict(snapshots=len(rows),cluster_dimensions=dict(Counter(r['q'] for r in rows)),
        categories=dict(Counter(r['category'] for r in rows)),capacities=capacities,
        approximations=approximations,svdvals_seconds=describe([r['svdvals_seconds'] for r in rows]),
        initialization=source['initialization'],scope=source['scope'],timing=source['timing'],
        limitations='Stratified diagnostic sample; no independent persistent-cache or truncated-capacity training',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def analyze(root):
    gate=json.loads((root/'PREFIX_READY.json').read_text())
    if not all(gate[k] for k in ('verified','exact_post_save_model_and_optimizer_restored','new_evaluation_batches')):
        raise ValueError('Restore gate failed')
    result=summarize(root,first=769,last=1024,policies=POLICIES,receipt='REPLICATION_COMPLETED.json')
    for r in result['results']:
        rows=read(root/('branch_'+r['policy'])/'metrics/trajectory_geometry.jsonl')
        updates=read(root/('branch_'+r['policy'])/'metrics/trajectory_update.jsonl')
        count=sum(not p.get('matching_satisfied',True) for p in rows)
        error=max(p['relative_error'] for p in rows)
        if count!=r['unmatched_components'] or abs(error-r['max_budget_error'])>1e-12:
            raise ValueError('Budget statistics differ from raw geometry')
        if count!=sum(p['unmatched_components'] for p in updates):
            raise ValueError('Update mismatch count differs')
        r['budget_error']=describe([p['relative_error'] for p in rows])
        r['geometry']['exceeds_top6']=sum(p['cluster_dimension']>6 for p in rows)
    cluster=next(r for r in result['results'] if r['policy']=='cluster_cone')
    for c in result['cluster_contrasts']:
        ref=next(r for r in result['results'] if r['policy']==c['control'])
        c['lower_loss_evaluations_excluding_initial']=sum(a['loss']<b['loss']
            for a,b in zip(cluster['curve'][1:],ref['curve'][1:]))
        c['post_update_evaluations']=len(cluster['curve'])-1
    result['snapshot_audit']=snapshot_summary(root/'SNAPSHOT_AUDIT_COMPLETED.json')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args();result=analyze(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.figure: plot(result,args.figure)
    print(json.dumps(result['cluster_contrasts'],indent=2))
