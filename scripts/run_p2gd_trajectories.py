"""Create one shared checkpoint, verify restore, then run five short branches."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from run import checkpoint_manifest, run
from p2gd_trajectory_geometry import POLICIES
from verify_p2gd_late import verify_initial_prefix
from verify_p2gd_six import read, verify_prefix_state


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def branch_settings(root, checkpoint, policy, last=768):
    old = Path(os.environ['ELR_REFERENCE_ROOT'])
    return dict(LOAD_CHECKPOINT=str(checkpoint),CHECKPOINT_PATH=str(checkpoint),SAVE_CHECKPOINT='0',
        DATA_CACHE_PATH=str(old/'data_cache'),EVAL_INTERVAL='10000',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_trajectory.py'),
        P2GD_POLICY=policy,ELR_FIRST='513',ELR_LAST=str(last),EXIT_INTERVAL=str(last),
        ELR_ACTIVATION_TOKENS='256',ELR_HELDOUT_BATCHES='20',ELR_FIXED_VALID_BATCHES='21',
        ELR_VALID_SKIP_BATCHES='128',ELR_COMMON_PREFIX='0',ELR_ASSERT_LAYOUT='1',
        ELR_NEAR_THRESHOLD='2e-4',TRANSFORMER_IMPL='local')


def verify_branch(metrics, last=768, first=513):
    updates=read(metrics/'trajectory_update.jsonl')
    if [r['iteration'] for r in updates]!=list(range(first,last+1)):
        raise RuntimeError('Missing branch updates')
    if not all(r['components']==252 and r['exact_commit'] and r['max_radius_error']<=5e-5 for r in updates):
        raise RuntimeError('Invalid branch returns/layout')
    stream=read(metrics/'training_stream.jsonl')
    if [r['iteration'] for r in stream]!=list(range(first,last+1)) or any(len(r['batches'])!=32 for r in stream):
        raise RuntimeError('Training data/accumulation count mismatch')
    evaluation=read(metrics/'trajectory_validation.jsonl')
    expected=sorted({first-1,last}|set(range(((first+31)//32)*32,last+1,32)))
    if [r['iteration'] for r in evaluation]!=expected:
        raise RuntimeError('Evaluation interval mismatch')
    data=read(metrics/'evaluation_data.jsonl')
    if len(data)!=1 or len(set(data[0]['heldout_tokens']))!=20 or data[0]['calibration_tokens'] in data[0]['heldout_tokens']:
        raise RuntimeError('Invalid fixed evaluation data')
    if any([b['tokens'] for b in r['batches']]!=data[0]['heldout_tokens'] for r in evaluation):
        raise RuntimeError('Evaluation data changed')
    return dict(updates=len(updates),final_loss=evaluation[-1]['loss'],
        last_four_mean_loss=sum(r['loss'] for r in evaluation[-4:])/min(4,len(evaluation)),
        near_components=sum(r['near_components'] for r in updates),
        unmatched_components=sum(r['unmatched_components'] for r in updates),
        max_budget_error=max(r['max_budget_error'] for r in updates),
        max_radius_error=max(r['max_radius_error'] for r in updates),
        mean_step_seconds=sum(r['seconds'] for r in stream)/len(stream),
        first32_mean_step_seconds=sum(r['seconds'] for r in stream[:32])/min(32,len(stream)),
        curves=evaluation,successful_updates=True)


def verify_initial(branch, reference, first=513):
    for name in ('trajectory_initial','evaluation_data'):
        if read(branch/(name+'.jsonl'))!=read(reference/(name+'.jsonl')):
            raise RuntimeError('Unmatched initial '+name)
    for a,b in zip(read(branch/'training_stream.jsonl'),read(reference/'training_stream.jsonl')):
        if a['iteration']!=b['iteration'] or a['batches']!=b['batches']:
            raise RuntimeError('Training token stream mismatch')
    a=read(branch/'trajectory_validation.jsonl')[0]
    b=read(reference/'trajectory_validation.jsonl')[0]
    if a['iteration']!=first-1 or a['batches']!=b['batches'] or abs(a['loss']-b['loss'])>1e-7:
        raise RuntimeError('Different starting held-out loss')
    verify_prefix_state(read(branch/'state.jsonl'),read(reference/'state.jsonl'),first,first)


def validate_checkpoint(root):
    checkpoint=root/'common_checkpoint'
    expected=json.loads((root/'results/common_checkpoint_manifest.json').read_text())
    if checkpoint_manifest(checkpoint,512)!=expected:
        raise RuntimeError('Shared checkpoint changed')
    return checkpoint


def prefix(root):
    late=Path(os.environ['ELR_LATE_REFERENCE_ROOT'])
    completed=json.loads((late/'results/LATE_COMPLETED.json').read_text())
    if completed['completed_steps']!=list(range(513,529)) or not completed['integrity']['verified'] or not completed['checkpoint_unchanged']:
        raise RuntimeError('Prerequisite replay did not pass')
    source=Path(os.environ['ELR_MATCHED_REFERENCE_ROOT'])
    source_checkpoint=source/'matched_checkpoint'
    expected=json.loads((source/'results/matched_reference_manifest.json').read_text())
    if checkpoint_manifest(source_checkpoint,252)!=expected:
        raise RuntimeError('Original checkpoint changed')
    folder=root/'results/common_prefix'
    checkpoint=root/'common_checkpoint'
    if folder.exists() or checkpoint.exists():
        raise RuntimeError('Refuse duplicate or partial prefix')
    settings=branch_settings(root,checkpoint,'half_d0',512)
    settings.update(LOAD_CHECKPOINT=str(source_checkpoint),SAVE_CHECKPOINT='1',SAVE_OPTIMIZER='1',SAVE_INTERVAL='512',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_late.py'),
        ELR_FIRST='253',ELR_VALID_SKIP_BATCHES='96')
    run(root,'common_prefix',settings)
    prior=Path(os.environ['ELR_PRIOR_REPLAY_ROOT'])/'results/budget_replay/metrics'
    proof=verify_initial_prefix(folder/'metrics',prior)
    continuation=read(folder/'metrics/continuation_update.jsonl')
    if [r['iteration'] for r in continuation]!=list(range(269,513)) or not all(r['components']==252 and r['exact_commit'] for r in continuation):
        raise RuntimeError('Common continuation incomplete')
    write(root/'results/common_checkpoint_manifest.json',checkpoint_manifest(checkpoint,512))
    if checkpoint_manifest(source_checkpoint,252)!=expected:
        raise RuntimeError('Original checkpoint modified')
    run(root,'restore_smoke',branch_settings(root,checkpoint,'half_d0',514))
    smoke=root/'results/restore_smoke/metrics'
    check=verify_branch(smoke,514)
    late_metrics=late/'results/late_replay/metrics'
    n=verify_prefix_state(read(smoke/'state.jsonl'),read(late_metrics/'state.jsonl'),513,514)
    stream=read(smoke/'training_stream.jsonl')
    probes={r['iteration']:r for r in read(late_metrics/'model_probe.jsonl')}
    for r in stream:
        if r['batches'][0]!=[probes[r['iteration']]['token_sha256'],probes[r['iteration']]['label_sha256']]:
            raise RuntimeError('Restore did not resume the training data')
    smoke_data=read(smoke/'evaluation_data.jsonl')[0]
    late_data=read(late_metrics/'activation_inputs.jsonl')[0]
    if set(smoke_data['heldout_tokens']+[smoke_data['calibration_tokens']]) & set(late_data['heldout_tokens']+[late_data['calibration_tokens']]):
        raise RuntimeError('New trajectory evaluation overlaps selection replay')
    validate_checkpoint(root)
    write(root/'results/PREFIX_READY.json',dict(prefix=proof,restore=check,
        restored_state_records=n,checkpoint=str(checkpoint),spectral_cache='none: exact spectra recomputed',
        new_evaluation_data=True))


def collect_if_complete(root):
    files=[root/'results'/('branch_'+p)/'BRANCH_COMPLETED.json' for p in POLICIES]
    if not all(p.exists() for p in files):
        return
    reference=root/'results/branch_half_d0/metrics'
    result=[]
    for name,path in zip(POLICIES,files):
        metrics=path.parent/'metrics'
        verify_initial(metrics,reference)
        a=read(metrics/'training_stream.jsonl');b=read(reference/'training_stream.jsonl')
        if len(a)!=len(b) or any(x['batches']!=y['batches'] for x,y in zip(a,b)):
            raise RuntimeError('Unpaired trajectory training data')
        result.append(dict(policy=name,**json.loads(path.read_text())))
    validate_checkpoint(root)
    target=root/'results/TRAJECTORIES_COMPLETED.json'
    temp=target.with_suffix('.'+str(os.getpid())+'.tmp')
    write(temp,dict(results=result,paired_data=True,paired_initial_state=True,
        common_updates=256,scope='Five independent paths from one shared state; not five independent seeds'))
    temp.replace(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prefix','branch'))
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    root=Path(os.environ['PROJECT_DIR'])
    if args.stage=='prefix':
        prefix(root)
        return
    if args.index is None or not 0<=args.index<len(POLICIES):
        raise ValueError('Need policy index 0..4')
    if not (root/'results/PREFIX_READY.json').exists():
        raise RuntimeError('Restore proof missing')
    checkpoint=validate_checkpoint(root)
    policy=POLICIES[args.index]
    name='branch_'+policy
    if (root/'results'/name).exists():
        raise RuntimeError('Refuse duplicate/partial trajectory')
    run(root,name,branch_settings(root,checkpoint,policy))
    folder=root/'results'/name
    result=verify_branch(folder/'metrics')
    verify_initial(folder/'metrics',root/'results/restore_smoke/metrics')
    validate_checkpoint(root)
    result['wall_seconds']=json.loads((folder/'wall_seconds.json').read_text())
    write(folder/'BRANCH_COMPLETED.json',result)
    collect_if_complete(root)


if __name__=='__main__':
    main()
