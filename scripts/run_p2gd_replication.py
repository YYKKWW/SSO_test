"""Later-start paired replication with a post-save optimizer-state restore gate."""
import argparse
import json
import os
from pathlib import Path

from run import checkpoint_manifest, run
from run_p2gd_trajectories import branch_settings, verify_branch, verify_initial, write
from verify_p2gd_six import read, verify_prefix_state

POLICIES=('half_d0','top1','cluster_cone','half_output_of_cluster')


def settings(root, checkpoint, policy, first=769, last=1024):
    value=branch_settings(root,checkpoint,policy,last)
    value.update(ELR_FIRST=str(first),ELR_VALID_SKIP_BATCHES='160',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_replication.py'),
        P2GD_SNAPSHOTS='1' if policy=='cluster_cone' else '0',P2GD_RECORD_POST_STATE='-1')
    return value


def validate(root):
    checkpoint=root/'common_checkpoint'
    if checkpoint_manifest(checkpoint,768)!=json.loads((root/'results/common_checkpoint_manifest.json').read_text()):
        raise RuntimeError('New shared checkpoint changed')
    return checkpoint


def verify_restored_state(post, initial):
    if len(post)!=1 or len(initial)!=1 or post[0]['iteration']!=768 or initial[0]['iteration']!=769:
        raise RuntimeError('Missing post-save / restored state')
    for key in ('model','optimizer'):
        if post[0][key]!=initial[0][key]:
            raise RuntimeError('Checkpoint restore differs: '+key)


def verify_repeated_prefix(metrics, old, first=513, last=768):
    """Check dense updates/streams but only the predeclared sparse state audits."""
    result=verify_branch(metrics,last=last,first=first)
    verify_branch(old,last=last,first=first)
    verify_initial(metrics,old,first=first)
    expected=sorted({first,last}|set(range(((first+31)//32)*32,last+1,32)))
    current_state,previous_state=read(metrics/'state.jsonl'),read(old/'state.jsonl')
    count=verify_prefix_state(current_state,previous_state,first,last,expected_iterations=expected)
    records_per_step=sum(r['iteration']==first for r in previous_state)
    for rows in (current_state,previous_state):
        if any(sum(r['iteration']==step for r in rows)!=records_per_step for step in expected):
            raise RuntimeError('Incomplete shared-prefix component audit')
    if read(metrics/'trajectory_validation.jsonl')!=read(old/'trajectory_validation.jsonl'):
        raise RuntimeError('Repeated reference path differs')
    if read(metrics/'trajectory_update.jsonl')!=read(old/'trajectory_update.jsonl'):
        raise RuntimeError('Repeated update/return records differ')
    a,b=read(metrics/'training_stream.jsonl'),read(old/'training_stream.jsonl')
    if len(a)!=len(b) or any((x['iteration'],x['batches'])!=(y['iteration'],y['batches']) for x,y in zip(a,b)):
        raise RuntimeError('Reference stream differs')
    return dict(updates=result['updates'],final_loss=result['final_loss'],
        last_four_mean_loss=result['last_four_mean_loss'],near_components=result['near_components'],
        max_radius_error=result['max_radius_error'],matched_reference_state_records=count,
        audited_iterations=expected,state_records_per_audit=records_per_step,
        exact_initial_model_and_optimizer=True,exact_training_stream=True,
        exact_update_records=True,
        exact_sampled_weight_gradient_momentum=True,exact_validation_curve=True,
        scope='Repeated reference path only; checkpoint restore is a separate gate')


def prefix(root):
    prior=Path(os.environ['P2GD_PREVIOUS_PATHS_ROOT'])
    completed=json.loads((prior/'results/TRAJECTORIES_COMPLETED.json').read_text())
    if not completed['paired_data'] or not completed['paired_initial_state'] or len(completed['results'])!=5:
        raise RuntimeError('Previous panel not verified')
    source=prior/'common_checkpoint'
    manifest=json.loads((prior/'results/common_checkpoint_manifest.json').read_text())
    if checkpoint_manifest(source,512)!=manifest:
        raise RuntimeError('Source checkpoint changed')
    if (root/'results/common_prefix').exists() or (root/'common_checkpoint').exists():
        raise RuntimeError('Refuse duplicate/partial continuation')
    checkpoint=root/'common_checkpoint'
    config=settings(root,checkpoint,'half_d0',513,768)
    config.update(LOAD_CHECKPOINT=str(source),SAVE_CHECKPOINT='1',SAVE_OPTIMIZER='1',SAVE_INTERVAL='768',
                  ELR_VALID_SKIP_BATCHES='128',P2GD_RECORD_POST_STATE='768')
    run(root,'common_prefix',config)
    metrics=root/'results/common_prefix/metrics'
    old=prior/'results/branch_half_d0/metrics'
    proof=verify_repeated_prefix(metrics,old)
    write(root/'results/common_checkpoint_manifest.json',checkpoint_manifest(checkpoint,768))
    restore_gate(root,proof,old)
    if checkpoint_manifest(source,512)!=manifest:
        raise RuntimeError('Original checkpoint modified')


def restore_gate(root,proof,old):
    checkpoint=validate(root)
    if (root/'results/restore_smoke').exists() or (root/'results/PREFIX_READY.json').exists():
        raise RuntimeError('Refuse duplicate/partial restore gate')
    run(root,'restore_smoke',settings(root,checkpoint,'half_d0',769,770))
    smoke=root/'results/restore_smoke/metrics'
    verify_branch(smoke,last=770,first=769)
    verify_restored_state(read(root/'results/common_prefix/metrics/checkpoint_post_state.jsonl'),
                          read(smoke/'trajectory_initial.jsonl'))
    new=read(smoke/'evaluation_data.jsonl')[0];previous=read(old/'evaluation_data.jsonl')[0]
    if set(new['heldout_tokens']+[new['calibration_tokens']]) & set(previous['heldout_tokens']+[previous['calibration_tokens']]):
        raise RuntimeError('Evaluation batches overlap previous panel')
    validate(root)
    write(root/'results/PREFIX_READY.json',dict(verified=True,
        matched_reference_state_records=proof['matched_reference_state_records'],
        exact_post_save_model_and_optimizer_restored=True,new_evaluation_batches=True,
        checkpoint=str(checkpoint),first=769,last=1024))


def recover(root):
    """Reuse the verified saved prefix; never replay its 256 training updates."""
    record=json.loads((root/'results/recovery_preparation.json').read_text())
    source=Path(record['source'])
    if (root/'common_checkpoint').resolve()!=(source/'common_checkpoint').resolve():
        raise RuntimeError('Recovery checkpoint does not match its source')
    if (root/'results/common_prefix').resolve()!=(source/'results/common_prefix').resolve():
        raise RuntimeError('Recovery prefix does not match its source')
    validate(root)
    old=Path(os.environ['P2GD_PREVIOUS_PATHS_ROOT'])/'results/branch_half_d0/metrics'
    proof=verify_repeated_prefix(root/'results/common_prefix/metrics',old)
    restore_gate(root,proof,old)


def branch(root,index):
    if not (root/'results/PREFIX_READY.json').exists():
        raise RuntimeError('Restore gate missing')
    checkpoint=validate(root)
    policy=POLICIES[index];name='branch_'+policy
    if (root/'results'/name).exists():
        raise RuntimeError('Refuse duplicate/partial branch')
    run(root,name,settings(root,checkpoint,policy))
    folder=root/'results'/name
    result=verify_branch(folder/'metrics',last=1024,first=769)
    verify_initial(folder/'metrics',root/'results/restore_smoke/metrics',first=769)
    verify_restored_state(read(root/'results/common_prefix/metrics/checkpoint_post_state.jsonl'),
                          read(folder/'metrics/trajectory_initial.jsonl'))
    validate(root)
    result.update(policy=policy,wall_seconds=json.loads((folder/'wall_seconds.json').read_text()))
    write(folder/'BRANCH_COMPLETED.json',result)
    files=[root/'results'/('branch_'+p)/'BRANCH_COMPLETED.json' for p in POLICIES]
    if all(p.exists() for p in files):
        reference=root/'results/branch_half_d0/metrics'
        for policy in POLICIES:
            verify_initial(root/'results'/('branch_'+policy)/'metrics',reference,first=769)
        temp=root/'results'/f'REPLICATION_COMPLETED.{os.getpid()}.tmp'
        write(temp,dict(results=[json.loads(p.read_text()) for p in files],
            paired_initial_state=True,paired_data=True,first=769,last=1024,
            scope='Later-start replication, same seed; not an independent initialization seed'))
        temp.replace(root/'results/REPLICATION_COMPLETED.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prefix','recover','branch'))
    parser.add_argument('--index',type=int,choices=range(4))
    args=parser.parse_args();root=Path(os.environ['PROJECT_DIR'])
    if args.stage=='prefix':
        prefix(root)
    elif args.stage=='recover':
        recover(root)
    else:
        if args.index is None:
            parser.error('branch needs --index')
        branch(root,args.index)
