"""One prefix, four paired paths (at most two GPUs), one dependent offline audit."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

POLICIES=('half_d0','top1','cluster_cone','half_output_of_cluster')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    root=Path(os.environ['PROJECT_DIR']).resolve()
    receipt=root/'results/replication_submission.json'
    pending=root/'results/replication_submission_pending.json'
    if receipt.exists():
        print(receipt.read_text());return
    if pending.exists():
        raise RuntimeError('Partial submission: inspect existing jobs before retrying')
    prior=Path(os.environ['P2GD_PREVIOUS_PATHS_ROOT']).resolve()
    proof=json.loads((prior/'results/TRAJECTORIES_COMPLETED.json').read_text())
    if not proof['paired_data'] or not proof['paired_initial_state']:
        raise RuntimeError('Previous panel incomplete')
    if shutil.disk_usage(root).free<6*1024**3:
        raise RuntimeError('Need 6 GB free; do not automatically delete anything')
    prep=json.loads((root/'results/replay_preparation.json').read_text())
    base=Path(prep['base'])
    old=json.loads((base/'results/confirmation_submission.json').read_text())['original_reference']
    config=json.loads((Path(old)/'results/prefix/launch_config.json').read_text())
    env=dict(os.environ,ELR_REFERENCE_ROOT=old)
    for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX','TOKENIZER_MODEL'):
        env[key]=config[key]
    entry='studies/p2gd_elr/p2gd_replication.sbatch'
    result=dict(project=str(root),prior=str(prior),width=256,seed=1234,
        prefix_first=513,prefix_last=768,first=769,last=1024,steps_per_arm=256,
        policies=POLICIES,max_concurrent_h20=2,valid_skip_batches=160,
        gap=2e-4,cluster_window=2e-4,checkpoint_saving='one common checkpoint at 768',
        snapshots='cluster path only, <=24 and <=256 MiB',
        scope='Later-start replication, not a second seed; no 1B/3B job',
        created_utc=datetime.now(timezone.utc).isoformat())
    write(pending,result)
    def submit(stage,args):
        value=subprocess.check_output(['sbatch','--parsable','--export=ALL',*args,entry],
            cwd=root,env=dict(env,P2GD_STAGE=stage),text=True).strip().split(';')[0]
        result[stage+'_job']=value;write(pending,result)
        return value
    prefix=submit('prefix',['--time=03:00:00','--job-name=w256_p2gd_rep_prefix'])
    array=submit('branch',['--array=0-3%2','--dependency=afterok:'+prefix,'--kill-on-invalid-dep=yes'])
    submit('audit',['--time=00:30:00','--job-name=w256_p2gd_snapshot','--dependency=afterok:'+array,'--kill-on-invalid-dep=yes'])
    pending.replace(receipt)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
