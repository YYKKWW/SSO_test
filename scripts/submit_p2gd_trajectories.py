"""Submit one prefix and a dependent five-arm array, throttled to two GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone


def main():
    root=Path(os.environ['PROJECT_DIR']).resolve()
    receipt=root/'results/trajectory_submission.json'
    if receipt.exists():
        print(receipt.read_text())
        return
    pending=root/'results/trajectory_submission_pending.json'
    if pending.exists():
        raise RuntimeError('Partial submission exists; inspect recorded jobs before retrying')
    preparation=json.loads((root/'results/replay_preparation.json').read_text())
    base=Path(preparation['base'])
    inherited=json.loads((base/'results/confirmation_submission.json').read_text())
    old=Path(inherited['original_reference'])
    config=json.loads((old/'results/prefix/launch_config.json').read_text())
    late=Path(os.environ['ELR_LATE_REFERENCE_ROOT']).resolve()
    proof=json.loads((late/'results/LATE_COMPLETED.json').read_text())
    if not proof['integrity']['verified'] or proof['completed_steps']!=list(range(513,529)):
        raise RuntimeError('Incomplete prerequisite')
    if shutil.disk_usage(root).free<6*1024**3:
        raise RuntimeError('Need 6 GB free for one shared checkpoint and logs')
    env=dict(os.environ,PROJECT_DIR=str(root),ELR_REFERENCE_ROOT=str(old),
        ELR_MATCHED_REFERENCE_ROOT=inherited['reference'],
        ELR_PRIOR_REPLAY_ROOT=preparation['prior_replay'])
    for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX','TOKENIZER_MODEL'):
        env[key]=config[key]
    entry='studies/p2gd_elr/p2gd_trajectory.sbatch'
    result=dict(project=str(root),width=256,seed=1234,first=513,last=768,steps_per_arm=256,
        policies=['half_d0','ambient','top1','cluster_cone','half_output_of_cluster'],
        max_concurrent_h20=2,checkpoint='one temporary common state at 512',
        branch_checkpoint_saving=False,prerequisite=str(late),
        decision='Mixed tiny BF16 signal: bounded cumulative-effect falsification, not a selected P2GD winner',
        created_utc=datetime.now(timezone.utc).isoformat())
    pending.write_text(json.dumps(result,indent=2)+'\n')
    prefix=subprocess.check_output(['sbatch','--parsable','--export=ALL','--time=03:00:00',
        '--job-name=w256_p2gd_prefix',entry],cwd=root,env=dict(env,P2GD_STAGE='prefix'),text=True).strip().split(';')[0]
    result['prefix_job']=prefix
    pending.write_text(json.dumps(result,indent=2)+'\n')
    array=subprocess.check_output(['sbatch','--parsable','--export=ALL','--array=0-4%2',
        '--dependency=afterok:'+prefix,'--kill-on-invalid-dep=yes',entry],
        cwd=root,env=dict(env,P2GD_STAGE='branch'),text=True).strip().split(';')[0]
    result['array_job']=array
    pending.write_text(json.dumps(result,indent=2)+'\n')
    pending.replace(receipt)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
