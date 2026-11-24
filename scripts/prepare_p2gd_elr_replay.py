"""Stage a frozen, isolated H20 replay. Never overwrite completed experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--study', choices=('aus','budget','six','late','paths'), default='aus')
    parser.add_argument('--prior-replay', type=Path)
    parser.add_argument('--six-reference', type=Path)
    args = parser.parse_args()
    if args.study=='paths' and args.submit:
        parser.error('Stage paths first; submit_p2gd_trajectories.py handles the dependent array')
    base, root = args.base.resolve(), args.target.resolve()
    if root == base or root.is_relative_to(base):
        raise ValueError('Use a separate project')
    previous = json.loads((base/'results/confirmation_submission.json').read_text())
    if args.study == 'budget' and (args.prior_replay is None or not
            (args.prior_replay/'results/REPLAY_COMPLETED.json').exists()):
        raise ValueError('Budget replay requires a completed prior AUS replay')
    if args.study in ('six','late','paths') and (args.prior_replay is None or not
            (args.prior_replay/'results/BUDGET_COMPLETED.json').exists()):
        raise ValueError('Six-way replay requires a completed prior budget replay')
    if args.study in ('late','paths'):
        if args.six_reference is None:
            raise ValueError('Later replay requires the six-way evaluation provenance')
        probes=[json.loads(line) for line in (args.six_reference/'results/six_replay/metrics/model_probe.jsonl').read_text().splitlines()]
        if [r['iteration'] for r in probes]!=list(range(253,301)) or not all(r['optimizer_update_successful'] for r in probes):
            raise ValueError('Six-way reference updates are incomplete')
    digest = sha(args.overlay)
    receipt = root/'results/replay_submission.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['overlay_sha256'] != digest:
            raise RuntimeError('Submitted source differs')
        print(json.dumps(result,indent=2))
        return
    if root.exists():
        raise RuntimeError('Refusing to overwrite a staged/partial project')
    if shutil.disk_usage(base).free < 4*1024**3:
        raise RuntimeError('Require 4 GB free; no automatic deletion')
    entries = set()
    root.mkdir(parents=True)
    for line in (base/'source.sha256').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        source = (base/name).resolve()
        if not source.is_relative_to(base) or sha(source) != checksum:
            raise RuntimeError('Frozen source changed: '+name)
        target = root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        entries.add(name)
    with tarfile.open(args.overlay) as archive:
        for item in archive.getmembers():
            if not item.isfile():
                continue
            target = (root/item.name).resolve()
            if not target.is_relative_to(root) or not item.name.startswith('studies/p2gd_elr/'):
                raise RuntimeError('Unexpected overlay path')
            if item.name in entries:
                raise RuntimeError('Overlay must not replace inherited files')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(item) as src, target.open('wb') as out:
                shutil.copyfileobj(src,out)
            entries.add(item.name)
    for name in ('results','logs'):
        (root/name).mkdir(exist_ok=True)
    (root/'source.sha256').write_text(''.join(f'{sha(root/p)}  {p}\n' for p in sorted(entries)))
    result = dict(base=str(base), target=str(root), overlay_sha256=digest,
        base_manifest_sha256=sha(base/'source.sha256'), width=256, seed=1234,
        study=args.study,common_updates={'six':48,'late':276,'paths':260}.get(args.study,16),
        directions={'aus':8,'budget':15,'six':6,'late':5,'paths':5}[args.study],
        near_threshold=2e-4, cluster_window=2e-4,
        gpus=1, time_limit_hours=3, checkpoint_saving=False,
        inherited_layout='252 components, FC1 unsplit',
        created_utc=datetime.now(timezone.utc).isoformat())
    if args.study=='six':
        result.update(prefix_updates=16,measured_updates=32,measured_first=269,measured_last=300,
            valid_skip_batches=68,prior_replay=str(args.prior_replay.resolve()))
    if args.study in ('late','paths'):
        required={'studies/p2gd_elr/'+name for name in ('verify_p2gd_six.py','verify_p2gd_late.py',
            'p2gd_late_geometry.py','p2gd_late_replay.py','run_p2gd_late.py','pretrain_p2gd_late.py',
            'summarize_p2gd_late.py','test_p2gd_late.py','p2gd_late.sbatch')}
        if not required<=entries:
            raise ValueError('Incomplete later replay overlay: '+str(sorted(required-entries)))
        result.update(prefix_updates=16,fast_continuation_updates=244,measured_updates=16,
            measured_first=513,measured_last=528,valid_skip_batches=96,alternative_policies=4,
            matching_space='BF16 forward weights',prior_replay=str(args.prior_replay.resolve()),
            six_reference=str(args.six_reference.resolve()))
    if args.study=='paths':
        required={'studies/p2gd_elr/'+name for name in ('p2gd_trajectory.py','p2gd_trajectory_geometry.py',
            'pretrain_p2gd_trajectory.py','run_p2gd_trajectories.py','submit_p2gd_trajectories.py',
            'p2gd_trajectory.sbatch','test_p2gd_trajectory.py','test_p2gd_trajectory_integrity.py')}
        if not required<=entries:
            raise ValueError('Incomplete trajectory overlay: '+str(sorted(required-entries)))
        result.update(measured_updates=256,measured_first=513,measured_last=768,
            valid_skip_batches=128,max_concurrent_h20=2,branch_time_limit_hours=6,
            checkpoint_saving='one common checkpoint at 512; no branch checkpoints',
            study_kind='five independent trajectories, same initial state and token order')
    (root/'results/replay_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.submit:
        old = Path(previous['original_reference'])
        config = json.loads((old/'results/prefix/launch_config.json').read_text())
        env = dict(os.environ, PROJECT_DIR=str(root), ELR_REFERENCE_ROOT=str(old),
            ELR_MATCHED_REFERENCE_ROOT=previous['reference'], ELR_CONFIRM_REFERENCE_ROOT=str(base))
        if args.study in ('budget','six','late'):
            env['ELR_PRIOR_REPLAY_ROOT'] = str(args.prior_replay.resolve())
        if args.study=='late':
            env['ELR_SIX_REFERENCE_ROOT']=str(args.six_reference.resolve())
        for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX','TOKENIZER_MODEL'):
            env[key] = config[key]
        for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX'):
            for suffix in ('.bin','.idx'):
                if not Path(env[key]+suffix).exists():
                    raise RuntimeError('Missing data: '+key)
        launcher = {'aus':'p2gd_elr_replay.sbatch','budget':'p2gd_budget.sbatch',
                    'six':'p2gd_six.sbatch','late':'p2gd_late.sbatch'}[args.study]
        job = subprocess.check_output(['sbatch','--parsable','--export=ALL',
            'studies/p2gd_elr/'+launcher],cwd=root,env=env,text=True).strip().split(';')[0]
        result.update(job_id=job, submitted_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
