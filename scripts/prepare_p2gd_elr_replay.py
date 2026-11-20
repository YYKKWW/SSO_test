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
    parser.add_argument('--study', choices=('aus','budget'), default='aus')
    parser.add_argument('--prior-replay', type=Path)
    args = parser.parse_args()
    base, root = args.base.resolve(), args.target.resolve()
    if root == base or root.is_relative_to(base):
        raise ValueError('Use a separate project')
    previous = json.loads((base/'results/confirmation_submission.json').read_text())
    if args.study == 'budget' and (args.prior_replay is None or not
            (args.prior_replay/'results/REPLAY_COMPLETED.json').exists()):
        raise ValueError('Budget replay requires a completed prior AUS replay')
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
        study=args.study,common_updates=16, directions=15 if args.study=='budget' else 8,
        near_threshold=2e-4, cluster_window=2e-4,
        gpus=1, time_limit_hours=3, checkpoint_saving=False,
        inherited_layout='252 components, FC1 unsplit',
        created_utc=datetime.now(timezone.utc).isoformat())
    (root/'results/replay_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    if args.submit:
        old = Path(previous['original_reference'])
        config = json.loads((old/'results/prefix/launch_config.json').read_text())
        env = dict(os.environ, PROJECT_DIR=str(root), ELR_REFERENCE_ROOT=str(old),
            ELR_MATCHED_REFERENCE_ROOT=previous['reference'], ELR_CONFIRM_REFERENCE_ROOT=str(base))
        if args.study == 'budget':
            env['ELR_PRIOR_REPLAY_ROOT'] = str(args.prior_replay.resolve())
        for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX','TOKENIZER_MODEL'):
            env[key] = config[key]
        for key in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX'):
            for suffix in ('.bin','.idx'):
                if not Path(env[key]+suffix).exists():
                    raise RuntimeError('Missing data: '+key)
        launcher = 'p2gd_budget.sbatch' if args.study=='budget' else 'p2gd_elr_replay.sbatch'
        job = subprocess.check_output(['sbatch','--parsable','--export=ALL',
            'studies/p2gd_elr/'+launcher],cwd=root,env=env,text=True).strip().split(';')[0]
        result.update(job_id=job, submitted_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
