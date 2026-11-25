"""Freeze a separate recovery overlay and reuse the saved step768 checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def verified_checkpoint(source,status):
    if not status['frozen_source_verified'] or status['checkpoint_iteration']!='768':
        raise RuntimeError('Expected a verified step768 source')
    receipt=json.loads((source/'results/replication_submission.json').read_text())
    for role,state in (('prefix','FAILED'),('branch','CANCELLED'),('audit','CANCELLED')):
        if status['jobs'][receipt[role+'_job']]['state']!=state:
            raise RuntimeError('Source job status does not match the interrupted batch')
    if ((source/'results/PREFIX_READY.json').exists() or (source/'results/restore_smoke').exists()
            or any((source/'results').glob('branch_*'))):
        raise RuntimeError('Source has already advanced; inspect before recovery')
    checkpoint=source/'common_checkpoint'
    if (checkpoint/'latest_checkpointed_iteration.txt').read_text().strip()!='768':
        raise RuntimeError('Source checkpoint is no longer step768')
    expected=status['checkpoint_files']
    paths={str(p.relative_to(checkpoint)):p for p in checkpoint.rglob('*') if p.is_file()}
    if set(paths)!=set(expected):
        raise RuntimeError('Checkpoint file set changed')
    for name,path in paths.items():
        if path.stat().st_size!=expected[name]['bytes'] or sha(path)!=expected[name]['sha256']:
            raise RuntimeError('Checkpoint changed: '+name)
    return dict(iteration=768,files={k:v['sha256'] for k,v in expected.items()})


def prepare(source,target,overlay,status_path):
    source,target=source.resolve(),target.resolve()
    if target==source or target.is_relative_to(source) or target.exists():
        raise RuntimeError('Use a new, separate recovery directory')
    if shutil.disk_usage(source).free<6*1024**3:
        raise RuntimeError('Need 6 GB free; no automatic cleanup')
    status=json.loads(status_path.read_text())
    if sha(source/'source.sha256')!=status['source_manifest_sha256']:
        raise RuntimeError('Frozen source manifest changed')
    checkpoint=verified_checkpoint(source,status)
    entries={}
    for line in (source/'source.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1);name=name.lstrip('*')
        path=(source/name).resolve()
        if not path.is_relative_to(source) or sha(path)!=digest:
            raise RuntimeError('Frozen source file changed: '+name)
        entries[name]=path
    with tarfile.open(overlay) as archive:
        members=[m for m in archive.getmembers() if m.isfile()]
        names=[m.name for m in members]
        if len(names)!=len(set(names)):
            raise RuntimeError('Duplicate overlay member')
        for name in names:
            if not name.startswith('studies/p2gd_elr/') or not (target/name).resolve().is_relative_to(target):
                raise RuntimeError('Unexpected recovery overlay path')
        target.mkdir(parents=True)
        for name,path in entries.items():
            output=target/name;output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,output)
        for member in members:
            output=target/member.name;output.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src,output.open('wb') as out:
                shutil.copyfileobj(src,out)
            entries[member.name]=output
    for name in ('results','logs'):
        (target/name).mkdir()
    (target/'common_checkpoint').symlink_to(source/'common_checkpoint',target_is_directory=True)
    (target/'results/common_prefix').symlink_to(source/'results/common_prefix',target_is_directory=True)
    (target/'source.sha256').write_text(''.join(f'{sha(target/name)}  {name}\n' for name in sorted(entries)))
    prep=json.loads((source/'results/replay_preparation.json').read_text())
    prep.update(target=str(target),recovery_source=str(source),prefix_updates=0,
        checkpoint_saving='none; read-only reuse of saved step768')
    record=dict(source=str(source),target=str(target),created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(source/'source.sha256'),overlay_sha256=sha(overlay),
        source_status=status,checkpoint_reused=True,repeated_prefix_updates=0)
    for name,value in (('replay_preparation',prep),('recovery_preparation',record),
                       ('common_checkpoint_manifest',checkpoint)):
        (target/'results'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','target','overlay','status'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.target,args.overlay,args.status),indent=2))
