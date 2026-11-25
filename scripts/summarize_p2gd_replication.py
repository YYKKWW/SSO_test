"""Verify the recorded prefix without relabeling failed jobs or releasing branches."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_p2gd_replication import verify_repeated_prefix
from verify_p2gd_six import read


def summarize(current, prior, status):
    receipt=json.loads((current/'replication_submission.json').read_text())
    expected_jobs={receipt[k+'_job'] for k in ('prefix','branch','audit')}
    if set(status['jobs'])!=expected_jobs:
        raise ValueError('Scheduler snapshot is for a different submission')
    if (current/'REPLICATION_COMPLETED.json').exists() or any(current.glob('branch_*')):
        raise ValueError('Branch results exist; use a completed-trajectory collector')
    if ((current/'restore_smoke').exists() or status['restore_smoke_present']
            or (current/'PREFIX_READY.json').exists() or status['prefix_ready']):
        raise ValueError('Restore work has started; this collector only handles the failed prefix gate')
    metrics=current/'common_prefix/metrics'
    reference=prior/'branch_half_d0/metrics'
    proof=verify_repeated_prefix(metrics,reference)
    post=read(metrics/'checkpoint_post_state.jsonl')
    if len(post)!=1 or post[0]['iteration']!=768 or not post[0]['model'] or not post[0]['optimizer']:
        raise ValueError('Missing post-update state signature')
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        scheduler=status,prefix_verification=proof,
        post_update_signature_recorded=True,
        restore_smoke_completed=False,prefix_ready=False,
        new_branch_results=0,snapshot_audit_completed=False,
        original_failure='Dense audit expectation applied to a sparse first/last/every-32 schedule',
        correction='Require all declared sparse audits and equal component counts; retain dense update/stream checks',
        limitation='Post-hoc prefix verification is not a restore proof or a later-start replication result',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (metrics,reference) for p in sorted(folder.glob('*.jsonl'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('current',type=Path)
    parser.add_argument('prior',type=Path)
    parser.add_argument('status',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=summarize(args.current,args.prior,json.loads(args.status.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
