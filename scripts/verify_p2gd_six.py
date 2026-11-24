"""Verify the common prefix, not unrelated later state-audit records."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_prefix_state(current, prior, first=253, last=268):
    expected = set(range(first,last+1))
    a = [r for r in current if first <= r['iteration'] <= last]
    b = [r for r in prior if first <= r['iteration'] <= last]
    if {r['iteration'] for r in a} != expected or {r['iteration'] for r in b} != expected:
        raise ValueError('Missing shared-prefix state audit')
    if a != b:
        raise ValueError('Shared-prefix weight/gradient/momentum state differs')
    return len(a)


def verify(directory, prior):
    probes, old = read(directory/'model_probe.jsonl'), read(prior/'model_probe.jsonl')
    if [r['iteration'] for r in probes] != list(range(253,301)):
        raise ValueError('Incomplete or duplicate common updates')
    if [r['iteration'] for r in old] != list(range(253,269)):
        raise ValueError('Invalid prior prefix')
    if not all(r['optimizer_update_successful'] for r in probes+old):
        raise ValueError('Unsuccessful optimizer update')
    max_loss_difference=0.
    for a,b in zip(probes[:16],old):
        if any(a[k]!=b[k] for k in ('iteration','token_sha256','label_sha256')):
            raise ValueError('Prefix training data mismatch')
        max_loss_difference=max(max_loss_difference,abs(a['loss_before']-b['loss_before']))
    if max_loss_difference>1e-6:
        raise ValueError('Prefix training loss mismatch')
    state_count=verify_prefix_state(read(directory/'state.jsonl'),read(prior/'state.jsonl'))
    a,b=read(directory/'optimizer_before.jsonl'),read(prior/'optimizer_before.jsonl')
    if not a or a!=b:
        raise ValueError('Initial optimizer state mismatch')
    current,previous=read(directory/'activation_inputs.jsonl'),read(prior/'activation_inputs.jsonl')
    if not current or not previous:
        raise ValueError('Missing evaluation data provenance')
    def tokens(rows):
        return {token for row in rows for token in row['heldout_tokens']+[row['calibration_tokens']]}
    if tokens(current)&tokens(previous):
        raise ValueError('New and old evaluation data overlap')
    first=current[0]
    if any(r['heldout_tokens']!=first['heldout_tokens'] or r['calibration_tokens']!=first['calibration_tokens']
           for r in current):
        raise ValueError('Evaluation batches changed within replay')
    if first['calibration_tokens'] in first['heldout_tokens'] or len(set(first['heldout_tokens']))!=20:
        raise ValueError('Invalid calibration/held-out split')
    return dict(common_updates=48, shared_prefix_steps=16, verified_state_records=state_count,
        max_prefix_loss_difference=max_loss_difference, optimizer_state_matched=True,
        all_optimizer_updates_successful=True, fresh_fixed_twenty_batch_evaluation=True,
        original_scheduler_state='FAILED (postprocessing only)',
        original_failure='Compared all 48-step state audit records with the prior 16-step audit',
        correction='Compare every state record in the common 253--268 prefix; preserve later records',
        checkpoint_unchanged='requires separate manifest verification',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (directory,prior) for p in sorted(folder.glob('*.jsonl'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    parser.add_argument('prior',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=verify(args.metrics,args.prior)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
