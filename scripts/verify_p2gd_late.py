"""Integrity checks for the later-state replay and its unaudited continuation."""
from verify_p2gd_six import read, verify_prefix_state


def verify_initial_prefix(metrics,prior):
    current=[r for r in read(metrics/'model_probe.jsonl') if 253<=r['iteration']<=268]
    previous=read(prior/'model_probe.jsonl')
    if [r['iteration'] for r in current]!=list(range(253,269)) or len(previous)!=16:
        raise ValueError('Incomplete common prefix')
    for a,b in zip(current,previous):
        if not a['optimizer_update_successful'] or any(a[k]!=b[k] for k in ('iteration','token_sha256','label_sha256')):
            raise ValueError('Prefix update or training data mismatch')
        if abs(a['loss_before']-b['loss_before'])>1e-6:
            raise ValueError('Prefix loss mismatch')
    count=verify_prefix_state(read(metrics/'state.jsonl'),read(prior/'state.jsonl'))
    if read(metrics/'optimizer_before.jsonl')!=read(prior/'optimizer_before.jsonl'):
        raise ValueError('Initial optimizer state mismatch')
    return dict(prefix_steps=16,state_records=count,verified=True)


def verify_completed(metrics,prior):
    result=verify_initial_prefix(metrics,prior)
    probes=read(metrics/'model_probe.jsonl')
    if [r['iteration'] for r in probes]!=list(range(253,269))+list(range(513,529)):
        raise ValueError('Incorrect audited windows')
    if not all(r['optimizer_update_successful'] for r in probes):
        raise ValueError('Unsuccessful optimizer update')
    continuation=read(metrics/'continuation_update.jsonl')
    if [r['iteration'] for r in continuation]!=list(range(269,513)):
        raise ValueError('Incomplete continuation')
    if not all(r['components']==252 and r['exact_commit'] and r['max_radius_error']<=5e-5 for r in continuation):
        raise ValueError('Continuation layout/return mismatch')
    activations=read(metrics/'activation_inputs.jsonl')
    first=activations[0]
    if len(set(first['heldout_tokens']))!=20 or first['calibration_tokens'] in first['heldout_tokens']:
        raise ValueError('Invalid heldout split')
    if any((r['heldout_tokens'],r['calibration_tokens'])!=(first['heldout_tokens'],first['calibration_tokens']) for r in activations):
        raise ValueError('Evaluation data changed')
    return dict(**result,common_updates=276,fast_continuation_updates=244,measured_updates=16,
        all_optimizer_updates_successful=True,fixed_evaluation=True)
