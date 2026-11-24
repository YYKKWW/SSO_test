"""Replay the old prefix, measure the next 32 states, never write a checkpoint."""
import json
import os
from pathlib import Path

from run import checkpoint_manifest, run
from summarize import rows
from summarize_p2gd_six import six_summary
from verify_p2gd_six import verify_prefix_state


def main():
    root=Path(os.environ['PROJECT_DIR']);old=Path(os.environ['ELR_REFERENCE_ROOT'])
    source=Path(os.environ['ELR_MATCHED_REFERENCE_ROOT'])
    prior=Path(os.environ['ELR_PRIOR_REPLAY_ROOT'])
    checkpoint=source/'matched_checkpoint'
    expected=json.loads((source/'results/matched_reference_manifest.json').read_text())
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Reference checkpoint changed')
    folder=root/'results/six_replay'
    if folder.exists():
        raise RuntimeError('Refusing duplicate or partial replay')
    run(root,'six_replay',dict(LOAD_CHECKPOINT=str(checkpoint),CHECKPOINT_PATH=str(checkpoint),
        SAVE_CHECKPOINT='0',DATA_CACHE_PATH=str(old/'data_cache'),EVAL_INTERVAL='10000',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_six.py'),
        ELR_FIXED_VALID_BATCHES='21',ELR_HELDOUT_BATCHES='20',ELR_VALID_SKIP_BATCHES='68',
        ELR_ACTIVATION_TOKENS='256',ELR_FIRST='253',ELR_LAST='300',EXIT_INTERVAL='300',
        ELR_COMMON_PREFIX='0',ELR_ASSERT_LAYOUT='1',TRANSFORMER_IMPL='local',ELR_NEAR_THRESHOLD='2e-4'))
    metrics=folder/'metrics';old_metrics=prior/'results/budget_replay/metrics'
    result=six_summary(metrics)
    if result['completed_steps']!=list(range(269,301)) or any(r['matching_violations'] for r in result['results']):
        raise RuntimeError('Incomplete replay or unmatched budget')
    probes=rows(metrics/'model_probe.jsonl');old_probes=rows(old_metrics/'model_probe.jsonl')
    if len(probes)!=48 or len(old_probes)!=16 or not all(r['optimizer_update_successful'] for r in probes):
        raise RuntimeError('Optimizer update failed or incomplete prefix')
    for a,b in zip(probes[:16],old_probes):
        if any(a[k]!=b[k] for k in ('iteration','token_sha256','label_sha256')) or abs(a['loss_before']-b['loss_before'])>1e-6:
            raise RuntimeError('Common prefix differs from prior budget study')
    verify_prefix_state(rows(metrics/'state.jsonl'),rows(old_metrics/'state.jsonl'))
    if rows(metrics/'optimizer_before.jsonl')!=rows(old_metrics/'optimizer_before.jsonl'):
        raise RuntimeError('Initial shared optimizer state differs')
    new=rows(metrics/'activation_inputs.jsonl')[0]; earlier=rows(old_metrics/'activation_inputs.jsonl')[0]
    if set(new['heldout_tokens']+[new['calibration_tokens']]) & set(earlier['heldout_tokens']+[earlier['calibration_tokens']]):
        raise RuntimeError('Evaluation data overlap')
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Checkpoint modified')
    result.update(checkpoint_unchanged=True,prior_prefix_matched=True,
        heldout_microbatches=20,width=256,seed=1234,production_timing=False)
    (root/'results/SIX_COMPLETED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
