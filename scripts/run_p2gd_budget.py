import json
import os
from pathlib import Path

from run import checkpoint_manifest, run
from summarize import rows
from summarize_p2gd_budget import summarize


def main():
    root=Path(os.environ['PROJECT_DIR'])
    old=Path(os.environ['ELR_REFERENCE_ROOT'])
    source=Path(os.environ['ELR_MATCHED_REFERENCE_ROOT'])
    prior=Path(os.environ['ELR_PRIOR_REPLAY_ROOT'])
    checkpoint=source/'matched_checkpoint'
    expected=json.loads((source/'results/matched_reference_manifest.json').read_text())
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Reference checkpoint changed')
    folder=root/'results/budget_replay'
    if folder.exists():
        raise RuntimeError('Refusing duplicate or partial replay')
    config=dict(LOAD_CHECKPOINT=str(checkpoint),CHECKPOINT_PATH=str(checkpoint),SAVE_CHECKPOINT='0',
        DATA_CACHE_PATH=str(old/'data_cache'),EVAL_INTERVAL='10000',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_budget.py'),
        ELR_FIXED_VALID_BATCHES='21',ELR_HELDOUT_BATCHES='20',ELR_VALID_SKIP_BATCHES='47',
        ELR_ACTIVATION_TOKENS='256',ELR_FIRST='253',ELR_LAST='268',EXIT_INTERVAL='268',
        ELR_COMMON_PREFIX='0',ELR_ASSERT_LAYOUT='1',TRANSFORMER_IMPL='local',ELR_NEAR_THRESHOLD='2e-4')
    run(root,'budget_replay',config)
    metrics=folder/'metrics';old_metrics=prior/'results/paired_replay/metrics'
    result=summarize(metrics)
    if result['completed_steps']!=list(range(253,269)) or any(r['matching_violations'] for r in result['results']):
        raise RuntimeError('Incomplete replay or unmatched budget')
    probes=rows(metrics/'model_probe.jsonl');old_probes=rows(old_metrics/'model_probe.jsonl')
    if len(probes)!=16 or not all(r['optimizer_update_successful'] for r in probes):
        raise RuntimeError('Optimizer update failed')
    for a,b in zip(probes,old_probes):
        if any(a[k]!=b[k] for k in ('iteration','token_sha256','label_sha256')) or abs(a['loss_before']-b['loss_before'])>1e-6:
            raise RuntimeError('Common training replay differs from prior AUS study')
    for key in ('state','optimizer_before'):
        if rows(metrics/(key+'.jsonl'))!=rows(old_metrics/(key+'.jsonl')):
            raise RuntimeError('Initial shared state differs: '+key)
    new=rows(metrics/'activation_inputs.jsonl')[0]
    earlier=rows(old_metrics/'activation_inputs.jsonl')[0]
    if set(new['heldout_tokens']+[new['calibration_tokens']]) & set(earlier['heldout_tokens']+[earlier['calibration_tokens']]):
        raise RuntimeError('Evaluation data overlap')
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Checkpoint modified')
    result.update(checkpoint_unchanged=True,prior_training_replay_matched=True,
        heldout_microbatches=20,width=256,seed=1234,production_timing=False)
    (root/'results/BUDGET_COMPLETED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
