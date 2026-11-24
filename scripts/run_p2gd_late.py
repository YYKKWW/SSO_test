"""One continuation from step 252, then a predeclared 513--528 replay."""
import json
import os
from pathlib import Path

from run import checkpoint_manifest, run
from summarize import rows
from summarize_p2gd_late import late_summary
from verify_p2gd_late import verify_completed


def main():
    root=Path(os.environ['PROJECT_DIR']);old=Path(os.environ['ELR_REFERENCE_ROOT'])
    source=Path(os.environ['ELR_MATCHED_REFERENCE_ROOT'])
    prior=Path(os.environ['ELR_PRIOR_REPLAY_ROOT'])
    checkpoint=source/'matched_checkpoint'
    expected=json.loads((source/'results/matched_reference_manifest.json').read_text())
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Reference checkpoint changed')
    folder=root/'results/late_replay'
    if folder.exists():
        raise RuntimeError('Refusing duplicate or partial late replay')
    run(root,'late_replay',dict(LOAD_CHECKPOINT=str(checkpoint),CHECKPOINT_PATH=str(checkpoint),
        SAVE_CHECKPOINT='0',DATA_CACHE_PATH=str(old/'data_cache'),EVAL_INTERVAL='10000',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_late.py'),
        ELR_FIXED_VALID_BATCHES='21',ELR_HELDOUT_BATCHES='20',ELR_VALID_SKIP_BATCHES='96',
        ELR_ACTIVATION_TOKENS='256',ELR_FIRST='253',ELR_LAST='528',EXIT_INTERVAL='528',
        ELR_COMMON_PREFIX='0',ELR_ASSERT_LAYOUT='1',TRANSFORMER_IMPL='local',ELR_NEAR_THRESHOLD='2e-4'))
    metrics=folder/'metrics';old_metrics=prior/'results/budget_replay/metrics'
    integrity=verify_completed(metrics,old_metrics)
    first=rows(metrics/'activation_inputs.jsonl')[0]
    fresh=set(first['heldout_tokens']+[first['calibration_tokens']])
    for path in (old_metrics,Path(os.environ['ELR_SIX_REFERENCE_ROOT'])/'results/six_replay/metrics'):
        previous=rows(path/'activation_inputs.jsonl')[0]
        if fresh & set(previous['heldout_tokens']+[previous['calibration_tokens']]):
            raise RuntimeError('Heldout data overlap')
    result=late_summary(metrics)
    if result['completed_steps']!=list(range(513,529)):
        raise RuntimeError('Incomplete measurement window')
    if checkpoint_manifest(checkpoint,252)!=expected:
        raise RuntimeError('Checkpoint modified')
    result.update(integrity=integrity,checkpoint_unchanged=True,heldout_microbatches=20,
        width=256,seed=1234,production_timing=False)
    (root/'results/LATE_COMPLETED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
