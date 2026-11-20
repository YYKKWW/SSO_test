"""Run 16 common updates, with eight same-state near-branch counterfactuals."""
import json
import os
from pathlib import Path

from p2gd_elr_geometry import METHODS
from run import checkpoint_manifest, run
from summarize import rows


def main():
    root = Path(os.environ['PROJECT_DIR'])
    old = Path(os.environ['ELR_REFERENCE_ROOT'])
    source = Path(os.environ['ELR_MATCHED_REFERENCE_ROOT'])
    checkpoint = source/'matched_checkpoint'
    expected = json.loads((source/'results/matched_reference_manifest.json').read_text())
    if checkpoint_manifest(checkpoint, 252) != expected:
        raise RuntimeError('Shared checkpoint changed')
    folder = root/'results/paired_replay'
    if folder.exists():
        raise RuntimeError('Refusing duplicate/partial replay; inspect before retry')
    config = dict(LOAD_CHECKPOINT=str(checkpoint), CHECKPOINT_PATH=str(checkpoint),
        SAVE_CHECKPOINT='0', DATA_CACHE_PATH=str(old/'data_cache'), EVAL_INTERVAL='10000',
        PRETRAIN_ENTRYPOINT=str(root/'studies/p2gd_elr/pretrain_p2gd_elr_replay.py'),
        ELR_FIXED_VALID_BATCHES='21', ELR_HELDOUT_BATCHES='20', ELR_VALID_SKIP_BATCHES='26',
        ELR_ACTIVATION_TOKENS='256', ELR_FIRST='253', ELR_LAST='268', EXIT_INTERVAL='268',
        ELR_COMMON_PREFIX='0', ELR_ASSERT_LAYOUT='1', TRANSFORMER_IMPL='local',
        ELR_NEAR_THRESHOLD='2e-4')
    run(root, 'paired_replay', config)
    loss = rows(folder/'metrics/repair_loss_replay.jsonl')
    geometry = rows(folder/'metrics/repair_geometry.jsonl')
    probes = rows(folder/'metrics/model_probe.jsonl')
    if [r['iteration'] for r in loss] != list(range(253,269)):
        raise RuntimeError('Missing replay iterations')
    if any(not r['restore_asserted'] for r in loss) or any(not r['optimizer_update_successful'] for r in probes):
        raise RuntimeError('Unsuccessful restoration or optimizer update')
    for r in loss:
        if [a['method'] for a in r['results']] != list(METHODS):
            raise RuntimeError('Missing counterfactual method')
    inputs = rows(folder/'metrics/activation_inputs.jsonl')[0]
    old_inputs = rows(Path(os.environ['ELR_CONFIRM_REFERENCE_ROOT'])/
        'results/confirm_2_mcsd_tp_aus/metrics/activation_inputs.jsonl')[0]
    if set(inputs['heldout_tokens']+[inputs['calibration_tokens']]) & set(
            old_inputs['heldout_tokens']+[old_inputs['calibration_tokens']]):
        raise RuntimeError('New diagnostic evaluation batches overlap old fixed batches')
    if checkpoint_manifest(checkpoint,252) != expected:
        raise RuntimeError('Shared checkpoint was modified')
    summaries = []
    informative = [r for r in loss if r['near_components']]
    for method in METHODS:
        selected = [next(a for a in r['results'] if a['method']==method) for r in informative]
        summaries.append(dict(method=method, informative_steps=len(selected),
            mean_loss_difference_vs_half=sum(r['heldout_difference_vs_half'] for r in selected)/len(selected)
                if selected else None,
            better_steps=sum(r['heldout_difference_vs_half'] < -1e-7 for r in selected),
            equal_steps=sum(abs(r['heldout_difference_vs_half']) <= 1e-7 for r in selected),
            changed_bf16_elements=sum(r['changed_bf16_elements_vs_half'] for r in selected)))
    result = dict(updates=16, width=256, seed=1234, matching='own-state half-D0 AUS',
        directions=list(METHODS), near_components=len(geometry),
        max_cluster_dimension=max((r['cluster_dimension'] for r in geometry), default=0),
        top4_overflow=sum(r['exceeds_top4'] for r in geometry),
        additional_training_tokens=16*128*4096, heldout_microbatches=20,
        checkpoint_unchanged=True, summaries=summaries,
        scope='Counterfactual complete LLM updates on one common trajectory; not eight training runs',
        production_timing=False)
    (root/'results/REPLAY_COMPLETED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
