# Reproducing the Current Cluster Experiments

[Overview](../../README.md) · [Results](../experiments/mcsd_p2gd_results_summary.md) · [Algorithms](../algorithms/current_comparison_pseudocode.md)

This runbook covers the `exp/mcsd-experiment-results` branch. It does not
replace the frozen historical paper configuration.

## Environment and Data

The tested H20 environment uses **Python 3.12.1, PyTorch 2.6.0+cu124, and CUDA
12.4**. Follow the existing [H20 setup guide](h20_spel_megatron_runbook.md)
and environment scripts. Installing PyTorch alone is insufficient for the
complete Megatron training stack.

Launchers use HKU's `c_math_gpu` partition, `gpu` QoS,
`nvidia_h20:1`, Python/CUDA modules, and `$HOME/envs/sso_h20`. Adapt these
site settings for another cluster without silently changing algorithm settings.

Prepare the Megatron training/validation `.bin/.idx` prefixes and a local
OLMo-2 tokenizer. Data are not included. Current 1B/3B runs share the
**3B indexed source**; 1B denotes the processed-token budget, not the historical
paper's separate sample. Use the original data order to attempt exact
reproduction; another sample is a replication, not an identical run.

## Tests

From the repository root, inside the configured training environment:

```bash
export PYTHONPATH="$PWD/Megatron-LM:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
P2GD_TEST_DEVICE=cpu python -B -m unittest discover -s scripts -p 'test_cluster*.py' -v
```

The cluster suite has 74 tests, covering directions, returns, cache semantics,
small-SVD calls, launch grids, and Slurm-spooled wrapper paths. CUDA preflight
runs on the compute node. Unit tests do not replace an actual training smoke
or full-budget endpoint verification.
An additional result-collector regression test checks both intermediate and
final validation log formats; all 75 tests pass locally and on the H20 host's CPU.

## Explicit Paths and Source Snapshot

On the cluster, use a separate checkout per batch. Do not modify its code
while a job is running.

```bash
export PROJECT_DIR="$PWD"
export ENV_DIR="$HOME/envs/sso_h20"
export TRAIN_DATA_PREFIX="/path/to/indexed/train/olmo_train_text_document"
export VALID_DATA_PREFIX="/path/to/indexed/valid/olmo_valid_text_document"
export TOKENIZER_MODEL="/path/to/OLMo-2-tokenizer"
mkdir -p logs
git ls-files -z Megatron-LM scripts slurm | xargs -0 sha256sum > source.sha256
```

Data prefixes omit the `.bin/.idx` extension. The manifest verifies this
checkout, not an earlier training snapshot. Completed results retain their
original manifests separately.

## Select a Comparison

| Launcher | Scope | Evaluation |
|---|---|---|
| [Six-arm 1B](../../slurm/cluster_cone_lmo_1b_20261031.sbatch) | MuonSphere, SSO, control, wide/narrow hybrid, single-LMO; seed 2345 | 1908 steps, 20 batches |
| [Paired 1B](../../slurm/cluster_replication_1b_20261101.sbatch) | K10 control and wide c8 hybrid; seed 1234 | 1908 steps, 20 batches |
| [Mechanisms and LMO scales](../../slurm/cluster_mechanism_scales_1b_20261102.sbatch) | Ambient/top-1 c8 and single-LMO c0.5/c2; seed 2345 | 1908 steps, 20 batches |
| [LMO rank and shape](../../slurm/cluster_lmo_shape_1b_20261104.sbatch) | Same-LMO top-1 and 25% shape blend, c0.5; seed 2345 | Separate four-step smoke, then fresh 1908 steps, 20 batches |
| [First-seed 3B](../../slurm/cluster_selected_3b_20261025.sbatch) | Baselines plus K10/K20 control/hybrid pairs | 5723 steps, 20 batches |
| [Earlier 1B grid](../../slurm/cluster_candidates_1b_20261024.sbatch) | Two LRs, c4/c8/c12, K20/K40 | 1908 steps, 5 batches |

Inspect configurations without launching training:

```bash
SLURM_ARRAY_TASK_ID=3 bash slurm/cluster_cone_lmo_1b_20261031.sbatch --describe
SLURM_ARRAY_TASK_ID=0 bash slurm/cluster_replication_1b_20261101.sbatch --describe
```

Choose one experiment; do not submit all historical grids just to get started.
For the latest LMO rank/shape pair, after environment and data validation:

```bash
sbatch --export=ALL --array=0-1%2 slurm/cluster_lmo_shape_1b_20261104.sbatch
```

Each arm uses one H20, at most two concurrently. Training starts from
initialization, not a 1B checkpoint. Checkpoints are disabled in these batches.

## Outputs and Verification

Each arm writes `results/<array_id>_<index>/`:

| File | Meaning |
|---|---|
| `launch_config.json` | Effective settings, solver, data, seed, and runtime |
| `train.log` | Train/validation loss and branch statistics |
| `training_wall_seconds.txt` | Training plus validation, excluding preflight/audits |
| `TRAINING_COMPLETED` | Training command exited successfully |
| `COMPLETED` | Training and required postprocessing succeeded |
| `source.sha256` | Training-source checksums |
| `snapshots/spectral_call_counts.json` | Small/full SVD counters for experimental arms |
| `snapshots/offline_audit.json` | Selected-sample numerical audit |

Baselines do not generate the experimental spectral counters. Confirm the
Slurm exit code, completion markers, and final validation iteration before
reporting a result. Queue disappearance alone is not completion.

Use [the collector](../../scripts/collect_cluster_results.py) with explicit
completed run directories and an output path to obtain a traceable JSON.
The latest rank/shape figure is generated by
[the LMO-shape plotting script](../../scripts/plot_cluster_lmo_shape_results.py),
which checks protocol fields before plotting. The earlier mechanism/step-scale
figure is generated by
[the follow-up plotting script](../../scripts/plot_cluster_followup_results.py).
The [six-arm plotting script](../../scripts/plot_cluster_mechanism_results.py)
preserves the original reference figure. These scripts require Matplotlib in a plotting
environment, not in the training job, and read recorded validation points only.

## Offline Direction Audit

[The scale audit](../../slurm/cluster_lmo_scales_20261101.sbatch) reads saved
matrices; it does not retrain or change weights. Set `SNAPSHOT_ROOT` to the
completed six-arm result directory. It compares cached/oracle directions and
c0.5/c1/c2 with the same trial matrices and learning rates.

Snapshots are intentionally not published because of size. To reproduce this
audit, regenerate them with the observed training entry point. Published JSON
contains selected-sample results and hashes, not population error guarantees.
