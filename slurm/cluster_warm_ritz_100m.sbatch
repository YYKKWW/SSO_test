#!/bin/bash
#SBATCH --job-name=w256_cluster_ritz_100m
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/ritz_%A_%a.out
#SBATCH --error=logs/ritz_%A_%a.err
set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0) mode=control; iterations=10 ;;
  1) mode=adaptive4; iterations=10 ;;
  2) mode=control; iterations=20 ;;
  3) mode=adaptive4; iterations=20 ;;
  *) echo "Unknown experiment index" >&2; exit 2 ;;
esac
if [[ "${1:-}" == "--describe" ]]; then
  printf '%s\t%s\t%s\t%s\n' "$SLURM_ARRAY_TASK_ID" "$mode" warm_block8 "$iterations"
  exit 0
fi

: "${PROJECT_DIR:?Use an isolated source snapshot}"
: "${TRAIN_DATA_PREFIX:?Explicit training data prefix required}"
: "${VALID_DATA_PREFIX:?Explicit validation data prefix required}"
: "${TOKENIZER_MODEL:?Explicit tokenizer required}"
cd "$PROJECT_DIR"
sha256sum --check source.sha256 > "logs/source_check_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p test_cluster_warm_ritz.py -v
python -B -m unittest discover -s scripts -p test_cluster_radial_production.py -v

export OPTIMIZER=spel_pgd_dist SEED=1234
export WIDTH=256 NUM_LAYERS=28 HEAD_DIM=128 SEQ_LENGTH=4096
export GLOBAL_BATCH=128 MICRO_BATCH=4 GPUS_PER_NODE=1 TP_SIZE=1 PP_SIZE=1
export TRAIN_ITER=191 TRAIN_TOKENS=100000000
export LR=1.5e-2 MIN_LR=1.5e-3 LR_WARMUP_ITERS=25 LR_DECAY_ITERS=191
export EVAL_INTERVAL=50 EVAL_ITERS=5 LOG_INTERVAL=10
export SAVE_CHECKPOINT=0 WEIGHT_DECAY=0.1
export SPEL_PGD_BRANCH_MODE=auto SPEL_PGD_GAP_THRESHOLD_REL=5e-5
export SPEL_PGD_CLUSTER_THRESHOLD_REL=2e-3
export SPEL_PGD_RADIAL_CLUSTER_MODE="$mode" SPEL_PGD_PGD_LR_SCALE=1.0
export SPEL_PGD_CLUSTER_SPECTRUM_MODE=warm_block8 SPEL_PGD_CLUSTER_ITERATIONS="$iterations"
# Legacy routing flags select the host dispatch; the cluster module overrides
# the spectral solver and both returns. No full SVD runs in optimizer updates.
export SPEL_PGD_PROJECTION_MODE=tp_radial_pgd_exact
export SPEL_PGD_POST_RETURN_SPECTRAL_CACHE=1
export SPEL_PGD_GAP_ESTIMATOR_MODE=exact_svd_gap_only
export SPEL_PGD_GAP_PROBE_INTERVAL=1 SPEL_PGD_GAP_AUDIT_INTERVAL=0
export SPEL_PGD_MAIN_POWER_DTYPE=fp32 SPEL_PGD_POWER_ITERATION_STEPS=10
export SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10 SPEL_PGD_MSIGN_STEPS=8
export SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1 SPEL_PGD_NORMALIZE_MSIGN_SPECTRAL=0
export SPEL_PGD_MATRIX_WEIGHT_DECAY=0 SPEL_PGD_DIRECTION_NORMALIZATION=none
export SPEL_PGD_FALLBACK_DIRECTION=pgd SPEL_PGD_WARM_START_UV=0
export JOB_NAME="w256_ritz8_k${iterations}_${mode}_g5e-5_t2e-3_c1_100m"
export RUN_ROOT="$PROJECT_DIR/results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export DATA_CACHE_PATH="$RUN_ROOT/data_cache"
export CLUSTER_SNAPSHOT_DIR="$RUN_ROOT/snapshots"
export PRETRAIN_ENTRYPOINT="$PROJECT_DIR/scripts/pretrain_cluster_ritz_observed.py"
mkdir -p "$RUN_ROOT"
cp source.sha256 "$RUN_ROOT/source.sha256"
python -c 'import json, os, torch
keys = {"WIDTH", "NUM_LAYERS", "HEAD_DIM", "SEQ_LENGTH", "GLOBAL_BATCH", "MICRO_BATCH",
        "TRAIN_ITER", "TRAIN_TOKENS", "LR", "MIN_LR", "LR_WARMUP_ITERS", "LR_DECAY_ITERS",
        "EVAL_INTERVAL", "EVAL_ITERS", "SEED", "WEIGHT_DECAY", "TRAIN_DATA_PREFIX",
        "VALID_DATA_PREFIX", "TOKENIZER_MODEL", "MEGATRON_PATH", "OPTIMIZER", "DATA_CACHE_PATH",
        "NVIDIA_TF32_OVERRIDE", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "PRETRAIN_ENTRYPOINT"}
config = {k: v for k,v in os.environ.items() if k in keys or k.startswith("SPEL_PGD_")}
config.update(actual_spectral_solver="fixed_warm_block8_fp32_ritz",
              actual_return="approximate_radial_both_branches", online_accuracy_gate=False,
              automatic_full_svd_fallback=False, matched_frobenius_length=True,
              offline_audit="after_training_only", reference_arrays=["4290663", "4290694"],
              torch_version=torch.__version__, cuda_version=torch.version.cuda,
              device=torch.cuda.get_device_name(0))
print(json.dumps(config, indent=2, sort_keys=True))' > "$RUN_ROOT/launch_config.json"
start=$(date +%s)
bash "$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch" 2>&1 | tee "$RUN_ROOT/train.log"
elapsed=$(( $(date +%s) - start ))
printf '%s\n' "$elapsed" > "$RUN_ROOT/training_wall_seconds.txt"
touch "$RUN_ROOT/TRAINING_COMPLETED"
python -c 'import json, os
from pathlib import Path
p = Path(os.environ["CLUSTER_SNAPSHOT_DIR"])
c = json.loads((p / "spectral_call_counts.json").read_text())
assert c["successful_training"] and c["component_updates"] > 0, c
assert c["full_svd"] == c["svdvals"] == 0, c
assert c["small_svd"] == c["component_updates"], c
print(c)'
# Exact decompositions below are offline diagnostics, excluded from training
# wall time, and cannot change any update or branch decision.
python scripts/audit_cluster_ritz_snapshots.py "$CLUSTER_SNAPSHOT_DIR" --device cuda \
  2>&1 | tee "$RUN_ROOT/offline_audit.log"
touch "$RUN_ROOT/COMPLETED"
