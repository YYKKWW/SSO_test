#!/bin/bash
#SBATCH --job-name=w256_cluster_followup_100m
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --array=0-13%2
#SBATCH --output=logs/followup_%A_%a.out
#SBATCH --error=logs/followup_%A_%a.err
set -euo pipefail

optimizer=spel_pgd_dist
mode=fixed4
coefficient=1.0
gap=5e-5
cluster=5e-5
seed=1234
case "${SLURM_ARRAY_TASK_ID:?An explicit array index is required}" in
  0) optimizer=muon_ball_dist; mode=none ;;
  1) optimizer=spectral_ball_dist; mode=none ;;
  2) mode=fixed2 ;;
  3) mode=fixed2; coefficient=0.5 ;;
  4) coefficient=0.5 ;;
  5) coefficient=0.25 ;;
  6) mode=adaptive4; cluster=5e-4 ;;
  7) mode=adaptive4; cluster=5e-4; coefficient=0.5 ;;
  8) mode=adaptive4; cluster=2e-3 ;;
  9) mode=adaptive4; cluster=2e-3; coefficient=0.5 ;;
  10) gap=1e-4; cluster=1e-4 ;;
  11) mode=adaptive4; gap=1e-4; cluster=5e-4 ;;
  12) mode=control; seed=2345 ;;
  13) seed=2345 ;;
  *) echo "Unknown experiment index" >&2; exit 2 ;;
esac
# Describe without loading CUDA, accessing data, or submitting anything.
if [[ "${1:-}" == "--describe" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SLURM_ARRAY_TASK_ID" "$optimizer" "$mode" "$gap" "$cluster" "$coefficient" "$seed"
  exit 0
fi

: "${PROJECT_DIR:?Use an isolated, verified source snapshot}"
: "${TRAIN_DATA_PREFIX:?Explicitly select the shared training source}"
: "${VALID_DATA_PREFIX:?Explicitly select the shared validation source}"
: "${TOKENIZER_MODEL:?Explicitly select the tokenizer}"
cd "$PROJECT_DIR"
sha256sum --check source.sha256
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p test_cluster_radial_production.py -v

export OPTIMIZER="$optimizer" SEED="$seed"
export WIDTH=256 NUM_LAYERS=28 HEAD_DIM=128 SEQ_LENGTH=4096
export GLOBAL_BATCH=128 MICRO_BATCH=4 GPUS_PER_NODE=1 TP_SIZE=1 PP_SIZE=1
export TRAIN_ITER=191 TRAIN_TOKENS=100000000
export LR=1.5e-2 MIN_LR=1.5e-3 LR_WARMUP_ITERS=25 LR_DECAY_ITERS=191
export EVAL_INTERVAL=50 EVAL_ITERS=5 LOG_INTERVAL=10
export SAVE_CHECKPOINT=0 WEIGHT_DECAY=0.1
export SPEL_PGD_BRANCH_MODE=auto SPEL_PGD_GAP_THRESHOLD_REL="$gap"
export SPEL_PGD_CLUSTER_THRESHOLD_REL="$cluster"
export SPEL_PGD_RADIAL_CLUSTER_MODE="$mode" SPEL_PGD_PGD_LR_SCALE="$coefficient"
export SPEL_PGD_PROJECTION_MODE=tp_radial_pgd_exact
export SPEL_PGD_POST_RETURN_SPECTRAL_CACHE=1
export SPEL_PGD_GAP_ESTIMATOR_MODE=exact_svd_gap_only
export SPEL_PGD_GAP_PROBE_INTERVAL=1 SPEL_PGD_GAP_AUDIT_INTERVAL=0
export SPEL_PGD_MAIN_POWER_DTYPE=fp32 SPEL_PGD_POWER_ITERATION_STEPS=10
export SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10 SPEL_PGD_MSIGN_STEPS=8
export SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1 SPEL_PGD_NORMALIZE_MSIGN_SPECTRAL=0
export SPEL_PGD_MATRIX_WEIGHT_DECAY=0 SPEL_PGD_DIRECTION_NORMALIZATION=none
export SPEL_PGD_FALLBACK_DIRECTION=pgd SPEL_PGD_WARM_START_UV=0
# Baselines keep their original PI10 / NS8 hard-retraction implementations.
export SPECTRAL_BALL_POWER_ITERATION_STEPS=10
export SPECTRAL_BALL_SOLVER_TOLERANCE_F=2e-4 SPECTRAL_BALL_SOLVER_MAX_ITERATIONS=20
export MUON_BALL_POWER_ITERATION_STEPS=10 MUON_BALL_MSIGN_STEPS=8
export JOB_NAME="w256_${optimizer}_${mode}_g${gap}_t${cluster}_c${coefficient}_s${seed}_100m"
export RUN_ROOT="$PROJECT_DIR/results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export DATA_CACHE_PATH="$RUN_ROOT/data_cache"
mkdir -p "$RUN_ROOT"
cp source.sha256 "$RUN_ROOT/source.sha256"
python -c 'import json, os, torch
keys = {
    "WIDTH", "NUM_LAYERS", "HEAD_DIM", "SEQ_LENGTH", "GLOBAL_BATCH",
    "MICRO_BATCH", "TRAIN_ITER", "TRAIN_TOKENS", "LR", "MIN_LR",
    "LR_WARMUP_ITERS", "LR_DECAY_ITERS", "EVAL_INTERVAL", "EVAL_ITERS",
    "SEED", "WEIGHT_DECAY", "TRAIN_DATA_PREFIX", "VALID_DATA_PREFIX",
    "TOKENIZER_MODEL", "MEGATRON_PATH", "OPTIMIZER", "DATA_CACHE_PATH",
    "NVIDIA_TF32_OVERRIDE", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"
}
config = {k: v for k, v in os.environ.items() if k in keys or
          k.startswith(("SPEL_PGD_", "MUON_BALL_", "SPECTRAL_BALL_"))}
experimental = os.environ["OPTIMIZER"] == "spel_pgd_dist"
config.update(
    actual_return="radial_both_branches" if experimental else "reference_hard_pre_retraction",
    matched_frobenius_length=experimental,
    cluster_options_active=experimental,
    control_source_array="4290663",
    torch_version=torch.__version__, cuda_version=torch.version.cuda,
    device=torch.cuda.get_device_name(0),
)
print(json.dumps(config, indent=2, sort_keys=True))' > "$RUN_ROOT/launch_config.json"
bash "$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch" 2>&1 | tee "$RUN_ROOT/train.log"
touch "$RUN_ROOT/COMPLETED"
