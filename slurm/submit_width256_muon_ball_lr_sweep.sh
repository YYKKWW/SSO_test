#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_1b_width256_muon_ball_lr_sweep}"

WIDTH="${WIDTH:-256}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-128}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
TRAIN_TOKENS="${TRAIN_TOKENS:-1000000000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
LRS="${LRS:-5e-3 7e-3 9e-3 1e-2 1.5e-2 2e-2 3e-2}"

MUON_BALL_MSIGN_STEPS="${MUON_BALL_MSIGN_STEPS:-8}"
MUON_BALL_POWER_ITERATION_STEPS="${MUON_BALL_POWER_ITERATION_STEPS:-10}"
MUON_BALL_RETRACT_ALPHA="${MUON_BALL_RETRACT_ALPHA:-0.05}"

mkdir -p "$RUN_ROOT"

lr_tag() {
  local lr=$1
  lr=${lr//./p}
  lr=${lr//-/m}
  printf "%s" "$lr"
}

min_lr_for() {
  local lr=$1
  awk -v lr="$lr" 'BEGIN { printf "%.8g", lr / 10.0 }'
}

submit_one() {
  local lr=$1
  local tag
  local min_lr
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")

  local job_name="muon_ball_w${WIDTH}_lr${tag}"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="muon_ball_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,MUON_BALL_MSIGN_STEPS="$MUON_BALL_MSIGN_STEPS",MUON_BALL_POWER_ITERATION_STEPS="$MUON_BALL_POWER_ITERATION_STEPS",MUON_BALL_RETRACT_ALPHA="$MUON_BALL_RETRACT_ALPHA" \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} MuonBall LR sweep"
echo "  LRS=${LRS}"
echo "  constants: momentum=0.9, nesterov=on, msign_steps=${MUON_BALL_MSIGN_STEPS}, radius/scale=spectral_mup, power_iteration_steps=${MUON_BALL_POWER_ITERATION_STEPS}, retract=hard"
echo "  TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "  RUN_ROOT=${RUN_ROOT}"

for lr in $LRS; do
  submit_one "$lr"
done
