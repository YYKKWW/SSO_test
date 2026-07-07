#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_1b_width256_sso_mcsd_lr_sweep}"

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
LRS="${LRS:-5e-3 7e-3 9e-3 1e-2 1.5e-2}"

mkdir -p "$RUN_ROOT"

submit_one() {
  local label=$1
  local optimizer=$2
  local lr=$3
  local lr_tag=${lr//./p}
  lr_tag=${lr_tag//-/m}
  local job_name="${label}_w${WIDTH}_lr${lr_tag}"
  local min_lr
  min_lr=$(awk -v lr="$lr" 'BEGIN { printf "%.8g", lr / 10.0 }')

  sbatch \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="$optimizer",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0 \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} SSO vs MCSD-TP LR sweep"
echo "  LRS=${LRS}"
echo "  GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, TRAIN_TOKENS=${TRAIN_TOKENS}"
echo "  RUN_ROOT=${RUN_ROOT}"

for lr in $LRS; do
  submit_one "sso" "spectral_ball_dist" "$lr"
done

for lr in $LRS; do
  submit_one "mcsd_tp" "spel_tp_dist" "$lr"
done
