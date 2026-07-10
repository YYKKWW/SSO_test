#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep}"

WIDTH="${WIDTH:-512}"
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
LRS="${LRS:-2e-2 3e-2}"

SPEL_PGD_BRANCH_MODE="${SPEL_PGD_BRANCH_MODE:-auto}"
SPEL_PGD_PROJECTION_MODE="${SPEL_PGD_PROJECTION_MODE:-shared_topk}"
SPEL_PGD_GAP_THRESHOLD_REL="${SPEL_PGD_GAP_THRESHOLD_REL:-1e-3}"
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-5}"
SPEL_PGD_DIRECTION_NORMALIZATION="${SPEL_PGD_DIRECTION_NORMALIZATION:-spectral}"
SPEL_PGD_PGD_LR_SCALE="${SPEL_PGD_PGD_LR_SCALE:-0.5}"

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

submit_sso() {
  local lr=$1
  local tag
  local min_lr
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")
  local job_name="sso_w${WIDTH}_lr${tag}_high"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spectral_ball_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0 \
    "$SBATCH_SCRIPT"
}

submit_spel_topk() {
  local lr=$1
  local rank=$2
  local tag
  local min_lr
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")
  local job_name="mcsd_tp_topk${rank}_w${WIDTH}_lr${tag}_high"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_tp_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PROJECTION_MODE=topk,SPEL_PROJECTION_RANK="$rank",SPEL_TANGENT_PROJECT_AFTER_MSIGN=1 \
    "$SBATCH_SCRIPT"
}

submit_pgd_shared_topk() {
  local lr=$1
  local rank=$2
  local tag
  local min_lr
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")
  local job_name="mcsd_pgd_shared-topk_k${rank}_w${WIDTH}_lr${tag}_high"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_pgd_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PGD_BRANCH_MODE="$SPEL_PGD_BRANCH_MODE",SPEL_PGD_PROJECTION_MODE="$SPEL_PGD_PROJECTION_MODE",SPEL_PGD_PROJECTION_RANK="$rank",SPEL_PGD_GAP_THRESHOLD_REL="$SPEL_PGD_GAP_THRESHOLD_REL",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="$SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS",SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION",SPEL_PGD_PGD_LR_SCALE="$SPEL_PGD_PGD_LR_SCALE" \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} high-LR projection sweep"
echo "  LRS=${LRS}"
echo "  configs: SSO, MCSD-TP/SpEL-TP topk k=4/8, MCSD-PGD shared_topk k=4/8"
echo "  TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "  RUN_ROOT=${RUN_ROOT}"

for lr in $LRS; do
  submit_sso "$lr"
  submit_spel_topk "$lr" 4
  submit_spel_topk "$lr" 8
  submit_pgd_shared_topk "$lr" 4
  submit_pgd_shared_topk "$lr" 8
done
