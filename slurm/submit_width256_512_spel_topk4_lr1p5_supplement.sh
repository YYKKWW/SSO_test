#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"

NUM_LAYERS="${NUM_LAYERS:-28}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-128}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
TRAIN_TOKENS="${TRAIN_TOKENS:-1000000000}"
LR="${LR:-1.5e-2}"
MIN_LR="${MIN_LR:-1.5e-3}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

submit_spel_topk4() {
  local width=$1
  local run_root=$2
  local lr_tag=${LR//./p}
  lr_tag=${lr_tag//-/m}
  local job_name="mcsd_topk4_w${width}_lr${lr_tag}"

  mkdir -p "$run_root"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_dist",WIDTH="$width",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$LR",MIN_LR="$MIN_LR",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$run_root",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PROJECTION_MODE=topk,SPEL_PROJECTION_RANK=4 \
    "$SBATCH_SCRIPT"
}

echo "Submitting SpEL top-k k=4 LR=${LR} supplements for width 256 and 512"
echo "  TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, CPUS_PER_TASK=${CPUS_PER_TASK}"

submit_spel_topk4 256 "$PROJECT_DIR/results/olmo_1b_width256_spel_topk8_pgd_topk_lr_sweep"
submit_spel_topk4 512 "$PROJECT_DIR/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep"
