#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"

WIDTHS="${WIDTHS:-256 512}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-128}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
TRAIN_TOKENS="${TRAIN_TOKENS:-1000000000}"
LRS="${LRS:-1.5e-2}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

SPEL_PGD_BRANCH_MODE="${SPEL_PGD_BRANCH_MODE:-auto}"
SPEL_PGD_GAP_THRESHOLD_REL="${SPEL_PGD_GAP_THRESHOLD_REL:-1e-3}"
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-3}"
SPEL_PGD_DIRECTION_NORMALIZATION="${SPEL_PGD_DIRECTION_NORMALIZATION:-none}"

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

run_root_for_width() {
  local width=$1
  printf "%s/results/olmo_1b_width%s_spel_mcsd_tp_pgd_projection_supplement" "$PROJECT_DIR" "$width"
}

submit_spel() {
  local width=$1
  local lr=$2
  local label=$3
  local projection_mode=$4
  local projection_rank=$5
  local tag
  local min_lr
  local run_root
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")
  run_root=$(run_root_for_width "$width")
  mkdir -p "$run_root"

  local job_name="spel_${label}_w${width}_lr${tag}"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_dist",WIDTH="$width",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$run_root",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PROJECTION_MODE="$projection_mode",SPEL_PROJECTION_RANK="$projection_rank",SPEL_TANGENT_PROJECT_AFTER_MSIGN=0 \
    "$SBATCH_SCRIPT"
}

submit_mcsd_tp_pgd() {
  local width=$1
  local lr=$2
  local label=$3
  local projection_mode=$4
  local projection_rank=$5
  local tag
  local min_lr
  local run_root
  tag=$(lr_tag "$lr")
  min_lr=$(min_lr_for "$lr")
  run_root=$(run_root_for_width "$width")
  mkdir -p "$run_root"

  local job_name="mcsd_tp_pgd_${label}_w${width}_lr${tag}"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_pgd_dist",WIDTH="$width",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$lr",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$run_root",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PGD_BRANCH_MODE="$SPEL_PGD_BRANCH_MODE",SPEL_PGD_PROJECTION_MODE="$projection_mode",SPEL_PGD_PROJECTION_RANK="$projection_rank",SPEL_PGD_GAP_THRESHOLD_REL="$SPEL_PGD_GAP_THRESHOLD_REL",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="$SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS",SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION",SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1 \
    "$SBATCH_SCRIPT"
}

echo "Submitting plain SpEL and MCSD-TP-PGD projection supplements"
echo "  WIDTHS=${WIDTHS}"
echo "  LRS=${LRS}"
echo "  projections: retraction, topk k=4, topk k=8"
echo "  TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, CPUS_PER_TASK=${CPUS_PER_TASK}"

for width in $WIDTHS; do
  for lr in $LRS; do
    submit_spel "$width" "$lr" "retr" "retraction" "1"
    submit_spel "$width" "$lr" "topk4" "topk" "4"
    submit_spel "$width" "$lr" "topk8" "topk" "8"

    submit_mcsd_tp_pgd "$width" "$lr" "shared-retr" "shared_retraction" "1"
    submit_mcsd_tp_pgd "$width" "$lr" "shared-topk_k4" "shared_topk" "4"
    submit_mcsd_tp_pgd "$width" "$lr" "shared-topk_k8" "shared_topk" "8"
  done
done
