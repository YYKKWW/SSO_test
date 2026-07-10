#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_250m_width256_spel_projection_ablation}"

WIDTH="${WIDTH:-256}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-128}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
TRAIN_TOKENS="${TRAIN_TOKENS:-250000000}"
LR="${LR:-1.5e-2}"
MIN_LR="${MIN_LR:-1.5e-3}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-50}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SPEL_PGD_GAP_THRESHOLD_REL="${SPEL_PGD_GAP_THRESHOLD_REL:-1e-3}"
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-5}"
SPEL_PGD_DIRECTION_NORMALIZATION="${SPEL_PGD_DIRECTION_NORMALIZATION:-spectral}"
SPEL_PGD_PGD_LR_SCALE="${SPEL_PGD_PGD_LR_SCALE:-0.5}"

mkdir -p "$RUN_ROOT"

submit_spel() {
  local label=$1
  local projection_mode=$2
  local projection_rank=$3
  local job_name="spel_tp_proj_w${WIDTH}_${label}"

  sbatch \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_tp_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$LR",MIN_LR="$MIN_LR",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PROJECTION_MODE="$projection_mode",SPEL_PROJECTION_RANK="$projection_rank",SPEL_TANGENT_PROJECT_AFTER_MSIGN=1 \
    "$SBATCH_SCRIPT"
}

submit_spel_pgd() {
  local label=$1
  local projection_mode=$2
  local projection_rank=$3
  local job_name="spelpgdtopk_w${WIDTH}_${label}"

  sbatch \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_pgd_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$LR",MIN_LR="$MIN_LR",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PGD_BRANCH_MODE=auto,SPEL_PGD_PROJECTION_MODE="$projection_mode",SPEL_PGD_PROJECTION_RANK="$projection_rank",SPEL_PGD_GAP_THRESHOLD_REL="$SPEL_PGD_GAP_THRESHOLD_REL",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="$SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS",SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION",SPEL_PGD_PGD_LR_SCALE="$SPEL_PGD_PGD_LR_SCALE" \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} MCSD-TP/SpEL-TP projection and SpEL-PGD top-k ablation"
echo "  LR=${LR}, TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}"
echo "  RUN_ROOT=${RUN_ROOT}"

# MCSD-TP/SpEL-TP canonical projection ablation.
submit_spel "retr" "retraction" "1"
submit_spel "exact" "exact" "1"
submit_spel "topk_k2" "topk" "2"
submit_spel "topk_k4" "topk" "4"
submit_spel "topk_k8" "topk" "8"

# SpEL-PGD with top-k projection. fallback_topk is the previous best rule;
# shared_topk applies top-k projection to both the safe SpEL branch and PGD fallback.
submit_spel_pgd "fallback_topk_k4" "fallback_topk" "4"
submit_spel_pgd "shared_topk_k2" "shared_topk" "2"
submit_spel_pgd "shared_topk_k4" "shared_topk" "4"
submit_spel_pgd "shared_topk_k8" "shared_topk" "8"
