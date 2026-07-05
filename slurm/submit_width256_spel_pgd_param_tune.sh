#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_250m_width256_spel_pgd_param_tune}"

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

mkdir -p "$RUN_ROOT"

submit_one() {
  local label=$1
  local branch_mode=$2
  local projection_mode=$3
  local gap=$4
  local direction_norm=$5
  local job_name="pgdtune_w${WIDTH}_${label}"

  sbatch \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_pgd_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$LR",MIN_LR="$MIN_LR",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PGD_BRANCH_MODE="$branch_mode",SPEL_PGD_PROJECTION_MODE="$projection_mode",SPEL_PGD_GAP_THRESHOLD_REL="$gap",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-3}",SPEL_PGD_DIRECTION_NORMALIZATION="$direction_norm" \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} SpEL-PGD parameter tuning"
echo "  LR=${LR}, TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}"
echo "  RUN_ROOT=${RUN_ROOT}"

# Baseline: optimized SpEL-PGD should match original SpEL exactly in the safe branch.
submit_one "spel_base" "spel" "fallback_exact" "5e-3" "none"

# Exact PGD fallback: tune gap threshold around the recommended range.
submit_one "exact_g1em4" "auto" "fallback_exact" "1e-4" "none"
submit_one "exact_g1em3" "auto" "fallback_exact" "1e-3" "none"
submit_one "exact_g5em3" "auto" "fallback_exact" "5e-3" "none"
submit_one "exact_g1em2" "auto" "fallback_exact" "1e-2" "none"

# Direction-scale ablation for PGD fallback.
submit_one "exact_g5em3_fro" "auto" "fallback_exact" "5e-3" "fro"

# Cheap fallback for scalable LLM runs.
submit_one "retr_g5em3" "auto" "fallback_retraction" "5e-3" "none"
submit_one "retr_g1em2" "auto" "fallback_retraction" "1e-2" "none"
