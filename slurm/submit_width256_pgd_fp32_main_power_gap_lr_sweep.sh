#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/submit_width256_pgd_gap_threshold_tune.sh}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$PROJECT_DIR/results/olmo_250m_width256_pgd_fp32_main_power_gap_lr_sweep}"

# Test whether running the ordinary MCSD/SpEL top-vector power iteration in
# FP32 changes the width=256 SpEL-PGD behavior.  The gap=0 cases are no-PGD
# baselines, so they are submitted once per configuration; gap=1e-4 sweeps the
# PGD fallback step size.
SPEL_PGD_MAIN_POWER_DTYPE="${SPEL_PGD_MAIN_POWER_DTYPE:-fp32}"
SPEL_PGD_ACTIVE_GAPS="${SPEL_PGD_ACTIVE_GAPS:-1e-4}"
SPEL_PGD_BASELINE_GAPS="${SPEL_PGD_BASELINE_GAPS:-0}"
SPEL_PGD_PGD_LR_SCALES="${SPEL_PGD_PGD_LR_SCALES:-0.2 0.5 1}"
SPEL_PGD_BASELINE_PGD_LR_SCALE="${SPEL_PGD_BASELINE_PGD_LR_SCALE:-0.5}"

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
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

SPEL_PGD_BRANCH_MODE="${SPEL_PGD_BRANCH_MODE:-auto}"
SPEL_PGD_PROJECTION_MODE="${SPEL_PGD_PROJECTION_MODE:-shared_topk}"
SPEL_PGD_PROJECTION_RANK="${SPEL_PGD_PROJECTION_RANK:-8}"
SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-10}"
SPEL_PGD_DIRECTION_NORMALIZATION="${SPEL_PGD_DIRECTION_NORMALIZATION:-spectral}"
SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN="${SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN:-0}"

tag_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  printf "%s" "$value"
}

submit_one() {
  local estimator=$1
  local warm=$2
  local gap=$3
  local scale=$4
  local warm_tag="cold"
  local gap_tag
  local scale_tag
  local main_dtype_tag
  if [[ "$warm" == "1" || "$warm" == "true" ]]; then
    warm_tag="warm"
  fi
  gap_tag=$(tag_value "$gap")
  scale_tag=$(tag_value "$scale")
  main_dtype_tag=${SPEL_PGD_MAIN_POWER_DTYPE//_/-}

  RUN_ROOT="$RUN_ROOT_BASE/${estimator}_${main_dtype_tag}_${warm_tag}_gap${gap_tag}_pgdlr${scale_tag}" \
  WIDTH="$WIDTH" \
  NUM_LAYERS="$NUM_LAYERS" \
  SEQ_LENGTH="$SEQ_LENGTH" \
  HEAD_DIM="$HEAD_DIM" \
  GLOBAL_BATCH="$GLOBAL_BATCH" \
  MICRO_BATCH="$MICRO_BATCH" \
  TRAIN_TOKENS="$TRAIN_TOKENS" \
  LR="$LR" \
  MIN_LR="$MIN_LR" \
  LR_WARMUP_ITERS="$LR_WARMUP_ITERS" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  EVAL_ITERS="$EVAL_ITERS" \
  LOG_INTERVAL="$LOG_INTERVAL" \
  CPUS_PER_TASK="$CPUS_PER_TASK" \
  SPEL_PGD_BRANCH_MODE="$SPEL_PGD_BRANCH_MODE" \
  SPEL_PGD_PROJECTION_MODE="$SPEL_PGD_PROJECTION_MODE" \
  SPEL_PGD_PROJECTION_RANK="$SPEL_PGD_PROJECTION_RANK" \
  SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="$SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS" \
  SPEL_PGD_GAP_ESTIMATOR_MODE="$estimator" \
  SPEL_PGD_MAIN_POWER_DTYPE="$SPEL_PGD_MAIN_POWER_DTYPE" \
  SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION" \
  SPEL_PGD_PGD_LR_SCALE="$scale" \
  SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN="$SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN" \
  SPEL_PGD_WARM_START_UV="$warm" \
  SPEL_PGD_GAP_THRESHOLDS="$gap" \
  bash "$SUBMIT_SCRIPT"
}

submit_config() {
  local estimator=$1
  local warm=$2
  local gap
  local scale
  for gap in $SPEL_PGD_BASELINE_GAPS; do
    submit_one "$estimator" "$warm" "$gap" "$SPEL_PGD_BASELINE_PGD_LR_SCALE"
  done
  for gap in $SPEL_PGD_ACTIVE_GAPS; do
    for scale in $SPEL_PGD_PGD_LR_SCALES; do
      submit_one "$estimator" "$warm" "$gap" "$scale"
    done
  done
}

echo "Submitting width=${WIDTH} SpEL-PGD FP32 main-power sweep"
echo "  main_power_dtype=${SPEL_PGD_MAIN_POWER_DTYPE}"
echo "  baseline_gaps=${SPEL_PGD_BASELINE_GAPS}, active_gaps=${SPEL_PGD_ACTIVE_GAPS}"
echo "  pgd_lr_scales=${SPEL_PGD_PGD_LR_SCALES}, baseline_pgdlr=${SPEL_PGD_BASELINE_PGD_LR_SCALE}"
echo "  sigma2_steps=${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS}"
echo "  projection=${SPEL_PGD_PROJECTION_MODE}, rank=${SPEL_PGD_PROJECTION_RANK}"
echo "  train_tokens=${TRAIN_TOKENS}, lr=${LR}, run_root_base=${RUN_ROOT_BASE}"

submit_config "block2_fp32_gap_only" "0"
submit_config "block2_fp32_gap_only" "1"
submit_config "block2_fp32" "1"
