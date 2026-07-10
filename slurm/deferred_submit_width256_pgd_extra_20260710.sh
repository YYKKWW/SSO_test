#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/submit_width256_pgd_gap_threshold_tune.sh}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$PROJECT_DIR/results/olmo_250m_width256_pgd_deferred_extra_20260710}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-26}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-28800}"

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

active_jobs() {
  squeue -h -u "$USER" | wc -l | tr -d ' '
}

submit_one() {
  local estimator=$1
  local warm=$2
  local scale=$3
  local gap=$4
  local group=$5
  local scale_tag
  local warm_tag
  scale_tag=$(tag_value "$scale")
  warm_tag="cold"
  if [[ "$warm" == "1" || "$warm" == "true" ]]; then
    warm_tag="warm"
  fi

  echo "[$(date)] submitting estimator=${estimator} warm=${warm} scale=${scale} gap=${gap} group=${group}"
  RUN_ROOT="$RUN_ROOT_BASE/${group}/${estimator}_${warm_tag}_pgdlr${scale_tag}" \
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
  SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION" \
  SPEL_PGD_PGD_LR_SCALE="$scale" \
  SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN="$SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN" \
  SPEL_PGD_WARM_START_UV="$warm" \
  SPEL_PGD_GAP_THRESHOLDS="$gap" \
  bash "$SUBMIT_SCRIPT"
}

tasks=(
  "block2_fp32_gap_only|1|0.05|1e-4|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.05|3e-4|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.1|3e-5|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.1|1e-4|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.1|3e-4|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.2|3e-5|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.2|1e-4|gap_only_lrscale_remaining"
  "block2_fp32_gap_only|1|0.2|3e-4|gap_only_lrscale_remaining"
  "deflated_power|0|0.5|0|deflated_power_baseline"
  "deflated_power|0|0.5|3e-5|deflated_power_baseline"
  "deflated_power|0|0.5|1e-4|deflated_power_baseline"
  "deflated_power|0|0.5|3e-4|deflated_power_baseline"
  "deflated_power|1|0.5|0|deflated_power_baseline"
  "deflated_power|1|0.5|3e-5|deflated_power_baseline"
  "deflated_power|1|0.5|1e-4|deflated_power_baseline"
  "deflated_power|1|0.5|3e-4|deflated_power_baseline"
)

start_time=$(date +%s)
echo "[$(date)] deferred submitter started with ${#tasks[@]} tasks, max_active=${MAX_ACTIVE_JOBS}"

for task in "${tasks[@]}"; do
  IFS='|' read -r estimator warm scale gap group <<< "$task"
  while true; do
    now=$(date +%s)
    if (( now - start_time > DEADLINE_SECONDS )); then
      echo "[$(date)] deadline reached before submitting: $task"
      exit 0
    fi

    current=$(active_jobs)
    if (( current < MAX_ACTIVE_JOBS )); then
      if submit_one "$estimator" "$warm" "$scale" "$gap" "$group"; then
        break
      fi
      echo "[$(date)] submit failed, probably QOS limit; retrying after sleep"
    else
      echo "[$(date)] active jobs=${current}, waiting for below ${MAX_ACTIVE_JOBS}"
    fi
    sleep "$SLEEP_SECONDS"
  done
done

echo "[$(date)] deferred submitter completed all tasks"
