#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$PROJECT_DIR/results/primary_1b_matrix}"
DRY_RUN="${DRY_RUN:-1}"
BATCH="${1:-}"

LRS_ALL="5e-3 7e-3 9e-3 1e-2 1.5e-2 2e-2 3e-2"
LRS_MISSING_SPEL_256_512="5e-3 7e-3 9e-3 1e-2 2e-2 3e-2"

tag_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  printf "%s" "$value"
}

min_lr_for() {
  local lr=$1
  awk -v lr="$lr" 'BEGIN { printf "%.8g", lr / 10.0 }'
}

resources_for_width() {
  local width=$1
  if [[ "$width" == "1024" ]]; then
    JOB_TIME="2-00:00:00"
    CPUS_PER_TASK=12
  elif [[ "$width" == "512" ]]; then
    JOB_TIME="1-00:00:00"
    CPUS_PER_TASK=12
  else
    JOB_TIME="1-00:00:00"
    CPUS_PER_TASK=8
  fi
}

run_sbatch() {
  local job_name=$1
  local export_values=$2
  local job_time=$3
  local cpus=$4
  local command=(
    sbatch
    --time="$job_time"
    --cpus-per-task="$cpus"
    -J "$job_name"
    --export="$export_values"
    "$SBATCH_SCRIPT"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY-RUN"
    printf " %q" "${command[@]}"
    printf "\n"
  else
    "${command[@]}"
  fi
}

common_exports() {
  local width=$1
  local lr=$2
  local run_root=$3
  local job_name=$4
  local min_lr
  min_lr=$(min_lr_for "$lr")
  printf "%s" "ALL,WIDTH=$width,NUM_LAYERS=28,HEAD_DIM=128,SEQ_LENGTH=4096,GLOBAL_BATCH=128,MICRO_BATCH=4,TRAIN_TOKENS=1000000000,LR=$lr,MIN_LR=$min_lr,LR_WARMUP_ITERS=250,EVAL_INTERVAL=250,EVAL_ITERS=5,LOG_INTERVAL=10,SEED=1234,RUN_ROOT=$run_root,JOB_NAME=$job_name,SAVE_CHECKPOINT=0"
}

submit_sso() {
  local width=$1
  local lr=$2
  local tag
  local job_name
  local run_root
  local exports
  tag=$(tag_value "$lr")
  job_name="primary_sso_w${width}_lr${tag}"
  run_root="$RUN_ROOT_BASE/w${width}/sso"
  resources_for_width "$width"
  exports="$(common_exports "$width" "$lr" "$run_root" "$job_name"),OPTIMIZER=spectral_ball_dist"
  run_sbatch "$job_name" "$exports" "$JOB_TIME" "$CPUS_PER_TASK"
}

submit_spel() {
  local width=$1
  local lr=$2
  local tag
  local job_name
  local run_root
  local exports
  tag=$(tag_value "$lr")
  job_name="primary_spel_bf16_topk8_w${width}_lr${tag}"
  run_root="$RUN_ROOT_BASE/w${width}/spel_bf16_topk8"
  resources_for_width "$width"
  exports="$(common_exports "$width" "$lr" "$run_root" "$job_name"),OPTIMIZER=spel_dist,SPEL_PROJECTION_MODE=topk,SPEL_PROJECTION_RANK=8,SPEL_TANGENT_PROJECT_AFTER_MSIGN=0"
  run_sbatch "$job_name" "$exports" "$JOB_TIME" "$CPUS_PER_TASK"
}

submit_pgd() {
  local width=$1
  local lr=$2
  local tag
  local job_name
  local run_root
  local exports
  tag=$(tag_value "$lr")
  job_name="primary_mcsd_pgd_bf16_gapfp32_k8_w${width}_lr${tag}"
  run_root="$RUN_ROOT_BASE/w${width}/mcsd_pgd_bf16_gapfp32_k8"
  resources_for_width "$width"
  exports="$(common_exports "$width" "$lr" "$run_root" "$job_name"),OPTIMIZER=spel_pgd_dist,SPEL_PGD_BRANCH_MODE=auto,SPEL_PGD_PROJECTION_MODE=shared_topk,SPEL_PGD_PROJECTION_RANK=8,SPEL_PGD_GAP_THRESHOLD_REL=3e-4,SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10,SPEL_PGD_GAP_ESTIMATOR_MODE=block2_fp32_gap_only,SPEL_PGD_GAP_PROBE_INTERVAL=5,SPEL_PGD_GAP_PROBE_SAFE_MULTIPLIER=10,SPEL_PGD_MAIN_POWER_DTYPE=bf16,SPEL_PGD_DIRECTION_NORMALIZATION=spectral,SPEL_PGD_PGD_LR_SCALE=0.5,SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0,SPEL_PGD_WARM_START_UV=0"
  run_sbatch "$job_name" "$exports" "$JOB_TIME" "$CPUS_PER_TASK"
}

submit_pgd_smoke_1024() {
  local lr="1.5e-2"
  local job_name="primary_mcsd_pgd_w1024_smoke"
  local run_root="$RUN_ROOT_BASE/smoke/mcsd_pgd_w1024"
  local exports
  exports="$(common_exports 1024 "$lr" "$run_root" "$job_name"),OPTIMIZER=spel_pgd_dist,TRAIN_ITER=2,SPEL_PGD_BRANCH_MODE=auto,SPEL_PGD_PROJECTION_MODE=shared_topk,SPEL_PGD_PROJECTION_RANK=8,SPEL_PGD_GAP_THRESHOLD_REL=3e-4,SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10,SPEL_PGD_GAP_ESTIMATOR_MODE=block2_fp32_gap_only,SPEL_PGD_GAP_PROBE_INTERVAL=5,SPEL_PGD_GAP_PROBE_SAFE_MULTIPLIER=10,SPEL_PGD_MAIN_POWER_DTYPE=bf16,SPEL_PGD_DIRECTION_NORMALIZATION=spectral,SPEL_PGD_PGD_LR_SCALE=0.5,SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0,SPEL_PGD_WARM_START_UV=0"
  run_sbatch "$job_name" "$exports" "00:20:00" 12
}

case "$BATCH" in
  smoke1024_pgd)
    submit_pgd_smoke_1024
    ;;
  width1024)
    for lr in $LRS_ALL; do
      submit_spel 1024 "$lr"
      submit_pgd 1024 "$lr"
    done
    ;;
  width512)
    for lr in $LRS_MISSING_SPEL_256_512; do
      submit_spel 512 "$lr"
    done
    for lr in $LRS_ALL; do
      submit_pgd 512 "$lr"
    done
    ;;
  width256)
    submit_sso 256 2e-2
    submit_sso 256 3e-2
    for lr in $LRS_MISSING_SPEL_256_512; do
      submit_spel 256 "$lr"
    done
    for lr in $LRS_ALL; do
      submit_pgd 256 "$lr"
    done
    ;;
  *)
    echo "Usage: DRY_RUN=1 bash $0 {smoke1024_pgd|width1024|width512|width256}" >&2
    exit 2
    ;;
esac
