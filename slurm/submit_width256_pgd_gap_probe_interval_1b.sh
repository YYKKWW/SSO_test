#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/submit_width256_pgd_gap_threshold_tune.sh}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$PROJECT_DIR/results/olmo_1b_width256_pgd_gap_probe_interval}"
SPEL_PGD_GAP_PROBE_INTERVALS="${SPEL_PGD_GAP_PROBE_INTERVALS:-5 10}"

echo "Submitting width-256 MCSD-PGD 1B gap-probe interval comparison"
echo "  intervals=${SPEL_PGD_GAP_PROBE_INTERVALS}"
echo "  estimator=block2_fp32_gap_only, main_power_dtype=fp32, warm_start_uv=0"
echo "  adaptive safe multiplier=10"
echo "  gap=3e-4, pgd_lr_scale=0.5, shared_topk k=8, sigma2_steps=10, seed=1234"

for interval in $SPEL_PGD_GAP_PROBE_INTERVALS; do
  RUN_ROOT="$RUN_ROOT_BASE/probe${interval}" \
  WIDTH=256 \
  NUM_LAYERS=28 \
  SEQ_LENGTH=4096 \
  HEAD_DIM=128 \
  GLOBAL_BATCH=128 \
  MICRO_BATCH=4 \
  TRAIN_TOKENS=1000000000 \
  LR=1.5e-2 \
  MIN_LR=1.5e-3 \
  LR_WARMUP_ITERS=250 \
  EVAL_INTERVAL=250 \
  EVAL_ITERS=5 \
  LOG_INTERVAL=10 \
  CPUS_PER_TASK=8 \
  SEED=1234 \
  SPEL_PGD_BRANCH_MODE=auto \
  SPEL_PGD_PROJECTION_MODE=shared_topk \
  SPEL_PGD_PROJECTION_RANK=8 \
  SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10 \
  SPEL_PGD_GAP_ESTIMATOR_MODE=block2_fp32_gap_only \
  SPEL_PGD_GAP_PROBE_INTERVAL="$interval" \
  SPEL_PGD_GAP_PROBE_SAFE_MULTIPLIER=10 \
  SPEL_PGD_MAIN_POWER_DTYPE=fp32 \
  SPEL_PGD_DIRECTION_NORMALIZATION=spectral \
  SPEL_PGD_PGD_LR_SCALE=0.5 \
  SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=0 \
  SPEL_PGD_WARM_START_UV=0 \
  SPEL_PGD_GAP_THRESHOLDS=3e-4 \
  bash "$SUBMIT_SCRIPT"
done
