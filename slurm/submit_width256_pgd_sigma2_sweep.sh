#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_1b_width256_pgd_sigma2_sweep}"

WIDTH="${WIDTH:-256}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
HEAD_DIM="${HEAD_DIM:-128}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
TRAIN_TOKENS="${TRAIN_TOKENS:-1000000000}"
LR="${LR:-1.5e-2}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"

SPEL_PGD_BRANCH_MODE="${SPEL_PGD_BRANCH_MODE:-auto}"
SPEL_PGD_PROJECTION_MODE="${SPEL_PGD_PROJECTION_MODE:-shared_topk}"
SPEL_PGD_RANKS="${SPEL_PGD_RANKS:-8}"
SPEL_PGD_GAP_THRESHOLD_REL="${SPEL_PGD_GAP_THRESHOLD_REL:-1e-3}"
SPEL_PGD_SIGMA2_STEPS="${SPEL_PGD_SIGMA2_STEPS:-5}"
SPEL_PGD_DIRECTION_NORMALIZATION="${SPEL_PGD_DIRECTION_NORMALIZATION:-spectral}"
SPEL_PGD_PGD_LR_SCALE="${SPEL_PGD_PGD_LR_SCALE:-0.5}"
SPEL_PGD_VARIANTS="${SPEL_PGD_VARIANTS:-plain}"

mkdir -p "$RUN_ROOT"

lr_tag() {
  local lr=$1
  lr=${lr//./p}
  lr=${lr//-/m}
  printf "%s" "$lr"
}

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

submit_pgd() {
  local label=$1
  local tp_after_msign=$2
  local rank=$3
  local sigma2_steps=$4
  local tag
  local min_lr
  local mode_tag
  tag=$(lr_tag "$LR")
  min_lr=$(min_lr_for "$LR")
  mode_tag=${SPEL_PGD_PROJECTION_MODE//_/-}

  local scale_tag
  scale_tag=$(tag_value "$SPEL_PGD_PGD_LR_SCALE")

  local job_name="${label}_${mode_tag}_k${rank}_s2${sigma2_steps}_pgdlr${scale_tag}_w${WIDTH}_lr${tag}"

  sbatch \
    --cpus-per-task="$CPUS_PER_TASK" \
    -J "$job_name" \
    --export=ALL,OPTIMIZER="spel_pgd_dist",WIDTH="$WIDTH",NUM_LAYERS="$NUM_LAYERS",HEAD_DIM="$HEAD_DIM",SEQ_LENGTH="$SEQ_LENGTH",GLOBAL_BATCH="$GLOBAL_BATCH",MICRO_BATCH="$MICRO_BATCH",TRAIN_TOKENS="$TRAIN_TOKENS",LR="$LR",MIN_LR="$min_lr",LR_WARMUP_ITERS="$LR_WARMUP_ITERS",EVAL_INTERVAL="$EVAL_INTERVAL",EVAL_ITERS="$EVAL_ITERS",LOG_INTERVAL="$LOG_INTERVAL",RUN_ROOT="$RUN_ROOT",JOB_NAME="$job_name",SAVE_CHECKPOINT=0,SPEL_PGD_BRANCH_MODE="$SPEL_PGD_BRANCH_MODE",SPEL_PGD_PROJECTION_MODE="$SPEL_PGD_PROJECTION_MODE",SPEL_PGD_PROJECTION_RANK="$rank",SPEL_PGD_GAP_THRESHOLD_REL="$SPEL_PGD_GAP_THRESHOLD_REL",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="$sigma2_steps",SPEL_PGD_DIRECTION_NORMALIZATION="$SPEL_PGD_DIRECTION_NORMALIZATION",SPEL_PGD_PGD_LR_SCALE="$SPEL_PGD_PGD_LR_SCALE",SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN="$tp_after_msign" \
    "$SBATCH_SCRIPT"
}

echo "Submitting width=${WIDTH} PGD sigma2 sweep"
echo "  LR=${LR}"
echo "  sigma2_steps=${SPEL_PGD_SIGMA2_STEPS}"
echo "  branch=${SPEL_PGD_BRANCH_MODE}, projection=${SPEL_PGD_PROJECTION_MODE}, ranks=${SPEL_PGD_RANKS}, gap=${SPEL_PGD_GAP_THRESHOLD_REL}"
echo "  direction_normalization=${SPEL_PGD_DIRECTION_NORMALIZATION}, pgd_lr_scale=${SPEL_PGD_PGD_LR_SCALE}"
echo "  variants=${SPEL_PGD_VARIANTS} (plain: tp_after_msign=0, tp: tp_after_msign=1)"
echo "  TRAIN_TOKENS=${TRAIN_TOKENS}, GLOBAL_BATCH=${GLOBAL_BATCH}, MICRO_BATCH=${MICRO_BATCH}, CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "  RUN_ROOT=${RUN_ROOT}"

for sigma2_steps in $SPEL_PGD_SIGMA2_STEPS; do
  for rank in $SPEL_PGD_RANKS; do
    for variant in $SPEL_PGD_VARIANTS; do
      case "$variant" in
        plain)
          submit_pgd "plain_pgd" "0" "$rank" "$sigma2_steps"
          ;;
        tp|spel_tp|spel-tp)
          submit_pgd "spel_tp_pgd" "1" "$rank" "$sigma2_steps"
          ;;
        *)
          echo "Unknown SPEL_PGD_VARIANTS entry: $variant" >&2
          exit 1
          ;;
      esac
    done
  done
done
