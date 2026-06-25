#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Single-GPU reproduction wrapper for the paper's dense width-scaling setup.
# It keeps the paper architecture rule:
#   layers=28, seq=4096, head_dim=128, ffn_hidden=3*hidden,
#   hidden swept over 256/512/1024/2048.
#
# Start with DRY_RUN=1 to print the manifest. Use SMOKE=1 before launching 10B.

WORKSPACE=${WORKSPACE:-/workspace}
REPO_PATH=${REPO_PATH:-${WORKSPACE}/results/optimizer_arena_v2/single_gpu_10b_paperarch}
SESSION_NAME=${SESSION_NAME:-single_gpu_10b_paperarch_spel}

OPTIMIZERS=${OPTIMIZERS:-"spel_dist"}
WIDTHS=${WIDTHS:-"256"}
LRS=${LRS:-"3e-3 5e-3 7e-3"}

TRAIN_TOKENS=${TRAIN_TOKENS:-10000000000}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
NUM_LAYERS=${NUM_LAYERS:-28}
HEAD_DIM=${HEAD_DIM:-128}

# GLOBAL_BATCH controls optimizer-step frequency. Total token compute is fixed by TRAIN_TOKENS.
# Larger values are closer to the paper but produce fewer, longer iterations on one GPU.
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-1}

SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-10}

TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fused}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
DRY_RUN=${DRY_RUN:-1}
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = "1" ]; then
    echo "Running a single paper-architecture smoke test: width=256, ${TRAIN_ITER:-10} iterations."
    OPTIMIZER=${OPTIMIZER:-spel_dist} \
    HIDDEN_SIZE=${HIDDEN_SIZE:-256} \
    NUM_LAYERS="$NUM_LAYERS" \
    HEAD_DIM="$HEAD_DIM" \
    SEQ_LENGTH="$SEQ_LENGTH" \
    GLOBAL_BATCH="$GLOBAL_BATCH" \
    MICRO_BATCH="$MICRO_BATCH" \
    TRAIN_ITER=${TRAIN_ITER:-10} \
    EVAL_INTERVAL=${SMOKE_EVAL_INTERVAL:-100} \
    EVAL_ITERS=${SMOKE_EVAL_ITERS:-1} \
    SAVE_INTERVAL=${SMOKE_SAVE_INTERVAL:-100} \
    MASTER_PORT=${MASTER_PORT:-29811} \
    JOB_NAME=${JOB_NAME:-paperarch_width256_smoke} \
    REPO_PATH="${REPO_PATH}_smoke" \
    TRANSFORMER_IMPL="$TRANSFORMER_IMPL" \
    ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    BASE_PATH=${BASE_PATH:-${WORKSPACE}/data/merged_data} \
    TRAIN_BASE_PATH=${TRAIN_BASE_PATH:-${WORKSPACE}/data/merged_data/train} \
    VALID_BASE_PATH=${VALID_BASE_PATH:-${WORKSPACE}/data/merged_data/valid} \
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-${WORKSPACE}/models/OLMo-2-1124-7B} \
    bash "${SCRIPT_DIR}/muonball.sh"
    exit 0
fi

echo "Single-GPU 10B paper-architecture sweep"
echo "  optimizers: ${OPTIMIZERS}"
echo "  widths: ${WIDTHS}"
echo "  lrs: ${LRS}"
echo "  train_tokens: ${TRAIN_TOKENS}"
echo "  seq_length: ${SEQ_LENGTH}"
echo "  layers: ${NUM_LAYERS}"
echo "  head_dim: ${HEAD_DIM}"
echo "  global_batch: ${GLOBAL_BATCH}"
echo "  repo_path: ${REPO_PATH}"

DRY_RUN="$DRY_RUN" \
SKIP_COMPLETED="$SKIP_COMPLETED" \
SESSION_NAME="$SESSION_NAME" \
REPO_PATH="$REPO_PATH" \
OPTIMIZERS="$OPTIMIZERS" \
WIDTHS="$WIDTHS" \
LRS="$LRS" \
TRAIN_TOKENS="$TRAIN_TOKENS" \
SEQ_LENGTH="$SEQ_LENGTH" \
GLOBAL_BATCH="$GLOBAL_BATCH" \
MICRO_BATCH="$MICRO_BATCH" \
NUM_LAYERS="$NUM_LAYERS" \
HEAD_DIM="$HEAD_DIM" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
SAVE_CHECKPOINT="$SAVE_CHECKPOINT" \
EVAL_INTERVAL="$EVAL_INTERVAL" \
EVAL_ITERS="$EVAL_ITERS" \
LOG_INTERVAL="$LOG_INTERVAL" \
TRANSFORMER_IMPL="$TRANSFORMER_IMPL" \
ATTENTION_BACKEND="$ATTENTION_BACKEND" \
bash "${SCRIPT_DIR}/run_mup_lr_width_sweep.sh"
