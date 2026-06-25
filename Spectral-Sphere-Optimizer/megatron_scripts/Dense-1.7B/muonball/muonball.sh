#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=${WANDB_MODE:-offline}

WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/code/Megatron-LM}
PRETRAIN_SCRIPT=${PRETRAIN_SCRIPT:-pretrain_gpt.py}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

LR=${LR:-5e-3}
MIN_LR=${MIN_LR:-$(awk -v lr="$LR" 'BEGIN { printf "%.8g", lr / 10.0 }')}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
TRAIN_ITER=${TRAIN_ITER:-}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-20}
TRAIN_TOKENS=${TRAIN_TOKENS:-}
OPTIMIZER=${OPTIMIZER:-muon_ball_dist}

HIDDEN_SIZE=${HIDDEN_SIZE:-256}
NUM_LAYERS=${NUM_LAYERS:-4}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-$((HIDDEN_SIZE * 3))}
SEQ_LENGTH=${SEQ_LENGTH:-512}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$((SEQ_LENGTH * 10))}
MICRO_BATCH=${MICRO_BATCH:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}

if [ -n "$TRAIN_TOKENS" ]; then
    TOKENS_PER_ITER=$((GLOBAL_BATCH * SEQ_LENGTH))
    TRAIN_ITER=$(((TRAIN_TOKENS + TOKENS_PER_ITER - 1) / TOKENS_PER_ITER))
fi
TRAIN_ITER=${TRAIN_ITER:-200}

if (( LR_WARMUP_ITERS >= TRAIN_ITER )); then
    if (( TRAIN_ITER <= 1 )); then
        LR_WARMUP_ITERS=0
    else
        LR_WARMUP_ITERS=$((TRAIN_ITER / 10))
        if (( LR_WARMUP_ITERS < 1 )); then
            LR_WARMUP_ITERS=1
        fi
    fi
fi

if (( HIDDEN_SIZE % 64 != 0 )); then
    echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by 64 for the default head_dim=64." >&2
    exit 1
fi

HEAD_DIM=${HEAD_DIM:-64}
if (( HIDDEN_SIZE % HEAD_DIM != 0 )); then
    echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by HEAD_DIM=${HEAD_DIM}." >&2
    exit 1
fi

NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-$((HIDDEN_SIZE / HEAD_DIM))}
if (( NUM_ATTENTION_HEADS < 1 )); then
    echo "NUM_ATTENTION_HEADS must be >= 1." >&2
    exit 1
fi

if (( NUM_ATTENTION_HEADS < 8 )); then
    DEFAULT_NUM_QUERY_GROUPS=${NUM_ATTENTION_HEADS}
else
    DEFAULT_NUM_QUERY_GROUPS=8
fi
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-${DEFAULT_NUM_QUERY_GROUPS}}
KV_CHANNELS=${KV_CHANNELS:-${HEAD_DIM}}

BASE_PATH=${BASE_PATH:-${WORKSPACE}/data/merged_data}
TRAIN_BASE_PATH=${TRAIN_BASE_PATH:-${BASE_PATH}/train}
VALID_BASE_PATH=${VALID_BASE_PATH:-${BASE_PATH}/valid}
DATA_PATH_CACHE=${DATA_PATH_CACHE:-${WORKSPACE}/data/merged_data_cache}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${WORKSPACE}/models/OLMo-2-1124-7B}

REPO_PATH=${REPO_PATH:-${WORKSPACE}/results/optimizer_arena_v2/single_gpu_debug}
JOB_NAME=${JOB_NAME:-muonball_smoke_h${HIDDEN_SIZE}_lr${LR}_it${TRAIN_ITER}}
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"
LOG_DIR="${REPO_PATH}/logs"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

mkdir -p "$TENSORBOARD_PATH" "$CHECKPOINT_PATH" "$WANDB_PATH" "$LOG_DIR" "$DATA_PATH_CACHE"

build_data_path() {
    local search_path=$1
    local data_path=""
    if [ -d "$search_path" ]; then
        while IFS= read -r file; do
            if [ ! -s "$file" ]; then
                echo "skip empty dataset shard: $file" >&2
                continue
            fi
            data_path+="1 ${file%.bin} "
        done < <(find "$search_path" -type f -name "*.bin" | sort)
    fi
    printf "%s" "$data_path"
}

if [ -d "$TRAIN_BASE_PATH" ]; then
    TRAIN_DATA_PATH=$(build_data_path "$TRAIN_BASE_PATH")
    VALID_DATA_PATH=$(build_data_path "$VALID_BASE_PATH")
else
    TRAIN_DATA_PATH=$(build_data_path "$BASE_PATH")
    VALID_DATA_PATH=""
fi

if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "No .bin files found. Expected converted Megatron data under:" >&2
    echo "  ${TRAIN_BASE_PATH}" >&2
    echo "or:" >&2
    echo "  ${BASE_PATH}" >&2
    exit 1
fi

cd "$MEGATRON_PATH"

DATA_COMMON_ARGS=(
    --tokenizer-model "$TOKENIZER_MODEL"
    --tokenizer-type HuggingFaceTokenizer
    --data-cache-path "$DATA_PATH_CACHE"
    --train-iters "$TRAIN_ITER"
    --num-dataset-builder-threads "${DATASET_BUILDER_THREADS:-8}"
    --num-workers "${NUM_WORKERS:-2}"
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
)

if [ -n "$VALID_DATA_PATH" ]; then
    DATA_ARGS=(
        "${DATA_COMMON_ARGS[@]}"
        --train-data-path $TRAIN_DATA_PATH
        --valid-data-path $VALID_DATA_PATH
    )
else
    DATA_ARGS=(
        "${DATA_COMMON_ARGS[@]}"
        --data-path $TRAIN_DATA_PATH
        --split 99,1,0
    )
fi

TRAINING_ARGS=(
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITER"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay "$WEIGHT_DECAY"
    --optimizer "$OPTIMIZER"
)

case "$OPTIMIZER" in
adam)
    if [ "${ADAMW_LR_MUP_SCALER:-1}" = "1" ]; then
        TRAINING_ARGS+=(--adamw-lr-mup-scaler)
    fi
    ;;
muon|dist_muon)
    TRAINING_ARGS+=(
        --muon-momentum 0.9
        --muon-use-nesterov
        --muon-scale-mode spectral_mup
        --muon-num-ns-steps "${MUON_NUM_NS_STEPS:-8}"
        --muon-qkv-split-mode head
    )
    ;;
spectral_ball|spectral_ball_dist)
    TRAINING_ARGS+=(
        --spectral-ball-momentum 0.9
        --spectral-ball-use-nesterov
        --spectral-ball-msign-steps "${SPECTRAL_BALL_MSIGN_STEPS:-8}"
        --spectral-ball-radius-mode spectral_mup
        --spectral-ball-scale-mode spectral_mup
        --spectral-ball-solver bisection
        --spectral-ball-solver-tolerance-f "${SPECTRAL_BALL_SOLVER_TOLERANCE_F:-2e-4}"
        --spectral-ball-power-iteration-steps "${SPECTRAL_BALL_POWER_ITERATION_STEPS:-10}"
        --spectral-ball-solver-max-iterations "${SPECTRAL_BALL_SOLVER_MAX_ITERATIONS:-20}"
        --spectral-ball-retract-mode hard
        --spectral-ball-qkv-split-mode head
    )
    ;;
muon_ball|muon_ball_dist)
    TRAINING_ARGS+=(
    --muon-ball-momentum 0.9
    --muon-ball-use-nesterov
    --muon-ball-msign-steps 8
    --muon-ball-radius-mode spectral_mup
    --muon-ball-scale-mode spectral_mup
    --muon-ball-power-iteration-steps "${MUON_POWER_ITERATION_STEPS:-10}"
    --muon-ball-retract-mode hard
    --muon-ball-qkv-split-mode head
)
    ;;
spel|spel_dist)
    TRAINING_ARGS+=(
    --spel-momentum 0.9
    --spel-use-nesterov
    --spel-msign-steps "${SPEL_MSIGN_STEPS:-8}"
    --spel-radius-mode spectral_mup
    --spel-scale-mode spectral_mup
    --spel-power-iteration-steps "${SPEL_POWER_ITERATION_STEPS:-10}"
    --spel-retract-mode hard
    --spel-qkv-split-mode head
)
    ;;
*)
    echo "Unsupported OPTIMIZER=${OPTIMIZER}. Use adam, dist_muon, spectral_ball_dist, muon_ball_dist, or spel_dist." >&2
    exit 1
    ;;
esac

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --group-query-attention
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"
    --norm-epsilon 1e-6
    --kv-channels "$KV_CHANNELS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --no-rope-fusion
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --no-persist-layer-norm
    --qk-layernorm
    --disable-bias-linear
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --transformer-impl "${TRANSFORMER_IMPL:-local}"
    --attention-backend "${ATTENTION_BACKEND:-unfused}"
    --init-method-std 0.02
    --split-qkv-init-mode head
    --spectral-mup-init
)

if [ "${USE_CPU_INITIALIZATION:-1}" = "1" ]; then
    MODEL_ARGS+=(--use-cpu-initialization)
fi

CKPT_ARGS=(
    --ckpt-format torch_dist
)

if [ "${SAVE_CHECKPOINT:-1}" = "1" ]; then
    CKPT_ARGS+=(
        --save-interval "${SAVE_INTERVAL:-100}"
        --no-save-optim
        --save "$CHECKPOINT_PATH"
    )
fi

if [ "${SAVE_CHECKPOINT:-1}" = "1" ] && [ "${ASYNC_SAVE:-0}" = "1" ]; then
    CKPT_ARGS+=(--async-save)
fi

if [ -n "${LOAD_PATH:-}" ]; then
    CKPT_ARGS+=(--load "$LOAD_PATH")
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --micro-batch-size "$MICRO_BATCH"
    --global-batch-size "$GLOBAL_BATCH"
)

LOGGER_ARGS=(
    --log-params-norm
    --log-throughput
    --log-interval "${LOG_INTERVAL:-1}"
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_PATH"
)

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

WANDB_ARGS=(
    --wandb-project "${WANDB_PROJECT:-optimizer_arena_v2}"
    --wandb-exp-name "$JOB_NAME"
    --wandb-save-dir "$WANDB_PATH"
)

echo "Running ${JOB_NAME}"
echo "  world_size=${WORLD_SIZE}, gpus_per_node=${GPUS_PER_NODE}, tp=${TP_SIZE}, pp=${PP_SIZE}"
echo "  hidden=${HIDDEN_SIZE}, layers=${NUM_LAYERS}, heads=${NUM_ATTENTION_HEADS}, q_groups=${NUM_QUERY_GROUPS}"
echo "  seq=${SEQ_LENGTH}, micro_batch=${MICRO_BATCH}, global_batch=${GLOBAL_BATCH}, train_iters=${TRAIN_ITER}"
echo "  optimizer=${OPTIMIZER}, lr=${LR}, min_lr=${MIN_LR}, warmup=${LR_WARMUP_ITERS}"
echo "  checkpoints=${CHECKPOINT_PATH}"
echo "  log=${LOG_FILE}"

{
    torchrun \
        "${DISTRIBUTED_ARGS[@]}" \
        "$PRETRAIN_SCRIPT" \
        "${DATA_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${PARALLEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${LOGGER_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        --eval-interval "${EVAL_INTERVAL:-50}" \
        --eval-iters "${EVAL_ITERS:-5}"
} 2>&1 | tee -a "$LOG_FILE"
