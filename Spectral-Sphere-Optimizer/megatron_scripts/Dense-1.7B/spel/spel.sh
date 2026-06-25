#!/bin/bash
set -euo pipefail

# Single-run SpEL launcher derived from the completed true-SpEL h2048 setup.
# Keep the architectural rule from the h2048 reference:
#   layers=28, seq=4096, head_dim=128, ffn_hidden=3*hidden,
# and vary width/lr through environment overrides such as:
#   HIDDEN_SIZE=256 LR=5e-3 bash spel.sh
#   HIDDEN_SIZE=512 LR=5e-3 bash spel.sh

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
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

OPTIMIZER=${OPTIMIZER:-spel_dist}
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-$(awk -v lr="$LR" 'BEGIN { printf "%.8g", lr / 10.0 }')}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
TRAIN_ITER=${TRAIN_ITER:-3815}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-500}
TRAIN_TOKENS=${TRAIN_TOKENS:-}

WIDTH=${WIDTH:-}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
if [ -n "$WIDTH" ]; then
    HIDDEN_SIZE=$WIDTH
fi
NUM_LAYERS=${NUM_LAYERS:-28}
HEAD_DIM=${HEAD_DIM:-128}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-$((HIDDEN_SIZE * 3))}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-40960}
MICRO_BATCH=${MICRO_BATCH:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}

if [ -n "$TRAIN_TOKENS" ]; then
    TOKENS_PER_ITER=$((GLOBAL_BATCH * SEQ_LENGTH))
    TRAIN_ITER=$(((TRAIN_TOKENS + TOKENS_PER_ITER - 1) / TOKENS_PER_ITER))
fi

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

REPO_PATH=${REPO_PATH:-${WORKSPACE}/results/optimizer_arena_v2/true_spel_from_h2048_ref}
job_lr=${LR//./p}
job_lr=${job_lr//-/m}
JOB_NAME=${JOB_NAME:-mup_spel_dist_h${HIDDEN_SIZE}_lr${job_lr}}
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"
LOG_DIR="${REPO_PATH}/logs"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-10}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fused}

mkdir -p "$TENSORBOARD_PATH" "$WANDB_PATH" "$LOG_DIR" "$DATA_PATH_CACHE"

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

TRAIN_DATA_PATH=$(build_data_path "$TRAIN_BASE_PATH")
VALID_DATA_PATH=$(build_data_path "$VALID_BASE_PATH")

if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "No .bin files found under ${TRAIN_BASE_PATH}." >&2
    exit 1
fi

cd "$MEGATRON_PATH"

DATA_ARGS=(
    --tokenizer-model "$TOKENIZER_MODEL"
    --tokenizer-type HuggingFaceTokenizer
    --data-cache-path "$DATA_PATH_CACHE"
    --train-data-path $TRAIN_DATA_PATH
    --valid-data-path $VALID_DATA_PATH
    --train-iters "$TRAIN_ITER"
    --num-dataset-builder-threads "${DATASET_BUILDER_THREADS:-8}"
    --num-workers "${NUM_WORKERS:-2}"
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
)

MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --group-query-attention
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --num-query-groups "$NUM_QUERY_GROUPS"
    --kv-channels "$KV_CHANNELS"
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --no-persist-layer-norm
    --qk-layernorm
    --disable-bias-linear
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --no-rope-fusion
    --transformer-impl "$TRANSFORMER_IMPL"
    --attention-backend "$ATTENTION_BACKEND"
    --init-method-std 0.02
    --split-qkv-init-mode head
    --spectral-mup-init
    --use-cpu-initialization
)

TRAINING_ARGS=(
    --optimizer "$OPTIMIZER"
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
    --spel-momentum 0.9
    --spel-use-nesterov
    --spel-qkv-split-mode head
    --spel-msign-steps 8
    --spel-radius-mode spectral_mup
    --spel-power-iteration-steps 10
    --spel-scale-mode spectral_mup
    --spel-retract-mode hard
    --spel-retract-alpha 0.05
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --pipeline-model-parallel-size "$PP_SIZE"
    --micro-batch-size "$MICRO_BATCH"
    --global-batch-size "$GLOBAL_BATCH"
)

LOGGER_ARGS=(
    --log-params-norm
    --log-throughput
    --log-interval "$LOG_INTERVAL"
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_PATH"
)

WANDB_ARGS=(
    --wandb-project "${WANDB_PROJECT:-optimizer_arena_v2}"
    --wandb-exp-name "$JOB_NAME"
    --wandb-save-dir "$WANDB_PATH"
)

CKPT_ARGS=(
    --ckpt-format torch_dist
)

if [ "$SAVE_CHECKPOINT" = "1" ]; then
    CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
    mkdir -p "$CHECKPOINT_PATH"
    CKPT_ARGS+=(
        --save "$CHECKPOINT_PATH"
        --save-interval "$SAVE_INTERVAL"
        --no-save-optim
    )
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

echo "Running ${JOB_NAME}"
echo "  hidden=${HIDDEN_SIZE}, layers=${NUM_LAYERS}, heads=${NUM_ATTENTION_HEADS}, q_groups=${NUM_QUERY_GROUPS}"
echo "  seq=${SEQ_LENGTH}, micro_batch=${MICRO_BATCH}, global_batch=${GLOBAL_BATCH}, train_iters=${TRAIN_ITER}"
echo "  optimizer=${OPTIMIZER}, lr=${LR}, min_lr=${MIN_LR}, warmup=${LR_WARMUP_ITERS}"
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
        --eval-interval "$EVAL_INTERVAL" \
        --eval-iters "$EVAL_ITERS"
} 2>&1 | tee -a "$LOG_FILE"
