#!/bin/bash
#SBATCH --job-name=spel_olmo_h20_smoke
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
MEGATRON_PATH="${MEGATRON_PATH:-$PROJECT_DIR/Megatron-LM}"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/olmo_mix_1124_1b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$DATA_ROOT/tokenizer/OLMo-2-1124-7B}"
TRAIN_DATA_PREFIX="${TRAIN_DATA_PREFIX:-$DATA_ROOT/indexed/train/olmo_mix_1124_1b_train_text_document}"
VALID_DATA_PREFIX="${VALID_DATA_PREFIX:-$DATA_ROOT/indexed/valid/olmo_mix_1124_1b_valid_text_document}"

WIDTH="${WIDTH:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-$WIDTH}"
NUM_LAYERS="${NUM_LAYERS:-4}"
HEAD_DIM="${HEAD_DIM:-128}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-$((HIDDEN_SIZE * 3))}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-1}"
TRAIN_ITER="${TRAIN_ITER:-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
EVAL_ITERS="${EVAL_ITERS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

OPTIMIZER="${OPTIMIZER:-spel_dist}"
LR="${LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-4}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29831}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

if (( HIDDEN_SIZE % HEAD_DIM != 0 )); then
  echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by HEAD_DIM=${HEAD_DIM}" >&2
  exit 1
fi

NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-$((HIDDEN_SIZE / HEAD_DIM))}"
if (( NUM_ATTENTION_HEADS < 8 )); then
  DEFAULT_NUM_QUERY_GROUPS="$NUM_ATTENTION_HEADS"
else
  DEFAULT_NUM_QUERY_GROUPS=8
fi
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-$DEFAULT_NUM_QUERY_GROUPS}"
KV_CHANNELS="${KV_CHANNELS:-$HEAD_DIM}"

JOB_TAG="spel_realdata_w${HIDDEN_SIZE}_l${NUM_LAYERS}_s${SEQ_LENGTH}_it${TRAIN_ITER}_${SLURM_JOB_ID:-manual}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-$PROJECT_DIR/data_cache/${JOB_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/${JOB_TAG}}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs" "$PROJECT_DIR/data_cache" "$DATA_CACHE_PATH" "$OUTPUT_DIR"

echo "==== job info ===="
date
hostname
whoami
pwd

echo "==== modules ===="
module purge
module load python/3.12.1
module load cuda/12.4
module list

echo "==== activate env ===="
source "$ENV_DIR/bin/activate"
which python
python --version

echo "==== GPU ===="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "==== torch ===="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==== data ===="
ls -lh "${TRAIN_DATA_PREFIX}.bin" "${TRAIN_DATA_PREFIX}.idx"
ls -lh "${VALID_DATA_PREFIX}.bin" "${VALID_DATA_PREFIX}.idx"
ls -lh "$TOKENIZER_MODEL"

echo "==== git ===="
cd "$MEGATRON_PATH"
if command -v git >/dev/null 2>&1; then
  git log -1 --oneline || true
  git status --short || true
else
  echo "git not found on compute node"
fi

export PYTHONPATH="$MEGATRON_PATH:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

DATA_ARGS=(
  --tokenizer-model "$TOKENIZER_MODEL"
  --tokenizer-type HuggingFaceTokenizer
  --data-cache-path "$DATA_CACHE_PATH"
  --train-data-path 1 "$TRAIN_DATA_PREFIX"
  --valid-data-path 1 "$VALID_DATA_PREFIX"
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
  --position-embedding-type rope
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
  --no-create-attention-mask-in-dataloader
  --transformer-impl "$TRANSFORMER_IMPL"
  --init-method-std 0.02
  --split-qkv-init-mode head
  --spectral-mup-init
  --use-cpu-initialization
)

if [ -n "$ATTENTION_BACKEND" ]; then
  MODEL_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

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
  --timing-log-level 0
)

CKPT_ARGS=(
  --ckpt-format torch_dist
)

DISTRIBUTED_ARGS=(
  --standalone
  --nproc_per_node "$GPUS_PER_NODE"
)

echo "==== launch ===="
echo "job_tag=$JOB_TAG"
echo "hidden=$HIDDEN_SIZE layers=$NUM_LAYERS heads=$NUM_ATTENTION_HEADS q_groups=$NUM_QUERY_GROUPS"
echo "seq=$SEQ_LENGTH micro=$MICRO_BATCH global=$GLOBAL_BATCH train_iters=$TRAIN_ITER"
echo "optimizer=$OPTIMIZER lr=$LR min_lr=$MIN_LR"
echo "train_data=$TRAIN_DATA_PREFIX"
echo "valid_data=$VALID_DATA_PREFIX"

torchrun \
  "${DISTRIBUTED_ARGS[@]}" \
  pretrain_gpt.py \
  "${DATA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${PARALLEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${LOGGER_ARGS[@]}" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-iters "$EVAL_ITERS"

echo "==== finished ===="
date
