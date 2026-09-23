#!/bin/bash
#SBATCH --job-name=manifold_smoke
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test-three-geometries}"
MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"
DATA_ROOT="${DATA_ROOT:-$HOME/projects/SSO_test/data/olmo_mix_1124_1b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$DATA_ROOT/tokenizer/OLMo-2-1124-7B}"
TRAIN_DATA_PREFIX="${TRAIN_DATA_PREFIX:-$DATA_ROOT/indexed/train/olmo_mix_1124_1b_train_text_document}"
VALID_DATA_PREFIX="${VALID_DATA_PREFIX:-$DATA_ROOT/indexed/valid/olmo_mix_1124_1b_valid_text_document}"

OPTIMIZER="${OPTIMIZER:-manifold_mcsd_tp}"
GEOMETRY="${GEOMETRY:-frobenius}"
LR="${LR:-0.01}"
MIN_LR="${MIN_LR:-0.001}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_ITERS="${EVAL_ITERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SEED="${SEED:-1234}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
MICRO_BATCH="${MICRO_BATCH:-4}"
SPECTRAL_SOLVER="${SPECTRAL_SOLVER:-pi_topk}"
LMO_MODE="${LMO_MODE:-ns}"
STIEFEL_RETURN_MODE="${STIEFEL_RETURN_MODE:-ns}"
POWER_STEPS="${POWER_STEPS:-10}"
TOPK_RANK="${TOPK_RANK:-8}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/manifold_smoke}"
RUN_DIR="$RUN_ROOT/${GEOMETRY}_${OPTIMIZER}_${SLURM_JOB_ID:-manual}"

[[ "$OPTIMIZER" == manifold_mcsd || "$OPTIMIZER" == manifold_mcsd_tp ]] || exit 2
[[ "$GEOMETRY" == frobenius || "$GEOMETRY" == spectral || "$GEOMETRY" == stiefel ]] || exit 2
[[ -f "${TRAIN_DATA_PREFIX}.idx" && -f "${VALID_DATA_PREFIX}.idx" ]] || {
  echo "Indexed data not found" >&2
  exit 2
}
[[ ! -e "$RUN_DIR" ]] || { echo "Run directory exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR" "$PROJECT_DIR/logs" "$PROJECT_DIR/data_cache/manifold_dense"

module purge
module load python/3.12.1
module load cuda/12.4
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$MEGATRON_PATH:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MCSD_MANIFOLD_MANIFEST_PATH="$RUN_DIR/parameter_manifest.json"

cd "$MEGATRON_PATH"
[[ -n "${SOURCE_COMMIT:-}" ]] || {
  echo "SOURCE_COMMIT is required; submit with scripts/manifold/launch.py" >&2
  exit 2
}
printf '%s\n' "$SOURCE_COMMIT" > "$RUN_DIR/source_commit.txt"
printf 'clean at submission (checked by launch.py)\n' > "$RUN_DIR/source_dirty.txt"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
  > "$RUN_DIR/runtime.txt"

ARGS=(
  --num-layers 28 --hidden-size 384 --ffn-hidden-size 1152
  --group-query-attention --num-attention-heads 6 --num-query-groups 3 --kv-channels 64
  --seq-length 2048 --max-position-embeddings 2048
  --attention-dropout 0 --hidden-dropout 0 --bf16
  --position-embedding-type rope --rotary-base 1000000
  --swiglu --untie-embeddings-and-output-weights
  --normalization RMSNorm --norm-epsilon 1e-6 --no-persist-layer-norm
  --qk-layernorm --disable-bias-linear
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion --no-rope-fusion
  --no-create-attention-mask-in-dataloader --transformer-impl local
  --init-method-std 0.02 --split-qkv-init-mode head --spectral-mup-init
  --use-cpu-initialization
  --tokenizer-model "$TOKENIZER_MODEL" --tokenizer-type HuggingFaceTokenizer
  --data-cache-path "$PROJECT_DIR/data_cache/manifold_dense"
  --train-data-path 1 "$TRAIN_DATA_PREFIX"
  --valid-data-path 1 "$VALID_DATA_PREFIX"
  --num-dataset-builder-threads 8 --num-workers 2 --no-mmap-bin-files
  --distributed-timeout-minutes 60
  --seed "$SEED" --optimizer "$OPTIMIZER" --manifold-geometry "$GEOMETRY"
  --manifold-momentum-beta 0.95 --manifold-aux-lr 0.001
  --manifold-lmo-mode "$LMO_MODE"
  --manifold-stiefel-return-mode "$STIEFEL_RETURN_MODE"
  --manifold-spectral-solver "$SPECTRAL_SOLVER"
  --manifold-power-steps "$POWER_STEPS" --manifold-topk-rank "$TOPK_RANK"
  --manifold-spectral-audit-interval 20
  --lr "$LR" --min-lr "$MIN_LR" --lr-warmup-iters "$WARMUP_ITERS"
  --lr-decay-style cosine --lr-decay-iters "$TRAIN_ITERS"
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
  --clip-grad 1.0 --weight-decay 0.1
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --micro-batch-size "$MICRO_BATCH" --global-batch-size "$GLOBAL_BATCH"
  --train-iters "$TRAIN_ITERS" --eval-interval "$EVAL_INTERVAL"
  --eval-iters "$EVAL_ITERS" --log-interval 10 --log-throughput
  --tensorboard-dir "$RUN_DIR/tensorboard" --ckpt-format torch_dist
)
if [[ "$SAVE_INTERVAL" -gt 0 ]]; then
  ARGS+=(--save "$RUN_DIR/checkpoints" --save-interval "$SAVE_INTERVAL")
fi
printf '%q ' torchrun --standalone --nproc_per_node "$GPUS_PER_NODE" pretrain_gpt.py "${ARGS[@]}" \
  > "$RUN_DIR/command.txt"
echo >> "$RUN_DIR/command.txt"
echo "Run: $RUN_DIR"
echo "Actual tokens: $((TRAIN_ITERS * GLOBAL_BATCH * 2048))"
torchrun --standalone --nproc_per_node "$GPUS_PER_NODE" pretrain_gpt.py "${ARGS[@]}"
