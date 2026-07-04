#!/bin/bash
#SBATCH --job-name=spel_dev_mock
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${MEGATRON_PATH:-$HOME/projects/Megatron-LM-active}"
mkdir -p logs outputs data_cache_mock

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

echo "==== env ===="
source "$HOME/envs/sso_h20/bin/activate"
which python
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==== git ===="
if command -v git >/dev/null 2>&1; then
  git log -1 --oneline || true
  git status --short || true
else
  echo "git not found on compute node"
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

echo "==== start mock SpEL pretrain ===="
torchrun --standalone --nproc_per_node=1 pretrain_gpt.py \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 384 \
  --num-attention-heads 1 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --position-embedding-type rope \
  --swiglu \
  --normalization RMSNorm \
  --no-persist-layer-norm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --no-rope-fusion \
  --transformer-impl local \
  --bf16 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --distributed-backend nccl \
  --distributed-timeout-minutes 30 \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 4096 \
  --split 99,1,0 \
  --data-cache-path data_cache_mock \
  --num-workers 0 \
  --no-create-attention-mask-in-dataloader \
  --train-iters 2 \
  --eval-interval 1000 \
  --eval-iters 0 \
  --log-interval 1 \
  --timing-log-level 0 \
  --optimizer spel_dist \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --lr-decay-style cosine \
  --lr-decay-iters 2 \
  --lr-warmup-iters 0 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --spel-momentum 0.9 \
  --spel-use-nesterov \
  --spel-qkv-split-mode head \
  --spel-msign-steps 8 \
  --spel-radius-mode spectral_mup \
  --spel-power-iteration-steps 10 \
  --spel-scale-mode spectral_mup \
  --spel-retract-mode hard \
  --spel-retract-alpha 0.05 \
  --split-qkv-init-mode head \
  --spectral-mup-init \
  --use-cpu-initialization \
  --ckpt-format torch_dist

echo "==== finished ===="
date
