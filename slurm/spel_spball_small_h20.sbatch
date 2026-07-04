#!/bin/bash
#SBATCH --job-name=spel_spball_small_h20
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"

cd "$PROJECT_DIR"
mkdir -p logs outputs

echo "==== job info ===="
date
hostname
whoami
pwd

echo "==== modules ===="
module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"
module list

echo "==== GPU ===="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "==== activate env ===="
source "$ENV_DIR/bin/activate"

echo "==== python env ===="
which python
python --version
python -c 'import torch; print("torch:", torch.__version__); print("torch cuda:", torch.version.cuda); print("cuda:", torch.cuda.is_available())'

echo "==== start optimizer compare smoke ===="
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/Megatron-LM:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=1 scripts/optimizer_compare_smoke.py \
    --device cuda \
    --mode "${MODE:-direct}" \
    --optimizers ${OPTIMIZERS:-spectral_ball_dist spel spel_pgd} \
    --steps "${STEPS:-3}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --seq-length "${SEQ_LENGTH:-16}" \
    --hidden-size "${HIDDEN_SIZE:-32}" \
    --ffn-hidden-size "${FFN_HIDDEN_SIZE:-64}" \
    --num-attention-heads "${NUM_ATTENTION_HEADS:-4}" \
    --num-query-groups "${NUM_QUERY_GROUPS:-1}" \
    --lr "${LR:-0.005}" \
    --weight-decay "${WEIGHT_DECAY:-0.1}" \
    --momentum "${MOMENTUM:-0.9}" \
    --msign-steps "${MSIGN_STEPS:-2}" \
    --power-iteration-steps "${POWER_ITERATION_STEPS:-3}" \
    --solver-max-iterations "${SOLVER_MAX_ITERATIONS:-10}" \
    --solver-tolerance-f "${SOLVER_TOLERANCE_F:-2e-4}" \
    --radius-mode "${RADIUS_MODE:-spectral_mup}" \
    --scale-mode "${SCALE_MODE:-spectral_mup}" \
    --retract-mode "${RETRACT_MODE:-hard}" \
    --qkv-split-mode "${QKV_SPLIT_MODE:-head}" \
    --spel-pgd-branch-mode "${SPEL_PGD_BRANCH_MODE:-auto}" \
    --spel-pgd-gap-threshold-rel "${SPEL_PGD_GAP_THRESHOLD_REL:-5e-3}" \
    --spel-pgd-sigma2-power-iteration-steps "${SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS:-3}" \
    --spel-pgd-direction-normalization "${SPEL_PGD_DIRECTION_NORMALIZATION:-none}" \
    --output "outputs/optimizer_compare_smoke_${SLURM_JOB_ID}.json"

echo "==== finished ===="
date
