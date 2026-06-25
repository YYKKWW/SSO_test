#!/bin/bash
#SBATCH --job-name=sso_test_h20
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"

cd "$PROJECT_DIR"
mkdir -p logs outputs checkpoints

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

echo "==== GPU ===="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "==== activate env ===="
source "$ENV_DIR/bin/activate"

echo "==== python env ===="
which python
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==== git info ===="
git log -1 --oneline || true

echo "==== start SSO test ===="
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

python -u Spectral-Sphere-Optimizer/run_sso_smoke.py \
    --device cuda \
    --steps "${STEPS:-20}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --input-dim "${INPUT_DIM:-256}" \
    --hidden-dim "${HIDDEN_DIM:-512}" \
    --output-dim "${OUTPUT_DIM:-128}" \
    --depth "${DEPTH:-2}" \
    --lr "${LR:-0.001}" \
    --output-dir outputs \
    --checkpoint-dir checkpoints

echo "==== finished ===="
date
