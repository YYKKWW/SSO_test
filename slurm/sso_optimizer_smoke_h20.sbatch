#!/bin/bash
#SBATCH --job-name=sso_py_smoke
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$HOME/projects/SSO_test"
mkdir -p logs outputs

echo "==== job info ===="
date
hostname
pwd

module purge
module load python/3.12.1
module load cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python scripts/check_sso_optimizer.py --repo-root "$PWD" --device cuda --steps 3

echo "==== finished ===="
date
