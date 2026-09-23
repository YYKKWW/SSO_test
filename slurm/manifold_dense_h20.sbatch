#!/bin/bash
#SBATCH --job-name=manifold_dense
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

: "${MANIFOLD_RUN_CONFIG:?Submit with scripts/manifold/launch.py}"
module purge
module load python/3.12.1
module load cuda/12.4
ENV_DIR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["env_dir"])' "$MANIFOLD_RUN_CONFIG")
PROJECT_DIR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["project"])' "$MANIFOLD_RUN_CONFIG")
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
cd "$PROJECT_DIR"
srun --ntasks=1 python scripts/manifold/run.py "$MANIFOLD_RUN_CONFIG"
