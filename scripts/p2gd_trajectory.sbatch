#!/bin/bash
#SBATCH --job-name=w256_p2gd_path
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/path_%A_%a.out
#SBATCH --error=logs/path_%A_%a.err
set -euo pipefail
cd "${PROJECT_DIR:?}"
sha256sum --check source.sha256 > "logs/source_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export PYTHONPATH="$PROJECT_DIR/studies/p2gd_elr:$PROJECT_DIR/elr/h20_elr:$PROJECT_DIR/elr:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ELR_TEST_DEVICE=cuda
python -B -m unittest test_p2gd_trajectory test_p2gd_trajectory_integrity -v
if [[ "${P2GD_STAGE:?}" == "prefix" ]]; then
  python -B studies/p2gd_elr/run_p2gd_trajectories.py prefix
else
  python -B studies/p2gd_elr/run_p2gd_trajectories.py branch --index "${SLURM_ARRAY_TASK_ID:?}"
fi
