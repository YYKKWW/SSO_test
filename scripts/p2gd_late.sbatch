#!/bin/bash
#SBATCH --job-name=w256_p2gd_late
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/late_%j.out
#SBATCH --error=logs/late_%j.err
set -euo pipefail
cd "${PROJECT_DIR:?}"
sha256sum --check source.sha256 > "logs/source_check_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export PYTHONPATH="$PROJECT_DIR/studies/p2gd_elr:$PROJECT_DIR/elr/h20_elr:$PROJECT_DIR/elr:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ELR_TEST_DEVICE=cuda
python -B -m unittest test_p2gd_late test_verify_p2gd_six -v
python -B studies/p2gd_elr/run_p2gd_late.py
