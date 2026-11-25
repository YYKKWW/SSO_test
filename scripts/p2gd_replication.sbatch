#!/bin/bash
#SBATCH --job-name=w256_p2gd_replicate
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/replicate_%A_%a.out
#SBATCH --error=logs/replicate_%A_%a.err
set -euo pipefail
cd "${PROJECT_DIR:?}"
sha256sum --check source.sha256 > "logs/source_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export PYTHONPATH="$PROJECT_DIR/studies/p2gd_elr:$PROJECT_DIR/elr/h20_elr:$PROJECT_DIR/elr:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ELR_TEST_DEVICE=cuda
python -B -m unittest test_p2gd_recovery test_p2gd_replication test_p2gd_trajectory test_p2gd_trajectory_integrity -v
case "${P2GD_STAGE:?}" in
  prefix) python -B studies/p2gd_elr/run_p2gd_replication.py prefix ;;
  recover) python -B studies/p2gd_elr/run_p2gd_replication.py recover ;;
  branch) python -B studies/p2gd_elr/run_p2gd_replication.py branch --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  audit) python -B studies/p2gd_elr/audit_p2gd_snapshots.py ;;
  *) exit 2 ;;
esac
