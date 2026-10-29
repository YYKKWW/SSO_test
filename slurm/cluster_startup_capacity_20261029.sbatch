#!/bin/bash
#SBATCH --job-name=cluster_startup_capacity
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/startup_capacity_%j.out
#SBATCH --error=logs/startup_capacity_%j.err
set -euo pipefail
: "${PROJECT_DIR:?Use an isolated source snapshot}"
: "${AUDIT_SOURCE:?Set the completed first-seed snapshot root}"
cd "$PROJECT_DIR"
sha256sum --check source.sha256 > "logs/source_check_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p test_cluster_replication.py -v
python -B scripts/audit_cluster_startup_capacity.py \
  "$AUDIT_SOURCE/results/4329039_4/snapshots" \
  "$AUDIT_SOURCE/results/4329039_5/snapshots" \
  --device cuda --output "$PROJECT_DIR/results/startup_capacity_${SLURM_JOB_ID}.json"
