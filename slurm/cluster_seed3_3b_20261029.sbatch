#!/bin/bash
#SBATCH --job-name=w256_cluster_seed3_3b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=1-06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/seed3_3b_%A_%a.out
#SBATCH --error=logs/seed3_3b_%A_%a.err
set -euo pipefail
case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0|1|2|3) ;;
  *) echo "Third-seed replication has exactly four arms" >&2; exit 2 ;;
esac
export CLUSTER_EXPERIMENT_SEED=3456
# Reuse the identical protocol; seed2 remains the shared launcher's default.
exec bash "$(dirname "${BASH_SOURCE[0]}")/cluster_seed_mechanism_3b_20261028.sbatch" "$@"
