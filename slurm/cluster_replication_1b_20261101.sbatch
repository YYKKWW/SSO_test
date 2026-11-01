#!/bin/bash
#SBATCH --job-name=w256_pair_1b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/pair_%A_%a.out
#SBATCH --error=logs/pair_%A_%a.err
set -euo pipefail
: "${PROJECT_DIR:?Use the isolated source snapshot, not the Slurm spool directory}"
case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0) export CLUSTER_ARM_INDEX=2 ;;
  1) export CLUSTER_ARM_INDEX=3 ;;
  *) echo "Unknown pair index" >&2; exit 2 ;;
esac
export CLUSTER_EXPERIMENT_SEED=1234
exec bash "$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch" "$@"
