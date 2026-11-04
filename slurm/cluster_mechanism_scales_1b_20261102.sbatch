#!/bin/bash
#SBATCH --job-name=w256_mechanism_1b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/mechanism_%A_%a.out
#SBATCH --error=logs/mechanism_%A_%a.err
set -euo pipefail
: "${PROJECT_DIR:?Use the isolated source snapshot, not the Slurm spool directory}"
launcher="$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch"
[[ -f "$launcher" ]] || { echo "Shared launcher not found: $launcher" >&2; exit 2; }
case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0) export CLUSTER_ARM_INDEX=6 ;;  # Ambient near direction, c8.
  1) export CLUSTER_ARM_INDEX=7 ;;  # Top-1 near correction, c8.
  2) export CLUSTER_ARM_INDEX=8 ;;  # Single-LMO, c0.5.
  3) export CLUSTER_ARM_INDEX=9 ;;  # Single-LMO, c2.
  *) echo "Unknown follow-up index" >&2; exit 2 ;;
esac
export CLUSTER_EXPERIMENT_SEED=2345
exec bash "$launcher" "$@"
