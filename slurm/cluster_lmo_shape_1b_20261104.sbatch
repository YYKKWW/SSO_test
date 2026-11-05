#!/bin/bash
#SBATCH --job-name=w256_lmo_shape_1b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/lmo_shape_%A_%a.out
#SBATCH --error=logs/lmo_shape_%A_%a.err
set -euo pipefail
: "${PROJECT_DIR:?Use the isolated source snapshot}"
case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0) export CLUSTER_ARM_INDEX=10 ;;
  1) export CLUSTER_ARM_INDEX=11 ;;
  *) echo "Unknown shape index" >&2; exit 2 ;;
esac
export CLUSTER_EXPERIMENT_SEED=2345
launcher="$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch"
if [[ "${1:-}" == "--describe" ]]; then
  exec bash "$launcher" "$@"
fi
# A separate process keeps smoke weights/RNG/data state out of the main run.
CLUSTER_SMOKE=1 CLUSTER_RUN_SUFFIX=_smoke bash "$launcher"
export CLUSTER_SMOKE=0 CLUSTER_RUN_SUFFIX=""
exec bash "$launcher" "$@"
