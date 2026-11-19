#!/bin/bash
#SBATCH --job-name=w256_lmo_confirm_1b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/lmo_confirmation_%A_%a.out
#SBATCH --error=logs/lmo_confirmation_%A_%a.err
set -euo pipefail
: "${PROJECT_DIR:?Use the isolated source snapshot}"
case "${SLURM_ARRAY_TASK_ID:?An array index is required}" in
  0) export CLUSTER_ARM_INDEX=2 ;;
  1) export CLUSTER_ARM_INDEX=8 ;;
  2) export CLUSTER_ARM_INDEX=11 ;;
  3) export CLUSTER_ARM_INDEX=0 ;;
  4) export CLUSTER_ARM_INDEX=1 ;;
  *) echo "Unknown confirmation index" >&2; exit 2 ;;
esac
export CLUSTER_EXPERIMENT_SEED=1234
launcher="$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch"
if [[ "${1:-}" == "--describe" ]]; then
  exec bash "$launcher" "$@"
fi
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -m unittest discover -s "$PROJECT_DIR/scripts" \
  -p test_cluster_lmo_confirmation.py -v
# Smoke and main training run in separate processes with fresh RNG/model state.
CLUSTER_SMOKE=1 CLUSTER_RUN_SUFFIX=_smoke bash "$launcher"
export CLUSTER_SMOKE=0 CLUSTER_RUN_SUFFIX=""
exec bash "$launcher" "$@"
