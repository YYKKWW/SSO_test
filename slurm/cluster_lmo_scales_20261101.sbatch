#!/bin/bash
#SBATCH --job-name=cone_scale_audit
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scales_%j.out
#SBATCH --error=logs/scales_%j.err
set -euo pipefail
: "${PROJECT_DIR:?Explicit source snapshot required}"
: "${SNAPSHOT_ROOT:?Existing completed six-arm result directory required}"
cd "$PROJECT_DIR"
sha256sum --check source.sha256 > "logs/source_check_scales_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p 'test_cluster*.py' -v
python scripts/audit_cluster_cone_lmo.py \
  "$SNAPSHOT_ROOT/4368382_2/snapshots" "$SNAPSHOT_ROOT/4368382_5/snapshots" \
  "$SNAPSHOT_ROOT/4368383_3/snapshots" "$SNAPSHOT_ROOT/4368383_4/snapshots" \
  --preset lmo_scales --device cuda --output audit/lmo_scales_20261101.json
