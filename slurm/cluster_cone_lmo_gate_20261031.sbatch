#!/bin/bash
#SBATCH --job-name=cone_lmo_gate
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=logs/cone_gate_%j.out
#SBATCH --error=logs/cone_gate_%j.err
set -euo pipefail
: "${PROJECT_DIR:?An isolated source snapshot is required}"
cd "$PROJECT_DIR"
test -f Megatron-LM/megatron/core/models/gpt/gpt_model.py
test -f Megatron-LM/megatron/training/models/gpt.py
sha256sum --check source.sha256 > "logs/gate_source_${SLURM_JOB_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p 'test_cluster*.py' -v
python scripts/audit_cluster_cone_lmo.py \
  /home/u3013198/projects/cluster-seed-mechanism-3b-20261028/results/4346705_2/snapshots \
  /home/u3013198/projects/cluster-seed-mechanism-3b-20261028/results/4346705_3/snapshots \
  /home/u3013198/projects/cluster-seed-mechanism-3b-20261028/results/4346705_4/snapshots \
  /home/u3013198/projects/cluster-seed-mechanism-3b-20261028/results/4346705_5/snapshots \
  --device cuda --output "$PROJECT_DIR/audit/paired_192.json"
python -c 'import json
from pathlib import Path
r=json.loads(Path("audit/paired_192.json").read_text())["summary"]
assert r["samples"]==192 and r["near_samples"]>0,r
for key in ("cached/cone_narrow_c8","cached/cone_lmo_narrow_c1"):
    a=r["arms"][key]
    assert a["direction_non_descent"]==0,(key,a)
    assert a["max_abs_radius_error"]<.01,(key,a)
print("OFFLINE_SANITY_PASSED")'
# This is a real Slurm-spooled launcher path, not a sibling-relative wrapper.
export CLUSTER_SMOKE=1 SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID"
export SLURM_ARRAY_TASK_ID=4
bash "$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch"
export SLURM_ARRAY_TASK_ID=5
bash "$PROJECT_DIR/slurm/cluster_cone_lmo_1b_20261031.sbatch"
touch "$PROJECT_DIR/audit/GATE_PASSED"
