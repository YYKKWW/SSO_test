#!/bin/bash
#SBATCH --job-name=w256_cone_lmo_1b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/cone_lmo_%A_%a.out
#SBATCH --error=logs/cone_lmo_%A_%a.err
set -euo pipefail

optimizer=spel_pgd_dist
mode=adaptive4
near=cone
cluster=2e-3
coefficient=8.0
experiment_seed=${CLUSTER_EXPERIMENT_SEED:-2345}
[[ "$experiment_seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed" >&2; exit 2; }
arm_index=${CLUSTER_ARM_INDEX:-${SLURM_ARRAY_TASK_ID:?An array index is required}}
case "$arm_index" in
  0) optimizer=muon_ball_dist; mode=none ;;
  1) optimizer=spectral_ball_dist; mode=none ;;
  2) mode=control; coefficient=1.0 ;;
  3) : ;;
  4) cluster=2e-4 ;;
  5) cluster=2e-4; near=cone_lmo; coefficient=1.0 ;;
  # Follow-up arms are selected only by the separate four-arm wrapper.
  6) near=ambient ;;
  7) near=tangent1 ;;
  8) cluster=2e-4; near=cone_lmo; coefficient=0.5 ;;
  9) cluster=2e-4; near=cone_lmo; coefficient=2.0 ;;
  10) cluster=2e-4; near=cone_lmo_top1; coefficient=0.5 ;;
  11) cluster=2e-4; near=cone_lmo_mix25; coefficient=0.5 ;;
  *) echo "Unknown experiment index" >&2; exit 2 ;;
esac
if [[ "${1:-}" == "--describe" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t1908\t20\n' \
    "$arm_index" "$optimizer" "$mode" "$near" "$cluster" "$coefficient" "$experiment_seed"
  exit 0
fi

# The selected values are local shell variables; do not leak overrides into
# subprocess configuration tests that also inspect the original six-arm grid.
unset CLUSTER_ARM_INDEX CLUSTER_EXPERIMENT_SEED
: "${PROJECT_DIR:?Use an isolated source snapshot}"
: "${TRAIN_DATA_PREFIX:?Explicit training data prefix required}"
: "${VALID_DATA_PREFIX:?Explicit validation data prefix required}"
: "${TOKENIZER_MODEL:?Explicit tokenizer required}"
cd "$PROJECT_DIR"
test -f Megatron-LM/megatron/core/models/gpt/gpt_model.py
test -f Megatron-LM/megatron/training/models/gpt.py
sha256sum --check source.sha256 > "logs/source_check_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export PYTHONPATH="$MEGATRON_PATH:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda
python -B -m unittest discover -s scripts -p test_cluster_cone_lmo.py -v
python -B -m unittest discover -s scripts -p test_cluster_near_projection.py -v
python -B -m unittest discover -s scripts -p test_cluster_followup_1b.py -v
python -B -m unittest discover -s scripts -p test_cluster_lmo_shape.py -v
export OPTIMIZER="$optimizer" SEED="$experiment_seed"
export WIDTH=256 NUM_LAYERS=28 HEAD_DIM=128 SEQ_LENGTH=4096
export GLOBAL_BATCH=128 MICRO_BATCH=4 GPUS_PER_NODE=1 TP_SIZE=1 PP_SIZE=1
export TRAIN_ITER=1908 TRAIN_TOKENS=1000000000
export LR=1.5e-2 MIN_LR=1.5e-3 LR_WARMUP_ITERS=250 LR_DECAY_ITERS=1908
export EVAL_INTERVAL=250 EVAL_ITERS=20 LOG_INTERVAL=10
if [[ "${CLUSTER_SMOKE:-0}" == "1" ]]; then
  export TRAIN_ITER=4 TRAIN_TOKENS=2097152
  export EVAL_ITERS=1 LOG_INTERVAL=1 LR_WARMUP_ITERS=1
fi
export SAVE_CHECKPOINT=0 WEIGHT_DECAY=0.1 LOAD_CHECKPOINT=""
export SPEL_PGD_BRANCH_MODE=auto SPEL_PGD_GAP_THRESHOLD_REL=5e-5
export SPEL_PGD_CLUSTER_THRESHOLD_REL="$cluster"
export SPEL_PGD_RADIAL_CLUSTER_MODE="$mode" SPEL_PGD_PGD_LR_SCALE="$coefficient"
export SPEL_PGD_CLUSTER_DIRECTION_MODE=momentum_fro SPEL_PGD_CLUSTER_NEAR_PROJECTION="$near"
export SPEL_PGD_CLUSTER_SPECTRUM_MODE=warm_block8 SPEL_PGD_CLUSTER_ITERATIONS=10
# Superseded by the cluster dispatch; actual solver is recorded below.
export SPEL_PGD_PROJECTION_MODE=tp_radial_pgd_exact SPEL_PGD_POST_RETURN_SPECTRAL_CACHE=1
export SPEL_PGD_GAP_ESTIMATOR_MODE=exact_svd_gap_only SPEL_PGD_GAP_PROBE_INTERVAL=1
export SPEL_PGD_GAP_AUDIT_INTERVAL=0 SPEL_PGD_MAIN_POWER_DTYPE=fp32
export SPEL_PGD_POWER_ITERATION_STEPS=10 SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS=10
export SPEL_PGD_MSIGN_STEPS=8 SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN=1
export SPEL_PGD_NORMALIZE_MSIGN_SPECTRAL=0 SPEL_PGD_MATRIX_WEIGHT_DECAY=0
export SPEL_PGD_DIRECTION_NORMALIZATION=none SPEL_PGD_FALLBACK_DIRECTION=pgd SPEL_PGD_WARM_START_UV=0
export SPECTRAL_BALL_POWER_ITERATION_STEPS=10 SPECTRAL_BALL_SOLVER_TOLERANCE_F=2e-4
export SPECTRAL_BALL_SOLVER_MAX_ITERATIONS=20 MUON_BALL_POWER_ITERATION_STEPS=10 MUON_BALL_MSIGN_STEPS=8
export JOB_NAME="w256_${optimizer}_${mode}_${near}_cluster${cluster}_c${coefficient}_seed${experiment_seed}_1b"
export RUN_ROOT="$PROJECT_DIR/results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}${CLUSTER_RUN_SUFFIX:-}"
export DATA_CACHE_PATH="$RUN_ROOT/data_cache"
if [[ "$optimizer" == "spel_pgd_dist" ]]; then
  export CLUSTER_SNAPSHOT_DIR="$RUN_ROOT/snapshots"
  export CLUSTER_SNAPSHOT_STEPS=2,50,250,750,1250,1750,1900
  export CLUSTER_SNAPSHOT_MAX=48 CLUSTER_SNAPSHOT_EARLY=8
  export CLUSTER_SNAPSHOT_LATE=8 CLUSTER_SNAPSHOT_LATE_AFTER=1000
  export PRETRAIN_ENTRYPOINT="$PROJECT_DIR/scripts/pretrain_cluster_ritz_observed.py"
else
  export PRETRAIN_ENTRYPOINT="$MEGATRON_PATH/pretrain_gpt.py"
fi
mkdir -p "$RUN_ROOT"
cp source.sha256 "$RUN_ROOT/source.sha256"
python -c 'import json,os,torch
keys={"WIDTH","NUM_LAYERS","HEAD_DIM","SEQ_LENGTH","GLOBAL_BATCH","MICRO_BATCH","TRAIN_ITER","TRAIN_TOKENS","LR","MIN_LR","LR_WARMUP_ITERS","LR_DECAY_ITERS","EVAL_INTERVAL","EVAL_ITERS","SEED","WEIGHT_DECAY","TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL","OPTIMIZER","NVIDIA_TF32_OVERRIDE","SLURM_JOB_ID","SLURM_ARRAY_TASK_ID","PRETRAIN_ENTRYPOINT"}
config={k:v for k,v in os.environ.items() if k in keys or k.startswith(("SPEL_PGD_","MUON_BALL_","SPECTRAL_BALL_","CLUSTER_SNAPSHOT_"))}
experimental=os.environ["OPTIMIZER"]=="spel_pgd_dist"
config.update(actual_spectral_solver="fixed_warm_block8_fp32_ritz" if experimental else "reference_bf16_pi10",actual_return="approximate_radial_both_branches" if experimental else "reference_hard_pre_retraction",online_accuracy_gate=False,automatic_full_svd_fallback=False,post_direction_normalization=False,matched_frobenius_length=False,data_protocol="1b_budget_on_existing_3b_indexed_source",initialization="fresh_seed"+os.environ["SEED"],processed_tokens=int(os.environ["TRAIN_ITER"])*128*4096,smoke=os.environ.get("CLUSTER_SMOKE")=="1",torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0))
mixed=os.environ["SPEL_PGD_CLUSTER_NEAR_PROJECTION"]=="cone_lmo_mix25"
config.update(lmo_blend_fraction=0.25 if mixed else 0.0,matched_unblended_near_fro=mixed,post_direction_normalization=mixed)
print(json.dumps(config,indent=2,sort_keys=True))' > "$RUN_ROOT/launch_config.json"
start=$(date +%s)
bash "$PROJECT_DIR/slurm/spel_olmo_1b_h20.sbatch" 2>&1 | tee "$RUN_ROOT/train.log"
printf '%s\n' "$(( $(date +%s) - start ))" > "$RUN_ROOT/training_wall_seconds.txt"
touch "$RUN_ROOT/TRAINING_COMPLETED"
if [[ "$optimizer" == "spel_pgd_dist" ]]; then
  python -c 'import json,os
from pathlib import Path
c=json.loads((Path(os.environ["CLUSTER_SNAPSHOT_DIR"])/"spectral_call_counts.json").read_text())
assert c["successful_training"] and c["component_updates"]>0,c
assert c["full_svd"]==c["svdvals"]==0,c
assert c["small_svd"]==c["component_updates"],c
print(c)'
  python scripts/audit_cluster_ritz_snapshots.py "$CLUSTER_SNAPSHOT_DIR" --device cuda \
    2>&1 | tee "$RUN_ROOT/offline_audit.log"
fi
touch "$RUN_ROOT/COMPLETED"
