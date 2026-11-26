#!/bin/bash
#SBATCH --job-name=w256_p2gd_smallstep_3b
#SBATCH --partition=c_math_gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/warm_p2gd_%A_%a.out
#SBATCH --error=logs/warm_p2gd_%A_%a.err
set -euo pipefail
: "${PROJECT_DIR:?Use the isolated warm-transfer snapshot}"
cd "$PROJECT_DIR"
sha256sum --check source.sha256 > "logs/source_check_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
module purge
module load python/3.12.1 cuda/12.4
source "${ENV_DIR:-$HOME/envs/sso_h20}/bin/activate"
export MEGATRON_PATH="$PROJECT_DIR/Megatron-LM"
export PYTHONPATH="$MEGATRON_PATH:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 P2GD_TEST_DEVICE=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest discover -s scripts -p test_warm_p2gd_transfer.py -v
modes=(cone top1 ambient)
python -B scripts/run_kernel_lmo_pilot.py --study direct_pk --arm 0 \
  --budget 3b --ambient-return radial --solver warm_block8 \
  --warm-p2gd-mode "${modes[$SLURM_ARRAY_TASK_ID]}" --warm-p2gd-ratio 0.075
