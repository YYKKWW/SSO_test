#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/submit_primary_1b_missing_matrix.sh}"
DEPENDENCY="${DEPENDENCY:-}"
SLURM_USER="${SLURM_USER:-u3013198}"
MAX_EXISTING_JOBS="${MAX_EXISTING_JOBS:-21}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-86400}"

if [[ -z "$DEPENDENCY" ]]; then
  echo "DEPENDENCY=afterok:<width512-job-ids> is required" >&2
  exit 2
fi

deadline=$((SECONDS + DEADLINE_SECONDS))

echo "Waiting to submit primary width-1024 full matrix"
echo "  user=${SLURM_USER}"
echo "  max_existing_jobs=${MAX_EXISTING_JOBS}"
echo "  dependency=${DEPENDENCY}"
echo "  deadline_seconds=${DEADLINE_SECONDS}"

while (( SECONDS < deadline )); do
  existing_primary=$(squeue -h -u "$SLURM_USER" -o "%j" | grep -Ec '^primary_(spel_bf16_topk8|mcsd_pgd_bf16_gapfp32_k8)_w1024_' || true)
  if (( existing_primary > 0 )); then
    echo "Primary width-1024 jobs already exist in squeue; refusing duplicate submission"
    exit 3
  fi

  submitted_count=$(squeue -h -u "$SLURM_USER" | wc -l | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') submitted_count=${submitted_count}"
  if (( submitted_count <= MAX_EXISTING_JOBS )); then
    cd "$PROJECT_DIR"
    DEPENDENCY="$DEPENDENCY" DRY_RUN=0 bash "$SUBMIT_SCRIPT" width1024
    echo "Width-1024 dependent batch submitted"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done

echo "Deadline reached before enough Slurm submit slots became available" >&2
exit 4
