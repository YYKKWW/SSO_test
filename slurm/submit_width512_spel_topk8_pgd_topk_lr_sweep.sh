#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/SSO_test}"

export PROJECT_DIR
export WIDTH="${WIDTH:-512}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/results/olmo_1b_width512_spel_topk8_pgd_topk_lr_sweep}"

exec bash "$PROJECT_DIR/slurm/submit_width256_spel_topk8_pgd_topk_lr_sweep.sh"
