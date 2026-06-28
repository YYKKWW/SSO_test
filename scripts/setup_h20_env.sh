#!/bin/bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20_cu118}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8}"

module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.6.0+cu118" \
  numpy

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
