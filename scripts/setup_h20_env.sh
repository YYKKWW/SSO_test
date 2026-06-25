#!/bin/bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/sso_h20}"

module purge
module load python/3.12.1
module load cuda/12.4

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
