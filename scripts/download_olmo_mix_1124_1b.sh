#!/bin/bash
set -euo pipefail

PROJECT=${PROJECT:-$HOME/projects/SSO_test}
BASE_DIR=${BASE_DIR:-$PROJECT/data/olmo_mix_1124_1b}
RAW_DIR=${RAW_DIR:-$BASE_DIR/jsonl}
TOKENIZER_DIR=${TOKENIZER_DIR:-$BASE_DIR/tokenizer/OLMo-2-1124-7B}
LOG_DIR=${LOG_DIR:-$PROJECT/logs}

TARGET_TOKENS=${TARGET_TOKENS:-1000000000}
VALID_TOKENS=${VALID_TOKENS:-10000000}
TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-50000000}
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-10000}
SEED=${SEED:-42}
FORCE=${FORCE:-0}

mkdir -p "$RAW_DIR" "$TOKENIZER_DIR" "$LOG_DIR"

module purge
module load python/3.12.1
source "$HOME/envs/sso_h20/bin/activate"

cd "$PROJECT"

echo "==== OLMo mix download ===="
date
hostname
pwd
python --version

echo "==== download tokenizer ===="
python scripts/download_hf_tokenizer.py \
  allenai/OLMo-2-1124-7B \
  "$TOKENIZER_DIR"

echo "==== sample dataset ===="
sample_args=(
  --output-dir "$RAW_DIR"
  --target-tokens "$TARGET_TOKENS"
  --valid-tokens "$VALID_TOKENS"
  --tokens-per-shard "$TOKENS_PER_SHARD"
  --shuffle-buffer "$SHUFFLE_BUFFER"
  --seed "$SEED"
)

if [ "$FORCE" = "1" ]; then
  sample_args+=(--force)
fi

if [ -f "$RAW_DIR/manifest.json" ] && [ "$FORCE" != "1" ]; then
  echo "manifest exists, skip sampling: $RAW_DIR/manifest.json"
else
  python -u scripts/sample_olmo_mix_1124.py "${sample_args[@]}"
fi

echo "==== summary ===="
find "$BASE_DIR" -maxdepth 3 -type f | sort | head -80
du -sh "$BASE_DIR"
date
