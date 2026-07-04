#!/bin/bash
set -euo pipefail

PROJECT=${PROJECT:-$HOME/projects/SSO_test}
MEGATRON=${MEGATRON:-$HOME/projects/Megatron-LM-active}
BASE_DIR=${BASE_DIR:-$PROJECT/data/olmo_mix_1124_1b}
RAW_DIR=${RAW_DIR:-$BASE_DIR/jsonl}
TOKENIZER_DIR=${TOKENIZER_DIR:-$BASE_DIR/tokenizer/OLMo-2-1124-7B}
INDEXED_DIR=${INDEXED_DIR:-$BASE_DIR/indexed}
WORKERS=${WORKERS:-8}
PARTITIONS=${PARTITIONS:-8}
WAIT_FOR_MANIFEST=${WAIT_FOR_MANIFEST:-1}
SPLIT=${SPLIT:-all}

mkdir -p "$INDEXED_DIR/train" "$INDEXED_DIR/valid" "$PROJECT/logs"

cleanup_partition_jsonl() {
  find "$RAW_DIR" -maxdepth 1 -type f \
    \( -name 'train_[*]_*.jsonl' -o -name 'valid_[*]_*.jsonl' \) \
    -delete
}

if command -v module >/dev/null 2>&1; then
  module purge
  module load python/3.12.1
fi

source "$HOME/envs/sso_h20/bin/activate"

if [ "$WAIT_FOR_MANIFEST" = "1" ]; then
  echo "waiting for $RAW_DIR/manifest.json"
  until [ -f "$RAW_DIR/manifest.json" ]; do
    date
    sleep 300
  done
fi

echo "==== OLMo mix Megatron preprocess ===="
date
hostname
echo "PROJECT=$PROJECT"
echo "MEGATRON=$MEGATRON"
echo "RAW_DIR=$RAW_DIR"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "INDEXED_DIR=$INDEXED_DIR"
echo "WORKERS=$WORKERS"
echo "PARTITIONS=$PARTITIONS"
echo "SPLIT=$SPLIT"
python --version

test -f "$TOKENIZER_DIR/tokenizer.json"
test -f "$RAW_DIR/manifest.json"
cleanup_partition_jsonl

cd "$MEGATRON"
export PYTHONPATH="$MEGATRON:${PYTHONPATH:-}"

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "train" ]; then
  echo "==== preprocess train ===="
  python -u tools/preprocess_data.py \
    --input "$RAW_DIR/train_*.jsonl" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_DIR" \
    --append-eod \
    --output-prefix "$INDEXED_DIR/train/olmo_mix_1124_1b_train" \
    --workers "$WORKERS" \
    --partitions "$PARTITIONS" \
    --log-interval 1000
  cleanup_partition_jsonl
fi

if [ "$SPLIT" = "all" ] || [ "$SPLIT" = "valid" ]; then
  echo "==== preprocess valid ===="
  python -u tools/preprocess_data.py \
    --input "$RAW_DIR/valid_*.jsonl" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_DIR" \
    --append-eod \
    --output-prefix "$INDEXED_DIR/valid/olmo_mix_1124_1b_valid" \
    --workers "$WORKERS" \
    --partitions "$PARTITIONS" \
    --log-interval 1000
  cleanup_partition_jsonl
fi

echo "==== indexed summary ===="
find "$INDEXED_DIR" -maxdepth 3 -type f | sort
du -sh "$INDEXED_DIR"
date
