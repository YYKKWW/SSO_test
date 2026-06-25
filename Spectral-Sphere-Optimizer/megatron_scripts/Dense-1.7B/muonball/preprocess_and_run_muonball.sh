#!/bin/bash
set -euo pipefail
shopt -s nullglob

WORKSPACE=${WORKSPACE:-/workspace}
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/code/Megatron-LM}
JSONL_DIR=${JSONL_DIR:-${WORKSPACE}/data/olmo_mix_1124_jsonl/diverse_10b}
OUT_BASE=${OUT_BASE:-${WORKSPACE}/data/merged_data}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${WORKSPACE}/models/OLMo-2-1124-7B}
PREPROCESS_WORKERS=${PREPROCESS_WORKERS:-16}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${OUT_BASE}/train" "${OUT_BASE}/valid"

preprocess_split() {
    local split=$1
    local out_dir="${OUT_BASE}/${split}"
    local files=("${JSONL_DIR}/${split}"_*.jsonl)

    if (( ${#files[@]} == 0 )); then
        echo "No ${split}_*.jsonl files found in ${JSONL_DIR}" >&2
        if [ "$split" = "train" ]; then
            exit 1
        fi
        return
    fi

    cd "$MEGATRON_PATH"
    for jsonl in "${files[@]}"; do
        local name
        name=$(basename "$jsonl" .jsonl)
        local prefix="${out_dir}/${name}"
        local bin_file="${prefix}_text_document.bin"
        local idx_file="${prefix}_text_document.idx"

        if [ -f "$bin_file" ] && [ -f "$idx_file" ]; then
            echo "skip existing ${bin_file}"
            continue
        fi

        echo "preprocess ${jsonl} -> ${prefix}_text_document.{bin,idx}"
        python tools/preprocess_data.py \
            --input "$jsonl" \
            --output-prefix "$prefix" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --json-keys text \
            --append-eod \
            --workers "$PREPROCESS_WORKERS"
    done
}

preprocess_split train
preprocess_split valid

BASE_PATH="$OUT_BASE" \
TRAIN_BASE_PATH="${OUT_BASE}/train" \
VALID_BASE_PATH="${OUT_BASE}/valid" \
TOKENIZER_MODEL="$TOKENIZER_MODEL" \
bash "${SCRIPT_DIR}/muonball.sh"
