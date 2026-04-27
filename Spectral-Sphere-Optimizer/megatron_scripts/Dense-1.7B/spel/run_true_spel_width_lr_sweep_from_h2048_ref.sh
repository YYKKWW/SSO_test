#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Single-GPU reproduction wrapper for the true SpEL h2048 reference log.
# It keeps the reference architecture rule:
#   layers=28, seq=4096, head_dim=128, ffn_hidden=3*hidden,
#   and sweeps width/lr under the same SpEL defaults.
#
# Start with DRY_RUN=1 to print the manifest. Use SMOKE=1 for a quick check.

WORKSPACE=${WORKSPACE:-/workspace}
SESSION_NAME=${SESSION_NAME:-true_spel_mup_width_lr_sweep_from_h2048_ref}
REPO_PATH=${REPO_PATH:-${WORKSPACE}/results/optimizer_arena_v2/true_spel_mup_width_lr_sweep_from_h2048_ref}

# Default to the small-width grid requested for muP transfer from the h2048 run.
WIDTHS=${WIDTHS:-"256 512"}
LRS=${LRS:-"1e-3 3e-3 5e-3 7e-3 9e-3"}
# Match the prior 1B-token budget used by the h2048 reference runs.
# With global_batch=64 and seq_length=4096, 1B tokens is about 3815 iterations.
TRAIN_TOKENS=${TRAIN_TOKENS:-1000000000}
TRAIN_ITER=${TRAIN_ITER:-3815}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
NUM_LAYERS=${NUM_LAYERS:-28}
HEAD_DIM=${HEAD_DIM:-128}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-4}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-500}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-10}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fused}
BASE_PORT=${BASE_PORT:-29700}
DRY_RUN=${DRY_RUN:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = "1" ]; then
    echo "Running a single true-SpEL reference smoke test: width=256, ${TRAIN_ITER:-10} iterations."
    OPTIMIZER=${OPTIMIZER:-spel_dist} \
    WIDTH=${WIDTH:-256} \
    NUM_LAYERS="$NUM_LAYERS" \
    HEAD_DIM="$HEAD_DIM" \
    SEQ_LENGTH="$SEQ_LENGTH" \
    GLOBAL_BATCH="$GLOBAL_BATCH" \
    MICRO_BATCH="$MICRO_BATCH" \
    TRAIN_TOKENS="${TRAIN_TOKENS:-}" \
    TRAIN_ITER=${TRAIN_ITER:-10} \
    LR_WARMUP_ITERS="$LR_WARMUP_ITERS" \
    EVAL_INTERVAL=${SMOKE_EVAL_INTERVAL:-100} \
    EVAL_ITERS=${SMOKE_EVAL_ITERS:-1} \
    SAVE_INTERVAL=${SMOKE_SAVE_INTERVAL:-100} \
    SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-0} \
    LOG_INTERVAL="$LOG_INTERVAL" \
    MASTER_PORT=${MASTER_PORT:-29821} \
    JOB_NAME=${JOB_NAME:-true_spel_ref_width256_smoke} \
    REPO_PATH="${REPO_PATH}_smoke" \
    TRANSFORMER_IMPL="$TRANSFORMER_IMPL" \
    ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    BASE_PATH=${BASE_PATH:-${WORKSPACE}/data/merged_data} \
    TRAIN_BASE_PATH=${TRAIN_BASE_PATH:-${WORKSPACE}/data/merged_data/train} \
    VALID_BASE_PATH=${VALID_BASE_PATH:-${WORKSPACE}/data/merged_data/valid} \
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-${WORKSPACE}/models/OLMo-2-1124-7B} \
    bash "${SCRIPT_DIR}/spel.sh"
    exit 0
fi

echo "Single-GPU true-SpEL width/lr sweep from h2048 reference"
echo "  widths: ${WIDTHS}"
echo "  lrs: ${LRS}"
echo "  train_tokens: ${TRAIN_TOKENS:-unset}"
echo "  train_iter: ${TRAIN_ITER}"
echo "  seq_length: ${SEQ_LENGTH}"
echo "  layers: ${NUM_LAYERS}"
echo "  head_dim: ${HEAD_DIM}"
echo "  global_batch: ${GLOBAL_BATCH}"
echo "  repo_path: ${REPO_PATH}"

MANIFEST="${REPO_PATH}/sweep_manifest.tsv"
mkdir -p "$REPO_PATH"

write_command() {
    local width=$1
    local lr=$2
    local port=$3
    local job_name=$4

    cat <<EOF
source /workspace/venv/bin/activate
WIDTH=${width} \\
LR=${lr} \\
TRAIN_TOKENS=${TRAIN_TOKENS} \\
TRAIN_ITER=${TRAIN_ITER} \\
LR_WARMUP_ITERS=${LR_WARMUP_ITERS} \\
NUM_LAYERS=${NUM_LAYERS} \\
HEAD_DIM=${HEAD_DIM} \\
SEQ_LENGTH=${SEQ_LENGTH} \\
GLOBAL_BATCH=${GLOBAL_BATCH} \\
MICRO_BATCH=${MICRO_BATCH} \\
SAVE_INTERVAL=${SAVE_INTERVAL} \\
SAVE_CHECKPOINT=${SAVE_CHECKPOINT} \\
EVAL_INTERVAL=${EVAL_INTERVAL} \\
EVAL_ITERS=${EVAL_ITERS} \\
LOG_INTERVAL=${LOG_INTERVAL} \\
TRANSFORMER_IMPL=${TRANSFORMER_IMPL} \\
ATTENTION_BACKEND=${ATTENTION_BACKEND} \\
MASTER_PORT=${port} \\
JOB_NAME=${job_name} \\
REPO_PATH=${REPO_PATH} \\
bash ${SCRIPT_DIR}/spel.sh
EOF
}

{
    printf "optimizer\twidth\tlr\tjob_name\tlog_file\n"
    run_id=0
    for width in $WIDTHS; do
        for lr in $LRS; do
            job_lr=${lr//./p}
            job_lr=${job_lr//-/m}
            job_name="mup_spel_dist_h${width}_lr${job_lr}"
            log_file="${REPO_PATH}/logs/${job_name}.log"
            printf "spel_dist\t%s\t%s\t%s\t%s\n" "$width" "$lr" "$job_name" "$log_file"
            run_id=$((run_id + 1))
        done
    done
} > "$MANIFEST"

if [ "$DRY_RUN" = "1" ]; then
    echo "Wrote manifest: $MANIFEST"
    echo "DRY_RUN=1, no jobs launched."
    echo "Launch with:"
    echo "  DRY_RUN=0 bash $0"
    exit 0
fi

tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "tmux session ${SESSION_NAME} already exists. Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
}

tmp_script=$(mktemp "/tmp/${SESSION_NAME}.XXXXXX.sh")
{
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    echo "mkdir -p ${REPO_PATH}/logs"
    run_id=0
    for width in $WIDTHS; do
        for lr in $LRS; do
            port=$((BASE_PORT + run_id))
            job_lr=${lr//./p}
            job_lr=${job_lr//-/m}
            job_name="mup_spel_dist_h${width}_lr${job_lr}"
            log_file="${REPO_PATH}/logs/${job_name}.log"
            echo "if [ \"${SKIP_COMPLETED}\" = \"1\" ] && [ -f \"${log_file}\" ] && grep -q 'validation loss' \"${log_file}\"; then"
            echo "  echo '===== SKIP completed ${job_name} ====='"
            echo "  continue"
            echo "fi"
            echo "echo '===== START ${job_name} ====='"
            write_command "$width" "$lr" "$port" "$job_name"
            echo "echo '===== DONE ${job_name} ====='"
            run_id=$((run_id + 1))
        done
    done
} > "$tmp_script"
chmod +x "$tmp_script"

tmux new-session -d -s "$SESSION_NAME" "bash $tmp_script"

echo "Launched sweep in tmux session: $SESSION_NAME"
echo "Manifest: $MANIFEST"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: ${REPO_PATH}/logs"
