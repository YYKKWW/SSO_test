#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKSPACE=${WORKSPACE:-/workspace}
SESSION_NAME=${SESSION_NAME:-mup_lr_width_sweep}
REPO_PATH=${REPO_PATH:-${WORKSPACE}/results/optimizer_arena_v2/mup_lr_width_sweep}

# Paper grid:
#   LRS="1e-3 3e-3 5e-3 7e-3 9e-3 1e-2 1.5e-2 2e-2 3e-2"
#   WIDTHS="256 512 1024 2048"
#   OPTIMIZERS="adam dist_muon spectral_ball_dist muon_ball_dist spel_dist"
LRS=${LRS:-"1e-3 3e-3 5e-3 7e-3 9e-3 1e-2 1.5e-2 2e-2 3e-2"}
WIDTHS=${WIDTHS:-"256 512 1024 2048"}
OPTIMIZERS=${OPTIMIZERS:-"adam dist_muon spectral_ball_dist"}

TRAIN_TOKENS=${TRAIN_TOKENS:-30000000000}
SEQ_LENGTH=${SEQ_LENGTH:-512}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
MICRO_BATCH=${MICRO_BATCH:-1}
NUM_LAYERS=${NUM_LAYERS:-4}
HEAD_DIM=${HEAD_DIM:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-10}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fused}
DRY_RUN=${DRY_RUN:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

BASE_PORT=${BASE_PORT:-29600}
MANIFEST="${REPO_PATH}/sweep_manifest.tsv"
mkdir -p "$REPO_PATH"

write_command() {
    local optimizer=$1
    local width=$2
    local lr=$3
    local port=$4
    local job_name=$5

    cat <<EOF
source /workspace/venv/bin/activate
OPTIMIZER=${optimizer} \\
LR=${lr} \\
HIDDEN_SIZE=${width} \\
HEAD_DIM=${HEAD_DIM} \\
NUM_LAYERS=${NUM_LAYERS} \\
SEQ_LENGTH=${SEQ_LENGTH} \\
GLOBAL_BATCH=${GLOBAL_BATCH} \\
MICRO_BATCH=${MICRO_BATCH} \\
TRAIN_TOKENS=${TRAIN_TOKENS} \\
LR_WARMUP_ITERS=\${LR_WARMUP_ITERS:-500} \\
SAVE_INTERVAL=${SAVE_INTERVAL} \\
SAVE_CHECKPOINT=${SAVE_CHECKPOINT} \\
EVAL_INTERVAL=${EVAL_INTERVAL} \\
EVAL_ITERS=${EVAL_ITERS} \\
LOG_INTERVAL=${LOG_INTERVAL} \\
MASTER_PORT=${port} \\
JOB_NAME=${job_name} \\
REPO_PATH=${REPO_PATH} \\
TRANSFORMER_IMPL=${TRANSFORMER_IMPL} \\
ATTENTION_BACKEND=${ATTENTION_BACKEND} \\
BASE_PATH=/workspace/data/merged_data \\
TRAIN_BASE_PATH=/workspace/data/merged_data/train \\
VALID_BASE_PATH=/workspace/data/merged_data/valid \\
TOKENIZER_MODEL=/workspace/models/OLMo-2-1124-7B \\
bash ${SCRIPT_DIR}/muonball.sh
EOF
}

{
    printf "optimizer\twidth\tlr\tjob_name\tlog_file\n"
    run_id=0
    for optimizer in $OPTIMIZERS; do
        for width in $WIDTHS; do
            for lr in $LRS; do
                job_lr=${lr//./p}
                job_lr=${job_lr//-/m}
                job_name="mup_${optimizer}_h${width}_lr${job_lr}"
                log_file="${REPO_PATH}/logs/${job_name}.log"
                printf "%s\t%s\t%s\t%s\t%s\n" "$optimizer" "$width" "$lr" "$job_name" "$log_file"
                run_id=$((run_id + 1))
            done
        done
    done
} > "$MANIFEST"

if [ "$DRY_RUN" = "1" ]; then
    echo "Wrote manifest: $MANIFEST"
    echo "DRY_RUN=1, no jobs launched."
    echo "Example launch for a tiny subset:"
    echo "  DRY_RUN=0 OPTIMIZERS='spel_dist' WIDTHS='256 512' LRS='3e-3 5e-3 7e-3' TRAIN_TOKENS=16000000 bash $0"
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
    for optimizer in $OPTIMIZERS; do
        for width in $WIDTHS; do
            for lr in $LRS; do
                port=$((BASE_PORT + run_id))
                job_lr=${lr//./p}
                job_lr=${job_lr//-/m}
                job_name="mup_${optimizer}_h${width}_lr${job_lr}"
                log_file="${REPO_PATH}/logs/${job_name}.log"
                echo "if [ \"${SKIP_COMPLETED}\" = \"1\" ] && [ -f \"${log_file}\" ] && grep -q 'validation loss' \"${log_file}\"; then"
                echo "  echo '===== SKIP completed ${job_name} ====='"
                echo "  continue"
                echo "fi"
                echo "echo '===== START ${job_name} ====='"
                write_command "$optimizer" "$width" "$lr" "$port" "$job_name"
                echo "echo '===== DONE ${job_name} ====='"
                run_id=$((run_id + 1))
            done
        done
    done
} > "$tmp_script"
chmod +x "$tmp_script"

tmux new-session -d -s "$SESSION_NAME" "bash $tmp_script"

echo "Launched sweep in tmux session: $SESSION_NAME"
echo "Manifest: $MANIFEST"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: ${REPO_PATH}/logs"
