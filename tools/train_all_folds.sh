#!/bin/bash
# =============================================================================
# Train all 5 folds sequentially (single-GPU or DDP via torchrun).
#
# Usage:
#   bash tools/train_all_folds.sh --config configs/maskpolish_swinumamba.yaml
#   NPROC_PER_NODE=4 bash tools/train_all_folds.sh --config configs/maskpolish_swinumamba.yaml
#
# Notes:
#   - NPROC_PER_NODE=1 (default) -> single-GPU launch
#   - NPROC_PER_NODE>1         -> DDP launch with torchrun per fold
#   - Folds run one after another (0 -> 1 -> 2 -> 3 -> 4)
#   - Each fold log is written to logs/nohup_fold{N}.log (override with --log-dir)
#   - Stop this script to terminate the active fold.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Parse --log-dir from arguments early
LOG_DIR="${PROJECT_DIR}/logs"
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--log-dir" ]]; then
        LOG_DIR="${ARGS[$((i+1))]}"
        break
    fi
done
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "============================================================"
    echo "  Starting fold ${FOLD} / 4  (nproc=${NPROC_PER_NODE})"
    echo "============================================================"

    PGID_FILE="${PROJECT_DIR}/.train_pgid_fold${FOLD}"
    LOG_FILE="${LOG_DIR}/nohup_fold${FOLD}.log"

    # Launch in a new session so kill_train.sh can kill the whole group
    setsid torchrun --nproc-per-node "${NPROC_PER_NODE}" train.py "$@" \
        --fold "$FOLD" > "${LOG_FILE}" 2>&1 &
    BG_PID=$!
    PGID=$(ps -o pgid= "$BG_PID" | tr -d ' ')
    echo "${PGID}" > "${PGID_FILE}"
    echo "  PGID=${PGID}  log=${LOG_FILE}"
    echo "  Stop this script to terminate the active fold."

    # Wait for this fold to finish before starting the next
    wait "${BG_PID}"
    rm -f "${PGID_FILE}"
done

echo ""
echo "All 5 folds completed. Each fold directory contains training_log.txt and checkpoint_best.pth."
