#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-500}"

echo "=== Gomoku MCTS Training ==="
echo "Checkpoints: checkpoints/train_loop/"
echo "Device: $DEVICE  Steps: $STEPS"
echo "Auto-resumes from latest checkpoint"
echo

exec python -u scripts/train_loop.py --device "$DEVICE" --steps "$STEPS" 2>&1
