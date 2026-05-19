#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"

echo "=== Gomoku MCTS Training ==="
echo "Checkpoints: checkpoints/train_loop/"
echo "Length curve: output/train_length.png"
echo "Device: $DEVICE"
echo

exec python -u scripts/train_loop.py --device "$DEVICE" 2>&1
