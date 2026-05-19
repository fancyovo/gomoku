#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
GAMES="${GAMES:-256}"

echo "=== Gomoku MCTS ELO Monitor ==="
echo "Watching: checkpoints/train_loop/"
echo "Games/pair: $GAMES"
echo "ELO curve: output/elo_curve.png"
echo "Device: $DEVICE"
echo

exec python -u scripts/elo_monitor.py --device "$DEVICE" --games_per_pair "$GAMES"
