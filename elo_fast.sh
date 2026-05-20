#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
GAMES="${GAMES:-256}"

echo "=== Gomoku ELO Fast (Pure Policy, No MCTS) ==="
echo "Watching: checkpoints/train_loop/"
echo "Games/pair: $GAMES (policy head direct sampling, no search)"
echo "Cache: output/elo_fast_cache.json"
echo "Plots: output/elo_fast_curve.png + output/elo_fast_heatmap.png"
echo "Device: $DEVICE"
echo

exec python -u scripts/elo_fast.py --device "$DEVICE" --games_per_pair "$GAMES"
