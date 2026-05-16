#!/bin/bash
# Continuous multi-experiment ELO monitor.
# Usage:
#   ./elo_watch.sh
#   GAMES=512 BATCH=256 ./elo_watch.sh

set -euo pipefail
cd "$(dirname "$0")"

GAMES="${GAMES:-128}"
N_MODELS="${N_MODELS:-5}"
INTERVAL="${INTERVAL:-10}"
DEVICE="${DEVICE:-cuda}"
MODEL_CONFIG="${MODEL_CONFIG:-}"

# Watch directories
WATCH_DIRS=(
    "checkpoints/base"
    "checkpoints/fixed_entropy"
    "checkpoints/reward_decay"
    "checkpoints/augment"
    "checkpoints/scale_up"
)

WATCH_ARGS=()
for d in "${WATCH_DIRS[@]}"; do
    [ -d "$d" ] && WATCH_ARGS+=(--watch_dir "$d")
done
[ ${#WATCH_ARGS[@]} -eq 0 ] && WATCH_ARGS=(--watch_dir "checkpoints/base")

MODEL_CONFIG_ARG=""
[ -n "$MODEL_CONFIG" ] && MODEL_CONFIG_ARG="--model_config $MODEL_CONFIG"

echo "ELO monitor starting..."
echo "  Watch dirs: ${WATCH_ARGS[*]}"
echo "  Games/pair: $GAMES  N models: $N_MODELS  Interval: ${INTERVAL}s"
echo

exec python scripts/elo_watch.py \
    --games_per_pair "$GAMES" \
    --n_models "$N_MODELS" \
    --interval "$INTERVAL" \
    --device "$DEVICE" \
    $MODEL_CONFIG_ARG \
    "${WATCH_ARGS[@]}"
