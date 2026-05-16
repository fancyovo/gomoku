#!/bin/bash
# Continuous multi-experiment ELO monitor.
# Usage:
#   ./elo_watch.sh
#   GAMES=512 BATCH=256 ./elo_watch.sh

set -euo pipefail
cd "$(dirname "$0")"

GAMES="${GAMES:-512}"
BATCH="${BATCH:-256}"
INTERVAL="${INTERVAL:-10}"
DEVICE="${DEVICE:-cuda}"
MODEL_CONFIG="${MODEL_CONFIG:-}"

# Watch directories — add new experiments here
WATCH_DIRS=(
    "checkpoints/base"
    "checkpoints/fixed_entropy"
    "checkpoints/reward_decay"
    "checkpoints/augment"
    "checkpoints/scale_up"
)

# Only watch dirs that exist
WATCH_ARGS=()
for d in "${WATCH_DIRS[@]}"; do
    if [ -d "$d" ]; then
        WATCH_ARGS+=(--watch_dir "$d")
    fi
done

# If no dirs exist yet, watch base
if [ ${#WATCH_ARGS[@]} -eq 0 ]; then
    WATCH_ARGS=(--watch_dir "checkpoints/base")
fi

MODEL_CONFIG_ARG=""
[ -n "$MODEL_CONFIG" ] && MODEL_CONFIG_ARG="--model_config $MODEL_CONFIG"

echo "ELO monitor starting..."
echo "  Watch dirs: ${WATCH_ARGS[*]}"
echo "  Games/pair: $GAMES  Batch: $BATCH  Interval: ${INTERVAL}s"
echo

exec python scripts/elo_watch.py \
    --games_per_pair "$GAMES" \
    --batch "$BATCH" \
    --interval "$INTERVAL" \
    --device "$DEVICE" \
    $MODEL_CONFIG_ARG \
    "${WATCH_ARGS[@]}"
