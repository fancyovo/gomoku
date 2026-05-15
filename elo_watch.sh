#!/bin/bash
# Continuous ELO monitor: watches for new checkpoints, plays matches, updates plot.
# Resume-safe: caches results in elo_cache.json.
#
# Usage:  ./elo_watch.sh
#         GAMES=400 BATCH=512 ./elo_watch.sh

set -euo pipefail
cd "$(dirname "$0")"

GAMES="${GAMES:-200}"
BATCH="${BATCH:-256}"
INTERVAL="${INTERVAL:-10}"
DEVICE="${DEVICE:-cuda}"

echo "ELO monitor starting..."
echo "  Games/pair: $GAMES  Batch: $BATCH  Interval: ${INTERVAL}s  Device: $DEVICE"
echo

exec python scripts/elo_watch.py \
    --games_per_pair "$GAMES" \
    --batch "$BATCH" \
    --interval "$INTERVAL" \
    --device "$DEVICE" \
    "$@"
