#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

GAMES="${GAMES:-1024}"
BATCH="${BATCH:-256}"
DEVICE="${DEVICE:-cuda}"

exec python scripts/elo_tournament.py \
    --games_per_pair "$GAMES" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    "$@"
