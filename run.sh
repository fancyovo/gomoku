#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-configs/default.yaml}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
RESUME="${RESUME:-}"
DEVICE="${DEVICE:-cuda}"

[ -f .env ] && source .env
RESUME_ARG=""
[ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"

export PYTORCH_ALLOC_CONF=expandable_segments:True

exec python scripts/train.py \
    --config "$CONFIG" \
    --total_steps "$TOTAL_STEPS" \
    --device "$DEVICE" \
    $RESUME_ARG
