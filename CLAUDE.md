# Gomoku Transformer — Agent Guide

## Quick Context
- Transformer-based Gomoku AI trained with REINFORCE policy gradient
- Board engine in C++ (bitboard, OpenMP), bridge via pybind11
- Inference uses KV cache, bf16 mixed precision
- Training: 32K self-play games/step, page_size=32

## Memory
Project memory is stored in `MEMORY.md` (index) + `memory/` directory (entries).
Read these first when starting a new session.

## Common Operations
```bash
# Training
CONFIG=configs/default.yaml ./run.sh

# ELO monitor (on idle GPU)
N_MODELS=8 GAMES=64 ./elo_watch.sh

# Profile multi-model batching
python scripts/bench_batch_elo.py --n_models 5 --games_per_dir 32
```

## Key Files
- `src/model/transformer.py` — Model (RMSNorm, SwiGLU, FlashAttention, KV cache)
- `src/training/self_play.py` — Self-play with page-based inference
- `src/training/trainer.py` — REINFORCE trainer with mask support
- `scripts/elo_watch.py` — Multi-model batched ELO monitor
- `src/cpp/board.h` — Bitboard engine with inline functions
- `configs/default.yaml` — Base experiment config
