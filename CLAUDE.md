# Gomoku Transformer — Agent Guide

## Quick Context
- Transformer-based Gomoku AI trained with AlphaZero-style self-play + MCTS distillation
- Board engine in C++ (bitboard, OpenMP), bridge via pybind11
- Inference uses KV cache, bf16 mixed precision
- Replay buffer training: G=512 games/step, pool=4096, alternating policy/value optimization
- Best models: scratch5/step_4 (ELO 1795), pretrain10/step_2 (ELO 1802)

## Cluster Resources
- Partition: Students, 2×GPU, 16×CPU, 1-day time limit
- Can run 2 jobs simultaneously (e.g., training + ELO monitor)
- SLURM defaults: `--gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00`

## Memory
Project memory is stored in `MEMORY.md` (index) + `memory/` directory (entries).
Read these first when starting a new session.

## Common Operations
```bash
# Training (from scratch, 10 steps)
sbatch job_run10_lr3e4.sh

# ELO monitor (on idle GPU)
N_MODELS=8 GAMES=64 ./elo_watch.sh

# Profile multi-model batching
python scripts/bench_batch_elo.py --n_models 5 --games_per_dir 32
```

## Key Files
- `src/model/transformer.py` — Model (RMSNorm, SwiGLU, FlashAttention, KV cache, 4+4+4 split)
- `src/training/replay.py` — Replay buffer self-play + training loop
- `src/training/loss.py` — AlphaGo Zero loss (CE policy + CE value, alternating)
- `scripts/train_replay.py` — Main training entry point
- `scripts/eval_elo_curve.py` — ELO evaluation + curve plot (S=16, noisy_uniform baseline)
- `src/cpp/board.h` — Bitboard engine with inline functions
