# Gomoku Transformer — Agent Guide

## Quick Context
- Transformer-based Gomoku AI trained with AlphaZero-style self-play + MCTS distillation
- Board engine in C++ (bitboard, OpenMP), bridge via pybind11
- Inference uses KV cache, bf16 mixed precision
- Replay buffer training: G=2048 games/step, pool=4096, policy+value joint optimization
- Current config (fix_v5): `--G 2048 --M 8 --S 64 --n_steps 500 --lr 1e-4 --c_puct 0.2 --single_loss --pool_mult 2 --train_fraction 0.625`

## Experiment Status (as of 2026-06-02)
- **Overall result: FAILURE** — model learns to cluster stones and has basic attack patterns (column/row 5-in-a-row setups) but CANNOT effectively organize offense or defend against opponent threats
- **Known limitations**:
  - Cannot defend sleep-4 (opponent 4-in-a-row threat) even with S=4096 MCTS
  - Value network barely converges (test_v ~0.95 after 200+ steps)
  - Policy network shows no improvement (test_p ~0.98 throughout)
  - Self-play B/W balance oscillates wildly, game length fluctuates
- **Bug history**:
  - `TransformerBlock.forward` residual bug (FFN used `x` instead of `h=x+attn`) — fixed
  - SDPA BoolTensor mask inverted in `forward_decode` and `prefill_extend` — fixed
  - `prefill_extend` missing branch cache for full sequence branch attention — fixed
  - Q_init=V_parent tried and reverted (caused instability)
- **Best models**: scratch5/step_4 (ELO 1795), pretrain10/step_2 (ELO 1802) — from very early experiments

## Cluster Resources
- Partition: Students, 2×GPU, 24×CPU, 1-day time limit
- Can run 2 jobs simultaneously (e.g., training + ELO monitor)
- SLURM defaults: `--gres=gpu:1 --cpus-per-task=12 --mem=32G --time=04:00:00`

## Memory
Project memory is stored in `MEMORY.md` (index) + `memory/` directory (entries).
Read these first when starting a new session.

## CRITICAL: Never Overwrite Past Experiments
- Every new experiment MUST use a NEW checkpoint directory (e.g., `checkpoints/run200_S128/`)
- NEVER `rm -rf checkpoints/` or delete any previous experiment's model weights or plots
- Checkpoint directories are permanent records — treat them as read-only after the experiment finishes
- Use descriptive directory names with key hyperparameters (e.g., `run200_S128_lr1e4_pool8`)

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
