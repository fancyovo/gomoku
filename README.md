# Gomoku AlphaZero

15×15 Gomoku AI trained via AlphaZero-style self-play with MCTS + policy/value distillation.

## Architecture

- **Model**: 4+4+4 Transformer (shared + policy + value branches), d_model=128, 4 heads, d_ff=256 (~2.15M params)
- **Value head**: 2-class softmax → scalar = p_win - p_lose (CE/ln(2) loss)
- **Policy head**: 225-class softmax (CE/ln(225) loss)
- **Inference**: KV cache, bf16 autocast
- **Board engine**: C++17 bitboard with OpenMP, pybind11 bridge

## Training Pipeline

```bash
# Generate initial pool data
python scripts/gen_data.py --output_dir data/init_pool --num_files 8

# Train from scratch (no pretrain)
python scripts/train_replay.py \
    --G 512 --M 8 --S 64 --n_steps 10 \
    --ckpt_dir checkpoints/run1 \
    --data_dir data/init_pool \
    --from_scratch

# ELO evaluation
python scripts/eval_elo_curve.py --ckpt_dir checkpoints/run1
```

**Replay buffer**: Pool size = 8×G (4096 games). Each step: self-play G games → discard G random old → add new → train 1 epoch on full pool.

## Experiment Record

### Fix v5 (latest, 2026-06, c_puct=0.2)
**Config**: `--G 2048 --M 8 --S 64 --n_steps 500 --lr 1e-4 --c_puct 0.2 --single_loss --pool_mult 2 --train_fraction 0.625 --from_scratch`

**Result: FAILURE** — model learns basic stone clustering and simple row/column attack patterns, but cannot:
- Defend against opponent sleep-4 threats (4-in-a-row)
- Execute coordinated attacks
- Improve policy beyond near-random (test_p ~0.98 throughout)
- Stabilize value predictions (test_v ~0.95 after 200+ steps)
- Self-play games shorten from ~60 to ~20-30 plies, B/W balance oscillates.

**Bugs fixed**: `TransformerBlock.forward` residual, SDPA mask inverted, branch cache missing in `prefill_extend`. All fixes are correct but did not resolve the fundamental failure: the model does not learn from MCTS self-play.

## Key Files

| File | Purpose |
|------|---------|
| `src/model/transformer.py` | GomokuTransformer (4+4+4 split architecture) |
| `src/model/embeddings.py` | ActionEmbedding (position + player + spatial coord) |
| `src/training/loss.py` | alphago_zero_loss (policy CE + value CE) |
| `src/training/replay.py` | Replay buffer, augment, collate, train loop, self-play |
| `src/training/augment.py` | 8× D4 symmetry augmentation |
| `src/cpp/mcts.cpp` | MCTS (PUCT, Dirichlet noise, virtual loss) |
| `scripts/train_replay.py` | Main training entry point |
| `scripts/eval_elo_curve.py` | ELO evaluation + curve plot |
| `scripts/gen_data.py` | Initial pool data generation |

## Results

| Experiment | Steps | Peak ELO | Notes |
|------------|-------|----------|-------|
| pretrain10 | pretrain + 10 | 1802 (step_2) | Old buggy pretrain data, best single model |
| scratch5 | 5 (from scratch) | 1795 (step_4) | No pretrain, consistent improvement |
| scratch15 | 15 (from scratch) | 1741 (step_11) | ΔELO=+626, step_0 below noisy_uniform |
| G=2048 | pretrain + 14 | 1729 (step_11) | Training unstable, step_14 collapse |

## Known Issues

1. **Policy head stagnation**: test_p flat at ~0.984 despite value head improvement. Model plays via strong value signal.
2. **S=16 sensitivity**: At low simulation counts, tiny policy non-uniformities cause large ELO shifts.
3. **Replay buffer slow turnover**: 1/8 replacement per step means old data dominates early training.