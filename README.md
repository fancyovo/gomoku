# Gomoku Transformer

A Transformer-based Gomoku (15×15) AI trained via pure REINFORCE policy gradient with self-play.

## Architecture

- **Model**: Decoder-only Transformer, 16 layers, d_model=128, 4 heads, d_ff=256 (~2.8M params)
- **Inference**: Autoregressive with KV cache, BF16 mixed precision, 4096-game micro-batches
- **Board engine**: C++17 bitboard with OpenMP parallelism, pybind11 bridge to Python
- **Training**: Pure REINFORCE — self-play → policy gradient, no value network, no MCTS
- **Data**: 32,768 games per step, page_size=32, action masking (occupied positions → -inf)

## Quick Start

```bash
# Install
pip install -e .

# Train (base experiment)
CONFIG=configs/default.yaml ./run.sh

# ELO monitor (separate GPU)
./elo_watch.sh
```

## Ablation Experiments

```bash
# Fixed entropy coefficient (no annealing)
CONFIG=configs/abl_fixed_entropy.yaml ./run.sh

# Exponential reward decay (half-life = 10 steps)
CONFIG=configs/abl_reward_decay.yaml ./run.sh

# Data augmentation (8× symmetry)
CONFIG=configs/abl_augment.yaml ./run.sh

# Scaled-up model (24 layers, d_ff=384)
CONFIG=configs/abl_scale_up.yaml ./run.sh
```

Checkpoints auto-save to `checkpoints/<experiment_name>/`. The ELO monitor watches all directories
and produces a multi-curve comparison plot at `output/elo_curve.png`.

## Base Experiment Results

**Training**: 1000 steps completed, 100 checkpoints (step 9–999), ~32K games/step.

**ELO trajectory** (512 games/pair):

| Phase | Steps | ELO Range | Characteristics |
|-------|-------|-----------|-----------------|
| Random | 0–29 | 578→1224 | Learned "play in center, not corners" |
| Growth | 29–149 | 1224→1512 | Learned directional play, building lines |
| Plateau | 149–349 | ~1400–1500 | Stable but not improving |
| Surge | 349–539 | 1497→**1909** | Developed sharp attacking tactics (column 9 vertical line) |
| Regression | 539–999 | 1909→1543 | Strategy narrowed, lost adaptability |

**Peak performance**: step_000539 (ELO 1909) — systematically builds 5-in-a-row in ~18 moves.

**Key findings**:

1. **REINFORCE without value function oscillates**. The model cycles between discovering new tactics
   and overfitting to them. No monotonic improvement.

2. **Action masking is essential**. Without it, the model converges to 100% illegal moves
   (copying the last input token). Masking occupied positions during both inference and training
   eliminates this failure mode.

3. **Logit-target shift is critical**. The model's `logits[:, t]` is computed from `positions[:, 0..t]`
   via causal attention, so it should predict `positions[:, t+1]`. Training it to predict
   `positions[:, t]` causes the model to learn a trivial copy task.

4. **Strong non-transitivity**. 38% of checkpoint pairs have the earlier model beating the
   later one. Different checkpoints develop mutually countering strategies (rock-paper-scissors).

5. **Game length decreases with skill**. Random models average 72 moves; trained models
   average 30–35 moves. Better models end games faster through deliberate attacking.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.d_model` | 128 | Hidden dimension |
| `model.n_layers` | 16 | Transformer layers |
| `model.n_heads` | 4 | Attention heads |
| `training.lr` | 1e-3 | Learning rate |
| `training.entropy_coef` | 0.01 | Initial entropy coefficient |
| `training.entropy_fixed` | null | Freeze entropy (skip annealing) |
| `training.reward_decay_half_life` | null | Exponential reward decay |
| `training.games_per_step` | 32768 | Self-play games per training step |
| `training.page_size` | 32 | Block size for batched inference |
| `training.train_batch_size` | 512 | Training batch size |
| `training.augment` | false | 8× symmetry data augmentation |

## ELO Monitor

```bash
./elo_watch.sh                           # default: 512 games/pair
GAMES=256 BATCH=128 ./elo_watch.sh       # faster, less accurate
```

- Watches all `checkpoints/*/` directories
- Randomly samples unplayed checkpoint pairs
- Caches results in `elo_caches/<experiment>.json`
- Produces `output/elo_curve.png` with all experiments overlaid

## Project Structure

```
gomoku_transformer/
├── configs/           # YAML configs (base + 4 ablations)
├── src/
│   ├── cpp/          # C++ board engine (bitboard + OpenMP)
│   ├── model/        # Transformer (RMSNorm, SwiGLU, FlashAttention)
│   ├── training/     # Self-play runner, REINFORCE trainer, DataLoader
│   └── monitoring/   # Wandb logger
├── scripts/          # train.py, elo_watch.py, elo_tournament.py
├── checkpoints/      # Per-experiment checkpoint directories
├── elo_caches/       # ELO pairwise result caches
├── output/           # Plots and analysis output
├── run.sh            # Training launcher
└── elo_watch.sh      # ELO monitor launcher
```
