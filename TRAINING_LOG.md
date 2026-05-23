# Training Debug Log

## Timeline

| # | Date | Description | Result |
|---|------|-------------|--------|
| 1 | 05/21 | Initial scan: found value target shift bug (tv[:,1:]→tv[:,:-1]) | Fixed |
| 2 | 05/21 | Found dangling pointer in bindings.cpp select_all numpy arrays | Fixed |
| 3 | 05/21 | Found N_total double-count in mcts.cpp select_leaf | Fixed |
| 4 | 05/21 | First run (all fixes, epoch=unlimited): ΔELO=-47, BWR oscillates | Failed |
| 5 | 05/21 | epoch=5: ΔELO=-10, BWR collapses to 34% | Failed |
| 6 | 05/21 | Self-play diagnosis: MCTS distributions near-uniform with random model | Root cause identified |
| 7 | 05/21 | Found N_total=0 after fix (missing init N_total=1) | Fixed |
| 8 | 05/21 | epoch=5 + N_total fix: BWR stable 47-54%, ΔELO=-23 | Not enough signal |
| 9 | 05/21 | Switched value head to 2-class classification (CE loss) | Architecture change |
| 10 | 05/21 | epoch=5: v=0.02 anomaly → found data leak in train/test split | Critical bug found |
| 11 | 05/21 | Fixed: split at game level before augmentation | Fixed |
| 12 | 05/21 | epoch=unlimited + fixed split: BWR=50-55%, ΔELO=-23 | Still not learning |
| 13 | 05/21 | G=2048 games/step: still p≈0.98, v≈1.0, epoch=1-4 | No improvement |
| 14 | 05/21 | Endgame diagnosis: value predictions IDENTICAL across all games | Model ignores board state |
| 15 | 05/21 | Exponential decay weights (half-life=5): p=0.98, v drops too fast → normalized | Fixed norm |
| 16 | 05/22 | Decay weights (re-normed): p still 0.98, BWR=50-57% | p doesn't move |
| 17 | 05/22 | Dual-head architecture (4 shared + 4 policy + 4 value): p still frozen | Failed |
| 18 | 05/22 | Policy-only training test: Top-1 match reaches 72% on training data | Policy CAN learn! |
| 19 | 05/22 | Policy-only with train/test split: train_p=0.95, test_p=0.99 → pure overfitting | Policy does NOT generalize |

## Current State

- **Value head**: 2-class softmax (CE loss, scaled 1/ln(2))
- **Policy head**: CE on MCTS visit distribution (scaled 1/ln(225))
- **Train/test split**: game-level (fixed data leak)
- **G=512 games/step**, 8×64 MCTS simulations
- **Model**: 16-layer transformer, d_model=128

## Key Findings

### Root Cause Analysis

1. **MCTS with random model produces near-uniform visit distributions** on 15×15 board with 512 simulations. MCTS target entropy ≈ 4.0-4.3 (uniform=5.42). Not informative enough for generalization.

2. **Policy CAN overfit** training data (p drops to 0.95, Top-1 match 72%) but **cannot generalize** to validation set (p stays at 0.99 ≈ random).

3. **Value head learns quickly** (BWR parity + board patterns at endgame) but doesn't help policy.

4. **With both heads active, early stopping kills training** before policy has any chance (value loss drops fast → total loss plateaus → stop at epoch 1-4).

5. **The fundamental bottleneck**: 512 random-model MCTS simulations on 15×15 produce too little signal for the model to learn generalizable Gomoku strategy. More games help overfitting but not generalization.

### Bugs Found & Fixed

| Bug | File | Impact |
|-----|------|--------|
| Value target shift tv[:,1:]→tv[:,:-1] | train_5steps.py | Wrong sign for value targets |
| select_all numpy array use-after-free | bindings.cpp | Memory corruption |
| N_total double-count | mcts.cpp | Inflated PUCT exploration |
| N_total=0 after expansion | mcts.cpp | PUCT U=0, random selection |
| Train/test split at sample level | inline scripts | Data leakage between train/val |
| ELO display missing models | inline scripts | uniform key collides with step_000000 |

### Files Modified

- `src/model/transformer.py` — value head 2-class, _value_to_scalar, dual-head attempt (reverted)
- `src/training/loss.py` — alphago_zero_loss with 2-class CE, decay weights, normalization
- `src/cpp/bindings.cpp` — fixed dangling pointers, added <cstring>
- `src/cpp/mcts.cpp` — removed N_total++, added N_total=1 on expand
- `scripts/train_5steps.py` — fixed value shift
- `scripts/train_5steps_with_elo.py` — fixed value shift
- `scripts/train_and_analyze.py` — fixed C++ API calls
- `.gitignore` — added .venv/

---

## Round 2: Data Generation & Pretraining (05/23)

| # | Description | Result |
|---|-------------|--------|
| 20 | CPU data generation: uniform dummy model + MCTS (512 sims, 1×512 serial) × 65536 games (128 files) | Data saved (~3.5GB) |
| 21 | Pretrain 1 epoch on 65536 games: p from 0.99→0.95, v from 1.0→0.9 | Slight improvement |
| 22 | Pretrained vs Uniform baseline (256 games, 8×64 MCTS): 115-141, WR=44.9% | Pretrained WORSE than uniform! |
| 23 | Policy-only pretrain (freeze value): vs Uniform 126-130 WR=49.2% | No improvement |
| 24 | Value-only pretrain (freeze policy): vs Uniform 123-133 WR=48.0% | No improvement |
| 25 | 50-step training from pretrained (8×64 MCTS, game-level split, decay hl=5): step 1 test_v=4.0 | Catastrophic value loss |
| 26 | Diagnosed test_v=4.0: caused by early_stop=20 running 20+ epochs, overfitting in 1 step | test_v stays 0.98-1.14 with 1-5 epochs |
| 27 | Fixed collate shape bug: `pol[i,:L_-1]=s['mcts_policies']` fails when padded to L entries | Fixed with [:L_-1] |
| 28 | Retried 50-step training with fixes: still no ELO improvement | Failed |
| 29 | MCTS diagnosis: 2048 simulations (1×2048 serial) produces concentrated endgame (top1=92-97%) vs 512 sims (top1=5-12%) | 512 sims too few for 15×15 |
| 30 | Dirichlet noise test: alpha=0.03 vs 0.3 vs eps=0.1 — no significant difference | Dirichlet not the bottleneck |
| 31 | Single-game MCTS trace: all moves before terminal have ent≈4.7-5.2 (near-uniform); only final winning move ent≈0.2 | MCTS can only find depth-1 terminals |

## Key Conclusions (Round 2)

1. **512 sims + uniform prior + 15×15 = near-uniform MCTS visit distributions** at all positions except the final winning move. Data lacks sufficient signal for policy learning.
2. **2048 sims** produces concentrated endgame distributions (top1=92-97%) but still can't see beyond depth 1.
3. **Pretraining on 65536 games of low-quality data** gives slight loss improvement (p: 0.99→0.95, v: 1.0→0.9) but doesn't improve playing strength.
4. **The fundamental bottleneck**: MCTS with uniform/random model on 15×15 cannot produce informative training targets. The search depth is limited to 1 (only direct winning moves are found).
5. **Possible ways forward**: (a) much more simulations (e.g., 3200+), (b) smaller board (9×9), (c) use a heuristic value function instead of zero/uniform for data generation, (d) pure RL approach without MCTS distillation.

## Current State (end of Round 2)

- All known code bugs are fixed
- Training pipeline is correct (game-level split, proper normalization, decay weights)
- Value head learns quickly but overfits; policy head learns almost nothing
- MCTS with random model on 15×15 cannot bootstrap the training
- The AlphaZero-style distillation paradigm is not viable with current compute budget (512 sims, 15×15 board)
