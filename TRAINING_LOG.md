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
- Note: Subsequent Round 3 findings invalidated this conclusion; the training pipeline was fixed and shown to work.

---

## Round 3: MCTS Sign Bug Fix & Cold-Start Suicide Discovery (05/24)

| # | Date | Description | Result |
|---|------|-------------|--------|
| 32 | 05/24 | **Discovered MCTS backup sign bug**: values[li] is from last mover's perspective (= first edge owner), but `v=-v` was applied BEFORE `e.W+=v`, inverting all Q values | Fixed |
| 33 | 05/24 | Terminal backup had same bug, accidentally canceled by terminal_value=-1.0 starting from leaf perspective. Unified: terminal_value=+1.0, add-then-flip | Fixed |
| 34 | 05/24 | Coord embedding experiment: sin/cos 4D→128D concat into token embedding. Pretrain on 65K data | No improvement (loss stuck at 1.0) |
| 35 | 05/24 | First sign-fixed 5-step training (train_v3): p_loss stuck ~0.98, v_loss ~0.99. ELO showed models losing to uniform baseline | Confusing results |
| 36 | 05/24 | **Ablation on step_0**: policy_only=8.2%, value_only=48.4%, both=8.2% WR vs uniform at S=16 | Policy suicide! |
| 37 | 05/24 | **Ablation on step_2**: policy_only=32.2%, value_only=81.6%, both=25.6% at S=16 | Value head remarkably strong, policy head dragging it down |
| 38 | 05/24 | **Diagnosis**: step_0 policy prior is completely static (~0.0048, near-uniform). Model doesn't look at board. MCTS stuck in BFS at depth 1 | Root cause identified |
| 39 | 05/24 | **Pretrain on old 65K data then ablation**: policy_only=28.3%, value_only=68.4%, both=38.5% | Same suicide pattern, confirming it's data-driven, not training-loop-specific |
| 40 | 05/24 | **Random (untrained) model ablation**: policy_only=5.1%, value_only=56.1%, both=5.7% vs uniform at S=16 | Even untrained random model suicides! |
| 41 | 05/24 | **Noise test**: Uniform+noise(std=0.001) vs clean Uniform = **5.9%** WR. Any non-uniformity causes suicide at S=16. Noisy vs Noisy = 53.1% (symmetric) | **Definitive proof**: S=16 + Q=0 MCTS + any non-uniform prior = catastrophic |
| 42 | 05/24 | **Root cause identified**: With S=16, M=4, only 64/225 root actions explored. The 64 explored are top-P by policy prior. Random/untrained priors select arbitrary subsets that are systematically worse than uniform's index-order subset. This is a structural MCTS property, not a code bug | Fundamental insight |
| 43 | 05/24 | **Noisy uniform baseline ELO** for train_v3 step_0~4 at S=16: step_0=1318, step_1=1564, step_2=1610, step_3=1623, step_4=1611. Δ=+293. Training IS working! | Clean uniform had unfair structural advantage |
| 44 | 05/24 | 50-step from-scratch training (train_50). p_loss slowly drops 0.98→0.96 | Slow but steady |
| 45 | 05/24 | **Pretrain + 10-step self-play (pretrain10)**: pretrain=912, step_0=939, step_1=1670, step_2=1802, step_3=1745, step_4=1701... ΔELO=+661 | Huge improvement! |
| 46 | 05/24 | Old 65K data pretraining is actively harmful (ELO 912, below noisy_uniform at 1119). But subsequent self-play overcomes it quickly | Pretraining on bad data worse than nothing |
| 47 | 05/24 | **step_2 diagnosis**: Policy prior concentrates on center (H8, I11, K9). Value head calibrated (V>0 when winning, V<0 when losing). MCTS focuses 90%+ on winning moves. step_0 MCTS is always uniform 0.0312 | Clear qualitative improvement |
| 48 | 05/24 | **step_9 diagnosis**: Value head broken — always outputs V=+0.08~+0.14 regardless of board state. MCTS degenerates back to BFS despite strong policy prior (H8=0.0126). Explains why step_9 (ELO 1600) < step_2 (ELO 1802) | Value overfitting identified |
| 49 | 05/24 | **step_2 policy-only ablation**: vs noisy_uniform=94.9%, vs step_0=95.7%, vs step_2=54.7%, vs step_9=57.8%. Policy head alone is excellent! | Confirms policy DID learn |

## Key Findings (Round 3)

1. **MCTS sign bug (48e2c21)**: The backup loop applied `v=-v` BEFORE `e.W+=v`, inverting all Q values. Fixed by swapping order (add-then-flip) and unifying terminal_value=+1.0.

2. **S=16 + Q=0 MCTS structural property**: With S=16 and no value signal, MCTS explores only M×S=64 of 225 root actions — those with highest policy prior. Uniform+noise(std=0.001) vs clean Uniform achieved only 5.9% WR. Two noisy models vs each other: 53.1%. This means any deviation from perfect uniformity causes large WR shifts at S=16, regardless of move quality. S=64 does not show this because 4×64=256 > 225.

3. **Training with noisy-uniform baseline (S=16, separate ELO runs — not cross-comparable)**:
   - train_v3 (from scratch, 5 steps, 1 run): step_0=1318, step_1=1564, step_2=1610, step_3=1623, step_4=1611. ΔELO=+293.
   - pretrain10 (pretrain 1 epoch on 65K old data + 10 self-play steps, separate run): pretrain=912, step_0=939, step_1=1670, step_2=1802, step_3=1745, step_4=1701, ..., step_9=1600. ΔELO(step_0→9)=+661.

4. **Policy learning**: step_2 policy-only vs noisy_uniform: 243-13 (94.9% WR); vs step_0 policy-only: 245-11 (95.7% WR); vs step_9 policy-only: 148-108 (57.8% WR); vs self: 140-116 (54.7% WR). Policy head improved dramatically from step_0 to step_2, concentrating on center positions (H8, I11, K9).

5. **Value learning**: step_2 value-only vs noisy_uniform at S=16: 81.6% WR (ablation on train_v3). In detailed game traces, step_2 value is calibrated (V>0 when winning, V<0 when losing), while step_9 value is near-constant +0.08~+0.14 regardless of board state, causing MCTS to degenerate to BFS.

6. **Pretrain10 ELO trend (single run, noisy_uniform baseline at S=16)**: pretrain=912, step_0=939, step_1=1670, step_2=1802. The pretrained model alone was weaker than noisy_uniform (1119) in this run, but self-play training produced rapid improvement (step_0→step_1 +731 ELO).

7. **Dirichlet noise confirmation**: `expand_roots` is called every move during self-play and always applies Dirichlet noise. The earlier analysis claiming missing noise was incorrect.

## Current State (end of Round 3)

- **MCTS sign bug fixed** (48e2c21)
- **Best model**: pretrain10/step_000002.pt (ELO 1802 at S=16, policy-only 94.9% vs noisy_uniform)
- **Pipeline verified**: self-play training produces measurable ELO improvement
- **S≥57 needed** for Q=0 MCTS to avoid structural concentration on ~64 actions
- **Open**: Value head overfitting in later steps (step_9); whether pretraining helps vs from-scratch not yet tested in a single ELO run
