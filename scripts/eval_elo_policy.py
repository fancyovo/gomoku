#!/usr/bin/env python3
"""ELO evaluation using pure policy sampling (no MCTS).

Usage: python scripts/eval_elo_policy.py --ckpt_dir checkpoints/run20_lr1e4 --max_gap 4
"""
import argparse, os, sys, time, glob
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOARD_SIZE = 15
N_CELLS = 225
MAX_MOVES = 225


# ── NoisyUniform baseline ─────────────────────────────────────
class NoisyUniform:
    def __init__(self, cfg=None):
        self.config = cfg
    def create_cache(self, max_games, max_cache_len=250):
        m = GomokuTransformer(self.config).to(DEVICE)
        return m.create_cache(max_games, max_cache_len)
    def sample_first_moves(self, bs, dev):
        return torch.randint(0, 225, (bs,), device=dev)
    def prefill(self, pos, plr, cache, indices):
        return (torch.randn(pos.shape[0], 225, device=DEVICE) * 0.02,
                torch.zeros(pos.shape[0], device=DEVICE))
    def decode(self, pos, plr, cache, indices):
        cache.advance(indices)
        return (torch.randn(len(indices), 225, device=DEVICE) * 0.02,
                torch.zeros(len(indices), device=DEVICE))


# ── Policy-only match ─────────────────────────────────────────
@torch.inference_mode()
def play_match_policy(ma, mb):
    """Play a match using only policy sampling — no MCTS, no value head."""
    G_ = G_ELO
    pool = gomoku_cpp.GamePool(G_)
    pool.reset_all()

    kva = ma.create_cache(max_games=G_)
    kvb = mb.create_cache(max_games=G_)
    a_black = torch.tensor([i % 2 == 0 for i in range(G_)], device=DEVICE)

    finished = torch.zeros(G_, dtype=torch.bool)
    winners = torch.zeros(G_, dtype=torch.long)
    occupied = torch.zeros(G_, N_CELLS, dtype=torch.bool, device=DEVICE)
    idx = torch.arange(G_, device=DEVICE)

    # First moves
    fa_a = ma.sample_first_moves(G_, DEVICE)
    fa_b = mb.sample_first_moves(G_, DEVICE)
    first = torch.where(a_black, fa_a, fa_b)
    occupied[idx, first] = True

    # Prefill both models with the shared first move
    fat = first.unsqueeze(1)
    pl0 = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    ma.prefill(fat, pl0, kva, list(range(G_)))
    mb.prefill(fat, pl0, kvb, list(range(G_)))

    # Step first moves on C++ board
    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first[g].item()))
        if r:
            finished[g] = True
            winners[g] = r

    last_act = first
    last_plr = torch.zeros(G_, dtype=torch.long, device=DEVICE)

    for move in range(1, MAX_MOVES):
        act = torch.where(~finished)[0]
        if len(act) == 0:
            break
        cp = move % 2

        # Both models decode the last move
        logits_a, _ = ma.decode(last_act, last_plr, kva, idx)
        logits_b, _ = mb.decode(last_act, last_plr, kvb, idx)

        # Use appropriate model's policy for current player
        logits = torch.where(a_black.unsqueeze(1) if cp == 0 else (~a_black).unsqueeze(1),
                            logits_a, logits_b)
        logits = logits.masked_fill(occupied, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1)

        occupied[idx, action] = True
        last_act = action
        last_plr = torch.full((G_,), cp, dtype=torch.long, device=DEVICE)

        for g_ in range(G_):
            if finished[g_]:
                continue
            r = gomoku_cpp.step(pool, g_, int(action[g_].item()))
            if r:
                finished[g_] = True
                winners[g_] = r

    # Games that didn't finish: draw
    winners[(winners == 0) & ~finished] = 3

    del kva, kvb
    torch.cuda.empty_cache()

    wa = wb = dr = 0
    for g in range(G_):
        w = winners[g].item()
        if w == 1:
            if a_black[g]: wa += 1
            else: wb += 1
        elif w == 2:
            if a_black[g]: wb += 1
            else: wa += 1
        else:
            dr += 1
    return wa, wb, dr


# ── ELO solver ─────────────────────────────────────────────────
def compute_elo(match_results):
    names = sorted(set(a for a, _, _, _ in match_results) | set(b for _, b, _, _ in match_results))
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(500):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            n = sa + sb
            if n == 0: continue
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            da = (sa - ea * n) * (32.0 / n)
            elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--max_gap', type=int, default=4)
    parser.add_argument('--G', type=int, default=256)
    args = parser.parse_args()

    global G_ELO, DEVICE
    G_ELO = args.G
    DEVICE = torch.device("cuda")

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    uni = NoisyUniform(cfg=cfg)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
    if not ckpts:
        print(f"ERROR: no step checkpoints in {args.ckpt_dir}")
        sys.exit(1)

    names = ['noisy_uniform'] + [os.path.basename(c) for c in ckpts]
    n_steps = len(ckpts)
    print(f"Policy-only ELO: {n_steps} checkpoints, max_gap={args.max_gap}")

    # Build pair list
    pairs = []
    for j in range(n_steps):
        pairs.append(('noisy_uniform', names[1 + j]))
    for i in range(n_steps):
        for j in range(i + 1, min(i + 1 + args.max_gap, n_steps)):
            pairs.append((names[1 + i], names[1 + j]))

    print(f"Pairs: {len(pairs)}")

    elo_results = []
    model_cache = {'noisy_uniform': uni}

    for na, nb in pairs:
        print(f'  {na} vs {nb} ...', end=' ', flush=True)
        t0 = time.perf_counter()

        if na not in model_cache:
            m = GomokuTransformer(cfg).to(DEVICE).eval()
            m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, na), map_location=DEVICE))
            model_cache[na] = m
        if nb not in model_cache:
            m = GomokuTransformer(cfg).to(DEVICE).eval()
            m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, nb), map_location=DEVICE))
            model_cache[nb] = m

        wa, wb, d = play_match_policy(model_cache[na], model_cache[nb])
        dt = time.perf_counter() - t0
        wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
        elo_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
        print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)', flush=True)

        if na != 'noisy_uniform': del model_cache[na]
        if nb != 'noisy_uniform': del model_cache[nb]
        torch.cuda.empty_cache()

    elo = compute_elo(elo_results)

    print(f"\n=== Policy-only ELO (no MCTS) ===")
    for n in names:
        r = elo.get(n, 1500)
        tag = ' (baseline)' if n == 'noisy_uniform' else ''
        print(f'  {n:20s}: ELO={r:.0f}{tag}')

    steps = [(int(n.split('_')[1].split('.')[0]), elo.get(n, 1500))
             for n in names if n.startswith('step_')]
    steps.sort(key=lambda x: x[0])
    if len(steps) >= 2:
        print(f'  DeltaELO (step_0->step_{steps[-1][0]}) = {steps[-1][1] - steps[0][1]:+.0f}')

    # Plot
    noisy_elo = elo.get('noisy_uniform', 1500)
    step_nums = [s[0] for s in steps]
    step_elos = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(step_nums, step_elos, 'r-o', markersize=4, label='Policy-only (no MCTS)')
    ax.axhline(y=noisy_elo, color='gray', linestyle=':',
               label=f'Noisy uniform (ELO={noisy_elo:.0f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('ELO')
    ax.set_title(f'Policy-Only ELO — No MCTS (G={G_ELO}, sparse |i-j|≤{args.max_gap})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = f'{args.ckpt_dir}/elo_policy_curve.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
