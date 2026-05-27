#!/usr/bin/env python3
"""ELO evaluation using value-only MCTS (uniform policy prior, model value head).

Usage: python scripts/eval_elo_value.py --ckpt_dir checkpoints/run20_lr1e4 --max_gap 4
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


# ── Uniform-policy wrapper: replaces policy head with uniform ──
class ValueOnlyModel:
    """Wraps a trained model, replacing policy logits with uniform."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.config = model.config

    def create_cache(self, max_games, max_cache_len=250):
        return self.model.create_cache(max_games, max_cache_len)

    def sample_first_moves(self, bs, dev):
        return torch.randint(0, N_CELLS, (bs,), device=dev)

    def prefill(self, pos, plr, kv_cache, branch_cache, indices):
        _, v = self.model.prefill(pos, plr, cache, _br_cache, indices)
        # Replace policy with uniform
        B = pos.shape[0] if isinstance(pos, torch.Tensor) else len(pos)
        u = torch.zeros(B, N_CELLS, device=self.device)
        return u, v

    def decode(self, pos, plr, kv_cache, branch_cache, indices):
        _, v = self.model.decode(pos, plr, cache, _br_cache, indices)
        u = torch.zeros(len(indices), N_CELLS, device=self.device)
        return u, v

    def evaluate_mcts_leaves(self, pos, plr, kv_cache, indices, plen):
        _, v = self.model.evaluate_mcts_leaves(pos, plr, cache, indices, plen)
        u = torch.zeros(pos.shape[0], N_CELLS, device=self.device)
        return u, v


class NoisyUniform:
    def __init__(self, cfg=None):
        self.config = cfg
    def create_cache(self, max_games, max_cache_len=250):
        m = GomokuTransformer(self.config).to(DEVICE)
        return m.create_cache(max_games, max_cache_len)
    def sample_first_moves(self, bs, dev):
        return torch.randint(0, 225, (bs,), device=dev)
    def prefill(self, pos, plr, kv_cache, branch_cache, indices):
        return (torch.randn(pos.shape[0], 225, device=DEVICE) * 0.02,
                torch.zeros(pos.shape[0], device=DEVICE))
    def decode(self, pos, plr, kv_cache, branch_cache, indices):
        kv_cache.advance(indices)
        return (torch.randn(len(indices), 225, device=DEVICE) * 0.02,
                torch.zeros(len(indices), device=DEVICE))
    def evaluate_mcts_leaves(self, pos, plr, kv_cache, indices, plen):
        return (torch.randn(pos.shape[0], 225, device=DEVICE) * 0.02,
                torch.zeros(pos.shape[0], device=DEVICE))


# ── MCTS match (same as eval_elo_sparse) ───────────────────────
@torch.inference_mode()
def play_match(ma, mb):
    G_ = G_ELO; M_ = M_ELO; S_ = S_ELO
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    def mm():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = M_; m.dirichlet_eps = 0.0
        m.init_roots(np.zeros((G_, 225), dtype=bool),
                     np.zeros((G_, 225), dtype=bool),
                     np.zeros(G_, dtype=np.int32))
        return m

    mga = mm(); mgb = mm()
    kva, _br_kva = ma.create_cache(max_games=G_); kvb, _br_kvb = mb.create_cache(max_games=G_)
    ab = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    fin = np.zeros(G_, dtype=bool); res = np.zeros(G_, dtype=np.int32)
    p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)

    fa_a = ma.sample_first_moves(G_, DEVICE)
    fa_b = mb.sample_first_moves(G_, DEVICE)
    fa = np.zeros(G_, dtype=np.int64)
    for g in range(G_):
        fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
        p0[g, fa[g]] = True

    fat = torch.tensor(fa, dtype=torch.long, device=DEVICE).unsqueeze(1)
    pl0 = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    ma.prefill(fat, pl0, kva, _br_kva, list(range(G_)))
    mb.prefill(fat, pl0, kvb, _br_kvb, list(range(G_)))
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(fa[g]))
        if r:
            fin[g] = True; res[g] = r
        else:
            bp = np.zeros(225, dtype=bool); bp[int(fa[g])] = True
            mga.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))
            mgb.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))

    for move in range(1, 240):
        act = np.where(~fin)[0]
        if len(act) == 0: break
        cp = move % 2
        for mgr, mdl, kv in [(mga, ma, kva), (mgb, mb, kvb)]:
            st = torch.from_numpy(act).to(DEVICE)
            dp = torch.zeros(len(act), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(act), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(og[act], -1e9)
            mgr.expand_roots(act.astype(np.int32),
                             torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                             lv.cpu().numpy().astype(np.float32))
            for _ in range(S_):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                lp2, lv2 = mdl.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32),
                                      torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                      lv2.cpu().numpy().astype(np.float32))
        rpa = mga.get_root_policies(); rpb = mgb.get_root_policies()
        na = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            ua = ((cp == 0) == ab[g])
            pol = (rpa[g] if ua else rpb[g]).copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0:
                a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True; og[g, a] = True
        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        ma.decode(dp_, dpl_, kva, _br_kva, torch.from_numpy(act).to(DEVICE))
        mb.decode(dp_, dpl_, kvb, _br_kvb, torch.from_numpy(act).to(DEVICE))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                mga.apply_move(g, int(na[i]), p0[g], p1[g])
                mgb.apply_move(g, int(na[i]), p0[g], p1[g])

    del kva, kvb; torch.cuda.empty_cache()
    wa = wb = dr = 0
    for g in range(G_):
        w = res[g]
        if w == 1:
            if ab[g]: wa += 1
            else: wb += 1
        elif w == 2:
            if ab[g]: wb += 1
            else: wa += 1
        else: dr += 1
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
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--S', type=int, default=16)
    args = parser.parse_args()

    global G_ELO, M_ELO, S_ELO, DEVICE
    G_ELO, M_ELO, S_ELO = args.G, args.M, args.S
    DEVICE = torch.device("cuda")

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
    if not ckpts:
        print(f"ERROR: no step checkpoints in {args.ckpt_dir}")
        sys.exit(1)

    # Build models with value-only wrapper
    raw_models = {}
    for ckpt_name in [os.path.basename(c) for c in ckpts]:
        m = GomokuTransformer(cfg).to(DEVICE).eval()
        m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, ckpt_name), map_location=DEVICE))
        raw_models[ckpt_name] = ValueOnlyModel(m, DEVICE)

    uni = NoisyUniform(cfg=cfg)

    names = ['noisy_uniform'] + [os.path.basename(c) for c in ckpts]
    n_steps = len(ckpts)
    print(f"Value-only ELO (uniform policy + model value): {n_steps} checkpoints, max_gap={args.max_gap}")

    pairs = []
    for j in range(n_steps):
        pairs.append(('noisy_uniform', names[1 + j]))
    for i in range(n_steps):
        for j in range(i + 1, min(i + 1 + args.max_gap, n_steps)):
            pairs.append((names[1 + i], names[1 + j]))

    print(f"Pairs: {len(pairs)}")

    elo_results = []
    all_models = {'noisy_uniform': uni, **raw_models}

    for na, nb in pairs:
        print(f'  {na} vs {nb} ...', end=' ', flush=True)
        t0 = time.perf_counter()
        wa, wb, d = play_match(all_models[na], all_models[nb])
        dt = time.perf_counter() - t0
        wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
        elo_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
        print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)', flush=True)
        torch.cuda.empty_cache()

    elo = compute_elo(elo_results)

    print(f"\n=== Value-only ELO (uniform policy, S={S_ELO}) ===")
    for n in names:
        r = elo.get(n, 1500)
        tag = ' (baseline)' if n == 'noisy_uniform' else ''
        print(f'  {n:20s}: ELO={r:.0f}{tag}')

    steps = [(int(n.split('_')[1].split('.')[0]), elo.get(n, 1500))
             for n in names if n.startswith('step_')]
    steps.sort(key=lambda x: x[0])
    if len(steps) >= 2:
        print(f'  DeltaELO (step_0->step_{steps[-1][0]}) = {steps[-1][1] - steps[0][1]:+.0f}')

    noisy_elo = elo.get('noisy_uniform', 1500)
    step_nums = [s[0] for s in steps]
    step_elos = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(step_nums, step_elos, 'g-o', markersize=4, label='Value-only (uniform policy + MCTS)')
    ax.axhline(y=noisy_elo, color='gray', linestyle=':',
               label=f'Noisy uniform (ELO={noisy_elo:.0f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('ELO')
    ax.set_title(f'Value-Only ELO — Uniform Policy + MCTS (S={S_ELO}, sparse |i-j|≤{args.max_gap})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = f'{args.ckpt_dir}/elo_value_curve.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f'\nSaved: {out_path}')
    plt.close()

    # Cleanup
    del raw_models
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
