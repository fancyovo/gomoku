#!/usr/bin/env python3
"""Compare MCTS depths: S=64 vs S=32 vs S=16 vs raw policy.
Uses value head only (uniform policy prior) to isolate value head quality.
256 games per pair."""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp as gcpp

DEVICE = torch.device('cuda')
N_CELLS = 225
CKPT_DIR = 'checkpoints/run200_G2048_pool2x_0530_1019'
STEP = 3


class ValueOnlyModel:
    """Wraps a model: returns uniform policy + model's value."""
    def __init__(self, model):
        self.model = model

    def create_cache(self, max_games, max_cache_len=250):
        return self.model.create_cache(max_games, max_cache_len)

    def sample_first_moves(self, bs, dev):
        return torch.randint(0, N_CELLS, (bs,), device=dev)

    def prefill(self, pos, plr, kv_cache, branch_cache, indices):
        _, v = self.model.prefill(pos, plr, kv_cache, branch_cache, indices)
        return torch.zeros(pos.shape[0], N_CELLS, device=DEVICE), v

    def decode(self, pos, plr, kv_cache, branch_cache, indices):
        _, v = self.model.decode(pos, plr, kv_cache, branch_cache, indices)
        return torch.zeros(len(indices), N_CELLS, device=DEVICE), v

    def evaluate_mcts_leaves(self, pos, plr, kv_cache, indices, plen):
        _, v = self.model.evaluate_mcts_leaves(pos, plr, kv_cache, indices, plen)
        return (torch.zeros(pos.shape[0], N_CELLS, device=DEVICE), v)


def run_match(mdl, S_a, S_b, G=256, temp_moves=0):
    """Play G games between two wrappers. S=0 means raw value-only (no MCTS)."""
    pool = gcpp.GamePool(G); pool.reset_all()

    def mm():
        m = gcpp.MCTSManager(G, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = 4; m.dirichlet_eps = 0.0
        m.init_roots(np.zeros((G, N_CELLS), dtype=bool),
                     np.zeros((G, N_CELLS), dtype=bool), np.zeros(G, dtype=np.int32))
        return m

    mga = mm() if S_a > 0 else None
    mgb = mm() if S_b > 0 else None
    kva, brka = mdl.create_cache(max_games=G, max_cache_len=250)
    kvb, brkb = mdl.create_cache(max_games=G, max_cache_len=250)
    ab = np.array([i % 2 == 0 for i in range(G)], dtype=bool)
    fin = np.zeros(G, dtype=bool); res = np.zeros(G, dtype=np.int32)
    p0 = np.zeros((G, N_CELLS), dtype=bool); p1 = np.zeros((G, N_CELLS), dtype=bool)

    # First moves: uniform random (since ValueOnlyModel uses randint)
    fa_a = mdl.sample_first_moves(G, DEVICE)
    fa_b = mdl.sample_first_moves(G, DEVICE)
    fa = np.zeros(G, dtype=np.int64)
    for g in range(G):
        fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
        p0[g, fa[g]] = True
    fat = torch.tensor(fa, dtype=torch.long, device=DEVICE).unsqueeze(1)
    pl0 = torch.zeros(G, 1, dtype=torch.long, device=DEVICE)
    rpa0, rva0 = mdl.prefill(fat, pl0, kva, brka, list(range(G)))
    rpb0, rvb0 = mdl.prefill(fat, pl0, kvb, brkb, list(range(G)))
    rp_a = rpa0.float().clone(); rv_a = rva0.float().clone()
    rp_b = rpb0.float().clone(); rv_b = rvb0.float().clone()
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G):
        r = gcpp.step(pool, g, int(fa[g]))
        if r: fin[g] = True; res[g] = r
        else:
            bp = np.zeros(N_CELLS, dtype=bool); bp[int(fa[g])] = True
            if mga: mga.apply_move(g, int(fa[g]), bp, np.zeros(N_CELLS, dtype=bool))
            if mgb: mgb.apply_move(g, int(fa[g]), bp, np.zeros(N_CELLS, dtype=bool))

    for move in range(1, 240):
        act = np.where(~fin)[0]
        if len(act) == 0: break
        cp = move % 2
        act_np = act.astype(np.int32)
        act_t = torch.from_numpy(act).to(DEVICE)

        for mgr, pol_buf, val_buf, kv, brk, S_use in [
            (mga, rp_a, rv_a, kva, brka, S_a),
            (mgb, rp_b, rv_b, kvb, brkb, S_b),
        ]:
            if S_use == 0: continue
            lp = pol_buf[act].masked_fill(og[act], -1e9)
            lv = val_buf[act]
            mgr.expand_roots(act_np,
                             torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                             lv.cpu().numpy().astype(np.float32))
            for _ in range(S_use):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                mdl_local = mdl  # ValueOnlyModel or raw model
                lp2, lv2 = mdl_local.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32),
                                      torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                      lv2.cpu().numpy().astype(np.float32))

        # Get policies and select actions
        rpa = mga.get_root_policies() if S_a > 0 else np.zeros((G, N_CELLS), dtype=np.float32)
        rpb = mgb.get_root_policies() if S_b > 0 else np.zeros((G, N_CELLS), dtype=np.float32)
        na = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            ua = ((cp == 0) == ab[g])
            if ua:
                if S_a > 0:
                    pol = rpa[g].copy(); pol[p0[g] | p1[g]] = 0
                else:
                    # Raw value-only: uniform policy
                    pol = np.ones(N_CELLS, dtype=np.float32)
                    pol[p0[g] | p1[g]] = 0
            else:
                if S_b > 0:
                    pol = rpb[g].copy(); pol[p0[g] | p1[g]] = 0
                else:
                    pol = np.ones(N_CELLS, dtype=np.float32)
                    pol[p0[g] | p1[g]] = 0
            pol_sum = pol.sum()
            if pol_sum > 0:
                if temp_moves > 0 and move > temp_moves:
                    # Greedy: pick the most visited action
                    a = int(np.argmax(pol))
                else:
                    a = int(np.random.choice(N_CELLS, p=pol / pol_sum))
            else:
                legal = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            na[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True
            og[g, a] = True

        # Decode
        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        ds_ = torch.from_numpy(act).to(DEVICE)
        npa, nva = mdl.decode(dp_, dpl_, kva, brka, ds_)
        npb, nvb = mdl.decode(dp_, dpl_, kvb, brkb, ds_)
        rp_a[act_t] = npa.float(); rv_a[act_t] = nva.float()
        rp_b[act_t] = npb.float(); rv_b[act_t] = nvb.float()

        for i, g in enumerate(act):
            r = gcpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                if mga: mga.apply_move(g, int(na[i]), p0[g], p1[g])
                if mgb: mgb.apply_move(g, int(na[i]), p0[g], p1[g])

    del kva, kvb; torch.cuda.empty_cache()
    wa = wb = dr = 0
    for g in range(G):
        w = res[g]
        if w == 1:
            if ab[g]: wa += 1
            else: wb += 1
        elif w == 2:
            if ab[g]: wb += 1
            else: wa += 1
        else: dr += 1
    return wa, wb, dr


def main():
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                      n_shared=4, n_policy=4, n_value=4)
    raw = GomokuTransformer(cfg).to(DEVICE).eval()
    raw.load_state_dict(torch.load(os.path.join(CKPT_DIR, f'step_{STEP:06d}.pt'), map_location=DEVICE))
    mdl = ValueOnlyModel(raw)  # uniform policy + model value

    pairs = [
        ("S=64   vs S=32",   64, 32),
        ("S=64   vs S=16",   64, 16),
        ("S=64   vs Uniform", 64, 0),
        ("S=32   vs S=16",   32, 16),
        ("S=32   vs Uniform", 32, 0),
        ("S=16   vs Uniform", 16, 0),
    ]

    print(f"=== Value-only MCTS Depth Comparison (step {STEP}) ===")
    print(f"==  Uniform prior + model value + temp=5 (then argmax)  ==\n")

    results = []
    for label, s_a, s_b in pairs:
        t0 = time.perf_counter()
        wa, wb, d = run_match(mdl, s_a, s_b, temp_moves=5)
        dt = time.perf_counter() - t0
        wr = wa / max(wa + wb, 1) * 100
        results.append((label, wa, wb, d, wr, dt))
        print(f"  {label}: {wa}-{wb} D={d} WR={wr:.1f}% ({dt:.0f}s)")

    print(f"\n{'='*55}")
    for label, wa, wb, d, wr, dt in results:
        print(f"  {label:<25s} {wa}-{wb:<4d} D={d} {wr:<7.1f}%")
    print(f"\n(Larger S means more search. If value head works, larger S = higher WR.)")

    del raw; torch.cuda.empty_cache()
    print("\nDone.")

if __name__ == '__main__':
    main()
