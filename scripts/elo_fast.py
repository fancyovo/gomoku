#!/usr/bin/env python3
"""Fast ELO tournament on existing checkpoints.
Uses temp=10 (sample from MCTS visits), then argmax.
S=16, G=256, M=4, noisy_uniform baseline."""
import sys, os, glob, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp as gcpp

DEVICE = torch.device('cuda')
N_CELLS = 225
CKPT_DIR = 'checkpoints/run200_G2048_pool2x_0530_1019'
TEMP_MOVES = 10


def make_mgr(G, seed=0):
    m = gcpp.MCTSManager(G, seed_base=seed + np.random.randint(0, 2**20))
    m.c_puct = 1.0; m.leaves_per_game = 4; m.dirichlet_eps = 0.0
    m.init_roots(np.zeros((G, N_CELLS), bool), np.zeros((G, N_CELLS), bool), np.zeros(G, np.int32))
    return m


def run_match(mdl_a, mdl_b, G=256, S=16):
    pool = gcpp.GamePool(G); pool.reset_all()
    mga = make_mgr(G); mgb = make_mgr(G)
    kva, bka = mdl_a.create_cache(G, 250)
    kvb, bkb = mdl_b.create_cache(G, 250)
    ab = np.array([i % 2 == 0 for i in range(G)], bool)
    fin = np.zeros(G, bool); res = np.zeros(G, np.int32)
    p0 = np.zeros((G, N_CELLS), bool); p1 = np.zeros((G, N_CELLS), bool)

    fa_a = mdl_a.sample_first_moves(G, DEVICE)
    fa_b = mdl_b.sample_first_moves(G, DEVICE)
    fa = np.zeros(G, np.int64)
    for g in range(G):
        fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
        p0[g, fa[g]] = True
    fat = torch.tensor(fa, dtype=torch.long, device=DEVICE).unsqueeze(1)
    pl0 = torch.zeros(G, 1, dtype=torch.long, device=DEVICE)
    pa0, va0 = mdl_a.prefill(fat, pl0, kva, bka, list(range(G)))
    pb0, vb0 = mdl_b.prefill(fat, pl0, kvb, bkb, list(range(G)))
    rpa = pa0.float().clone(); rva = va0.float().clone()
    rpb = pb0.float().clone(); rvb = vb0.float().clone()
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G):
        r = gcpp.step(pool, g, int(fa[g]))
        if r: fin[g] = True; res[g] = r
        else:
            bp = np.zeros(N_CELLS, bool); bp[int(fa[g])] = True
            mga.apply_move(g, int(fa[g]), bp, np.zeros(N_CELLS, bool))
            mgb.apply_move(g, int(fa[g]), bp, np.zeros(N_CELLS, bool))

    for move in range(1, 240):
        act = np.where(~fin)[0]
        if len(act) == 0: break
        cp = move % 2
        anp = act.astype(np.int32); at = torch.from_numpy(act).to(DEVICE)

        for mgr, polb, valb, kv, bk, S_ in [(mga, rpa, rva, kva, bka, S), (mgb, rpb, rvb, kvb, bkb, S)]:
            lp = polb[act].masked_fill(og[act], -1e9)
            lv = valb[act]
            mgr.expand_roots(anp, torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
            for _ in range(S_):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                mdl = mdl_a if mgr is mga else mdl_b
                lp2, lv2 = mdl.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp2, -1).cpu().numpy().astype(np.float32), lv2.cpu().numpy().astype(np.float32))

        rp_ar = mga.get_root_policies(); rp_br = mgb.get_root_policies()
        na = np.zeros(len(act), np.int64)
        for i, g in enumerate(act):
            ua = ((cp == 0) == ab[g])
            pol = (rp_ar[g] if ua else rp_br[g]).copy(); pol[p0[g] | p1[g]] = 0
            ps = pol.sum()
            if ps > 0:
                if move > TEMP_MOVES:
                    na[i] = int(np.argmax(pol))
                else:
                    na[i] = int(np.random.choice(N_CELLS, p=pol / ps))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                na[i] = int(np.random.choice(leg)) if len(leg) > 0 else 0
            if cp == 0: p0[g, na[i]] = True
            else: p1[g, na[i]] = True
            og[g, na[i]] = True

        ds = torch.from_numpy(act).to(DEVICE)
        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        npa, nva = mdl_a.decode(dp_, dpl_, kva, bka, ds)
        npb, nvb = mdl_b.decode(dp_, dpl_, kvb, bkb, ds)
        rpa[at] = npa.float(); rva[at] = nva.float()
        rpb[at] = npb.float(); rvb[at] = nvb.float()

        for i, g in enumerate(act):
            r = gcpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                mga.apply_move(g, int(na[i]), p0[g], p1[g])
                mgb.apply_move(g, int(na[i]), p0[g], p1[g])

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

    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'step_*.pt')))
    names = []
    for c in ckpts:
        s = int(os.path.basename(c).split('_')[1].split('.')[0])
        if s % 5 == 0:
            names.append(os.path.basename(c))
    names.sort(key=lambda n: int(n.split('_')[1].split('.')[0]))

    print(f"Checkpoints: {names}")
    print(f"Temp={TEMP_MOVES} (sample), then argmax. G=256, S=16, M=4\n")

    class Uniform:
        def __init__(self):
            self.config = cfg
        def create_cache(self, *a, **kw): return GomokuTransformer(cfg).to(DEVICE).create_cache(*a, **kw)
        def sample_first_moves(self, bs, dev): return torch.randint(0, N_CELLS, (bs,), device=dev)
        def prefill(self, pos, plr, kv, br, idx): return (torch.randn(pos.shape[0], N_CELLS, device=DEVICE)*0.02, torch.zeros(pos.shape[0], device=DEVICE))
        def decode(self, pos, plr, kv, br, idx):
            kv.advance(idx)
            return (torch.randn(len(idx), N_CELLS, device=DEVICE)*0.02, torch.zeros(len(idx), device=DEVICE))
        def evaluate_mcts_leaves(self, pos, plr, kv, idx, plen): return (torch.randn(pos.shape[0], N_CELLS, device=DEVICE)*0.02, torch.zeros(pos.shape[0], device=DEVICE))

    uni = Uniform()
    all_names = ['noisy_uniform'] + names
    cache = {}
    for i, na in enumerate(all_names):
        for nb in all_names[i+1:]:
            print(f"  {na} vs {nb} ...", end=' ', flush=True)
            t0 = time.perf_counter()
            if na == 'noisy_uniform': ma = uni
            else:
                m = GomokuTransformer(cfg).to(DEVICE).eval()
                m.load_state_dict(torch.load(os.path.join(CKPT_DIR, na), map_location=DEVICE))
                ma = m
            if nb == 'noisy_uniform': mb = uni
            else:
                m = GomokuTransformer(cfg).to(DEVICE).eval()
                m.load_state_dict(torch.load(os.path.join(CKPT_DIR, nb), map_location=DEVICE))
                mb = m
            wa, wb, d = run_match(ma, mb)
            dt = time.perf_counter() - t0
            wr = wa / max(wa + wb, 1) * 100
            cache[f"{na}|{nb}"] = (wa, wb, d)
            print(f"{wa}-{wb} D={d} WR={wr:.1f}% ({dt:.0f}s)")
            if na != 'noisy_uniform': del ma
            if nb != 'noisy_uniform': del mb
            torch.cuda.empty_cache()

    elo = {n: 1500.0 for n in all_names}
    entries = []
    for pair, (wa, wb, d) in cache.items():
        a, b = pair.split('|')
        entries.append((a, b, wa + d*0.5, wb + d*0.5))
    for _ in range(500):
        dmax = 0.0
        for a, b, sa, sb in entries:
            n = sa + sb
            if n == 0: continue
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            da = (sa - ea * n) * (32.0 / n)
            elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break

    print(f"\nELO Results (temp={TEMP_MOVES} then argmax):")
    for n in sorted(all_names, key=lambda x: -elo.get(x, 1500)):
        print(f"  {n:25s}: {elo.get(n, 1500):.0f}")

if __name__ == '__main__':
    main()
