#!/usr/bin/env python3
"""Compare two checkpoints: self-play traces + head-to-head MCTS match."""
import sys, os, time, numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda")
N_CELLS = 225
G_SELF = 3
G_MATCH = 256


def pos_to_str(p):
    return f"{chr(ord('A') + p % 15)}{p // 15 + 1}"


def load_model(path):
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    m = GomokuTransformer(cfg).to(DEVICE).eval()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m


def run_selfplay(model, G, M=8, S=64, label=""):
    pool = gomoku_cpp.GamePool(G); pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G, seed_base=np.random.randint(0, 2**31))
    mgr.c_puct = 1.0; mgr.dirichlet_eps = 0.0
    mgr.dirichlet_alpha = 0.03; mgr.leaves_per_game = M
    p0 = np.zeros((G, 225), dtype=bool); p1 = np.zeros((G, 225), dtype=bool)
    mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
    kv = model.create_cache(max_games=G, max_cache_len=250)

    fa = model.sample_first_moves(G, DEVICE)
    model.prefill(fa.unsqueeze(1), torch.zeros(G, 1, dtype=torch.long, device=DEVICE), kv, list(range(G)))

    ph = [[] for _ in range(G)]; plh = [[] for _ in range(G)]
    p0_c = np.zeros((G, 225), dtype=bool); p1_c = np.zeros((G, 225), dtype=bool)
    plen = np.zeros(G, dtype=np.int32); fin = np.zeros(G, dtype=bool); res = np.zeros(G, dtype=np.int32)

    for g in range(G):
        a = int(fa[g].item()); ph[g].append(a); plh[g].append(0); p0_c[g, a] = True; plen[g] = 1
        r = gomoku_cpp.step(pool, g, a)
        if r: fin[g] = True; res[g] = r; mgr.reset_game(g)
        else: mgr.apply_move(g, a, p0_c[g], p1_c[g])

    occ_g = torch.from_numpy(p0_c | p1_c).to(DEVICE)

    while True:
        act = np.where(~fin)[0]
        if len(act) == 0: break
        st = torch.from_numpy(act).to(DEVICE); cp = int(plen[act[0]]) % 2
        dp = torch.zeros(len(act), 1, dtype=torch.long, device=DEVICE)
        dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=DEVICE)
        dl = torch.ones(len(act), dtype=torch.long, device=DEVICE)
        lp, lv = model.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
        lp = lp.masked_fill(occ_g[act], -1e9)
        mgr.expand_roots(act.astype(np.int32),
                         torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                         lv.cpu().numpy().astype(np.float32))
        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0: continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0: continue
            pt_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
            pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
            lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
            sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
            lp2, lv2 = model.evaluate_mcts_leaves(pt_t, pl2, kv, sl, lt)
            ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
            lp2 = lp2.masked_fill(ot, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32),
                                  torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                  lv2.cpu().numpy().astype(np.float32))
        rp = mgr.get_root_policies()
        na = np.zeros(len(act), dtype=np.int64); np_ = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            pol = rp[g].copy(); pol[p0_c[g] | p1_c[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0_c[g] | p1_c[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a; np_[i] = plen[g] % 2
            ph[g].append(a); plh[g].append(np_[i])
            if np_[i] == 0: p0_c[g, a] = True
            else: p1_c[g, a] = True
            plen[g] += 1
        dec_p = torch.from_numpy(na).to(DEVICE); dec_pl = torch.from_numpy(np_).to(DEVICE)
        model.decode(dec_p, dec_pl, kv, torch.from_numpy(act).to(DEVICE))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r; mgr.reset_game(g)
            else: mgr.apply_move(g, int(na[i]), p0_c[g], p1_c[g])

    del kv; torch.cuda.empty_cache()
    rn = {1: "B_WIN", 2: "W_WIN", 3: "DRAW"}
    for g in range(G):
        L = plen[g]; rv = res[g] if res[g] != 0 else 3
        print(f"\n  [{label}] Game {g+1}: {L} moves, {rn.get(rv, rv)}")
        moves = [pos_to_str(a) for a in ph[g]]
        for i in range(0, len(moves), 15):
            print(f"    {i+1:3d}: " + " ".join(moves[i:i+15]))
    return ph, plh, plen


@torch.inference_mode()
def play_match(ma, mb, G_=256, M_=4, S_=16):
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    def mm():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = M_; m.dirichlet_eps = 0.0
        m.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
        return m
    mga = mm(); mgb = mm()
    kva = ma.create_cache(max_games=G_); kvb = mb.create_cache(max_games=G_)
    ab = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    fin = np.zeros(G_, dtype=bool); res = np.zeros(G_, dtype=np.int32)
    p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)

    fa_a = ma.sample_first_moves(G_, DEVICE); fa_b = mb.sample_first_moves(G_, DEVICE)
    fa = np.zeros(G_, dtype=np.int64)
    for g in range(G_):
        fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item()); p0[g, fa[g]] = True
    fat = torch.tensor(fa, dtype=torch.long, device=DEVICE).unsqueeze(1)
    pl0 = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    ma.prefill(fat, pl0, kva, list(range(G_))); mb.prefill(fat, pl0, kvb, list(range(G_)))
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(fa[g]))
        if r: fin[g] = True; res[g] = r
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
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True; og[g, a] = True
        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        ma.decode(dp_, dpl_, kva, torch.from_numpy(act).to(DEVICE))
        mb.decode(dp_, dpl_, kvb, torch.from_numpy(act).to(DEVICE))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                mga.apply_move(g, int(na[i]), p0[g], p1[g])
                mgb.apply_move(g, int(na[i]), p0[g], p1[g])

    del kva, kvb; torch.cuda.empty_cache()
    wa = wb = dr = 0
    for g in range(G_):
        w = res[g]; is_a = ab[g]
        if w == 1: wa += 1 if is_a else 0; wb += 0 if is_a else 1
        elif w == 2: wa += 0 if is_a else 1; wb += 1 if is_a else 0
        else: dr += 1
    return wa, wb, dr


def main():
    ckpt_dir = sys.argv[1]
    idx_a = int(sys.argv[2]) if len(sys.argv) > 2 else None
    idx_b = int(sys.argv[3]) if len(sys.argv) > 3 else None

    ckpts = sorted([int(f.split('_')[1].split('.')[0])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith('step_') and f.endswith('.pt')])
    a_idx = idx_a if idx_a is not None else ckpts[-1]
    b_idx = idx_b if idx_b is not None else ckpts[-2]

    name_a = f"step_{a_idx:06d}.pt"
    name_b = f"step_{b_idx:06d}.pt"
    print(f"Model A: {name_a}  Model B: {name_b}")

    m_a = load_model(os.path.join(ckpt_dir, name_a))
    m_b = load_model(os.path.join(ckpt_dir, name_b))

    print(f"\n{'='*60}")
    print(f"Self-play A: {name_a} ({G_SELF} games, M=8 S=64)")
    print(f"{'='*60}")
    run_selfplay(m_a, G_SELF, label=f"A_{name_a}")

    print(f"\n{'='*60}")
    print(f"Self-play B: {name_b} ({G_SELF} games, M=8 S=64)")
    print(f"{'='*60}")
    run_selfplay(m_b, G_SELF, label=f"B_{name_b}")

    print(f"\n{'='*60}")
    print(f"Match: A({name_a}) vs B({name_b}) (S=16, {G_MATCH} games)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    wa, wb, d = play_match(m_a, m_b, G_MATCH)
    dt = time.perf_counter() - t0
    wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
    print(f"  A({name_a}): {wa}  B({name_b}): {wb}  draws: {d}")
    print(f"  A WR = {wr:.1f}%  ({dt:.0f}s)")

    del m_a, m_b; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
