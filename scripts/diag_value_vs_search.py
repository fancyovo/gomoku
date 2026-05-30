#!/usr/bin/env python3
"""Diagnose MCTS value usage: one game, compare S=64 vs S=16.
At each step runs BOTH searches on independent game states.
Prints: raw policy top5, S=64 visit top5, S=16 visit top5, root value."""
import sys, os, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp as gcpp

DEVICE = torch.device('cuda')
N_CELLS = 225
CKPT_DIR = 'checkpoints/run200_G2048_pool2x_0530_1019'
STEP = 3
COLS = 'ABCDEFGHIJKLMNO'
def pos_str(p):
    return f"{COLS[p % 15]}{p // 15 + 1}"
def top5(arr, occ):
    a = arr.copy(); a[occ] = 0
    s = a.sum()
    if s == 0: return []
    a /= s
    return [(pos_str(i), a[i]) for i in np.argsort(a)[::-1][:5] if a[i] > 0]

def run_once(S_play, S_cmp, label):
    raw = GomokuTransformer(ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                                         n_shared=4, n_policy=4, n_value=4)).to(DEVICE).eval()
    raw.load_state_dict(torch.load(f'{CKPT_DIR}/step_{STEP:06d}.pt', map_location=DEVICE))

    def new_state():
        pool = gcpp.GamePool(1); pool.reset_all()
        mgr = gcpp.MCTSManager(1, np.random.randint(0, 2**31))
        mgr.c_puct = 1.0; mgr.leaves_per_game = 4; mgr.dirichlet_eps = 0.0
        mgr.init_roots(np.zeros((1,N_CELLS),bool), np.zeros((1,N_CELLS),bool), np.zeros(1,np.int32))
        kv, br = raw.create_cache(1, 250)
        return pool, mgr, kv, br, np.zeros(N_CELLS,bool), np.zeros((1,N_CELLS),bool), np.zeros((1,N_CELLS),bool)

    def replay_upto(moves, my_state):
        pool, mgr, kv, br, occ, p0, p1 = my_state
        for i, (pos, pl) in enumerate(moves):
            if i == 0:
                fat = torch.tensor([[pos]], dtype=torch.long, device=DEVICE)
                pl0 = torch.zeros(1,1,dtype=torch.long,device=DEVICE)
                rp, rv = raw.prefill(fat, pl0, kv, br, [0])
                rpb, rvb = rp.float().clone(), rv.float().clone()
            else:
                dp = torch.tensor([pos], dtype=torch.long, device=DEVICE)
                dpl = torch.tensor([pl], dtype=torch.long, device=DEVICE)
                ds = torch.tensor([0], device=DEVICE)
                np_, nv_ = raw.decode(dp, dpl, kv, br, ds)
                rpb[0] = np_.float(); rvb[0] = nv_.float()
            occ[pos] = True
            if pl == 0: p0[0, pos] = True
            else: p1[0, pos] = True
            gcpp.step(pool, 0, pos)
            mgr.apply_move(0, pos, p0[0], p1[0])
        return rpb, rvb

    def search(mgr, kv, br, rpb, rvb, occ, S):
        lp = rpb[0].cpu().numpy().copy(); lp[occ] = -1e9
        lv = rvb[0].cpu().numpy()
        mgr.expand_roots(np.array([0],np.int32),
            torch.softmax(torch.from_numpy(lp),-1).numpy().astype(np.float32),
            lv.astype(np.float32))
        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0: continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0: continue
            pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
            pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
            lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
            sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
            lp2, lv2 = raw.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
            ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
            lp2 = lp2.masked_fill(ot, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32),
                torch.softmax(lp2,-1).cpu().numpy().astype(np.float32),
                lv2.cpu().numpy().astype(np.float32))
        return mgr.get_root_policies()[0].copy()

    st_a = new_state()
    st_b = new_state()

    ft = raw.sample_first_moves(1, DEVICE)
    fp = int(ft[0].item())
    move_seq = [(fp, 0)]

    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    print(f"  Move 1: Black {pos_str(fp)}  (S_play={S_play}, S_cmp={S_cmp})")

    for _ in range(1, 225):
        # Replay to current position for both states
        rpb_a, rvb_a = replay_upto(move_seq, st_a)
        rpb_b, rvb_b = replay_upto(move_seq, st_b)

        cp = len(move_seq) % 2
        plr = 'Black' if cp == 0 else 'White'

        visits_play = search(st_a[1], st_a[2], st_a[3], rpb_a, rvb_a, st_a[4], S_play)
        visits_cmp = search(st_b[1], st_b[2], st_b[3], rpb_b, rvb_b, st_b[4], S_cmp)

        # Action via S_play
        pol = visits_play.copy(); pol[st_a[4]] = 0
        ps = pol.sum()
        action = int(np.random.choice(N_CELLS, p=pol/ps)) if ps > 0 else int(np.random.choice(np.where(~st_a[4])[0]))

        raw_pol = rpb_a[0].cpu().numpy().copy()
        print(f"  Move {len(move_seq)+1}: {plr} {pos_str(action)}")
        print(f"    Root value: {rvb_a[0].item():+.4f}")
        print(f"    Raw policy:   [{', '.join(f'{p}({v:.3f})' for p,v in top5(raw_pol, st_a[4]))}]")
        print(f"    S={S_play} visits: [{', '.join(f'{p}({v:.3f})' for p,v in top5(visits_play, st_a[4]))}]")
        print(f"    S={S_cmp} visits:  [{', '.join(f'{p}({v:.3f})' for p,v in top5(visits_cmp, st_b[4]))}]")

        move_seq.append((action, cp))
        r = gcpp.step(st_a[0], 0, action)
        if r:
            print(f"  *** {['Black','White','Draw'][min(r,3)-1]} wins ***")
            break

    del raw; torch.cuda.empty_cache()

if __name__ == '__main__':
    run_once(64, 16, "Full model: S=64 plays, S=16 compare")
    run_once(16, 64, "Full model: S=16 plays, S=64 compare")
