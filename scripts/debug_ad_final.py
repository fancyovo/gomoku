"""Replay viz game, detect sleep-4, trace MCTS with score breakdown."""
import sys, os, numpy as np, torch, math, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp
DEVICE = torch.device('cuda')
N_CELLS = 225; C_PUCT = 0.2

def pos(r,c): return r*15+c

def find_sleep4(p0, p1):
    """Return (color, r, c_start, open_end) for any sleep-4 found."""
    for color, stones in [(0,p0),(1,p1)]:
        for r in range(15):
            for cc in range(12):
                if all(stones[r*15+cc+j] for j in range(4)):
                    left = cc - 1; right = cc + 4
                    left_free = left >= 0 and not (p0[r*15+left] or p1[r*15+left])
                    right_free = right < 15 and not (p0[r*15+right] or p1[r*15+right])
                    if left_free ^ right_free:
                        open_end = r*15+left if left_free else r*15+right
                        return color, r, cc, open_end
    return None

def trace_mcts(model, kv, br, hp, hp2, cur, correct, label, S_vals=[16,32,64,128,256,512,1024,2048,4096]):
    """Trace MCTS at position: target vs top-3 scores."""
    pos_t = torch.tensor([hp], dtype=torch.long, device=DEVICE)
    plr_t = torch.tensor([hp2], dtype=torch.long, device=DEVICE)
    kv2, br2 = model.create_cache(1, 250)
    rpol, rval = model.prefill(pos_t, plr_t, kv2, br2, [0])
    raw = torch.softmax(rpol.float(), dim=-1).cpu().squeeze(0).numpy()

    p0b = np.zeros(N_CELLS, bool); p1b = np.zeros(N_CELLS, bool)
    for i in range(len(hp)):
        if hp2[i] == 0: p0b[hp[i]] = True
        else: p1b[hp[i]] = True
    occ = p0b | p1b

    leg = raw.copy(); leg[occ] = 0; leg /= leg.sum()
    print(f"  Raw P(correct={correct})={leg[correct] if correct<225 else 'N/A':.6f}")
    top5 = np.argsort(-leg)[:5]
    print(f"  Top-5 raw P: "+", ".join(f"{a}({leg[a]:.4f})" for a in top5))

    print(f"  {'S':>5s} {'Ntot':>5s} | target N Q scr | top3_by_score")
    print(f"  {'-'*55}")

    for S in S_vals:
        mgr = gomoku_cpp.MCTSManager(1, 42)
        mgr.c_puct = C_PUCT; mgr.dirichlet_eps = 0.0; mgr.leaves_per_game = 1
        mgr.init_roots(p0b.reshape(1,-1), p1b.reshape(1,-1), np.array([cur], np.int32))
        lp = raw.copy(); lp[occ] = 0
        mgr.expand_roots(np.array([0], np.int32), lp.reshape(1,-1).astype(np.float32),
                          rval.cpu().numpy().astype(np.float32))
        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0: continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0: continue
            pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
            pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
            lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
            sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
            lp2, lv2 = model.evaluate_mcts_leaves(pt, pl2, kv2, br2, sl, lt)
            torch.cuda.synchronize()
            ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
            lp2 = lp2.masked_fill(ot, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp2,-1).cpu().numpy().astype(np.float32),
                                  lv2.cpu().numpy().astype(np.float32))
        ed = mgr.debug_root_edges(0)
        acts = np.array(ed['actions']); Ns = np.array(ed['N']); Ws = np.array(ed['W']); Ps = np.array(ed['P'])
        ntot = int(ed['N_total']); sqrtN = math.sqrt(max(ntot, 1))

        def qf(w,n): return w/n if n>0 else 0.0
        def sc_fn(j): return qf(float(Ws[j]),int(Ns[j])) + C_PUCT*float(Ps[j])*sqrtN/(1+int(Ns[j]))

        # Target
        mask = (acts == correct) if correct < 225 else np.zeros(len(acts), bool)
        if mask.any():
            i = np.where(mask)[0][0]; Nt = int(Ns[i]); Wt = float(Ws[i]); Pt = float(Ps[i])
            Qt = qf(Wt,Nt); Ut = C_PUCT*Pt*sqrtN/(1+Nt); scr_t = Qt+Ut
        else:
            Nt=0; Qt=0; Ut=0; scr_t=0

        # Top-3 by score
        all_s = [(sc_fn(j), j) for j in range(len(acts))]; all_s.sort(reverse=True)
        top3_s = " ".join(f"{int(acts[j])}({s:.3f})" for s,j in all_s[:3])

        print(f"  {S:5d} {ntot:5d} | target({correct}) N={Nt:3d} Q={Qt:.3f} scr={scr_t:.3f} | {top3_s}")
        del mgr; torch.cuda.empty_cache()
    del kv2, br2; torch.cuda.empty_cache()

def main():
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15, n_shared=4, n_policy=4, n_value=4)
    m = GomokuTransformer(cfg).to(DEVICE).eval()
    ckpt_dir = 'checkpoints/run500_fix_v5_0601_1046_25521'
    m.load_state_dict(torch.load(sorted(glob.glob(f'{ckpt_dir}/step_*.pt'))[-1], map_location=DEVICE))

    # Game from viz output
    RC = [(9,7),(10,6),(7,9),(7,7),(7,8),(4,6),(7,6),(8,7),(8,6),(8,8),(6,7),(5,6),
          (8,9),(5,7),(6,8),(9,8),(6,6),(6,9),(6,5),(14,7),(5,8),(7,5),(8,5),(2,14),
          (9,5),(5,5),(9,4)]
    moves = [pos(r,c) for r,c in RC]
    plrs = [i%2 for i in range(len(moves))]

    # Replay find sleep-4
    p0 = np.zeros(N_CELLS, bool); p1 = np.zeros(N_CELLS, bool)
    sleep4_step = None; sleep4_info = None
    for i in range(len(moves)):
        a = moves[i]; pl = plrs[i]
        if pl == 0: p0[a] = True
        else: p1[a] = True
        s4 = find_sleep4(p0, p1)
        if s4 is not None:
            sleep4_step = i+1
            sleep4_info = (s4, p0.copy(), p1.copy(), moves[:i+1], plrs[:i+1])
            color, r, cc, open_end = s4
            print(f"Sleep-4 at step {i+1}: {'Black' if color==0 else 'White'} row {r} cols {cc}-{cc+3}, open={open_end}")
            break

    if sleep4_info is None:
        print("No sleep-4 found in this game.")
        return

    (s4, p0_s4, p1_s4, hist_moves, hist_plrs) = sleep4_info
    color, r, cc, open_end = s4
    next_player = len(hist_moves) % 2

    # PART 1: Trace MCTS for defending player (should block at open_end)
    print(f"\n{'='*70}")
    print(f"PART 1: Defending player's turn at sleep-4 position")
    print(f"  Correct block: {open_end}")
    trace_mcts(m, None, None, hist_moves, hist_plrs, next_player, open_end,
               f"Defense: {'Black' if next_player==0 else 'White'} must block at {open_end}")

    # PART 2: Make the defender play a wrong move (the actual game's move)
    # Find what was actually played next
    if len(moves) > len(hist_moves):
        wrong_move = moves[len(hist_moves)]
        print(f"\n{'='*70}")
        print(f"PART 2: Defender played wrong move {wrong_move} instead of blocking")
        # New state after wrong move
        p0_wrong = p0_s4.copy(); p1_wrong = p1_s4.copy()
        if next_player == 0: p0_wrong[wrong_move] = True
        else: p1_wrong[wrong_move] = True
        new_player = 1 - next_player
        new_hist = hist_moves + [wrong_move]
        new_plr = hist_plrs + [next_player]

        # Sleep-4 owner now has winning move
        # The correct winning move is at the open end (completes 5)
        winning = open_end  # same position, now it's an attack

        trace_mcts(m, None, None, new_hist, new_plr, new_player, winning,
                   f"Attack: {'Black' if new_player==0 else 'White'} can win at {winning}")

    print("\n=== Done ===")

if __name__ == '__main__':
    main()
