#!/usr/bin/env python3
"""Head-to-head match with MCTS top-5 trace + GIF with overlay.

Usage: python scripts/viz_match.py <ckpt_dir> <step_a> <step_b>
"""
import sys, os, time, numpy as np, argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda")
N_CELLS = 225; BS = 15


def pos_to_str(p):
    return f"{chr(ord('A') + p % BS)}{p // BS + 1}"


def load_model(path):
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    m = GomokuTransformer(cfg).to(DEVICE).eval()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m


def draw_board(ax, p0, p1, last_move, move_num, top5, next_player):
    """Board + top-5 MCTS recommendations with probability labels."""
    ax.clear()
    ax.set_xlim(-0.5, BS - 0.5); ax.set_ylim(-0.5, BS - 0.5)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    for i in range(BS):
        ax.axhline(i, color="black", linewidth=0.3, alpha=0.3)
        ax.axvline(i, color="black", linewidth=0.3, alpha=0.3)
    for r, c in [(3,3),(3,7),(3,11),(7,3),(7,7),(7,11),(11,3),(11,7),(11,11)]:
        ax.add_patch(Circle((c, r), 0.12, color="black", zorder=2))
    for pos in range(N_CELLS):
        r, c = pos // BS, pos % BS
        if p0[pos]:
            ax.add_patch(Circle((c, r), 0.42, color="black", zorder=3, ec="none"))
        elif p1[pos]:
            ax.add_patch(Circle((c, r), 0.42, color="white", zorder=3, ec="black", linewidth=0.5))
    if last_move is not None:
        r, c = last_move // BS, last_move % BS
        ax.add_patch(Circle((c, r), 0.12, color="red", zorder=4))
    # Top-5 overlay
    if top5:
        max_p = max(p for _, p in top5)
        for pos, prob in top5:
            if p0[pos] or p1[pos]: continue
            r, c = pos // BS, pos % BS
            alpha = 0.15 + 0.7 * (prob / max(max_p, 1e-6))
            ax.add_patch(Circle((c, r), 0.43, color="blue", zorder=5,
                                alpha=alpha, ec="blue", linewidth=1.2, fill=False))
            ax.text(c, r - 0.6, f"{prob*100:.1f}%", fontsize=6, ha="center",
                    color="darkblue", zorder=6)
    pstr = "Black" if next_player == 0 else "White"
    ax.set_title(f"Move {move_num} — {pstr}", fontsize=11)


@torch.inference_mode()
def play_match(ma, mb, G, S_elo=256):
    """Play G games between ma and mb with S_elo MCTS sims.
    Returns match results + per-game traces."""
    G_ = G; M_ = 4
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    def mk_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = M_; m.dirichlet_eps = 0.0
        m.init_roots(np.zeros((G_, 225), dtype=bool),
                     np.zeros((G_, 225), dtype=bool),
                     np.zeros(G_, dtype=np.int32))
        return m

    mga = mk_mgr(); mgb = mk_mgr()
    kva, bra = ma.create_cache(max_games=G_)
    kvb, brb = mb.create_cache(max_games=G_)
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
    ma.prefill(fat, pl0, kva, bra, list(range(G_)))
    mb.prefill(fat, pl0, kvb, brb, list(range(G_)))
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    # Per-game traces
    moves = [[] for _ in range(G_)]      # positions
    players = [[] for _ in range(G_)]    # who played
    which_model = [[] for _ in range(G_)]  # 0=ma, 1=mb
    mcts_all = [[] for _ in range(G_)]   # MCTS top-5 BEFORE each move

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(fa[g]))
        if r: fin[g] = True; res[g] = r
        else:
            bp = np.zeros(225, dtype=bool); bp[int(fa[g])] = True
            mga.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))
            mgb.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))
        moves[g].append(int(fa[g])); players[g].append(0)
        which_model[g].append(0 if ab[g] else 1)

    for move_idx in range(1, 240):
        act = np.where(~fin)[0]
        if len(act) == 0: break
        cp = move_idx % 2

        # Each MCTS manager runs S_elo simulations independently
        for mgr, mdl, kv, br in [(mga, ma, kva, bra), (mgb, mb, kvb, brb)]:
            st = torch.from_numpy(act).to(DEVICE)
            dp = torch.zeros(len(act), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(act), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(og[act], -1e9)
            mgr.expand_roots(act.astype(np.int32),
                             torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                             lv.cpu().numpy().astype(np.float32))
            for _ in range(S_elo):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                lp2, lv2 = mdl.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                ot2 = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot2, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32),
                                      torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                      lv2.cpu().numpy().astype(np.float32))

        rpa = mga.get_root_policies(); rpb = mgb.get_root_policies()
        na = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            ua = ((cp == 0) == ab[g])
            pol = (rpa[g] if ua else rpb[g]).copy()
            # Save MCTS top-5 before masking occupied
            top5 = sorted([(a, float(pol[a])) for a in range(225) if pol[a] > 0],
                          key=lambda x: -x[1])[:5]
            mcts_all[g].append(top5)
            pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True; og[g, a] = True
            moves[g].append(a); players[g].append(cp)
            which_model[g].append(0 if ua else 1)

        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        ma.decode(dp_, dpl_, kva, bra, torch.from_numpy(act).to(DEVICE))
        mb.decode(dp_, dpl_, kvb, brb, torch.from_numpy(act).to(DEVICE))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                mga.apply_move(g, int(na[i]), p0[g], p1[g])
                mgb.apply_move(g, int(na[i]), p0[g], p1[g])

    del kva, kvb, bra, brb; torch.cuda.empty_cache()

    wa = wb = dr = 0
    for g in range(G_):
        w = res[g]
        if w == 1: wa += 1 if ab[g] else 0; wb += 0 if ab[g] else 1
        elif w == 2: wa += 0 if ab[g] else 1; wb += 1 if ab[g] else 0
        else: dr += 1
    return wa, wb, dr, moves, players, which_model, mcts_all, res, p0, p1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt_dir')
    p.add_argument('step_a', type=int)
    p.add_argument('step_b', type=int)
    p.add_argument('--games', type=int, default=4)
    p.add_argument('--S', type=int, default=256)
    args = p.parse_args()

    na = f"step_{args.step_a:06d}.pt"; nb = f"step_{args.step_b:06d}.pt"
    print(f"A({na}) vs B({nb})  G={args.games} S={args.S}")
    ma = load_model(os.path.join(args.ckpt_dir, na))
    mb = load_model(os.path.join(args.ckpt_dir, nb))

    t0 = time.perf_counter()
    wa, wb, dr, moves, players, which, mcts_all, res, p0_arr, p1_arr = \
        play_match(ma, mb, args.games, args.S)
    dt = time.perf_counter() - t0
    wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50
    print(f"Result: A={wa} B={wb} draws={dr}  WR_A={wr:.1f}%  ({dt:.0f}s)\n")

    rn = {1: "B_WIN", 2: "W_WIN", 3: "DRAW"}

    for g in range(args.games):
        L = len(moves[g]); rv = res[g] if res[g] != 0 else 3
        occ = p0_arr[g] | p1_arr[g]
        print(f"{'=' * 60}")
        print(f"Game {g+1}: {L} moves, {rn.get(rv, rv)}")
        print(f"{'=' * 60}")

        for i in range(L):
            a = moves[g][i]; plr = players[g][i]; wm = which[g][i]
            label = "A" if wm == 0 else "B"
            ps = "B" if plr == 0 else "W"
            pos = pos_to_str(a)
            print(f"\n  Move {i+1}: {ps} {pos} [{label}]")

            # MCTS top-5 that was computed BEFORE this move was chosen
            mcts = mcts_all[g][i] if i < len(mcts_all[g]) else []
            if mcts:
                print(f"    MCTS top-5:   ", end="")
                for idx, prob in mcts:
                    occ_str = "(occ)" if occ[idx] else ""
                    print(f"{pos_to_str(idx)}={prob*100:.1f}%{occ_str} ", end="")
                print()

        # GIF for first game
        if g == 0:
            out_gif = f"{args.ckpt_dir}/match_{args.step_a}_vs_{args.step_b}.gif"
            p0_g = np.zeros(N_CELLS, dtype=bool); p1_g = np.zeros(N_CELLS, dtype=bool)
            frames = []; fig, ax = plt.subplots(figsize=(7, 7))
            top5_0 = mcts_all[g][0] if len(mcts_all[g]) > 0 else []
            draw_board(ax, p0_g, p1_g, None, 0, top5_0, players[g][0])
            fig.tight_layout(); fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()))

            for i, a in enumerate(moves[g]):
                if players[g][i] == 0: p0_g[a] = True
                else: p1_g[a] = True
                next_p = players[g][i+1] if i+1 < len(players[g]) else (1 - players[g][i])
                top5 = mcts_all[g][i+1] if i+1 < len(mcts_all[g]) else []
                draw_board(ax, p0_g, p1_g, a, i+1, top5, next_p)
                fig.tight_layout(); fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()))
            plt.close()
            frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                           duration=1000, loop=0)
            print(f"\nGIF: {out_gif} ({len(frames)} frames)")

    del ma, mb; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
