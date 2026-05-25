#!/usr/bin/env python3
"""Generate self-play data using uniform dummy model + MCTS. CPU-only.
Fixed version: proper MCTS for first move, random leaf values, no placeholders."""

import sys, os, time, json, pickle, numpy as np, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gomoku_cpp

BOARD_SIZE = 15
N_CELLS = BOARD_SIZE * BOARD_SIZE


def pos_to_coord(p):
    return f"{chr(ord('A')+p%BOARD_SIZE)}{p//BOARD_SIZE+1}"


class DummyModel:
    """Uniform random model — all-zero logits (uniform policy), random value."""

    def __init__(self):
        self.n_positions = N_CELLS

    def sample_first_moves(self, batch_size, device=None):
        return np.random.randint(0, N_CELLS, size=batch_size).astype(np.int64)

    def get_policy_value(self, n_eval):
        """Uniform policy + random 2-class value prediction."""
        policy = np.zeros((n_eval, N_CELLS), dtype=np.float32)
        v_logits = np.random.randn(n_eval, 2).astype(np.float32) * 0.1
        v_probs = np.exp(v_logits - v_logits.max(axis=-1, keepdims=True))
        v_probs /= v_probs.sum(axis=-1, keepdims=True)
        value = v_probs[:, 0] - v_probs[:, 1]
        return policy, value

    def clear_cache(self):
        pass


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def generate_batch(batch_size, num_simulations, leaves_per_game,
                   c_puct, dirichlet_eps, dirichlet_alpha):
    """Generate one batch of self-play games with proper MCTS."""
    G, M, S = batch_size, leaves_per_game, num_simulations
    model = DummyModel()

    pool = gomoku_cpp.GamePool(G)
    pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G, seed_base=np.random.randint(0, 2 ** 31))
    mgr.c_puct = c_puct
    mgr.dirichlet_eps = dirichlet_eps
    mgr.dirichlet_alpha = dirichlet_alpha
    mgr.leaves_per_game = M

    p0 = np.zeros((G, N_CELLS), dtype=bool)
    p1 = np.zeros((G, N_CELLS), dtype=bool)
    mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))

    pos_hist = [[] for _ in range(G)]
    plr_hist = [[] for _ in range(G)]
    mcts_pols = [[] for _ in range(G)]
    pos_lens = np.zeros(G, dtype=np.int32)
    finished = np.zeros(G, dtype=bool)
    results = np.zeros(G, dtype=np.int32)
    occ = np.zeros((G, N_CELLS), dtype=bool)

    # ── First move: run MCTS on empty board ──
    active = np.arange(G, dtype=np.int32)
    lp, lv = model.get_policy_value(G)
    mgr.expand_roots(active, softmax(lp).astype(np.float32), lv.astype(np.float32))

    for _ in range(S):
        sel = mgr.select_all()
        if sel['max_path_len'] == 0:
            continue
        vi = np.where(sel['valid_mask'])[0]
        if len(vi) == 0:
            continue
        n_eval = len(vi)
        lp2, lv2 = model.get_policy_value(n_eval)
        occ_t = sel['occ_dense'][vi].astype(bool)
        for i in range(n_eval):
            lp2[i][occ_t[i]] = -1e9
        mgr.expand_and_backup(vi.astype(np.int32),
                              softmax(lp2).astype(np.float32),
                              lv2.astype(np.float32))

    rp = mgr.get_root_policies()
    for g in range(G):
        pol = rp[g].copy()
        s = pol.sum()
        if s > 0:
            pol /= s
            a = int(np.random.choice(N_CELLS, p=pol))
        else:
            legal = np.where(~occ[g])[0]
            a = int(np.random.choice(legal)) if len(legal) > 0 else 0

        pos_hist[g].append(a)
        plr_hist[g].append(0)
        mcts_pols[g].append(rp[g].copy())  # Store MCTS policy for first move
        p0[g, a] = True
        occ[g, a] = True
        pos_lens[g] = 1

        r = gomoku_cpp.step(pool, g, a)
        if r:
            finished[g] = True
            results[g] = r
            mgr.reset_game(g)
        else:
            mgr.apply_move(g, a, p0[g], p1[g])

    # ── Main loop ──
    while True:
        active = np.where(~finished)[0]
        n_active = len(active)
        if n_active == 0:
            break

        cp = int(pos_lens[active[0]]) % 2

        lp, lv = model.get_policy_value(n_active)
        for i, g in enumerate(active):
            lp[i][occ[g]] = -1e9

        mgr.expand_roots(active.astype(np.int32),
                         softmax(lp).astype(np.float32),
                         lv.astype(np.float32))

        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0:
                continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0:
                continue
            n_eval = len(vi)
            lp2, lv2 = model.get_policy_value(n_eval)
            occ_t = sel['occ_dense'][vi].astype(bool)
            for i in range(n_eval):
                lp2[i][occ_t[i]] = -1e9
            mgr.expand_and_backup(vi.astype(np.int32),
                                  softmax(lp2).astype(np.float32),
                                  lv2.astype(np.float32))

        rp = mgr.get_root_policies()
        for i, g in enumerate(active):
            pol = rp[g].copy()
            pol[occ[g]] = 0.0
            s = pol.sum()
            if s > 0:
                pol /= s
                a = int(np.random.choice(N_CELLS, p=pol))
            else:
                legal = np.where(~occ[g])[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0

            plr = pos_lens[g] % 2
            pos_hist[g].append(a)
            plr_hist[g].append(plr)
            mcts_pols[g].append(rp[g].copy())

            if plr == 0:
                p0[g, a] = True
            else:
                p1[g, a] = True
            occ[g, a] = True
            pos_lens[g] += 1

            r = gomoku_cpp.step(pool, g, a)
            if r:
                finished[g] = True
                results[g] = r
                mgr.reset_game(g)
            else:
                mgr.apply_move(g, a, p0[g], p1[g])

    # ── Build trajectories ──
    # mcts_pols has L entries (π_0..π_{L-1}). Drop π_0 so format matches
    # self-play data (L-1 entries, π_1..π_{L-1}), aligned with collate_fn[:L-1].
    trajectories = []
    for g in range(G):
        L = pos_lens[g]
        r_val = results[g]
        r_val = 3 if r_val == 0 else r_val

        vt = np.zeros((L, 2), dtype=np.float32)
        for i in range(L):
            plr = plr_hist[g][i]
            if r_val == 3:
                vt[i] = [0.5, 0.5]
            elif (r_val == 1 and plr == 0) or (r_val == 2 and plr == 1):
                vt[i] = [1.0, 0.0]
            else:
                vt[i] = [0.0, 1.0]

        pols = np.array(mcts_pols[g][1:], dtype=np.float32)  # drop π_0, keep π_1..π_{L-1}

        trajectories.append({
            'positions': np.array(pos_hist[g], dtype=np.int64),
            'players': np.array(plr_hist[g], dtype=np.int64),
            'actions': np.array(pos_hist[g], dtype=np.int64),
            'mcts_policies': pols,
            'value_targets': vt,
            'actual_len': L,
            'result': int(r_val),
        })

    return trajectories, results, pos_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_simulations', type=int, default=256)
    parser.add_argument('--leaves_per_game', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='data/selfplay_fixed')
    parser.add_argument('--num_files', type=int, default=16)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        'batch_size': args.batch_size,
        'num_simulations': args.num_simulations,
        'leaves_per_game': args.leaves_per_game,
        'c_puct': 1.0,
        'dirichlet_eps': 0.25,
        'dirichlet_alpha': 0.03,
    }

    for fi in range(args.num_files):
        t0 = time.perf_counter()
        trajectories, results, pos_lens = generate_batch(**config)
        dt = time.perf_counter() - t0

        bw = int((results == 1).sum())
        ww = int((results == 2).sum())
        dr = int((results == 3).sum()) + int((results == 0).sum())
        avg_len = pos_lens.mean()

        file_idx = args.start_idx + fi
        out_path = os.path.join(args.output_dir, f'selfplay_{file_idx:04d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(trajectories, f)

        meta = {
            'batch_size': args.batch_size,
            'num_simulations': args.num_simulations,
            'leaves_per_game': args.leaves_per_game,
            'black_wins': bw, 'white_wins': ww, 'draws': dr,
            'avg_length': float(avg_len),
            'time_seconds': dt,
        }
        with open(out_path.replace('.pkl', '_meta.json'), 'w') as f:
            json.dump(meta, f)

        print(f"[{fi + 1}/{args.num_files}] {out_path}: "
              f"len={avg_len:.0f} B={bw} W={ww} D={dr} "
              f"BWR={bw / (bw + ww) * 100:.0f}% "
              f"time={dt:.0f}s")

        # Verify first file
        if file_idx == args.start_idx:
            print(f"\n=== Verification: first 3 games ===")
            for g in range(min(3, len(trajectories))):
                t = trajectories[g]
                L = t['actual_len']
                r_val = t['result']
                r_str = {1: 'B_WIN', 2: 'W_WIN', 3: 'DRAW'}.get(r_val, f'r={r_val}')
                print(f"\n  Game {g}: len={L} result={r_str}")
                # Check π_0 and π_1
                pol = t['mcts_policies']
                print(f"  π_0 (empty board): max={pol[0].max():.4f} nonzero={(pol[0]>1e-6).sum()}")
                if len(pol) > 1:
                    print(f"  π_1 (after 1 move): max={pol[1].max():.4f} nonzero={(pol[1]>1e-6).sum()}")


if __name__ == '__main__':
    main()
