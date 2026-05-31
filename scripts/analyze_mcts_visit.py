#!/usr/bin/env python3
"""MCTS visit distribution analysis for specific board scenarios.

Scenarios (sleep-4 = 4 in a row, one end blocked by board edge):
  1. Black sleep-4, Black to play → correct: complete 5 at open end
  2. Black sleep-4, White to play → correct: block open end
  3. White sleep-4, White to play → correct: complete 5 at open end
  4. White sleep-4, Black to play → correct: block open end

For each scenario runs MCTS with S=16,32,64,128,256 and reports top visit
distribution for both a trained model and a random baseline.
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device('cuda')
N_CELLS = 225


def pos(r, c):
    return r * 15 + c


# ── Scenarios ─────────────────────────────────────────────────────
# Using horizontal 4-in-a-row at row 7, cols 1-4, left end blocked by board edge.
# Open end at (7,5). All scenarios share the same critical point.
SCENARIOS = [
    {
        'name': 'B_sleep4_B_turn',
        'desc': 'Black has sleep-4, Black to play → complete 5 at (7,5)',
        'p0': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],  # 8 black (4 sleep-4 + 4 filler)
        'p1': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1), pos(11,11)],  # 7 white filler
        'current_player': 0,
        'correct': pos(7,5),
    },
    {
        'name': 'B_sleep4_W_turn',
        'desc': 'Black has sleep-4, White to play → block at (7,5)',
        'p0': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],  # 8 black
        'p1': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1)],  # 6 white
        'current_player': 1,
        'correct': pos(7,5),
    },
    {
        'name': 'W_sleep4_W_turn',
        'desc': 'White has sleep-4, White to play → complete 5 at (7,5)',
        'p0': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1), pos(11,11)],  # 7 black filler
        'p1': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],  # 8 white (4 sleep-4 + 4 filler)
        'current_player': 1,
        'correct': pos(7,5),
    },
    {
        'name': 'W_sleep4_B_turn',
        'desc': 'White has sleep-4, Black to play → block at (7,5)',
        'p0': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1)],  # 6 black filler
        'p1': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],  # 8 white
        'current_player': 0,
        'correct': pos(7,5),
    },
]


def make_history(p0_stones, p1_stones):
    """Build alternating move sequence from stone lists.
    Black (player 0) goes first, then white, alternating."""
    pos_list, plr_list = [], []
    max_len = max(len(p0_stones), len(p1_stones))
    for i in range(max_len):
        if i < len(p0_stones):
            pos_list.append(p0_stones[i])
            plr_list.append(0)
        if i < len(p1_stones):
            pos_list.append(p1_stones[i])
            plr_list.append(1)
    return pos_list, plr_list


def top5_str(rp, N_CELLS=N_CELLS):
    ranked = [(rp[a], a) for a in range(N_CELLS) if rp[a] > 0]
    ranked.sort(reverse=True)
    return " ".join(f"{a:3d}({v:.3f})" for v, a in ranked[:5])


def run_analysis(model, model_name, scenarios, S_values, c_puct=1.0, label=""):
    """Run MCTS analysis for all scenarios with a given model."""
    for sc in scenarios:
        print(f"\n{'=' * 65}")
        print(f"[{model_name}] {sc['name']}: {sc['desc']}")
        print(f"{'=' * 65}")

        history_pos, history_plr = make_history(sc['p0'], sc['p1'])

        # Prefill game history into KV cache
        pos_t = torch.tensor(history_pos, dtype=torch.long, device=DEVICE).unsqueeze(0)
        plr_t = torch.tensor(history_plr, dtype=torch.long, device=DEVICE).unsqueeze(0)

        kv_cache, br_cache = model.create_cache(max_games=1, max_cache_len=250)
        root_pol, root_val = model.prefill(pos_t, plr_t, kv_cache, br_cache, [0])

        p0_bool = np.zeros(N_CELLS, dtype=bool)
        p1_bool = np.zeros(N_CELLS, dtype=bool)
        for s in sc['p0']: p0_bool[s] = True
        for s in sc['p1']: p1_bool[s] = True
        occ_bool = p0_bool | p1_bool

        # Model's raw policy (before MCTS)
        raw_policy = torch.softmax(root_pol.float(), dim=-1).cpu().squeeze(0)  # (225,)
        legal_policy = raw_policy.clone()
        legal_policy[occ_bool] = 0
        legal_policy /= legal_policy.sum()
        correct_raw = legal_policy[sc['correct']].item()

        print(f"  History: {len(history_pos)} plies, current={'Black' if sc['current_player']==0 else 'White'}")
        print(f"  Correct: {sc['correct']} (r={sc['correct']//15},c={sc['correct']%15})")
        print(f"  Raw policy → correct: {correct_raw:.4f}  top5: {top5_str(legal_policy.numpy())}")
        print(f"  {'S':>4s} | {'correct_visit':>14s} | {'correct_Q':>10s} | top5 visits                 | time")
        print(f"  {'-'*4} | {'-'*14} | {'-'*10} | {'-'*30} | {'-'*6}")

        for S in S_values:
            t0 = time.perf_counter()

            mgr = gomoku_cpp.MCTSManager(1, seed_base=42)
            mgr.c_puct = c_puct
            mgr.dirichlet_eps = 0.0
            mgr.leaves_per_game = 1
            mgr.init_roots(p0_bool.reshape(1, -1), p1_bool.reshape(1, -1),
                           np.array([sc['current_player']], dtype=np.int32))

            # Root expansion
            lp = raw_policy.clone()
            lp[occ_bool] = 0
            mgr.expand_roots(np.array([0], dtype=np.int32),
                             lp.unsqueeze(0).numpy().astype(np.float32),
                             root_val.cpu().numpy().astype(np.float32))

            for _ in range(S):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0:
                    continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0:
                    continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                lp2, lv2 = model.evaluate_mcts_leaves(pt, pl2, kv_cache, sl, lt)
                torch.cuda.synchronize()
                ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32),
                                      torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                      lv2.cpu().numpy().astype(np.float32))

            rp = mgr.get_root_policies().flatten()

            correct_visit = rp[sc['correct']]
            dt = time.perf_counter() - t0

            # Also fetch Q for the correct edge
            # The MCTSManager doesn't expose individual edge Q easily, so just show visits
            print(f"  {S:4d} | {correct_visit:14.4f} | {'':>10s} | {top5_str(rp):30s} | {dt:.1f}s")

            del mgr
            torch.cuda.empty_cache()

        del kv_cache, br_cache
        torch.cuda.empty_cache()
        print()


class RandomModel:
    """Random baseline: noisy uniform policy, zero value."""
    def __init__(self, cfg):
        self.config = cfg
        self._dummy = GomokuTransformer(cfg).to(DEVICE)  # for create_cache format

    def create_cache(self, max_games=1, max_cache_len=250):
        return self._dummy.create_cache(max_games, max_cache_len)

    def prefill(self, pos, plr, kv_cache, branch_cache, indices):
        bs = pos.shape[0]
        return (torch.randn(bs, 225, device=DEVICE) * 0.02,
                torch.zeros(bs, device=DEVICE))

    def evaluate_mcts_leaves(self, positions, players, kv_cache, indices, path_lengths):
        bs = positions.shape[0]
        return (torch.randn(bs, 225, device=DEVICE) * 0.02,
                torch.zeros(bs, device=DEVICE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',
                        default='checkpoints/run500_fix1stB_0531_1423_1972/step_000010.pt')
    parser.add_argument('--S', type=int, nargs='+', default=[16, 32, 64, 128, 256])
    parser.add_argument('--c_puct', type=float, default=1.0)
    parser.add_argument('--random', action='store_true',
                        help='Only run random baseline (skip model load)')
    args = parser.parse_args()

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                      n_shared=4, n_policy=4, n_value=4)

    if not args.random:
        print(f"Loading model from {args.ckpt}")
        model = GomokuTransformer(cfg).to(DEVICE).eval()
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        run_analysis(model, f"Training c_puct=0.2 eval c_puct={args.c_puct}",
                     SCENARIOS, args.S, args.c_puct)
    else:
        print("Skipping trained model (--random).")

    print(f"\n{'#' * 65}")
    print(f"# RANDOM BASELINE (eval c_puct={args.c_puct})")
    print(f"{'#' * 65}")
    rnd = RandomModel(cfg)
    run_analysis(rnd, f"Random eval c_puct={args.c_puct}",
                 SCENARIOS, args.S, args.c_puct)


if __name__ == '__main__':
    main()
