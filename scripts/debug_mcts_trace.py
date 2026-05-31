#!/usr/bin/env python3
"""Trace MCTS step-by-step: track score components for target (7,5) vs top competitors.

For blocking scenario, prints at each simulation the root edge stats for the
top-5 edges by visit count plus the blocking edge, showing Q, U, P, N, score.
Also tracks whether 2-ply discovery happens (opponent finds winning move).
"""
import argparse, os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device('cuda')
N_CELLS = 225


def pos(r, c):
    return r * 15 + c


SCENARIO = {
    'name': 'W_sleep4_B_turn',
    'desc': 'White has sleep-4 at (7,1-4), Black to play → block at (7,5)',
    'p0': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
           pos(10,10), pos(1,1)],
    'p1': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
           pos(3,3), pos(5,5), pos(8,8), pos(0,0)],
    'current_player': 0,
    'correct': pos(7,5),
}


def make_history(p0_stones, p1_stones):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/run500_fix1stB_0531_1423_1972/step_000109.pt')
    parser.add_argument('--S', type=int, default=512)
    parser.add_argument('--c_puct', type=float, default=1.0)
    parser.add_argument('--print_every', type=int, default=16,
                        help='Print trace every N simulations')
    args = parser.parse_args()

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                      n_shared=4, n_policy=4, n_value=4)
    model = GomokuTransformer(cfg).to(DEVICE).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))

    sc = SCENARIO
    history_pos, history_plr = make_history(sc['p0'], sc['p1'])

    # Prefill
    pos_t = torch.tensor(history_pos, dtype=torch.long, device=DEVICE).unsqueeze(0)
    plr_t = torch.tensor(history_plr, dtype=torch.long, device=DEVICE).unsqueeze(0)
    kv_cache, br_cache = model.create_cache(max_games=1, max_cache_len=250)
    root_pol, root_val = model.prefill(pos_t, plr_t, kv_cache, br_cache, [0])

    p0_bool = np.zeros(N_CELLS, dtype=bool)
    p1_bool = np.zeros(N_CELLS, dtype=bool)
    for s in sc['p0']: p0_bool[s] = True
    for s in sc['p1']: p1_bool[s] = True
    occ_bool = p0_bool | p1_bool

    raw_policy = torch.softmax(root_pol.float(), dim=-1).cpu().squeeze(0)
    legal_policy = raw_policy.clone()
    legal_policy[occ_bool] = 0
    legal_policy /= legal_policy.sum()

    sc['correct'] = pos(7,5)
    correct_raw = legal_policy[sc['correct']].item()

    print(f"{'='*80}")
    print(f"TRACING: {sc['name']} — {sc['desc']}")
    print(f"S={args.S}, c_puct={args.c_puct}")
    print(f"Correct block: {sc['correct']} (r={sc['correct']//15},c={sc['correct']%15})")
    print(f"Raw policy @ block: {correct_raw:.4f}")
    print(f"{'='*80}")

    # Show top-10 raw policy
    top10_idx = np.argsort(-legal_policy.numpy())[:10]
    print("\nTop-10 raw policy (legal moves):")
    for i, aidx in enumerate(top10_idx):
        print(f"  {i+1}. {aidx:3d} (r={aidx//15},c={aidx%15:2d}): {legal_policy[aidx].item():.4f}")
    print(f"  Block(7,5) rank: {list(legal_policy.numpy()).index(correct_raw) if any(legal_policy.numpy() > 0) else 'N/A'}")

    # Initialize MCTS
    mgr = gomoku_cpp.MCTSManager(1, seed_base=42)
    mgr.c_puct = args.c_puct
    mgr.dirichlet_eps = 0.0
    mgr.leaves_per_game = 1
    mgr.init_roots(p0_bool.reshape(1, -1), p1_bool.reshape(1, -1),
                    np.array([sc['current_player']], dtype=np.int32))
    lp = raw_policy.clone()
    lp[occ_bool] = 0
    mgr.expand_roots(np.array([0], dtype=np.int32),
                      lp.unsqueeze(0).numpy().astype(np.float32),
                      root_val.cpu().numpy().astype(np.float32))

    # Track stats
    block_edge_idx = None  # we'll figure this out

    for sim in range(1, args.S + 1):
        sel = mgr.select_all()
        vi = np.where(sel['valid_mask'])[0] if sel['max_path_len'] > 0 else []

        if len(vi) > 0:
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

        # Print trace periodically
        if sim % args.print_every == 0 or sim == 1:
            rp = mgr.get_root_policies().flatten()
            # Get root node info via C++ — we can access root edges through the policy
            # Sort all edges by N (visit count), descending
            ranked = [(rp[a], a) for a in range(N_CELLS) if rp[a] > 0]
            ranked.sort(reverse=True)

            block_visit = float(rp[sc['correct']])
            top1_action = ranked[0][1]
            top1_visit = ranked[0][0]

            # Compute N_total for denominator
            total_visits = sum(rp[a] for a in range(N_CELLS))
            # root_N_total from total_visits rounded

            print(f"\n--- Sim {sim}/{args.S} ---")
            print(f"  Block(7,5): visit={block_visit:.4f}  top1: {top1_action}({top1_visit:.4f})")

            # Top-5 by visit
            print(f"  Top5 by visit:")
            for k in range(min(5, len(ranked))):
                v, a = ranked[k]
                raw_p = legal_policy[a].item()
                print(f"    {a:3d}(r={a//15},c={a%15:2d}): visit={v:.4f}  rawP={raw_p:.4f}")

            # Check: has 2-ply discovery happened? See if ANY opponent-win terminal
            # detection has backed up negative Q to non-blocking edges
            # If block visit > 0 and other edges have very negative Q, we know it did
            if block_visit < 0.01 and sim > 220:
                # After all edges visited once, check if any non-block edge has very low visit
                # (indicating negative Q from opponent response)
                n_negative = sum(1 for v, a in ranked if v < 0.001 and a != sc['correct'])
                print(f"  Non-block edges with near-zero visit: {n_negative}/{len(ranked)}")

            # Count how many distinct root edges have been visited
            visited_edges = len(ranked)
            legal_moves = int((~occ_bool).sum())
            print(f"  Visited root edges: {visited_edges}/{legal_moves} total_visits={int(total_visits+0.5)}")

    # Final result
    rp = mgr.get_root_policies().flatten()
    ranked = [(rp[a], a) for a in range(N_CELLS) if rp[a] > 0]
    ranked.sort(reverse=True)
    block_visit = float(rp[sc['correct']])

    print(f"\n{'='*80}")
    print(f"FINAL (S={args.S})")
    print(f"  Block(7,5) visit: {block_visit:.4f}")
    print(f"  Top-10 by visit:")
    for k in range(min(10, len(ranked))):
        v, a = ranked[k]
        raw_p = legal_policy[a].item()
        print(f"    {a:3d}(r={a//15},c={a%15:2d}): visit={v:.4f}  rawP={raw_p:.4f}")
    print(f"{'='*80}")

    del kv_cache, br_cache, mgr
    torch.cuda.empty_cache()
    print("=== Done ===")


if __name__ == '__main__':
    main()
