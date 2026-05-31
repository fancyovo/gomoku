#!/usr/bin/env python3
"""After playing MCTS's top (non-blocking) move, check if (7,5) becomes a winning attack."""
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


# Only blocking scenarios
BLOCK_SCENARIOS = [
    {
        'name': 'B_sleep4_W_turn',
        'desc': 'Black has sleep-4 at (7,1-4), White to play',
        'p0': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],
        'p1': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1)],
        'current_player': 1,  # white
        'correct': pos(7,5),
        'next_player': 0,  # black after white plays
    },
    {
        'name': 'W_sleep4_B_turn',
        'desc': 'White has sleep-4 at (7,1-4), Black to play',
        'p0': [pos(2,2), pos(4,4), pos(6,6), pos(9,9),
               pos(10,10), pos(1,1)],
        'p1': [pos(7,1), pos(7,2), pos(7,3), pos(7,4),
               pos(3,3), pos(5,5), pos(8,8), pos(0,0)],
        'current_player': 0,  # black
        'correct': pos(7,5),
        'next_player': 1,  # white after black plays
    },
]


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


def run_mcts(model, history_pos, history_plr, p0_bool, p1_bool, current_player, S, c_puct=1.0):
    """Run MCTS on a position, return (visit_distribution, model_raw_policy)."""
    pos_t = torch.tensor(history_pos, dtype=torch.long, device=DEVICE).unsqueeze(0)
    plr_t = torch.tensor(history_plr, dtype=torch.long, device=DEVICE).unsqueeze(0)
    kv_cache, br_cache = model.create_cache(max_games=1, max_cache_len=250)
    root_pol, root_val = model.prefill(pos_t, plr_t, kv_cache, br_cache, [0])

    occ_bool = p0_bool | p1_bool
    raw_policy = torch.softmax(root_pol.float(), dim=-1).cpu().squeeze(0)

    mgr = gomoku_cpp.MCTSManager(1, seed_base=42)
    mgr.c_puct = c_puct
    mgr.dirichlet_eps = 0.0
    mgr.leaves_per_game = 1
    mgr.init_roots(p0_bool.reshape(1, -1), p1_bool.reshape(1, -1),
                    np.array([current_player], dtype=np.int32))
    lp = raw_policy.clone()
    lp[occ_bool] = 0
    mgr.expand_roots(np.array([0], dtype=np.int32),
                      lp.unsqueeze(0).numpy().astype(np.float32),
                      root_val.cpu().numpy().astype(np.float32))
    for _ in range(S):
        sel = mgr.select_all()
        if sel['max_path_len'] == 0: continue
        vi = np.where(sel['valid_mask'])[0]
        if len(vi) == 0: continue
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
    del mgr, kv_cache, br_cache
    torch.cuda.empty_cache()
    return rp, raw_policy


def top5_str(rp):
    ranked = [(rp[a], a, a//15, a%15) for a in range(N_CELLS) if rp[a] > 0]
    ranked.sort(reverse=True)
    return " ".join(f"{a:3d}({v:.3f})" for v, a, _, _ in ranked[:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/run500_fix1stB_0531_1423_1972/step_000109.pt')
    parser.add_argument('--S', type=int, default=256)
    parser.add_argument('--c_puct', type=float, default=1.0)
    args = parser.parse_args()

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                      n_shared=4, n_policy=4, n_value=4)
    model = GomokuTransformer(cfg).to(DEVICE).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))

    for sc in BLOCK_SCENARIOS:
        print(f"\n{'='*70}")
        print(f"STEP 1: {sc['name']} — {sc['desc']}")
        print(f"{'='*70}")

        history_pos, history_plr = make_history(sc['p0'], sc['p1'])
        p0_bool = np.zeros(N_CELLS, dtype=bool)
        p1_bool = np.zeros(N_CELLS, dtype=bool)
        for s in sc['p0']: p0_bool[s] = True
        for s in sc['p1']: p1_bool[s] = True

        # Run MCTS on original position
        rp, raw_pol = run_mcts(model, history_pos, history_plr,
                                p0_bool, p1_bool, sc['current_player'],
                                args.S, args.c_puct)

        # MCTS's top-1 move
        ranked = [(rp[a], a) for a in range(N_CELLS) if rp[a] > 0]
        ranked.sort(reverse=True)
        top1_move = ranked[0][1]
        top1_visit = ranked[0][0]
        block_visit = rp[sc['correct']]

        raw_legal = raw_pol.clone()
        raw_legal[p0_bool | p1_bool] = 0
        raw_legal /= raw_legal.sum()
        raw_block_prob = raw_legal[sc['correct']].item()
        raw_top1 = raw_legal.argmax().item()
        raw_top1_prob = raw_legal[raw_top1].item()

        print(f"  Correct block: {sc['correct']} (r={sc['correct']//15},c={sc['correct']%15})")
        print(f"  Model raw policy → top1: {raw_top1}({raw_top1_prob:.3f})  block: {raw_block_prob:.3f}")
        print(f"  MCTS S={args.S}   → top1: {top1_move}({top1_visit:.3f})  block visit: {block_visit:.3f}")
        print(f"  MCTS top5: {top5_str(rp)}")

        # STEP 2: Apply top1 move (the non-blocking move) and see if (7,5) emerges as attack
        print(f"\n  ---> Playing MCTS top-1 move: {top1_move} (r={top1_move//15},c={top1_move%15})")
        print(f"  ---> Now it's opponent's turn, (7,5) should become a WINNING attack...")

        # Update board state with the non-blocking move
        cur_player = sc['current_player']
        if cur_player == 0:  # black just played
            p0_bool[top1_move] = True
        else:  # white just played
            p1_bool[top1_move] = True
        next_player = 1 - cur_player

        # New history: append the non-blocking move
        new_history_pos = history_pos + [top1_move]
        new_history_plr = history_plr + [cur_player]

        # Run MCTS on new position
        rp2, raw_pol2 = run_mcts(model, new_history_pos, new_history_plr,
                                  p0_bool, p1_bool, next_player,
                                  args.S, args.c_puct)

        ranked2 = [(rp2[a], a) for a in range(N_CELLS) if rp2[a] > 0]
        ranked2.sort(reverse=True)
        top1_2 = ranked2[0][1] if ranked2 else -1
        top1_2_visit = ranked2[0][0] if ranked2 else 0
        final5_visit = rp2[sc['correct']]

        raw_legal2 = raw_pol2.clone()
        raw_legal2[p0_bool | p1_bool] = 0
        raw_legal2 /= raw_legal2.sum()
        raw_final5_prob = raw_legal2[sc['correct']].item()

        print(f"  Opponent MCTS S={args.S} → top1: {top1_2}({top1_2_visit:.3f})  (7,5) visit: {final5_visit:.3f}")
        print(f"  Opponent raw policy → (7,5) prob: {raw_final5_prob:.3f}")
        print(f"  Opponent MCTS top5: {top5_str(rp2)}")
        print()

    print("=== Done ===")


if __name__ == '__main__':
    main()
