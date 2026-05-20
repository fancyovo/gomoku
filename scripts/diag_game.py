#!/usr/bin/env python3
"""Diagnostic: random model self-play, dump last 15 moves + MCTS stats for 3 games."""
import torch, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = 64; M = 4; S = 64  # fewer games but full sims for detail

def row_col(a):
    return f"{'ABCDEFGHIJKLMNO'[a//15]}{a%15+1}"

cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
model = GomokuTransformer(cfg).to(DEVICE).eval()
print("Random model", flush=True)

pool = gomoku_cpp.GamePool(G); pool.reset_all()
mgr = gomoku_cpp.MCTSManager(G, seed_base=42)
mgr.c_puct = 1.0; mgr.dirichlet_eps = 0.25; mgr.dirichlet_alpha = 0.03
mgr.leaves_per_game = M
p0 = np.zeros((G, 225), dtype=bool); p1 = np.zeros((G, 225), dtype=bool)
mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
kv = model.create_cache(max_games=G, max_cache_len=250)
fa = model.sample_first_moves(G, DEVICE)
model.prefill(fa.unsqueeze(1), torch.zeros(G, 1, dtype=torch.long, device=DEVICE), kv, list(range(G)))

pos_hist = [[] for _ in range(G)]; plr_hist = [[] for _ in range(G)]
pos_lens = np.zeros(G, dtype=np.int32)
finished = np.zeros(G, dtype=bool); results = np.zeros(G, dtype=np.int32)

for g in range(G):
    a = int(fa[g].item()); pos_hist[g].append(a); plr_hist[g].append(0)
    p0[g, a] = True; pos_lens[g] = 1
    r = gomoku_cpp.step(pool, g, a)
    if r: finished[g] = True; results[g] = r; mgr.reset_game(g)
    else: mgr.apply_move(g, a, p0[g], p1[g])
occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)

while True:
    active = np.where(~finished)[0]
    if len(active) == 0: break
    st = torch.from_numpy(active).to(DEVICE)
    cp = int(pos_lens[active[0]]) % 2
    dp = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
    dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=DEVICE)
    dl = torch.ones(len(active), dtype=torch.long, device=DEVICE)
    lp, lv = model.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
    lp = lp.masked_fill(occ_gpu[active], -1e9); torch.cuda.synchronize()
    mgr.expand_roots(active.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
    for _ in range(S):
        sel = mgr.select_all()
        if sel['max_path_len'] == 0: continue
        vi = np.where(sel['valid_mask'])[0]
        if len(vi) == 0: continue
        pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
        plr_t2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
        lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
        slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
        lp, lv = model.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
        occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
        lp = lp.masked_fill(occ_t, -1e9)
        mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
    rp = mgr.get_root_policies()
    new_actions = np.zeros(len(active), dtype=np.int64); new_plrs = np.zeros(len(active), dtype=np.int64)
    for i, g in enumerate(active):
        pol = rp[g].copy(); pol[p0[g] | p1[g]] = 0
        if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
        else:
            legal = np.where(~(p0[g] | p1[g]))[0]
            a = int(np.random.choice(legal)) if len(legal) > 0 else 0
        new_actions[i] = a; new_plrs[i] = pos_lens[g] % 2
        pos_hist[g].append(a); plr_hist[g].append(new_plrs[i])
        if new_plrs[i] == 0: p0[g, a] = True
        else: p1[g, a] = True
        pos_lens[g] += 1
    dec_pos = torch.from_numpy(new_actions).to(DEVICE)
    dec_plr = torch.from_numpy(new_plrs).to(DEVICE)
    dec_slots = torch.from_numpy(active).to(DEVICE)
    model.decode(dec_pos, dec_plr, kv, dec_slots)
    for i, g in enumerate(active):
        r = gomoku_cpp.step(pool, g, int(new_actions[i]))
        if r: finished[g] = True; results[g] = r; mgr.reset_game(g)
        else: mgr.apply_move(g, int(new_actions[i]), p0[g], p1[g])

del kv; torch.cuda.empty_cache()

# Print 3 complete games with MCTS policy stats for last moves
print(f"\nAvg length: {pos_lens.mean():.1f}")
print(f"Black wins: {(results==1).sum()}, White wins: {(results==2).sum()}, Draws: {(results==0).sum()+(results==3).sum()}")
print()

# Analyze policy entropy: does MCTS concentrate mass near game end?
for g_idx in [0, 1, 2]:
    L = pos_lens[g_idx]
    result = {1: "Black", 2: "White", 3: "Draw", 0: "Draw"}[results[g_idx]]
    print(f"=== Game {g_idx}: {result} wins, {L} moves ===")
    # Print all moves in compact form
    for i in range(L):
        a = pos_hist[g_idx][i]
        p = plr_hist[g_idx][i]
        print(f"  {i:3d}: {'B' if p==0 else 'W'} {row_col(a):>3s}", end="")
        if i % 5 == 4 or i == L-1:
            print()
    print(f"\n  Board state at end:")
    print(f"  p0 stones: {p0[g_idx].sum()}, p1 stones: {p1[g_idx].sum()}")
    print(f"  occupied: {(p0[g_idx] | p1[g_idx]).sum()}")
    print()

# Check: any game going past 150 moves without a win condition being possible?
long_games = [(g, pos_lens[g]) for g in range(G) if pos_lens[g] > 150]
if long_games:
    print(f"WARNING: {len(long_games)} games > 150 moves!")
    for g, l in long_games[:3]:
        print(f"  Game {g}: {int(l)} moves, result={results[g]}")
else:
    print("OK: No games > 150 moves")

# Check the shortest games
short_games = sorted([(g, pos_lens[g]) for g in range(G)], key=lambda x: x[1])
print(f"\nShortest games:")
for g, l in short_games[:5]:
    print(f"  Game {g}: {int(l)} moves, result={results[g]}")
