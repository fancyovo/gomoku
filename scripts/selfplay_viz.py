#!/usr/bin/env python3
"""Self-play visualization: play a few games with a trained model and render one as GIF."""
import sys, os, time, numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/run10_lr3e4/step_000009.pt"
OUT_GIF = sys.argv[2] if len(sys.argv) > 2 else "output/selfplay_game.gif"
N_GAMES = 5

BOARD_SIZE = 15
N_CELLS = 225

# ── Board rendering ──────────────────────────────────────────
def draw_board(ax, p0, p1, last_move=None, move_num=0):
    """Draw a Gomoku board."""
    ax.clear()
    ax.set_xlim(-0.5, BOARD_SIZE + 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE + 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # Grid
    for i in range(BOARD_SIZE):
        ax.axhline(i, color="black", linewidth=0.3, alpha=0.5)
        ax.axvline(i, color="black", linewidth=0.3, alpha=0.5)

    # Star points
    stars = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11),
             (11, 3), (11, 7), (11, 11)]
    for r, c in stars:
        ax.add_patch(Circle((c, r), 0.12, color="black", zorder=3))

    # Stones
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pos = r * BOARD_SIZE + c
            if p0[pos]:
                ax.add_patch(Circle((c, r), 0.42, color="black", zorder=4, ec="none"))
            elif p1[pos]:
                ax.add_patch(Circle((c, r), 0.42, color="white", zorder=4, ec="black", linewidth=0.5))

    # Last move marker
    if last_move is not None:
        r, c = last_move // BOARD_SIZE, last_move % BOARD_SIZE
        ax.add_patch(Circle((c, r), 0.12, color="red", zorder=5, alpha=0.8))

    ax.set_title(f"Move {move_num}", fontsize=10)


def pos_to_str(p):
    return f"{chr(ord('A') + p % BOARD_SIZE)}{p // BOARD_SIZE + 1}"


# ── Load model ───────────────────────────────────────────────
cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
model = GomokuTransformer(cfg).to(DEVICE).eval()
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
print(f"Loaded: {CKPT}")

# ── Self-play ────────────────────────────────────────────────
G, M, S = N_GAMES, 8, 64
pool = gomoku_cpp.GamePool(G)
pool.reset_all()
mgr = gomoku_cpp.MCTSManager(G, seed_base=42)
mgr.c_puct = 1.0
mgr.dirichlet_eps = 0.0  # no noise for demonstration
mgr.dirichlet_alpha = 0.03
mgr.leaves_per_game = M

p0 = np.zeros((G, N_CELLS), dtype=bool)
p1 = np.zeros((G, N_CELLS), dtype=bool)
mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
kv = model.create_cache(max_games=G, max_cache_len=250)

fa = model.sample_first_moves(G, DEVICE)
model.prefill(fa.unsqueeze(1), torch.zeros(G, 1, dtype=torch.long, device=DEVICE), kv, list(range(G)))

ph = [[] for _ in range(G)]
plh = [[] for _ in range(G)]
p0_c = np.zeros((G, N_CELLS), dtype=bool)
p1_c = np.zeros((G, N_CELLS), dtype=bool)
plen = np.zeros(G, dtype=np.int32)
fin = np.zeros(G, dtype=bool)
res = np.zeros(G, dtype=np.int32)

for g in range(G):
    a = int(fa[g].item())
    ph[g].append(a); plh[g].append(0); p0_c[g, a] = True; plen[g] = 1
    r = gomoku_cpp.step(pool, g, a)
    if r:
        fin[g] = True; res[g] = r; mgr.reset_game(g)
    else:
        mgr.apply_move(g, a, p0_c[g], p1_c[g])

occ_g = torch.from_numpy(p0_c | p1_c).to(DEVICE)

move_count = 1
while True:
    act = np.where(~fin)[0]
    if len(act) == 0:
        break
    st = torch.from_numpy(act).to(DEVICE)
    cp = int(plen[act[0]]) % 2
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
        if sel['max_path_len'] == 0:
            continue
        vi = np.where(sel['valid_mask'])[0]
        if len(vi) == 0:
            continue
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
    na = np.zeros(len(act), dtype=np.int64)
    for i, g in enumerate(act):
        pol = rp[g].copy()
        pol[p0_c[g] | p1_c[g]] = 0
        if pol.sum() > 0:
            a = int(np.random.choice(225, p=pol / pol.sum()))
        else:
            leg = np.where(~(p0_c[g] | p1_c[g]))[0]
            a = int(np.random.choice(leg)) if len(leg) > 0 else 0
        na[i] = a
        ph[g].append(a); plh[g].append(plen[g] % 2)
        if plen[g] % 2 == 0:
            p0_c[g, a] = True
        else:
            p1_c[g, a] = True
        plen[g] += 1
    dec_p = torch.from_numpy(na).to(DEVICE)
    dec_pl = torch.from_numpy((np.array([plen[g] % 2 for g in act]) ^ 1)).to(DEVICE)
    actually, wait — let me just use the correct player:

Wait, I need to fix this. The player who just played is (plen[g] - 1) % 2. The next player is plen[g] % 2. But decode needs the player who just played.

Actually, looking at the original run_selfplay code:
```python
dec_pl = torch.from_numpy(np_).to(DEVICE)
```
Where np_[i] = plen[g] % 2. And plen[g] is the length BEFORE appending. So np_[i] is the player of the new move.

Wait, looking more carefully at the original code:
```python
na[i] = a; np_[i] = plen[g] % 2
ph[g].append(a); plh[g].append(np_[i])
```

So plen[g] is the length before appending. np_[i] = plen[g] % 2 is the player of the NEW move. After appending: plen[g] += 1.

Then:
```python
dec_pl = torch.from_numpy(np_).to(DEVICE)
model.decode(dec_p, dec_pl, kv, ...)
```

decode expects the player who just played. The new move's player IS the player who just played. So this is correct.

Let me fix my code.

Actually, I realize there's a bug in my visualization script. Let me re-examine and fix it.<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Write">
<｜｜DSML｜｜parameter name="content" string="true">#!/usr/bin/env python3
"""Self-play visualization: play a few games and render one as GIF.

Usage: python scripts/selfplay_viz.py [ckpt_path] [output_gif]
"""
import sys, os, time, numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/run10_lr3e4/step_000009.pt"
OUT_GIF = sys.argv[2] if len(sys.argv) > 2 else "output/selfplay_game.gif"
N_GAMES = 5
BOARD_SIZE = 15
N_CELLS = 225


def pos_to_str(p):
    return f"{chr(ord('A') + p % BOARD_SIZE)}{p // BOARD_SIZE + 1}"


def draw_board(ax, p0, p1, last_move=None, move_num=0):
    """Draw a Gomoku board state."""
    ax.clear()
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    for i in range(BOARD_SIZE):
        ax.axhline(i, color="black", linewidth=0.3, alpha=0.4)
        ax.axvline(i, color="black", linewidth=0.3, alpha=0.4)

    stars = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11),
             (11, 3), (11, 7), (11, 11)]
    for r, c in stars:
        ax.add_patch(Circle((c, r), 0.12, color="black", zorder=3))

    for pos in range(N_CELLS):
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        if p0[pos]:
            ax.add_patch(Circle((c, r), 0.42, color="black", zorder=4, ec="none"))
        elif p1[pos]:
            ax.add_patch(Circle((c, r), 0.42, color="white", zorder=4,
                                ec="black", linewidth=0.5))

    if last_move is not None:
        r, c = last_move // BOARD_SIZE, last_move % BOARD_SIZE
        ax.add_patch(Circle((c, r), 0.14, color="red", zorder=5, alpha=0.85))

    ax.set_title(f"Move {move_num}", fontsize=10)


# ── Load model ──────────────────────────────────────────────────
cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
model = GomokuTransformer(cfg).to(DEVICE).eval()
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
print(f"Loaded: {CKPT}")

# ── Self-play ───────────────────────────────────────────────────
G, M, S = N_GAMES, 8, 64
pool = gomoku_cpp.GamePool(G)
pool.reset_all()
mgr = gomoku_cpp.MCTSManager(G, seed_base=42)
mgr.c_puct = 1.0
mgr.dirichlet_eps = 0.0   # deterministic for clean demo
mgr.dirichlet_alpha = 0.03
mgr.leaves_per_game = M

p0 = np.zeros((G, N_CELLS), dtype=bool)
p1 = np.zeros((G, N_CELLS), dtype=bool)
mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
kv = model.create_cache(max_games=G, max_cache_len=250)

# First moves
fa = model.sample_first_moves(G, DEVICE)
model.prefill(fa.unsqueeze(1),
              torch.zeros(G, 1, dtype=torch.long, device=DEVICE),
              kv, list(range(G)))

ph = [[] for _ in range(G)]   # position history
plh = [[] for _ in range(G)]  # player history
p0_c = np.zeros((G, N_CELLS), dtype=bool)
p1_c = np.zeros((G, N_CELLS), dtype=bool)
plen = np.zeros(G, dtype=np.int32)
fin = np.zeros(G, dtype=bool)
res = np.zeros(G, dtype=np.int32)

for g in range(G):
    a = int(fa[g].item())
    ph[g].append(a); plh[g].append(0); p0_c[g, a] = True; plen[g] = 1
    r = gomoku_cpp.step(pool, g, a)
    if r:
        fin[g] = True; res[g] = r; mgr.reset_game(g)
    else:
        mgr.apply_move(g, a, p0_c[g], p1_c[g])

occ_g = torch.from_numpy(p0_c | p1_c).to(DEVICE)

# Main loop
while True:
    act = np.where(~fin)[0]
    if len(act) == 0:
        break
    st = torch.from_numpy(act).to(DEVICE)
    cp = int(plen[act[0]]) % 2
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
        if sel['max_path_len'] == 0:
            continue
        vi = np.where(sel['valid_mask'])[0]
        if len(vi) == 0:
            continue
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
    na = np.zeros(len(act), dtype=np.int64)
    np_ = np.zeros(len(act), dtype=np.int64)
    for i, g in enumerate(act):
        pol = rp[g].copy()
        pol[p0_c[g] | p1_c[g]] = 0
        if pol.sum() > 0:
            a = int(np.random.choice(225, p=pol / pol.sum()))
        else:
            leg = np.where(~(p0_c[g] | p1_c[g]))[0]
            a = int(np.random.choice(leg)) if len(leg) > 0 else 0
        na[i] = a; np_[i] = plen[g] % 2
        ph[g].append(a); plh[g].append(np_[i])
        if np_[i] == 0:
            p0_c[g, a] = True
        else:
            p1_c[g, a] = True
        plen[g] += 1
    dec_p = torch.from_numpy(na).to(DEVICE)
    dec_pl = torch.from_numpy(np_).to(DEVICE)
    model.decode(dec_p, dec_pl, kv, torch.from_numpy(act).to(DEVICE))
    for i, g in enumerate(act):
        r = gomoku_cpp.step(pool, g, int(na[i]))
        if r:
            fin[g] = True; res[g] = r; mgr.reset_game(g)
        else:
            mgr.apply_move(g, int(na[i]), p0_c[g], p1_c[g])

del kv; torch.cuda.empty_cache()

# ── Print games ─────────────────────────────────────────────────
result_names = {1: "B_WIN", 2: "W_WIN", 3: "DRAW"}
for g in range(N_GAMES):
    L = plen[g]
    rv = res[g] if res[g] != 0 else 3
    print(f"\n=== Game {g + 1}: len={L} result={result_names.get(rv, rv)} ===")
    moves_str = " ".join(pos_to_str(a) for a in ph[g])
    # Print 15 moves per line
    for i in range(0, len(moves_str.split()), 15):
        print("  " + " ".join(moves_str.split()[i:i + 15]))

# ── Pick the shortest game for GIF ──────────────────────────────
best_g = min(range(N_GAMES), key=lambda g: plen[g])
L = plen[best_g]
print(f"\nRendering GIF: Game {best_g + 1} ({L} moves) -> {OUT_GIF}")

# Replay the game move by move, capturing frames
b0 = np.zeros(N_CELLS, dtype=bool)
b1 = np.zeros(N_CELLS, dtype=bool)
frames = []
fig, ax = plt.subplots(figsize=(6, 6))

# Empty board
draw_board(ax, b0, b1, move_num=0)
fig.tight_layout()
fig.canvas.draw()
buf = np.asarray(fig.canvas.buffer_rgba())
frames.append(buf)

for i, a in enumerate(ph[best_g]):
    plr = plh[best_g][i]
    if plr == 0:
        b0[a] = True
    else:
        b1[a] = True
    draw_board(ax, b0, b1, last_move=a, move_num=i + 1)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frames.append(buf)

plt.close()

# Save GIF at 0.5s/frame
imageio.mimsave(OUT_GIF, frames, duration=0.5, loop=0)
print(f"Saved: {OUT_GIF} ({len(frames)} frames)")
