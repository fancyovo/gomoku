#!/usr/bin/env python3
"""Quick ELO tournament + heatmap for checkpoints/train_loop/"""
import torch, sys, os, time, numpy as np, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_ = 128   # games per pair (fast)
M = 4
S = 16
CHECKPOINT_DIR = "checkpoints/train_loop"
OUTPUT_DIR = "output"

def compute_elo(match_results):
    names = sorted(set(a for a, _, _, _ in match_results) | set(b for _, b, _, _ in match_results))
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(200):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n = sa + sb
            if n == 0: continue
            da = (sa - ea * n) * (32.0 / n); elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo

@torch.inference_mode()
def play_match(model_a, model_b):
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    def make_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = M
        m.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
        return m
    mgr_a = make_mgr(); mgr_b = make_mgr()
    kva = model_a.create_cache(max_games=G_, max_cache_len=250)
    kvb = model_b.create_cache(max_games=G_, max_cache_len=250)
    a_black = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    finished = np.zeros(G_, dtype=bool); winners = np.zeros(G_, dtype=np.int32)
    fa_a = model_a.sample_first_moves(G_, DEVICE); fa_b = model_b.sample_first_moves(G_, DEVICE)
    first_acts = np.zeros(G_, dtype=np.int64)
    p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)
    for g in range(G_):
        first_acts[g] = int(fa_a[g].item()) if a_black[g] else int(fa_b[g].item())
        p0[g, first_acts[g]] = True
    fa_t = torch.tensor(first_acts, dtype=torch.long, device=DEVICE).unsqueeze(1)
    plr_t = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    model_a.prefill(fa_t, plr_t, kva, list(range(G_)))
    model_b.prefill(fa_t, plr_t, kvb, list(range(G_)))
    occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)
    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first_acts[g]))
        if r: finished[g] = True; winners[g] = r
        else:
            bp0 = np.zeros(225, dtype=bool); bp0[int(first_acts[g])] = True
            mgr_a.apply_move(g, int(first_acts[g]), bp0, np.zeros(225, dtype=bool))
            mgr_b.apply_move(g, int(first_acts[g]), bp0, np.zeros(225, dtype=bool))
    for move in range(1, 200):
        active = np.where(~finished)[0]
        if len(active) == 0: break
        cp = move % 2
        for mgr, mdl, kv in [(mgr_a, model_a, kva), (mgr_b, model_b, kvb)]:
            st = torch.from_numpy(active).to(DEVICE)
            dp = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(active), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
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
                lp, lv = mdl.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
                occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp = lp.masked_fill(occ_t, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
        rp_a = mgr_a.get_root_policies(); rp_b = mgr_b.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            use_a = ((cp == 0) == a_black[g])
            pol = (rp_a[g] if use_a else rp_b[g]).copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True; occ_gpu[g, a] = True
        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.full((len(active),), cp, dtype=torch.long, device=DEVICE)
        model_a.decode(dec_pos, dec_plr, kva, torch.from_numpy(active).to(DEVICE))
        model_b.decode(dec_pos, dec_plr, kvb, torch.from_numpy(active).to(DEVICE))
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; winners[g] = r
            else:
                mgr_a.apply_move(g, int(new_actions[i]), p0[g], p1[g])
                mgr_b.apply_move(g, int(new_actions[i]), p0[g], p1[g])
    wa = wb = dr = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wa += 1 if a_black[g] else 0; wb += 0 if a_black[g] else 1
        elif w == 2: wa += 0 if a_black[g] else 1; wb += 1 if a_black[g] else 0
        else: dr += 1
    del kva, kvb; torch.cuda.empty_cache()
    return wa, wb, dr

# ─── Main ────────────────────────────────────────────────────────
cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
names = [os.path.basename(c) for c in ckpts]
print(f"Checkpoints: {len(names)}")
for n in names: print(f"  {n}")

# Load all models once
cache = {}
models = {}
for name in names:
    m = GomokuTransformer(cfg).to(DEVICE).eval()
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name), map_location=DEVICE))
    models[name] = m
    print(f"Loaded {name}")

match_results = []
for i, na in enumerate(names):
    for nb in names[i + 1:]:
        print(f"  {na} vs {nb} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        wa, wb, d = play_match(models[na], models[nb])
        dt = time.perf_counter() - t0
        wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5
        match_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
        print(f"{wa}-{wb} D={d} WR={wr:.1%} ({dt:.0f}s)")

# Compute ELO
elo = compute_elo(match_results)
items = sorted((int(n.split("_")[1].split(".")[0]), r) for n, r in elo.items())
steps_elo, ratings = zip(*items)
print("\nELO ratings:")
for s, r in items: print(f"  step_{s:06d}: {r:.0f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(steps_elo, ratings, ".-", color="C1", markersize=10, linewidth=2)
ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Training Step")
ax.set_ylabel("ELO Rating")
ax.set_title(f"ELO Curve ({G_} games/pair)")
ax.grid(True, alpha=0.3)

ax = axes[1]
steps_list = sorted(set(int(n.split("_")[1].split(".")[0]) for n in names))
n = len(steps_list)
if n >= 2:
    sti = {s: i for i, s in enumerate(steps_list)}
    wr_mat = np.full((n, n), np.nan)
    for na, nb, sa, sb in match_results:
        s1 = int(na.split("_")[1].split(".")[0])
        s2 = int(nb.split("_")[1].split(".")[0])
        total = sa + sb - (abs(sa - sb) * 0)  # approximate
        # sa = wa + d*0.5, sb = wb + d*0.5
        wa_real = sa - (total - (sa + sb)) * 0.5
        wb_real = sb - (total - (sa + sb)) * 0.5
        # simpler: just use win rate
        wra = (sa - sb + total) / (2 * total) if total > 0 else 0.5
        wr_mat[sti[s1], sti[s2]] = wra
    for ii in range(n):
        for jj in range(ii + 1, n):
            if not np.isnan(wr_mat[ii, jj]):
                wr_mat[jj, ii] = 1.0 - wr_mat[ii, jj]
    im = ax.imshow(wr_mat, cmap=plt.cm.RdYlBu_r, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(steps_list, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(steps_list, fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Step")
    ax.set_title("Win Rate Heatmap")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "elo_heatmap_5steps.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nPlot saved: {out_path}")
