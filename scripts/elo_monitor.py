#!/usr/bin/env python3
"""ELO monitor — watches for new checkpoints, plays matches, updates ELO plot."""

import torch, sys, os, time, json, numpy as np, argparse, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Config ─────────────────────────────────────────────────────
ELO_G = 256       # games per pair (128 black + 128 white)
ELO_M = 4         # leaves per round (small for speed)
ELO_S = 1         # rounds → 4 total sims (policy-weighted, near-zero search)
WATCH_DIR = "checkpoints/train_loop"
CACHE_FILE = "output/elo_cache.json"
PLOT_FILE = "output/elo_curve.png"
HEATMAP_FILE = "output/elo_heatmap.png"
POLL_INTERVAL = 30  # seconds between checks

# ─── ELO Computation ────────────────────────────────────────────

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


# ─── Game Engine ────────────────────────────────────────────────

@torch.inference_mode()
def play_match(model_a, model_b):
    G_ = ELO_G
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    def make_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = ELO_M
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
        p0[g, first_acts[g]] = True  # first move is always black (player 0)

    fa_t = torch.tensor(first_acts, dtype=torch.long, device=DEVICE).unsqueeze(1)
    plr_t = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    slots_all = list(range(G_))
    model_a.prefill(fa_t, plr_t, kva, slots_all)
    model_b.prefill(fa_t, plr_t, kvb, slots_all)
    occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first_acts[g]))
        occ = np.zeros(225, dtype=bool); occ[int(first_acts[g])] = True
        if r: finished[g] = True; winners[g] = r
        else:
            bp0 = occ  # first move is always black (player 0)
            bp1 = np.zeros(225, dtype=bool)
            mgr_a.apply_move(g, int(first_acts[g]), bp0, bp1)
            mgr_b.apply_move(g, int(first_acts[g]), bp0, bp1)

    for move in range(1, 200):
        active = np.where(~finished)[0]
        if len(active) == 0: break
        cp = move % 2

        for mgr, mdl, kv in [(mgr_a, model_a, kva), (mgr_b, model_b, kvb)]:
            st = torch.from_numpy(active).to(DEVICE)
            dp = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(active), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(occ_gpu[active], -1e9); torch.cuda.synchronize()
            mgr.expand_roots(active.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
            for _ in range(ELO_S):
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
            else: p1[g, a] = True
            occ_gpu[g, a] = True

        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.full((len(active),), cp, dtype=torch.long, device=DEVICE)
        dec_slots = torch.from_numpy(active).to(DEVICE)
        model_a.decode(dec_pos, dec_plr, kva, dec_slots)
        model_b.decode(dec_pos, dec_plr, kvb, dec_slots)

        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; winners[g] = r
            else:
                mgr_a.apply_move(g, int(new_actions[i]), p0[g], p1[g])
                mgr_b.apply_move(g, int(new_actions[i]), p0[g], p1[g])

    wins_a = wins_b = draws = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wins_a += 1 if a_black[g] else 0; wins_b += 0 if a_black[g] else 1
        elif w == 2: wins_a += 0 if a_black[g] else 1; wins_b += 1 if a_black[g] else 0
        else: draws += 1

    del kva, kvb; torch.cuda.empty_cache()
    return wins_a, wins_b, draws


# ─── Cache + Plot ───────────────────────────────────────────────

def safe_load_ckpt(path, map_location, max_retries=3, delay=1.0):
    """Load checkpoint with retry in case file is still being written."""
    for attempt in range(max_retries):
        try:
            return torch.load(path, map_location=map_location)
        except RuntimeError:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f: return json.load(f)
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE + ".tmp", "w") as f: json.dump(cache, f)
    os.replace(CACHE_FILE + ".tmp", CACHE_FILE)

def cache_key(a_name, b_name):
    sa = int(a_name.split("_")[1].split(".")[0])
    sb = int(b_name.split("_")[1].split(".")[0])
    return f"{min(sa, sb)}_{max(sa, sb)}"

def plot_elo(cache):
    match_results = []
    for key, val in cache.items():
        if key.startswith("_"): continue
        try:
            wa, wb, d = val
            s1, s2 = key.split("_")
            a_name = f"step_{int(s1):06d}.pt"
            b_name = f"step_{int(s2):06d}.pt"
            score_a = wa + d * 0.5; score_b = wb + d * 0.5
            match_results.append((a_name, b_name, score_a, score_b))
        except (ValueError, IndexError):
            pass

    if not match_results: return

    elo = compute_elo(match_results)
    # Sort by step number
    items = []
    for name, rating in elo.items():
        step = int(name.split("_")[1].split(".")[0])
        items.append((step, rating))
    items.sort()

    steps = [s for s, _ in items]; ratings = [r for _, r in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, ratings, ".-", color="C1", markersize=6, linewidth=2)
    ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step"); ax.set_ylabel("ELO Rating")
    ax.set_title(f"ELO Curve ({ELO_G} games/pair, {ELO_M*ELO_S} sims)")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)
    plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=120); plt.close()
    return elo


def plot_heatmap(cache):
    """Plot win-rate heatmap for all checkpoint pairs."""
    # Collect all step numbers
    steps_set = set()
    for key, val in cache.items():
        if key.startswith("_"): continue
        try:
            s1, s2 = key.split("_")
            steps_set.add(int(s1)); steps_set.add(int(s2))
        except (ValueError, IndexError):
            pass
    if len(steps_set) < 2: return
    steps = sorted(steps_set)
    n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}

    # Build win-rate matrix: row beats column
    wr = np.full((n, n), np.nan)
    for key, val in cache.items():
        if key.startswith("_"): continue
        try:
            wa, wb, d = val
            s1, s2 = key.split("_")
            i1, i2 = int(s1), int(s2)
            total = wa + wb
            if total == 0: continue
            wr[step_to_idx[i1], step_to_idx[i2]] = wa / total
        except (ValueError, IndexError):
            pass
    # Above only filled one side (i1 < i2). Fill the other side.
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(wr[i, j]):
                wr[j, i] = 1.0 - wr[i, j]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(wr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Label every few steps
    tick_step = max(1, n // 20)
    tick_idx = list(range(0, n, tick_step))
    tick_labels = [str(steps[i]) for i in tick_idx]
    ax.set_xticks(tick_idx); ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Step"); ax.set_ylabel("Step")
    ax.set_title(f"Win Rate Heatmap (row vs col, {ELO_G} games/pair)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Win Rate")

    os.makedirs(os.path.dirname(HEATMAP_FILE), exist_ok=True)
    plt.tight_layout(); plt.savefig(HEATMAP_FILE, dpi=120); plt.close()


# ─── Main Loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games_per_pair", type=int, default=256)
    args = parser.parse_args()
    global DEVICE, ELO_G
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ELO_G = args.games_per_pair

    print(f"ELO Monitor: watching {WATCH_DIR}/")
    print(f"Config: G={ELO_G} M={ELO_M} S={ELO_S} ({ELO_M*ELO_S} sims)")
    print(f"Cache: {CACHE_FILE}")
    print(f"Plot: {PLOT_FILE}")
    print()

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    cache = load_cache()
    n_prev = 0

    while True:
        ckpts = sorted(glob.glob(os.path.join(WATCH_DIR, "step_*.pt")))
        ckpt_names = [os.path.basename(c) for c in ckpts]

        if len(ckpts) != n_prev:
            print(f"\n[{time.strftime('%H:%M:%S')}] {len(ckpts)} checkpoints")
            n_prev = len(ckpts)

        # Find all missing pairs among existing checkpoints
        missing = []
        for i, name_a in enumerate(ckpt_names):
            for name_b in ckpt_names[i + 1:]:
                key = cache_key(name_a, name_b)
                if key not in cache:
                    missing.append((name_a, name_b))

        if missing:
            print(f"  {len(missing)} missing pairs to play")
            elo = None
            for name_a, name_b in missing:
                key = cache_key(name_a, name_b)
                print(f"  {name_a} vs {name_b} ...", end=" ", flush=True)
                t0 = time.perf_counter()

                model_a = GomokuTransformer(cfg).to(DEVICE).eval()
                model_a.load_state_dict(safe_load_ckpt(os.path.join(WATCH_DIR, name_a), DEVICE))
                model_b = GomokuTransformer(cfg).to(DEVICE).eval()
                model_b.load_state_dict(safe_load_ckpt(os.path.join(WATCH_DIR, name_b), DEVICE))

                wa, wb, d = play_match(model_a, model_b)
                dt = time.perf_counter() - t0
                wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5
                cache[key] = [wa, wb, d]
                save_cache(cache)
                print(f"{wa}-{wb} D={d} WR={wr:.2%} ({dt:.0f}s)")

                del model_a, model_b; torch.cuda.empty_cache()

                # Update plots after each pair
                elo = plot_elo(cache)
                plot_heatmap(cache)

            if elo:
                items = sorted([(int(k.split("_")[1].split(".")[0]), v) for k, v in elo.items()])
                latest_elo = items[-1][1] if items else 1500
                print(f"  ELO range: {min(items)[1]:.0f} - {max(items)[1]:.0f}, latest: {latest_elo:.0f}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] waiting... ({len(ckpts)} checkpoints, 0 missing)", flush=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
