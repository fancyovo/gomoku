#!/usr/bin/env python3
"""ELO fast — pure policy head sampling, no MCTS. Separate cache/plots from elo_monitor."""

import torch, sys, os, time, json, numpy as np, argparse, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ELO_G = 256       # games per pair (128 black + 128 white)
WATCH_DIR = "checkpoints/train_loop"
CACHE_FILE = "output/elo_fast_cache.json"
PLOT_FILE = "output/elo_fast_curve.png"
HEATMAP_FILE = "output/elo_fast_heatmap.png"
POLL_INTERVAL = 30


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
def play_match_policy(model_a, model_b):
    """Pure policy head — no MCTS. Key: reuse decode output as next-step policy
       to avoid redundant evaluate_mcts_leaves calls (2x forward instead of 4x)."""
    G_ = ELO_G
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    a_black = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    finished = np.zeros(G_, dtype=bool); winners = np.zeros(G_, dtype=np.int32)

    fa_a = model_a.sample_first_moves(G_, DEVICE)
    fa_b = model_b.sample_first_moves(G_, DEVICE)
    first_acts = np.zeros(G_, dtype=np.int64)

    # occ_gpu on GPU from the start — single source of truth for board occupancy
    occ_gpu = torch.zeros(G_, 225, dtype=torch.bool, device=DEVICE)
    for g in range(G_):
        first_acts[g] = int(fa_a[g].item()) if a_black[g] else int(fa_b[g].item())
        occ_gpu[g, first_acts[g]] = True

    kva = model_a.create_cache(max_games=G_, max_cache_len=250)
    kvb = model_b.create_cache(max_games=G_, max_cache_len=250)

    fa_t = torch.tensor(first_acts, dtype=torch.long, device=DEVICE).unsqueeze(1)
    plr_t = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    pol_a_full = torch.zeros(G_, 225, device=DEVICE)
    pol_b_full = torch.zeros(G_, 225, device=DEVICE)
    raw_a, _ = model_a.prefill(fa_t, plr_t, kva, list(range(G_)))
    raw_b, _ = model_b.prefill(fa_t, plr_t, kvb, list(range(G_)))
    pol_a_full[:] = raw_a; pol_b_full[:] = raw_b

    a_black_t = torch.from_numpy(a_black).to(DEVICE)
    finished_t = torch.zeros(G_, dtype=torch.bool, device=DEVICE)

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first_acts[g]))
        if r:
            finished[g] = True; winners[g] = r
            finished_t[g] = True

    for move in range(1, 200):
        active_t = torch.where(~finished_t)[0]
        if len(active_t) == 0: break
        cp = move % 2
        n_active = len(active_t)

        # Mask and sample — all on GPU
        pol_a_act = pol_a_full[active_t]
        pol_b_act = pol_b_full[active_t]
        occ_act = occ_gpu[active_t]
        pol_a_act = pol_a_act.masked_fill(occ_act, -1e9)
        pol_b_act = pol_b_act.masked_fill(occ_act, -1e9)
        probs_a = torch.softmax(pol_a_act, -1)
        probs_b = torch.softmax(pol_b_act, -1)
        use_a = ((cp == 0) == a_black_t[active_t])
        probs = torch.where(use_a.unsqueeze(1), probs_a, probs_b)
        new_acts_t = torch.multinomial(probs, 1).squeeze(-1)

        # Update occupied board ON GPU (scatter), sync to CPU only for gomoku_cpp.step
        occ_gpu[active_t, new_acts_t] = True
        new_actions = new_acts_t.cpu().numpy()
        active = active_t.cpu().numpy()

        # decode returns policy for NEXT move
        dec_plr = torch.full((n_active,), cp, dtype=torch.long, device=DEVICE)
        raw_a, _ = model_a.decode(new_acts_t, dec_plr, kva, active_t)
        raw_b, _ = model_b.decode(new_acts_t, dec_plr, kvb, active_t)
        pol_a_full[active_t] = raw_a
        pol_b_full[active_t] = raw_b

        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r:
                finished[g] = True; winners[g] = r
                finished_t[g] = True

    wins_a = wins_b = draws = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wins_a += 1 if a_black[g] else 0; wins_b += 0 if a_black[g] else 1
        elif w == 2: wins_a += 0 if a_black[g] else 1; wins_b += 1 if a_black[g] else 0
        else: draws += 1

    del kva, kvb; torch.cuda.empty_cache()
    return wins_a, wins_b, draws


def safe_load_ckpt(path, map_location, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            return torch.load(path, map_location=map_location)
        except RuntimeError:
            if attempt < max_retries - 1: time.sleep(delay)
            else: raise


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
        except (ValueError, IndexError): pass
    if not match_results: return None
    elo = compute_elo(match_results)
    items = sorted((int(n.split("_")[1].split(".")[0]), r) for n, r in elo.items())
    steps, ratings = zip(*items)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, ratings, ".-", color="C2", markersize=6, linewidth=2)
    ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step"); ax.set_ylabel("ELO Rating")
    ax.set_title(f"ELO Curve — Pure Policy (no MCTS, {ELO_G} games/pair)")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)
    plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=120); plt.close()
    return elo


def plot_heatmap(cache):
    steps_set = set()
    for key, val in cache.items():
        if key.startswith("_"): continue
        try:
            s1, s2 = key.split("_")
            steps_set.add(int(s1)); steps_set.add(int(s2))
        except (ValueError, IndexError): pass
    if len(steps_set) < 2: return
    steps = sorted(steps_set); n = len(steps)
    step_to_idx = {s: i for i, s in enumerate(steps)}
    wr = np.full((n, n), np.nan)
    for key, val in cache.items():
        if key.startswith("_"): continue
        try:
            wa, wb, d = val; s1, s2 = key.split("_")
            i1, i2 = int(s1), int(s2)
            total = wa + wb
            if total > 0: wr[step_to_idx[i1], step_to_idx[i2]] = wa / total
        except (ValueError, IndexError): pass
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(wr[i, j]): wr[j, i] = 1.0 - wr[i, j]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(wr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    tick_step = max(1, n // 20)
    tick_idx = list(range(0, n, tick_step))
    tick_labels = [str(steps[i]) for i in tick_idx]
    ax.set_xticks(tick_idx); ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Step"); ax.set_ylabel("Step")
    ax.set_title(f"Win Rate Heatmap — Pure Policy ({ELO_G} games/pair)")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Win Rate")
    os.makedirs(os.path.dirname(HEATMAP_FILE), exist_ok=True)
    plt.tight_layout(); plt.savefig(HEATMAP_FILE, dpi=120); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games_per_pair", type=int, default=256)
    args = parser.parse_args()
    global DEVICE, ELO_G
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ELO_G = args.games_per_pair

    print(f"ELO Fast (pure policy, no MCTS): watching {WATCH_DIR}/")
    print(f"Config: G={ELO_G} (no MCTS, policy head direct sampling)")
    print(f"Cache: {CACHE_FILE}")
    print(f"Plots: {PLOT_FILE} + {HEATMAP_FILE}")
    print()

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    cache = load_cache()
    n_prev = 0

    def gather_missing():
        ckpts = sorted(glob.glob(os.path.join(WATCH_DIR, "step_*.pt")))
        ckpt_names = [os.path.basename(c) for c in ckpts]
        missing = []
        for i, name_a in enumerate(ckpt_names):
            for name_b in ckpt_names[i + 1:]:
                key = cache_key(name_a, name_b)
                if key not in cache:
                    missing.append((name_a, name_b, key))
        return ckpt_names, missing

    ckpt_names, missing = gather_missing()
    n_prev = len(ckpt_names)
    if n_prev > 0:
        print(f"\n[{time.strftime('%H:%M:%S')}] {n_prev} checkpoints, {len(missing)} missing pairs")

    batch_count = 0
    while True:
        if not missing:
            print(f"[{time.strftime('%H:%M:%S')}] waiting... ({n_prev} checkpoints, 0 missing)", flush=True)
            time.sleep(POLL_INTERVAL)
            ckpt_names, missing = gather_missing()
            if len(ckpt_names) != n_prev:
                print(f"\n[{time.strftime('%H:%M:%S')}] {len(ckpt_names)} checkpoints ({len(missing)} missing)")
                n_prev = len(ckpt_names)
            continue

        # Randomly pick one pair
        import random
        idx = random.randrange(len(missing))
        name_a, name_b, key = missing.pop(idx)
        batch_count += 1

        print(f"  [{batch_count}] {name_a} vs {name_b} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        model_a = GomokuTransformer(cfg).to(DEVICE).eval()
        model_a.load_state_dict(safe_load_ckpt(os.path.join(WATCH_DIR, name_a), DEVICE))
        model_b = GomokuTransformer(cfg).to(DEVICE).eval()
        model_b.load_state_dict(safe_load_ckpt(os.path.join(WATCH_DIR, name_b), DEVICE))

        wa, wb, d = play_match_policy(model_a, model_b)
        dt = time.perf_counter() - t0
        wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5
        cache[key] = [wa, wb, d]
        save_cache(cache)
        print(f"{wa}-{wb} D={d} WR={wr:.2%} ({dt:.0f}s)")

        del model_a, model_b; torch.cuda.empty_cache()

        # Update plots: every 100 pairs or when all missing pairs are done
        if batch_count % 100 == 0 or not missing:
            plot_elo(cache)
            plot_heatmap(cache)
        # Re-scan for new checkpoints periodically or when exhausted
        if batch_count % 100 == 0 or not missing:
            new_names, new_missing = gather_missing()
            if len(new_names) != n_prev:
                print(f"  [{time.strftime('%H:%M:%S')}] {len(new_names)} checkpoints (+{len(new_names) - n_prev} new)")
                old_keys = set(k for _, _, k in missing)
                for na, nb, nk in new_missing:
                    if nk not in old_keys and nk not in cache:
                        missing.append((na, nb, nk))
                n_prev = len(new_names)
                import random as _random
                _random.shuffle(missing)


if __name__ == "__main__":
    main()
