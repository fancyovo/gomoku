#!/usr/bin/env python3
"""Continuous multi-experiment ELO monitor with random pair sampling.

Uses full-sequence forward passes for simplicity. The model processes
the growing move history each step — O(L²) per game. For ~50-move games
this is fast enough on a 3060 for 128 games/pair (~3s/pair).
"""

import argparse, json, math, os, random, sys, time
import torch, numpy as np, gomoku_cpp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15; N_CELLS = 225; PAGE = 32
CACHE_DIR = "elo_caches"
MAX_MOVES = 80  # most games end by 50-60; cap KV cache at 80 steps
PLOT_FILE = "output/elo_curve.png"
MAX_MOVES = 180  # hard cap


# ── ELO solver ──────────────────────────────────────────────

def compute_elo(match_results):
    names = set();
    for a, b, _, _ in match_results: names.add(a); names.add(b)
    names = sorted(names)
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(200):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n = sa + sb
            if n == 0: continue
            da = (sa - ea * n) * (32.0 / n)
            elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo


# ── Game engine (full-sequence, no KV cache) ────────────────

def load_model(cfg, path, device):
    model = GomokuTransformer(cfg).to(device).eval()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def play_batch(model_a, model_b, b, a_black, device):
    """Play b games using per-model KV caches. Full batch parallelism.

    Both models decode ALL games every step (full batch b, not split).
    For games where a model doesn't play this turn, its decode output is
    discarded but its cache stays synchronized with the full move history.
    """
    occupied = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    cache_a = model_a.create_cache(max_games=b, max_cache_len=MAX_MOVES)
    cache_b = model_b.create_cache(max_games=b, max_cache_len=MAX_MOVES)

    pool = gomoku_cpp.GamePool(b)
    PAGE = 32
    all_done  = torch.zeros(b, dtype=torch.bool, device=device)
    winners   = torch.zeros(b, dtype=torch.long, device=device)
    accum_acts = torch.zeros(b, PAGE, dtype=torch.long, device=device)
    accum_cnt  = 0
    idx_b = torch.arange(b, device=device)

    def flush_cpp():
        nonlocal accum_cnt
        if accum_cnt == 0: return
        acts_cpu = accum_acts[:, :accum_cnt].cpu().numpy().astype(np.int32)
        for pg_start in range(0, accum_cnt, PAGE):
            pg_end = min(pg_start + PAGE, accum_cnt)
            pg_acts = np.zeros((b, PAGE), dtype=np.int32)
            pg_acts[:, :pg_end - pg_start] = acts_cpu[:, pg_start:pg_end]
            results = pool.execute_block(np.arange(b, dtype=np.int32), pg_acts)
            for i in range(b):
                if all_done[i]: continue
                es = int(results[i, 0])
                if es >= 0:
                    all_done[i] = True
                    winners[i] = int(results[i, 1])
        accum_cnt = 0

    # ── Step 0: first move (black) from first_move_logits ──
    fm_a = model_a.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_b = model_b.first_move_logits.unsqueeze(0).expand(b, -1)
    fm = torch.where(a_black.unsqueeze(1), fm_a, fm_b)
    fm = fm.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm.float(), dim=-1)
    first = torch.multinomial(probs, 1).squeeze(-1)
    occupied[idx_b, first] = True
    accum_acts[:, 0] = first; accum_cnt = 1
    last_act = first
    last_plr = torch.zeros(b, dtype=torch.long, device=device)

    # ── Main loop: each step, both models decode the FULL batch ──
    for step in range(1, MAX_MOVES):
        cp = step % 2

        # Both models decode the last action (full batch parallelism!)
        logits_a = model_a.decode(last_act, last_plr, cache_a, idx_b)
        logits_b = model_b.decode(last_act, last_plr, cache_b, idx_b)

        # Use the appropriate model's logits for the current player
        if cp == 0:  # black's turn
            logits = torch.where(a_black.unsqueeze(1), logits_a, logits_b)
        else:  # white's turn
            logits = torch.where(a_black.unsqueeze(1), logits_b, logits_a)

        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        occupied.scatter_(1, actions.unsqueeze(1), True)

        last_act = actions
        last_plr = torch.full((b,), cp, dtype=torch.long, device=device)

        accum_acts[:, accum_cnt] = actions
        accum_cnt += 1
        if accum_cnt == PAGE:
            flush_cpp()
            if all_done.all(): break

    flush_cpp()
    winners[(winners == 0) & ~all_done] = 3

    wins_a = wins_b = draws = 0
    for i in range(b):
        w = winners[i].item()
        if w == 1:
            wins_a += 1 if a_black[i] else 0; wins_b += 0 if a_black[i] else 1
        elif w == 2:
            wins_a += 0 if a_black[i] else 1; wins_b += 1 if a_black[i] else 0
        else: draws += 1
    return wins_a, wins_b, draws


def play_match(model_a, model_b, n_games, batch_size, device):
    wins_a = wins_b = draws = 0
    for start in range(0, n_games, batch_size):
        b = min(batch_size, n_games - start)
        a_black = torch.tensor([(start + i) % 2 == 0 for i in range(b)], device=device)
        wa, wb, d = play_batch(model_a, model_b, b, a_black, device)
        wins_a += wa; wins_b += wb; draws += d
    return wins_a, wins_b, draws


# ── Cache ───────────────────────────────────────────────────

def cache_path(watch_dir):
    exp = os.path.basename(watch_dir.rstrip("/"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{exp}.json")

def load_cache(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {"_meta": {"games_per_pair": 0}}

def save_cache(path, cache):
    with open(path, "w") as f: json.dump(cache, f)

def cache_key(a_name, b_name):
    sa = int(a_name.split("_")[1].split(".")[0])
    sb = int(b_name.split("_")[1].split(".")[0])
    return f"{min(sa,sb)}_{max(sa,sb)}"

def match_ok(val, games_per_pair):
    if isinstance(val, list) and len(val) >= 4:
        return val[3] == games_per_pair
    return False

def cross_key(exp_a, a_name, exp_b, b_name):
    sa = int(a_name.split("_")[1].split(".")[0])
    sb = int(b_name.split("_")[1].split(".")[0])
    return f"{exp_a}:{sa}|{exp_b}:{sb}"

CROSS_CACHE = os.path.join(CACHE_DIR, "_cross.json")


# ── Plot ────────────────────────────────────────────────────

def plot_all(watch_dirs, games_per_pair):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return

    try:
        all_results = []
        for d in watch_dirs:
            exp = os.path.basename(d.rstrip("/"))
            cp = cache_path(d)
            if not os.path.exists(cp): continue
            cache = load_cache(cp)
            for key, val in cache.items():
                if key == "_meta": continue
                try:
                    wa, wb, d = val[0], val[1], val[2]
                    s1, s2 = key.split("_")
                    a = f"{exp}:step_{int(s1):06d}"
                    b = f"{exp}:step_{int(s2):06d}"
                    all_results.append((a, b, wa + d*0.5, wb + d*0.5))
                except (ValueError, IndexError): pass

        if os.path.exists(CROSS_CACHE):
            cc = load_cache(CROSS_CACHE)
            for key, val in cc.items():
                if key == "_meta": continue
                try:
                    wa, wb, d = val[0], val[1], val[2]
                    a_raw, b_raw = key.split("|")
                    ea, sa = a_raw.split(":")
                    eb, sb = b_raw.split(":")
                    a = f"{ea}:step_{int(sa):06d}"
                    b = f"{eb}:step_{int(sb):06d}"
                    all_results.append((a, b, wa + d*0.5, wb + d*0.5))
                except (ValueError, IndexError): pass

        joint_elo = compute_elo(all_results) if all_results else {}
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(watch_dirs), 1)))
        for di, d in enumerate(watch_dirs):
            exp = os.path.basename(d.rstrip("/"))
            exp_elo = {}
            for name, rating in joint_elo.items():
                if not name.startswith(exp + ":"): continue
                try:
                    step_str = name.split(":step_")[1]
                    exp_elo[int(step_str)] = rating
                except (ValueError, IndexError): pass
            if not exp_elo: continue
            steps = sorted(exp_elo.keys())
            ratings = [exp_elo[s] for s in steps]
            ax.plot(steps, ratings, ".-", color=colors[di], markersize=3,
                    linewidth=1.5, label=exp)
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Training Step"); ax.set_ylabel("ELO Rating")
        ax.set_title(f"ELO Comparison ({games_per_pair} games/pair, joint calibration)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        os.makedirs("output", exist_ok=True)
        plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=120); plt.close()
    except Exception as e:
        print(f"  Plot error: {e}")


# ── Pair enumeration ─────────────────────────────────────────

def all_ckpts(watch_dir):
    if not os.path.isdir(watch_dir): return []
    return sorted([f for f in os.listdir(watch_dir) if f.endswith(".pt")],
                  key=lambda x: int(x.split("_")[1].split(".")[0]))

def missing_pairs(watch_dir, cache, games_per_pair):
    ckpts = all_ckpts(watch_dir)
    missing = []
    for i, a in enumerate(ckpts):
        for b in ckpts[i+1:]:
            key = cache_key(a, b)
            if key not in cache or not match_ok(cache.get(key, []), games_per_pair):
                missing.append((a, b))
    return missing


# ── Main loop ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch_dir", type=str, action="append", default=[])
    parser.add_argument("--games_per_pair", type=int, default=128)
    parser.add_argument("--n_models", type=int, default=5,
                        help="Number of models to load per round (multi-model batching)")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f: cfg = yaml.safe_load(f)
    model_cfg_raw = cfg["model"]
    if args.model_config:
        with open(args.model_config) as f: mc = yaml.safe_load(f)
        model_cfg_raw = mc["model"]
    model_cfg = ModelConfig.from_dict(model_cfg_raw)
    device = torch.device(args.device)

    if not args.watch_dir:
        args.watch_dir = ["checkpoints/base"]

    gpp = args.games_per_pair
    for d in args.watch_dir:
        cp = cache_path(d)
        cache = load_cache(cp)
        cache["_meta"]["games_per_pair"] = gpp
        save_cache(cp, cache)

    print(f"Watching {len(args.watch_dir)} dirs, {gpp} games/pair, n_models={args.n_models}")
    for d in args.watch_dir: print(f"  {d}")
    print(f"Model: d_model={model_cfg_raw['d_model']}, n_layers={model_cfg_raw['n_layers']}\n")

    # Per-experiment model configs
    exp_cfgs = {}
    if os.path.exists("configs/abl_scale_up.yaml"):
        with open("configs/abl_scale_up.yaml") as f:
            exp_cfgs["scale_up"] = ModelConfig.from_dict(yaml.safe_load(f)["model"])

    def get_model_cfg(dir_path):
        exp = os.path.basename(dir_path.rstrip("/"))
        return exp_cfgs.get(exp, model_cfg)

    model_cache = {}
    def get_model(dir_path, ckpt_name):
        key = (dir_path, ckpt_name)
        if key not in model_cache:
            path = os.path.join(dir_path, ckpt_name)
            model_cache[key] = load_model(get_model_cfg(dir_path), path, device)
            if len(model_cache) > 32:
                del model_cache[list(model_cache.keys())[0]]
                torch.cuda.empty_cache()
        return model_cache[key]

    last_stats_time = 0.0
    def print_stats():
        nonlocal last_stats_time
        now = time.time()
        if now - last_stats_time < 30.0: return
        last_stats_time = now
        print(f"\n  {'─'*55}")
        print(f"  {'Experiment':<20s} {'CKPTs':>6s} {'Matched':>8s} {'Total':>8s} {'Done':>7s}")
        print(f"  {'─'*55}")
        total_intra_m, total_intra_t = 0, 0
        for d in args.watch_dir:
            exp = os.path.basename(d.rstrip("/"))
            ckpts = all_ckpts(d); n = len(ckpts)
            tp = n*(n-1)//2 if n >= 2 else 0
            cache = load_cache(cache_path(d))
            m = sum(1 for k in cache if k != "_meta" and match_ok(cache[k], gpp))
            print(f"  {exp:<20s} {n:6d} {m:8d} {tp:8d} {f'{100*m/tp:.0f}%' if tp>0 else '-':>7s}")
            total_intra_m += m; total_intra_t += tp
        cc = load_cache(CROSS_CACHE)
        cm = sum(1 for k in cc if k != "_meta" and match_ok(cc[k], gpp))
        ct = sum(len(all_ckpts(da))*len(all_ckpts(db))
                 for i, da in enumerate(args.watch_dir)
                 for db in args.watch_dir[i+1:])
        print(f"  {'cross':<20s} {'-':>6s} {cm:8d} {ct:8d} {f'{100*cm/ct:.0f}%' if ct>0 else '-':>7s}")
        om, ot = total_intra_m + cm, total_intra_t + ct
        print(f"  {'TOTAL':<20s} {'-':>6s} {om:8d} {ot:8d} {f'{100*om/ot:.0f}%' if ot>0 else '-':>7s}")
        print(f"  {'─'*55}\n")

    # ── Multi-model batched play ────────────────────────────────

    @torch.inference_mode()
    def play_multi_batch(loaded_models, pair_list, games_per_dir, device):
        """Play all given pairs in one batch. loaded_models: list of (name, model).
        pair_list: [(a_idx, b_idx)] — each pair gets games_per_dir games (half a=black).
        Returns: {(a_name, b_name): (wa, wb, dg)}.
        """
        total_games = 0
        game_info = []
        black_m_idx = []; white_m_idx = []
        for a_idx, b_idx in pair_list:
            for g in range(games_per_dir):
                is_a_black = (g % 2 == 0)
                black_m_idx.append(a_idx if is_a_black else b_idx)
                white_m_idx.append(b_idx if is_a_black else a_idx)
                game_info.append((a_idx, b_idx, g))
                total_games += 1
        b = total_games
        if b == 0: return {}

        M = len(loaded_models)
        caches = [loaded_models[mi][1].create_cache(max_games=b, max_cache_len=MAX_MOVES) for mi in range(M)]
        black_m = torch.tensor(black_m_idx, device=device)
        white_m = torch.tensor(white_m_idx, device=device)
        occupied = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
        pool = gomoku_cpp.GamePool(b)
        all_done = torch.zeros(b, dtype=torch.bool, device=device)
        winners  = torch.zeros(b, dtype=torch.long, device=device)
        accum_acts = torch.zeros(b, PAGE, dtype=torch.long, device=device)
        accum_cnt = 0
        idx_b = torch.arange(b, device=device)

        def flush():
            nonlocal accum_cnt
            if accum_cnt == 0: return
            acts_cpu = accum_acts[:, :accum_cnt].cpu().numpy().astype(np.int32)
            for pg_start in range(0, accum_cnt, PAGE):
                pg_end = min(pg_start + PAGE, accum_cnt)
                pg_acts = np.zeros((b, PAGE), dtype=np.int32)
                pg_acts[:, :pg_end - pg_start] = acts_cpu[:, pg_start:pg_end]
                results = pool.execute_block(np.arange(b, dtype=np.int32), pg_acts)
                for i in range(b):
                    if all_done[i]: continue
                    es = int(results[i, 0])
                    if es >= 0:
                        all_done[i] = True
                        winners[i] = int(results[i, 1])
            accum_cnt = 0

        # First move
        fm_all = torch.stack([m[1].first_move_logits for m in loaded_models])
        fm_buf = fm_all[black_m]
        fm_buf = fm_buf.masked_fill(occupied, -1e9)
        probs = torch.softmax(fm_buf.float(), dim=-1)
        first = torch.multinomial(probs, 1).squeeze(-1)
        occupied[idx_b, first] = True
        accum_acts[:, 0] = first; accum_cnt = 1
        last_act = first
        last_plr = torch.zeros(b, dtype=torch.long, device=device)

        for step in range(1, MAX_MOVES):
            cp = step % 2
            all_logits = torch.zeros(M, b, N_CELLS, device=device)
            for mi in range(M):
                all_logits[mi] = loaded_models[mi][1].decode(
                    last_act, last_plr, caches[mi], idx_b)

            model_for_game = black_m if cp == 0 else white_m
            logits = all_logits[model_for_game, idx_b]
            logits = logits.masked_fill(occupied, -1e9)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            occupied.scatter_(1, actions.unsqueeze(1), True)
            last_act = actions
            last_plr = torch.full((b,), cp, dtype=torch.long, device=device)

            accum_acts[:, accum_cnt] = actions
            accum_cnt += 1
            if accum_cnt == PAGE:
                flush()
                if all_done.all(): break
        flush()
        winners[(winners == 0) & ~all_done] = 3

        # Aggregate per pair
        results = {}
        for pi, (a_idx, b_idx) in enumerate(pair_list):
            wa = wb = dg = 0
            for g in range(games_per_dir):
                gi = pi * games_per_dir + g
                w = winners[gi].item()
                is_a_black = (g % 2 == 0)
                if w == 1:
                    wa += 1 if is_a_black else 0; wb += 0 if is_a_black else 1
                elif w == 2:
                    wa += 0 if is_a_black else 1; wb += 1 if is_a_black else 0
                else: dg += 1
            a_name = loaded_models[a_idx][0]
            b_name = loaded_models[b_idx][0]
            results[(a_name, b_name)] = (wa, wb, dg)

        # Free KV caches immediately — iterate over all layers
        for cache in caches:
            for k in cache.k: del k
            for v in cache.v: del v
            cache.k.clear(); cache.v.clear()
        del caches
        torch.cuda.empty_cache()
        return results

    # ── Main sampling loop ─────────────────────────────────────

    # How many models to load per round
    ACTIVE_MODELS = args.n_models

    round_num = 0
    while True:
        # Collect all (dir, ckpt) that have unplayed pairs
        ckpt_missing = {}  # (dir, ckpt) → count of missing pairs
        for d in args.watch_dir:
            cache = load_cache(cache_path(d))
            ckpt_count = {}
            for a, b in missing_pairs(d, cache, gpp):
                ckpt_count[a] = ckpt_count.get(a, 0) + 1
                ckpt_count[b] = ckpt_count.get(b, 0) + 1
            for ckpt, cnt in ckpt_count.items():
                ckpt_missing[(d, ckpt)] = cnt

        if len(ckpt_missing) < 2:
            print_stats()
            time.sleep(args.interval)
            try: plot_all(args.watch_dir, gpp)
            except Exception as e: print(f"  Plot error: {e}")
            continue

        # Pick ACTIVE_MODELS checkpoints: 3 weighted + 2 uniform
        keys = list(ckpt_missing.keys())
        n_pick = min(ACTIVE_MODELS, len(keys))
        n_weighted = min(3, n_pick)
        n_uniform = n_pick - n_weighted

        # Weighted picks (favor sparse checkpoints)
        weights = [ckpt_missing[k] for k in keys]
        chosen = list(random.choices(keys, weights=weights, k=n_weighted))
        # Deduplicate
        chosen = list(dict.fromkeys(chosen))

        # Uniform picks from remaining (ensure everyone gets a chance)
        remaining = [k for k in keys if k not in chosen]
        for _ in range(n_uniform):
            if not remaining: break
            extra = random.choice(remaining)
            chosen.append(extra)
            remaining.remove(extra)

        if len(chosen) < 2:
            time.sleep(args.interval)
            continue

        # Load models
        round_num += 1
        t_load = time.perf_counter()
        loaded = []
        for d, ckpt in chosen:
            path = os.path.join(d, ckpt)
            exp = os.path.basename(d.rstrip("/"))
            cfg = get_model_cfg(d)
            m = GomokuTransformer(cfg).to(device).eval()
            m.load_state_dict(torch.load(path, map_location=device))
            loaded.append((f"{exp}:{ckpt}", m))
        load_time = time.perf_counter() - t_load

        # Build all pairs among chosen models
        pair_list = [(i, j) for i in range(len(loaded)) for j in range(len(loaded)) if i < j]
        n_pairs = len(pair_list)
        total_games = n_pairs * gpp

        exp_names = list(set(os.path.basename(d.rstrip("/")) for d, _ in chosen))
        tag = f"Round {round_num}: {len(chosen)} models"
        if len(exp_names) <= 3:
            tag += f" [{','.join(exp_names)}]"
        print(f"\n[{time.strftime('%H:%M:%S')}] {tag} — {n_pairs} pairs × {gpp} games "
              f"(load {load_time:.1f}s) ...", end=" ", flush=True)

        t0 = time.perf_counter()
        results = play_multi_batch(loaded, pair_list, gpp, device)
        dt = time.perf_counter() - t0

        # Save results
        n_new = 0
        for pi, (i, j) in enumerate(pair_list):
            a_name, b_name = loaded[i][0], loaded[j][0]
            wa, wb, dg = results[(a_name, b_name)]
            # Determine which cache to write to
            a_exp, a_ckpt = a_name.split(":", 1)
            b_exp, b_ckpt = b_name.split(":", 1)
            if a_exp == b_exp:
                # Intra-experiment
                d = next(d_ for d_ in args.watch_dir if os.path.basename(d_.rstrip("/")) == a_exp)
                cache = load_cache(cache_path(d))
                key = cache_key(a_ckpt, b_ckpt)
                if key not in cache or not match_ok(cache.get(key, []), gpp):
                    cache[key] = (wa, wb, dg, gpp)
                    save_cache(cache_path(d), cache)
                    n_new += 1
            else:
                # Cross-experiment
                cross_cache = load_cache(CROSS_CACHE)
                key = cross_key(a_exp, a_ckpt, b_exp, b_ckpt)
                if key not in cross_cache or not match_ok(cross_cache.get(key, []), gpp):
                    cross_cache[key] = (wa, wb, dg, gpp)
                    save_cache(CROSS_CACHE, cross_cache)
                    n_new += 1

        total_wr = sum(wa/(wa+wb) if wa+wb>0 else 0.5 for wa,wb,_ in results.values()) / n_pairs
        print(f"{total_games} games in {dt:.1f}s ({total_games/dt:.0f} games/s), "
              f"{n_new} new pairs saved")

        # Free GPU memory
        del loaded
        torch.cuda.empty_cache()

        print_stats()
        try: plot_all(args.watch_dir, gpp)
        except Exception as e: print(f"  Plot error: {e}")


if __name__ == "__main__":
    main()
