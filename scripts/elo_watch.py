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

BOARD_SIZE = 15; N_CELLS = 225
CACHE_DIR = "elo_caches"
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
    parser.add_argument("--batch", type=int, default=256)
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

    print(f"Watching {len(args.watch_dir)} dirs, {gpp} games/pair, batch={args.batch}")
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

    while True:
        # Build per-checkpoint missing pairs for uniform coverage
        # intra_ckpt: {d: {ckpt: [(opp_ckpt, a, b), ...]}}
        intra_ckpt = {}
        for d in args.watch_dir:
            cache = load_cache(cache_path(d))
            d_ckpts = {}
            for a, b in missing_pairs(d, cache, gpp):
                d_ckpts.setdefault(a, []).append(("intra", d, d, a, b))
                d_ckpts.setdefault(b, []).append(("intra", d, d, a, b))
            if d_ckpts:
                intra_ckpt[d] = d_ckpts

        # Collect all (dir, ckpt) keys for uniform sampling
        all_ckpt_keys = []
        for d, d_ckpts in intra_ckpt.items():
            for ckpt in d_ckpts:
                all_ckpt_keys.append((d, ckpt))

        # Pick: 70% intra, 30% cross
        pick_cross = False
        cross_cache = load_cache(CROSS_CACHE)
        if len(args.watch_dir) >= 2 and random.random() < 0.3:
            pick_cross = True

        if (not pick_cross or not all_ckpt_keys) and all_ckpt_keys:
            # Intra: pick random checkpoint, then random opponent
            d, ckpt = random.choice(all_ckpt_keys)
            choice = random.choice(intra_ckpt[d][ckpt])
        elif pick_cross:
            # Cross: pick two random experiments, then one ckpt from each
            dirs_with = [d_ for d_ in args.watch_dir if all_ckpts(d_)]
            if len(dirs_with) >= 2:
                da, db = random.sample(dirs_with, 2)
                ca, cb = all_ckpts(da), all_ckpts(db)
                if ca and cb:
                    a, b = random.choice(ca), random.choice(cb)
                    key = cross_key(da, a, db, b)
                    if key in cross_cache and match_ok(cross_cache.get(key, []), gpp):
                        choice = None  # already done, fall through
                    else:
                        choice = ("cross", da, db, a, b)
                else:
                    choice = None
            else:
                choice = None
        else:
            choice = None

        if choice is None:
            print_stats()
            time.sleep(args.interval)
            try: plot_all(args.watch_dir, gpp)
            except Exception as e: print(f"  Plot error: {e}")
            continue

        kind, da, db, a_name, b_name = choice
        if kind == "intra":
            cache = load_cache(cache_path(da))
            key = cache_key(a_name, b_name)
            tag = os.path.basename(da.rstrip("/"))
        else:
            cache = cross_cache
            key = cross_key(da, a_name, db, b_name)
            tag = f"{os.path.basename(da.rstrip('/'))} vs {os.path.basename(db.rstrip('/'))}"
        if key in cache and match_ok(cache.get(key, []), gpp): continue

        model_a = get_model(da, a_name)
        model_b = get_model(db, b_name)
        t0 = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {a_name} vs {b_name} ...",
              end=" ", flush=True)

        wa, wb, dg = play_match(model_a, model_b, gpp, args.batch, device)
        cache[key] = (wa, wb, dg, gpp)
        save_cache(cache_path(da) if kind == "intra" else CROSS_CACHE, cache)
        wr = wa/(wa+wb) if wa+wb>0 else 0.5
        dt = time.perf_counter() - t0
        print(f"{wa}-{wb} (D={dg}) WR={wr:.2%}  [{dt:.1f}s]")

        print_stats()
        try: plot_all(args.watch_dir, gpp)
        except Exception as e: print(f"  Plot error: {e}")


if __name__ == "__main__":
    main()
