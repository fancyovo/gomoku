#!/usr/bin/env python3
"""Continuous multi-experiment ELO monitor with random pair sampling.

Watches multiple checkpoint directories. Each cycle:
  1. Scan all dirs for new checkpoints
  2. If unplayed pairs exist, randomly pick one, play match, save to per-dir cache
  3. Recompute ELO for all experiments, plot multi-curve chart
  4. Repeat (idle if all pairs done)

Usage:
    python scripts/elo_watch.py --watch_dir checkpoints/base --watch_dir checkpoints/fixed_entropy
"""

import argparse
import json
import math
import os
import random
import sys
import time
import torch
import numpy as np
import gomoku_cpp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15; N_CELLS = 225
CACHE_DIR = "elo_caches"
PLOT_FILE = "output/elo_curve.png"


# ── ELO solver ──────────────────────────────────────────────

def compute_elo(match_results):
    names = set()
    for a, b, _, _ in match_results:
        names.add(a); names.add(b)
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


# ── Game engine ─────────────────────────────────────────────

def load_model(cfg, path, device):
    model = GomokuTransformer(cfg).to(device).eval()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def play_batch(model_a, model_b, b, a_black, device):
    # GPU state
    positions = torch.zeros(b, 0, dtype=torch.long, device=device)
    players   = torch.zeros(b, 0, dtype=torch.long, device=device)
    occupied  = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    active    = torch.ones(b, dtype=torch.bool, device=device)

    # C++ batch processor
    pool = gomoku_cpp.GamePool(b)
    PAGE = 32
    all_done = torch.zeros(b, dtype=torch.bool, device=device)  # already finished
    winners  = torch.zeros(b, dtype=torch.long, device=device)

    idx_batch = torch.arange(b, device=device)
    accum_acts = torch.zeros(b, PAGE, dtype=torch.int32)  # action buffer
    accum_cnt = 0

    # First move (black)
    fm_a = model_a.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_b = model_b.first_move_logits.unsqueeze(0).expand(b, -1)
    fm = torch.where(a_black.unsqueeze(1), fm_a, fm_b)
    fm = fm.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm.float(), dim=-1)
    first = torch.multinomial(probs, 1).squeeze(-1)
    occupied[idx_batch, first] = True
    positions = first.unsqueeze(1)
    players = torch.zeros(b, 1, dtype=torch.long, device=device)
    accum_acts[:, 0] = first.cpu().int()
    accum_cnt = 1

    def flush_cpp():
        nonlocal accum_cnt
        if accum_cnt == 0: return
        pad = accum_acts[:, :accum_cnt].cpu().numpy()  # (b, accum_cnt)
        # Flatten to (b * accum_cnt), send to C++ in pages of 32
        indices_np = np.arange(b, dtype=np.int32)
        for pg_start in range(0, accum_cnt, PAGE):
            pg_end = min(pg_start + PAGE, accum_cnt)
            pg_acts = np.zeros((b, PAGE), dtype=np.int32)
            pg_acts[:, :pg_end - pg_start] = pad[:, pg_start:pg_end]
            results = pool.execute_block(indices_np, pg_acts)
            for i in range(b):
                if all_done[i]: continue
                es = int(results[i, 0])
                if es >= 0:
                    all_done[i] = True
                    winners[i] = int(results[i, 1])
        accum_cnt = 0

    for step in range(1, N_CELLS):
        cp = step % 2
        need_a = ((cp == 0) & a_black) | ((cp == 1) & ~a_black)
        need_b = ~need_a
        logits = torch.zeros(b, N_CELLS, device=device)

        if need_a.any():
            ia = need_a.nonzero(as_tuple=True)[0]
            if ia.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model_a(positions[ia], players[ia])
                logits[ia] = out.float()[:, -1, :]

        if need_b.any():
            ib = need_b.nonzero(as_tuple=True)[0]
            if ib.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model_b(positions[ib], players[ib])
                logits[ib] = out.float()[:, -1, :]

        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)

        occupied[~all_done] = occupied[~all_done].scatter_(
            1, actions[~all_done].unsqueeze(1), True)
        positions = torch.cat([positions, actions.unsqueeze(1)], dim=1)
        players = torch.cat([players, torch.full((b, 1), cp, dtype=torch.long, device=device)], dim=1)

        accum_acts[:, accum_cnt] = actions.cpu().int()
        accum_cnt += 1
        if accum_cnt == PAGE:
            flush_cpp()
            if all_done.all(): break

    flush_cpp()
    # Any remaining active games are draws
    winners[(winners == 0) & ~all_done] = 3

    wins_a = wins_b = draws = 0
    for i in range(b):
        w = winners[i].item()
        if w == 1: wins_a += 1 if a_black[i] else 0; wins_b += 0 if a_black[i] else 1
        elif w == 2: wins_a += 0 if a_black[i] else 1; wins_b += 1 if a_black[i] else 0
        else: draws += 1
    return wins_a, wins_b, draws


def play_match(model_a, model_b, n_games, batch_size, device):
    wins_a = wins_b = draws = 0
    for start in range(0, n_games, batch_size):
        b = min(batch_size, n_games - start)
        a_black = torch.tensor([(start+i)%2==0 for i in range(b)], device=device)
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
    """Check cached result has matching game count (backward compat)."""
    if isinstance(val, list) and len(val) >= 4:
        return val[3] == games_per_pair
    return False  # old format without game count → recompute

def cross_key(exp_a, a_name, exp_b, b_name):
    sa = int(a_name.split("_")[1].split(".")[0])
    sb = int(b_name.split("_")[1].split(".")[0])
    return f"{exp_a}:{sa}|{exp_b}:{sb}"  # | cannot appear in path names

CROSS_CACHE = os.path.join(CACHE_DIR, "_cross.json")


# ── Plot ────────────────────────────────────────────────────

def plot_all(watch_dirs, games_per_pair):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return

    try:
        # Collect ALL match results (intra + cross) for joint ELO
        all_results = []
        for d in watch_dirs:
            exp = os.path.basename(d.rstrip("/"))
            cp = cache_path(d)
            if not os.path.exists(cp): continue
            cache = load_cache(cp)
            for key, val in cache.items():
                if key == "_meta": continue
                try:
                    wa, wb, d = val[0], val[1], val[2]  # first 3 elements, ignore game count
                    s1, s2 = key.split("_")
                    a = f"{exp}:step_{int(s1):06d}"
                    b = f"{exp}:step_{int(s2):06d}"
                    all_results.append((a, b, wa + d*0.5, wb + d*0.5))
                except (ValueError, IndexError):
                    pass  # skip malformed entries

        # Add cross-experiment results
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
                except (ValueError, IndexError):
                    pass

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
                except (ValueError, IndexError):
                    pass
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
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] Plot updated: {PLOT_FILE}")


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
    parser.add_argument("--watch_dir", type=str, action="append", default=[],
                        help="Checkpoint directories to watch (repeatable)")
    parser.add_argument("--games_per_pair", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Overriding model config (for scale_up etc.)")
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

    games_per_pair = args.games_per_pair
    # Update all caches' meta
    for d in args.watch_dir:
        cp = cache_path(d)
        cache = load_cache(cp)
        cache["_meta"]["games_per_pair"] = games_per_pair
        save_cache(cp, cache)

    print(f"Watching {len(args.watch_dir)} directories:")
    for d in args.watch_dir:
        print(f"  {d}")
    print(f"Games/pair: {games_per_pair}, batch: {args.batch}")
    print(f"Scan interval: {args.interval}s")
    print(f"Model config: d_model={model_cfg_raw['d_model']}, "
          f"n_layers={model_cfg_raw['n_layers']}")
    print()

    # Per-experiment model configs (different architectures)
    exp_configs = {}
    # Detect scale_up config
    scale_up_cfg = None
    if os.path.exists("configs/abl_scale_up.yaml"):
        with open("configs/abl_scale_up.yaml") as f:
            sc = yaml.safe_load(f)
            scale_up_cfg = ModelConfig.from_dict(sc["model"])

    def get_model_cfg(dir_path):
        exp = os.path.basename(dir_path.rstrip("/"))
        if exp == "scale_up" and scale_up_cfg is not None:
            return scale_up_cfg
        return model_cfg

    # Model cache — loaded on demand per directory
    model_cache = {}

    def get_model(dir_path, ckpt_name):
        key = (dir_path, ckpt_name)
        if key not in model_cache:
            path = os.path.join(dir_path, ckpt_name)
            cfg = get_model_cfg(dir_path)
            model_cache[key] = load_model(cfg, path, device)
            if len(model_cache) > 32:
                oldest = list(model_cache.keys())[0]
                del model_cache[oldest]
                torch.cuda.empty_cache()
        return model_cache[key]

    last_stats_time = 0.0
    stats_interval = 30.0  # print stats every 30s

    def print_stats():
        nonlocal last_stats_time
        now = time.time()
        if now - last_stats_time < stats_interval:
            return
        last_stats_time = now
        print(f"\n  {'─'*55}")
        print(f"  {'Experiment':<20s} {'CKPTs':>6s} {'Matched':>8s} {'Total':>8s} {'Done':>7s}")
        print(f"  {'─'*55}")
        total_intra_matched = 0; total_intra_pairs = 0
        for d in args.watch_dir:
            exp = os.path.basename(d.rstrip("/"))
            ckpts = all_ckpts(d)
            n = len(ckpts)
            total_pairs = n * (n - 1) // 2 if n >= 2 else 0
            cache = load_cache(cache_path(d))
            matched = sum(1 for k in cache if k != "_meta")
            pct = f"{100*matched/total_pairs:.0f}%" if total_pairs > 0 else "-"
            print(f"  {exp:<20s} {n:6d} {matched:8d} {total_pairs:8d} {pct:>7s}")
            total_intra_matched += matched; total_intra_pairs += total_pairs
        # Cross stats
        cc = load_cache(CROSS_CACHE)
        cross_matched = sum(1 for k in cc if k != "_meta")
        n_dirs = len([d for d in args.watch_dir if all_ckpts(d)])
        cross_total = 0
        if n_dirs >= 2:
            for i, da in enumerate(args.watch_dir):
                for db in args.watch_dir[i+1:]:
                    cross_total += len(all_ckpts(da)) * len(all_ckpts(db))
        cross_pct = f"{100*cross_matched/cross_total:.0f}%" if cross_total > 0 else "-"
        print(f"  {'cross':<20s} {'-':>6s} {cross_matched:8d} {cross_total:8d} {cross_pct:>7s}")
        overall_matched = total_intra_matched + cross_matched
        overall_total = total_intra_pairs + cross_total
        overall_pct = f"{100*overall_matched/overall_total:.0f}%" if overall_total > 0 else "-"
        print(f"  {'TOTAL':<20s} {'-':>6s} {overall_matched:8d} {overall_total:8d} {overall_pct:>7s}")
        print(f"  {'─'*55}\n")

    while True:
        # Collect intra-experiment missing pairs
        intra_missing = []
        for d in args.watch_dir:
            cache = load_cache(cache_path(d))
            for a, b in missing_pairs(d, cache, games_per_pair):
                intra_missing.append(("intra", d, d, a, b))

        # Collect cross-experiment missing pairs (sample up to 100)
        cross_missing = []
        cross_cache = load_cache(CROSS_CACHE)
        if len(args.watch_dir) >= 2:
            dirs_with_ckpts = [d for d in args.watch_dir if all_ckpts(d)]
            for _ in range(100):
                if len(dirs_with_ckpts) < 2: break
                da, db = random.sample(dirs_with_ckpts, 2)
                ckpts_a = all_ckpts(da)
                ckpts_b = all_ckpts(db)
                if not ckpts_a or not ckpts_b: continue
                a = random.choice(ckpts_a)
                b = random.choice(ckpts_b)
                key = cross_key(da, a, db, b)
                if key not in cross_cache or not match_ok(cross_cache.get(key, []), games_per_pair):
                    cross_missing.append(("cross", da, db, a, b))

        # Pick one pair: 70% intra, 30% cross (if both available)
        pool = intra_missing + cross_missing
        if not pool:
            print_stats()
            time.sleep(args.interval)
            try: plot_all(args.watch_dir, games_per_pair)
            except Exception as e: print(f"  Plot error: {e}")
            continue

        # Prefer cross-experiment occasionally for ELO anchoring
        if cross_missing and (not intra_missing or random.random() < 0.3):
            choice = random.choice(cross_missing)
        else:
            choice = random.choice(intra_missing)

        kind, da, db, a_name, b_name = choice
        exp_a = os.path.basename(da.rstrip("/"))
        exp_b = os.path.basename(db.rstrip("/"))

        if kind == "intra":
            cache = load_cache(cache_path(da))
            key = cache_key(a_name, b_name)
            tag = exp_a
        else:
            cache = cross_cache
            key = cross_key(da, a_name, db, b_name)
            tag = f"{exp_a} vs {exp_b}"

        if key in cache:
            continue

        model_a = get_model(da, a_name)
        model_b = get_model(db, b_name)
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: {a_name} vs {b_name} ...",
              end=" ", flush=True)

        wa, wb, dg = play_match(model_a, model_b, games_per_pair, args.batch, device)
        cache[key] = (wa, wb, dg, games_per_pair)
        if kind == "intra":
            save_cache(cache_path(da), cache)
        else:
            save_cache(CROSS_CACHE, cache)
        wr = wa/(wa+wb) if wa+wb>0 else 0.5
        print(f"{wa}-{wb} (D={dg}) WR={wr:.2%}")

        print_stats()
        try: plot_all(args.watch_dir, games_per_pair)
        except Exception as e: print(f"  Plot error: {e}")


if __name__ == "__main__":
    main()
