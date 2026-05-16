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
    positions = torch.zeros(b, 0, dtype=torch.long, device=device)
    players   = torch.zeros(b, 0, dtype=torch.long, device=device)
    occupied  = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    stones    = torch.zeros(b, N_CELLS, dtype=torch.uint8, device=device)
    active    = torch.ones(b, dtype=torch.bool, device=device)
    winners   = torch.zeros(b, dtype=torch.long, device=device)

    fm_a = model_a.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_b = model_b.first_move_logits.unsqueeze(0).expand(b, -1)
    fm = torch.where(a_black.unsqueeze(1), fm_a, fm_b)
    fm = fm.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm.float(), dim=-1)
    first = torch.multinomial(probs, 1).squeeze(-1)
    idx = torch.arange(b, device=device)
    occupied[idx, first] = True; stones[idx, first] = 1
    positions = first.unsqueeze(1)
    players = torch.zeros(b, 1, dtype=torch.long, device=device)

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

        just_won = torch.zeros(b, dtype=torch.bool, device=device)
        for i in range(b):
            if not active[i]: continue
            a = actions[i].item(); r, c = a // BOARD_SIZE, a % BOARD_SIZE
            for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                cnt = 1
                for sign in [1,-1]:
                    for k in range(1,5):
                        nr, nc = r+dr*k*sign, c+dc*k*sign
                        if not (0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE): break
                        if stones[i, nr*BOARD_SIZE+nc].item() != (cp+1): break
                        cnt += 1
                if cnt >= 5: just_won[i] = True; break

        occupied[active] = occupied[active].scatter_(1, actions[active].unsqueeze(1), True)
        stones[active] = stones[active].scatter_(1, actions[active].unsqueeze(1), cp+1)
        positions = torch.cat([positions, actions.unsqueeze(1)], dim=1)
        players = torch.cat([players, torch.full((b,1), cp, dtype=torch.long, device=device)], dim=1)

        newly_done = just_won & active
        winners[newly_done] = cp + 1
        active[newly_done] = False
        if not active.any(): break

    winners[(winners==0) & ~active] = 3
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


# ── Plot ────────────────────────────────────────────────────

def plot_all(watch_dirs, games_per_pair):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(watch_dirs), 1)))

    for di, d in enumerate(watch_dirs):
        exp = os.path.basename(d.rstrip("/"))
        cp = cache_path(d)
        if not os.path.exists(cp): continue
        cache = load_cache(cp)
        match_results = []
        for key, val in cache.items():
            if key == "_meta": continue
            wa, wb, d = val
            s1, s2 = key.split("_")
            a = f"step_{int(s1):06d}.pt"; b = f"step_{int(s2):06d}.pt"
            match_results.append((a, b, wa + d*0.5, wb + d*0.5))
        if len(match_results) < 1: continue

        elo = compute_elo(match_results)
        steps = sorted(int(k.split("_")[1].split(".")[0]) for k in elo)
        ratings = [elo[f"step_{s:06d}.pt"] for s in steps]
        ax.plot(steps, ratings, ".-", color=colors[di], markersize=3, linewidth=1.5, label=exp)

    ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Step"); ax.set_ylabel("ELO Rating")
    ax.set_title(f"ELO Comparison ({games_per_pair} games/pair)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    os.makedirs("output", exist_ok=True)
    plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=120); plt.close()
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] Plot updated: {PLOT_FILE}")


# ── Pair enumeration ─────────────────────────────────────────

def all_ckpts(watch_dir):
    if not os.path.isdir(watch_dir): return []
    return sorted([f for f in os.listdir(watch_dir) if f.endswith(".pt")],
                  key=lambda x: int(x.split("_")[1].split(".")[0]))

def missing_pairs(watch_dir, cache):
    ckpts = all_ckpts(watch_dir)
    missing = []
    for i, a in enumerate(ckpts):
        for b in ckpts[i+1:]:
            key = cache_key(a, b)
            if key not in cache:
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

    # Model cache — loaded on demand per directory
    model_cache = {}

    def get_model(dir_path, ckpt_name, load_new=True):
        key = (dir_path, ckpt_name)
        if key not in model_cache:
            path = os.path.join(dir_path, ckpt_name)
            model_cache[key] = load_model(model_cfg, path, device)
            # Limit model cache size
            if len(model_cache) > 32:
                oldest = list(model_cache.keys())[0]
                del model_cache[oldest]
                torch.cuda.empty_cache()
        return model_cache[key]

    while True:
        # Collect all missing pairs across all directories
        all_missing = []
        for d in args.watch_dir:
            cache = load_cache(cache_path(d))
            missing = missing_pairs(d, cache)
            for a, b in missing:
                all_missing.append((d, a, b))

        if all_missing:
            # Randomly pick one pair to evaluate
            d, a_name, b_name = random.choice(all_missing)
            exp = os.path.basename(d.rstrip("/"))
            cache = load_cache(cache_path(d))
            key = cache_key(a_name, b_name)
            if key in cache:  # race condition guard
                continue

            model_a = get_model(d, a_name)
            model_b = get_model(d, b_name)
            print(f"[{time.strftime('%H:%M:%S')}] {exp}: {a_name} vs {b_name} ...",
                  end=" ", flush=True)

            wa, wb, dg = play_match(model_a, model_b, games_per_pair, args.batch, device)
            cache[key] = (wa, wb, dg)
            save_cache(cache_path(d), cache)
            wr = wa/(wa+wb) if wa+wb>0 else 0.5
            print(f"{wa}-{wb} (D={dg}) WR={wr:.2%}")

        # Plot all
        plot_all(args.watch_dir, games_per_pair)

        if not all_missing:
            time.sleep(args.interval)
        # If there were missing pairs, loop immediately to pick another


if __name__ == "__main__":
    main()
