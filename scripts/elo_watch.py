#!/usr/bin/env python3
"""Continuous ELO monitor: watch checkpoints, play new ones, update plot.

Cache file (elo_cache.json) stores all pairwise results, so restart is cheap.

Usage:
    python scripts/elo_watch.py [--games_per_pair 200] [--batch 256] [--interval 10]
"""

import argparse
import json
import math
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15
N_CELLS = BOARD_SIZE * BOARD_SIZE
CACHE_FILE = "elo_cache.json"
PLOT_FILE = "output/elo_curve.png"


# ── ELO solver ──────────────────────────────────────────────

def compute_elo(match_results, initial=1500.0, max_iter=200):
    names = set()
    for a, b, _, _ in match_results:
        names.add(a); names.add(b)
    elo = {n: initial for n in sorted(names)}

    for _ in range(max_iter):
        delta_max = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n = sa + sb
            if n == 0:
                continue
            da = (sa - ea * n) * (32.0 / n)
            db = (sb - (1 - ea) * n) * (32.0 / n)
            elo[a] += da; elo[b] += db
            delta_max = max(delta_max, abs(da), abs(db))
        if delta_max < 1e-6:
            break
    return elo


# ── Game engine ─────────────────────────────────────────────

def load_model(path, config, device):
    model = GomokuTransformer(config).to(device).eval()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def play_batch(model_a, model_b, b, a_black, device):
    """Play b games. a_black[i]=True → model_a is black, else model_b is black.

    Returns (wins_a, wins_b, draws).
    """
    positions = torch.zeros(b, 0, dtype=torch.long, device=device)
    players   = torch.zeros(b, 0, dtype=torch.long, device=device)
    occupied  = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    active    = torch.ones(b, dtype=torch.bool, device=device)
    winners   = torch.zeros(b, dtype=torch.long, device=device)

    # First move from black's model
    fm_a = model_a.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_b = model_b.first_move_logits.unsqueeze(0).expand(b, -1)
    fm = torch.where(a_black.unsqueeze(1), fm_a, fm_b)
    fm = fm.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm.float(), dim=-1)
    first = torch.multinomial(probs, 1).squeeze(-1)
    occupied[torch.arange(b, device=device), first] = True
    positions = first.unsqueeze(1)
    players = torch.zeros(b, 1, dtype=torch.long, device=device)

    idx_a_list = []
    idx_b_list = []
    for i in range(b):
        (idx_a_list if a_black[i] else idx_b_list).append(i)
    idx_a = torch.tensor(idx_a_list, dtype=torch.long, device=device) if idx_a_list else None
    idx_b = torch.tensor(idx_b_list, dtype=torch.long, device=device) if idx_b_list else None

    for step in range(1, N_CELLS):
        cp = step % 2  # current player

        # Build indices for model_a (games where model_a plays this step)
        need_a = (cp == 0) & a_black
        need_a |= (cp == 1) & ~a_black
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

        # Check wins (scalar loop, fast enough)
        just_won = torch.zeros(b, dtype=torch.bool, device=device)
        for i in range(b):
            if not active[i]:
                continue
            a = actions[i].item()
            r, c = a // BOARD_SIZE, a % BOARD_SIZE
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                cnt = 1
                for sign in [1, -1]:
                    for k in range(1, 5):
                        nr, nc = r + dr * k * sign, c + dc * k * sign
                        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                            break
                        if not occupied[i, nr * BOARD_SIZE + nc]:
                            break
                        cnt += 1
                if cnt >= 5:
                    just_won[i] = True
                    break

        # Update
        occupied[active] = occupied[active].scatter_(
            1, actions[active].unsqueeze(1), True
        )
        positions = torch.cat([positions, actions.unsqueeze(1)], dim=1)
        plr_col = torch.full((b, 1), cp, dtype=torch.long, device=device)
        players = torch.cat([players, plr_col], dim=1)

        newly_done = just_won & active
        winners[newly_done] = cp + 1
        active[newly_done] = False
        if not active.any():
            break

    winners[(winners == 0) & ~active] = 3

    wins_a = wins_b = draws = 0
    for i in range(b):
        w = winners[i].item()
        if w == 1:
            wins_a += 1 if a_black[i] else 0
            wins_b += 0 if a_black[i] else 1
        elif w == 2:
            wins_a += 0 if a_black[i] else 1
            wins_b += 1 if a_black[i] else 0
        else:
            draws += 1
    return wins_a, wins_b, draws


def play_match(model_a, model_b, n_games, batch_size, device):
    wins_a = wins_b = draws = 0
    for start in range(0, n_games, batch_size):
        b = min(batch_size, n_games - start)
        a_black = torch.tensor(
            [(start + i) % 2 == 0 for i in range(b)], device=device
        )
        wa, wb, d = play_batch(model_a, model_b, b, a_black, device)
        wins_a += wa; wins_b += wb; draws += d
    return wins_a, wins_b, draws


# ── Cache ───────────────────────────────────────────────────

def cache_key(a, b):
    sa = int(a.split("_")[1].split(".")[0])
    sb = int(b.split("_")[1].split(".")[0])
    return f"{min(sa,sb)}_{max(sa,sb)}"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ── Plot ────────────────────────────────────────────────────

def plot_elo(cache, ckpts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Build match results from cache
    match_results = []
    for key, (wa, wb, d) in cache.items():
        s1, s2 = key.split("_")
        name_a = f"step_{int(s1):06d}.pt"
        name_b = f"step_{int(s2):06d}.pt"
        match_results.append((name_a, name_b, wa + d * 0.5, wb + d * 0.5))

    if len(match_results) < 1:
        return

    elo = compute_elo(match_results)
    steps = sorted(int(c.split("_")[1].split(".")[0]) for c in ckpts)
    ratings = [elo.get(f"step_{s:06d}.pt", 1500) for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ratings, "b.-", markersize=8, linewidth=1.5)
    ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.5)

    # Annotate each point
    for s, r in zip(steps, ratings):
        ax.annotate(f"{r:.0f}", (s, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("ELO Rating")
    ax.set_title("ELO Rating vs Training Step")
    ax.grid(True, alpha=0.3)

    ymin, ymax = min(ratings) - 20, max(ratings) + 20
    ax.set_ylim(ymin, ymax)

    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=120)
    plt.close()
    print(f"  Plot updated: {PLOT_FILE}")


# ── Main loop ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games_per_pair", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--interval", type=int, default=10,
                        help="Seconds between checkpoint scans")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    cache = load_cache()
    print(f"Loaded cache: {len(cache)} pairwise results")
    print(f"Watching {args.checkpoint_dir}/ for new checkpoints...")
    print(f"Games per pair: {args.games_per_pair}, batch: {args.batch}")
    print(f"Scan interval: {args.interval}s\n")

    # Load all existing models once
    loaded_models = {}

    def get_model(name):
        if name not in loaded_models:
            path = os.path.join(args.checkpoint_dir, name)
            loaded_models[name] = load_model(path, model_cfg, device)
        return loaded_models[name]

    current_ckpts = set()

    while True:
        if os.path.isdir(args.checkpoint_dir):
            ckpts = sorted([
                f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")
            ], key=lambda x: int(x.split("_")[1].split(".")[0]))
        else:
            ckpts = []

        ckpt_set = set(ckpts)

        # Find all missing pairs among current checkpoints
        all_missing = []
        for a in sorted(ckpt_set):
            for b in sorted(ckpt_set):
                if a >= b:
                    continue
                key = cache_key(a, b)
                if key not in cache:
                    all_missing.append((a, b))

        if all_missing:
            n = len(all_missing)
            print(f"\n[{time.strftime('%H:%M:%S')}] {n} missing pair(s) to evaluate")

            for a_name, b_name in sorted(all_missing):
                key = cache_key(a_name, b_name)
                if key in cache:
                    continue

                model_a = get_model(a_name)
                model_b = get_model(b_name)
                print(f"  {a_name} vs {b_name} ...", end=" ", flush=True)

                wa, wb, d = play_match(
                    model_a, model_b,
                    args.games_per_pair, args.batch, device
                )
                cache[key] = (wa, wb, d)
                save_cache(cache)
                wr = wa / (wa + wb) if wa + wb > 0 else 0.5
                print(f"{wa}-{wb} (D={d}) WR={wr:.2%}")

            # Update plot after all missing pairs done
            plot_elo(cache, sorted(ckpt_set))

        elif ckpt_set != current_ckpts:
            # Checkpoints changed but no new pairs (e.g., checkpoint deleted)
            if len(ckpt_set) >= 2:
                plot_elo(cache, ckpts)

        current_ckpts = ckpt_set
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
