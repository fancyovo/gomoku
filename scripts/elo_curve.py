#!/usr/bin/env python3
"""Incremental ELO curve with caching — re-run safely while training.

Usage:
    python scripts/elo_curve.py --checkpoint_dir checkpoints/reward_decay_long \
        --config configs/abl_reward_decay_long.yaml --games_per_pair 512 --batch 256
"""

import argparse, itertools, json, math, os, sys, time, yaml
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15; N_CELLS = 225


# ═══════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════

def load_model(path, config, device):
    model = GomokuTransformer(config).to(device).eval()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


# ═══════════════════════════════════════════════════════════════
# Game engine (adapted from elo_tournament.py)
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def play_games_batch(model_a, model_b, batch_size, n_games, device):
    wins_a = 0; wins_b = 0; draws = 0
    for start in range(0, n_games, batch_size):
        b = min(batch_size, n_games - start)
        wa, wb, d = _play_batch(model_a, model_b, b, start, device)
        wins_a += wa; wins_b += wb; draws += d
    return wins_a, wins_b, draws


@torch.inference_mode()
def _play_batch(model_a, model_b, b, start_offset, device):
    positions = torch.zeros(b, 0, dtype=torch.long, device=device)
    players   = torch.zeros(b, 0, dtype=torch.long, device=device)
    occupied  = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    active    = torch.ones(b, dtype=torch.bool, device=device)
    winners   = torch.zeros(b, dtype=torch.long, device=device)

    is_a_black = torch.tensor(
        [(start_offset + i) % 2 == 0 for i in range(b)], device=device)

    fm_logits_a = model_a.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_logits_b = model_b.first_move_logits.unsqueeze(0).expand(b, -1)
    fm_logits = torch.where(is_a_black.unsqueeze(1), fm_logits_a, fm_logits_b)
    fm_logits = fm_logits.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm_logits.float(), dim=-1)
    first_act = torch.multinomial(probs, 1).squeeze(-1)
    occupied[torch.arange(b, device=device), first_act] = True
    positions = first_act.unsqueeze(1)
    players = torch.zeros(b, 1, dtype=torch.long, device=device)

    for step in range(1, N_CELLS):
        current_player = step % 2
        use_a = (current_player == 0) & is_a_black
        use_a |= (current_player == 1) & ~is_a_black

        logits = torch.zeros(b, N_CELLS, device=device)

        if use_a.any():
            idx_a = use_a.nonzero(as_tuple=True)[0]
            if idx_a.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_a = model_a(positions[idx_a], players[idx_a])
                logits[idx_a] = out_a.float()[:, -1, :]

        if (~use_a).any():
            idx_b = (~use_a).nonzero(as_tuple=True)[0]
            if idx_b.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_b = model_b(positions[idx_b], players[idx_b])
                logits[idx_b] = out_b.float()[:, -1, :]

        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)

        just_won = _batch_check_win(positions, occupied, actions, active, device)

        occupied[active] = occupied[active].scatter_(
            1, actions[active].unsqueeze(1), True)
        positions = torch.cat([positions, actions.unsqueeze(1)], dim=1)
        new_plr = torch.full((b, 1), current_player, dtype=torch.long, device=device)
        players = torch.cat([players, new_plr], dim=1)

        newly_done = just_won & active
        winners[newly_done] = current_player + 1
        active[newly_done] = False

        if not active.any():
            break

    winners[(winners == 0) & ~active] = 3  # draw

    wins_a = 0; wins_b = 0; draws_count = 0
    for i in range(b):
        w = winners[i].item()
        a_black = is_a_black[i].item()
        if w == 1:
            if a_black: wins_a += 1
            else: wins_b += 1
        elif w == 2:
            if a_black: wins_b += 1
            else: wins_a += 1
        else:
            draws_count += 1

    return wins_a, wins_b, draws_count


def _batch_check_win(positions, occupied, actions, active, device):
    b = actions.shape[0]
    won = torch.zeros(b, dtype=torch.bool, device=device)
    for i in range(b):
        if not active[i]:
            continue
        action = actions[i].item()
        r, c = action // BOARD_SIZE, action % BOARD_SIZE
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                for k in range(1, 5):
                    nr, nc = r + dr * k * sign, c + dc * k * sign
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        break
                    if not occupied[i, nr * BOARD_SIZE + nc]:
                        break
                    count += 1
            if count >= 5:
                won[i] = True
                break
    return won


# ═══════════════════════════════════════════════════════════════
# ELO solver
# ═══════════════════════════════════════════════════════════════

def compute_elo(match_results, initial=1500.0, max_iter=200):
    names = set()
    for a, b, _, _ in match_results:
        names.add(a); names.add(b)
    names = sorted(names)
    elo = {n: initial for n in names}
    for _ in range(max_iter):
        delta_max = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n_games = sa + sb
            delta_a = (sa - ea * n_games) * (32.0 / max(n_games, 1))
            elo[a] += delta_a; elo[b] -= delta_a
            delta_max = max(delta_max, abs(delta_a))
        if delta_max < 1e-6:
            break
    return elo


# ═══════════════════════════════════════════════════════════════
# Cache management
# ═══════════════════════════════════════════════════════════════

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}

def save_cache(cache_path, cache):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)

def sorted_key(ckpt_name):
    """Extract step number from checkpoint filename like 'step_000050.pt'."""
    try:
        return int(ckpt_name.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return 0


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_results(elo, win_rates, steps, cache, out_dir, exp_name):
    os.makedirs(out_dir, exist_ok=True)

    # ── ELO curve ──
    fig, ax = plt.subplots(figsize=(14, 6))
    if elo:
        sorted_names = sorted(elo.keys(), key=lambda k: int(k))
        x = [int(n) for n in sorted_names]
        y = [elo[n] for n in sorted_names]
        ax.plot(x, y, ".-", color="C0", markersize=6, linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("ELO Rating")
        ax.set_title(f"ELO Curve — {exp_name}")
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    elo_path = os.path.join(out_dir, "elo_curve.png")
    plt.savefig(elo_path, dpi=150); plt.close()
    print(f"Saved ELO curve to {elo_path}")

    # ── Win rate heatmap ──
    if len(steps) >= 2:
        n = len(steps)
        mat = np.full((n, n), np.nan)
        for i, si in enumerate(steps):
            for j, sj in enumerate(steps):
                if i >= j:
                    continue
                pair_key = f"{si}|{sj}"
                if pair_key in cache:
                    wa, wb, d = cache[pair_key]
                    total = wa + wb
                    mat[i, j] = wa / total if total > 0 else 0.5
                    mat[j, i] = wb / total if total > 0 else 0.5

        fig, ax = plt.subplots(figsize=(12, 10))
        labels = [f"{s}" for s in steps]
        mask = np.isnan(mat)
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="Win Rate (row vs col)")

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        # Only label every Nth tick to avoid crowding
        tick_step = max(1, n // 30)
        tick_positions = list(range(0, n, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([labels[i] for i in tick_positions], fontsize=8)

        ax.set_title(f"Win Rate Matrix — {exp_name}")
        plt.tight_layout()
        hm_path = os.path.join(out_dir, "winrate_heatmap.png")
        plt.savefig(hm_path, dpi=150); plt.close()
        print(f"Saved heatmap to {hm_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def evaluate_all_pairs(ckpts, sorted_key, model_cfg, cache, cache_path, args, device):
    """Evaluate all uncached pairs. Returns (new_count, updated cache)."""
    pairs = list(itertools.combinations(ckpts, 2))
    total_new = 0

    for ai, (a_name, b_name) in enumerate(pairs):
        a_step = sorted_key(a_name)
        b_step = sorted_key(b_name)
        pair_key = f"{a_step}|{b_step}"

        if pair_key in cache:
            continue

        total_new += 1
        a_path = os.path.join(args.checkpoint_dir, a_name)
        b_path = os.path.join(args.checkpoint_dir, b_name)

        model_a = load_model(a_path, model_cfg, device)
        model_b = load_model(b_path, model_cfg, device)

        wins_a, wins_b, draws = play_games_batch(
            model_a, model_b, args.batch, args.games_per_pair, device)

        cache[pair_key] = [wins_a, wins_b, draws]

        del model_a, model_b
        torch.cuda.empty_cache()

        total = wins_a + wins_b + draws
        wr = wins_a / (wins_a + wins_b) if (wins_a + wins_b) > 0 else 0.5
        print(f"  [{ai+1}/{len(pairs)}] step_{a_step:06d} vs step_{b_step:06d}:  "
              f"{wins_a}-{wins_b} (D={draws})  WR={wr:.2%}")

        if total_new % 5 == 0:
            save_cache(cache_path, cache)

    return total_new, cache


def build_elo_and_plot(cache, ckpts, sorted_key, args, exp_name):
    """Compute ELO from cache and regenerate plots."""
    steps = [sorted_key(c) for c in ckpts]

    match_results = []
    for pair_key, val in cache.items():
        try:
            sa, sb = pair_key.split("|")
            wa, wb, d = val
            score_a = wa + d * 0.5
            score_b = wb + d * 0.5
            match_results.append((sa, sb, score_a, score_b))
        except (ValueError, AttributeError):
            continue

    elo = {}
    if match_results:
        elo_raw = compute_elo(match_results)
        for step in steps:
            key = str(step)
            if key in elo_raw:
                elo[key] = elo_raw[key]

    if elo:
        ranked = sorted(elo.items(), key=lambda x: -x[1])
        print(f"\n{'='*50}")
        print("ELO Rankings (top 10):")
        print("=" * 50)
        for rank, (step_str, rating) in enumerate(ranked[:10], 1):
            step_num = int(step_str)
            bar = "█" * max(1, int((rating - 1400) / 4))
            print(f"  {rank:2d}. step_{step_num:06d}  {rating:7.1f}  {bar}")

    plot_results(elo, {}, steps, cache, args.output_dir, exp_name)
    return elo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--games_per_pair", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Continuously watch for new checkpoints and update ELO")
    parser.add_argument("--interval", type=int, default=120,
                        help="Polling interval in seconds (default: 120)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    exp_name = cfg.get("wandb", {}).get("name", os.path.basename(args.checkpoint_dir))
    device = torch.device(args.device)

    if args.output_dir is None:
        args.output_dir = f"output/{exp_name}_elo"

    cache_path = os.path.join(args.output_dir, "pair_cache.json")
    cache = load_cache(cache_path)

    known_ckpts = set()

    while True:
        ckpts = sorted(
            [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")],
            key=sorted_key
        )

        if len(ckpts) < 2:
            print(f"[{time.strftime('%H:%M:%S')}] Need >= 2 checkpoints, found {len(ckpts)}. "
                  f"Waiting {args.interval}s...")
            if not args.watch:
                return
            time.sleep(args.interval)
            continue

        steps = [sorted_key(c) for c in ckpts]
        new_ckpts = set(ckpts) - known_ckpts

        if not new_ckpts and known_ckpts:
            if not args.watch:
                print(f"\nAll {len(cache)} pairs already cached — no new checkpoints.")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {len(ckpts)} ckpts, "
                      f"no new checkpoints. Waiting {args.interval}s...")
            # Still update plots (ELO may change slightly or cache was updated)
            build_elo_and_plot(cache, ckpts, sorted_key, args, exp_name)
            if not args.watch:
                return
            time.sleep(args.interval)
            continue

        if not known_ckpts:
            print(f"Experiment: {exp_name}")
            print(f"Checkpoints: {len(ckpts)}  ({steps[0]} .. {steps[-1]})")
            print(f"Games/pair: {args.games_per_pair}  batch: {args.batch}")
            print(f"Pairs: {len(ckpts)*(len(ckpts)-1)//2}  "
                  f"Already cached: {len([k for k in cache if '|' in str(k)])}")
            print(f"Cache: {cache_path}")
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoints: {len(new_ckpts)} "
                  f"({[sorted_key(c) for c in sorted(new_ckpts, key=sorted_key)]})")
            print(f"Total: {len(ckpts)} ckpts, "
                  f"{len(ckpts)*(len(ckpts)-1)//2 - len(cache):,} new pairs to evaluate")

        known_ckpts = set(ckpts)
        print()

        total_new, cache = evaluate_all_pairs(
            ckpts, sorted_key, model_cfg, cache, cache_path, args, device)

        if total_new > 0:
            save_cache(cache_path, cache)
            print(f"\n{total_new} new pairs played, cache saved to {cache_path}")

        build_elo_and_plot(cache, ckpts, sorted_key, args, exp_name)

        if not args.watch:
            break

        print(f"\n[{time.strftime('%H:%M:%S')}] Waiting for next checkpoint "
              f"(interval={args.interval}s)...\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
