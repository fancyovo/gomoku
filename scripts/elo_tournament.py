#!/usr/bin/env python3
"""Batched ELO tournament: all checkpoint pairs play against each other.

Usage:
    python scripts/elo_tournament.py --games_per_pair 200 --batch 256
"""

import argparse
import os
import sys
import math
import torch
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15
N_CELLS = BOARD_SIZE * BOARD_SIZE


def load_model(path, config, device):
    model = GomokuTransformer(config).to(device).eval()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def play_games_batch(model_a, model_b, batch_size, n_games, device):
    """Play n_games between model_a and model_b, alternating colors.

    Returns: (wins_a, wins_b, draws)
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for start in range(0, n_games, batch_size):
        b = min(batch_size, n_games - start)
        wins_a_batch, wins_b_batch, draws_batch = _play_batch(
            model_a, model_b, b, start, device
        )
        wins_a += wins_a_batch
        wins_b += wins_b_batch
        draws += draws_batch

    return wins_a, wins_b, draws


@torch.inference_mode()
def _play_batch(model_a, model_b, b, start_offset, device):
    """Play a batch of b games. Games alternate colors starting from offset."""
    # State: (B, max_len) tensors, grow dynamically
    positions = torch.zeros(b, 0, dtype=torch.long, device=device)
    players   = torch.zeros(b, 0, dtype=torch.long, device=device)
    occupied  = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    active    = torch.ones(b, dtype=torch.bool, device=device)
    winners   = torch.zeros(b, dtype=torch.long, device=device)  # 0=ongoing, 1,2,3

    # First move (black) — always from model_a for even games, model_b for odd
    # We handle the color alternation by swapping models at each step
    is_a_black = torch.tensor(
        [(start_offset + i) % 2 == 0 for i in range(b)],
        device=device
    )

    # First move from the black player's model
    # Black's first_move_logits come from the model that plays black
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
        current_player = step % 2  # 0=black, 1=white

        # For black moves at games where a_is_black: use model_a, else model_b
        use_a = (current_player == 0) & is_a_black
        use_a |= (current_player == 1) & ~is_a_black

        # Forward pass — we need both models if the batch has mixed colors
        logits = torch.zeros(b, N_CELLS, device=device)

        # Games using model_a
        if use_a.any():
            idx_a = use_a.nonzero(as_tuple=True)[0]
            if idx_a.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_a = model_a(positions[idx_a], players[idx_a])
                logits[idx_a] = out_a.float()[:, -1, :]

        # Games using model_b
        if (~use_a).any():
            idx_b = (~use_a).nonzero(as_tuple=True)[0]
            if idx_b.numel() > 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_b = model_b(positions[idx_b], players[idx_b])
                logits[idx_b] = out_b.float()[:, -1, :]

        # Mask occupied and sample
        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)

        # Check win via 5-in-a-row (vectorized)
        just_won = _batch_check_win(positions, occupied, actions, active, device)

        # Update state for active games
        occupied[active] = occupied[active].scatter_(
            1, actions[active].unsqueeze(1), True
        )
        positions = torch.cat([positions, actions.unsqueeze(1)], dim=1)
        # players: append current_player for each game
        new_plr = torch.full((b, 1), current_player, dtype=torch.long, device=device)
        players = torch.cat([players, new_plr], dim=1)

        # Mark finished
        newly_done = just_won & active
        winners[newly_done] = current_player + 1  # 1 or 2
        active[newly_done] = False

        if not active.any():
            break

    # Check for draws (board full, no one won)
    winners[(winners == 0) & ~active] = 3  # draw

    wins_a = 0
    wins_b = 0
    draws_count = 0
    for i in range(b):
        w = winners[i].item()
        a_black = is_a_black[i].item()
        if w == 1:  # black wins
            if a_black: wins_a += 1
            else: wins_b += 1
        elif w == 2:  # white wins
            if a_black: wins_b += 1
            else: wins_a += 1
        else:
            draws_count += 1

    return wins_a, wins_b, draws_count


def _batch_check_win(positions, occupied, actions, active, device):
    """Check if the latest action creates 5-in-a-row in any active game."""
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


def compute_elo_ratings(match_results, initial=1500.0, max_iter=200):
    """Iterative ELO solver using all match results simultaneously.

    match_results: list of (name_a, name_b, score_a, score_b)
        where score_a is points earned by a (win=1, draw=0.5, loss=0).
    """
    names = set()
    for a, b, _, _ in match_results:
        names.add(a); names.add(b)
    names = sorted(names)
    elo = {n: initial for n in names}

    for _ in range(max_iter):
        delta_max = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            eb = 1.0 - ea
            n_games = sa + sb
            delta_a = (sa - ea * n_games) * (32.0 / max(n_games, 1))
            delta_b = (sb - eb * n_games) * (32.0 / max(n_games, 1))
            elo[a] += delta_a
            elo[b] += delta_b
            delta_max = max(delta_max, abs(delta_a), abs(delta_b))
        if delta_max < 1e-6:
            break
    return elo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--games_per_pair", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    ckpts = sorted(
        [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if len(ckpts) < 2:
        print(f"Need >= 2 checkpoints, found {len(ckpts)}")
        return

    print(f"Checkpoints: {len(ckpts)}, games/pair: {args.games_per_pair}, batch: {args.batch}")
    print(f"Device: {device}")

    # Load all models
    models = {}
    for name in ckpts:
        path = os.path.join(args.checkpoint_dir, name)
        models[name] = load_model(path, model_cfg, device)
        print(f"  Loaded {name}")

    pairs = list(itertools.combinations(ckpts, 2))
    print(f"\n{len(pairs)} pairs, {args.games_per_pair} games each...\n")

    match_results = []
    for a_name, b_name in pairs:
        model_a = models[a_name]
        model_b = models[b_name]

        wins_a, wins_b, draws = play_games_batch(
            model_a, model_b, args.batch, args.games_per_pair, device
        )

        total = wins_a + wins_b + draws
        wr = wins_a / (wins_a + wins_b) if (wins_a + wins_b) > 0 else 0.5
        score_a = wins_a + draws * 0.5
        score_b = wins_b + draws * 0.5
        match_results.append((a_name, b_name, score_a, score_b))
        print(f"  {a_name} vs {b_name}: {wins_a}-{wins_b} (D={draws}) WR={wr:.2%}")

    # Compute ELO from all results simultaneously
    elo = compute_elo_ratings(match_results)
    print(f"\n{'='*50}")
    print("ELO Rankings:")
    print('='*50)
    ranked = sorted(elo.items(), key=lambda x: -x[1])
    base = ranked[0][1]
    for rank, (name, rating) in enumerate(ranked, 1):
        step_num = int(name.split("_")[1].split(".")[0])
        bar = "█" * max(1, int((rating - 1450) / 5))
        print(f"  {rank:2d}. step_{step_num:06d}  {rating:7.1f}  ({rating-base:+.1f})  {bar}")


if __name__ == "__main__":
    main()
