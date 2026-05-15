#!/usr/bin/env python3
"""ELO tournament: all checkpoint pairs play against each other.

Usage:
    python scripts/elo_tournament.py [--games_per_pair 20] [--device cuda]
"""

import argparse
import os
import sys
import math
import torch
import numpy as np
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
def play_game(model_black, model_white, device):
    """Play one game. Returns 1=black_win, 2=white_win, 3=draw."""
    occupied = torch.zeros(1, N_CELLS, dtype=torch.bool, device=device)
    positions = torch.zeros(1, 0, dtype=torch.long, device=device)
    players = torch.zeros(1, 0, dtype=torch.long, device=device)

    # First move (black) — from model_black's first_move_logits
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fm_logits = model_black.first_move_logits.unsqueeze(0)
        fm_logits = fm_logits.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm_logits.float(), dim=-1)
    action = torch.multinomial(probs, 1).squeeze(-1).item()
    occupied[0, action] = True
    positions = torch.tensor([[action]], dtype=torch.long, device=device)
    players = torch.tensor([[0]], dtype=torch.long, device=device)

    for step in range(1, N_CELLS):
        current_player = step % 2  # 0=black, 1=white
        model = model_black if current_player == 0 else model_white

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(positions, players)
        logits = logits.float()[:, -1, :]
        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1).item()

        # Check if illegal (shouldn't happen with masking, but safety)
        if occupied[0, action]:
            return 2 if current_player == 0 else 1  # illegal = lose

        # Check win: 5 in a row
        r, c = action // BOARD_SIZE, action % BOARD_SIZE
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                for i in range(1, 5):
                    nr, nc = r + dr * i * sign, c + dc * i * sign
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        break
                    if not occupied[0, nr * BOARD_SIZE + nc]:
                        break
                    count += 1
            if count >= 5:
                return current_player + 1  # 1=black, 2=white

        occupied[0, action] = True
        positions = torch.cat([positions, torch.tensor([[action]], device=device)], dim=1)
        players = torch.cat([players, torch.tensor([[current_player]], device=device)], dim=1)

    return 3  # draw


def elo_update(rating_a, rating_b, result_a, k=32):
    """result_a: 1=win, 0.5=draw, 0=loss"""
    ea = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    return rating_a + k * (result_a - ea)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--games_per_pair", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])

    # Find checkpoints
    ckpt_dir = args.checkpoint_dir
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoint dir: {ckpt_dir}")
        return

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if len(ckpts) < 2:
        print(f"Need at least 2 checkpoints, found {len(ckpts)}")
        return

    print(f"Found {len(ckpts)} checkpoints: {ckpts}")
    print(f"Games per pair: {args.games_per_pair}")
    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Load all models
    models = {}
    for name in ckpts:
        path = os.path.join(ckpt_dir, name)
        models[name] = load_model(path, model_cfg, device)
        print(f"Loaded {name}")

    # Initialize ELO
    elo = {name: 1500.0 for name in ckpts}

    # Play round-robin
    pairs = list(itertools.combinations(ckpts, 2))
    print(f"\nPlaying {len(pairs)} pairs × {args.games_per_pair} games each...\n")

    for a_name, b_name in pairs:
        model_a = models[a_name]
        model_b = models[b_name]
        wins_a = 0
        wins_b = 0
        draws = 0

        for g in range(args.games_per_pair):
            # Alternate colors
            if g % 2 == 0:
                result = play_game(model_a, model_b, device)
                if result == 1: wins_a += 1
                elif result == 2: wins_b += 1
                else: draws += 1
            else:
                result = play_game(model_b, model_a, device)
                if result == 1: wins_b += 1
                elif result == 2: wins_a += 1
                else: draws += 1

        # Update ELO
        for _ in range(wins_a):
            elo[a_name] = elo_update(elo[a_name], elo[b_name], 1.0, k=16)
            elo[b_name] = elo_update(elo[b_name], elo[a_name], 0.0, k=16)
        for _ in range(wins_b):
            elo[a_name] = elo_update(elo[a_name], elo[b_name], 0.0, k=16)
            elo[b_name] = elo_update(elo[b_name], elo[a_name], 1.0, k=16)
        for _ in range(draws):
            elo[a_name] = elo_update(elo[a_name], elo[b_name], 0.5, k=16)
            elo[b_name] = elo_update(elo[b_name], elo[a_name], 0.5, k=16)

        wr = wins_a / (wins_a + wins_b) if (wins_a + wins_b) > 0 else 0.5
        print(f"  {a_name} vs {b_name}: {wins_a}-{wins_b} (D={draws}), "
              f"WR={wr:.2%}, ELO Δ={elo[a_name]-elo[b_name]:+.0f}")

    # Final rankings
    print(f"\n{'='*50}")
    print(f"Final ELO rankings:")
    print(f"{'='*50}")
    ranked = sorted(elo.items(), key=lambda x: -x[1])
    base = ranked[0][1]
    for rank, (name, rating) in enumerate(ranked, 1):
        step_num = int(name.split("_")[1].split(".")[0])
        print(f"  {rank:2d}. step_{step_num:06d}  {rating:6.1f}  ({rating-base:+.1f})")


if __name__ == "__main__":
    main()
