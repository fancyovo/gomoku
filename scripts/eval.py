#!/usr/bin/env python3
"""Evaluate two checkpoints against each other."""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training.self_play import _infer_batch


def play_game(model_black, model_white, device):
    """Play a single game: model_black vs model_white. Returns winner (0=black, 1=white, 2=draw)."""
    try:
        import gomoku_cpp
    except ImportError:
        raise RuntimeError("gomoku_cpp not built")

    board = gomoku_cpp.Board()
    positions_b = []
    players_b = []
    positions_w = []
    players_w = []

    while board.result == 0:
        current_player = board.current_player
        if current_player == 0:  # black
            model = model_black
            positions_b.append(board.get_moves())
            # Build input sequence
            moves = board.get_moves()
            positions = [m for m in moves]  # all positions
            players = [i % 2 for i in range(len(moves))]
        else:
            model = model_white
            positions_w.append(board.get_moves())
            moves = board.get_moves()
            positions = [m for m in moves]
            players = [i % 2 for i in range(len(moves))]

        if len(positions) == 0:
            # First move: random
            action = np.random.randint(0, 225)
        else:
            pos_t = torch.tensor([positions], dtype=torch.long, device=device)
            plr_t = torch.tensor([players], dtype=torch.long, device=device)
            action = _infer_batch(model, [pos_t[0]], [plr_t[0]], device)
            action = int(action[0])

        result = board.play_move(action)
        if result == -1:  # illegal
            # Current player loses
            return 1 if current_player == 0 else 0

    if board.result == 1:
        return 0  # black wins
    elif board.result == 2:
        return 1  # white wins
    else:
        return 2  # draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_games", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig.from_dict(config["model"])

    # Load models
    model_a = GomokuTransformer(model_cfg).to(device)
    ckpt_a = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(ckpt_a)
    model_a.eval()

    model_b = GomokuTransformer(model_cfg).to(device)
    ckpt_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(ckpt_b)
    model_b.eval()

    # Play
    wins_a = 0
    wins_b = 0
    draws = 0

    for i in range(args.num_games):
        # Alternate colors
        if i % 2 == 0:
            winner = play_game(model_a, model_b, device)
            if winner == 0:
                wins_a += 1
            elif winner == 1:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_game(model_b, model_a, device)
            if winner == 0:
                wins_b += 1
            elif winner == 1:
                wins_a += 1
            else:
                draws += 1

        if (i + 1) % 50 == 0:
            print(f"  played {i + 1}/{args.num_games} games")

    print(f"\nResults ({args.num_games} games):")
    print(f"  A wins: {wins_a}")
    print(f"  B wins: {wins_b}")
    print(f"  Draws:  {draws}")
    print(f"  A winrate: {wins_a / args.num_games:.2%}")

    # Simple Elo: +400 * log10(winrate / (1 - winrate))
    if wins_a + wins_b > 0:
        wr = wins_a / (wins_a + wins_b)
        if wr > 0 and wr < 1:
            elo_diff = 400 * np.log10(wr / (1 - wr))
            print(f"  Elo(A) - Elo(B): {elo_diff:+.1f}")


if __name__ == "__main__":
    main()
