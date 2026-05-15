#!/usr/bin/env python3
"""Human vs AI play on the terminal."""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training.self_play import _infer_batch


def print_board(state):
    """Print 15x15 board. state: list of 225 values (0=empty, 1=black, -1=white)."""
    symbols = {0: ".", 1: "X", -1: "O"}
    print("   " + "".join(f"{c:2}" for c in range(15)))
    for r in range(15):
        row = [symbols[state[r * 15 + c]] for c in range(15)]
        print(f"{r:2} " + " ".join(f"{s:2}" for s in row))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ai_color", type=str, default="white",
                        choices=["black", "white"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig.from_dict(config["model"])

    model = GomokuTransformer(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    try:
        import gomoku_cpp
    except ImportError:
        raise RuntimeError("gomoku_cpp not built. Run: pip install -e .")

    board = gomoku_cpp.Board()
    ai_color = 0 if args.ai_color == "black" else 1

    print(f"You play {'black (X)' if ai_color == 1 else 'white (O)'}")
    print("Enter move as 'row col' (0-14)\n")

    while board.result == 0:
        print_board(board.get_state())

        if board.current_player == ai_color:
            # AI's turn
            moves = board.get_moves()
            positions = [m for m in moves]
            players = [i % 2 for i in range(len(moves))]

            if len(positions) == 0:
                action = np.random.randint(0, 225)
            else:
                pos_t = torch.tensor([positions], dtype=torch.long, device=device)
                plr_t = torch.tensor([players], dtype=torch.long, device=device)
                action = int(_infer_batch(model, [pos_t[0]], [plr_t[0]], device)[0])

            row, col = action // 15, action % 15
            print(f"AI plays: ({row}, {col})")
            result = board.play_move(action)
            if board.result != 0:
                break
        else:
            # Human's turn
            while True:
                try:
                    inp = input("Your move (row col): ").strip().split()
                    row, col = int(inp[0]), int(inp[1])
                    action = row * 15 + col
                    if action < 0 or action >= 225:
                        print("Invalid position")
                        continue
                    if board.is_occupied(action):
                        print("Position already occupied")
                        continue
                    break
                except (ValueError, IndexError):
                    print("Enter as: row col")
                    continue

            board.play_move(action)

    print_board(board.get_state())
    if board.result == 1:
        winner = "Black (X)" if ai_color == 0 else "AI"
    elif board.result == 2:
        winner = "White (O)" if ai_color == 1 else "AI"
    else:
        winner = "Draw"

    print(f"Game over! Winner: {winner}")


if __name__ == "__main__":
    main()
