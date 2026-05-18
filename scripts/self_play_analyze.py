#!/usr/bin/env python3
"""Self-play a few games from a checkpoint and output full game records."""
import argparse, math, os, sys, yaml, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer

BOARD_SIZE = 15; N_CELLS = 225
COLS = "ABCDEFGHIJKLMNO"


def load_model(path, config, device):
    m = GomokuTransformer(config).to(device).eval()
    state = torch.load(path, map_location=device)
    m.load_state_dict(state)
    return m


@torch.inference_mode()
def self_play_game(model, device):
    """Play one game (model vs itself). Returns list of (position, player)."""
    positions = torch.zeros(1, 0, dtype=torch.long, device=device)
    players = torch.zeros(1, 0, dtype=torch.long, device=device)
    occupied = torch.zeros(1, N_CELLS, dtype=torch.bool, device=device)
    black_stones = torch.zeros(1, N_CELLS, dtype=torch.bool, device=device)
    white_stones = torch.zeros(1, N_CELLS, dtype=torch.bool, device=device)

    fm = model.first_move_logits.unsqueeze(0)
    fm = fm.masked_fill(occupied, -1e9)
    act = torch.multinomial(torch.softmax(fm.float(), dim=-1), 1).squeeze(-1)
    occupied[0, act] = True
    black_stones[0, act] = True
    positions = act.unsqueeze(1)
    players = torch.zeros(1, 1, dtype=torch.long, device=device)
    moves = [(act.item(), 0)]

    for step in range(1, N_CELLS):
        cp = step % 2
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(positions, players)
        logits = out.float()[:, -1, :]
        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        act = torch.multinomial(probs, 1).squeeze(-1)

        # Check win using color-specific board
        winner = _check_win_for_player(black_stones[0] if cp == 0 else white_stones[0], act.item())
        if winner:
            moves.append((act.item(), cp))
            return moves, cp

        occupied[0, act] = True
        if cp == 0:
            black_stones[0, act] = True
        else:
            white_stones[0, act] = True
        positions = torch.cat([positions, act.unsqueeze(1)], dim=1)
        players = torch.cat([players, torch.full((1, 1), cp, dtype=torch.long, device=device)], dim=1)
        moves.append((act.item(), cp))

    return moves, None  # draw


def _check_win_for_player(stones, action):
    """Check if 'action' completes 5-in-a-row on the given stones bitmask."""
    r, c = action // BOARD_SIZE, action % BOARD_SIZE
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        for sign in [1, -1]:
            for k in range(1, 5):
                nr, nc = r + dr * k * sign, c + dc * k * sign
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    break
                if not stones[nr * BOARD_SIZE + nc]:
                    break
                count += 1
        if count >= 5:
            return True
    return False


def render_board(moves):
    """ASCII board render."""
    board = [["." for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for i, (pos, plr) in enumerate(moves):
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        ch = "●" if plr == 0 else "○"
        board[r][c] = ch

    header = "   " + " ".join(COLS)
    lines = [header]
    for r in range(BOARD_SIZE):
        row_num = f"{r+1:2d} "
        line = row_num + " ".join(board[r])
        lines.append(line)
    return "\n".join(lines)


def describe_moves(moves):
    """Text description of each move."""
    lines = []
    for i, (pos, plr) in enumerate(moves):
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        color = "● Black" if plr == 0 else "○ White"
        lines.append(f"  {i+1:3d}. {COLS[c]}{r+1:2d}  {color}")
    return "\n".join(lines)


def analyze_game(moves, winner):
    """Print move-by-move analysis of key patterns."""
    print(f"\n  Total moves: {len(moves)}")
    print(f"  Winner: {'Black' if winner == 0 else 'White' if winner == 1 else 'Draw'}")

    # Move distribution by quadrant
    quads = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right
    mid = BOARD_SIZE // 2
    for pos, _ in moves:
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        q = (0 if r < mid else 2) + (0 if c < mid else 1)
        quads[q] += 1
    print(f"  Quadrants (TL/TR/BL/BR): {[f'{q/len(moves):.1%}' for q in quads]}")

    # Row distribution
    rows = [0] * BOARD_SIZE
    for pos, _ in moves:
        rows[pos // BOARD_SIZE] += 1
    top3_rows = sorted(enumerate(rows), key=lambda x: -x[1])[:3]
    print(f"  Top 3 rows: {[(r, f'{c/len(moves):.1%}') for r,c in top3_rows if c>0]}")

    # Distance between consecutive moves
    dists = []
    for i in range(1, len(moves)):
        r1, c1 = moves[i-1][0] // BOARD_SIZE, moves[i-1][0] % BOARD_SIZE
        r2, c2 = moves[i][0] // BOARD_SIZE, moves[i][0] % BOARD_SIZE
        dists.append(abs(r1-r2) + abs(c1-c2))
    print(f"  Move distances: min={min(dists)} max={max(dists)} avg={np.mean(dists):.1f}")

    # Check for "race" pattern: both players building lines independently
    black_moves = [(i, pos) for i, (pos, plr) in enumerate(moves) if plr == 0]
    white_moves = [(i, pos) for i, (pos, plr) in enumerate(moves) if plr == 1]

    # Black's line-building pattern: check if moves cluster near each other
    if len(black_moves) >= 3:
        b_rows = [pos // BOARD_SIZE for _, pos in black_moves]
        b_cols = [pos % BOARD_SIZE for _, pos in black_moves]
        print(f"  Black: row_range={max(b_rows)-min(b_rows)+1} col_range={max(b_cols)-min(b_cols)+1}")

    if len(white_moves) >= 3:
        w_rows = [pos // BOARD_SIZE for _, pos in white_moves]
        w_cols = [pos % BOARD_SIZE for _, pos in white_moves]
        print(f"  White: row_range={max(w_rows)-min(w_rows)+1} col_range={max(w_cols)-min(w_cols)+1}")

    # Detect "edge/center" preference
    edge = 0
    for pos, _ in moves:
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        if r <= 2 or r >= 12 or c <= 2 or c >= 12:
            edge += 1
    print(f"  Edge (<=2 or >=12): {edge}/{len(moves)} ({edge/len(moves):.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_games", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    print(f"Loading {args.checkpoint}...")
    model = load_model(args.checkpoint, model_cfg, device)
    print(f"Playing {args.num_games} games of model vs itself...\n")

    for g in range(args.num_games):
        moves, winner = self_play_game(model, device)
        print("=" * 70)
        print(f"Game {g+1}")
        print(render_board(moves))
        print()
        print(describe_moves(moves))
        print()
        analyze_game(moves, winner)
        print()

    # Summary
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
