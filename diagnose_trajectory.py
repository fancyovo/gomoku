"""
Diagnostic script: dump full self-play trajectory details for manual verification.

Usage:
    python diagnose_trajectory.py [--num-games 10] [--checkpoint path/to/model.pt]
"""
import torch
import argparse
import numpy as np
import sys

from src.model.config import ModelConfig
from src.model.transformer import GomokuTransformer
from src.training.self_play import SelfPlayRunner


def load_config():
    import yaml
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def pprint_trajectory(traj, idx):
    """Pretty-print a single trajectory with full detail."""
    positions = traj["positions"].tolist()
    players = traj["players"].tolist()
    actions = traj["actions"].tolist()
    rewards = traj["rewards"].tolist()
    actual_len = traj["actual_len"]
    result = traj["result"]
    end_reason = traj["end_reason"]

    end_reason_str = {1: "WIN(5-in-a-row)", 2: "ILLEGAL", 3: "DRAW"}
    result_str = {1: "BLACK_WINS", 2: "WHITE_WINS", 3: "DRAW"}

    print(f"\n{'='*80}")
    print(f"Trajectory #{idx + 1}")
    print(f"{'='*80}")
    print(f"  actual_len  = {actual_len}")
    print(f"  end_reason  = {end_reason} ({end_reason_str.get(end_reason, '?')})")
    print(f"  result      = {result} ({result_str.get(result, '?')})")
    print(f"  seq_length  = {len(positions)} (= existing_history + page_size)")
    print()

    # Show full timeline: positions, players, actions, rewards
    print(f"  {'Step':<6} {'Player':<8} {'Pos':<6} {'Action':<6} {'Reward':<8} {'Legal?':<8} {'Note'}")
    print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*20}")

    # Build set of occupied positions up to each step
    occupied = set()
    for step in range(len(positions)):
        pos = positions[step]
        plr = players[step]
        act = actions[step]
        rew = rewards[step]
        plr_str = "BLACK" if plr == 0 else "WHITE"

        # Check if the action (which equals position) is legal at this step
        # Position is legal if it wasn't already occupied
        legal = pos not in occupied

        note = ""
        if step < actual_len:
            if legal:
                occupied.add(pos)
                note = "legal, added to board"
            else:
                # This is the illegal move that ended the game
                note = f"ILLEGAL! Pos {pos} already occupied"
        else:
            if step == actual_len:
                note = "--- game ended here ---"
            else:
                note = "post-game padding (reward=0)"

        print(f"  {step:<6} {plr_str:<8} {pos:<6} {act:<6} {rew:>+7.1f}  {str(legal):<8} {note}")

    # Verify: positions and actions should be identical
    if positions == actions:
        print(f"\n  [OK] positions == actions (as expected for Gomoku)")
    else:
        print(f"\n  [ERROR] positions != actions!")
        for i, (p, a) in enumerate(zip(positions, actions)):
            if p != a:
                print(f"    step {i}: pos={p}, act={a}")

    # Check reward consistency
    print(f"\n  Reward summary:")
    black_moves = [(i, pos, rew) for i, (plr, pos, rew)
                   in enumerate(zip(players, positions, rewards))
                   if plr == 0 and i < actual_len]
    white_moves = [(i, pos, rew) for i, (plr, pos, rew)
                   in enumerate(zip(players, positions, rewards))
                   if plr == 1 and i < actual_len]

    print(f"    BLACK moves ({len(black_moves)}): {[(p, f'{r:+.1f}') for _, p, r in black_moves]}")
    print(f"    WHITE moves ({len(white_moves)}): {[(p, f'{r:+.1f}') for _, p, r in white_moves]}")

    # Find the illegal move if any
    if end_reason == 2:
        occupied_check = set()
        for step in range(actual_len):
            pos = positions[step]
            if pos in occupied_check:
                print(f"\n  [CONFIRMED] Illegal move at step {step}: "
                      f"{'BLACK' if players[step] == 0 else 'WHITE'} played position {pos} "
                      f"which was already occupied")
                print(f"    Occupied before this move: {sorted(occupied_check)}")
                break
            occupied_check.add(pos)

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config()
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = GomokuTransformer(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["training"]
    train_cfg["games_per_step"] = args.num_games  # small for debugging

    runner = SelfPlayRunner(model, device, train_cfg)
    model.eval()

    print(f"Running self-play with {args.num_games} games (random init, seed={args.seed})...")
    print(f"  page_size={train_cfg['page_size']}, board_size={model_cfg.board_size}")
    trajectories = runner.run_one_wave()

    print(f"\nGenerated {len(trajectories)} trajectories\n")

    lens = [t["actual_len"] for t in trajectories]
    n_illegal = sum(1 for t in trajectories if t["end_reason"] == 2)
    n_win = sum(1 for t in trajectories if t["end_reason"] == 1)
    n_draw = sum(1 for t in trajectories if t["end_reason"] == 3)

    print(f"Summary: avg_len={sum(lens)/len(lens):.1f}, max_len={max(lens)}, "
          f"min_len={min(lens)}")
    print(f"  Wins: {n_win}, Illegals: {n_illegal}, Draws: {n_draw}")
    print(f"  Illegal rate: {n_illegal/len(trajectories):.2%}")

    for i, t in enumerate(trajectories):
        pprint_trajectory(t, i)

    # =========================================================================
    # After printing trajectories, do a manual check: simulate the training
    # data flow for the first trajectory to verify the shift fix.
    # =========================================================================
    print(f"\n{'='*80}")
    print("MANUAL TRAINING DATA FLOW CHECK (first trajectory)")
    print(f"{'='*80}")

    t = trajectories[0]
    pos = t["positions"].unsqueeze(0)  # (1, L_total)
    plr = t["players"].unsqueeze(0)
    rew = t["rewards"].unsqueeze(0)
    mask = torch.zeros(1, pos.shape[1], dtype=torch.bool)
    mask[0, :t["actual_len"]] = True
    actual_len = t["actual_len"]

    L = pos.shape[1]
    print(f"  Input shape: positions={list(pos.shape)}, actual_len={actual_len}")

    # Simulate the shift
    if L > 1:
        pred_logits_shape = (1, L - 1, model_cfg.n_positions)
        pred_act = pos[:, 1:]    # (1, L-1)
        pred_rew = rew[:, 1:]    # (1, L-1)
        pred_mask = mask[:, 1:]  # (1, L-1)

        print(f"\n  Shifted predictions (logits[:, :-1] → action[:, 1:]):")
        print(f"  {'Shift idx':<10} {'Predicts':<12} {'From inputs':<25} {'Action':<8} {'Reward':<8} {'Mask'}")
        print(f"  {'-'*10} {'-'*12} {'-'*25} {'-'*8} {'-'*8} {'-'*5}")
        for shift_i in range(L - 1):
            pred_step = shift_i + 1  # predicting action at this step
            predicts = f"step {pred_step}"
            from_inputs = f"positions[0..{shift_i}]"
            action_val = pred_act[0, shift_i].item()
            reward_val = pred_rew[0, shift_i].item()
            mask_val = pred_mask[0, shift_i].item()
            print(f"  {shift_i:<10} {predicts:<12} {from_inputs:<25} {action_val:<8} {reward_val:>+7.1f} {bool(mask_val)}")
    else:
        print("  L <= 1, no shifted predictions")

    # Check first_move_logits training
    print(f"\n  first_move_logits training:")
    print(f"    act[:, 0] = {pos[0, 0].item()}")
    print(f"    rew[:, 0] = {rew[0, 0].item():+.1f}")
    print(f"    mask[:, 0] = {mask[0, 0].item()}")

    # Verify correctness
    print(f"\n  Verification:")
    # For a game of actual_len A, valid training positions are 0..A-2 (shifted)
    # Action at position t (0-indexed) should be the (t+1)-th move
    # logits[:, t, :] sees positions[:, 0..t] and predicts action[:, t+1]
    errors = []
    for shift_i in range(min(actual_len - 1, L - 1)):
        pred_step = shift_i + 1
        from_inputs = list(range(shift_i + 1))
        action = pos[0, pred_step].item()
        # Check: can the model see the action it's trying to predict?
        # It CAN if action appears in positions[:, 0..shift_i]
        # That would be a data leak (predicting an already-seen position)
        input_positions = [pos[0, j].item() for j in range(shift_i + 1)]
        if action in input_positions:
            errors.append(
                f"  shift_i={shift_i}: predicting step {pred_step} pos={action}, "
                f"but this position already appears in inputs {input_positions}! "
                f"This means the model is trained to predict an occupied position "
                f"(= illegal move), which gets reward {pred_rew[0, shift_i].item():+.1f}"
            )

    if errors:
        print(f"  [NOTE] Found {len(errors)} positions where target = already-occupied position:")
        for e in errors:
            print(e)
        print(f"  This is EXPECTED when the game ended by illegal move — the illegal")
        print(f"  move IS in the training data with negative reward.")
    else:
        print(f"  [OK] No data leaks in shifted predictions.")


if __name__ == "__main__":
    main()
