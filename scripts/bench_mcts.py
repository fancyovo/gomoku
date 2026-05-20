#!/usr/bin/env python3
"""Benchmark MCTS self-play pipeline and profile performance bottlenecks.

Usage:
    python scripts/bench_mcts.py --batch_size 2048 --device cuda
"""

import argparse, os, sys, time, math
import torch, numpy as np, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
from training.mcts_self_play import MCTSSelfPlayRunner, N_CELLS

COLS = "ABCDEFGHIJKLMNO"


def render_board(moves):
    """ASCII board render of a game."""
    board = [["." for _ in range(15)] for _ in range(15)]
    for i, pos in enumerate(moves):
        r, c = pos // 15, pos % 15
        ch = "X" if i % 2 == 0 else "O"
        board[r][c] = ch
    header = "   " + " ".join(COLS)
    lines = [header]
    for r in range(15):
        lines.append(f"{r+1:2d} " + " ".join(board[r]))
    return "\n".join(lines)


def describe_game(pos_seq, plr_seq, result, mcts_pols=None):
    """Print game description."""
    print(f"\n  Game length: {len(pos_seq)} moves")
    result_str = {1: "Black wins", 2: "White wins", 3: "Draw", 0: "Unknown"}[result]
    print(f"  Result: {result_str}")

    print(f"\n  Move list:")
    for i, (pos, plr) in enumerate(zip(pos_seq, plr_seq)):
        r, c = pos // 15, pos % 15
        color = "B" if plr == 0 else "W"
        print(f"    {i+1:3d}. {color} {COLS[c]}{r+1:2d}", end="")
        if mcts_pols is not None and i < len(mcts_pols):
            top3 = np.argsort(mcts_pols[i])[-3:][::-1]
            print(f"  top3:", end="")
            for a in top3:
                ar, ac = a // 15, a % 15
                print(f" {COLS[ac]}{ar+1:2d}({mcts_pols[i][a]:.3f})", end="")
        print()

    print(f"\n  Board:")
    print(render_board(pos_seq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    print(f"Creating random model (d_model={model_cfg.d_model}, "
          f"n_layers={model_cfg.n_layers})...")
    model = GomokuTransformer(model_cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    config = {
        "mcts_batch_size": args.batch_size,
        "mcts_simulations": 4,
        "c_puct": 1.0,
        "dirichlet_eps": 0.25,
        "dirichlet_alpha": 0.03,
        "root_temperature": 1.0,
        "max_cache_len": 100,
        "max_path_len": 40,
    }

    # Try memory probe first
    print(f"\n--- Memory probe: creating runner with batch_size={args.batch_size} ---")
    try:
        probe_config = dict(config)
        probe_config["mcts_simulations"] = 2
        runner = MCTSSelfPlayRunner(model, device, probe_config)
        print("  Runner created OK, testing one wave...")
        t0 = time.perf_counter()
        trajs = runner.run_one_wave()
        dt = time.perf_counter() - t0
        print(f"  Probe OK: {len(trajs)} trajectories in {dt:.1f}s "
              f"({len(trajs)/dt:.1f} games/s)")
        del runner, trajs
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("  OOM! Reducing batch_size by half...")
        args.batch_size //= 2
        config["mcts_batch_size"] = args.batch_size
        print(f"  Retrying with batch_size={args.batch_size}")
        probe_config = dict(config)
        probe_config["mcts_simulations"] = 2
        runner = MCTSSelfPlayRunner(model, device, probe_config)
        trajs = runner.run_one_wave()
        print(f"  OK: {len(trajs)} trajectories")
        del runner, trajs
        torch.cuda.empty_cache()

    # ── Main benchmark: double num_simulations until >3min ──
    print(f"\n{'='*70}")
    print(f"Benchmark: batch_size={args.batch_size}")
    print(f"{'='*70}")
    print(f"{'Sims':>6s}  {'Time':>8s}  {'Games/s':>10s}  {'Select':>8s}  {'Eval':>8s}  {'Expand':>8s}  {'Prefill':>8s}  {'Step':>8s}  {'Cleanup':>8s}")
    print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    TIME_LIMIT = 180  # 3 minutes

    sims = 4
    last_trajs = None
    last_runner = None

    while sims <= 512:
        config["mcts_simulations"] = sims
        runner = MCTSSelfPlayRunner(model, device, config)

        t0 = time.perf_counter()
        trajs = runner.run_one_wave()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        timing = runner.timing
        n_games = len(trajs)
        gps = n_games / dt if dt > 0 else 0

        print(f"{sims:6d}  {dt:8.1f}s  {gps:10.1f}  "
              f"{timing.get('mcts_select',0):8.1f}s  "
              f"{timing.get('mcts_eval',0):8.1f}s  "
              f"{timing.get('mcts_expand',0):8.1f}s  "
              f"{timing.get('kv_prefill',0):8.1f}s  "
              f"{timing.get('game_step',0):8.1f}s  "
              f"{timing.get('kv_cleanup',0):8.1f}s")

        # Percentage breakdown
        total_t = sum(v for k, v in timing.items())
        if total_t > 0:
            parts = []
            for k in ["mcts_select", "mcts_eval", "mcts_expand", "kv_prefill", "game_step", "kv_cleanup"]:
                v = timing.get(k, 0)
                parts.append(f"{k}={v/total_t*100:.1f}%")
            print(f"        Breakdown: {' | '.join(parts)}")

        last_trajs = trajs
        last_runner = runner

        del runner, trajs
        torch.cuda.empty_cache()

        if dt > TIME_LIMIT:
            print(f"\n  Time {dt:.1f}s exceeds {TIME_LIMIT}s limit, stopping at sims={sims}")
            break

        sims *= 2

    # ── Inspect a sample trajectory ──
    print(f"\n{'='*70}")
    print("Sample Trajectory Inspection")
    print(f"{'='*70}")

    if last_trajs:
        # Pick an interesting game (not too short, not the longest)
        valid = [t for t in last_trajs if t is not None and t.get("actual_len", 0) > 5]
        if valid:
            # Pick a game near the median length
            valid.sort(key=lambda t: t["actual_len"])
            traj = valid[len(valid) // 2]

            pos_seq = traj["positions"].tolist()
            plr_seq = traj["players"].tolist()
            result = traj["result"]
            mcts_pols = traj["mcts_policies"].numpy() if hasattr(traj["mcts_policies"], "numpy") else traj["mcts_policies"]

            describe_game(pos_seq, plr_seq, result, mcts_pols)

            # MCTS policy statistics
            print(f"\n  MCTS Policy Statistics:")
            for i in range(min(len(mcts_pols), 5)):
                pol = mcts_pols[i]
                nonzero = (pol > 0).sum()
                top_val = pol.max()
                entropy = -(pol[pol > 0] * np.log(pol[pol > 0] + 1e-9)).sum()
                print(f"    Move {i+1}: nonzero={nonzero}, max={top_val:.4f}, entropy={entropy:.3f}")
            if len(mcts_pols) > 5:
                pol_last = mcts_pols[-1]
                nonzero = (pol_last > 0).sum()
                top_val = pol_last.max()
                print(f"    Move {len(mcts_pols)}: nonzero={nonzero}, max={top_val:.4f}")

            # Check basic correctness
            print(f"\n  Correctness checks:")
            L = len(pos_seq)
            assert len(plr_seq) == L, "player seq length mismatch"
            assert len(mcts_pols) >= L - 1, f"MCTS policies missing: {len(mcts_pols)} vs {L}"
            assert result in [1, 2, 3], f"Invalid result: {result}"
            # Check first move is black
            assert plr_seq[0] == 0, "First move not black"
            print(f"    All checks passed.")

            # Print trajectory structure
            print(f"\n  Trajectory keys: {list(traj.keys())}")
            for k, v in traj.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"    {k}: {v}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
