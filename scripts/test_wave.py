#!/usr/bin/env python3
"""Run one wave of self-play and report timing and game length distribution."""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import yaml
from collections import Counter

from model import ModelConfig, GomokuTransformer
from training.self_play import SelfPlayRunner


def theoretical_distribution():
    """Compute expected game length distribution for random (illegal-move-only) play."""
    probs = []
    surv = 1.0
    N = 225
    for k in range(1, N + 2):
        prob_end = surv * (k - 1) / N  # chance to end at step k due to illegal move
        if prob_end < 1e-10 and k > 2:
            break
        probs.append((k, prob_end))
        surv *= (N - (k - 1)) / N
    return probs


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Games per step: {config['training']['games_per_step']}")
    print(f"Page size: {config['training']['page_size']}")
    print(f"Base batch: {config['training']['base_batch']}, base seq: {config['training']['base_seq_len']}")
    print()

    runner = SelfPlayRunner(model, device, config["training"])

    t0 = time.perf_counter()
    trajectories = runner.run_one_wave()
    total_time = time.perf_counter() - t0

    python_time = runner.timing["python"]
    cpp_time = runner.timing["cpp"]

    print(f"Wave complete: {len(trajectories)} games in {total_time:.3f}s")
    print(f"  Python (inference): {python_time:.3f}s ({python_time/total_time*100:.1f}%)")
    print(f"  C++   (execute):    {cpp_time*1000:.3f}ms ({cpp_time/total_time*100:.3f}%)")
    print()

    # Game length distribution
    lengths = [t["actual_len"] for t in trajectories]
    results = [t["result"] for t in trajectories]

    len_counter = Counter(lengths)
    result_counter = Counter(results)

    print("Game length distribution:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.2f}")
    for lo in range(0, max(lengths) + 1, 10):
        hi = lo + 9
        count = sum(len_counter.get(l, 0) for l in range(lo, hi + 1))
        if count > 0:
            bar = "#" * (count * 50 // max(len_counter.values()))
            print(f"  [{lo:3d}-{hi:3d}]: {count:5d} ({count/len(trajectories)*100:5.1f}%) {bar}")

    print(f"\nEnd reasons:")
    for k, v in sorted(result_counter.items()):
        label = {1: "black_win", 2: "white_win", 3: "draw"}.get(k, f"unknown({k})")
        print(f"  {label}: {v} ({v/len(trajectories)*100:.1f}%)")

    # Compare with theoretical (random illegal-move) distribution
    print(f"\nTheoretical (random illegal-move only):")
    theo = theoretical_distribution()
    theo_total = sum(p for _, p in theo)
    cum = 0
    for k, p in theo:
        cum += p
        if k <= 5 or k % 10 == 0:
            print(f"  step {k:3d}: P(end)={p:.4f}  cum={cum:.4f}")

    expected_len = sum(k * p for k, p in theo) / theo_total
    print(f"  Expected length: {expected_len:.2f}")
    print(f"  Actual mean length: {sum(lengths)/len(lengths):.2f}")


if __name__ == "__main__":
    main()
