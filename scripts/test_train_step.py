#!/usr/bin/env python3
"""Run one full training step and plot perplexity curves."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ModelConfig, GomokuTransformer
from training import Trainer


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")
    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Games: {config['training']['games_per_step']}, "
          f"augment={config['training']['augment']}, "
          f"batch={config['training']['train_batch_size']}, "
          f"dtype=bfloat16")
    print()

    trainer = Trainer(model, config["training"], device)

    t0 = time.perf_counter()
    metrics = trainer.train_step(0, 10000, max_batches=None, ppl_interval=5)
    total = time.perf_counter() - t0

    # Print summary
    print(f"\n=== Full step summary ===")
    print(f"  Self-play:   {metrics['perf/self_play_time']:.1f}s")
    print(f"  Training:    {metrics['train/time']:.1f}s")
    print(f"  Total:       {total:.1f}s")
    print(f"  Games:       {metrics['perf/raw_games']}")
    print(f"  Batches:     {metrics['train/batches']}")
    print(f"  Valid moves: {metrics['train/n_valid_moves']:,}")
    print(f"  Loss:        {metrics['loss/total']:.4f}")
    print(f"  Policy loss: {metrics['loss/policy']:.4f}")
    print(f"  Entropy:     {metrics['loss/entropy']:.4f}")
    print(f"  Black win%:  {metrics['game/black_winrate']:.2%}")
    print(f"  Avg length:  {metrics['game/avg_len']:.1f}")

    # Perplexity curve during training
    ppl_curve = metrics["ppl_curve"]
    batches, ppls = zip(*ppl_curve) if ppl_curve else ([], [])

    # Perplexity by sequence length
    ppl_by_len = metrics["ppl_by_len"]
    lens, len_ppls = zip(*ppl_by_len) if ppl_by_len else ([], [])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: perplexity vs training progress
    ax = axes[0]
    if batches:
        ax.plot(batches, ppls, "b.-", markersize=4)
    ax.axhline(y=225, color="gray", linestyle="--", alpha=0.5, label="225 (uniform)")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Perplexity exp(entropy)")
    ax.set_title("Perplexity during training (per-batch)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: perplexity vs sequence length
    ax = axes[1]
    if lens:
        ax.plot(lens, len_ppls, "r.-", markersize=4)
        # Expected: 225 - remaining_cells = 225 - (225 - i) = i ...
        # Actually: at step i (0-indexed), i moves have been played
        # remaining empty = 225 - i
        # If model is uniform over all positions: ppl = 225 (constant)
        # If model avoids occupied: ppl ≈ 225 - i (linear decrease)
        # Reference line for "ideal": ppl = 225 - len
        ideal_x = np.array(lens)
        ideal_y = 225.0 - ideal_x
        ax.plot(ideal_x, ideal_y, "gray", linestyle="--", alpha=0.5, label="ideal (225 - step)")
    ax.axhline(y=225, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Sequence position (step)")
    ax.set_ylabel("Perplexity exp(entropy)")
    ax.set_title("Perplexity by sequence position (post-training)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "output/perplexity.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
