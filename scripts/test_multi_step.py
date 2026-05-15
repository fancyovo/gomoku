#!/usr/bin/env python3
"""Run N training steps, track perplexity-by-len evolution across iterations."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch, yaml, numpy as np
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
    print(f"LR: {config['training']['lr']}, augment: {config['training']['augment']}")
    print(f"Running 10 steps...\n")

    trainer = Trainer(model, config["training"], device)
    n_steps = 10
    ppl_history = []  # step → list of (seq_pos, ppl)
    step_times = []

    for step in range(n_steps):
        t0 = time.perf_counter()
        metrics = trainer.train_step(step, n_steps, ppl_interval=999)  # no per-batch ppl logging
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        ppl_by_len = metrics["ppl_by_len"]
        ppl_history.append(ppl_by_len)

        print(f"Step {step+1}/{n_steps}: {elapsed:.1f}s | "
              f"loss={metrics['loss/total']:.4f} | "
              f"entropy={metrics['loss/entropy']:.4f} | "
              f"games={metrics['perf/raw_games']} | "
              f"avg_len={metrics['game/avg_len']:.1f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

    # Left: perplexity by sequence length, all steps overlaid
    ax = axes[0]
    for i, ppl_data in enumerate(ppl_history):
        if ppl_data:
            steps, ppls = zip(*ppl_data)
            ax.plot(steps, ppls, color=colors[i], linewidth=1.2, alpha=0.8,
                    label=f"step {i+1}")
    # Ideal reference
    max_steps = max(len(d) for d in ppl_history if d)
    ideal_x = np.arange(max_steps)
    ideal_y = 225.0 - ideal_x
    ax.plot(ideal_x, ideal_y, "k--", linewidth=1.0, alpha=0.5, label="ideal (225−step)")
    ax.axhline(y=225, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Sequence position (step)")
    ax.set_ylabel("Perplexity exp(entropy)")
    ax.set_title("PPL by sequence position — 10 steps overlay")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: first vs last step comparison
    ax = axes[1]
    if len(ppl_history) >= 2:
        first = ppl_history[0]
        last = ppl_history[-1]
        if first and last:
            s1, p1 = zip(*first)
            s2, p2 = zip(*last)
            ax.plot(s1, p1, "r.-", markersize=3, linewidth=1.0, label=f"step 1", alpha=0.7)
            ax.plot(s2, p2, "b.-", markersize=3, linewidth=1.0, label=f"step {n_steps}", alpha=0.7)
    ideal_x = np.arange(max(len(d) for d in ppl_history if d))
    ax.plot(ideal_x, 225.0 - ideal_x, "k--", linewidth=1.0, alpha=0.5, label="ideal")
    ax.axhline(y=225, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Sequence position (step)")
    ax.set_ylabel("Perplexity exp(entropy)")
    ax.set_title("Step 1 vs Step 10 perplexity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "output/ppl_evolution_10steps.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")

    # Print numerical comparison
    print(f"\n=== First vs last step ppl-by-len ===")
    if len(ppl_history) >= 2:
        first = ppl_history[0]
        last = ppl_history[-1]
        for j in range(min(10, len(first), len(last))):
            s1, p1 = first[j]
            s2, p2 = last[j]
            print(f"  pos {s1:3d}: step1 ppl={p1:.1f} → step{n_steps} ppl={p2:.1f} (Δ={p2-p1:+.1f})")


if __name__ == "__main__":
    main()
