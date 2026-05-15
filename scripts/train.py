#!/usr/bin/env python3
"""Training entry point."""

import argparse
import os
import sys
import yaml
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit training batches per step (for testing)")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Training config: {config['training']}")
    print()

    trainer = Trainer(model, config["training"], device)

    start_step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        start_step = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from step {start_step}")

    for step in range(start_step, args.total_steps):
        t0 = time.perf_counter()
        metrics = trainer.train_step(step, args.total_steps,
                                      max_batches=args.max_batches)
        step_time = time.perf_counter() - t0

        print(f"Step {step:4d} | "
              f"loss: {metrics['loss/total']:.4f} | "
              f"policy: {metrics['loss/policy']:.4f} | "
              f"entropy: {metrics['loss/entropy']:.4f} | "
              f"ppl: {metrics['policy/perplexity']:.1f} | "
              f"games: {metrics['perf/raw_games']} | "
              f"batches: {metrics['train/batches']} | "
              f"moves: {metrics['train/n_valid_moves']:,} | "
              f"sp: {metrics['perf/self_play_time']:.1f}s | "
              f"tr: {metrics['train/time']:.1f}s | "
              f"total: {step_time:.1f}s")

        if args.max_batches:
            print(f"\nStopping after {args.max_batches} batches (test mode).")
            break

        if (step + 1) % config["training"]["checkpoint_interval"] == 0:
            path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            trainer.save_checkpoint(path, step)
            print(f"  -> checkpoint: {path}")


if __name__ == "__main__":
    main()
