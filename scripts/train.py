#!/usr/bin/env python3
"""Entry point for training the Gomoku Transformer."""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training import Trainer
from monitoring import WandbLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Logger
    wandb_enabled = not args.no_wandb
    logger = WandbLogger(
        project=config.get("wandb", {}).get("project", "gomoku-transformer"),
        entity=config.get("wandb", {}).get("entity"),
        config=config,
        enabled=wandb_enabled,
    )

    if not wandb_enabled:
        print("wandb logging disabled")

    # Trainer
    trainer = Trainer(model, config, device, logger)
    start_step = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        start_step = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from step {start_step}")

    # Training loop
    for step in range(start_step, args.total_steps):
        logger.log_step_start(step)

        metrics = trainer.train_step(step, args.total_steps)
        logger.log_step_end(step, metrics)

        print(
            f"Step {step:5d} | "
            f"loss: {metrics['loss/total']:.4f} | "
            f"policy: {metrics['loss/policy']:.4f} | "
            f"entropy: {metrics['loss/entropy']:.4f} | "
            f"winrate_b: {metrics['game/black_winrate']:.2f} | "
            f"len: {metrics['game/avg_length']:.1f} | "
            f"n_games: {metrics['perf/n_games']}"
        )

        if (step + 1) % config["training"]["checkpoint_interval"] == 0:
            path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            trainer.save_checkpoint(path, step)
            print(f"  -> checkpoint: {path}")

    logger.finish()


if __name__ == "__main__":
    main()
