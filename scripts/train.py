#!/usr/bin/env python3
"""Training script for Gomoku Transformer."""

import argparse
import os
import sys
import time
import yaml
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training import Trainer
from monitoring.logger import WandbLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default from config wandb.name)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = ModelConfig.from_dict(config["model"])
    model = GomokuTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Config: games={config['training']['games_per_step']}, "
          f"batch={config['training']['train_batch_size']}, "
          f"lr={config['training']['lr']}, "
          f"augment={config['training']['augment']}")
    print(f"Total steps: {args.total_steps}, "
          f"checkpoint every: {config['training']['checkpoint_interval']}")
    print()

    # Wandb
    use_wandb = not args.no_wandb
    logger = WandbLogger(
        project=config.get("wandb", {}).get("project", "gomoku-transformer"),
        entity=config.get("wandb", {}).get("entity"),
        name=config.get("wandb", {}).get("name"),
        config=config,
        enabled=use_wandb,
    )
    if use_wandb:
        print("wandb: enabled")
    else:
        print("wandb: disabled")
    print()

    # Trainer
    trainer = Trainer(model, config["training"], device)

    # Default checkpoint dir from experiment name
    if args.checkpoint_dir == "checkpoints":
        exp_name = config.get("wandb", {}).get("name", "default")
        args.checkpoint_dir = f"checkpoints/{exp_name}"

    start_step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        # Parse step from filename: step_XXXXXX.pt
        import re
        m = re.search(r"step_(\d+)", args.resume)
        start_step = int(m.group(1)) + 1 if m else 0
        print(f"Resumed from step {start_step}")

    print(f"Training from step {start_step} to {args.total_steps}")
    print("=" * 60)

    for step in range(start_step, args.total_steps):
        logger.log_step_start(step)

        t0 = time.perf_counter()
        metrics = trainer.train_step(step, args.total_steps)
        step_time = time.perf_counter() - t0

        # --- Log to wandb ---
        wandb_metrics = {
            "loss/total": metrics["loss/total"],
            "loss/policy": metrics["loss/policy"],
            "loss/entropy": metrics["loss/entropy"],
            "game/avg_len": metrics["game/avg_len"],
            "game/max_len": metrics["game/max_len"],
            "game/black_winrate": metrics["game/black_winrate"],
            "game/white_winrate": metrics["game/white_winrate"],
            "game/illegal_rate": metrics["game/illegal_rate"],
            "game/win_rate_5": metrics["game/win_rate_5"],
            "game/total_moves": metrics["game/total_moves"],
            "perf/self_play_time": metrics["perf/self_play_time"],
            "perf/train_time": metrics["train/time"],
            "perf/step_time": step_time,
            "perf/raw_games": metrics["perf/raw_games"],
            "perf/n_batches": metrics["train/batches"],
            "perf/n_valid_moves": metrics["train/n_valid_moves"],
            "train/lr": metrics["train/lr"],
            "train/entropy_coef": metrics["train/entropy_coef"],
        }

        # PPL by sequence position (first and last few)
        ppl_by_len = metrics.get("ppl_by_len", [])
        if ppl_by_len:
            wandb_metrics["ppl/by_pos_0"] = ppl_by_len[0][1]
            wandb_metrics["ppl/by_pos_mid"] = ppl_by_len[len(ppl_by_len)//2][1]
            wandb_metrics["ppl/by_pos_last"] = ppl_by_len[-1][1]

        logger.log_step_end(step, wandb_metrics)

        # --- Console ---
        ppl_pos0 = ppl_by_len[0][1] if ppl_by_len else float("nan")
        ppl_last = ppl_by_len[-1][1] if ppl_by_len else float("nan")
        print(
            f"[{step:4d}/{args.total_steps}] "
            f"loss={metrics['loss/total']:+.4f} | "
            f"ent={metrics['loss/entropy']:.3f} | "
            f"ppl(0)={ppl_pos0:.0f} ppl(-1)={ppl_last:.0f} | "
            f"len={metrics['game/avg_len']:.1f} | "
            f"ill={metrics['game/illegal_rate']:.1%} | "
            f"win_b={metrics['game/black_winrate']:.2%} | "
            f"games={metrics['perf/raw_games']} | "
            f"sp={metrics['perf/self_play_time']:.0f}s "
            f"tr={metrics['train/time']:.0f}s "
            f"tot={step_time:.0f}s"
        )

        # --- Checkpoint ---
        ckpt_interval = config["training"]["checkpoint_interval"]
        if (step + 1) % ckpt_interval == 0 or step == args.total_steps - 1:
            path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            trainer.save_checkpoint(path)
            print(f"  ==> checkpoint: {path}")

    logger.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
