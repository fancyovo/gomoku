import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

from .loss import reinforce_loss
from .self_play import SelfPlayRunner


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        logger=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.logger = logger

        self.optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.entropy_coef = config["training"]["entropy_coef"]
        self.entropy_coef_min = config["training"]["entropy_coef_min"]
        self.grad_clip = config["training"]["grad_clip"]
        self.train_micro_batch = config["training"]["train_micro_batch"]

        self.runner = SelfPlayRunner(
            model,
            device,
            config["training"]["games_per_step"],
            config["training"]["infer_micro_batch"],
        )

    def _entropy_coef_for_step(self, step: int, total_steps: int) -> float:
        """Cosine anneal entropy coefficient."""
        progress = step / max(total_steps - 1, 1)
        coef = self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        return coef

    def train_step(self, step: int, total_steps: int):
        """Run one full training step: self-play → gradient update."""
        self.model.eval()

        # === Phase 1: Self-play ===
        trajectories = self.runner.run()

        # Flatten all moves
        all_positions = torch.cat([t["positions"] for t in trajectories])
        all_players = torch.cat([t["players"] for t in trajectories])
        all_actions = torch.cat([t["actions"] for t in trajectories])
        all_rewards = torch.cat([t["rewards"] for t in trajectories])
        n_moves = len(all_actions)

        # Compute metrics
        game_lengths = [len(t["positions"]) for t in trajectories]
        black_wins = sum(
            1 for t in trajectories
            if t["rewards"][0].item() > 0
        )
        white_wins = sum(
            1 for t in trajectories
            if t["rewards"][0].item() < 0
        )
        illegal_count = sum(
            1 for t in trajectories
            if len(t["positions"]) > 0 and t["rewards"][-1].item() < 0
            and abs(t["rewards"][-1].item()) > 0.5
        )  # rough estimate: last move bad → could be illegal

        # === Phase 2: Training ===
        self.model.train()
        self.optimizer.zero_grad()

        entropy_coef = self._entropy_coef_for_step(step, total_steps)

        total_policy_loss = 0.0
        total_entropy = 0.0
        micro_batches = 0

        # Gradient accumulation over micro-batches
        perm = torch.randperm(n_moves)
        for start in range(0, n_moves, self.train_micro_batch):
            indices = perm[start : start + self.train_micro_batch]

            pos_batch = all_positions[indices].unsqueeze(0).to(self.device)  # (1, b)
            plr_batch = all_players[indices].unsqueeze(0).to(self.device)
            act_batch = all_actions[indices].to(self.device)
            rew_batch = all_rewards[indices].to(self.device)

            # Forward pass (FP16)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(pos_batch, plr_batch)  # (1, b, 225)
                logits = logits.squeeze(0)  # (b, 225)

            loss, policy_loss, entropy = reinforce_loss(
                logits, act_batch, rew_batch, entropy_coef
            )
            loss.backward()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            micro_batches += 1

        # Gradient clipping
        grad_norm_raw = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm_raw += p.grad.norm().item() ** 2
        grad_norm_raw = math.sqrt(grad_norm_raw)

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = math.sqrt(grad_norm)

        self.optimizer.step()

        # === Metrics ===
        metrics = {
            "loss/policy": total_policy_loss / max(micro_batches, 1),
            "loss/entropy": total_entropy / max(micro_batches, 1),
            "loss/total": (total_policy_loss + total_entropy) / max(micro_batches, 1),
            "game/avg_length": sum(game_lengths) / max(len(game_lengths), 1),
            "game/black_winrate": black_wins / max(len(trajectories), 1),
            "game/white_winrate": white_wins / max(len(trajectories), 1),
            "grad/norm": grad_norm,
            "grad/norm_raw": grad_norm_raw,
            "train/entropy_coef": entropy_coef,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "perf/n_moves": n_moves,
            "perf/n_games": len(trajectories),
        }

        return metrics

    def save_checkpoint(self, path: str, step: int):
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["step"]
