import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import math
import time

from .loss import reinforce_loss
from .self_play import SelfPlayRunner
from .dataset import create_dataloader


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.cfg = config

        self.optimizer = AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        self.entropy_coef = config["entropy_coef"]
        self.entropy_coef_min = config["entropy_coef_min"]
        self.loss_scale = config.get("loss_scale", 1.0)
        self.grad_clip = config["grad_clip"]
        self.augment = config.get("augment", True)

        self.runner = SelfPlayRunner(model, device, config)

    def _entropy_coef(self, step: int, total_steps: int) -> float:
        progress = step / max(total_steps - 1, 1)
        return self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (
            1.0 + math.cos(math.pi * progress)
        )

    def train_step(self, step: int, total_steps: int, max_batches: int | None = None,
                   ppl_interval: int = 10):
        """One training step.

        Args:
            ppl_interval: log perplexity every N batches.
        Returns:
            metrics dict with extra 'ppl_curve' and 'ppl_by_len'.
        """
        metrics = {}

        # Phase 1: Self-play
        t0 = time.perf_counter()
        self.model.eval()
        trajectories = self.runner.run_one_wave()
        metrics["perf/self_play_time"] = time.perf_counter() - t0
        metrics["perf/raw_games"] = len(trajectories)
        metrics["perf/python_infer"] = self.runner.timing["python"]
        metrics["perf/cpp_time_ms"] = self.runner.timing["cpp"] * 1000

        lens = [t["actual_len"] for t in trajectories]
        metrics["game/avg_len"] = sum(lens) / len(lens)
        metrics["game/max_len"] = max(lens)
        metrics["game/total_moves"] = sum(lens)
        n_black_wins = sum(1 for t in trajectories if t["result"] == 1)
        n_white_wins = sum(1 for t in trajectories if t["result"] == 2)
        metrics["game/black_winrate"] = n_black_wins / len(trajectories)
        metrics["game/white_winrate"] = n_white_wins / len(trajectories)

        # Phase 2: Training
        t0 = time.perf_counter()
        self.model.train()

        dataloader = create_dataloader(
            trajectories, batch_size=self.cfg["train_batch_size"],
            augment=self.augment, shuffle=True,
        )

        total_loss = 0.0
        total_policy = 0.0
        total_entropy = 0.0
        n_batches = 0
        n_valid_moves = 0
        ppl_curve = []  # (batch_idx, perplexity)
        loss_curve = []  # (batch_idx, loss)

        ent_coef = self._entropy_coef(step, total_steps)
        self.optimizer.zero_grad()

        for batch in dataloader:
            pos = batch["positions"].to(self.device)
            plr = batch["players"].to(self.device)
            act = batch["actions"].to(self.device)
            rew = batch["rewards"].to(self.device)
            mask = batch["mask"].to(self.device)

            B, L = pos.shape

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(pos, plr)  # (B, L, n_positions)

            # Perplexity logging
            if n_batches % ppl_interval == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits.float(), dim=-1)
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    ent = -(probs * log_probs).sum(dim=-1)  # (B, L)
                    # Perplexity per valid position
                    valid_ent = ent[mask]
                    if valid_ent.numel() > 0:
                        ppl = valid_ent.mean().exp().item()
                        ppl_curve.append((n_batches, ppl))

            # Flatten and compute loss
            logits_flat = logits.reshape(B * L, -1).float()
            act_flat = act.reshape(B * L)
            rew_flat = rew.reshape(B * L)
            mask_flat = mask.reshape(B * L)

            loss, policy_loss, entropy = reinforce_loss(
                logits_flat, act_flat, rew_flat, mask_flat, ent_coef, self.loss_scale,
            )
            loss.backward()

            total_loss += loss.item()
            if n_batches % ppl_interval == 0:
                loss_curve.append((n_batches, loss.item()))
            total_policy += policy_loss.item()
            total_entropy += entropy.item()
            n_valid_moves += mask_flat.sum().item()
            n_batches += 1

            if max_batches and n_batches >= max_batches:
                break

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        metrics["train/time"] = time.perf_counter() - t0
        metrics["train/batches"] = n_batches
        metrics["train/n_valid_moves"] = int(n_valid_moves)
        metrics["loss/total"] = total_loss / max(n_batches, 1)
        metrics["loss/policy"] = total_policy / max(n_batches, 1)
        metrics["loss/entropy"] = total_entropy / max(n_batches, 1)
        metrics["train/entropy_coef"] = ent_coef
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        avg_policy = total_policy / max(n_batches, 1)
        metrics["policy/perplexity"] = math.exp(max(avg_policy, -10))
        metrics["ppl_curve"] = ppl_curve
        metrics["loss_curve"] = loss_curve

        # Phase 3: Perplexity by sequence position (on one batch, no grad)
        ppl_by_len = self._eval_ppl_by_len(trajectories[:1024])
        metrics["ppl_by_len"] = ppl_by_len

        return metrics

    @torch.no_grad()
    def _eval_ppl_by_len(self, trajectories: list[dict]):
        """Compute mean perplexity at each sequence position."""
        self.model.eval()

        # Build a batch from trajectories (use raw, no augment)
        dataloader = create_dataloader(
            trajectories, batch_size=512, augment=False, shuffle=False,
        )
        batch = next(iter(dataloader))

        pos = batch["positions"].to(self.device)
        plr = batch["players"].to(self.device)
        mask = batch["mask"].to(self.device)

        B, L = pos.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(pos, plr).float()

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1)  # (B, L)

        # Average entropy per sequence position (over games that are that long)
        ppl_by_len = []
        for i in range(min(L, 225)):
            col_mask = mask[:, i]  # which games have this position
            if col_mask.sum() > 0:
                avg_ent = ent[col_mask, i].mean().item()
                ppl_by_len.append((i, math.exp(avg_ent)))
            else:
                break

        self.model.train()
        return ppl_by_len

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
