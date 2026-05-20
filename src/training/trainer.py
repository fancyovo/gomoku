import torch
import torch.nn as nn
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
        self.entropy_fixed = config.get("entropy_fixed", None)
        self.reward_decay_hl = config.get("reward_decay_half_life", None)
        self.loss_scale = config.get("loss_scale", 1.0)
        self.grad_clip = config["grad_clip"]
        self.augment = config.get("augment", True)

        self.runner = SelfPlayRunner(model, device, config)

    def _entropy_coef(self, step: int, total_steps: int) -> float:
        if self.entropy_fixed is not None:
            return self.entropy_fixed
        progress = step / max(total_steps - 1, 1)
        return self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply_reward_decay(self, trajectories: list[dict]):
        """Apply exponential reward decay in-place."""
        if self.reward_decay_hl is None:
            return
        hl = self.reward_decay_hl
        decay_factor = 0.5 ** (1.0 / hl)
        for traj in trajectories:
            L = traj["actual_len"]
            rewards = traj["rewards"]
            for i in range(L):
                dist_from_end = L - 1 - i
                rewards[i] *= decay_factor ** dist_from_end

    def train_step(self, step: int, total_steps: int, max_batches: int | None = None,
                   ppl_interval: int = 10):
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
        n_illegal = sum(1 for t in trajectories if t.get("end_reason") == 2)
        n_win = sum(1 for t in trajectories if t.get("end_reason") == 1)
        metrics["game/black_winrate"] = n_black_wins / len(trajectories)
        metrics["game/white_winrate"] = n_white_wins / len(trajectories)
        metrics["game/illegal_rate"] = n_illegal / len(trajectories)
        metrics["game/win_rate_5"] = n_win / len(trajectories)

        self._apply_reward_decay(trajectories)

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
        ppl_curve = []
        loss_curve = []

        ent_coef = self._entropy_coef(step, total_steps)

        for batch in dataloader:
            pos = batch["positions"].to(self.device, non_blocking=True)
            plr = batch["players"].to(self.device, non_blocking=True)
            act = batch["actions"].to(self.device, non_blocking=True)
            rew = batch["rewards"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            B, L = pos.shape

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = self.model(pos, plr)

            if L > 1:
                pred_logits = logits[:, :-1, :].contiguous()
                pred_act    = act[:, 1:].contiguous()
                pred_rew    = rew[:, 1:].contiguous()
                pred_mask   = mask[:, 1:].contiguous()

                # Build action mask on GPU with scatter + cummax
                n_pos = logits.size(-1)
                oh = torch.zeros(B, L, n_pos, dtype=torch.bool, device=self.device)
                oh.scatter_(2, pos.unsqueeze(-1), True)
                occ_cum = oh.cummax(dim=1).values
                action_mask = occ_cum[:, :-1, :]
            else:
                pred_logits = logits.new_empty(B, 0, logits.size(-1))
                pred_act    = act.new_empty(B, 0)
                pred_rew    = rew.new_empty(B, 0)
                pred_mask   = mask.new_empty(B, 0)
                action_mask = None

            if n_batches % ppl_interval == 0 and pred_logits.size(1) > 0:
                with torch.no_grad():
                    probs = torch.softmax(pred_logits.float(), dim=-1)
                    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
                    ent = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
                    valid_ent = ent[pred_mask]
                    if valid_ent.numel() > 0:
                        ppl_curve.append((n_batches, valid_ent.mean().exp().item()))

            pred_logits_flat = pred_logits.reshape(-1, pred_logits.size(-1)).float()
            pred_act_flat    = pred_act.reshape(-1)
            pred_rew_flat    = pred_rew.reshape(-1)
            pred_mask_flat   = pred_mask.reshape(-1)
            action_mask_flat = action_mask.reshape(-1, action_mask.size(-1)) if action_mask is not None else None

            loss, policy_loss, entropy = reinforce_loss(
                pred_logits_flat, pred_act_flat, pred_rew_flat, pred_mask_flat,
                ent_coef, self.loss_scale, action_mask=action_mask_flat,
            )

            first_mask = mask[:, 0]
            if first_mask.any():
                fm_logits = self.model.first_move_logits.unsqueeze(0).expand(B, -1)
                fm_loss, _, _ = reinforce_loss(
                    fm_logits, act[:, 0], rew[:, 0], first_mask,
                    ent_coef, self.loss_scale,
                )
                loss = loss + fm_loss

            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            if n_batches % ppl_interval == 0:
                loss_curve.append((n_batches, loss.item()))
            total_policy += policy_loss.item()
            total_entropy += entropy.item()
            n_valid_moves += pred_mask_flat.sum().item() + first_mask.sum().item()
            n_batches += 1

            if max_batches and n_batches >= max_batches:
                break

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

        ppl_by_len = self._eval_ppl_by_len(trajectories[:1024])
        metrics["ppl_by_len"] = ppl_by_len

        return metrics

    @torch.no_grad()
    def _eval_ppl_by_len(self, trajectories: list[dict]):
        self.model.eval()

        dataloader = create_dataloader(
            trajectories, batch_size=512, augment=False, shuffle=False,
        )
        batch = next(iter(dataloader))

        pos = batch["positions"].to(self.device)
        plr = batch["players"].to(self.device)
        mask = batch["mask"].to(self.device)
        B, L = pos.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = self.model(pos, plr)
            logits = logits.float()

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        ent = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)

        ppl_by_len = []
        for i in range(min(L - 1, 225)):
            next_mask = mask[:, i + 1]
            if next_mask.sum() > 0:
                avg_ent = ent[next_mask, i].mean().item()
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
