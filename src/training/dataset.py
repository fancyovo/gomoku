"""Gomoku dataset: trajectories → training batches with padding and collation."""

import torch
from torch.utils.data import Dataset, DataLoader

from .augment import augment_trajectory, N_SYMS


class GomokuDataset(Dataset):
    """Wraps a list of trajectories, optionally with symmetry augmentation.

    Each item: dict with 'positions', 'players', 'actions', 'rewards'.
    All are 1D tensors of equal length (= padded_len, which is a multiple
    of page_size; only the first actual_len steps have non-zero reward).
    """

    def __init__(self, trajectories: list[dict], augment: bool = True):
        self.samples = []
        for traj in trajectories:
            if augment:
                self.samples.extend(augment_trajectory(traj, N_SYMS))
            else:
                self.samples.append(traj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "positions": s["positions"].clone(),
            "players": s["players"].clone(),
            "actions": s["actions"].clone(),
            "rewards": s["rewards"].clone(),
            "actual_len": s["actual_len"],
            "result": s["result"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length trajectories to max_len in batch.

    Returns dict with:
        positions: (B, max_len)  int64
        players:   (B, max_len)  int64
        actions:   (B, max_len)  int64
        rewards:   (B, max_len)  float32
        mask:      (B, max_len)  bool  — True for valid positions
    """
    max_len = max(s["positions"].shape[0] for s in batch)
    B = len(batch)

    pos = torch.zeros(B, max_len, dtype=torch.long)
    plr = torch.zeros(B, max_len, dtype=torch.long)
    act = torch.zeros(B, max_len, dtype=torch.long)
    rew = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        L = s["positions"].shape[0]
        pos[i, :L] = s["positions"]
        plr[i, :L] = s["players"]
        act[i, :L] = s["actions"]
        rew[i, :L] = s["rewards"]
        # Mark valid positions (actual game steps + padding for page alignment)
        # Only the first actual_len positions have non-zero reward
        mask[i, :s["actual_len"]] = True

    return {
        "positions": pos,
        "players": plr,
        "actions": act,
        "rewards": rew,
        "mask": mask,
    }


def create_dataloader(trajectories: list[dict], batch_size: int,
                       augment: bool = True, shuffle: bool = True,
                       num_workers: int = 0) -> DataLoader:
    """Create a DataLoader from trajectories.

    Args:
        trajectories: list of trajectory dicts from SelfPlayRunner.
        batch_size: training batch size (e.g., 2048).
        augment: whether to apply 8× symmetry augmentation.
        shuffle: whether to shuffle.
        num_workers: number of CPU workers for data loading.
    """
    dataset = GomokuDataset(trajectories, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
