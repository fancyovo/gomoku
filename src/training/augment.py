"""Data augmentation via the 8 symmetries of a square (D4 dihedral group).

Each symmetry maps a 15×15 board position to a new position.
Precomputed lookup table: sym_pos[sym][old_pos] = new_pos.
"""

import torch
import numpy as np

BOARD_SIZE = 15
N_CELLS = BOARD_SIZE * BOARD_SIZE  # 225
N_SYMS = 8


def _build_sym_table():
    """Build [8][225] lookup: sym_idx → (old_pos → new_pos)."""
    table = np.zeros((N_SYMS, N_CELLS), dtype=np.int64)
    for pos in range(N_CELLS):
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        maps = [
            (r, c),                    # 0: identity
            (c, BOARD_SIZE - 1 - r),   # 1: rotate 90°
            (BOARD_SIZE - 1 - r,
             BOARD_SIZE - 1 - c),      # 2: rotate 180°
            (BOARD_SIZE - 1 - c, r),   # 3: rotate 270°
            (r, BOARD_SIZE - 1 - c),   # 4: flip horizontal
            (BOARD_SIZE - 1 - r, c),   # 5: flip vertical
            (c, r),                    # 6: transpose (main diagonal)
            (BOARD_SIZE - 1 - c,
             BOARD_SIZE - 1 - r),      # 7: anti-diagonal
        ]
        for s, (nr, nc) in enumerate(maps):
            table[s, pos] = nr * BOARD_SIZE + nc
    return torch.from_numpy(table)


SYM_TABLE = _build_sym_table()  # (8, 225)


def augment_trajectory(traj: dict, n_syms: int = N_SYMS) -> list[dict]:
    """Apply symmetry augmentations to one trajectory.

    Args:
        traj: dict with keys 'positions', 'players', 'actions', 'rewards',
              'actual_len', 'result'. All tensors are 1D of shape (padded_len,).
        n_syms: number of symmetries to use (default 8).

    Returns:
        list of n_syms dicts, each with the same structure.
        positions and actions are remapped; players, rewards, result unchanged.
    """
    out = []
    pos = traj["positions"]  # (L,)
    act = traj["actions"]    # (L,)

    for s in range(n_syms):
        remap = SYM_TABLE[s]  # (225,)
        out.append({
            "positions": remap[pos],
            "players": traj["players"],
            "actions": remap[act],
            "rewards": traj["rewards"],
            "actual_len": traj["actual_len"],
            "result": traj["result"],
        })
    return out
