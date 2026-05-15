import torch
import numpy as np
from collections import defaultdict

try:
    import gomoku_cpp
except ImportError:
    gomoku_cpp = None


@torch.inference_mode()
def _infer_batch(model, positions_list, players_list, device):
    """Pack variable-length sequences → batch inference → sampled actions."""
    max_len = max(p.shape[0] for p in positions_list)
    b = len(positions_list)

    pos_pad = torch.full((b, max_len), 0, dtype=torch.long, device=device)
    plr_pad = torch.full((b, max_len), 0, dtype=torch.long, device=device)

    for i, (pos, plr) in enumerate(zip(positions_list, players_list)):
        seq_len = pos.shape[0]
        pos_pad[i, :seq_len] = pos
        plr_pad[i, :seq_len] = plr

    actions = model.sample_actions(pos_pad, plr_pad)
    return actions.cpu().numpy()


class SelfPlayRunner:
    def __init__(self, model, device, games_per_step, infer_micro_batch, seed=42):
        self.model = model
        self.device = device
        self.games_per_step = games_per_step
        self.infer_micro_batch = infer_micro_batch

        if gomoku_cpp is None:
            raise RuntimeError("gomoku_cpp module not built. Run: pip install -e .")

        self.mgr = gomoku_cpp.GameManager(games_per_step, seed)

    def run(self):
        """Run self-play until we have games_per_step completed games.

        Returns:
            trajectories: list of dicts with keys:
                positions: (seq_len,) tensor — position indices
                players:   (seq_len,) tensor — player ids
                actions:   (seq_len,) tensor — chosen actions (= positions for next step)
                rewards:   (seq_len,) tensor — +1/-1 per move
        """
        trajectories = []
        # Track ongoing games: pool_idx → (positions[], players[])
        ongoing: dict[int, tuple[list[int], list[int]]] = {}

        for idx in self.mgr.active_indices:
            ongoing[idx] = ([], [])

        while len(trajectories) < self.games_per_step:
            active = list(ongoing.keys())

            # Micro-batched inference
            all_actions = {}
            for i in range(0, len(active), self.infer_micro_batch):
                batch_indices = active[i : i + self.infer_micro_batch]

                positions_list = []
                players_list = []
                for idx in batch_indices:
                    p, pl = ongoing[idx]
                    positions_list.append(torch.tensor(p, dtype=torch.long))
                    players_list.append(torch.tensor(pl, dtype=torch.long))

                sampled = _infer_batch(
                    self.model, positions_list, players_list, self.device
                )
                for j, idx in enumerate(batch_indices):
                    all_actions[idx] = int(sampled[j])

            # Apply actions via C++
            results = self.mgr.step(active, [all_actions[i] for i in active])

            # Update ongoing sequences
            for idx in active:
                action = all_actions[idx]
                positions, players = ongoing[idx]
                board = self.mgr.boards[idx]  # need to access board state
                # We need current_player BEFORE the move was applied
                # The board stores it, let's get it from move count parity
                current_player = len(positions) % 2
                positions.append(action)
                players.append(current_player)

            # Process finished games
            for finished_idx in results:
                pos_seq, plr_seq = ongoing.pop(finished_idx)
                if len(pos_seq) == 0:
                    continue

                board = self.mgr.boards[finished_idx]
                result = board.result

                # Compute rewards from black's perspective
                if result == 1:  # black win
                    r_black = 1.0
                elif result == 2:  # white win
                    r_black = -1.0
                else:  # draw
                    r_black = 0.0

                rewards = []
                for pid in plr_seq:
                    rewards.append(r_black if pid == 0 else -r_black)

                # If the last move was illegal, it gets penalized
                # (illegal moves cause immediate loss, so reward is already correct)

                trajectories.append({
                    "positions": torch.tensor(pos_seq, dtype=torch.long),
                    "players": torch.tensor(plr_seq, dtype=torch.long),
                    "actions": torch.tensor(pos_seq, dtype=torch.long),
                    "rewards": torch.tensor(rewards, dtype=torch.float32),
                })

            # Replenish
            added = self.mgr.replenish()
            for idx in self.mgr.active_indices:
                if idx not in ongoing:
                    ongoing[idx] = ([], [])

            if len(trajectories) >= self.games_per_step:
                break

        return trajectories[: self.games_per_step]
