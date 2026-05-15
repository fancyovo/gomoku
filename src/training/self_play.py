import torch
import numpy as np
import time
import math

try:
    import gomoku_cpp
except ImportError:
    gomoku_cpp = None


def _floor_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


class SelfPlayRunner:
    def __init__(self, model, device, config: dict):
        self.model = model
        self.device = device
        self.games_per_step = config["games_per_step"]
        self.page_size = config["page_size"]
        self.base_batch = config["base_batch"]
        self.base_seq_len = config["base_seq_len"]

        if gomoku_cpp is None:
            raise RuntimeError("gomoku_cpp not built. Run: pip install -e .")

        self.pool = gomoku_cpp.GamePool(self.games_per_step)
        self.timing = {"python": 0.0, "cpp": 0.0}

    def _batch_size(self, seq_len: int) -> int:
        # KV cache grows to seq_len + page_size during decode
        effective = max(seq_len + self.page_size, 1)
        raw = self.base_batch * self.base_seq_len / effective
        return min(max(1, _floor_pow2(int(raw))), 16384)

    def run_one_wave(self):
        self.pool.reset_all()
        sequences = {i: [] for i in range(self.games_per_step)}
        trajectories = []
        page = self.page_size

        python_time = 0.0
        cpp_time = 0.0

        # Round counter for the "all games at same length" invariant
        while True:
            active_list = self.pool.active_indices()
            if not active_list:
                break

            # All active games have the same current length
            L = len(sequences[active_list[0]])
            batch_size = self._batch_size(L)

            for start in range(0, len(active_list), batch_size):
                micro_indices = active_list[start:start + batch_size]
                b = len(micro_indices)

                kv_cache = self.model.create_cache(max_games=b,
                                                      max_cache_len=L + page)
                local_idx = list(range(b))

                t0 = time.perf_counter()

                if L == 0:
                    # --- First round: L=0, empty boards ---
                    # Step 1: sample first move from learnable distribution (black)
                    first_act = self.model.sample_first_moves(b, self.device).cpu()

                    # Prefill this 1 token
                    pos_t = first_act.unsqueeze(1).to(self.device)
                    plr_t = torch.zeros(b, 1, dtype=torch.long, device=self.device)
                    logits = self.model.prefill(pos_t, plr_t, kv_cache, local_idx)

                    # Sample second move (white) from prefill output
                    probs = torch.softmax(logits, dim=-1)
                    second_act = torch.multinomial(probs, 1).squeeze(-1).cpu()

                    # all_actions[i] will grow to length `page`
                    all_actions = [[int(first_act[i]), int(second_act[i])] for i in range(b)]
                    n_decode = page - 2  # remaining after the 2 already-sampled moves
                else:
                    # --- L > 0: prefill full sequence, then decode ---
                    pos_list, plr_list = [], []
                    for idx in micro_indices:
                        seq = sequences[idx]
                        pos_list.append([p for p, _ in seq])
                        plr_list.append([pl for _, pl in seq])

                    pos_t = torch.tensor(pos_list, dtype=torch.long, device=self.device)
                    plr_t = torch.tensor(plr_list, dtype=torch.long, device=self.device)

                    logits = self.model.prefill(pos_t, plr_t, kv_cache, local_idx)
                    probs = torch.softmax(logits, dim=-1)
                    first_block_act = torch.multinomial(probs, 1).squeeze(-1).cpu()

                    all_actions = [[int(first_block_act[i])] for i in range(b)]
                    n_decode = page - 1

                # --- Common decode loop ---
                idx_tensor = torch.arange(b, device=self.device)
                for _ in range(n_decode):
                    # The token just sampled (input to decode) is the last in each list
                    current_pos = torch.tensor(
                        [all_actions[i][-1] for i in range(b)],
                        dtype=torch.long, device=self.device
                    )
                    # Actual global step of this token = L + len(all_actions[i]) - 1
                    current_step = L + len(all_actions[0]) - 1
                    current_plr = torch.full((b,), current_step % 2,
                                             dtype=torch.long, device=self.device)

                    logits = self.model.decode(
                        current_pos, current_plr, kv_cache, idx_tensor
                    )
                    probs = torch.softmax(logits, dim=-1)
                    new_act = torch.multinomial(probs, 1).squeeze(-1).cpu()
                    for i in range(b):
                        all_actions[i].append(int(new_act[i]))

                python_time += time.perf_counter() - t0

                # --- Send to C++ ---
                actions_block = np.array(all_actions, dtype=np.int32)
                assert actions_block.shape == (b, page)

                t1 = time.perf_counter()
                results = self.pool.execute_block(
                    np.array(micro_indices, dtype=np.int32), actions_block
                )
                cpp_time += time.perf_counter() - t1

                # --- Process C++ results ---
                for j, idx in enumerate(micro_indices):
                    end_step = int(results[j, 0])
                    result = int(results[j, 1])

                    # Build full timeline: L existing steps + this block's actions
                    pos_seq = []
                    plr_seq = []
                    if L > 0:
                        seq = sequences[idx]
                        pos_seq = [p for p, _ in seq]
                        plr_seq = [pl for _, pl in seq]

                    if end_step >= 0:
                        # Game ended within this block
                        actual_len = L + end_step + 1

                        # Compute reward sign from black's perspective
                        if result == 1:    r_black = 1.0
                        elif result == 2:  r_black = -1.0
                        else:              r_black = 0.0

                        rewards = []

                        # Rewards for existing history (L moves, all valid)
                        for _, pl in (sequences[idx] if L > 0 else []):
                            rewards.append(r_black if pl == 0 else -r_black)

                        # Rewards for this block's page actions
                        for k in range(page):
                            action = int(actions_block[j, k])
                            player = (L + k) % 2
                            pos_seq.append(action)
                            plr_seq.append(player)
                            if k <= end_step:
                                reward = r_black if player == 0 else -r_black
                            else:
                                reward = 0.0
                            rewards.append(reward)

                        trajectories.append({
                            "positions": torch.tensor(pos_seq, dtype=torch.long),
                            "players": torch.tensor(plr_seq, dtype=torch.long),
                            "actions": torch.tensor(pos_seq, dtype=torch.long),
                            "rewards": torch.tensor(rewards, dtype=torch.float32),
                            "actual_len": actual_len,
                            "result": result,
                        })

                        sequences.pop(idx, None)
                    else:
                        # Game continues — store this block's timeline
                        for k in range(page):
                            sequences[idx].append(
                                (int(actions_block[j, k]), (L + k) % 2)
                            )

                del kv_cache

        self.timing["python"] = python_time
        self.timing["cpp"] = cpp_time
        return trajectories

    def run(self):
        return self.run_one_wave()
