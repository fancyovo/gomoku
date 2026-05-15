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
        self.n_positions = model.config.n_positions

        if gomoku_cpp is None:
            raise RuntimeError("gomoku_cpp not built. Run: pip install -e .")

        self.pool = gomoku_cpp.GamePool(self.games_per_step)
        self.timing = {"python": 0.0, "cpp": 0.0}

        # Pre-allocated index tensor (max batch size)
        self._idx_cache = {}

    def _batch_size(self, seq_len: int) -> int:
        effective = max(seq_len + self.page_size, 1)
        raw = self.base_batch * self.base_seq_len / effective
        return min(max(1, _floor_pow2(int(raw))), 16384)

    def _get_idx(self, b: int) -> torch.Tensor:
        if b not in self._idx_cache:
            self._idx_cache[b] = torch.arange(b, device=self.device)
        return self._idx_cache[b]

    def run_one_wave(self):
        self.pool.reset_all()
        sequences = {i: [] for i in range(self.games_per_step)}
        trajectories = []
        page = self.page_size

        python_time = 0.0
        cpp_time = 0.0

        while True:
            active_list = self.pool.active_indices()
            if not active_list:
                break

            L = len(sequences[active_list[0]])
            batch_size = self._batch_size(L)

            for start in range(0, len(active_list), batch_size):
                micro_indices = active_list[start:start + batch_size]
                b = len(micro_indices)

                kv_cache = self.model.create_cache(max_games=b,
                                                    max_cache_len=L + page)
                local_idx = list(range(b))
                idx_t = self._get_idx(b)
                t0 = time.perf_counter()

                if L == 0:
                    first_act = self.model.sample_first_moves(b, self.device)
                    pos_t = first_act.unsqueeze(1)
                    plr_t = torch.zeros(b, 1, dtype=torch.long, device=self.device)
                    logits = self.model.prefill(pos_t, plr_t, kv_cache, local_idx)

                    occupied = torch.zeros(b, self.n_positions, dtype=torch.bool,
                                          device=self.device)
                    occupied[idx_t, first_act] = True

                    logits = logits.masked_fill(occupied, float('-inf'))
                    probs = torch.softmax(logits, dim=-1)
                    second_act = torch.multinomial(probs, 1).squeeze(-1)
                    occupied[idx_t, second_act] = True

                    # Store actions on GPU: (b, page)
                    actions_gpu = torch.zeros(b, page, dtype=torch.long, device=self.device)
                    actions_gpu[:, 0] = first_act
                    actions_gpu[:, 1] = second_act
                    cur_col = 2
                    n_decode = page - 2
                else:
                    pos_list, plr_list = [], []
                    for idx in micro_indices:
                        seq = sequences[idx]
                        pos_list.append([p for p, _ in seq])
                        plr_list.append([pl for _, pl in seq])

                    pos_t = torch.tensor(pos_list, dtype=torch.long, device=self.device)
                    plr_t = torch.tensor(plr_list, dtype=torch.long, device=self.device)

                    occupied = torch.zeros(b, self.n_positions, dtype=torch.bool,
                                          device=self.device)
                    occupied[idx_t.unsqueeze(1).expand(b, L).reshape(-1),
                             pos_t.reshape(-1)] = True

                    logits = self.model.prefill(pos_t, plr_t, kv_cache, local_idx)
                    logits = logits.masked_fill(occupied, float('-inf'))
                    probs = torch.softmax(logits, dim=-1)
                    first_block_act = torch.multinomial(probs, 1).squeeze(-1)
                    occupied[idx_t, first_block_act] = True

                    actions_gpu = torch.zeros(b, page, dtype=torch.long, device=self.device)
                    actions_gpu[:, 0] = first_block_act
                    cur_col = 1
                    n_decode = page - 1

                # Decode loop — everything stays on GPU
                for step in range(n_decode):
                    current_pos = actions_gpu[:, cur_col - 1]
                    current_step = L + cur_col - 1
                    current_plr = torch.full((b,), current_step % 2,
                                             dtype=torch.long, device=self.device)

                    logits = self.model.decode(
                        current_pos, current_plr, kv_cache, idx_t
                    )
                    logits = logits.masked_fill(occupied, float('-inf'))
                    probs = torch.softmax(logits, dim=-1)
                    new_act = torch.multinomial(probs, 1).squeeze(-1)
                    occupied[idx_t, new_act] = True
                    actions_gpu[:, cur_col] = new_act
                    cur_col += 1

                torch.cuda.synchronize()
                python_time += time.perf_counter() - t0

                # Transfer actions to CPU once
                actions_np = actions_gpu.cpu().numpy()
                del kv_cache

                t1 = time.perf_counter()
                results = self.pool.execute_block(
                    np.array(micro_indices, dtype=np.int32), actions_np
                )
                cpp_time += time.perf_counter() - t1

                # Process results
                for j, idx in enumerate(micro_indices):
                    end_step = int(results[j, 0])
                    result = int(results[j, 1])
                    end_reason = int(results[j, 2])

                    pos_seq = []
                    plr_seq = []
                    if L > 0:
                        seq = sequences[idx]
                        pos_seq = [p for p, _ in seq]
                        plr_seq = [pl for _, pl in seq]

                    if end_step >= 0:
                        actual_len = L + end_step + 1
                        r_black = 1.0 if result == 1 else (-1.0 if result == 2 else 0.0)

                        rewards = []
                        for _, pl in (sequences[idx] if L > 0 else []):
                            rewards.append(r_black if pl == 0 else -r_black)

                        for k in range(page):
                            action = int(actions_np[j, k])
                            player = (L + k) % 2
                            pos_seq.append(action)
                            plr_seq.append(player)
                            if k <= end_step:
                                rewards.append(r_black if player == 0 else -r_black)
                            else:
                                rewards.append(0.0)

                        trajectories.append({
                            "positions": torch.tensor(pos_seq, dtype=torch.long),
                            "players": torch.tensor(plr_seq, dtype=torch.long),
                            "actions": torch.tensor(pos_seq, dtype=torch.long),
                            "rewards": torch.tensor(rewards, dtype=torch.float32),
                            "actual_len": actual_len,
                            "result": result,
                            "end_reason": end_reason,
                        })
                        sequences.pop(idx, None)
                    else:
                        for k in range(page):
                            sequences[idx].append(
                                (int(actions_np[j, k]), (L + k) % 2)
                            )

        self.timing["python"] = python_time
        self.timing["cpp"] = cpp_time
        return trajectories

    def run(self):
        return self.run_one_wave()
