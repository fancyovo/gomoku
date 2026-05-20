import torch
import numpy as np
import time

try:
    import gomoku_cpp
except ImportError:
    gomoku_cpp = None

N_CELLS = 225


class MCTSSelfPlayRunner:
    def __init__(self, model, device, config: dict):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = config["mcts_batch_size"]
        self.num_simulations = config["mcts_simulations"]
        self.c_puct = config.get("c_puct", 1.0)
        self.dirichlet_eps = config.get("dirichlet_eps", 0.25)
        self.dirichlet_alpha = config.get("dirichlet_alpha", 0.03)
        self.root_temperature = config.get("root_temperature", 1.0)
        self.n_positions = model.config.n_positions
        self.max_cache_len = config.get("max_cache_len", 250)
        self.max_path_len = config.get("max_path_len", 40)

        if gomoku_cpp is None:
            raise RuntimeError("gomoku_cpp not built. Run: pip install -e .")

        self.pool = gomoku_cpp.GamePool(self.batch_size)
        self.timing = {}

    @torch.inference_mode()
    def run_one_wave(self):
        G = self.batch_size
        self.pool.reset_all()
        device = self.device

        # ── State: keep everything on GPU where possible ──
        pos_hist = [[] for _ in range(G)]
        plr_hist = [[] for _ in range(G)]
        mcts_policies = [[] for _ in range(G)]
        occupied_gpu = torch.zeros(G, N_CELLS, dtype=torch.bool, device=device)
        finished_cpu = [False] * G  # Python bool list, scanned once per move

        timing = {
            "mcts_select": 0.0, "mcts_eval": 0.0, "mcts_expand": 0.0,
            "game_decode": 0.0, "game_step": 0.0, "setup": 0.0,
        }

        t0 = time.perf_counter()

        mcts_mgr = gomoku_cpp.MCTSManager(G, seed_base=42)
        mcts_mgr.c_puct = self.c_puct
        mcts_mgr.dirichlet_eps = self.dirichlet_eps
        mcts_mgr.dirichlet_alpha = self.dirichlet_alpha
        mcts_mgr.init_roots(np.zeros((G, N_CELLS), dtype=bool),
                            np.zeros((G, N_CELLS), dtype=bool),
                            np.zeros(G, dtype=np.int32))

        kv = self.model.create_cache(max_games=G, max_cache_len=self.max_cache_len)

        # ── First moves (batched prefill) ──
        first_acts = self.model.sample_first_moves(G, device)
        for g in range(G):
            a = int(first_acts[g].item())
            pos_hist[g].append(a)
            plr_hist[g].append(0)
            occupied_gpu[g, a] = True

        pos_batch = first_acts.unsqueeze(1)
        plr_batch = torch.zeros(G, 1, dtype=torch.long, device=device)
        self.model.prefill(pos_batch, plr_batch, kv, list(range(G)))

        for g in range(G):
            a = int(first_acts[g].item())
            r = gomoku_cpp.step(self.pool, g, a)
            if r != 0:
                finished_cpu[g] = True
                mcts_mgr.reset_game(g)
            else:
                mcts_mgr.apply_move(g, a, occupied_gpu[g].cpu().numpy())

        timing["setup"] = time.perf_counter() - t0

        # ── Pre-allocated tensors for eval batch (max size = G) ──
        max_d = self.max_path_len
        new_pos_buf = torch.zeros(G, max_d, dtype=torch.long, device=device)
        new_plr_buf = torch.zeros(G, max_d, dtype=torch.long, device=device)
        path_lens_buf = torch.zeros(G, dtype=torch.long, device=device)
        slots_buf = torch.zeros(G, dtype=torch.long, device=device)

        active_indices_arr = np.zeros(G, dtype=np.int32)
        nonterm_to_active_arr = np.zeros(G, dtype=np.int32)

        # ── Main loop ──
        move_count = 1
        n_active = G

        while move_count < 225:
            # Build active list once per move
            n_active = 0
            for g in range(G):
                if not finished_cpu[g]:
                    active_indices_arr[n_active] = g
                    n_active += 1
            if n_active == 0:
                break

            active = active_indices_arr[:n_active]  # slice view for numpy indexing

            # ── MCTS simulations ──
            for sim in range(self.num_simulations):
                t_s = time.perf_counter()
                sel = mcts_mgr.select_all()
                timing["mcts_select"] += time.perf_counter() - t_s

                game_indices = sel["game_indices"]
                path_actions = sel["path_actions"]
                max_pl = sel["max_path_len"]

                if max_pl == 0:
                    # Roots not yet evaluated — use a dummy token to get model output
                    n_root = n_active
                    cp = int(cache.seq_lens[active[0]].item()) % 2 if len(active) > 0 else 0
                    dummy_pos = torch.zeros(n_root, 1, dtype=torch.long, device=device)
                    dummy_plr = torch.full((n_root, 1), cp, dtype=torch.long, device=device)
                    dummy_lens = torch.ones(n_root, dtype=torch.long, device=device)
                    slots_t = torch.from_numpy(active).to(device)

                    t_e = time.perf_counter()
                    policy_logits, values = self.model.evaluate_mcts_leaves(
                        dummy_pos, dummy_plr, kv, slots_t, dummy_lens)
                    torch.cuda.synchronize()
                    timing["mcts_eval"] += time.perf_counter() - t_e

                    # Mask occupied (already on GPU) then softmax
                    occ_mask = occupied_gpu[active]
                    policy_logits = policy_logits.masked_fill(occ_mask, -1e9)
                    pp = torch.softmax(policy_logits, dim=-1)
                    # Transfer to CPU for C++ (unavoidable — C++ needs CPU data)
                    pp_cpu = pp.cpu().numpy().astype(np.float32)
                    vv_cpu = values.cpu().numpy().astype(np.float32)

                    t_x = time.perf_counter()
                    mcts_mgr.expand_and_backup(
                        np.arange(n_root, dtype=np.int32), pp_cpu, vv_cpu)
                    timing["mcts_expand"] += time.perf_counter() - t_x
                    continue

                # ── Build eval batch in one pass ──
                n_eval = 0
                nta_idx = 0
                d_max = 0

                n_sel = sel["n_active"]
                for ai in range(n_sel):
                    g = game_indices[ai]
                    if finished_cpu[g]:
                        continue
                    pa = path_actions[ai]
                    d = len(pa)
                    if d == 0:
                        continue

                    # Fill pre-allocated buffers
                    slots_buf[n_eval] = g
                    path_lens_buf[n_eval] = d
                    root_last = plr_hist[g][-1] if plr_hist[g] else 0
                    for j, a in enumerate(pa):
                        new_pos_buf[n_eval, j] = a
                        new_plr_buf[n_eval, j] = (root_last + 1 + j) % 2
                    nonterm_to_active_arr[nta_idx] = g  # build nta inline
                    n_eval += 1
                    nta_idx += 1
                    if d > d_max:
                        d_max = d

                if n_eval == 0:
                    continue

                # Slice pre-allocated buffers
                new_pos = new_pos_buf[:n_eval, :d_max]
                new_plr = new_plr_buf[:n_eval, :d_max]
                path_lens = path_lens_buf[:n_eval]
                slots_t = slots_buf[:n_eval]
                nta = nonterm_to_active_arr[:n_eval]

                t_e = time.perf_counter()
                policy_logits, values = self.model.evaluate_mcts_leaves(
                    new_pos, new_plr, kv, slots_t, path_lens)
                torch.cuda.synchronize()
                timing["mcts_eval"] += time.perf_counter() - t_e

                # Build occupied mask on GPU: game occupied + path actions
                occ_mask = occupied_gpu[slots_t].clone()
                for i in range(n_eval):
                    d_i = int(path_lens[i].item())
                    if d_i > 0:
                        # Mark path actions as occupied in one vectorized scatter
                        actions = new_pos_buf[i, :d_i]  # (d_i,)
                        occ_mask[i, actions] = True

                policy_logits = policy_logits.masked_fill(occ_mask, -1e9)
                pp = torch.softmax(policy_logits, dim=-1)
                pp_cpu = pp.cpu().numpy().astype(np.float32)
                vv_cpu = values.cpu().numpy().astype(np.float32)

                t_x = time.perf_counter()
                mcts_mgr.expand_and_backup(nta, pp_cpu, vv_cpu)
                timing["mcts_expand"] += time.perf_counter() - t_x

            # ── Select actions ──
            root_policies = mcts_mgr.get_root_policies()

            decode_pos = np.zeros(n_active, dtype=np.int64)
            decode_plr = np.zeros(n_active, dtype=np.int64)
            decode_slots = np.zeros(n_active, dtype=np.int32)

            for i, g in enumerate(active):
                policy = root_policies[g]
                # Mask occupied
                rp_gpu = torch.from_numpy(policy).to(device)
                rp_gpu[occupied_gpu[g]] = 0.0
                rp_sum = rp_gpu.sum()
                if rp_sum > 0:
                    rp_gpu /= rp_sum
                    action = int(torch.multinomial(rp_gpu, 1).item())
                else:
                    # Fallback: pick from unoccupied
                    legal = torch.where(~occupied_gpu[g])[0]
                    if len(legal) > 0:
                        action = int(legal[torch.randint(0, len(legal), (1,))].item())
                    else:
                        action = 0

                mcts_policies[g].append(policy.copy())
                pos_hist[g].append(action)
                plr = (len(pos_hist[g]) - 1) % 2
                plr_hist[g].append(plr)
                occupied_gpu[g, action] = True

                decode_pos[i] = action
                decode_plr[i] = plr
                decode_slots[i] = g

            # ── Batch decode ──
            t_d = time.perf_counter()
            pos_t = torch.from_numpy(decode_pos).to(device)
            plr_t = torch.from_numpy(decode_plr).to(device)
            slots_t = torch.from_numpy(decode_slots.astype(np.int64)).to(device)
            self.model.decode(pos_t, plr_t, kv, slots_t)
            torch.cuda.synchronize()
            timing["game_decode"] += time.perf_counter() - t_d

            # ── Execute moves on C++ board ──
            for i, g in enumerate(active):
                action = int(decode_pos[i])
                t_s = time.perf_counter()
                r = gomoku_cpp.step(self.pool, g, action)
                timing["game_step"] += time.perf_counter() - t_s

                if r != 0:
                    finished_cpu[g] = True
                    mcts_mgr.reset_game(g)
                else:
                    mcts_mgr.apply_move(g, action, occupied_gpu[g].cpu().numpy())

            move_count += 1

        # ── Build trajectories ──
        trajectories = []
        for g in range(G):
            r = gomoku_cpp.get_result(self.pool, g)
            L = len(pos_hist[g])
            if r == 0:
                r = 3  # draw: board full or no winner
            traj = self._build_trajectory(pos_hist[g], plr_hist[g], mcts_policies[g], r)
            trajectories.append(traj)

        del kv
        torch.cuda.empty_cache()
        self.timing = timing
        return trajectories

    def _build_trajectory(self, pos_seq, plr_seq, mcts_pols, result):
        L = len(pos_seq)
        if L == 0:
            return None

        val_targets = []
        for plr in plr_seq:
            if result == 3:
                val_targets.append(0.0)
            elif result == 1:
                val_targets.append(1.0 if plr == 0 else -1.0)
            else:
                val_targets.append(1.0 if plr == 1 else -1.0)

        if not mcts_pols:
            mcts_pols = [np.ones(N_CELLS, dtype=np.float32) / N_CELLS]
        if len(mcts_pols) < L:
            mcts_pols.append(np.ones(N_CELLS, dtype=np.float32) / N_CELLS)

        mcts_arr = np.array(mcts_pols, dtype=np.float32)
        return {
            "positions": torch.tensor(pos_seq, dtype=torch.long),
            "players": torch.tensor(plr_seq, dtype=torch.long),
            "actions": torch.tensor(pos_seq, dtype=torch.long),
            "mcts_policies": torch.from_numpy(mcts_arr),
            "value_targets": torch.tensor(val_targets, dtype=torch.float32),
            "actual_len": L,
            "result": result,
        }

    def run(self):
        return self.run_one_wave()
