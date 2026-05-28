#!/usr/bin/env python3
"""Full MCTS self-play + training pipeline benchmark.
Usage: python scripts/bench_mcts_pipeline.py
"""

import torch, sys, os, time, math, numpy as np, argparse
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import ModelConfig, GomokuTransformer
from training.augment import SYM_TABLE, N_SYMS, augment_trajectory
from training.loss import alphago_zero_loss
import gomoku_cpp


# ─── MCTS Self-Play (integrated runner) ─────────────────────────

def run_mcts_selfplay(model, device, G, M, S):
    """Run MCTS self-play for G games, M leaves/round, S rounds. Returns list of trajectories."""
    pool = gomoku_cpp.GamePool(G); pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G, seed_base=np.random.randint(0, 2**31))
    mgr.c_puct = 1.0; mgr.dirichlet_eps = 0.25; mgr.dirichlet_alpha = 0.03
    mgr.leaves_per_game = M
    mgr.init_roots(np.zeros((G, 225), dtype=bool), np.zeros((G, 225), dtype=bool), np.zeros(G, dtype=np.int32))
    kv = model.create_cache(max_games=G, max_cache_len=250)

    fa = model.sample_first_moves(G, device)
    model.prefill(fa.unsqueeze(1), torch.zeros(G, 1, dtype=torch.long, device=device), kv, list(range(G)))

    occupied = np.zeros((G, 225), dtype=bool)
    finished_np = np.zeros(G, dtype=bool)
    mcts_policies = [[] for _ in range(G)]
    pos_hist = [[] for _ in range(G)]  # track move sequences
    plr_hist = [[] for _ in range(G)]
    pos_lens = np.zeros(G, dtype=np.int32)

    for g in range(G):
        a = int(fa[g].item()); occupied[g, a] = True; pos_lens[g] = 1
        pos_hist[g].append(a); plr_hist[g].append(0)
        r = gomoku_cpp.step(pool, g, a)
        if r: finished_np[g] = True; mgr.reset_game(g)
        else: mgr.apply_move(g, a, occupied[g])
    occ_gpu = torch.from_numpy(occupied).to(device)

    move = 0
    while True:
        active = np.where(~finished_np)[0]
        if len(active) == 0 or move > 200: break
        move += 1

        # Root eval
        st = torch.from_numpy(active).to(device)
        cp = int(pos_lens[active[0]]) % 2
        dp = torch.zeros(len(active), 1, dtype=torch.long, device=device)
        dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=device)
        dl = torch.ones(len(active), dtype=torch.long, device=device)
        lp, lv = model.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
        lp = lp.masked_fill(occ_gpu[active], -1e9)
        torch.cuda.synchronize()
        pp = torch.softmax(lp, -1).cpu().numpy().astype(np.float32)
        mgr.expand_roots(active.astype(np.int32), pp, lv.cpu().numpy().astype(np.float32))

        for sim in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0: continue
            vi = np.where(sel['valid_mask'])[0]
            n_eval = len(vi)
            if n_eval == 0: continue

            pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(device)
            plr_t = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(device)
            lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(device)
            slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(device)

            lp, lv = model.evaluate_mcts_leaves(pos_t, plr_t, kv, slots_t, lens_t)
            torch.cuda.synchronize()

            occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(device).bool()
            lp = lp.masked_fill(occ_t, -1e9)
            pp = torch.softmax(lp, -1).cpu().numpy().astype(np.float32)
            mgr.expand_and_backup(vi.astype(np.int32), pp, lv.cpu().numpy().astype(np.float32))

        # Select actions
        rp = mgr.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            pol = rp[g].copy(); pol[occupied[g]] = 0
            if pol.sum() > 0:
                a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~occupied[g])[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a
            mcts_policies[g].append(pol.copy())
            plr = (pos_lens[g]) % 2
            pos_hist[g].append(a); plr_hist[g].append(plr)
            occupied[g, a] = True; occ_gpu[g, a] = True; pos_lens[g] += 1

        # Decode
        dec_pos = torch.from_numpy(new_actions).to(device)
        plr_now = np.array([1 - (pos_lens[g] % 2) for g in active], dtype=np.int64)
        dec_plr = torch.from_numpy(plr_now).to(device)
        dec_slots = torch.from_numpy(active).to(device)
        model.decode(dec_pos, dec_plr, kv, dec_slots)

        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished_np[g] = True; mgr.reset_game(g)
            else: mgr.apply_move(g, int(new_actions[i]), occupied[g])

    # Build trajectories
    trajectories = []
    for g in range(G):
        r = gomoku_cpp.get_result(pool, g)
        L = pos_lens[g]
        if r == 0: r = 3 if L >= 225 else 3

        # Value targets from game outcome
        val_targets = np.zeros(L, dtype=np.float32)
        for i in range(L):
            plr = i % 2  # alternating players
            if r == 3: val_targets[i] = 0.0
            elif r == 1: val_targets[i] = 1.0 if plr == 0 else -1.0
            else: val_targets[i] = 1.0 if plr == 1 else -1.0

        # Pad mcts_policies
        pols = mcts_policies[g]
        if len(pols) < L:
            pols.append(np.ones(225, dtype=np.float32) / 225)

        trajectories.append({
            "positions": np.array(pos_hist[g], dtype=np.int64),
            "players": np.array(plr_hist[g], dtype=np.int64),
            "actions": np.array(pos_hist[g], dtype=np.int64),
            "mcts_policies": np.array(pols, dtype=np.float32),
            "value_targets": val_targets,
            "actual_len": L,
            "result": r,
        })

    del kv; torch.cuda.empty_cache()
    return trajectories, mgr


# ─── Data Augmentation for MCTS trajectories ─────────────────────

def augment_mcts_traj(traj):
    """Apply 8x symmetry augmentation to a trajectory with MCTS policies."""
    out = []
    L = traj["actual_len"]
    pos_seq = traj["positions"][:L]
    act_seq = traj["actions"][:L]
    mcts_pols = traj["mcts_policies"][:L]  # (L, 225)
    val_targs = traj["value_targets"][:L]

    for s in range(N_SYMS):
        remap = SYM_TABLE[s]  # (225,) int64
        # Remap positions and actions
        new_pos = remap[pos_seq].numpy()
        new_act = remap[act_seq].numpy()
        # Remap MCTS policies: permute the 225-dim vector
        new_pols = mcts_pols[:, remap.numpy()]

        out.append({
            "positions": torch.from_numpy(new_pos),
            "players": torch.from_numpy(traj["players"][:L].copy()),
            "actions": torch.from_numpy(new_act),
            "mcts_policies": torch.from_numpy(new_pols.copy()),
            "value_targets": torch.from_numpy(val_targs.copy()),
            "actual_len": L,
            "result": traj["result"],
        })
    return out


# ─── Dataset + DataLoader ────────────────────────────────────────

class MCTSDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories, augment=True):
        self.samples = []
        for traj in trajectories:
            if augment:
                self.samples.extend(augment_mcts_traj(traj))
            else:
                L = traj["actual_len"]
                self.samples.append({
                    "positions": torch.from_numpy(traj["positions"][:L].copy()),
                    "players": torch.from_numpy(traj["players"][:L].copy()),
                    "actions": torch.from_numpy(traj["actions"][:L].copy()),
                    "mcts_policies": torch.from_numpy(traj["mcts_policies"][:L].copy()),
                    "value_targets": torch.from_numpy(traj["value_targets"][:L].copy()),
                    "actual_len": L,
                    "result": traj["result"],
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_mcts(batch):
    max_len = max(s["positions"].shape[0] for s in batch)
    B = len(batch)

    pos = torch.zeros(B, max_len, dtype=torch.long)
    plr = torch.zeros(B, max_len, dtype=torch.long)
    act = torch.zeros(B, max_len, dtype=torch.long)
    pol = torch.zeros(B, max_len, 225, dtype=torch.float32)
    val = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        L = s["actual_len"]
        pos[i, :L] = s["positions"]
        plr[i, :L] = s["players"]
        act[i, :L] = s["actions"]
        pol[i, :L] = s["mcts_policies"]
        val[i, :L] = s["value_targets"]
        mask[i, :L] = True

    return {"positions": pos, "players": plr, "actions": act,
            "mcts_policies": pol, "value_targets": val, "mask": mask}


# ─── Training ────────────────────────────────────────────────────

def train_one_step(model, trajectories, device, batch_size=512, augment=True, lr=1e-3):
    """Run one training step on MCTS trajectories."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = MCTSDataset(trajectories, augment=augment)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_mcts, num_workers=0, pin_memory=True)

    total_loss = 0.0; total_policy = 0.0; total_value = 0.0; n_batches = 0

    for batch in dataloader:
        pos = batch["positions"].to(device, non_blocking=True)
        plr = batch["players"].to(device, non_blocking=True)
        pol_target = batch["mcts_policies"].to(device, non_blocking=True)
        val_target = batch["value_targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        B, L = pos.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_logits, value_preds = model(pos, plr)

        if L > 1:
            # Shift: predict at t from history 0..t-1
            pred_policy = policy_logits[:, :-1, :].contiguous()
            pred_value = value_preds[:, :-1].contiguous()
            target_policy = pol_target[:, 1:, :].contiguous()
            target_value = val_target[:, 1:].contiguous()
            pred_mask = mask[:, 1:].contiguous()

            loss, pol_loss, val_loss = alphago_zero_loss(
                pred_policy.reshape(-1, 225).float(),
                target_policy.reshape(-1, 225),
                pred_value.reshape(-1).float(),
                target_value.reshape(-1),
                pred_mask.reshape(-1))

            # First move loss (from first_move_logits)
            fm_logits = model.first_move_logits.unsqueeze(0).expand(B, -1)
            fm_loss, _, _ = alphago_zero_loss(
                fm_logits, pol_target[:, 0, :],
                value_preds[:, 0].float(), val_target[:, 0],
                mask[:, 0])
            loss = loss + fm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_policy += pol_loss.item()
            total_value += val_loss.item()
            n_batches += 1

    model.eval()
    return {"loss": total_loss / max(n_batches, 1),
            "policy_loss": total_policy / max(n_batches, 1),
            "value_loss": total_value / max(n_batches, 1),
            "n_batches": n_batches}


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--leaves", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--train_batch", type=int, default=512)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    G, M, S = args.batch_size, args.leaves, args.rounds
    device = torch.device(args.device)
    print(f"MCTS Pipeline: G={G} M={M} S={S} total_sims={M*S} train_bs={args.train_batch} augment={not args.no_augment}")

    # Create random model
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    model = GomokuTransformer(cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # ── Phase 1: Self-play ──
    t0 = time.perf_counter()
    trajectories, mgr = run_mcts_selfplay(model, device, G, M, S)
    torch.cuda.synchronize()
    t_sp = time.perf_counter() - t0
    n_games = len(trajectories)

    lens = [t["actual_len"] for t in trajectories]
    results = Counter(t["result"] for t in trajectories)
    print(f"\nPhase 1 - Self-play: {n_games} games in {t_sp:.1f}s ({n_games/t_sp:.1f} games/s)")
    print(f"  Lengths: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.1f}")
    print(f"  Results: B={results.get(1,0)} W={results.get(2,0)} D={results.get(3,0)}")
    print(f"  Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    torch.cuda.reset_peak_memory_stats()

    # ── Phase 2: Training ──
    n_aug = 1 if args.no_augment else 8
    total_samples = n_games * n_aug
    print(f"\nPhase 2 - Training: {n_games} traj × {n_aug} aug = {total_samples} samples")
    print(f"  Augment: {not args.no_augment}")

    t0 = time.perf_counter()
    metrics = train_one_step(model, trajectories, device,
                             batch_size=args.train_batch,
                             augment=not args.no_augment)
    torch.cuda.synchronize()
    t_train = time.perf_counter() - t0

    print(f"  Training: {t_train:.1f}s ({metrics['n_batches']} batches)")
    print(f"  Loss: total={metrics['loss']:.4f} policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}")

    # ── Summary ──
    total = t_sp + t_train
    print(f"\n{'='*55}")
    print(f"Total step time: {total:.1f}s (self-play={t_sp:.1f}s + train={t_train:.1f}s)")
    print(f"Self-play fraction: {t_sp/total*100:.0f}%")
    print(f"games/s (self-play only): {n_games/t_sp:.1f}")
    print(f"Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
