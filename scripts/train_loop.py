#!/usr/bin/env python3
"""Training loop — self-play + training for N steps. Saves checkpoints and length curve."""

import torch, sys, os, time, json, numpy as np, argparse, glob
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
from training.augment import SYM_TABLE, N_SYMS
from training.loss import alphago_zero_loss
import gomoku_cpp

# Inverse symmetry lookup: policy transform needs new_pos → old_pos (inverse of SYM_TABLE)
# SYM_TABLE[s][old] = new. We need inv_map[new] = old.
# Inverse of D4 group: [id, rot270, rot180, rot90, flipH, flipV, transp, antidiag]
_INV_SYM = [0, 3, 2, 1, 4, 5, 6, 7]
INV_SYM_TABLE = SYM_TABLE[_INV_SYM]  # (8, 225), inv_map[new] = old

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Config ─────────────────────────────────────────────────────
G = 512          # self-play batch size
M = 8            # leaves per round
S = 64           # rounds → 512 total sims
TRAIN_BATCH = 128
MAX_EPOCHS = 100000
EARLY_STOP = 20
N_STEPS = 500
CHECKPOINT_DIR = "checkpoints/train_loop"
OUTPUT_DIR = "output"

# ─── Self-Play ──────────────────────────────────────────────────

def run_selfplay(model):
    pool = gomoku_cpp.GamePool(G); pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G, seed_base=np.random.randint(0, 2**31))
    mgr.c_puct = 1.0; mgr.dirichlet_eps = 0.25; mgr.dirichlet_alpha = 0.03
    mgr.leaves_per_game = M
    p0 = np.zeros((G, 225), dtype=bool); p1 = np.zeros((G, 225), dtype=bool)
    mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
    kv = model.create_cache(max_games=G, max_cache_len=250)

    fa = model.sample_first_moves(G, DEVICE)
    model.prefill(fa.unsqueeze(1), torch.zeros(G, 1, dtype=torch.long, device=DEVICE), kv, list(range(G)))

    pos_hist = [[] for _ in range(G)]; plr_hist = [[] for _ in range(G)]
    mcts_pols = [[] for _ in range(G)]
    pos_lens = np.zeros(G, dtype=np.int32)
    finished = np.zeros(G, dtype=bool); results = np.zeros(G, dtype=np.int32)

    for g in range(G):
        a = int(fa[g].item()); pos_hist[g].append(a); plr_hist[g].append(0)
        p0[g, a] = True; pos_lens[g] = 1
        r = gomoku_cpp.step(pool, g, a)
        if r: finished[g] = True; results[g] = r; mgr.reset_game(g)
        else: mgr.apply_move(g, a, p0[g], p1[g])
    occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)

    while True:
        active = np.where(~finished)[0]
        if len(active) == 0: break

        st = torch.from_numpy(active).to(DEVICE)
        cp = int(pos_lens[active[0]]) % 2  # current player for this turn
        dp = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
        dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=DEVICE)
        dl = torch.ones(len(active), dtype=torch.long, device=DEVICE)
        lp, lv = model.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
        lp = lp.masked_fill(occ_gpu[active], -1e9); torch.cuda.synchronize()
        mgr.expand_roots(active.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))

        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0: continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0: continue
            pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
            plr_t = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
            lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
            slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
            lp, lv = model.evaluate_mcts_leaves(pos_t, plr_t, kv, slots_t, lens_t); torch.cuda.synchronize()
            occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
            lp = lp.masked_fill(occ_t, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))

        rp = mgr.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64); new_plrs = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            pol = rp[g].copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a; new_plrs[i] = pos_lens[g] % 2
            pos_hist[g].append(a); plr_hist[g].append(new_plrs[i])
            mcts_pols[g].append(pol.copy())
            if new_plrs[i] == 0: p0[g, a] = True
            else: p1[g, a] = True
            pos_lens[g] += 1

        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.from_numpy(new_plrs).to(DEVICE)
        dec_slots = torch.from_numpy(active).to(DEVICE)
        model.decode(dec_pos, dec_plr, kv, dec_slots)
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; results[g] = r; mgr.reset_game(g)
            else: mgr.apply_move(g, int(new_actions[i]), p0[g], p1[g])

    trajectories = []
    for g in range(G):
        L = pos_lens[g]; pols = mcts_pols[g]
        r_val = results[g]; r_val = 3 if r_val == 0 else r_val
        val_t = np.zeros(L, dtype=np.float32)
        for i in range(L):
            plr = plr_hist[g][i]
            if r_val == 3: val_t[i] = 0.0
            elif r_val == 1: val_t[i] = 1.0 if plr == 0 else -1.0
            else: val_t[i] = 1.0 if plr == 1 else -1.0
        trajectories.append({"positions": np.array(pos_hist[g], dtype=np.int64),
                             "players": np.array(plr_hist[g], dtype=np.int64),
                             "actions": np.array(pos_hist[g], dtype=np.int64),
                             "mcts_policies": np.array(pols, dtype=np.float32),
                             "value_targets": val_t, "actual_len": L, "result": r_val})

    del kv; torch.cuda.empty_cache()
    black_wins = int((results == 1).sum())
    white_wins = int((results == 2).sum())
    draws = int((results == 3).sum()) + int((results == 0).sum())
    return trajectories, results, pos_lens, black_wins, white_wins, draws


# ─── Training ───────────────────────────────────────────────────

def train_with_early_stop(model, trajectories):
    samples = []
    for traj in trajectories:
        L = traj["actual_len"]
        pos = torch.from_numpy(traj["positions"][:L]); plr = torch.from_numpy(traj["players"][:L])
        pol = torch.from_numpy(traj["mcts_policies"][:L]); val = torch.from_numpy(traj["value_targets"][:L])
        for s in range(N_SYMS):
            remap = SYM_TABLE[s]  # old→new, for position tokens
            inv_remap = INV_SYM_TABLE[s]  # new→old, for policy vectors
            samples.append({"positions": remap[pos], "players": plr,
                            "mcts_policies": pol[:, inv_remap], "value_targets": val,
                            "actual_len": L, "sym": s})

    n = len(samples); n_train = int(n * 0.8)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    class DS(torch.utils.data.Dataset):
        def __init__(self, s, indices): self.s = [s[i] for i in indices]
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    def collate(batch):
        max_len = max(s["positions"].shape[0] for s in batch); B_ = len(batch)
        pos = torch.zeros(B_, max_len, dtype=torch.long); plr = torch.zeros(B_, max_len, dtype=torch.long)
        pol = torch.zeros(B_, max_len, 225); val_t = torch.zeros(B_, max_len)
        mask = torch.zeros(B_, max_len, dtype=torch.bool); syms = torch.zeros(B_, dtype=torch.long)
        for i, s in enumerate(batch):
            L_ = s["actual_len"]
            pos[i, :L_] = s["positions"]; plr[i, :L_] = s["players"]
            pol[i, :L_-1] = s["mcts_policies"]; val_t[i, :L_] = s["value_targets"]; mask[i, :L_] = True
            syms[i] = s.get("sym", 0)
        return {"positions": pos, "players": plr, "mcts_policies": pol, "value_targets": val_t, "mask": mask, "sym": syms}

    train_ds, test_ds = DS(samples, train_idx), DS(samples, test_idx)

    def eval_epoch(ds):
        model.eval(); dl = torch.utils.data.DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=False, collate_fn=collate)
        total, n_batch = 0, 0
        with torch.inference_mode():
            for batch in dl:
                pos = batch["positions"].to(DEVICE); plr = batch["players"].to(DEVICE)
                pol_t = batch["mcts_policies"].to(DEVICE); val_t = batch["value_targets"].to(DEVICE)
                m = batch["mask"].to(DEVICE); B_, L_ = pos.shape
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    p, v = model(pos, plr)
                if L_ > 1:
                    pp = p[:, :-1, :].float(); vv = v[:, :-1].float()
                    tp = pol_t[:, :-1, :]; tv = val_t[:, :-1]; pm = m[:, :-1]
                    loss, _, _ = alphago_zero_loss(pp.reshape(-1, 225), tp.reshape(-1, 225), vv.reshape(-1), tv.reshape(-1), pm.reshape(-1))
                    total += loss.item(); n_batch += 1
        return total / max(n_batch, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_test_loss = float('inf'); best_state = None; best_epoch = 0; no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); dl = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, collate_fn=collate)
        for batch in dl:
            pos = batch["positions"].to(DEVICE); plr = batch["players"].to(DEVICE)
            pol_t = batch["mcts_policies"].to(DEVICE); val_t = batch["value_targets"].to(DEVICE)
            m = batch["mask"].to(DEVICE); B_, L_ = pos.shape
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(pos, plr)
            if L_ > 1:
                pp = p[:, :-1, :].contiguous(); vv = v[:, :-1].contiguous()
                tp = pol_t[:, :-1, :].contiguous(); tv = val_t[:, :-1].contiguous(); pm = m[:, :-1].contiguous()
                loss, _, _ = alphago_zero_loss(pp.reshape(-1, 225).float(), tp.reshape(-1, 225), vv.reshape(-1).float(), tv.reshape(-1), pm.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()

        test_loss = eval_epoch(test_ds)
        if test_loss < best_test_loss:
            best_test_loss = test_loss; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; no_improve = 0
        else: no_improve += 1
        if no_improve >= EARLY_STOP: break

    model.load_state_dict(best_state); model.eval()
    return best_epoch, best_test_loss


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    args = parser.parse_args()
    global DEVICE; DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    total_steps = args.steps

    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)

    # Auto-resume from latest checkpoint
    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    start_step = 0
    history = {"avg_len": [], "sp_time": [], "tr_time": [], "epochs": [], "test_loss": [], "black_wr": []}
    length_file = os.path.join(OUTPUT_DIR, "train_length.json")

    if existing:
        latest = existing[-1]
        start_step = int(os.path.basename(latest).split("_")[1].split(".")[0]) + 1
        model = GomokuTransformer(cfg).to(DEVICE).eval()
        model.load_state_dict(torch.load(latest, map_location=DEVICE))
        if os.path.exists(length_file):
            with open(length_file) as f: history = json.load(f)
        print(f"Resumed from {os.path.basename(latest)} (start_step={start_step})")
    else:
        model = GomokuTransformer(cfg).to(DEVICE).eval()
        print(f"Starting from scratch")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: G={G} M={M} S={S} total_sims={M*S}")
    print(f"Training: batch={TRAIN_BATCH} early_stop={EARLY_STOP} max_epochs={MAX_EPOCHS}")
    print(f"Steps: {start_step} -> {start_step + total_steps - 1} ({total_steps} steps)")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print()

    for step in range(start_step, start_step + total_steps):
        t0_step = time.perf_counter()

        # Self-play
        t0 = time.perf_counter()
        model.eval()
        traj, results, pos_lens, bw, ww, dr = run_selfplay(model)
        torch.cuda.synchronize()
        t_sp = time.perf_counter() - t0

        avg_len = pos_lens.mean()
        rc = Counter(results)
        history["avg_len"].append(float(avg_len)); history["sp_time"].append(t_sp)
        history.setdefault("black_wr", []).append(bw / (bw + ww) if (bw + ww) > 0 else 0.5)

        # Training
        t0 = time.perf_counter()
        best_ep, best_loss = train_with_early_stop(model, traj)
        torch.cuda.synchronize()
        t_tr = time.perf_counter() - t0
        history["tr_time"].append(t_tr); history["epochs"].append(best_ep); history["test_loss"].append(float(best_loss))

        step_time = time.perf_counter() - t0_step
        print(f"[step {step:4d}] len={avg_len:.0f} sp={t_sp:.0f}s tr={t_tr:.0f}s "
              f"epoch={best_ep} loss={best_loss:.4f} B={rc.get(1,0)} W={rc.get(2,0)} D={rc.get(3,0)} "
              f"BWR={bw/(bw+ww)*100:.0f}% total={step_time:.0f}s")

        # Save checkpoint (atomic: write to tmp then rename)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
        tmp_path = ckpt_path + ".tmp"
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, ckpt_path)

        # Save length history and plot
        with open(length_file, "w") as f: json.dump(history, f)

        if (step + 1) % 5 == 0 or step == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
            ax1.plot(history["avg_len"], "o-", color="C0", markersize=4)
            ax1.set_xlabel("Step"); ax1.set_ylabel("Avg Game Length")
            ax1.set_title("Game Length vs Training"); ax1.grid(True, alpha=0.3)
            ax2.plot(history["black_wr"], "o-", color="C3", markersize=4)
            ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
            ax2.set_xlabel("Step"); ax2.set_ylabel("Black Win Rate")
            ax2.set_title("Self-Play Black Win Rate"); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "train_length.png"), dpi=100); plt.close()

    print(f"\nTraining complete. Checkpoints: {CHECKPOINT_DIR}/")
    print(f"Curves: {OUTPUT_DIR}/train_length.png")


if __name__ == "__main__":
    main()
