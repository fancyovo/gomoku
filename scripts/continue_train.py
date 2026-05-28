#!/usr/bin/env python3
"""Continue training from step_000004 for 5 more steps, then ELO on all 10."""
import torch, sys, os, time, json, numpy as np, glob
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
from training.augment import SYM_TABLE, N_SYMS
_INV_SYM = [0, 3, 2, 1, 4, 5, 6, 7]
INV_SYM_TABLE = SYM_TABLE[_INV_SYM]
from training.loss import alphago_zero_loss, reinforce_loss
import gomoku_cpp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = 512; M = 8; S = 64; TRAIN_BATCH = 128; MAX_EPOCHS = 50; EARLY_STOP = 5
N_STEPS = 5; START_STEP = 5
ELO_G = 256; ELO_M = 4; ELO_S = 1
CHECKPOINT_DIR = "checkpoints/train_loop"; OUTPUT_DIR = "output"

print("Loading step_000004 checkpoint...")
cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
model = GomokuTransformer(cfg).to(DEVICE).eval()
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "step_000004.pt"), map_location=DEVICE)
model.load_state_dict(ckpt)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Load history
hist_file = os.path.join(OUTPUT_DIR, "train_length.json")
if os.path.exists(hist_file):
    with open(hist_file) as f: history = json.load(f)
else:
    history = {"avg_len": [], "sp_time": [], "tr_time": [], "epochs": [], "test_loss": [], "black_wr": []}


# ─── Self-Play (same as train_loop.py) ────────────────────
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
        cp = int(pos_lens[active[0]]) % 2
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
            plr_t2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
            lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
            slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
            lp, lv = model.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
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
    bw = int((results == 1).sum()); ww = int((results == 2).sum())
    dr = int((results == 3).sum()) + int((results == 0).sum())
    return trajectories, results, pos_lens, bw, ww, dr


# ─── Training ───────────────────────────────────────────
def train_with_early_stop(model, trajectories):
    samples = []
    for traj in trajectories:
        L = traj["actual_len"]
        pos = torch.from_numpy(traj["positions"][:L]); plr = torch.from_numpy(traj["players"][:L])
        pol = torch.from_numpy(traj["mcts_policies"][:L]); val = torch.from_numpy(traj["value_targets"][:L])
        for s in range(N_SYMS):
            remap = SYM_TABLE[s]
            inv_remap = INV_SYM_TABLE[s]
            samples.append({"positions": remap[pos], "players": plr,
                            "mcts_policies": pol[:, inv_remap], "value_targets": val, "actual_len": L})
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
        mask = torch.zeros(B_, max_len, dtype=torch.bool)
        for i, s in enumerate(batch):
            L_ = s["actual_len"]
            pos[i, :L_] = s["positions"]; plr[i, :L_] = s["players"]
            pol[i, :L_] = s["mcts_policies"]; val_t[i, :L_] = s["value_targets"]; mask[i, :L_] = True
        return {"positions": pos, "players": plr, "mcts_policies": pol, "value_targets": val_t, "mask": mask}
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
                    # Train first_move_logits with REINFORCE using game outcome
                    fm = model.first_move_logits.unsqueeze(0).expand(B_, -1)
                    fm_loss, _, _ = reinforce_loss(fm.float(), pos[:, 0], val_t[:, 0], m[:, 0])
                    loss = loss + fm_loss
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
                # Train first_move_logits with REINFORCE using game outcome
                fm = model.first_move_logits.unsqueeze(0).expand(B_, -1)
                fm_loss, _, _ = reinforce_loss(fm.float(), pos[:, 0], val_t[:, 0], m[:, 0])
                loss = loss + fm_loss
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


# ─── ELO ────────────────────────────────────────────────
def compute_elo(match_results):
    names = sorted(set(a for a, _, _, _ in match_results) | set(b for _, b, _, _ in match_results))
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(200):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n = sa + sb
            if n == 0: continue
            da = (sa - ea * n) * (32.0 / n); elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo

@torch.inference_mode()
def play_match(model_a, model_b):
    G_ = ELO_G
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    def make_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = ELO_M
        m.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
        return m
    mgr_a = make_mgr(); mgr_b = make_mgr()
    kva, brkva = model_a.create_cache(max_games=G_, max_cache_len=250)
    kvb, brkvb = model_b.create_cache(max_games=G_, max_cache_len=250)
    a_black = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    finished = np.zeros(G_, dtype=bool); winners = np.zeros(G_, dtype=np.int32)
    fa_a = model_a.sample_first_moves(G_, DEVICE); fa_b = model_b.sample_first_moves(G_, DEVICE)
    first_acts = np.zeros(G_, dtype=np.int64)
    p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)
    for g in range(G_):
        first_acts[g] = int(fa_a[g].item()) if a_black[g] else int(fa_b[g].item())
        p0[g, first_acts[g]] = True
    fa_t = torch.tensor(first_acts, dtype=torch.long, device=DEVICE).unsqueeze(1)
    plr_t = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    pa0, va0 = model_a.prefill(fa_t, plr_t, kva, brkva, list(range(G_)))
    pb0, vb0 = model_b.prefill(fa_t, plr_t, kvb, brkvb, list(range(G_)))
    occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)
    root_pol_a = pa0.float().clone()
    root_val_a = va0.float().clone()
    root_pol_b = pb0.float().clone()
    root_val_b = vb0.float().clone()
    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first_acts[g]))
        if r: finished[g] = True; winners[g] = r
        else:
            bp0 = np.zeros(225, dtype=bool); bp0[int(first_acts[g])] = True
            bp1 = np.zeros(225, dtype=bool)
            mgr_a.apply_move(g, int(first_acts[g]), bp0, bp1)
            mgr_b.apply_move(g, int(first_acts[g]), bp0, bp1)
    for move in range(1, 200):
        active = np.where(~finished)[0]
        if len(active) == 0: break
        cp = move % 2
        act_np = active.astype(np.int32)
        act_t = torch.from_numpy(active).to(DEVICE)
        for mgr, pol_buf, val_buf, kv in [(mgr_a, root_pol_a, root_val_a, kva),
                                          (mgr_b, root_pol_b, root_val_b, kvb)]:
            lp = pol_buf[active].masked_fill(occ_gpu[active], -1e9); torch.cuda.synchronize()
            lv = val_buf[active]
            mgr.expand_roots(act_np, torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
            for _ in range(ELO_S):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                plr_t2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                mdl = model_a if mgr is mgr_a else model_b
                lp, lv = mdl.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
                occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp = lp.masked_fill(occ_t, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
        rp_a = mgr_a.get_root_policies(); rp_b = mgr_b.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            use_a = ((cp == 0) == a_black[g])
            pol = (rp_a[g] if use_a else rp_b[g]).copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True
            occ_gpu[g, a] = True
        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.full((len(active),), cp, dtype=torch.long, device=DEVICE)
        dec_slots_t = torch.from_numpy(active).to(DEVICE)
        new_pa, new_va = model_a.decode(dec_pos, dec_plr, kva, brkva, dec_slots_t)
        new_pb, new_vb = model_b.decode(dec_pos, dec_plr, kvb, brkvb, dec_slots_t)
        root_pol_a[act_t] = new_pa.float()
        root_val_a[act_t] = new_va.float()
        root_pol_b[act_t] = new_pb.float()
        root_val_b[act_t] = new_vb.float()
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; winners[g] = r
            else:
                mgr_a.apply_move(g, int(new_actions[i]), p0[g], p1[g])
                mgr_b.apply_move(g, int(new_actions[i]), p0[g], p1[g])
    wins_a = wins_b = draws = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wins_a += 1 if a_black[g] else 0; wins_b += 0 if a_black[g] else 1
        elif w == 2: wins_a += 0 if a_black[g] else 1; wins_b += 1 if a_black[g] else 0
        else: draws += 1
    del kva, kvb; torch.cuda.empty_cache()
    return wins_a, wins_b, draws


# ─── Main ────────────────────────────────────────────────
for step in range(N_STEPS):
    real_step = START_STEP + step
    t0_step = time.perf_counter()

    # Self-play
    t0 = time.perf_counter(); model.eval()
    traj, results, pos_lens, bw, ww, dr = run_selfplay(model)
    torch.cuda.synchronize(); t_sp = time.perf_counter() - t0
    avg_len = pos_lens.mean(); rc = Counter(results)
    history["avg_len"].append(float(avg_len)); history["sp_time"].append(t_sp)
    history["black_wr"].append(bw / (bw + ww) if (bw + ww) > 0 else 0.5)

    # Training
    t0 = time.perf_counter()
    best_ep, best_loss = train_with_early_stop(model, traj)
    torch.cuda.synchronize(); t_tr = time.perf_counter() - t0
    history["tr_time"].append(t_tr); history["epochs"].append(best_ep); history["test_loss"].append(float(best_loss))

    step_time = time.perf_counter() - t0_step
    print(f"[step {real_step+1:4d}/10] len={avg_len:.0f} sp={t_sp:.0f}s tr={t_tr:.0f}s "
          f"epoch={best_ep} loss={best_loss:.4f} B={rc.get(1,0)} W={rc.get(2,0)} D={rc.get(3,0)} "
          f"BWR={bw/(bw+ww)*100:.0f}% total={step_time:.0f}s")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{real_step:06d}.pt")
    tmp_path = ckpt_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, ckpt_path)

    with open(hist_file, "w") as f: json.dump(history, f)

print(f"\nTraining complete ({N_STEPS} more steps). Running ELO tournament...")

# ─── ELO Tournament ─────────────────────────────────────
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
ckpt_names = [os.path.basename(c) for c in ckpts]
cache = {}

n_pairs = len(ckpt_names) * (len(ckpt_names) - 1) // 2
print(f"{len(ckpt_names)} checkpoints, {n_pairs} pairs")
for i, name_a in enumerate(ckpt_names):
    for name_b in ckpt_names[i + 1:]:
        sa = int(name_a.split("_")[1].split(".")[0])
        sb = int(name_b.split("_")[1].split(".")[0])
        key = f"{min(sa, sb)}_{max(sa, sb)}"
        print(f"  {name_a} vs {name_b} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        model_a = GomokuTransformer(cfg).to(DEVICE).eval()
        model_a.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name_a), map_location=DEVICE))
        model_b = GomokuTransformer(cfg).to(DEVICE).eval()
        model_b.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name_b), map_location=DEVICE))
        wa, wb, d = play_match(model_a, model_b)
        dt = time.perf_counter() - t0
        wr_val = wa / (wa + wb) if (wa + wb) > 0 else 0.5
        cache[key] = [wa, wb, d]
        print(f"{wa}-{wb} D={d} WR={wr_val:.2%} ({dt:.0f}s)")
        del model_a, model_b; torch.cuda.empty_cache()

# ─── 4-panel Plot ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ax.plot(history["avg_len"], "o-", color="C0", markersize=6)
ax.set_xlabel("Training Step"); ax.set_ylabel("Avg Game Length")
ax.set_title("Game Length vs Training"); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(history["black_wr"], "o-", color="C3", markersize=6)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Training Step"); ax.set_ylabel("Black Win Rate")
ax.set_title("Self-Play Black Win Rate"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
match_results = []
for key, val in cache.items():
    try:
        wa, wb, d = val
        s1, s2 = key.split("_")
        a_name = f"step_{int(s1):06d}.pt"
        b_name = f"step_{int(s2):06d}.pt"
        score_a = wa + d * 0.5; score_b = wb + d * 0.5
        match_results.append((a_name, b_name, score_a, score_b))
    except: pass
if match_results:
    elo = compute_elo(match_results)
    items = sorted((int(n.split("_")[1].split(".")[0]), r) for n, r in elo.items())
    steps_elo, ratings = zip(*items)
    ax.plot(steps_elo, ratings, ".-", color="C1", markersize=8, linewidth=2)
    ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Step"); ax.set_ylabel("ELO Rating")
    ax.set_title(f"ELO Curve ({ELO_G} games/pair)")
    ax.grid(True, alpha=0.3)

ax = axes[1, 1]
steps_list = sorted(set(int(n.split("_")[1].split(".")[0]) for n in ckpt_names))
n = len(steps_list)
if n >= 2:
    step_to_idx = {s: i for i, s in enumerate(steps_list)}
    wr = np.full((n, n), np.nan)
    for key, val in cache.items():
        try:
            wa, wb, d = val
            s1, s2 = key.split("_")
            i1, i2 = int(s1), int(s2)
            total = wa + wb
            if total > 0: wr[step_to_idx[i1], step_to_idx[i2]] = wa / total
        except: pass
    for ii in range(n):
        for jj in range(ii + 1, n):
            if not np.isnan(wr[ii, jj]):
                wr[jj, ii] = 1.0 - wr[ii, jj]
    im = ax.imshow(wr, cmap=plt.cm.RdYlBu_r, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(steps_list, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(steps_list, fontsize=7)
    ax.set_xlabel("Step"); ax.set_ylabel("Step")
    ax.set_title(f"Win Rate Heatmap ({ELO_G} games/pair)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "analysis.png")
plt.savefig(plot_path, dpi=150); plt.close()
print(f"\nPlot: {plot_path}")
print(f"Done. {len(cache)} ELO pairs.")
