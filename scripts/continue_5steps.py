#!/usr/bin/env python3
"""Continue training from latest checkpoint for 5 more steps, then full ELO."""
import torch, sys, os, time, json, numpy as np, glob
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

import importlib.util
spec = importlib.util.spec_from_file_location("train_loop", os.path.join(os.path.dirname(__file__), "train_loop.py"))
train_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_loop)
run_selfplay = train_loop.run_selfplay
train_with_early_stop = train_loop.train_with_early_stop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 5
ELO_G, ELO_M, ELO_S = 128, 4, 16
CHECKPOINT_DIR, OUTPUT_DIR = "checkpoints/train_loop", "output"


@torch.inference_mode()
def play_match(model_a, model_b):
    G_ = ELO_G; pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    def make_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = ELO_M
        m.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
        return m
    mgr_a = make_mgr(); mgr_b = make_mgr()
    kva = model_a.create_cache(max_games=G_, max_cache_len=250)
    kvb = model_b.create_cache(max_games=G_, max_cache_len=250)
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
    model_a.prefill(fa_t, plr_t, kva, list(range(G_)))
    model_b.prefill(fa_t, plr_t, kvb, list(range(G_)))
    occ_gpu = torch.from_numpy(p0 | p1).to(DEVICE)
    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first_acts[g]))
        if r: finished[g] = True; winners[g] = r
        else:
            bp0 = np.zeros(225, dtype=bool); bp0[int(first_acts[g])] = True
            mgr_a.apply_move(g, int(first_acts[g]), bp0, np.zeros(225, dtype=bool))
            mgr_b.apply_move(g, int(first_acts[g]), bp0, np.zeros(225, dtype=bool))
    for move in range(1, 200):
        active = np.where(~finished)[0]
        if len(active) == 0: break
        cp = move % 2
        for mgr, mdl, kv in [(mgr_a, model_a, kva), (mgr_b, model_b, kvb)]:
            st = torch.from_numpy(active).to(DEVICE)
            dp = torch.zeros(len(active), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(active), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(occ_gpu[active], -1e9); torch.cuda.synchronize()
            mgr.expand_roots(active.astype(np.int32), torch.softmax(lp, -1).cpu().numpy().astype(np.float32), lv.cpu().numpy().astype(np.float32))
            for _ in range(ELO_S):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                plr_t2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
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
            else: p1[g, a] = True; occ_gpu[g, a] = True
        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.full((len(active),), cp, dtype=torch.long, device=DEVICE)
        model_a.decode(dec_pos, dec_plr, kva, torch.from_numpy(active).to(DEVICE))
        model_b.decode(dec_pos, dec_plr, kvb, torch.from_numpy(active).to(DEVICE))
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; winners[g] = r
            else: mgr_a.apply_move(g, int(new_actions[i]), p0[g], p1[g]); mgr_b.apply_move(g, int(new_actions[i]), p0[g], p1[g])
    wa = wb = dr = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wa += 1 if a_black[g] else 0; wb += 0 if a_black[g] else 1
        elif w == 2: wa += 0 if a_black[g] else 1; wb += 1 if a_black[g] else 0
        else: dr += 1
    del kva, kvb; torch.cuda.empty_cache()
    return wa, wb, dr


def compute_elo(match_results):
    names = sorted(set(a for a, _, _, _ in match_results) | set(b for _, b, _, _ in match_results))
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(200):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0)); n = sa + sb
            if n == 0: continue
            da = (sa - ea * n) * (32.0 / n); elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)

    # Load latest checkpoint
    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    if not existing:
        print("ERROR: No existing checkpoint found. Run train_5steps_fixed.py first.")
        return
    latest = existing[-1]
    start_step = int(os.path.basename(latest).split("_")[1].split(".")[0]) + 1
    model = GomokuTransformer(cfg).to(DEVICE).eval()
    model.load_state_dict(torch.load(latest, map_location=DEVICE))
    print(f"Resumed from {os.path.basename(latest)} (start_step={start_step})")

    # Load history
    length_file = os.path.join(OUTPUT_DIR, "train_length.json")
    if os.path.exists(length_file):
        with open(length_file) as f:
            history = json.load(f)
    else:
        history = {"avg_len": [], "sp_time": [], "tr_time": [], "epochs": [], "test_loss": [], "black_wr": []}

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Continue from step {start_step} for {N_STEPS} steps (→ step {start_step + N_STEPS - 1})")

    for step in range(start_step, start_step + N_STEPS):
        t0 = time.perf_counter(); model.eval()
        traj, results, pos_lens, bw, ww, dr = run_selfplay(model)
        torch.cuda.synchronize(); t_sp = time.perf_counter() - t0

        avg_len = pos_lens.mean(); rc = Counter(results)
        history["avg_len"].append(float(avg_len)); history["sp_time"].append(t_sp)
        history["black_wr"].append(bw / (bw + ww) if (bw + ww) > 0 else 0.5)

        t0 = time.perf_counter()
        best_ep, best_loss = train_with_early_stop(model, traj)
        torch.cuda.synchronize(); t_tr = time.perf_counter() - t0
        history["tr_time"].append(t_tr); history["epochs"].append(best_ep); history["test_loss"].append(float(best_loss))

        print(f"[step {step:4d}] len={avg_len:.0f} sp={t_sp:.0f}s tr={t_tr:.0f}s "
              f"epoch={best_ep} loss={best_loss:.4f} B={rc.get(1, 0)} W={rc.get(2, 0)} D={rc.get(3, 0)} "
              f"BWR={bw / (bw + ww) * 100:.0f}%", flush=True)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
        torch.save(model.state_dict(), ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)
        with open(length_file, "w") as f: json.dump(history, f)

    # Full ELO tournament
    print(f"\nRunning ELO tournament (all checkpoints)...")
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt")))
    names = [os.path.basename(c) for c in ckpts]

    # Only play pairs we don't already have cached
    cache = {}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            sa = int(na.split("_")[1].split(".")[0])
            sb = int(nb.split("_")[1].split(".")[0])
            key = f"{min(sa, sb)}_{max(sa, sb)}"
            print(f"  {na} vs {nb} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            ma = GomokuTransformer(cfg).to(DEVICE).eval(); ma.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, na), map_location=DEVICE))
            mb = GomokuTransformer(cfg).to(DEVICE).eval(); mb.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, nb), map_location=DEVICE))
            wa, wb, d = play_match(ma, mb); dt = time.perf_counter() - t0
            wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5; cache[key] = [wa, wb, d]
            print(f"{wa}-{wb} WR={wr:.1%} ({dt:.0f}s)")
            del ma, mb; torch.cuda.empty_cache()

    # Compute ELO
    results = []
    for key, val in cache.items():
        try:
            wa, wb, d = val; s1, s2 = key.split("_")
            results.append((f"step_{int(s1):06d}.pt", f"step_{int(s2):06d}.pt", wa + d * 0.5, wb + d * 0.5))
        except: pass
    elo = compute_elo(results)
    items = sorted((int(n.split("_")[1].split(".")[0]), r) for n, r in elo.items())

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Panel 1: Game length
    ax = axes[0, 0]
    ax.plot(history["avg_len"], "o-", color="C0", markersize=6)
    ax.set_xlabel("Step"); ax.set_ylabel("Avg Game Length"); ax.set_title("Game Length"); ax.grid(True, alpha=0.3)
    # Panel 2: Black WR
    ax = axes[0, 1]
    ax.plot(history["black_wr"], "o-", color="C3", markersize=6)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Step"); ax.set_ylabel("Black Win Rate"); ax.set_title("Self-Play Black WR"); ax.grid(True, alpha=0.3)
    # Panel 3: ELO
    ax = axes[1, 0]
    if items:
        se, ra = zip(*items)
        ax.plot(se, ra, ".-", color="C1", markersize=8, linewidth=2)
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Step"); ax.set_ylabel("ELO"); ax.set_title(f"ELO ({ELO_G} games/pair)"); ax.grid(True, alpha=0.3)
    # Panel 4: Heatmap
    ax = axes[1, 1]
    steps_list = sorted(set(int(n.split("_")[1].split(".")[0]) for n in names)); n = len(steps_list)
    if n >= 2:
        sti = {s: i for i, s in enumerate(steps_list)}; wr_mat = np.full((n, n), np.nan)
        for key, val in cache.items():
            try:
                wa, wb, d = val; s1, s2 = key.split("_"); i1, i2 = int(s1), int(s2)
                if wa + wb > 0: wr_mat[sti[i1], sti[i2]] = wa / (wa + wb)
            except: pass
        for ii in range(n):
            for jj in range(ii + 1, n):
                if not np.isnan(wr_mat[ii, jj]): wr_mat[jj, ii] = 1.0 - wr_mat[ii, jj]
        im = ax.imshow(wr_mat, cmap=plt.cm.RdYlBu_r, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(steps_list, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(steps_list, fontsize=7)
        ax.set_xlabel("Step"); ax.set_ylabel("Step"); ax.set_title("Win Rate Heatmap")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "analysis_10steps.png"), dpi=150); plt.close()

    print(f"\nELO (all {len(items)} checkpoints):")
    for s, r in items:
        print(f"  Step {s}: ELO={r:.0f}")
    if len(items) >= 2:
        print(f"  ΔELO (step{items[0][0]}→step{items[-1][0]}) = {items[-1][1] - items[0][1]:.0f}")
    print(f"\nPlot: {OUTPUT_DIR}/analysis_10steps.png")


if __name__ == "__main__":
    main()
