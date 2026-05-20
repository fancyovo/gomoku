#!/usr/bin/env python3
"""Run N training steps then analyze: game length curve, ELO curve, winrate heatmap."""

import torch, sys, os, time, math, numpy as np, argparse
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp


# ─── Config ─────────────────────────────────────────────────────

G = 512  # self-play batch size
M = 8    # leaves per round
S = 64   # rounds per move (total sims = M*S = 512)

ELO_G = 128   # ELO tournament batch size
ELO_M = 4     # smaller for speed
ELO_S = 4     # total = 16 sims

N_STEPS = 10

# ─── MCTS Self-Play ─────────────────────────────────────────────

def run_selfplay(model, device):
    G_ = G; pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
    mgr.c_puct = 1.0; mgr.dirichlet_eps = 0.25; mgr.dirichlet_alpha = 0.03
    mgr.leaves_per_game = M
    mgr.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
    kv = model.create_cache(max_games=G_, max_cache_len=250)

    fa = model.sample_first_moves(G_, device)
    model.prefill(fa.unsqueeze(1), torch.zeros(G_, 1, dtype=torch.long, device=device), kv, list(range(G_)))

    occupied = np.zeros((G_, 225), dtype=bool); finished_np = np.zeros(G_, dtype=bool)
    mcts_pols = [[] for _ in range(G_)]
    pos_hist = [[] for _ in range(G_)]; plr_hist = [[] for _ in range(G_)]
    pos_lens = np.zeros(G_, dtype=np.int32)

    for g in range(G_):
        a = int(fa[g].item()); occupied[g, a] = True; pos_lens[g] = 1
        pos_hist[g].append(a); plr_hist[g].append(0)
        r = gomoku_cpp.step(pool, g, a)
        if r: finished_np[g] = True; mgr.reset_game(g)
        else: mgr.apply_move(g, a, occupied[g])
    occ_gpu = torch.from_numpy(occupied).to(device)

    while True:
        active = np.where(~finished_np)[0]
        if len(active) == 0: break

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

        rp = mgr.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            pol = rp[g].copy(); pol[occupied[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~occupied[g])[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a
            mcts_pols[g].append(pol.copy())
            plr = pos_lens[g] % 2
            pos_hist[g].append(a); plr_hist[g].append(plr)
            occupied[g, a] = True; occ_gpu[g, a] = True; pos_lens[g] += 1

        dec_pos = torch.from_numpy(new_actions).to(device)
        plr_now = np.array([1 - (pos_lens[g] % 2) for g in active], dtype=np.int64)
        dec_plr = torch.from_numpy(plr_now).to(device)
        dec_slots = torch.from_numpy(active).to(device)
        model.decode(dec_pos, dec_plr, kv, dec_slots)
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished_np[g] = True; mgr.reset_game(g)
            else: mgr.apply_move(g, int(new_actions[i]), occupied[g])

    trajectories = []
    for g in range(G_):
        r = gomoku_cpp.get_result(pool, g)
        L = pos_lens[g]; pols = mcts_pols[g]
        val_t = np.zeros(L, dtype=np.float32)
        for i in range(L):
            plr = i % 2
            if r == 3: val_t[i] = 0.0
            elif r == 1: val_t[i] = 1.0 if plr == 0 else -1.0
            else: val_t[i] = 1.0 if plr == 1 else -1.0
        if len(pols) < L: pols.append(np.ones(225, dtype=np.float32) / 225)
        trajectories.append({
            "positions": np.array(pos_hist[g], dtype=np.int64),
            "players": np.array(plr_hist[g], dtype=np.int64),
            "actions": np.array(pos_hist[g], dtype=np.int64),
            "mcts_policies": np.array(pols, dtype=np.float32),
            "value_targets": val_t, "actual_len": L, "result": r,
        })
    del kv; torch.cuda.empty_cache()
    return trajectories


# ─── Training ────────────────────────────────────────────────────

def train_step(model, trajectories, device, batch_size=512):
    from training.augment import SYM_TABLE, N_SYMS
    _INV_SYM = [0, 3, 2, 1, 4, 5, 6, 7]
    INV_SYM_TABLE = SYM_TABLE[_INV_SYM]

    # Augment
    samples = []
    for traj in trajectories:
        L = traj["actual_len"]
        pos = torch.from_numpy(traj["positions"][:L].copy())
        plr = torch.from_numpy(traj["players"][:L].copy())
        act = torch.from_numpy(traj["actions"][:L].copy())
        pol = torch.from_numpy(traj["mcts_policies"][:L].copy())
        val = torch.from_numpy(traj["value_targets"][:L].copy())
        for s in range(N_SYMS):
            remap = SYM_TABLE[s]
            inv_remap = INV_SYM_TABLE[s]
            samples.append({
                "positions": remap[pos], "players": plr, "actions": remap[act],
                "mcts_policies": pol[:, inv_remap], "value_targets": val,
                "actual_len": L,
            })

    # DataLoader
    class DS(torch.utils.data.Dataset):
        def __init__(self, s): self.samples = s
        def __len__(self): return len(self.samples)
        def __getitem__(self, i): return self.samples[i]

    def collate(batch):
        max_len = max(s["positions"].shape[0] for s in batch)
        B_ = len(batch)
        pos = torch.zeros(B_, max_len, dtype=torch.long)
        plr = torch.zeros(B_, max_len, dtype=torch.long)
        pol = torch.zeros(B_, max_len, 225)
        val = torch.zeros(B_, max_len)
        mask = torch.zeros(B_, max_len, dtype=torch.bool)
        for i, s in enumerate(batch):
            L_ = s["actual_len"]
            pos[i, :L_] = s["positions"]; plr[i, :L_] = s["players"]
            pol[i, :L_] = s["mcts_policies"]; val[i, :L_] = s["value_targets"]
            mask[i, :L_] = True
        return {"positions": pos, "players": plr, "mcts_policies": pol,
                "value_targets": val, "mask": mask}

    dl = torch.utils.data.DataLoader(DS(samples), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    for batch in dl:
        pos = batch["positions"].to(device); plr = batch["players"].to(device)
        pol_t = batch["mcts_policies"].to(device); val_t = batch["value_targets"].to(device)
        m = batch["mask"].to(device)
        B_, L_ = pos.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            p, v = model(pos, plr)

        if L_ > 1:
            pp = p[:, :-1, :].contiguous(); vv = v[:, :-1].contiguous()
            tp = pol_t[:, :-1, :].contiguous(); tv = val_t[:, :-1].contiguous()
            pm = m[:, :-1].contiguous()

            # Loss in float32
            from training.loss import alphago_zero_loss, reinforce_loss
            loss, _, _ = alphago_zero_loss(
                pp.reshape(-1, 225).float(), tp.reshape(-1, 225),
                vv.reshape(-1).float(), tv.reshape(-1), pm.reshape(-1))

            # Train first_move_logits with REINFORCE using game outcome
            fm = model.first_move_logits.unsqueeze(0).expand(B_, -1)
            fl, _, _ = reinforce_loss(fm.float(), pos[:, 0], val_t[:, 0], m[:, 0])
            loss = loss + fl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

    model.eval()


# ─── ELO Tournament ──────────────────────────────────────────────

@torch.inference_mode()
def play_match(model_a, model_b, device, n_games=128):
    """Play n_games between two models using small MCTS. Returns (wins_a, wins_b, draws)."""
    G_ = min(ELO_G, n_games)
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    mgr_a = gomoku_cpp.MCTSManager(G_, seed_base=0)
    mgr_a.c_puct = 1.0; mgr_a.leaves_per_game = ELO_M
    mgr_a.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))

    mgr_b = gomoku_cpp.MCTSManager(G_, seed_base=1)
    mgr_b.c_puct = 1.0; mgr_b.leaves_per_game = ELO_M
    mgr_b.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))

    kva = model_a.create_cache(max_games=G_, max_cache_len=250)
    kvb = model_b.create_cache(max_games=G_, max_cache_len=250)

    fa_a = model_a.sample_first_moves(G_, device)
    fa_b = model_b.sample_first_moves(G_, device)

    occupied = np.zeros((G_, 225), dtype=bool)
    finished = np.zeros(G_, dtype=bool)
    winners = np.zeros(G_, dtype=np.int32)
    a_is_black = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)

    # First move: choose from black's model
    first_acts = np.zeros(G_, dtype=np.int64)
    for g in range(G_):
        fm = fa_a[g].item() if a_is_black[g] else fa_b[g].item()
        first_acts[g] = fm; occupied[g, fm] = True

    pos_batch_a = torch.tensor([[first_acts[g]] for g in range(G_) if a_is_black[g]], dtype=torch.long, device=device)
    pos_batch_b = torch.tensor([[first_acts[g]] for g in range(G_) if not a_is_black[g]], dtype=torch.long, device=device)

    for g in range(G_):
        a = int(first_acts[g])
        r = gomoku_cpp.step(pool, g, a)
        if r: finished[g] = True; winners[g] = r
        else:
            mgr_a.apply_move(g, a, np.zeros(225, dtype=bool))
            mgr_b.apply_move(g, a, np.zeros(225, dtype=bool))

    # Prefill with first move
    slots_all = list(range(G_))
    # Use model_a's cache for all games
    fa_t = torch.tensor(first_acts, dtype=torch.long, device=device).unsqueeze(1)
    model_a.prefill(fa_t, torch.zeros(G_, 1, dtype=torch.long, device=device), kva, slots_all)
    model_b.prefill(fa_t, torch.zeros(G_, 1, dtype=torch.long, device=device), kvb, slots_all)

    occ_gpu = torch.from_numpy(occupied).to(device)
    pos_hist = [list(first_acts)] * G_  # simplified, just for length tracking

    for move in range(1, 200):
        active = np.where(~finished)[0]
        if len(active) == 0: break

        # MCTS for both models (root eval first, then multi-leaf)
        for mgr, mdl, kv in [(mgr_a, model_a, kva), (mgr_b, model_b, kvb)]:
            st = torch.from_numpy(active).to(device)
            dp = torch.zeros(len(active), 1, dtype=torch.long, device=device)
            dplr = torch.full((len(active), 1), cp, dtype=torch.long, device=device)
            dl = torch.ones(len(active), dtype=torch.long, device=device)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(occ_gpu[active], -1e9)
            pp = torch.softmax(lp, -1).cpu().numpy().astype(np.float32)
            mgr.expand_roots(active.astype(np.int32), pp, lv.cpu().numpy().astype(np.float32))

            for sim in range(ELO_S):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pos_t = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(device)
                plr_t = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(device)
                lens_t = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(device)
                slots_t = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(device)
                lp, lv = mdl.evaluate_mcts_leaves(pos_t, plr_t, kv, slots_t, lens_t)
                torch.cuda.synchronize()
                occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(device).bool()
                lp = lp.masked_fill(occ_t, -1e9)
                pp = torch.softmax(lp, -1).cpu().numpy().astype(np.float32)
                mgr.expand_and_backup(vi.astype(np.int32), pp, lv.cpu().numpy().astype(np.float32))

        # Select moves: current player uses their model's policy
        cp = move % 2
        rp_a = mgr_a.get_root_policies()
        rp_b = mgr_b.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)

        for i, g in enumerate(active):
            pol = rp_a[g] if ((cp == 0) == a_is_black[g]) else rp_b[g]
            pol = pol.copy(); pol[occupied[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~occupied[g])[0]
                a = int(np.random.choice(legal)) if len(legal) > 0 else 0
            new_actions[i] = a; occupied[g, a] = True; occ_gpu[g, a] = True

        dec_pos = torch.from_numpy(new_actions).to(device)
        dec_plr = torch.full((len(active),), cp, dtype=torch.long, device=device)
        dec_slots = torch.from_numpy(active).to(device)
        model_a.decode(dec_pos, dec_plr, kva, dec_slots)
        model_b.decode(dec_pos, dec_plr, kvb, dec_slots)

        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; winners[g] = r
            else:
                mgr_a.apply_move(g, int(new_actions[i]), occupied[g])
                mgr_b.apply_move(g, int(new_actions[i]), occupied[g])

    wins_a = wins_b = draws = 0
    for g in range(G_):
        w = winners[g]
        if w == 1: wins_a += 1 if a_is_black[g] else 0; wins_b += 0 if a_is_black[g] else 1
        elif w == 2: wins_a += 0 if a_is_black[g] else 1; wins_b += 1 if a_is_black[g] else 0
        else: draws += 1

    del kva, kvb; torch.cuda.empty_cache()
    return wins_a, wins_b, draws


def compute_elo(match_results):
    names = set()
    for a, b, _, _ in match_results: names.add(a); names.add(b)
    names = sorted(names)
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(200):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            n = sa + sb
            if n == 0: continue
            da = (sa - ea * n) * (32.0 / n)
            elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo


# ─── Main ────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    model = GomokuTransformer(cfg).to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training: G={G} M={M} S={S} (total={M*S} sims)")
    print(f"ELO eval: G={ELO_G} M={ELO_M} S={ELO_S} (total={ELO_M*ELO_S} sims)")
    print(f"Steps: {N_STEPS}")

    os.makedirs("checkpoints/mcts_pipeline", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    avg_lens = []
    train_times = []
    sp_times = []

    for step in range(N_STEPS):
        print(f"\n{'='*50}")
        print(f"Step {step+1}/{N_STEPS}")

        # Self-play
        t0 = time.perf_counter()
        model.eval()
        trajectories = run_selfplay(model, device)
        torch.cuda.synchronize()
        t_sp = time.perf_counter() - t0

        lens = [t["actual_len"] for t in trajectories]
        avg_len = sum(lens) / len(lens)
        avg_lens.append(avg_len)
        sp_times.append(t_sp)

        # Training
        t0 = time.perf_counter()
        train_step(model, trajectories, device)
        torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        train_times.append(t_train)

        results = Counter(t["result"] for t in trajectories)
        print(f"  Self-play: {t_sp:.1f}s  Train: {t_train:.1f}s  "
              f"Avg len: {avg_len:.0f}  B={results.get(1,0)} W={results.get(2,0)}")

        # Save checkpoint
        ckpt_path = f"checkpoints/mcts_pipeline/step_{step:04d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # ── ELO Tournament ──
    print(f"\n{'='*50}")
    print("ELO Tournament (all checkpoint pairs)...")
    ckpts = sorted([f for f in os.listdir("checkpoints/mcts_pipeline") if f.endswith(".pt")])
    print(f"  Checkpoints: {len(ckpts)}")

    match_results = []
    winrate_matrix = np.zeros((len(ckpts), len(ckpts)))

    for i, ca in enumerate(ckpts):
        model_a = GomokuTransformer(cfg).to(device).eval()
        model_a.load_state_dict(torch.load(f"checkpoints/mcts_pipeline/{ca}", map_location=device))

        for j, cb in enumerate(ckpts):
            if i >= j: continue

            model_b = GomokuTransformer(cfg).to(device).eval()
            model_b.load_state_dict(torch.load(f"checkpoints/mcts_pipeline/{cb}", map_location=device))

            wa, wb, d = play_match(model_a, model_b, device)
            total = wa + wb + d
            score_a = wa + d * 0.5
            score_b = wb + d * 0.5
            match_results.append((ca, cb, score_a, score_b))

            wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5
            winrate_matrix[i, j] = wr
            winrate_matrix[j, i] = 1 - wr
            print(f"    {ca} vs {cb}: {wa}-{wb} (D={d}) WR={wr:.2%}")

            del model_b; torch.cuda.empty_cache()
        del model_a; torch.cuda.empty_cache()

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Game length over steps
    ax = axes[0]
    ax.plot(range(1, N_STEPS + 1), avg_lens, "o-", color="C0", markersize=6)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Avg Game Length")
    ax.set_title("Game Length vs Training")
    ax.grid(True, alpha=0.3)

    # 2. ELO over steps
    ax = axes[1]
    if match_results:
        elo = compute_elo(match_results)
        steps = []
        ratings = []
        for ckpt in ckpts:
            try:
                s = int(ckpt.split("_")[1].split(".")[0])
                steps.append(s)
                ratings.append(elo.get(ckpt, 1500))
            except: pass
        ax.plot(steps, ratings, "o-", color="C1", markersize=8, linewidth=2)
        ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Training Step"); ax.set_ylabel("ELO")
        ax.set_title("ELO Rating vs Training")
        ax.grid(True, alpha=0.3)

    # 3. Winrate heatmap
    ax = axes[2]
    n = len(ckpts)
    if n > 1:
        # Fill diagonal with 0.5
        for i in range(n): winrate_matrix[i, i] = 0.5
        im = ax.imshow(winrate_matrix, cmap="RdYlBu_r", vmin=0.3, vmax=0.7, aspect="auto")
        plt.colorbar(im, ax=ax, label="Win Rate (row vs col)")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([c.replace(".pt","") for c in ckpts], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([c.replace(".pt","") for c in ckpts], fontsize=7)
        ax.set_title("Win Rate Heatmap")

    plt.tight_layout()
    plt.savefig("output/mcts_pipeline_analysis.png", dpi=150)
    print(f"\nPlots saved to output/mcts_pipeline_analysis.png")

    # Summary
    print(f"\nTraining summary:")
    print(f"  Avg step time: {sum(sp_times)/len(sp_times):.0f}s (sp) + {sum(train_times)/len(train_times):.0f}s (train)")
    print(f"  Game lengths: {avg_lens}")


if __name__ == "__main__":
    main()
