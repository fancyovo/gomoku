#!/usr/bin/env python3
"""Continuous ELO monitor: watches for new checkpoints, evaluates against previous ones.

- Full MCTS matches AND policy-only matches
- Reverse order: step_i vs step_{i-1}, step_{i-2}, ..., noisy_uniform
- Separate ELO curves for full and policy-only
- Replots after each checkpoint is fully evaluated
"""
import argparse, json, os, sys, time, glob
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G_ELO, M_ELO, S_ELO = 256, 4, 16
N_CELLS = 225
DEVICE = None
CACHE_FILE = None


# ── NoisyUniform ───────────────────────────────────────────────
class NoisyUniform:
    def __init__(self, cfg=None):
        self.config = cfg
    def create_cache(self, max_games, max_cache_len=250):
        m = GomokuTransformer(self.config).to(DEVICE)
        return m.create_cache(max_games, max_cache_len)
    def sample_first_moves(self, bs, dev):
        return torch.randint(0, 225, (bs,), device=dev)
    def prefill(self, pos, plr, cache, indices):
        return (torch.randn(pos.shape[0], 225, device=DEVICE) * 0.02,
                torch.zeros(pos.shape[0], device=DEVICE))
    def decode(self, pos, plr, cache, indices):
        cache.advance(indices)
        return (torch.randn(len(indices), 225, device=DEVICE) * 0.02,
                torch.zeros(len(indices), device=DEVICE))
    def evaluate_mcts_leaves(self, pos, plr, cache, indices, plen):
        return (torch.randn(pos.shape[0], 225, device=DEVICE) * 0.02,
                torch.zeros(pos.shape[0], device=DEVICE))


# ── Full MCTS match ────────────────────────────────────────────
@torch.inference_mode()
def play_match_mcts(ma, mb):
    G_ = G_ELO; M_ = M_ELO; S_ = S_ELO
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    def mm():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = M_; m.dirichlet_eps = 0.0
        m.init_roots(np.zeros((G_, 225), dtype=bool),
                     np.zeros((G_, 225), dtype=bool),
                     np.zeros(G_, dtype=np.int32))
        return m

    mga = mm(); mgb = mm()
    kva = ma.create_cache(max_games=G_); kvb = mb.create_cache(max_games=G_)
    ab = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
    fin = np.zeros(G_, dtype=bool); res = np.zeros(G_, dtype=np.int32)
    p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)

    fa_a = ma.sample_first_moves(G_, DEVICE)
    fa_b = mb.sample_first_moves(G_, DEVICE)
    fa = np.zeros(G_, dtype=np.int64)
    for g in range(G_):
        fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
        p0[g, fa[g]] = True

    fat = torch.tensor(fa, dtype=torch.long, device=DEVICE).unsqueeze(1)
    pl0 = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    ma.prefill(fat, pl0, kva, list(range(G_)))
    mb.prefill(fat, pl0, kvb, list(range(G_)))
    og = torch.from_numpy(p0 | p1).to(DEVICE)

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(fa[g]))
        if r: fin[g] = True; res[g] = r
        else:
            bp = np.zeros(225, dtype=bool); bp[int(fa[g])] = True
            mga.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))
            mgb.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))

    for move in range(1, 240):
        act = np.where(~fin)[0]
        if len(act) == 0: break
        cp = move % 2
        for mgr, mdl, kv in [(mga, ma, kva), (mgb, mb, kvb)]:
            st = torch.from_numpy(act).to(DEVICE)
            dp = torch.zeros(len(act), 1, dtype=torch.long, device=DEVICE)
            dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=DEVICE)
            dl = torch.ones(len(act), dtype=torch.long, device=DEVICE)
            lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
            lp = lp.masked_fill(og[act], -1e9)
            mgr.expand_roots(act.astype(np.int32),
                             torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                             lv.cpu().numpy().astype(np.float32))
            for _ in range(S_):
                sel = mgr.select_all()
                if sel['max_path_len'] == 0: continue
                vi = np.where(sel['valid_mask'])[0]
                if len(vi) == 0: continue
                pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(DEVICE)
                pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(DEVICE)
                lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(DEVICE)
                sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(DEVICE)
                lp2, lv2 = mdl.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(ot, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32),
                                      torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                      lv2.cpu().numpy().astype(np.float32))
        rpa = mga.get_root_policies(); rpb = mgb.get_root_policies()
        na = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            ua = ((cp == 0) == ab[g])
            pol = (rpa[g] if ua else rpb[g]).copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0:
                a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a
            if cp == 0: p0[g, a] = True
            else: p1[g, a] = True; og[g, a] = True
        dp_ = torch.from_numpy(na).to(DEVICE)
        dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=DEVICE)
        ma.decode(dp_, dpl_, kva, torch.from_numpy(act).to(DEVICE))
        mb.decode(dp_, dpl_, kvb, torch.from_numpy(act).to(DEVICE))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r: fin[g] = True; res[g] = r
            else:
                mga.apply_move(g, int(na[i]), p0[g], p1[g])
                mgb.apply_move(g, int(na[i]), p0[g], p1[g])

    del kva, kvb; torch.cuda.empty_cache()
    wa = wb = dr = 0
    for g in range(G_):
        w = res[g]
        if w == 1:
            if ab[g]: wa += 1
            else: wb += 1
        elif w == 2:
            if ab[g]: wb += 1
            else: wa += 1
        else: dr += 1
    return wa, wb, dr


# ── Policy-only match ──────────────────────────────────────────
@torch.inference_mode()
def play_match_policy(ma, mb):
    G_ = G_ELO
    pool = gomoku_cpp.GamePool(G_); pool.reset_all()

    kva = ma.create_cache(max_games=G_); kvb = mb.create_cache(max_games=G_)
    a_black = torch.tensor([i % 2 == 0 for i in range(G_)], device=DEVICE)
    finished = torch.zeros(G_, dtype=torch.bool)
    winners = torch.zeros(G_, dtype=torch.long)
    occupied = torch.zeros(G_, N_CELLS, dtype=torch.bool, device=DEVICE)
    idx = torch.arange(G_, device=DEVICE)

    fa_a = ma.sample_first_moves(G_, DEVICE)
    fa_b = mb.sample_first_moves(G_, DEVICE)
    first = torch.where(a_black, fa_a, fa_b)
    occupied[idx, first] = True

    fat = first.unsqueeze(1)
    pl0 = torch.zeros(G_, 1, dtype=torch.long, device=DEVICE)
    ma.prefill(fat, pl0, kva, list(range(G_)))
    mb.prefill(fat, pl0, kvb, list(range(G_)))

    for g in range(G_):
        r = gomoku_cpp.step(pool, g, int(first[g].item()))
        if r: finished[g] = True; winners[g] = r

    last_act = first
    last_plr = torch.zeros(G_, dtype=torch.long, device=DEVICE)

    for move in range(1, 240):
        act = torch.where(~finished)[0]
        if len(act) == 0: break
        cp = move % 2

        logits_a, _ = ma.decode(last_act, last_plr, kva, idx)
        logits_b, _ = mb.decode(last_act, last_plr, kvb, idx)

        logits = torch.where(a_black.unsqueeze(1) if cp == 0 else (~a_black).unsqueeze(1),
                            logits_a, logits_b)
        logits = logits.masked_fill(occupied, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1)

        occupied[idx, action] = True
        last_act = action
        last_plr = torch.full((G_,), cp, dtype=torch.long, device=DEVICE)

        for g_ in range(G_):
            if finished[g_]: continue
            r = gomoku_cpp.step(pool, g_, int(action[g_].item()))
            if r: finished[g_] = True; winners[g_] = r

    winners[(winners == 0) & ~finished] = 3
    del kva, kvb; torch.cuda.empty_cache()

    wa = wb = dr = 0
    for g in range(G_):
        w = winners[g].item()
        if w == 1:
            if a_black[g]: wa += 1
            else: wb += 1
        elif w == 2:
            if a_black[g]: wb += 1
            else: wa += 1
        else: dr += 1
    return wa, wb, dr


# ── ELO solver ─────────────────────────────────────────────────
def compute_elo(match_results):
    names = sorted(set(a for a, _, _, _ in match_results) | set(b for _, b, _, _ in match_results))
    if not names: return {}
    elo = {n: 1500.0 for n in names}
    for _ in range(500):
        dmax = 0.0
        for a, b, sa, sb in match_results:
            n = sa + sb
            if n == 0: continue
            ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            da = (sa - ea * n) * (32.0 / n)
            elo[a] += da; elo[b] -= da
            dmax = max(dmax, abs(da))
        if dmax < 1e-6: break
    return elo


# ── Plot ───────────────────────────────────────────────────────
def plot_elo(elo_dict, ckpt_dir, suffix, title_suffix):
    names = sorted(elo_dict.keys(), key=lambda n: (-1 if n == 'noisy_uniform' else int(n.split('_')[1].split('.')[0])))
    steps = [(int(n.split('_')[1].split('.')[0]), elo_dict[n])
             for n in names if n.startswith('step_')]
    steps.sort(key=lambda x: x[0])
    if not steps: return

    noisy_elo = elo_dict.get('noisy_uniform', 1500)
    step_nums = [s[0] for s in steps]
    step_elos = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(step_nums, step_elos, '.-', markersize=4, linewidth=1.5)
    ax.axhline(y=noisy_elo, color='gray', linestyle=':',
               label=f'Noisy uniform (ELO={noisy_elo:.0f})')
    ax.set_xlabel('Step'); ax.set_ylabel('ELO')
    ax.set_title(f'ELO {title_suffix} (G=256, S=16)')
    ax.legend(); ax.grid(True, alpha=0.3)
    out_path = f'{ckpt_dir}/elo_{suffix}_curve.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Plot: {out_path}')


# ── Main loop ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--G', type=int, default=256)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--S', type=int, default=16)
    parser.add_argument('--max_gap', type=int, default=5,
                        help='Only evaluate pairs with |i-j| <= max_gap (plus noisy_uniform)')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()

    global G_ELO, M_ELO, S_ELO, DEVICE, CACHE_FILE
    G_ELO, M_ELO, S_ELO = args.G, args.M, args.S
    DEVICE = torch.device("cuda")
    CACHE_FILE = os.path.join(args.ckpt_dir, "elo_monitor_cache.json")

    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
    uni = NoisyUniform(cfg=cfg)
    uni_name = 'noisy_uniform'

    # Load or init cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    else:
        cache = {"mcts": {}, "policy": {}}

    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Continuous ELO monitor: {args.ckpt_dir}")
    print(f"  G={G_ELO} M={M_ELO} S={S_ELO} interval={args.interval}s")
    print(f"  Cache: {CACHE_FILE}")

    def load_model(ckpt_name):
        m = GomokuTransformer(cfg).to(DEVICE).eval()
        m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, ckpt_name), map_location=DEVICE))
        return m

    def get_pair_key(a, b):
        """Key that preserves original call order: a was ma, b was mb."""
        return f"{a}|{b}"

    def eval_pair(ma, mb, a_name, b_name, cache_key, match_type):
        """Evaluate one pair, update cache if not already done."""
        pair_key = get_pair_key(a_name, b_name)
        # Also check reverse key (same pair, opposite order)
        rev_key = get_pair_key(b_name, a_name)
        if cache[cache_key].get(pair_key) is not None:
            val = cache[cache_key][pair_key]
            if len(val) >= 4 and val[3] == G_ELO:
                return val[0], val[1], val[2]
        if cache[cache_key].get(rev_key) is not None:
            val = cache[cache_key][rev_key]
            if len(val) >= 4 and val[3] == G_ELO:
                # Reverse: wa was for b_name, wb was for a_name
                return val[1], val[0], val[2]

        if match_type == 'mcts':
            wa, wb, d = play_match_mcts(ma, mb)
        else:
            wa, wb, d = play_match_policy(ma, mb)

        cache[cache_key][pair_key] = [wa, wb, d, G_ELO]
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        return wa, wb, d

    def build_results(match_type):
        """Build results list from cache for ELO computation."""
        cache_key = 'mcts' if match_type == 'mcts' else 'policy'
        results = []
        for key, val in cache[cache_key].items():
            if len(val) < 4 or val[3] != G_ELO:
                continue
            wa, wb, d = val[0], val[1], val[2]
            try:
                a, b = key.split('|')
                results.append((a, b, wa + d * 0.5, wb + d * 0.5))
            except (ValueError, IndexError):
                pass
        return results

    def plot_both():
        """Recompute ELO from all results and plot both curves."""
        for match_type, suffix, title in [
            ('mcts', 'mcts', 'MCTS'),
            ('policy', 'policy', 'Policy-only')
        ]:
            results = build_results(match_type)
            if results:
                elo = compute_elo(results)
                plot_elo(elo, args.ckpt_dir, suffix, title)
        print()  # blank line

    # Track which checkpoints have been fully evaluated
    evaluated = set()

    while True:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
        ckpt_names = [os.path.basename(c) for c in ckpts]

        if not ckpt_names:
            print(f"[{time.strftime('%H:%M:%S')}] No checkpoints yet, sleeping {args.interval}s...")
            time.sleep(args.interval)
            continue

        # Find checkpoints not yet evaluated
        pending = [n for n in ckpt_names if n not in evaluated]
        if pending:
            # Sort by step number, process smallest first
            pending.sort(key=lambda n: int(n.split('_')[1].split('.')[0]))
            ckpt_name = pending[0]  # smallest un-evaluated
            step_i = int(ckpt_name.split('_')[1].split('.')[0])
            print(f"\n[{time.strftime('%H:%M:%S')}] New checkpoint: {ckpt_name}")

            # Load model once
            model_i_mcts = load_model(ckpt_name)
            model_i_policy = load_model(ckpt_name)

            # Sparse opponents: previous steps within max_gap (higher → lower)
            opponents = []
            for j in range(step_i - 1, max(step_i - 1 - args.max_gap, -1), -1):
                if j >= 0:
                    opponents.insert(0, f"step_{j:06d}.pt")
            # noisy_uniform only vs steps 0..4 (treated as step -1)
            if step_i <= 4:
                opponents.append(uni_name)
            existing_opponents = [o for o in opponents if o == uni_name or o in ckpt_names]

            for opp_name in existing_opponents:
                print(f'  MCTS:  {ckpt_name} vs {opp_name} ...', end=' ', flush=True)
                t0 = time.perf_counter()

                if opp_name == uni_name:
                    opp_model = uni
                else:
                    opp_model = load_model(opp_name)

                wa, wb, d = eval_pair(model_i_mcts, opp_model, ckpt_name, opp_name, 'mcts', 'mcts')
                dt = time.perf_counter() - t0
                wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
                print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)')

                if opp_name != uni_name:
                    del opp_model
                torch.cuda.empty_cache()

            # Policy-only matches
            for opp_name in existing_opponents:
                print(f'  Policy:{ckpt_name} vs {opp_name} ...', end=' ', flush=True)
                t0 = time.perf_counter()

                if opp_name == uni_name:
                    opp_model = uni
                else:
                    opp_model = load_model(opp_name)

                wa, wb, d = eval_pair(model_i_policy, opp_model, ckpt_name, opp_name, 'policy', 'policy')
                dt = time.perf_counter() - t0
                wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
                print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)')

                if opp_name != uni_name:
                    del opp_model
                torch.cuda.empty_cache()

            del model_i_mcts, model_i_policy
            torch.cuda.empty_cache()

            evaluated.add(ckpt_name)
            plot_both()
        else:
            # All evaluated, check for new checkpoints
            new_ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
            new_names = [os.path.basename(c) for c in new_ckpts]
            new_pending = [n for n in new_names if n not in evaluated]
            if not new_pending:
                print(f"[{time.strftime('%H:%M:%S')}] All {len(evaluated)} checkpoints evaluated, "
                      f"waiting for new ones...")
                time.sleep(args.interval)


if __name__ == '__main__':
    main()
