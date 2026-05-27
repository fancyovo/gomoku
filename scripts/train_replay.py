#!/usr/bin/env python3
"""Replay-buffer AlphaZero training for Gomoku.

Pool size = 8 * G (before symmetry augmentation).
Each step: self-play G games, discard G random old games, add new ones,
train 1 epoch on the full pool, save checkpoint.
After all steps, run ELO tournament.

Usage: python scripts/train_replay.py [--ckpt_dir DIR] [--data_dir DIR]
"""

import argparse, os, sys, time, glob, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
from training.replay import (
    load_old_data, augment_trajectories, GameDataset, collate_fn,
    evaluate, train_one_epoch, run_selfplay,
)
from training.loss import alphago_zero_loss
import gomoku_cpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/replay')
    parser.add_argument('--data_dir', default='data/selfplay')
    parser.add_argument('--G', type=int, default=512)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--S', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--pool_mult', type=int, default=8,
                        help='pool size = pool_mult * G')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--skip_elo', action='store_true',
                        help='Skip ELO tournament after training')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Start from random model, no pretraining')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--n_shared', type=int, default=4)
    parser.add_argument('--n_policy', type=int, default=4)
    parser.add_argument('--n_value', type=int, default=4)
    parser.add_argument('--single_loss', action='store_true',
                        help='Use combined policy+value loss (one backward per batch)')
    parser.add_argument('--raw_loss', action='store_true',
                        help='Use raw CE (no ln(N) normalization) so policy and value contribute equally')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='Fraction of training data to use per step (subsampled randomly)')
    args = parser.parse_args()

    torch.set_num_threads(2)  # limit PyTorch CPU parallelism, GPU is the bottleneck
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device('cuda')
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15,
                      n_shared=args.n_shared, n_policy=args.n_policy, n_value=args.n_value)
    pool_size = args.pool_mult * args.G

    print(f"Replay Buffer Training")
    print(f"  Pool: {pool_size} games (before aug), G={args.G}, M={args.M}, S={args.S}")
    print(f"  Steps: {args.n_steps}, lr={args.lr}")
    print(f"  Checkpoints: {args.ckpt_dir}")

    # ── Load initial pool data ──
    all_data = load_old_data(args.data_dir)
    print(f"  Initial data: {len(all_data)} games")

    # ── Initialize model (4 shared + 4 policy + 4 value layers) ──
    model = GomokuTransformer(cfg).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    if args.from_scratch:
        print("\nStarting from scratch (random init, no pretrain).")
        pool_rng = random.Random(123)
        pool = pool_rng.sample(all_data, min(pool_size, len(all_data)))
        print(f"  Using {len(pool)} games as initial pool (sampled from {len(all_data)} available).")
    else:
        # ── Pretrain on ENTIRE dataset (1 epoch) ──
        print("\nPretraining on full dataset (1 epoch)...")
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        rng = random.Random(42)
        rng.shuffle(all_data)
        n_tr = int(len(all_data) * 0.8)
        tr_pool = all_data[:n_tr]
        te_pool = all_data[n_tr:]
        tr_aug = augment_trajectories(tr_pool)
        te_aug = augment_trajectories(te_pool)
        tr_ds = GameDataset(tr_aug)
        te_ds = GameDataset(te_aug)

        tp, tv, _, _ = train_one_epoch(model, pretrain_opt, tr_ds, te_ds, device, args.train_batch, args.num_workers, normalize_loss=not args.raw_loss)
        print(f"  Pretrain done: test_p={tp:.4f} test_v={tv:.4f}")
        model.eval()
        torch.save(model.state_dict(), f'{args.ckpt_dir}/pretrain.pt')

        pool_rng = random.Random(123)
        pool = pool_rng.sample(all_data, min(pool_size, len(all_data)))
        print(f"  Initial pool: {len(pool)} games (not trained on)")

    # ── Self-play steps ──
    import csv as _csv
    t_start = time.perf_counter()
    game_lens = []; test_ps = []; test_vs = []; train_ps = []; train_vs = []

    def _plot_losses(step_idx):
        """Plot game length + policy/value loss curves."""
        n = len(game_lens)
        xs = list(range(n))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(xs, game_lens, '.-', markersize=4)
        axes[0].set_title('Avg Game Length'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(xs, test_ps, '.-', label='test_p', markersize=4)
        axes[1].plot(xs, train_ps, '.-', label='train_p', markersize=4)
        axes[1].set_title('Policy Loss (norm CE)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(xs, test_vs, '.-', label='test_v', markersize=4)
        axes[2].plot(xs, train_vs, '.-', label='train_v', markersize=4)
        axes[2].set_title('Value Loss (norm CE)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{args.ckpt_dir}/training_curves.png', dpi=100, bbox_inches='tight')
        plt.close()

        # CSV export
        with open(f'{args.ckpt_dir}/training_log.csv', 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['step', 'avg_len', 'test_p', 'test_v', 'train_p', 'train_v'])
            for i in range(n):
                w.writerow([i, game_lens[i], test_ps[i], test_vs[i], train_ps[i], train_vs[i]])

    for step in range(args.n_steps):
        print(f"\n{'=' * 60}")
        print(f"Step {step}/{args.n_steps - 1}")
        print(f"{'=' * 60}")

        # Self-play
        t0 = time.perf_counter()
        model.eval()
        new_traj, avg_len, bw, ww, dr = run_selfplay(model, device, args.G, args.M, args.S)
        tsp = time.perf_counter() - t0
        print(f"  Self-play: G={len(new_traj)} len={avg_len:.0f} "
              f"B={bw} W={ww} D={dr} ({tsp:.0f}s)")

        # Update pool
        pool_rng.shuffle(pool)
        pool = pool[args.G:] + new_traj
        print(f"  Pool: {len(pool)} games")

        # Train/test split
        pool_rng.shuffle(pool)
        n_tr = int(len(pool) * 0.8)
        tr_pool = pool[:n_tr]
        te_pool = pool[n_tr:]

        t0 = time.perf_counter()
        tr_aug = augment_trajectories(tr_pool)
        te_aug = augment_trajectories(te_pool)
        if args.train_fraction < 1.0:
            n_tr_aug = int(len(tr_aug) * args.train_fraction)
            random.shuffle(tr_aug)
            tr_aug = tr_aug[:n_tr_aug]
        tr_ds = GameDataset(tr_aug)
        te_ds = GameDataset(te_aug)
        taug = time.perf_counter() - t0

        # Train 1 epoch
        t0 = time.perf_counter()
        step_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        tp, tv, tr_p_loss, tr_v_loss = train_one_epoch(
            model, step_opt, tr_ds, te_ds, device, args.train_batch, args.num_workers,
            single_loss=args.single_loss, normalize_loss=not args.raw_loss)
        ttr = time.perf_counter() - t0

        model.eval()
        torch.save(model.state_dict(), f'{args.ckpt_dir}/step_{step:06d}.pt')
        print(f"  Train: test_p={tp:.4f} test_v={tv:.4f} "
              f"train_p={tr_p_loss:.4f} train_v={tr_v_loss:.4f} "
              f"aug={taug:.0f}s tr={ttr:.0f}s")

        # Track
        game_lens.append(avg_len); test_ps.append(tp); test_vs.append(tv)
        train_ps.append(tr_p_loss); train_vs.append(tr_v_loss)
        _plot_losses(step)

    dt_total = time.perf_counter() - t_start
    print(f"\nTraining done: {dt_total / 60:.0f}min")

    if args.skip_elo:
        print("Skipping ELO tournament (--skip_elo).")
        return

    # ── ELO Tournament ──
    print(f"\n{'=' * 60}")
    print("ELO Tournament (S=16, noisy uniform baseline)")
    print("=" * 60)

    G_ELO, M_ELO, S_ELO = 256, 4, 16

    class NoisyUniform:
        def __init__(self, ns=0.02):
            self.ns = ns
            self.config = cfg

        def create_cache(self, max_games, max_cache_len=250):
            m = GomokuTransformer(cfg).to(device)
            return m.create_cache(max_games, max_cache_len)

        def sample_first_moves(self, bs, dev):
            return torch.randint(0, 225, (bs,), device=dev)

        def prefill(self, pos, plr, kv_cache, branch_cache, indices):
            return (torch.randn(pos.shape[0], 225, device=device) * self.ns,
                    torch.zeros(pos.shape[0], device=device))

        def decode(self, pos, plr, kv_cache, branch_cache, indices):
            kv_cache.advance(indices)
            return (torch.randn(len(indices), 225, device=device) * self.ns,
                    torch.zeros(len(indices), device=device))

        def evaluate_mcts_leaves(self, pos, plr, cache, indices, path_lengths):
            return (torch.randn(pos.shape[0], 225, device=device) * self.ns,
                    torch.zeros(pos.shape[0], device=device))

    @torch.inference_mode()
    def play_match(model_a, model_b):
        G_ = G_ELO
        pool = gomoku_cpp.GamePool(G_)
        pool.reset_all()

        def mm():
            m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2 ** 31))
            m.c_puct = 1.0; m.leaves_per_game = M_ELO; m.dirichlet_eps = 0.0
            m.init_roots(np.zeros((G_, 225), dtype=bool),
                         np.zeros((G_, 225), dtype=bool),
                         np.zeros(G_, dtype=np.int32))
            return m

        mga = mm(); mgb = mm()
        kva, _br_kva = model_a.create_cache(max_games=G_)
        kvb, _br_kvb = model_b.create_cache(max_games=G_)
        ab = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
        fin = np.zeros(G_, dtype=bool)
        res = np.zeros(G_, dtype=np.int32)
        p0 = np.zeros((G_, 225), dtype=bool)
        p1 = np.zeros((G_, 225), dtype=bool)

        fa_a = model_a.sample_first_moves(G_, device)
        fa_b = model_b.sample_first_moves(G_, device)
        fa = np.zeros(G_, dtype=np.int64)
        for g in range(G_):
            fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
            p0[g, fa[g]] = True

        fat = torch.tensor(fa, dtype=torch.long, device=device).unsqueeze(1)
        pl0 = torch.zeros(G_, 1, dtype=torch.long, device=device)
        model_a.prefill(fat, pl0, kva, _br_kva, list(range(G_)))
        model_b.prefill(fat, pl0, kvb, _br_kvb, list(range(G_)))
        og = torch.from_numpy(p0 | p1).to(device)

        for g in range(G_):
            r = gomoku_cpp.step(pool, g, int(fa[g]))
            if r:
                fin[g] = True; res[g] = r
            else:
                bp = np.zeros(225, dtype=bool); bp[int(fa[g])] = True
                mga.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))
                mgb.apply_move(g, int(fa[g]), bp, np.zeros(225, dtype=bool))

        for move in range(1, 240):
            act = np.where(~fin)[0]
            if len(act) == 0: break
            cp = move % 2
            for mgr, mdl, kv in [(mga, model_a, kva), (mgb, model_b, kvb)]:
                st = torch.from_numpy(act).to(device)
                dp = torch.zeros(len(act), 1, dtype=torch.long, device=device)
                dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=device)
                dl = torch.ones(len(act), dtype=torch.long, device=device)
                lp, lv = mdl.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
                lp = lp.masked_fill(og[act], -1e9)
                mgr.expand_roots(act.astype(np.int32),
                                 torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                                 lv.cpu().numpy().astype(np.float32))
                for _ in range(S_ELO):
                    sel = mgr.select_all()
                    if sel['max_path_len'] == 0: continue
                    vi = np.where(sel['valid_mask'])[0]
                    if len(vi) == 0: continue
                    pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(device)
                    pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(device)
                    lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(device)
                    sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(device)
                    lp2, lv2 = mdl.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
                    ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(device).bool()
                    lp2 = lp2.masked_fill(ot, -1e9)
                    mgr.expand_and_backup(vi.astype(np.int32),
                                          torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                          lv2.cpu().numpy().astype(np.float32))
            rpa = mga.get_root_policies(); rpb = mgb.get_root_policies()
            na = np.zeros(len(act), dtype=np.int64)
            for i, g in enumerate(act):
                ua = ((cp == 0) == ab[g])
                pol = (rpa[g] if ua else rpb[g]).copy()
                pol[p0[g] | p1[g]] = 0
                if pol.sum() > 0:
                    a = int(np.random.choice(225, p=pol / pol.sum()))
                else:
                    leg = np.where(~(p0[g] | p1[g]))[0]
                    a = int(np.random.choice(leg)) if len(leg) > 0 else 0
                na[i] = a
                if cp == 0:
                    p0[g, a] = True
                else:
                    p1[g, a] = True; og[g, a] = True
            dp_ = torch.from_numpy(na).to(device)
            dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=device)
            model_a.decode(dp_, dpl_, kva, _br_kva, torch.from_numpy(act).to(device))
            model_b.decode(dp_, dpl_, kvb, _br_kvb, torch.from_numpy(act).to(device))
            for i, g in enumerate(act):
                r = gomoku_cpp.step(pool, g, int(na[i]))
                if r:
                    fin[g] = True; res[g] = r
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
            else:
                dr += 1
        return wa, wb, dr

    def compute_elo(er):
        names = sorted(set(a for a, _, _, _ in er) | set(b for _, b, _, _ in er))
        if not names: return {}
        elo = {n: 1500.0 for n in names}
        for _ in range(500):
            dmax = 0.0
            for a, b, sa, sb in er:
                n = sa + sb
                if n == 0: continue
                ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
                da = (sa - ea * n) * (32.0 / n)
                elo[a] += da; elo[b] -= da
                dmax = max(dmax, abs(da))
            if dmax < 1e-6: break
        return elo

    uni = NoisyUniform()
    pretrain_path = f'{args.ckpt_dir}/pretrain.pt'
    has_pretrain = os.path.exists(pretrain_path)
    if has_pretrain:
        pretrain_m = GomokuTransformer(cfg).to(device).eval()
        pretrain_m.load_state_dict(torch.load(pretrain_path, map_location=device))

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
    names = ['noisy_uniform']
    if has_pretrain:
        names.append('pretrain')
    names += [os.path.basename(c) for c in ckpts]
    print(f"Models in ELO: {len(names)}")
    elo_results = []

    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            print(f'  {na} vs {nb} ...', end=' ', flush=True)
            t0 = time.perf_counter()
            if na == 'noisy_uniform': ma = uni
            elif na == 'pretrain': ma = pretrain_m
            else: ma = GomokuTransformer(cfg).to(device).eval(); ma.load_state_dict(torch.load(os.path.join(args.ckpt_dir, na), map_location=device))
            if nb == 'noisy_uniform': mb = uni
            elif nb == 'pretrain': mb = pretrain_m
            else: mb = GomokuTransformer(cfg).to(device).eval(); mb.load_state_dict(torch.load(os.path.join(args.ckpt_dir, nb), map_location=device))
            wa, wb, d = play_match(ma, mb)
            dt = time.perf_counter() - t0
            wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
            elo_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
            print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)', flush=True)
            baseline = ('noisy_uniform', 'pretrain')
            if na not in baseline: del ma
            if nb not in baseline: del mb
            torch.cuda.empty_cache()

    elo = compute_elo(elo_results)
    print(f'\nELO Results (S={S_ELO}, replay buffer, {args.n_steps} steps):')
    for n in names:
        r = elo.get(n, 1500)
        tag = ' (baseline)' if n == 'noisy_uniform' else ''
        if n == 'pretrain': tag = ' [pretrain]'
        print(f'  {n:20s}: ELO={r:.0f}{tag}')
    steps = [(n, elo.get(n, 1500)) for n in names if n.startswith('step_')]
    steps.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))
    if len(steps) >= 2:
        print(f'  ΔELO (step_0→{steps[-1][0].split(".")[0]}) = {steps[-1][1] - steps[0][1]:+.0f}')


if __name__ == '__main__':
    main()
