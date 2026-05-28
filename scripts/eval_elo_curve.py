#!/usr/bin/env python3
"""ELO evaluation for replay buffer training and ELO curve plot.

Usage: python scripts/eval_elo_curve.py --ckpt_dir checkpoints/replay20
"""
import argparse, os, sys, time, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
import gomoku_cpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/replay20')
    parser.add_argument('--G', type=int, default=256)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--S', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda')
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)

    class NoisyUniform:
        def __init__(self, ns=0.02): self.ns = ns; self.config = cfg
        def create_cache(self, max_games, max_cache_len=250):
            m = GomokuTransformer(cfg).to(device)
            return m.create_cache(max_games, max_cache_len)
        def sample_first_moves(self, bs, dev):
            return torch.randint(0, 225, (bs,), device=dev)
        def prefill(self, pos, plr, cache, indices):
            return (torch.randn(pos.shape[0], 225, device=device) * self.ns,
                    torch.zeros(pos.shape[0], device=device))
        def decode(self, pos, plr, cache, indices):
            cache.advance(indices)
            return (torch.randn(len(indices), 225, device=device) * self.ns,
                    torch.zeros(len(indices), device=device))
        def evaluate_mcts_leaves(self, pos, plr, cache, indices, plen):
            return (torch.randn(pos.shape[0], 225, device=device) * self.ns,
                    torch.zeros(pos.shape[0], device=device))

    @torch.inference_mode()
    def play_match(ma, mb):
        G_ = args.G; M_ = args.M; S_ = args.S
        pool = gomoku_cpp.GamePool(G_); pool.reset_all()
        def mm():
            m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
            m.c_puct = 1.0; m.leaves_per_game = M_; m.dirichlet_eps = 0.0
            m.init_roots(np.zeros((G_, 225), dtype=bool),
                         np.zeros((G_, 225), dtype=bool),
                         np.zeros(G_, dtype=np.int32))
            return m
        mga = mm(); mgb = mm()
        ab = np.array([i % 2 == 0 for i in range(G_)], dtype=bool)
        fin = np.zeros(G_, dtype=bool); res = np.zeros(G_, dtype=np.int32)
        p0 = np.zeros((G_, 225), dtype=bool); p1 = np.zeros((G_, 225), dtype=bool)
        fa_a = ma.sample_first_moves(G_, device); fa_b = mb.sample_first_moves(G_, device)
        fa = np.zeros(G_, dtype=np.int64)
        for g in range(G_):
            fa[g] = int(fa_a[g].item()) if ab[g] else int(fa_b[g].item())
            p0[g, fa[g]] = True
        fat = torch.tensor(fa, dtype=torch.long, device=device).unsqueeze(1)
        pl0 = torch.zeros(G_, 1, dtype=torch.long, device=device)
        kva, brkva = ma.create_cache(max_games=G_)
        kvb, brkvb = mb.create_cache(max_games=G_)
        pa0, va0 = ma.prefill(fat, pl0, kva, brkva, list(range(G_)))
        pb0, vb0 = mb.prefill(fat, pl0, kvb, brkvb, list(range(G_)))
        og = torch.from_numpy(p0 | p1).to(device)
        root_pol_a = pa0.float().clone()
        root_val_a = va0.float().clone()
        root_pol_b = pb0.float().clone()
        root_val_b = vb0.float().clone()
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
            act_np = act.astype(np.int32)
            act_t = torch.from_numpy(act).to(device)
            for mgr, pol_buf, val_buf, kv in [(mga, root_pol_a, root_val_a, kva),
                                              (mgb, root_pol_b, root_val_b, kvb)]:
                lp = pol_buf[act].masked_fill(og[act], -1e9)
                lv = val_buf[act]
                mgr.expand_roots(act_np,
                                 torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                                 lv.cpu().numpy().astype(np.float32))
                for _ in range(S_):
                    sel = mgr.select_all()
                    if sel['max_path_len'] == 0: continue
                    vi = np.where(sel['valid_mask'])[0]
                    if len(vi) == 0: continue
                    pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(device)
                    pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(device)
                    lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(device)
                    sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(device)
                    mdl = ma if mgr is mga else mb
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
                pol = (rpa[g] if ua else rpb[g]).copy(); pol[p0[g] | p1[g]] = 0
                if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
                else:
                    leg = np.where(~(p0[g] | p1[g]))[0]
                    a = int(np.random.choice(leg)) if len(leg) > 0 else 0
                na[i] = a
                if cp == 0: p0[g, a] = True
                else: p1[g, a] = True
                og[g, a] = True
            dp_ = torch.from_numpy(na).to(device)
            dpl_ = torch.full((len(act),), cp, dtype=torch.long, device=device)
            dec_slots = torch.from_numpy(act).to(device)
            new_pa, new_va = ma.decode(dp_, dpl_, kva, brkva, dec_slots)
            new_pb, new_vb = mb.decode(dp_, dpl_, kvb, brkvb, dec_slots)
            root_pol_a[act_t] = new_pa.float()
            root_val_a[act_t] = new_va.float()
            root_pol_b[act_t] = new_pb.float()
            root_val_b[act_t] = new_vb.float()
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

    # Collect models
    pretrain_path = f'{args.ckpt_dir}/pretrain.pt'
    has_pretrain = os.path.exists(pretrain_path)
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pt')))
    if not ckpts:
        print(f"ERROR: no step checkpoints found in {args.ckpt_dir}")
        sys.exit(1)

    names = ['noisy_uniform']
    if has_pretrain:
        names.append('pretrain')
    names += [os.path.basename(c) for c in ckpts]
    print(f"Models: {len(names)}")
    elo_results = []

    uni = NoisyUniform()
    model_cache = {'noisy_uniform': uni}
    if has_pretrain:
        m = GomokuTransformer(cfg).to(device).eval()
        m.load_state_dict(torch.load(pretrain_path, map_location=device))
        model_cache['pretrain'] = m

    print(f"Pairwise matches: {len(names) * (len(names) - 1) // 2}")
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            print(f'  {na} vs {nb} ...', end=' ', flush=True)
            t0 = time.perf_counter()

            if na not in model_cache:
                m = GomokuTransformer(cfg).to(device).eval()
                m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, na),
                                             map_location=device))
                model_cache[na] = m
            if nb not in model_cache:
                m = GomokuTransformer(cfg).to(device).eval()
                m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, nb),
                                             map_location=device))
                model_cache[nb] = m

            wa, wb, d = play_match(model_cache[na], model_cache[nb])
            dt = time.perf_counter() - t0
            wr = wa / (wa + wb) * 100 if (wa + wb) > 0 else 50.0
            elo_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
            print(f'{wa}-{wb} WR={wr:.1f}% ({dt:.0f}s)', flush=True)

            # Free non-baseline models to save VRAM
            if na not in ('noisy_uniform', 'pretrain'):
                del model_cache[na]
            if nb not in ('noisy_uniform', 'pretrain'):
                del model_cache[nb]
            torch.cuda.empty_cache()

    elo = compute_elo(elo_results)

    # Print results
    print(f"\nELO Results (S={args.S}, replay buffer, {len(ckpts)} steps):")
    for n in names:
        r = elo.get(n, 1500)
        tag = ' (baseline)' if n == 'noisy_uniform' else ''
        if n == 'pretrain': tag = ' [pretrain]'
        print(f'  {n:20s}: ELO={r:.0f}{tag}')

    steps = [(int(n.split('_')[1].split('.')[0]), elo.get(n, 1500))
             for n in names if n.startswith('step_')]
    steps.sort(key=lambda x: x[0])
    if len(steps) >= 2:
        print(f'  ΔELO (step_0→step_{steps[-1][0]}) = {steps[-1][1] - steps[0][1]:+.0f}')

    # ── Plot ELO curve ──
    noisy_elo = elo.get('noisy_uniform', 1500)

    step_nums = [s[0] for s in steps]
    step_elos = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(step_nums, step_elos, 'b-o', markersize=4, label='Self-play steps')
    if has_pretrain:
        pretrain_elo = elo.get('pretrain', 1500)
        ax.axhline(y=pretrain_elo, color='green', linestyle='--',
                   label=f'Pretrain (ELO={pretrain_elo:.0f})')
    ax.axhline(y=noisy_elo, color='gray', linestyle=':',
               label=f'Noisy uniform (ELO={noisy_elo:.0f})')

    ax.set_xlabel('Step')
    ax.set_ylabel('ELO')
    ax.set_title(f'Replay Buffer Training (G=512, S={args.S})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = f'{args.ckpt_dir}/elo_curve.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f'\nELO curve saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
