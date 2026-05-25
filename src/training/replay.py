"""Replay buffer training for Gomoku AlphaZero."""
import pickle, glob, random, os, time, math
import numpy as np
import torch
from model import ModelConfig, GomokuTransformer
from training.augment import SYM_TABLE, N_SYMS, INV_SYM_TABLE
from training.loss import alphago_zero_loss
import gomoku_cpp


def load_old_data(data_dir: str) -> list:
    """Load all old self-play data into memory."""
    files = sorted(glob.glob(os.path.join(data_dir, 'selfplay_*.pkl')))
    all_data = []
    for fp in files:
        with open(fp, 'rb') as f:
            all_data.extend(pickle.load(f))
    return all_data


def augment_trajectories(trajs: list) -> list:
    """Apply 8x symmetry augmentation to a list of trajectories.

    Policy alignment: mcts_policies[k] = MCTS policy for board after k moves,
    predicting move k. Old data has L entries (π_0..π_{L-1}), self-play has
    L-1 entries (π_1..π_{L-1}). collate_fn uses [:L-1] which for old data
    takes π_0..π_{L-2} and for self-play takes π_1..π_{L-1}. The first
    prediction target π_0 is perfectly uniform (empty board + DummyModel),
    which is beneficial at S=16."""
    out = []
    for t in trajs:
        L = t['actual_len']
        pos = torch.from_numpy(t['positions'][:L])
        plr = torch.from_numpy(t['players'][:L])
        pol_raw = torch.from_numpy(t['mcts_policies'][:L])
        val = torch.from_numpy(t['value_targets'][:L])
        for sym in range(N_SYMS):
            rm = SYM_TABLE[sym]
            irm = INV_SYM_TABLE[sym]
            out.append({
                'positions': rm[pos],
                'players': plr,
                'mcts_policies': pol_raw[:, irm],
                'value_targets': val,
                'actual_len': L,
            })
    return out


class GameDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch):
    """Collate trajectories. mcts_policies has L or L-1 entries (see augment).
    We use [:L-1] which gives policy targets for positions 0..L-2.
    For old data this includes π_0 (uniform empty-board policy) at position 0."""
    ml = max(s['positions'].shape[0] for s in batch)
    B_ = len(batch)
    pos = torch.zeros(B_, ml, dtype=torch.long)
    plr = torch.zeros(B_, ml, dtype=torch.long)
    pol = torch.zeros(B_, ml, 225)
    vt = torch.zeros(B_, ml, 2)
    mask = torch.zeros(B_, ml, dtype=torch.bool)
    for i, s in enumerate(batch):
        L_ = s['actual_len']
        pos[i, :L_] = s['positions']
        plr[i, :L_] = s['players']
        pol[i, :L_ - 1] = s['mcts_policies'][:L_ - 1]
        vt[i, :L_] = s['value_targets']
        mask[i, :L_] = True
    return {
        'positions': pos, 'players': plr,
        'mcts_policies': pol, 'value_targets': vt, 'mask': mask,
    }


def evaluate(model, ds, device, batch_size=128):
    """Evaluate alphago_zero_loss on a dataset. Returns (policy_loss, value_loss)
    averaged over all VALID (non-padding) tokens."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    total_samples = 0
    tp_sum, tv_sum = 0.0, 0.0
    with torch.inference_mode():
        for batch in dl:
            pos = batch['positions'].to(device)
            plr = batch['players'].to(device)
            pt = batch['mcts_policies'].to(device)
            vt = batch['value_targets'].to(device)
            m = batch['mask'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p, v = model(pos, plr)
            if pos.shape[1] > 1:
                pp = p[:, :-1].float()
                vv = v[:, :-1].float()
                tp_ = pt[:, :-1]
                tv_ = vt[:, :-1]
                pm = m[:, :-1]
                n_valid = pm.sum().item()
                _, pl, vl = alphago_zero_loss(
                    pp.reshape(-1, 225), tp_.reshape(-1, 225),
                    vv.reshape(-1, 2), tv_.reshape(-1, 2), pm.reshape(-1))
                tp_sum += pl.item() * n_valid
                tv_sum += vl.item() * n_valid
                total_samples += n_valid
    return tp_sum / max(total_samples, 1), tv_sum / max(total_samples, 1)


def train_one_epoch(model, opt, train_ds, test_ds, device, batch_size=128):
    """Train 1 epoch. Alternating optimization: policy and value each get
    their own backward+step per batch, avoiding gradient scale imbalance."""
    tr_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                        shuffle=True, collate_fn=collate_fn)
    opt.zero_grad()
    model.train()
    for batch in tr_dl:
        pos = batch['positions'].to(device)
        plr = batch['players'].to(device)
        pt = batch['mcts_policies'].to(device)
        vt = batch['value_targets'].to(device)
        m = batch['mask'].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            p, v = model(pos, plr)
        if pos.shape[1] > 1:
            pp = p[:, :-1].contiguous()
            vv = v[:, :-1].contiguous()
            tp_ = pt[:, :-1].contiguous()
            tv_ = vt[:, :-1].contiguous()
            pm = m[:, :-1].contiguous()
            # Policy-only step
            loss_p, _, _ = alphago_zero_loss(
                pp.reshape(-1, 225).float(), tp_.reshape(-1, 225),
                vv.reshape(-1, 2).float(), tv_.reshape(-1, 2),
                pm.reshape(-1), policy_weight=1.0, value_weight=0.0)
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            # Value-only step
            loss_v, _, _ = alphago_zero_loss(
                pp.reshape(-1, 225).float(), tp_.reshape(-1, 225),
                vv.reshape(-1, 2).float(), tv_.reshape(-1, 2),
                pm.reshape(-1), policy_weight=0.0, value_weight=1.0)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    return evaluate(model, test_ds, device, batch_size)


def run_selfplay(model, device, G, M, S):
    """Run G games of MCTS self-play with M leaves and S sims/leaf.
    Returns list of trajectory dicts plus stats."""
    pool = gomoku_cpp.GamePool(G)
    pool.reset_all()
    mgr = gomoku_cpp.MCTSManager(G, seed_base=np.random.randint(0, 2 ** 31))
    mgr.c_puct = 1.0
    mgr.dirichlet_eps = 0.25
    mgr.dirichlet_alpha = 0.03
    mgr.leaves_per_game = M

    p0 = np.zeros((G, 225), dtype=bool)
    p1 = np.zeros((G, 225), dtype=bool)
    mgr.init_roots(p0, p1, np.zeros(G, dtype=np.int32))
    kv = model.create_cache(max_games=G, max_cache_len=250)

    fa = model.sample_first_moves(G, device)
    model.prefill(fa.unsqueeze(1),
                  torch.zeros(G, 1, dtype=torch.long, device=device),
                  kv, list(range(G)))

    ph = [[] for _ in range(G)]
    plh = [[] for _ in range(G)]
    mp = [[] for _ in range(G)]
    plen = np.zeros(G, dtype=np.int32)
    fin = np.zeros(G, dtype=bool)
    res = np.zeros(G, dtype=np.int32)

    for g in range(G):
        a = int(fa[g].item())
        ph[g].append(a); plh[g].append(0); p0[g, a] = True; plen[g] = 1
        r = gomoku_cpp.step(pool, g, a)
        if r:
            fin[g] = True; res[g] = r; mgr.reset_game(g)
        else:
            mgr.apply_move(g, a, p0[g], p1[g])

    occ_g = torch.from_numpy(p0 | p1).to(device)

    while True:
        act = np.where(~fin)[0]
        if len(act) == 0:
            break
        st = torch.from_numpy(act).to(device)
        cp = int(plen[act[0]]) % 2
        dp = torch.zeros(len(act), 1, dtype=torch.long, device=device)
        dplr = torch.full((len(act), 1), cp, dtype=torch.long, device=device)
        dl = torch.ones(len(act), dtype=torch.long, device=device)
        lp, lv = model.evaluate_mcts_leaves(dp, dplr, kv, st, dl)
        lp = lp.masked_fill(occ_g[act], -1e9)
        torch.cuda.synchronize()
        mgr.expand_roots(act.astype(np.int32),
                         torch.softmax(lp, -1).cpu().numpy().astype(np.float32),
                         lv.cpu().numpy().astype(np.float32))
        for _ in range(S):
            sel = mgr.select_all()
            if sel['max_path_len'] == 0:
                continue
            vi = np.where(sel['valid_mask'])[0]
            if len(vi) == 0:
                continue
            pt = torch.from_numpy(np.ascontiguousarray(sel['pos_dense'][vi])).to(device)
            pl2 = torch.from_numpy(np.ascontiguousarray(sel['plr_dense'][vi])).to(device)
            lt = torch.from_numpy(np.ascontiguousarray(sel['leaf_lengths'][vi])).to(device)
            sl = torch.from_numpy(np.ascontiguousarray(sel['game_indices'][vi])).to(device)
            lp2, lv2 = model.evaluate_mcts_leaves(pt, pl2, kv, sl, lt)
            torch.cuda.synchronize()
            ot = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(device).bool()
            lp2 = lp2.masked_fill(ot, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32),
                                  torch.softmax(lp2, -1).cpu().numpy().astype(np.float32),
                                  lv2.cpu().numpy().astype(np.float32))
        rp = mgr.get_root_policies()
        na = np.zeros(len(act), dtype=np.int64)
        np_ = np.zeros(len(act), dtype=np.int64)
        for i, g in enumerate(act):
            pol = rp[g].copy()
            pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0:
                a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                leg = np.where(~(p0[g] | p1[g]))[0]
                a = int(np.random.choice(leg)) if len(leg) > 0 else 0
            na[i] = a; np_[i] = plen[g] % 2
            ph[g].append(a); plh[g].append(np_[i])
            mp[g].append(rp[g].copy())
            if np_[i] == 0:
                p0[g, a] = True
            else:
                p1[g, a] = True
            plen[g] += 1
        dec_p = torch.from_numpy(na).to(device)
        dec_pl = torch.from_numpy(np_).to(device)
        model.decode(dec_p, dec_pl, kv, torch.from_numpy(act).to(device))
        for i, g in enumerate(act):
            r = gomoku_cpp.step(pool, g, int(na[i]))
            if r:
                fin[g] = True; res[g] = r; mgr.reset_game(g)
            else:
                mgr.apply_move(g, int(na[i]), p0[g], p1[g])

    del kv; torch.cuda.empty_cache()

    traj = []
    for g in range(G):
        L = plen[g]; rv = res[g]
        rv = 3 if rv == 0 else rv
        vt = np.zeros((L, 2), dtype=np.float32)
        for i in range(L):
            pl = plh[g][i]
            if rv == 3:
                vt[i] = [0.5, 0.5]
            elif (rv == 1 and pl == 0) or (rv == 2 and pl == 1):
                vt[i] = [1.0, 0.0]
            else:
                vt[i] = [0.0, 1.0]
        traj.append({
            'positions': np.array(ph[g], dtype=np.int64),
            'players': np.array(plh[g], dtype=np.int64),
            'actions': np.array(ph[g], dtype=np.int64),
            'mcts_policies': np.array(mp[g], dtype=np.float32),
            'value_targets': vt,
            'actual_len': L,
            'result': rv,
        })
    bw = int((res == 1).sum())
    ww = int((res == 2).sum())
    dr = int((res == 3).sum()) + int((res == 0).sum())
    return traj, plen.mean(), bw, ww, dr
