#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=gm5_fix
#SBATCH --output=slurm_logs/slurm_5steps_fix_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku

source .venv/bin/activate

echo "=== Rebuilding C++ module (MCTS sign fix) ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -3
# Copy to venv so it's importable
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/
echo "Build done."

python -u << 'PYEOF'
import torch, sys, os, time, json, numpy as np, glob
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from model import ModelConfig, GomokuTransformer
from training.augment import SYM_TABLE, N_SYMS
_INV_SYM = [0, 3, 2, 1, 4, 5, 6, 7]
INV_SYM_TABLE = SYM_TABLE[_INV_SYM]
from training.loss import alphago_zero_loss, reinforce_loss
import gomoku_cpp

DEVICE = torch.device('cuda')
BOARD_SIZE = 15
G, M, S = 512, 8, 64
TRAIN_BATCH = 128
MAX_EPOCHS = 100000
EARLY_STOP = 20
N_STEPS = 5
ELO_G, ELO_M, ELO_S = 128, 4, 16
CHECKPOINT_DIR, OUTPUT_DIR = 'checkpoints/train_fix', 'output'

# ========== Self-play (Dirichlet ON for exploration) ==========
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
    mcts_pols = [[] for _ in range(G)]; pos_lens = np.zeros(G, dtype=np.int32)
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
            lp2, lv2 = model.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
            occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
            lp2 = lp2.masked_fill(occ_t, -1e9)
            mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp2, -1).cpu().numpy().astype(np.float32), lv2.cpu().numpy().astype(np.float32))
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
            mcts_pols[g].append(rp[g].copy())
            if new_plrs[i] == 0: p0[g, a] = True
            else: p1[g, a] = True
            pos_lens[g] += 1
        dec_pos = torch.from_numpy(new_actions).to(DEVICE)
        dec_plr = torch.from_numpy(new_plrs).to(DEVICE)
        model.decode(dec_pos, dec_plr, kv, torch.from_numpy(active).to(DEVICE))
        for i, g in enumerate(active):
            r = gomoku_cpp.step(pool, g, int(new_actions[i]))
            if r: finished[g] = True; results[g] = r; mgr.reset_game(g)
            else: mgr.apply_move(g, int(new_actions[i]), p0[g], p1[g])
    trajectories = []
    for g in range(G):
        L = pos_lens[g]; r_val = results[g]; r_val = 3 if r_val == 0 else r_val
        val_t = np.zeros((L, 2), dtype=np.float32)
        for i in range(L):
            plr = plr_hist[g][i]
            if r_val == 3:
                val_t[i] = [0.5, 0.5]
            elif (r_val == 1 and plr == 0) or (r_val == 2 and plr == 1):
                val_t[i] = [1.0, 0.0]
            else:
                val_t[i] = [0.0, 1.0]
        trajectories.append({'positions': np.array(pos_hist[g], dtype=np.int64),
                             'players': np.array(plr_hist[g], dtype=np.int64),
                             'actions': np.array(pos_hist[g], dtype=np.int64),
                             'mcts_policies': np.array(mcts_pols[g], dtype=np.float32),
                             'value_targets': val_t, 'actual_len': L, 'result': r_val})
    del kv; torch.cuda.empty_cache()
    bw = int((results == 1).sum()); ww = int((results == 2).sum())
    dr = int((results == 3).sum()) + int((results == 0).sum())
    return trajectories, results, pos_lens, bw, ww, dr

# ========== Training ==========
def train_with_early_stop(model, trajectories):
    n_traj = len(trajectories)
    idx = np.random.permutation(n_traj)
    n_train_traj = int(n_traj * 0.8)

    def augment_trajs(traj_list):
        samples = []
        for traj in traj_list:
            L = traj['actual_len']
            pos = torch.from_numpy(traj['positions'][:L]); plr = torch.from_numpy(traj['players'][:L])
            pol = torch.from_numpy(traj['mcts_policies'][:L]); val = torch.from_numpy(traj['value_targets'][:L])
            for s in range(N_SYMS):
                remap = SYM_TABLE[s]; inv_remap = INV_SYM_TABLE[s]
                samples.append({'positions': remap[pos], 'players': plr,
                                'mcts_policies': pol[:, inv_remap], 'value_targets': val,
                                'actual_len': L})
        return samples

    train_samples = augment_trajs([trajectories[i] for i in idx[:n_train_traj]])
    test_samples = augment_trajs([trajectories[i] for i in idx[n_train_traj:]])
    samples = train_samples + test_samples
    n_train = len(train_samples)
    class DS(torch.utils.data.Dataset):
        def __init__(self, s, indices): self.s = [s[i] for i in indices]
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]
    def collate(batch):
        max_len = max(s['positions'].shape[0] for s in batch); B_ = len(batch)
        pos = torch.zeros(B_, max_len, dtype=torch.long)
        plr = torch.zeros(B_, max_len, dtype=torch.long)
        pol = torch.zeros(B_, max_len, 225)
        val_t = torch.zeros(B_, max_len, 2)
        val_w = torch.zeros(B_, max_len)
        mask = torch.zeros(B_, max_len, dtype=torch.bool)
        for i, s in enumerate(batch):
            L_ = s['actual_len']
            pos[i, :L_] = s['positions']; plr[i, :L_] = s['players']
            pol[i, :L_-1] = s['mcts_policies']
            val_t[i, :L_] = s['value_targets']
            mask[i, :L_] = True
            for j in range(L_):
                d = L_ - 1 - j
                val_w[i, j] = 0.5 ** (d / 5.0)
        return {'positions': pos, 'players': plr, 'mcts_policies': pol,
                'value_targets': val_t, 'value_weights': val_w, 'mask': mask}
    train_ds = DS(train_samples, list(range(len(train_samples))))
    test_ds = DS(test_samples, list(range(len(test_samples))))

    def eval_epoch(ds):
        model.eval(); dl = torch.utils.data.DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=False, collate_fn=collate)
        total_loss, n_batch = 0.0, 0
        with torch.inference_mode():
            for batch in dl:
                pos = batch['positions'].to(DEVICE); plr = batch['players'].to(DEVICE)
                pol_t = batch['mcts_policies'].to(DEVICE); val_t = batch['value_targets'].to(DEVICE)
                m = batch['mask'].to(DEVICE); B_, L_ = pos.shape
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    p, v = model(pos, plr)
                if L_ > 1:
                    pp = p[:, :-1, :].float(); vv = v[:, :-1, :].float()
                    tp = pol_t[:, :-1, :]; tv = val_t[:, :-1, :]
                    tw = batch['value_weights'].to(DEVICE)[:, :-1]; pm = m[:, :-1]
                    loss, pl, vl = alphago_zero_loss(
                        pp.reshape(-1, 225), tp.reshape(-1, 225),
                        vv.reshape(-1, 2), tv.reshape(-1, 2), pm.reshape(-1),
                        value_weights=tw.reshape(-1))
                    rew_scalar = val_t[:, 0, 0] - val_t[:, 0, 1]
                    fm = model.first_move_logits.unsqueeze(0).expand(B_, -1)
                    fl, _, _ = reinforce_loss(fm.float(), pos[:, 0], rew_scalar, m[:, 0])
                    total_loss += (loss + fl).item(); n_batch += 1
        return total_loss / max(n_batch, 1) if n_batch > 0 else 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_test_loss = float('inf'); best_state = None; best_epoch = 0; no_improve = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); dl = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, collate_fn=collate)
        for batch in dl:
            pos = batch['positions'].to(DEVICE); plr = batch['players'].to(DEVICE)
            pol_t = batch['mcts_policies'].to(DEVICE); val_t = batch['value_targets'].to(DEVICE)
            m = batch['mask'].to(DEVICE); B_, L_ = pos.shape
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p, v = model(pos, plr)
            if L_ > 1:
                pp = p[:, :-1, :].contiguous(); vv = v[:, :-1, :].contiguous()
                tp = pol_t[:, :-1, :].contiguous(); tv = val_t[:, :-1, :].contiguous()
                pm = m[:, :-1].contiguous()
                loss, pl, vl = alphago_zero_loss(
                    pp.reshape(-1, 225).float(), tp.reshape(-1, 225),
                    vv.reshape(-1, 2).float(), tv.reshape(-1, 2), pm.reshape(-1))
                rew_scalar = val_t[:, 0, 0] - val_t[:, 0, 1]
                fm = model.first_move_logits.unsqueeze(0).expand(B_, -1)
                fl, _, _ = reinforce_loss(fm.float(), pos[:, 0], rew_scalar, m[:, 0])
                loss = loss + fl; loss.backward()
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

# ========== ELO match (Dirichlet OFF for eval) ==========
@torch.inference_mode()
def play_match(model_a, model_b):
    G_ = ELO_G; pool = gomoku_cpp.GamePool(G_); pool.reset_all()
    def make_mgr():
        m = gomoku_cpp.MCTSManager(G_, seed_base=np.random.randint(0, 2**31))
        m.c_puct = 1.0; m.leaves_per_game = ELO_M
        m.dirichlet_eps = 0.0  # NO noise in evaluation!
        m.init_roots(np.zeros((G_, 225), dtype=bool), np.zeros((G_, 225), dtype=bool), np.zeros(G_, dtype=np.int32))
        return m
    mgr_a = make_mgr(); mgr_b = make_mgr()
    kva = model_a.create_cache(max_games=G_, max_cache_len=250); kvb = model_b.create_cache(max_games=G_, max_cache_len=250)
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
    model_a.prefill(fa_t, plr_t, kva, list(range(G_))); model_b.prefill(fa_t, plr_t, kvb, list(range(G_)))
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
                lp2, lv2 = mdl.evaluate_mcts_leaves(pos_t, plr_t2, kv, slots_t, lens_t); torch.cuda.synchronize()
                occ_t = torch.from_numpy(np.ascontiguousarray(sel['occ_dense'][vi])).to(DEVICE).bool()
                lp2 = lp2.masked_fill(occ_t, -1e9)
                mgr.expand_and_backup(vi.astype(np.int32), torch.softmax(lp2, -1).cpu().numpy().astype(np.float32), lv2.cpu().numpy().astype(np.float32))
        rp_a = mgr_a.get_root_policies(); rp_b = mgr_b.get_root_policies()
        new_actions = np.zeros(len(active), dtype=np.int64)
        for i, g in enumerate(active):
            use_a = ((cp == 0) == a_black[g])
            pol = (rp_a[g] if use_a else rp_b[g]).copy(); pol[p0[g] | p1[g]] = 0
            if pol.sum() > 0: a = int(np.random.choice(225, p=pol / pol.sum()))
            else:
                legal = np.where(~(p0[g] | p1[g]))[0]; a = int(np.random.choice(legal)) if len(legal) > 0 else 0
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

# ========== Main ==========
os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15)
model = GomokuTransformer(cfg).to(DEVICE).eval()
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
print(f'MCTS sign fix applied. Self-play: Dirichlet ON. ELO eval: Dirichlet OFF.')
print(f'G={G} M={M} S={S} | ELO_G={ELO_G} ELO_M={ELO_M} ELO_S={ELO_S}')
history = {'avg_len': [], 'sp_time': [], 'tr_time': [], 'epochs': [], 'test_loss': [], 'black_wr': [], 'train_p': [], 'train_v': []}

for step in range(N_STEPS):
    t0 = time.perf_counter(); model.eval()
    traj, results, pos_lens, bw, ww, dr = run_selfplay(model)
    torch.cuda.synchronize(); t_sp = time.perf_counter() - t0
    avg_len = pos_lens.mean(); rc = Counter(results)
    history['avg_len'].append(float(avg_len)); history['sp_time'].append(t_sp)
    history['black_wr'].append(bw / (bw + ww) if (bw + ww) > 0 else 0.5)
    t0 = time.perf_counter()
    best_ep, best_loss = train_with_early_stop(model, traj)
    torch.cuda.synchronize(); t_tr = time.perf_counter() - t0
    history['tr_time'].append(t_tr); history['epochs'].append(best_ep); history['test_loss'].append(float(best_loss))

    # Per-component loss on a subset
    model.eval()
    samples = []
    for t_ in traj[:128]:
        L = t_['actual_len']
        pos = torch.from_numpy(t_['positions'][:L]); plr = torch.from_numpy(t_['players'][:L])
        pol = torch.from_numpy(t_['mcts_policies'][:L]); val = torch.from_numpy(t_['value_targets'][:L])
        samples.append({'positions': pos, 'players': plr, 'mcts_policies': pol, 'value_targets': val, 'actual_len': L})
    max_len = max(s['actual_len'] for s in samples)
    B_ = len(samples)
    pos_b = torch.zeros(B_, max_len, dtype=torch.long); plr_b = torch.zeros(B_, max_len, dtype=torch.long)
    pol_b = torch.zeros(B_, max_len, 225); val_b = torch.zeros(B_, max_len, 2)
    val_w_b = torch.zeros(B_, max_len); mask_b = torch.zeros(B_, max_len, dtype=torch.bool)
    for i, s in enumerate(samples):
        L_ = s['actual_len']
        pos_b[i,:L_] = s['positions']; plr_b[i,:L_] = s['players']
        pol_b[i,:L_-1] = s['mcts_policies']; val_b[i,:L_] = s['value_targets']; mask_b[i,:L_] = True
        for j in range(L_):
            d = L_ - 1 - j
            val_w_b[i, j] = 0.5 ** (d / 5.0)
    pos_b = pos_b.to(DEVICE); plr_b = plr_b.to(DEVICE); pol_b = pol_b.to(DEVICE); val_b = val_b.to(DEVICE); val_w_b = val_w_b.to(DEVICE); mask_b = mask_b.to(DEVICE)
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            p, v = model(pos_b, plr_b)
    pp = p[:, :-1, :].float(); vv = v[:, :-1, :].float()
    tp = pol_b[:, :-1, :]; tv = val_b[:, :-1, :]; tw = val_w_b[:, :-1]; pm = mask_b[:, :-1]
    _, pl, vl = alphago_zero_loss(pp.reshape(-1,225), tp.reshape(-1,225), vv.reshape(-1,2).float(), tv.reshape(-1,2), pm.reshape(-1), value_weights=tw.reshape(-1))
    history['train_p'].append(float(pl.item())); history['train_v'].append(float(vl.item()))

    print(f'[step {step+1:3d}/{N_STEPS}] len={avg_len:.0f} sp={t_sp:.0f}s tr={t_tr:.0f}s '
          f'epoch={best_ep} test_loss={best_loss:.4f} '
          f'train_p={pl.item():.3f} train_v={vl.item():.3f} '
          f'B={rc.get(1, 0)} W={rc.get(2, 0)} D={rc.get(3, 0)} '
          f'BWR={bw/(bw+ww)*100:.0f}%', flush=True)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'step_{step:06d}.pt')
    torch.save(model.state_dict(), ckpt_path + '.tmp'); os.replace(ckpt_path + '.tmp', ckpt_path)
    with open(os.path.join(OUTPUT_DIR, 'train_fix_history.json'), 'w') as f: json.dump(history, f)

# ========== ELO ==========
class UniformModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.config = cfg
        self.device = device
        self.first_move_logits = torch.nn.Parameter(torch.zeros(225))
    def forward(self, pos, plr):
        b, s = pos.shape
        return torch.zeros(b, s, 225, device=self.device), torch.zeros(b, s, 2, device=self.device)
    def create_cache(self, max_games, max_cache_len=250):
        m = GomokuTransformer(self.config).to(self.device)
        return m.create_cache(max_games, max_cache_len)
    def prefill(self, pos, plr, cache, indices):
        b = pos.shape[0]
        return torch.zeros(b, 225, device=self.device), torch.zeros(b, device=self.device)
    def decode(self, pos, plr, cache, indices):
        cache.advance(indices)
        return torch.zeros(len(indices), 225, device=self.device), torch.zeros(len(indices), device=self.device)
    def evaluate_mcts_leaves(self, pos, plr, cache, indices, path_lengths):
        b = pos.shape[0]
        return torch.zeros(b, 225, device=self.device), torch.zeros(b, device=self.device)
    @torch.inference_mode()
    def sample_first_moves(self, batch_size, device):
        return torch.randint(0, 225, (batch_size,), device=device)
uni_baseline = UniformModel(DEVICE).eval()

print(f'\nRunning ELO (no Dirichlet in eval)...')
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'step_*.pt')))
names = ['uniform'] + [os.path.basename(c) for c in ckpts]
elo_results = []
for i, na in enumerate(names):
    for nb in names[i + 1:]:
        print(f'  {na} vs {nb} ...', end=' ', flush=True)
        t0 = time.perf_counter()
        if na == 'uniform': ma = uni_baseline
        else:
            ma = GomokuTransformer(cfg).to(DEVICE).eval(); ma.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, na), map_location=DEVICE))
        if nb == 'uniform': mb = uni_baseline
        else:
            mb = GomokuTransformer(cfg).to(DEVICE).eval(); mb.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, nb), map_location=DEVICE))
        wa, wb, d = play_match(ma, mb); dt = time.perf_counter() - t0
        wr = wa / (wa + wb) if (wa + wb) > 0 else 0.5
        elo_results.append((na, nb, wa + d * 0.5, wb + d * 0.5))
        print(f'{wa}-{wb} WR={wr:.1%} ({dt:.0f}s)')
        if na != 'uniform': del ma
        if nb != 'uniform': del mb
        torch.cuda.empty_cache()

elo = compute_elo(elo_results)

print(f'\nELO Results:')
for n in ['uniform'] + [os.path.basename(c) for c in ckpts]:
    r = elo.get(n, 1500)
    tag = ' (baseline)' if n == 'uniform' else ''
    print(f'  {n:20s}: ELO={r:.0f}{tag}')
items_trained = [(int(n.split('_')[1].split('.')[0]), elo.get(n, 1500)) for n in [os.path.basename(c) for c in ckpts]]
items_trained.sort()
uni_elo = elo.get('uniform', None)
if len(items_trained) >= 2:
    delta = items_trained[-1][1] - items_trained[0][1]
    print(f'  ΔELO (step 0→{items_trained[-1][0]}) = {delta:+.0f}')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax in [axes[0,0], axes[0,1], axes[1,0]]: ax.grid(True, alpha=0.3)
axes[0,0].plot(history['avg_len'], 'o-', color='C0', markersize=6); axes[0,0].set_title('Game Length')
axes[0,1].plot(history['black_wr'], 'o-', color='C3', markersize=6)
axes[0,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4); axes[0,1].set_title('Black WR')
# p/v loss
ax_pv = axes[0,0].twinx()
ax_pv.plot(history['train_p'], 's-', color='C1', markersize=4, alpha=0.7, label='p_loss')
ax_pv.plot(history['train_v'], '^-', color='C2', markersize=4, alpha=0.7, label='v_loss')
ax_pv.legend(loc='upper right')
if items_trained:
    se, ra = zip(*items_trained)
    axes[1,0].plot(se, ra, '.-', color='C1', markersize=8, linewidth=2, label='trained')
    if uni_elo is not None:
        axes[1,0].axhline(y=uni_elo, color='red', linestyle='--', alpha=0.5, label=f'uniform baseline')
    axes[1,0].axhline(y=1500, color='gray', linestyle='--', alpha=0.4)
    axes[1,0].legend()
axes[1,0].set_title(f'ELO ({ELO_G} games/pair, Dirichlet OFF)')
steps_list = sorted(set(int(n.split('_')[1].split('.')[0]) for n in names if n.startswith('step_')))
n = len(steps_list)
if n >= 2:
    sti = {s: i for i, s in enumerate(steps_list)}; wr_mat = np.full((n, n), np.nan)
    for name_a, name_b, score_a, score_b in elo_results:
        if name_a == 'uniform' or name_b == 'uniform': continue
        s_a = int(name_a.split('_')[1].split('.')[0])
        s_b = int(name_b.split('_')[1].split('.')[0])
        if s_a in sti and s_b in sti and score_a + score_b > 0:
            wr_mat[sti[s_a], sti[s_b]] = score_a / (score_a + score_b)
    for ii in range(n):
        for jj in range(ii + 1, n):
            if not np.isnan(wr_mat[ii, jj]): wr_mat[jj, ii] = 1.0 - wr_mat[ii, jj]
    im = axes[1,1].imshow(wr_mat, cmap=plt.cm.RdYlBu_r, vmin=0.0, vmax=1.0, aspect='auto')
    axes[1,1].set_xticks(range(n)); axes[1,1].set_yticks(range(n))
    axes[1,1].set_xticklabels(steps_list, rotation=45, ha='right', fontsize=7)
    axes[1,1].set_yticklabels(steps_list, fontsize=7); axes[1,1].set_title('Win Rate Heatmap')
    plt.colorbar(im, ax=axes[1,1])
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'train_fix_analysis.png'), dpi=150); plt.close()
print(f'Plot: {OUTPUT_DIR}/train_fix_analysis.png')
PYEOF

echo "=== Done ==="
echo "Date: $(date)"
ls -la output/train_fix_analysis.png 2>/dev/null
ls -la checkpoints/train_fix/ 2>/dev/null
