#!/usr/bin/env python3
"""Benchmark: batch N models against each other, all pairs in parallel."""
import sys, os, time, torch, yaml, numpy as np
sys.path.insert(0, '..' if os.path.dirname(__file__) == 'scripts' else '.')
sys.path.insert(0, 'src')
from model import ModelConfig, GomokuTransformer
import gomoku_cpp

BOARD_SIZE = 15; N_CELLS = 225; MAX_MOVES = 150; PAGE = 32

@torch.inference_mode()
def play_multi_model(models, pair_assignments, games_per_dir, device):
    """models: list of (name, model)
    pair_assignments: list of (model_a_idx, model_b_idx, n_games)
        Half games: model_a=black, half: model_b=black.
    Returns list of (wa, wb, d) per pair.
    """
    # Build flat batch: all games from all pairs
    total_games = 0
    game_info = []
    black_model_idx = []   # which model plays black for each game
    white_model_idx = []   # which model plays white for each game
    for pi, (a_idx, b_idx, n) in enumerate(pair_assignments):
        for g in range(n):
            total_games += 1
            is_a_black = (g % 2 == 0)
            black_model_idx.append(a_idx if is_a_black else b_idx)
            white_model_idx.append(b_idx if is_a_black else a_idx)
            game_info.append((pi, is_a_black))
    b = total_games
    if b == 0: return []
    black_m = torch.tensor(black_model_idx, device=device)
    white_m = torch.tensor(white_model_idx, device=device)

    # Per-model KV caches
    caches = []
    for mi in range(len(models)):
        caches.append(models[mi][1].create_cache(max_games=b, max_cache_len=MAX_MOVES))

    a_black = torch.tensor([gi[1] for gi in game_info], device=device)
    occupied = torch.zeros(b, N_CELLS, dtype=torch.bool, device=device)
    pool = gomoku_cpp.GamePool(b)
    all_done = torch.zeros(b, dtype=torch.bool, device=device)
    winners  = torch.zeros(b, dtype=torch.long, device=device)
    accum_acts = torch.zeros(b, PAGE, dtype=torch.long, device=device)
    accum_cnt = 0
    idx_b = torch.arange(b, device=device)

    def flush_cpp():
        nonlocal accum_cnt
        if accum_cnt == 0: return
        acts_cpu = accum_acts[:, :accum_cnt].cpu().numpy().astype(np.int32)
        for pg_start in range(0, accum_cnt, PAGE):
            pg_end = min(pg_start + PAGE, accum_cnt)
            pg_acts = np.zeros((b, PAGE), dtype=np.int32)
            pg_acts[:, :pg_end - pg_start] = acts_cpu[:, pg_start:pg_end]
            results = pool.execute_block(np.arange(b, dtype=np.int32), pg_acts)
            for i in range(b):
                if all_done[i]: continue
                es = int(results[i, 0])
                if es >= 0:
                    all_done[i] = True
                    winners[i] = int(results[i, 1])
        accum_cnt = 0

    # First move: gather all models' first_move_logits, index by black_m
    M = len(models)
    fm_all = torch.stack([m[1].first_move_logits for m in models])  # (M, 225)
    fm_buf = fm_all[black_m]  # (b, 225)

    fm_buf = fm_buf.masked_fill(occupied, -1e9)
    probs = torch.softmax(fm_buf.float(), dim=-1)
    first = torch.multinomial(probs, 1).squeeze(-1)
    occupied[idx_b, first] = True
    accum_acts[:, 0] = first; accum_cnt = 1
    last_act = first
    last_plr = torch.zeros(b, dtype=torch.long, device=device)

    # Main loop
    M = len(models)
    for step in range(1, MAX_MOVES):
        cp = step % 2

        # ALL M models decode ALL b games
        all_logits = torch.zeros(M, b, N_CELLS, device=device)
        for mi in range(M):
            all_logits[mi] = models[mi][1].decode(last_act, last_plr, caches[mi], idx_b)

        # Pick correct model's logits per game (vectorized)
        model_for_game = black_m if cp == 0 else white_m
        logits = all_logits[model_for_game, idx_b]

        logits = logits.masked_fill(occupied, -1e9)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        occupied.scatter_(1, actions.unsqueeze(1), True)
        last_act = actions
        last_plr = torch.full((b,), cp, dtype=torch.long, device=device)

        accum_acts[:, accum_cnt] = actions
        accum_cnt += 1
        if accum_cnt == PAGE:
            flush_cpp()
            if all_done.all(): break

    flush_cpp()
    winners[(winners == 0) & ~all_done] = 3

    results = []
    for pi, (a_idx, b_idx, n) in enumerate(pair_assignments):
        start = sum(p[2] for p in pair_assignments[:pi])
        wa = wb = dg = 0
        for g in range(n):
            gi = start + g
            w = winners[gi].item()
            is_a_black = (g % 2 == 0)
            if w == 1:
                wa += 1 if is_a_black else 0; wb += 0 if is_a_black else 1
            elif w == 2:
                wa += 0 if is_a_black else 1; wb += 1 if is_a_black else 0
            else: dg += 1
        results.append((wa, wb, dg))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--games_per_dir", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open("configs/default.yaml") as f: cfg = yaml.safe_load(f)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    device = torch.device(args.device)

    # Load N different checkpoints (or same checkpoint duplicated for benchmark)
    ckpt_dir = "checkpoints/base"
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                   key=lambda x: int(x.split("_")[1].split(".")[0]))
    if not ckpts:
        print("No checkpoints found, creating identical models")
        models = [(f"m{i}", GomokuTransformer(model_cfg).to(device).eval()) for i in range(args.n_models)]
    else:
        step = max(1, len(ckpts) // args.n_models)
        sel = [ckpts[i * step] for i in range(min(args.n_models, len(ckpts)))]
        while len(sel) < args.n_models:
            sel.append(sel[-1])  # duplicate last if not enough
        print(f"Using checkpoints: {sel}")
        models = []
        for s in sel:
            m = GomokuTransformer(model_cfg).to(device).eval()
            m.load_state_dict(torch.load(os.path.join(ckpt_dir, s), map_location=device))
            models.append((s, m))

    # Build pair assignments (all pairs)
    n = args.n_models
    pairs = [(i, j, args.games_per_dir) for i in range(n) for j in range(n) if i != j]
    total_games = sum(p[2] for p in pairs)
    print(f"\n{n} models → {len(pairs)} pairs × {args.games_per_dir} games = {total_games} total games")
    print(f"KV caches: {n} × {total_games} × {MAX_MOVES}")
    print(f"Per step: {n} decode calls on batch size {total_games}")

    # Warmup
    print("\nWarmup...")
    play_multi_model(models, pairs[:2], 2, device)
    torch.cuda.synchronize()

    # Time it
    print("Benchmarking...")
    t0 = time.perf_counter()
    results = play_multi_model(models, pairs, args.games_per_dir, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"\nTotal: {elapsed:.2f}s")
    print(f"Per game: {elapsed/total_games*1000:.1f}ms")
    print(f"Per pair ({args.games_per_dir} games): {elapsed/len(pairs):.3f}s")
    print(f"Games/sec: {total_games/elapsed:.0f}")


if __name__ == "__main__":
    main()
