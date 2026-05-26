#!/usr/bin/env python3
"""Recover ELO plots from monitor log and game length from training log."""
import re, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def parse_monitor_log(path):
    """Extract MCTS and policy match results from monitor log."""
    mcts_results = []
    policy_results = []
    with open(path) as f:
        for line in f:
            # MCTS:  step_000038.pt vs step_000036.pt ... 200-56 WR=78.1% (25s)
            m = re.match(r'\s+(MCTS|Policy):\s*(\S+)\s+vs\s+(\S+)\s+\.\.\.\s+(\d+)-(\d+)', line)
            if m:
                match_type = m.group(1)
                a = m.group(2); b = m.group(3)
                wa = int(m.group(4)); wb = int(m.group(5))
                # draws embedded in WR calculation but not shown directly; assume 0
                if match_type == 'MCTS':
                    mcts_results.append((a, b, wa, wb))
                else:
                    policy_results.append((a, b, wa, wb))
    return mcts_results, policy_results

def parse_train_log(path):
    """Extract game lengths from training log."""
    game_lens = []
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for line in f:
            # Self-play: G=512 len=60 B=265 W=247 D=0 (144s)
            m = re.search(r'Self-play:.*len=(\d+)', line)
            if m:
                game_lens.append(int(m.group(1)))
    return game_lens

def plot_elo(results, out_path, title, noisy_elo=None):
    if not results:
        print(f"No data for {out_path}")
        return
    elo = compute_elo([(a, b, wa, wb) for a, b, wa, wb in results])
    names = sorted(elo.keys(), key=lambda n: (-1 if n == 'noisy_uniform' else int(n.split('_')[1].split('.')[0])))
    steps = [(int(n.split('_')[1].split('.')[0]), elo[n]) for n in names if n.startswith('step_')]
    steps.sort()
    if not steps:
        return
    if noisy_elo is None:
        noisy_elo = elo.get('noisy_uniform', 1500)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([s[0] for s in steps], [s[1] for s in steps], '.-', markersize=3, linewidth=1.5)
    ax.axhline(y=noisy_elo, color='gray', linestyle=':', label=f'Noisy uniform (ELO={noisy_elo:.0f})')
    ax.set_xlabel('Step'); ax.set_ylabel('ELO')
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} ({len(steps)} steps, max ELO={max(s[1] for s in steps):.0f} at step {max(steps, key=lambda x: x[1])[0]})")

def plot_game_len(lens, out_path):
    if not lens:
        print("No game length data")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(lens)), lens, '.-', markersize=4)
    ax.set_xlabel('Step'); ax.set_ylabel('Avg Game Length')
    ax.set_title('Self-Play Game Length per Step (S=64)')
    ax.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} ({len(lens)} steps)")

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output/recovered"
    os.makedirs(out_dir, exist_ok=True)

    # Find monitor log
    monitor_logs = [
        "slurm_logs/slurm_monitor200_8821.out",
        "slurm_logs/history/slurm_monitor200_8821.out",
    ]
    monitor_path = None
    for p in monitor_logs:
        if os.path.exists(p):
            monitor_path = p; break
    if not monitor_path:
        # Find largest monitor log
        import glob
        candidates = glob.glob("slurm_logs/*monitor*8821*") + glob.glob("slurm_logs/history/*monitor*8821*")
        if candidates:
            monitor_path = max(candidates, key=os.path.getsize)
    if not monitor_path:
        print("No monitor log found!")
        return

    print(f"Parsing: {monitor_path}")
    mcts_results, policy_results = parse_monitor_log(monitor_path)
    print(f"  MCTS pairs: {len(mcts_results)}, Policy pairs: {len(policy_results)}")

    plot_elo(mcts_results, f"{out_dir}/elo_mcts_recovered.png",
             f"ELO MCTS (S=16, S64 training, recovered)")
    plot_elo(policy_results, f"{out_dir}/elo_policy_recovered.png",
             f"ELO Policy-only (S=16, recovered)")

    # Game length from training log
    train_logs = [
        "slurm_logs/slurm_train200_8809.out",
        "slurm_logs/history/slurm_train200_8809.out",
    ]
    train_path = None
    for p in train_logs:
        if os.path.exists(p):
            train_path = p; break
    if train_path:
        lens = parse_train_log(train_path)
        print(f"  Game lengths: {len(lens)} steps")
        plot_game_len(lens, f"{out_dir}/game_length_recovered.png")

if __name__ == '__main__':
    main()
