#!/usr/bin/env python3
"""Generate comprehensive ablation experiment report with visualizations."""

import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

B = 15; N = 225; cols = "ABCDEFGHIJKLMNO"
OUT = "output/report"

def entropy(cnt, total):
    return -sum((c/total)*np.log(c/total) for c in cnt.values())

def compute_elo(match_results):
    names = set()
    for a, b, _, _ in match_results: names.add(a); names.add(b)
    if not names: return {}
    elo = {n: 1500.0 for n in sorted(names)}
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

# Load data
raw_data = json.load(open("output/all_game_data.json"))
data = {}
for exp, exp_data in raw_data.items():
    data[exp] = {int(k): v for k, v in exp_data.items()}
stage_data = json.load(open("output/stage_data.json"))

# Load ELO
all_results = []
for fname in sorted(os.listdir("elo_caches")):
    if fname == ".gitkeep": continue
    cache = json.load(open(f"elo_caches/{fname}"))
    is_cross = (fname == "_cross.json")
    for key, val in cache.items():
        if key == "_meta": continue
        try: wa, wb, d = val[0], val[1], val[2]
        except: continue
        if is_cross:
            try:
                a_raw, b_raw = key.split("|")
                ea, sa = a_raw.split(":"); eb, sb = b_raw.split(":")
                a = f"{ea}:step_{int(sa):06d}"; b = f"{eb}:step_{int(sb):06d}"
                all_results.append((a, b, wa + d*0.5, wb + d*0.5))
            except: pass
        else:
            exp = fname.replace(".json", "")
            try:
                s1, s2 = key.split("_")
                a = f"{exp}:step_{int(s1):06d}"; b = f"{exp}:step_{int(s2):06d}"
                all_results.append((a, b, wa + d*0.5, wb + d*0.5))
            except: pass

joint_elo = compute_elo(all_results)

experiments = ["base", "fixed_entropy", "reward_decay", "scale_up"]
exp_colors = {"base": "C0", "fixed_entropy": "C1", "reward_decay": "C2", "scale_up": "C3"}

os.makedirs(OUT, exist_ok=True)

# ── Figure 1: Joint ELO with stages marked ─────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for exp in experiments:
    exp_elo = {}
    for name, r in joint_elo.items():
        if name.startswith(exp + ":"):
            step = int(name.split(":step_")[1])
            exp_elo[step] = r
    if not exp_elo: continue
    steps = sorted(exp_elo.keys())
    ratings = [exp_elo[s] for s in steps]
    ax.plot(steps, ratings, ".-", color=exp_colors[exp], markersize=2, linewidth=1.5, label=exp)
    # Mark stages
    for s in sorted(stage_data[exp]["stages"]):
        if s in exp_elo:
            ax.plot(s, exp_elo[s], "o", color=exp_colors[exp], markersize=8, markeredgecolor="white", markeredgewidth=1)

ax.axhline(y=1500, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("Training Step"); ax.set_ylabel("Joint ELO Rating")
ax.set_title("Figure 1: Joint ELO Trajectories (5 analysis stages marked per experiment)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_elo_trajectories.png", dpi=150); plt.close()

# ── Figure 2: Statistics evolution ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ["avg_len", "row_ent", "avg_dist"]
metric_labels = {"avg_len": "Avg Game Length", "row_ent": "Row Distribution Entropy", "avg_dist": "Avg Distance Between Consecutive Moves"}
for mi, metric in enumerate(metrics):
    ax = axes[mi//2][mi%2]
    for exp in experiments:
        if exp not in data: continue
        steps = sorted(int(k) for k in data[exp].keys())
        vals = [data[exp][s][metric] for s in steps]
        ax.plot(steps, vals, ".-", color=exp_colors[exp], markersize=6, linewidth=1.5, label=exp)
    ax.set_xlabel("Training Step"); ax.set_ylabel(metric_labels[metric])
    ax.set_title(metric_labels[metric]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Add entropy ratio subplot
ax = axes[1][1]
for exp in experiments:
    if exp not in data: continue
    steps = sorted(int(k) for k in data[exp].keys())
    vals = [data[exp][s]["row_ent"] / data[exp][s]["max_ent"] for s in steps]
    ax.plot(steps, vals, ".-", color=exp_colors[exp], markersize=6, linewidth=1.5, label=exp)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("Training Step"); ax.set_ylabel("Entropy / Max Entropy")
ax.set_title("Normalized Row Entropy (1.0 = uniform)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle("Figure 2: Game Statistics Evolution Across Training", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_statistics.png", dpi=150); plt.close()

# ── Figure 3-6: Per-experiment game boards ─────────────────
def plot_board(moves, ax, title):
    ax.set_xlim(-0.5, B-0.5); ax.set_ylim(-0.5, B-0.5); ax.set_aspect("equal"); ax.invert_yaxis()
    for i in range(B):
        ax.axhline(i, color="black", linewidth=0.2, alpha=0.3)
        ax.axvline(i, color="black", linewidth=0.2, alpha=0.3)
    for r,c in [(3,3),(3,7),(3,11),(7,3),(7,7),(7,11),(11,3),(11,7),(11,11)]:
        ax.plot(c, r, "ko", markersize=3)

    for i, pos in enumerate(moves[:99]):
        r, c = pos//B, pos%B
        is_black = (i%2==0); is_last = (i==len(moves)-1)
        color = "black" if is_black else "white"
        edge = "red" if is_last else ("#444" if is_black else "#888")
        circle = patches.Circle((c, r), 0.42, facecolor=color, edgecolor=edge,
                                linewidth=2 if is_last else 1, zorder=5)
        ax.add_patch(circle)
        nc = "white" if is_black else "black"
        ax.text(c, r, str(i+1), ha="center", va="center", fontsize=6, color=nc, fontweight="bold", zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

for exp in experiments:
    if exp not in data: continue
    stages = sorted(int(k) for k in data[exp].keys())
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4.5))
    if n == 1: axes = [axes]
    for i, s in enumerate(stages):
        elo_val = None
        for name, r in joint_elo.items():
            if name == f"{exp}:step_{s:06d}":
                elo_val = r; break
        d = data[exp][s]
        title = f"step {s}\nlen={d['avg_len']:.0f} ent={d['row_ent']:.2f}\nELO={elo_val:.0f}" if elo_val else f"step {s}\nlen={d['avg_len']:.0f}"
        moves = d["sample_game"]
        plot_board(moves, axes[i], title)
    plt.suptitle(f"Figure: {exp} — Game Board Evolution", fontsize=14)
    plt.tight_layout()
    fig_idx = 3 + experiments.index(exp)
    plt.savefig(f"{OUT}/fig{fig_idx}_{exp}_boards.png", dpi=150); plt.close()

# ── Figure 7: Row distribution heatmaps ────────────────────
fig, axes = plt.subplots(len(experiments), 5, figsize=(20, 4*len(experiments)))
for ei, exp in enumerate(experiments):
    if exp not in data: continue
    stages = sorted(int(k) for k in data[exp].keys())
    for si, s in enumerate(stages):
        ax = axes[ei][si]
        all_moves = data[exp][s]["all_moves"]
        row_dist = np.zeros(15)
        for g in all_moves:
            for m in g:
                row_dist[m//B] += 1
        row_dist /= row_dist.sum()
        ax.barh(range(15), row_dist, color=exp_colors[exp], alpha=0.7)
        ax.set_xlim(0, max(row_dist)*1.2)
        ax.invert_yaxis()
        ax.set_yticks(range(15))
        if si == 0: ax.set_ylabel(f"{exp}\nRow")
        if ei == 0: ax.set_title(f"step {s}")
        if si > 0: ax.set_yticklabels([])
        ax.grid(True, alpha=0.2, axis="x")

plt.suptitle("Figure 7: Row Distribution Evolution (per experiment, per stage)", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_row_distributions.png", dpi=150); plt.close()

# ── Figure 8: Summary comparison ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Game Length", "Row Entropy", "Move Distance"]
for ei, (metric, title) in enumerate(zip(["avg_len", "row_ent", "avg_dist"], titles)):
    ax = axes[ei]
    x = np.arange(len(experiments))
    width = 0.15
    for exp in experiments:
        if exp not in data: continue
        stages = sorted(int(k) for k in data[exp].keys())
        vals = [data[exp][s][metric] for s in stages]
        # Show first and last stage
        idx = experiments.index(exp)
        ax.bar(idx - width, vals[0], width, color=exp_colors[exp], alpha=0.4, label=f"{exp} start" if ei==0 else "")
        ax.bar(idx + width, vals[-1], width, color=exp_colors[exp], alpha=0.9, label=f"{exp} end" if ei==0 else "")
    ax.set_xticks(x); ax.set_xticklabels(experiments)
    ax.set_title(title)
    if ei == 0: ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Figure 8: Start vs End Comparison Across Experiments", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_summary_comparison.png", dpi=150); plt.close()

print(f"All figures saved to {OUT}/")
print(f"Files: {sorted(os.listdir(OUT))}")
