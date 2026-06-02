#!/usr/bin/env python3
"""Self-play one game with fix_v3 latest model, render as GIF."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ModelConfig, GomokuTransformer
from training.replay import run_selfplay
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
DEVICE = torch.device('cuda')

def board_to_img(p0, p1, last_move=None, move_num=None, total=None, result_text=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(15):
        ax.plot([i, i], [0, 14], 'k-', lw=0.5); ax.plot([0, 14], [i, i], 'k-', lw=0.5)
    for r in range(15):
        for c in range(15):
            idx = r*15+c
            if p0[idx]: ax.add_patch(plt.Circle((c, 14-r), 0.42, color='black', zorder=3))
            elif p1[idx]: ax.add_patch(plt.Circle((c, 14-r), 0.42, color='white', ec='black', lw=1, zorder=3))
    if last_move is not None:
        ax.add_patch(plt.Circle((last_move%15, 14-last_move//15), 0.12, color='red', zorder=4))
    ax.set_xlim(-0.5, 14.5); ax.set_ylim(-0.5, 14.5); ax.set_aspect('equal'); ax.axis('off')
    title = f"Move {move_num}/{total}" if move_num else ''
    if result_text: title = result_text
    ax.set_title(title, fontsize=14)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3].copy()
    plt.close(fig); return img

def main():
    ckpt_dir = 'checkpoints/run500_fix_v3_0601_1156_31735'
    steps = sorted(glob.glob(f'{ckpt_dir}/step_*.pt'))
    ckpt = steps[-1]; step = int(ckpt.split('_')[-1].split('.')[0])
    print(f"Using {ckpt} (step {step})")
    cfg = ModelConfig(d_model=128, n_layers=16, n_heads=4, d_ff=256, board_size=15, n_shared=4, n_policy=4, n_value=4)
    model = GomokuTransformer(cfg).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    traj, avg_len, bw, ww, dr = run_selfplay(model, DEVICE, G=1, M=8, S=256, c_puct=1.0)
    g = traj[0]; L = g['actual_len']
    result = g['result']; winner = 'Black' if result==1 else 'White' if result==2 else 'Draw'
    print(f"Game: {L} plies, result={winner}")
    frames = []
    p0 = np.zeros(225, bool); p1 = np.zeros(225, bool)
    for i in range(L):
        a = int(g['positions'][i]); pl = int(g['players'][i])
        print(f"  Step {i+1}: {'Black' if pl==0 else 'White'} plays ({a//15},{a%15})")
        if pl == 0: p0[a] = True
        else: p1[a] = True
        frames.append(board_to_img(p0, p1, last_move=a, move_num=i+1, total=L))
    frames.append(board_to_img(p0, p1, result_text=f"{winner} wins ({L} moves)"))
    out = f'gomoku_fix_v3_step{step}.gif'
    imageio.mimsave(out, frames, duration=1000, loop=0)
    print(f"GIF: {out}")
    torch.cuda.empty_cache()
    print("=== Done ===")
if __name__ == '__main__': main()
