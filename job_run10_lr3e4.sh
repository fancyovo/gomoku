#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=run10_lr3e4
#SBATCH --output=slurm_logs/slurm_run10_lr3e4_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo ""
echo "=== Training 10 steps from scratch (lr=3e-4) ==="
python -u scripts/train_replay.py \
    --G 512 --M 8 --S 64 --n_steps 10 \
    --ckpt_dir checkpoints/run10_lr3e4 \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 3e-4 \
    --skip_elo

echo ""
echo "=== ELO evaluation + plotting (S=16, with noisy_uniform baseline) ==="
python -u scripts/eval_elo_curve.py --ckpt_dir checkpoints/run10_lr3e4 --G 256 --M 4 --S 16
