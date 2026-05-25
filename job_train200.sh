#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train200
#SBATCH --output=slurm_logs/slurm_train200_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=4

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo "=== Training 200 steps from scratch (lr=1e-4) ==="
python -u scripts/train_replay.py \
    --G 512 --M 8 --S 64 --n_steps 200 \
    --ckpt_dir checkpoints/run200 \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 1e-4 \
    --skip_elo
