#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=elo_lr1e4
#SBATCH --output=slurm_logs/slurm_elo_lr1e4_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=4

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo "=== ELO: run20_lr1e4 (sparse, |i-j|<=4) ==="
python -u scripts/eval_elo_sparse.py \
    --ckpt_dir checkpoints/run20_lr1e4 \
    --max_gap 4 --G 256 --M 4 --S 16
