#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=vizmatch
#SBATCH --output=slurm_logs/slurm_vizmatch_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate
export OMP_NUM_THREADS=6

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo "=== Head-to-head: step_38 vs step_199 (S=256, branch fix experiment) ==="
mkdir -p output
python -u scripts/viz_match.py \
    checkpoints/run200_G1024_0527_0740 \
    34 199 \
    --games 4 --S 256
