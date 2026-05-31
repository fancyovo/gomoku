#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=analyze_mcts_visit
#SBATCH --output=slurm_logs/slurm_analyze_mcts_visit_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate
export OMP_NUM_THREADS=4
echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo "=== Running MCTS visit analysis ==="
python -u scripts/analyze_mcts_visit.py \
    --ckpt checkpoints/run500_fix1stB_0531_1423_1972/step_000109.pt \
    --S 16 32 64 128 256 \
    --c_puct 1.0 \
    2>&1
echo "=== Done ==="
