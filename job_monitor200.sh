#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --nodelist=anode07
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=monitor200
#SBATCH --output=slurm_logs/slurm_monitor200_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=4

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo "=== Continuous ELO monitor: checkpoints/run200 ==="
python -u scripts/elo_monitor_continuous.py \
    --ckpt_dir checkpoints/run200 \
    --G 256 --M 4 --S 16 \
    --max_gap 8 --interval 30
