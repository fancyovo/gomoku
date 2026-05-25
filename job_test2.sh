#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=test2
#SBATCH --output=slurm_logs/slurm_test2_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/
python -u scripts/train_replay.py --G 512 --M 8 --S 64 --n_steps 2 --ckpt_dir checkpoints/test2 --data_dir data/init_pool --from_scratch
