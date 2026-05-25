#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=run20_pool16
#SBATCH --output=slurm_logs/slurm_run20_pool16_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

echo ""
echo "=== Generating extra 8*G initial pool data ==="
python -u scripts/gen_data.py \
    --batch_size 512 --num_simulations 256 --leaves_per_game 4 \
    --output_dir data/init_pool \
    --num_files 8 --start_idx 8

echo ""
echo "=== Training 20 steps from scratch (pool_mult=16, lr=3e-4) ==="
python -u scripts/train_replay.py \
    --G 512 --M 8 --S 64 --n_steps 20 \
    --ckpt_dir checkpoints/run20_pool16 \
    --data_dir data/init_pool \
    --pool_mult 16 \
    --from_scratch \
    --lr 3e-4 \
    --skip_elo

echo ""
echo "=== ELO evaluation + plotting (S=16) ==="
python -u scripts/eval_elo_curve.py --ckpt_dir checkpoints/run20_pool16 --G 256 --M 4 --S 16
