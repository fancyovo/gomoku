#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --exclude=anode05
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=G1024
#SBATCH --output=slurm_logs/slurm_G1024_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

CKPT_DIR="checkpoints/run200_G1024_$(date +%m%d_%H%M)"
mkdir -p "$CKPT_DIR"

echo "=== Launching training (G=1024, single loss, train_fraction=0.5) ==="
python -u scripts/train_replay.py \
    --G 1024 --M 8 --S 64 --n_steps 200 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 1e-4 \
    --single_loss \
    --train_fraction 0.5 \
    --skip_elo \
    > slurm_logs/train_G1024_$$.log 2>&1 &
TRAIN_PID=$!

echo "=== Launching monitor (max_gap=5, MCTS+policy+value) ==="
python -u scripts/elo_monitor_continuous.py \
    --ckpt_dir "$CKPT_DIR" \
    --G 256 --M 4 --S 16 \
    --max_gap 5 --interval 30 \
    > slurm_logs/monitor_G1024_$$.log 2>&1 &
MONITOR_PID=$!

echo "Train PID: $TRAIN_PID  Monitor PID: $MONITOR_PID"
echo "CKPT: $CKPT_DIR"

# Wait for BOTH to finish
wait $TRAIN_PID
RC_TRAIN=$?
echo "Training finished (exit=$RC_TRAIN), waiting for monitor to catch up..."
wait $MONITOR_PID
RC_MON=$?
echo "Monitor finished (exit=$RC_MON)"
echo "=== All done ==="
