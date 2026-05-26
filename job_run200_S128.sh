#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=run200_S128
#SBATCH --output=slurm_logs/slurm_run200_S128_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

CKPT_DIR="checkpoints/run200_S128_$(date +%m%d_%H%M)"
mkdir -p "$CKPT_DIR" slurm_logs

echo "=== Launching training (S=128) in background ==="
python -u scripts/train_replay.py \
    --G 512 --M 8 --S 128 --n_steps 200 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 1e-4 \
    --skip_elo \
    > slurm_logs/train_$$.log 2>&1 &
TRAIN_PID=$!

echo "=== Launching monitor (max_gap=8) in background ==="
python -u scripts/elo_monitor_continuous.py \
    --ckpt_dir "$CKPT_DIR" \
    --G 256 --M 4 --S 16 \
    --max_gap 8 --interval 30 \
    > slurm_logs/monitor_$$.log 2>&1 &
MONITOR_PID=$!

echo "Train PID: $TRAIN_PID  Monitor PID: $MONITOR_PID"

# Wait for training to finish; kill monitor after
wait $TRAIN_PID
RC=$?
echo "Training finished (exit=$RC), stopping monitor..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

echo "=== All done ==="
cat slurm_logs/train_$$.log | tail -5
