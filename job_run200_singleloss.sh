#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=singleloss
#SBATCH --output=slurm_logs/slurm_singleloss_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate

export OMP_NUM_THREADS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

CKPT_DIR="checkpoints/run200_S64_singleloss_$(date +%m%d_%H%M)"
mkdir -p "$CKPT_DIR"

echo "=== Launching training (S=64, single combined loss) ==="
python -u scripts/train_replay.py \
    --G 512 --M 8 --S 64 --n_steps 200 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 1e-4 \
    --single_loss \
    --skip_elo \
    > slurm_logs/train_singleloss_$$.log 2>&1 &
TRAIN_PID=$!

echo "=== Launching monitor (max_gap=5) ==="
python -u scripts/elo_monitor_continuous.py \
    --ckpt_dir "$CKPT_DIR" \
    --G 256 --M 4 --S 16 \
    --max_gap 5 --interval 30 \
    > slurm_logs/monitor_singleloss_$$.log 2>&1 &
MONITOR_PID=$!

echo "Train PID: $TRAIN_PID  Monitor PID: $MONITOR_PID"
echo "CKPT: $CKPT_DIR"

wait $TRAIN_PID
RC=$?
echo "Training finished (exit=$RC), stopping monitor..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true
echo "=== All done ==="
