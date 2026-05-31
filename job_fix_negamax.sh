#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --exclude=anode05
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fix_negamax
#SBATCH --output=slurm_logs/slurm_fix_negamax_%j.out
set -euo pipefail
cd /home/scc/pb24511935/gomoku
source .venv/bin/activate
export OMP_NUM_THREADS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== Building C++ module ==="
CPLUS_INCLUDE_PATH=/usr/include/python3.12 python setup.py build_ext --inplace 2>&1 | tail -1
cp build/lib.linux-x86_64-cpython-312/gomoku_cpp*.so .venv/lib/python3.12/site-packages/

CKPT_DIR="checkpoints/run500_fix_negamax_$(date +%m%d_%H%M)_$RANDOM"
mkdir -p "$CKPT_DIR"

python -u scripts/train_replay.py \
    --G 2048 --M 8 --S 64 --n_steps 500 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir data/init_pool \
    --from_scratch \
    --lr 1e-4 \
    --single_loss \
    --pool_mult 2 \
    --train_fraction 0.625 \
    --c_puct 1.0 \
    --skip_elo \
    > slurm_logs/train_fix_negamax_$$.log 2>&1 &
TRAIN_PID=$!

python -u scripts/elo_monitor_continuous.py \
    --ckpt_dir "$CKPT_DIR" \
    --G 256 --M 4 --S 16 \
    --max_gap 5 --interval 30 \
    --no-value \
    > slurm_logs/monitor_fix_negamax_$$.log 2>&1 &
MONITOR_PID=$!

echo "Train PID: $TRAIN_PID  Monitor PID: $MONITOR_PID"
echo "CKPT: $CKPT_DIR"

wait $TRAIN_PID
RC=$?
echo "Training finished (exit=$RC). Signalling monitor to finish..."
touch "$CKPT_DIR/.done"
echo "Waiting up to 30min for monitor to evaluate remaining pairs..."
for i in $(seq 1 30); do
    sleep 60
    if ! kill -0 $MONITOR_PID 2>/dev/null; then echo "Monitor finished."; break; fi
done
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true
echo "=== All done ==="
