#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=gomoku5
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail
PROJ_DIR="/home/scc/pb24511935/gomoku"
cd "$PROJ_DIR"

echo "=== Gomoku 5-Step Training + ELO Analysis ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source .venv/bin/activate

echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo
echo "Running train_5steps_fixed.py..."
python -u scripts/train_5steps_fixed.py

echo
echo "=== Done ==="
echo "Date: $(date)"
echo "Output in: output/"
echo "Checkpoints in: checkpoints/train_loop/"
ls -la output/analysis_after.png 2>/dev/null || echo "Plot not found, check logs"
