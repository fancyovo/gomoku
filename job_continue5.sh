#!/bin/bash
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=gomoku5b
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail
PROJ_DIR="/home/scc/pb24511935/gomoku"
cd "$PROJ_DIR"

echo "=== Gomoku Continue 5 Steps + Full ELO ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source .venv/bin/activate

echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo
echo "Running continue_5steps.py..."
python -u scripts/continue_5steps.py

echo
echo "=== Done ==="
echo "Date: $(date)"
echo "Plot: output/analysis_10steps.png"
ls -la output/analysis_10steps.png 2>/dev/null || echo "Plot not found, check logs"
