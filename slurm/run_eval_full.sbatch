#!/bin/bash
#SBATCH --job-name=eval-full
#SBATCH --output=logs/eval-full-%j.out
#SBATCH --error=logs/eval-full-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail
mkdir -p logs

GROUP_FILTER="${1:-1b-scale}"
EXTRA="${2:-}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR/experiment"

echo "== Eval (full, no loggy filter) on $(hostname) =="
nvidia-smi -L || true

export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
export PYTHONPATH="$TRAIN_DIR:${PYTHONPATH:-}"

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"
which python3; python3 -V

# No --loggy-checkpoints: evaluate ALL saved checkpoints for the group
python3 evaluate_checkpoints_fineweb_dedup.py \
    --scan-dir \
    --groups "$GROUP_FILTER" \
    --rank 0 --world-size 1 \
    $EXTRA

echo "Eval finished with rc=$?"
