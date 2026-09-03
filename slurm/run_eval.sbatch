#!/bin/bash
#SBATCH --job-name=eval-1b
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
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

# Args:
#   $1 = comma-separated group names to filter (e.g., "1b-scale")
#   $2 = optional extra flags (e.g., "--force")
# NB: avoid bash builtin var name GROUPS — that's an array of the user's GIDs.
GROUP_FILTER="${1:-1b-scale}"
EXTRA="${2:-}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR/experiment"

echo "== Eval on $(hostname) =="
echo "GROUP_FILTER=$GROUP_FILTER  EXTRA=$EXTRA"
nvidia-smi -L || true

# STORAGE_ROOT for the eval script: where checkpoints actually live on ORC
export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
export PYTHONPATH="$TRAIN_DIR:${PYTHONPATH:-}"

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"
which python3; python3 -V

# Single-GPU eval. Uses --scan-dir so wandb is not needed (works for offline runs too).
python3 evaluate_checkpoints_fineweb_dedup.py \
    --scan-dir \
    --groups "$GROUP_FILTER" \
    --loggy-checkpoints \
    --rank 0 --world-size 1 \
    $EXTRA

echo "Eval finished with rc=$?"
