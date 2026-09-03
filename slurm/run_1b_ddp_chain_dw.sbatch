#!/bin/bash
#SBATCH --job-name=1b-chain-dw
#SBATCH --output=logs/1b-chain-%j.out
#SBATCH --error=logs/1b-chain-%j.err
#SBATCH --partition=dw
#SBATCH --account=dw87
#SBATCH --qos=dw87long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --exclude=dw-2-4
#SBATCH --time=7-00:00:00

set -euo pipefail
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

CONFIG_FILE="${1:?config file required}"
RESUME_DIR="${2:?out_dir required}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR"

echo "== 1B chain (dw, 7-day) on $(hostname) job=$SLURM_JOB_ID =="
echo "CONFIG=$CONFIG_FILE  OUT_DIR=$RESUME_DIR"

# Early exit if model already at max_iters.
mkdir -p "$RESUME_DIR/checkpoints"
LATEST=$(ls "$RESUME_DIR/checkpoints/" 2>/dev/null | { grep "^step-" || true; } | sed 's/step-0*//' | sort -n | tail -1)
MAX_ITERS=$(grep -oP 'max_iters\s*=\s*\K[0-9]+' "$CONFIG_FILE" | head -1)
if [ -n "$LATEST" ] && [ -n "$MAX_ITERS" ] && [ "$LATEST" -ge "$MAX_ITERS" ]; then
    echo "Already at max_iters ($LATEST >= $MAX_ITERS); skipping"
    exit 0
fi

# Coarse lock with stale detection (7 days).
LOCK="$RESUME_DIR/.training_lock"
if [ -e "$LOCK" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
    if [ "$AGE" -lt 604800 ]; then
        echo "FATAL: lock $LOCK held (age ${AGE}s)"
        cat "$LOCK"
        exit 1
    fi
    echo "Lock stale (age ${AGE}s); overriding"
fi
echo "$(hostname) pid=$$ slurm=$SLURM_JOB_ID at $(date)" > "$LOCK"
trap "rm -f $LOCK" EXIT

nvidia-smi -L || true

export WANDB_MODE=offline
export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
export OUT_DIR="$RESUME_DIR"

source .supercomputer-venv/bin/activate
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"
which python3; python3 -V

torchrun --standalone --nproc_per_node=8 \
    train.py --job.config-file "$CONFIG_FILE"

echo "Training finished with rc=$?"
