#!/bin/bash
# Standalone single-GPU train.py runner. Parameterized by CONFIG env var.
#
# Usage:
#   sbatch --partition=cs3 --gpus=1 --mem=64G \
#     --export=ALL,CONFIG=/path/to/config.toml slurm/single_train.sbatch
#
# NAME is derived from CONFIG basename; OUT_DIR is under $TC_STORAGE_ROOT/out/.
# train.py acquires flock on $OUT_DIR/.train.lock — if held by another Python,
# this exits rc=2 without corrupting state.

#SBATCH --job-name=single-train
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/single/%j.out
#SBATCH --error=logs/single/%j.out

set -uo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
: "${TC_STORAGE_ROOT:=/nobackup/autodelete/grp/grp_pccl/vin/tc-code5}"
: "${CONFIG:?CONFIG env var not set; pass via --export=ALL,CONFIG=/path/to/config.toml}"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG"; exit 1; }

NAME=$(basename "$CONFIG" .toml)
export OUT_DIR="$TC_STORAGE_ROOT/out/$NAME"
export RUN_ID="$NAME"
export WANDB_MODE=offline
export GIT_DIR="$TRAIN_DIR/.git"
export GIT_WORK_TREE="$TRAIN_DIR"
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"
export TC_STORAGE_ROOT

mkdir -p "$TRAIN_DIR/logs/single"

echo "[single-train] job=$SLURM_JOB_ID host=$(hostname) name=$NAME config=$CONFIG"
nvidia-smi -L

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"

# cd into TC_STORAGE_ROOT so relative bin paths in configs resolve.
cd "$TC_STORAGE_ROOT"
python "$TRAIN_DIR/train.py" --job.config-file "$CONFIG"
