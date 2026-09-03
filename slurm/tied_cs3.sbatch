#!/bin/bash
#SBATCH --job-name=tied-b200
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/tied/tied-%j.out
#SBATCH --error=logs/tied/tied-%j.out

set -uo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
export TC_STORAGE_ROOT=/nobackup/autodelete/grp/grp_pccl/vin/tc-code5
CONFIG="$TC_STORAGE_ROOT/queue/running/g6__8-512-n8-tr01-tied.toml"

# Fallbacks in case the pool moved the config between now and when this fires.
if [ ! -f "$CONFIG" ]; then
    for cand in \
        "$TC_STORAGE_ROOT"/queue/done/8-512-n8-tr01-tied.toml \
        "$TC_STORAGE_ROOT"/queue/failed/8-512-n8-tr01-tied.toml \
        "$TC_STORAGE_ROOT"/queue/pending/8-512-n8-tr01-tied.toml \
        "$TRAIN_DIR"/config/8-512-n8-tr01-tied.toml
    do
        [ -f "$cand" ] && CONFIG="$cand" && break
    done
fi
[ -f "$CONFIG" ] || { echo "no tied config found"; exit 1; }

export OUT_DIR="$TC_STORAGE_ROOT/out/8-512-n8-tr01-tied"
export RUN_ID=8-512-n8-tr01-tied
export WANDB_MODE=offline
export GIT_DIR="$TRAIN_DIR/.git"
export GIT_WORK_TREE="$TRAIN_DIR"
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"

mkdir -p "$TRAIN_DIR"/logs/tied

echo "[tied-b200] job=$SLURM_JOB_ID host=$(hostname) config=$CONFIG"
nvidia-smi -L

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"

# cd into TC_STORAGE_ROOT so relative bin paths in configs resolve.
# The updated train.py acquires flock on $OUT_DIR/.train.lock; if the old
# cs-2-2 process is still running, this exits rc=2 immediately — a safety net.
cd "$TC_STORAGE_ROOT"
python "$TRAIN_DIR/train.py" --job.config-file "$CONFIG"
