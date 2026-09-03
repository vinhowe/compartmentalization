#!/bin/bash
#SBATCH --job-name=1b-resume-3d
#SBATCH --output=logs/1b-resume-3d-%j.out
#SBATCH --error=logs/1b-resume-3d-%j.err
#SBATCH --partition=dw
#SBATCH --account=dw87
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=3-00:00:00

set -euo pipefail
ulimit -c 0
mkdir -p logs

# Args:
#   $1 = config file
#   $2 = OUT_DIR (absolute path to original run dir for resume)
CONFIG_FILE="${1:?config file required}"
RESUME_DIR="${2:?out_dir required for resume}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR"

echo "== 1B DDP Resume on $(hostname) =="
echo "CONFIG=$CONFIG_FILE  OUT_DIR=$RESUME_DIR"
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
