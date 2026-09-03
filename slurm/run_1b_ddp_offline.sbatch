#!/bin/bash
#SBATCH --job-name=1b-8comp-ddp
#SBATCH --output=logs/1b-ddp-%j.out
#SBATCH --error=logs/1b-ddp-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=1-00:00:00

set -euo pipefail
mkdir -p logs

CONFIG_FILE="${1:-config/1b-8comp-bpe16384.toml}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR"

echo "== 1B DDP Training on $(hostname) =="
echo "CONFIG_FILE=$CONFIG_FILE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L || true

export WANDB_MODE=offline
export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"

source .supercomputer-venv/bin/activate
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"
which python3; python3 -V

echo "Launching torchrun --nproc_per_node=8 train.py --job.config-file $CONFIG_FILE"
torchrun --standalone --nproc_per_node=8 \
    train.py --job.config-file "$CONFIG_FILE"

echo "Training finished with rc=$?"
