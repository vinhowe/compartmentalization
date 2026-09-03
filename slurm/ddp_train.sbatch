#!/bin/bash
# 8-way single-node DDP train.py runner. Parameterized by CONFIG env var.
#
# Usage:
#   sbatch --export=ALL,CONFIG=/path/to/config.toml,TC_STORAGE_ROOT=/nobackup/... \
#     slurm/ddp_train.sbatch
#
# Requires that batch_size in the config is divisible by SLURM_GPUS_ON_NODE.
# NAME derived from CONFIG basename; OUT_DIR under $TC_STORAGE_ROOT/out/$NAME.

#SBATCH --job-name=ddp-train
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --partition=cs3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/ddp/%j.out
#SBATCH --error=logs/ddp/%j.out

set -uo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
: "${TC_STORAGE_ROOT:?TC_STORAGE_ROOT env var not set}"
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

mkdir -p "$TRAIN_DIR/logs/ddp"
mkdir -p "$OUT_DIR"

echo "[ddp-train] job=$SLURM_JOB_ID host=$(hostname) name=$NAME config=$CONFIG gpus=$SLURM_GPUS_ON_NODE"
nvidia-smi -L

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"

# cd into TC_STORAGE_ROOT so relative bin paths in configs resolve.
cd "$TC_STORAGE_ROOT"

# Trap SIGTERM (slurm preempt) so torchrun children get it too — train.py will
# see it and save a rolling checkpoint before exiting.
cleanup() {
    echo "[ddp-train] SIGTERM received at $(date '+%F %T')"
    trap - SIGTERM SIGINT EXIT
    pkill -TERM -P $$ 2>/dev/null || true
    wait
    exit 0
}
trap cleanup SIGTERM SIGINT

# SLURM_GPUS_ON_NODE is the NODE's gpu count, not this job's allocation. On a
# shared node that spawns ranks for GPUs we do not own -> "CUDA-capable
# device(s) is/are busy or unavailable". CUDA_VISIBLE_DEVICES is exactly our
# allocation, so count that instead.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NGPU=${SLURM_GPUS_ON_NODE:-1}
fi
echo "[ddp-train] NGPU=$NGPU (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
torchrun --standalone --nproc_per_node="$NGPU" \
    "$TRAIN_DIR/train.py" --job.config-file "$CONFIG" \
    ${EXTRA_TRAIN_ARGS:-} &
wait $!
rc=$?
echo "[ddp-train] $(date '+%F %T') done rc=$rc name=$NAME"
exit $rc
