#!/usr/bin/env bash
# Convenience wrapper for slurm/pool.sbatch. Ensures the queue and log dirs
# exist, then submits. Override QUEUE / OUT_ROOT via env if you want an
# experiment-specific queue rather than the default shared pool.
#
# Usage:
#   ./slurm/pool_submit.sh                                    # default pool
#   QUEUE=/nobackup/.../vin/tc-code5/queue \
#     OUT_ROOT=/nobackup/.../vin/tc-code5/out \
#     ./slurm/pool_submit.sh                                  # experiment-scoped
#
# Add configs by dropping .toml files into $QUEUE/pending/. Each config lives
# in exactly one of pending/ | running/ | done/ | failed/ at a time.

set -euo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
QUEUE=${QUEUE:-/nobackup/autodelete/grp/grp_pccl/vin/pool/queue}
OUT_ROOT=${OUT_ROOT:-/nobackup/autodelete/grp/grp_pccl/vin/pool/out}

mkdir -p "$QUEUE"/{pending,running,done,failed}
mkdir -p "$OUT_ROOT"
mkdir -p "$TRAIN_DIR/logs/pool"

cd "$TRAIN_DIR"
export QUEUE OUT_ROOT
sbatch --export=ALL slurm/pool.sbatch
