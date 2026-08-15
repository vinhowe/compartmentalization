#!/bin/bash
# One 30B 1B-scale run on all 8 B200s. Submitted once per LR-sweep arm.
#
#   CFG=config/redesign-1b-lr4e-4-c8.toml sbatch --export=ALL slurm/lrsweep.sbatch
#
# PARTITION cs3 IS DELIBERATE, not just a preference for the fastest hardware.
# cs3 contains exactly one node, cs-3-1, and every known-bad node lives in a
# different partition (cs-1-1 in `cs`, dw-2-4 in `dw`, m13l-2-1 in `m13l`). So
# targeting cs3 makes the shredder nodes structurally unreachable -- which
# matters because `--exclude` is a no-op for nodes outside the job's partition,
# and a job landing on one of them fails in seconds while looking healthy.
#
#SBATCH --job-name=lrsweep
#SBATCH --partition=cs3
# account+qos are REQUIRED: without them sbatch rejects the job with
# "Cannot access partition cs3 due to QOS gpu not being allowed". Copied from
# pool.sbatch, which is the known-good submission for this partition.
#SBATCH --account=dw87
#SBATCH --qos=cs
# Redundant given cs3 holds only cs-3-1, but harmless and it keeps the exclusion
# in the file where a future partition change would otherwise silently drop it.
#SBATCH --exclude=cs-1-1,cs-1-2,m13l-2-1,dw-2-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
# 24h is a HARD CEILING: QOS `cs` caps at 1440 minutes, regardless of the cs3
# partition advertising a 7-day MaxTime. A fresh 0->100B run at c=8 measures
# ~24.4h, i.e. it CANNOT finish in one job -- submit it as a chained pair
# (second job --dependency=afterany, resuming from _rolling). One such run was
# killed at 93% before this was understood.
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nobackup/autodelete/grp/grp_pccl/vin/logs/lrsweep-%j.out

set -uo pipefail

# A full core dump per crashed rank once filled /grphome and made every
# subsequent job die at 0:53 with no log. Never remove this.
ulimit -c 0

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
cd "$TRAIN_DIR"
# shellcheck disable=SC1091
source .supercomputer-venv/bin/activate
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"
export WANDB_MODE=offline

# MANDATORY on cs-3-1. NVLink SHARP is advertised and hangs the first NCCL
# collective: init_process_group succeeds, world_size prints, then the job sits
# until its wallclock expires with no error. Verified 2026-08-07.
export NCCL_NVLS_ENABLE=0

: "${CFG:?set CFG to a config .toml path}"
# OUT_NAME lets an extension config (e.g. ...-100b.toml) resume the ORIGINAL
# run directory. Without it the new config name would mint a fresh OUT_DIR and
# the run would silently start from scratch instead of continuing.
NAME=${OUT_NAME:-$(basename "$CFG" .toml)}
# Outputs to /nobackup: /grphome has 571G free and one 30B run's checkpoints
# would eat a meaningful fraction of that.
export OUT_DIR=/nobackup/autodelete/grp/grp_pccl/vin/out/redesign-1b-lrsweep/$NAME
export RUN_ID=$NAME
mkdir -p "$OUT_DIR"

echo "[lrsweep] job=$SLURM_JOB_ID host=$(hostname) cfg=$CFG out=$OUT_DIR"
nvidia-smi -L | head -8 || true

# rdzv-endpoint port 0 lets the OS pick: a stale run holding a fixed port made
# every subsequent DDP launch fail with EADDRINUSE.
srun --ntasks=1 torchrun --nproc_per_node=8 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    train.py --job.config-file "$CFG"
rc=$?
echo "[lrsweep] finished rc=$rc cfg=$CFG"
exit $rc
