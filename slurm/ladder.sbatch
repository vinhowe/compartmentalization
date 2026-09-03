#!/bin/bash
# One ladder rung/arm on 8 A100s (partition dw). Submitted once per arm:
#
#   CFG=config/ladder-v2-final/ladder-v2-final-r5-c8.toml \
#     sbatch --export=ALL,CFG=... slurm/ladder.sbatch
#
# Derived from lrsweep_a100.sbatch. The difference is OUT_ROOT: ladder runs land
# under out/<suite>/ rather than out/redesign-1b-lrsweep/, so the two families
# stay separable in analysis. Mixing them is how a recipe change silently
# contaminates a result.
#
# CHAINING. QOS dw87 caps a job at 24h. Measured 8-GPU wall times at 30B tokens
# (8.9 s/iter at 16x2048 c=8, from the 1B runs) are:
#
#   R5-c1 ~11h   R5-c8 ~20h   R6-c1 ~24h   R6-c8 ~35h
#
# so R6 needs a chained pair. Submit the second with
# --dependency=afterany:<first>; it resumes from _rolling, and if the run
# already finished it exits immediately rather than retraining.
#
#SBATCH --job-name=ladder
#SBATCH --partition=dw
# account+qos are REQUIRED; without them sbatch rejects the job outright.
#SBATCH --account=dw87
#SBATCH --qos=dw87
# dw-2-4 has a GPU-0 ECC fault that silently shreds configs; dw-2-1 and dw-2-2
# are the lab workstations and are down in slurm. --exclude is a no-op for nodes
# outside the job's partition, so this only works because all three are in dw.
#SBATCH --exclude=dw-2-4,dw-2-1,dw-2-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/nobackup/autodelete/grp/grp_pccl/vin/logs/ladder-%j.out

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

# Advertised NVLink SHARP hangs the first NCCL collective on cs-3-1: the job
# sits until wallclock with no error. Harmless on A100, kept so a partition
# change cannot silently reintroduce the hang.
export NCCL_NVLS_ENABLE=0

: "${CFG:?set CFG to a config .toml path}"
NAME=${OUT_NAME:-$(basename "$CFG" .toml)}
# SUITE controls the parent directory; default groups by the config's own dir.
SUITE=${SUITE:-$(basename "$(dirname "$CFG")")}
# Outputs to /nobackup: /grphome is small and one 30B run's checkpoints would
# eat a meaningful fraction of it.
export OUT_DIR=/nobackup/autodelete/grp/grp_pccl/vin/out/$SUITE/$NAME
export RUN_ID=$NAME
mkdir -p "$OUT_DIR"

echo "[ladder] job=$SLURM_JOB_ID host=$(hostname) cfg=$CFG out=$OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

srun --export=ALL torchrun \
    --standalone --nproc_per_node=8 \
    train.py --job.config-file "$CFG"
rc=$?
echo "[ladder] finished rc=$rc cfg=$CFG"
exit $rc
