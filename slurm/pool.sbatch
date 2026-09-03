#!/bin/bash
# Long-running worker pool on cs3 (8x B200). Each GPU polls a shared queue
# directory and runs one config at a time. Drop configs into $QUEUE/pending/
# to add work; results go to $OUT_ROOT/<config-basename>/.
#
# Layout (both under $QUEUE):
#   pending/    configs waiting to run
#   running/    configs currently claimed by a GPU (named g<N>__<orig>.toml)
#   done/       configs that finished with rc=0
#   failed/     configs that finished with rc!=0
#
# Resumability:
#   - Pool startup sweeps running/ back to pending/ (in case a prior sbatch was
#     killed mid-run). Training resumes from rolling checkpoint on relaunch.
#   - Each config gets deterministic OUT_DIR=$OUT_ROOT/<basename> and
#     RUN_ID=<basename>, so re-runs of the same config land in the same dir
#     and pick up from the latest step-N/ checkpoint.
#
# Env overrides (optional):
#   QUEUE       shared queue dir (default: /nobackup/.../vin/pool/queue)
#   OUT_ROOT    where run outputs live (default: /nobackup/.../vin/pool/out)
#   LOG_DIR     per-worker logs (default: $TRAIN_DIR/logs/pool)
#   POLL_SLEEP  seconds to wait when queue empty (default: 20)
#
# Submit with:
#   sbatch slurm/pool.sbatch
# or with overrides:
#   QUEUE=/some/other/queue sbatch --export=ALL slurm/pool.sbatch
#SBATCH --job-name=tc-pool
#SBATCH --output=/grphome/grp_pccl/vin/dev/translation-compression/logs/pool/pool-%j.out
#SBATCH --error=/grphome/grp_pccl/vin/dev/translation-compression/logs/pool/pool-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
# Nodes with uncorrectable ECC / flaky GPUs. These do not merely fail a job:
# before the checkpoint-loading fixes, a CUDA failure here made the worker fall
# back to the newest NAMED checkpoint, silently rewinding the run by up to 200k
# steps AND resetting Adam (named checkpoints carry no optimizer state). Single
# reseed runs were thrown back to step-500000 four times over this way.
#
# Choosing a partition cannot avoid them: every node belongs to ~35 partitions
# at once, so cs-1-1 is reachable from cs3, dwmatrix, m13h, eng and the rest
# alike. --exclude is a NODE-level constraint and is the only thing that works.
#SBATCH --exclude=cs-1-1,cs-1-2,m13l-2-1,dw-2-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60

set -uo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
QUEUE=${QUEUE:-/nobackup/autodelete/grp/grp_pccl/vin/pool/queue}
OUT_ROOT=${OUT_ROOT:-/nobackup/autodelete/grp/grp_pccl/vin/pool/out}
LOG_DIR=${LOG_DIR:-$TRAIN_DIR/logs/pool}
POLL_SLEEP=${POLL_SLEEP:-20}

mkdir -p "$QUEUE"/{pending,running,done,failed}
mkdir -p "$OUT_ROOT" "$LOG_DIR"

cd "$TRAIN_DIR"
# shellcheck disable=SC1091
source .supercomputer-venv/bin/activate
export WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"

# MANDATORY on cs-3-1 (B200). NVLink SHARP is advertised on that node and hangs
# the first NCCL collective -- init_process_group succeeds, world_size prints,
# and then the job sits forever with no error until its wallclock expires.
# Verified 2026-08-07: a bare 2-GPU all_reduce hangs without this; with it,
# 512MB all_reduce runs at 429 GB/s and 8-GPU DDP scales at 92%.
#
# Harmless on single-GPU workers (no collectives) and on non-Blackwell nodes,
# so it is set unconditionally rather than guarded by hostname.
export NCCL_NVLS_ENABLE=0

echo "[pool] job=$SLURM_JOB_ID host=$(hostname) queue=$QUEUE out_root=$OUT_ROOT"
nvidia-smi -L | head -8 || true

# Sweep any leftover running/ configs back to pending/. Runs on startup and
# again on graceful shutdown. The 'g<N>__' prefix is stripped so the original
# basename is preserved (which also drives OUT_DIR/RUN_ID => resume).
# Match ANY .toml in running/, not just g*__*.toml, and strip to after the LAST
# "__". Two failures made this necessary:
#   * an older pool tagged claims "<jobid>_g<N>__<orig>.toml", which begins with a
#     digit, so the g*__ glob never matched it — five configs sat in running/
#     owned by a dead job, invisible to every subsequent sweep.
#   * "${base#g*__}" strips only a leading g-prefix, so requeuing a file that
#     already carried one and re-claiming it produced
#     "g2__<jobid>_g1__<orig>.toml". The basename drives OUT_DIR, so that starts
#     a BRAND NEW run in a garbage directory instead of resuming.
# Config basenames never contain "__" (asserted below), so "${base##*__}" is
# exact and idempotent no matter how many prefixes have accreted.
requeue_running() {
    shopt -s nullglob
    for f in "$QUEUE"/running/*.toml; do
        base=$(basename "$f")
        orig="${base##*__}"
        if mv "$f" "$QUEUE/pending/$orig" 2>/dev/null; then
            echo "[pool] requeued: $base -> $orig"
        fi
    done
    shopt -u nullglob
}
requeue_running

# On SIGTERM (slurm timeout, --requeue, manual scancel), requeue in-flight and
# exit cleanly. cs3 has GraceTime=0 so the --signal=B:SIGTERM@60 above is what
# gives us the 60 s window before SIGKILL. If we miss the window, the next pool
# submission will requeue on startup — training-side rolling checkpoints keep
# progress either way.
cleanup() {
    echo "[pool] SIGTERM received at $(date '+%F %T'); requeuing in-flight"
    trap - SIGTERM SIGINT EXIT
    # kill worker loops
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 3
    requeue_running
    exit 0
}
trap cleanup SIGTERM SIGINT

worker() {
    local GPU=$1
    local WLOG="$LOG_DIR/g${GPU}-${SLURM_JOB_ID}.log"
    echo "[g$GPU] $(date '+%F %T') starting" | tee -a "$WLOG"
    while true; do
        # Atomic claim: iterate pending/, try to mv into running/ with our GPU tag.
        # Only one worker will win any given mv (rename is atomic on same fs).
        local CFG=""
        shopt -s nullglob
        for cand in "$QUEUE"/pending/*.toml; do
            local claimed="$QUEUE/running/g${GPU}__$(basename "$cand")"
            if mv "$cand" "$claimed" 2>/dev/null; then
                CFG="$claimed"
                break
            fi
        done
        shopt -u nullglob

        if [ -z "$CFG" ]; then
            sleep "$POLL_SLEEP"
            continue
        fi

        # Strip prefix to get original name; that drives OUT_DIR/RUN_ID so a
        # re-run of the same config resumes from the same checkpoint dir.
        local BASE
        BASE=$(basename "$CFG" .toml)
        local NAME="${BASE#g*__}"
        local RUN_OUT="$OUT_ROOT/$NAME"
        mkdir -p "$RUN_OUT"

        echo "[g$GPU] $(date '+%F %T') starting $NAME (out=$RUN_OUT)" | tee -a "$WLOG"
        CUDA_VISIBLE_DEVICES=$GPU \
            OUT_DIR="$RUN_OUT" \
            RUN_ID="$NAME" \
            CONFIG_PATH="$CFG" \
            python train.py --job.config-file "$CFG" >> "$WLOG" 2>&1
        local rc=$?

        local ORIG_NAME="${NAME}.toml"
        if [ $rc -eq 0 ]; then
            mv "$CFG" "$QUEUE/done/$ORIG_NAME"
            echo "[g$GPU] $(date '+%F %T') done rc=0 $NAME" | tee -a "$WLOG"
        else
            mv "$CFG" "$QUEUE/failed/$ORIG_NAME"
            echo "[g$GPU] $(date '+%F %T') FAILED rc=$rc $NAME" | tee -a "$WLOG"
        fi
    done
}

for GPU in 0 1 2 3 4 5 6 7; do
    worker "$GPU" &
done

# `wait` returns when all children exit OR when interrupted by our trap.
wait
