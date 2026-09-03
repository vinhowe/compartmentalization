#!/bin/bash
# Periodic queue reconciler. CPU-only, no GPU, negligible cost.
#
# WHY THIS EXISTS. reconcile_queue.sh already runs at pool startup, but that is
# not enough: when a pool hits its wallclock, its configs stop, and the *next*
# pool typically starts within a few minutes -- at which point those configs
# still look "fresh" (checkpoint mtime under FRESH_MIN) so the startup
# reconcile correctly skips them. Nothing then revisits, and they sit dead.
#
# That is not hypothetical. On 2026-07-27 the dw pools expired, three backfill
# pools started minutes later, and 37 of 62 reseed configs sat idle for 7.7
# hours before anyone noticed -- roughly 280 GPU-hours lost. A startup-only
# reconcile cannot catch that by construction; only a periodic one can.
#
# Runs the reconciler every INTERVAL seconds and logs anything it requeues.
#
#SBATCH --job-name=tc-watchdog
#SBATCH --account=dw87
#SBATCH --qos=standby
#SBATCH --partition=m9g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/grphome/grp_pccl/vin/dev/translation-compression/logs/watchdog-%j.out

set -uo pipefail
TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
CFG_DIR=${CFG_DIR:?config dir}
QUEUE=${QUEUE:?queue dir}
OUT_ROOT=${OUT_ROOT:?out dir}
INTERVAL=${INTERVAL:-600}

echo "[watchdog] start $(date) interval=${INTERVAL}s"
echo "[watchdog] cfg=$CFG_DIR queue=$QUEUE out=$OUT_ROOT"

while true; do
    out=$("$TRAIN_DIR/slurm/reconcile_queue.sh" "$CFG_DIR" "$QUEUE" "$OUT_ROOT" --apply 2>&1)
    # only speak up when it actually did something, so the log stays readable
    if echo "$out" | grep -qE "requeue: |done="; then
        n=$(echo "$out" | grep -c "requeue: ")
        if [ "$n" -gt 0 ]; then
            echo "[watchdog] $(date '+%F %T') requeued $n stalled config(s):"
            echo "$out" | grep "requeue: " | sed 's/^/    /'
        fi
    fi
    sleep "$INTERVAL"
done
