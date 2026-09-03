#!/bin/bash
# Self-healing queue reconciler for the pool.sbatch file-queue.
#
# The queue dirs (pending/running/done/failed) are advisory bookkeeping, and
# they drift from reality: a worker can die without requeueing, and the old
# requeue_all_stale bug moved entries out from under live workers. We ended up
# with 12 configs sitting in failed/ while their training processes were very
# much alive -- which would have silently orphaned them at the next wallclock
# churn, because requeue only ever looks at running/.
#
# Ground truth is not the queue. It is:
#   (a) the rolling checkpoint's iter_num  -> how far the run actually got
#   (b) that file's mtime                  -> whether a process is touching it
#
# For each config in the repo set:
#   * at/near max_iters      -> move to done/, never re-run
#   * checkpoint is fresh    -> a worker owns it, leave it completely alone
#   * otherwise              -> it is genuinely idle, ensure exactly one
#                               pending/ entry so a free worker resumes it
#
# Safe to run repeatedly, including while the pools are running.
#
# Usage: reconcile_queue.sh <config_dir> <queue_dir> <out_dir> [--apply]
set -u

CFG_DIR=${1:?config dir}
QUEUE=${2:?queue dir}
OUT=${3:?out dir}
APPLY=${4:-}

# checkpoint touched this recently => assume a worker owns it. Default 15 min
# is right for steady state (rolling checkpoints land every 1000 iters, and the
# slowest configs take a few minutes per 1000). Lower it right after a
# deliberate scancel, when you know the processes are gone and want the freed
# configs requeued promptly instead of waiting out the window.
FRESH_MIN=${FRESH_MIN:-15}
DONE_FRAC=${DONE_FRAC:-0.999}   # >= this fraction of max_iters counts as finished

now=$(date +%s)
n_done=0; n_live=0; n_requeue=0; n_nostart=0

for cfg in "$CFG_DIR"/*.toml; do
    [ -e "$cfg" ] || continue
    name=$(basename "$cfg" .toml)
    state="$OUT/$name/checkpoints/_rolling/trainer_state.json"

    max_iters=$(grep -E "^max_iters" "$cfg" | head -1 | tr -dc '0-9')
    max_iters=${max_iters:-1000000}

    if [ -f "$state" ]; then
        iter=$(tr -dc '0-9,:{}"a-z_' < "$state" | sed 's/.*iter_num":\([0-9]*\).*/\1/')
        iter=${iter:-0}
        mtime=$(stat -c %Y "$state" 2>/dev/null || echo 0)
        age_min=$(( (now - mtime) / 60 ))
    else
        iter=0; age_min=999999
    fi

    thresh=$(awk -v m="$max_iters" -v f="$DONE_FRAC" 'BEGIN{printf "%d", m*f}')

    if [ "$iter" -ge "$thresh" ]; then
        n_done=$((n_done+1))
        if [ -z "$APPLY" ]; then continue; fi
        rm -f "$QUEUE/pending/$name.toml" "$QUEUE/failed/$name.toml"
        rm -f "$QUEUE"/running/*__"$name".toml
        cp "$cfg" "$QUEUE/done/$name.toml" 2>/dev/null
        continue
    fi

    if [ "$age_min" -lt "$FRESH_MIN" ]; then
        # A live worker owns this. Do not put it in pending -- a second worker
        # would collide on the run lock and fail the run.
        n_live=$((n_live+1))
        if [ -n "$APPLY" ]; then rm -f "$QUEUE/pending/$name.toml"; fi
        continue
    fi

    # Genuinely idle and unfinished -> make sure it is queued exactly once.
    if [ "$iter" -eq 0 ]; then n_nostart=$((n_nostart+1)); else n_requeue=$((n_requeue+1)); fi
    if [ -n "$APPLY" ]; then
        rm -f "$QUEUE/failed/$name.toml"
        rm -f "$QUEUE"/running/*__"$name".toml
        cp "$cfg" "$QUEUE/pending/$name.toml"
    fi
    echo "  requeue: $name (iter=$iter/$max_iters age=${age_min}m)"
done

echo "reconcile: done=$n_done live=$n_live requeue=$n_requeue never_started=$n_nostart $([ -z "$APPLY" ] && echo '(DRY RUN -- pass --apply)')"
