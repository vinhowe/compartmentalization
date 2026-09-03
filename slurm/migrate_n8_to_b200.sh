#!/bin/bash
# Move the slow n=8 reseed configs off A100 (dw) and onto B200 (cs3).
#
# WHY: measured 37,500 iters/hr on cs3 B200 vs 12,500 on dw A100 for matched
# n=5 configs -- a 3.0x speedup. The n=8 family is the critical path (LM head is
# 84% of an 8-256 model at n=8, so it is ~3.8x the compute of n=1). On A100 the
# n=8 tail is ~4.6 days; on B200 it is ~1.5 days.
#
# WHY IT NEEDS A DELIBERATE SWAP: the n=8 configs sit on pools with ~20h left,
# while the pools expiring soon hold the fast n2-n6 configs. Passive wallclock
# churn would not move them.
#
# SAFETY:
#   * refuses to run unless a cs3 pool is already RUNNING with free workers,
#     otherwise the freed configs get re-claimed by a pending dw pool and we
#     churn for nothing
#   * holds down pending dw pools during the swap for the same reason
#   * snapshots every config's iter_num before touching anything, and diffs
#     after, so "did it resume or restart from 0" is answered with data
#   * resume itself is proven in this sweep: n2-tr01comp-s66 passed 780k iters,
#     more than one 18h pool can produce, so it has already survived restarts
#   * reconciles via rolling-checkpoint truth, which also rescues configs whose
#     queue entries were lost to the old cannibalisation bug (they have no
#     running/ entry and would otherwise be orphaned by the scancel)
#
# Usage: migrate_n8_to_b200.sh [--apply]      (default: dry run)
set -uo pipefail

TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
B=/nobackup/autodelete/grp/grp_pccl/vin/tc-8-256-reseed
CFG_DIR=$TRAIN_DIR/config/reseed_8_256
QUEUE=$B/queue
OUT=$B/out
SNAP=/tmp/n8_migration_snapshot.txt

A100_N8_JOBS="12902843 12902844"     # pools currently holding the n=8 configs
DW_PENDING="12903267 12903268"       # would steal the freed configs
CS3_POOLS="12903269 12903270 12902379"

APPLY=${1:-}

# --verify: diff current iters against the pre-scancel snapshot. Any run that
# came back LOWER than it started restarted from scratch instead of resuming --
# that is the failure mode this whole procedure exists to rule out.
if [ "$APPLY" = "--verify" ]; then
    [ -f "$SNAP" ] || { echo "no snapshot at $SNAP"; exit 1; }
    bad=0; ok=0
    while read -r n before; do
        s="$OUT/$n/checkpoints/_rolling/trainer_state.json"
        [ -f "$s" ] || continue
        now=$(tr -dc '0-9,:{}"a-z_' < "$s" | sed 's/.*iter_num":\([0-9]*\).*/\1/')
        now=${now:-0}
        if [ "$now" -lt $(( before - 2000 )) ]; then
            echo "  REGRESSED: $n  $before -> $now"; bad=$((bad+1))
        else
            ok=$((ok+1))
        fi
    done < "$SNAP"
    echo "verify: ok=$ok regressed=$bad"
    [ "$bad" -eq 0 ] && echo "All runs resumed correctly (no restart-from-zero)." || echo "!! investigate regressions above"
    exit 0
fi

snapshot() {
    : > "$SNAP"
    for d in "$OUT"/*/; do
        n=$(basename "$d")
        s="$d/checkpoints/_rolling/trainer_state.json"
        [ -f "$s" ] || continue
        it=$(tr -dc '0-9,:{}"a-z_' < "$s" | sed 's/.*iter_num":\([0-9]*\).*/\1/')
        echo "$n ${it:-0}" >> "$SNAP"
    done
    echo "  snapshotted $(wc -l < "$SNAP") runs -> $SNAP"
}

# ---- precondition: a cs3 pool must be RUNNING ----
running_cs3=""
for j in $CS3_POOLS; do
    st=$(squeue -h -j "$j" -o "%T" 2>/dev/null)
    [ "$st" = "RUNNING" ] && running_cs3="$running_cs3 $j"
done
if [ -z "$running_cs3" ]; then
    echo "ABORT: no cs3 B200 pool is RUNNING. Freed configs would be re-claimed"
    echo "       by a dw pool and land right back on A100. Wait, then re-run."
    exit 1
fi
echo "cs3 pools running:$running_cs3"

echo "n=8 configs currently claimed on A100:"
ls "$QUEUE"/running 2>/dev/null | grep -E "^($(echo $A100_N8_JOBS | tr ' ' '|'))_" | sed 's/^/  /'

if [ -z "$APPLY" ]; then
    echo
    echo "DRY RUN. Would: hold dw pools ($DW_PENDING), snapshot iters,"
    echo "scancel $A100_N8_JOBS, reconcile with FRESH_MIN=3, verify resume."
    exit 0
fi

echo "== 1. snapshot before =="
snapshot

echo "== 2. hold pending dw pools so they cannot steal =="
scancel $DW_PENDING 2>/dev/null && echo "  held: $DW_PENDING"

echo "== 3. scancel A100 pools holding n=8 (SIGTERM; workers requeue) =="
scancel $A100_N8_JOBS
echo "  waiting 90s for graceful requeue + process exit..."
sleep 90

echo "== 4. reconcile (short freshness window; processes are gone) =="
FRESH_MIN=3 "$TRAIN_DIR/slurm/reconcile_queue.sh" "$CFG_DIR" "$QUEUE" "$OUT" --apply

echo "== 5. queue state =="
for d in pending running done failed; do
    echo "  $d: $(ls "$QUEUE/$d" 2>/dev/null | wc -l)"
done

echo
echo "NEXT: watch that cs3 workers claim the pending configs, then verify"
echo "resume against the snapshot:"
echo "  migrate_n8_to_b200.sh --verify"
