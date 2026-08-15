#!/bin/bash
# Telegram on LR-sweep completion, failure, or stall.
#
# The six arms run back-to-back on cs-3-1 (~7h each, ~42h total). They are slurm
# jobs on a remote cluster, so nothing local notices if one dies -- and a slurm
# job that fails at minute two looks exactly like one that has not started yet.
#
# Progress comes from scripts staged on ORC (sweep_status.sh), which re-execs
# itself under a login shell because slurm's sacct is a SHELL FUNCTION there,
# absent from any PATH in a non-login shell.
#
# ANSI STRIPPING IS LOAD-BEARING: ORC prints a login banner whose colour reset
# codes prefix the first line of real output, so a naive `grep '^redesign'`
# silently drops one arm -- reporting five jobs where there are six.
set -uo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-/nobackup/autodelete/grp/grp_pccl/vin/profiling/sweep_status.sh}
SSH_CFG=${SSH_CFG:-/root/.ssh/orc_config}
HOST=${HOST:-orc-th443}
POLL=${POLL:-900}                 # 15 min
# checkpoint_steps is LOG-SPACED: late in a run the gap is ~2000 steps, which at
# ~2s/step is ~65 min of legitimate silence. A 45-min threshold fired four false
# alarms on the first sweep. Must exceed the largest real gap, not the typical one.
# 30 min is now meaningful: a live run advances its iteration counter
# every ~20 seconds, so 30 min of no movement is genuinely wedged.
STALL_MIN=${STALL_MIN:-30}
SSH_FAIL_MAX=${SSH_FAIL_MAX:-6}   # ~90 min of unreachable before complaining
STATE=${STATE:-/tmp/watch_lrsweep.state}
NOTIFY=${NOTIFY:-$HOME/.claude/notify.sh}
PROBE=${PROBE:-}                  # tests override this to fake the remote call

fired() { grep -qxF "$1" "$STATE" 2>/dev/null; }
mark()  { echo "$1" >> "$STATE"; }
say()   { echo "[$(date '+%F %T')] $*"; bash "$NOTIFY" "$*"; }
touch "$STATE"

probe() {
    if [ -n "$PROBE" ]; then bash "$PROBE"; return $?; fi
    timeout 120 ssh -F "$SSH_CFG" -o ConnectTimeout=20 "$HOST" "$REMOTE" 2>/dev/null
}

declare -A last_tok last_change last_state
ssh_fails=0

while true; do
    now=$(date +%s)
    # strip ANSI, keep only pipe-delimited data lines
    raw=$(probe | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^(redesign|ERROR)' || true)

    if [ -z "$raw" ]; then
        ssh_fails=$((ssh_fails+1))
        if [ "$ssh_fails" -ge "$SSH_FAIL_MAX" ] && ! fired "unreachable"; then
            say "LR SWEEP — cannot read job status from ORC for ~$((ssh_fails*POLL/60))min. Sweep may be fine; the watcher is blind."
            mark "unreachable"
        fi
        sleep "$POLL"; continue
    fi
    ssh_fails=0

    done_n=0; total=0; running=""; summary=""
    while IFS='|' read -r name state el n mx it; do
        [ "$name" = "ERROR" ] && continue
        total=$((total+1))
        summary="$summary\n  $name  $state  iter ${it}  ${mx}M tok  ($n ckpts, $el)"
        key="$name"
        # Reset the stall clock on ANY state change too, not just token progress:
        # otherwise a job that waited hours in PENDING is judged "stalled since
        # first seen" the moment it starts running, before it can write anything.
        # Progress = the training log's ITERATION counter (advances every ~20s),
        # not the checkpoint count. Checkpoints are log-spaced with gaps up to
        # 9000 steps (~4.9h at c=8), so a checkpoint-based stall test is either
        # blind for hours or cries wolf -- it did both before this change.
        if [ "${last_tok[$key]:-}" != "$it" ] || [ "${last_state[$key]:-}" != "$state" ]; then
            last_tok[$key]=$it; last_state[$key]=$state; last_change[$key]=$now
        fi
        case "$state" in
            COMPLETED) done_n=$((done_n+1)) ;;
            RUNNING)
                running="$name"
                age=$(( (now - ${last_change[$key]:-$now}) / 60 ))
                if [ "$age" -ge "$STALL_MIN" ] && ! fired "stall:$name:$mx"; then
                    say "LR SWEEP STALLED — $name running but its iteration counter has not moved from ${it} for ${age}min (last checkpoint ${mx}M)."
                    mark "stall:$name:$mx"
                fi ;;
            FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                if ! fired "dead:$name:$state"; then
                    say "LR SWEEP ARM $state — $name died at ${mx}M tokens after $el. Log: /nobackup/autodelete/grp/grp_pccl/vin/logs/"
                    mark "dead:$name:$state"
                fi ;;
        esac
    done <<< "$raw"

    if [ "$total" -gt 0 ] && [ "$done_n" -eq "$total" ] && ! fired "alldone"; then
        say "LR SWEEP DONE — all $total arms finished.$(echo -e "$summary")\n\nNext: compare final val loss per (LR, c) to see whether the optimum depends on c, then the wd arm."
        mark "alldone"
        exit 0
    fi
    sleep "$POLL"
done
