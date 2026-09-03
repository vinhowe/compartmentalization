#!/usr/bin/env bash
# Login-side guard for a per-job loopback microsocks proxy.
#
# Keeps a microsocks instance bound to 127.0.0.1:$PORT alive while
#   (a) the worker's sentinel file still exists, AND
#   (b) the originating Slurm job is still in squeue.
# Cleans up its microsocks child, the sentinel, and the lock dir on
# any exit (normal completion, SIGTERM, SIGHUP, etc.). The squeue
# check is throttled so a guard whose worker dies without sending a
# signal still self-terminates within at most SQUEUE_INTERVAL seconds.
#
# Usage:  proxy-guard-login.sh <port> <sentinel_path> <log_path>
# Sentinel filename convention: sentinel-<SLURM_JOB_ID>-<node>.on
#
# Environment overrides:
#   SQUEUE_INTERVAL  seconds between squeue queries (default 300)
#   POLL_INTERVAL    seconds between local sentinel/PID polls (default 10)

set -euo pipefail

PORT="${1:?port}"
SENTINEL="${2:?sentinel}"
LOG="${3:?log}"

PROXY_DIR="$HOME/.proxyd"
mkdir -p "$(dirname "$LOG")" "$PROXY_DIR"

# ---------------------------------------------------------------------------
# Single-instance lock per port. mkdir is atomic on POSIX filesystems, unlike
# a "test then touch" check, and unlike flock(1) on a regular file it leaves
# no readable stale lock file behind once we clean up.
# ---------------------------------------------------------------------------
LOCKDIR="$PROXY_DIR/lock-${PORT}.d"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  prior=""
  [ -r "$LOCKDIR/pid" ] && prior="$(cat "$LOCKDIR/pid" 2>/dev/null || true)"
  # Bail if the owner is live OR still bootstrapping (empty pid file).
  # Only reclaim when we can prove the prior holder is dead.
  if [ -z "$prior" ] || kill -0 "$prior" 2>/dev/null; then
    exit 0
  fi
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR" 2>/dev/null || exit 0
fi
echo "$$" > "$LOCKDIR/pid"

# Extract Slurm job ID from sentinel filename (sentinel-<JOBID>-<node>.on)
SLURM_JOB_ID=""
if [[ "$(basename "$SENTINEL")" =~ ^sentinel-([0-9]+)- ]]; then
  SLURM_JOB_ID="${BASH_REMATCH[1]}"
fi

touch "$SENTINEL"

MS_PID=""
SLEEP_PID=""
cleanup() {
  trap - EXIT INT TERM HUP
  # Stop the pending interruptible-sleep child so it doesn't outlive us.
  if [ -n "$SLEEP_PID" ] && kill -0 "$SLEEP_PID" 2>/dev/null; then
    kill -TERM "$SLEEP_PID" 2>/dev/null || true
  fi
  if [ -n "$MS_PID" ] && kill -0 "$MS_PID" 2>/dev/null; then
    kill -TERM "$MS_PID" 2>/dev/null || true
    for _ in 1 2 3 4; do
      kill -0 "$MS_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$MS_PID" 2>/dev/null || true
  fi
  rm -f "$SENTINEL"
  rm -f "$LOCKDIR/pid"
  rmdir "$LOCKDIR" 2>/dev/null || true
  echo "[guard $$] cleanup done port=$PORT job=$SLURM_JOB_ID" >> "$LOG"
  exit 0
}
trap cleanup EXIT INT TERM HUP

start_microsocks() {
  microsocks -i 127.0.0.1 -p "$PORT" >>"$LOG" 2>&1 &
  MS_PID=$!
}

SQUEUE_INTERVAL="${SQUEUE_INTERVAL:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
NEXT_SQUEUE_TS=0
job_alive() {
  [ -z "$SLURM_JOB_ID" ] && return 0
  local now
  now=$(date +%s)
  if [ "$now" -lt "$NEXT_SQUEUE_TS" ]; then
    return 0
  fi
  NEXT_SQUEUE_TS=$(( now + SQUEUE_INTERVAL ))
  # Distinguish "job is gone" (clean shutdown) from "squeue failed"
  # (transient: don't abandon a healthy job just because slurmctld blipped).
  local out rc
  out="$(squeue -j "$SLURM_JOB_ID" --noheader 2>/dev/null)" || true
  rc=$?
  if [ "$rc" -ne 0 ]; then
    return 0
  fi
  [ -n "$out" ]
}

echo "[guard $$] start port=$PORT job=$SLURM_JOB_ID squeue_every=${SQUEUE_INTERVAL}s poll=${POLL_INTERVAL}s" >> "$LOG"
start_microsocks
echo "[guard $$] microsocks pid=$MS_PID" >> "$LOG"

while :; do
  if [ ! -f "$SENTINEL" ]; then
    echo "[guard $$] sentinel removed; exiting" >> "$LOG"
    break
  fi
  if ! job_alive; then
    echo "[guard $$] slurm job $SLURM_JOB_ID no longer in squeue; exiting" >> "$LOG"
    break
  fi
  if ! kill -0 "$MS_PID" 2>/dev/null; then
    echo "[guard $$] microsocks pid=$MS_PID died; restarting" >> "$LOG"
    start_microsocks
    echo "[guard $$] microsocks pid=$MS_PID" >> "$LOG"
  fi
  # Interruptible sleep. The external sleep(1) command defers bash trap
  # handling until it returns (signals are queued but the trap runs only
  # after the foreground command exits). Backgrounding sleep and using
  # the `wait` builtin lets signals fire the cleanup trap immediately.
  sleep "$POLL_INTERVAL" &
  SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null || true
done
