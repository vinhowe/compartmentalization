#!/bin/bash
# Telegram when the FineWeb corpus build finishes -- or when it quietly stops.
#
# WHY. Tokenization (data/fineweb2.py) and the ORC sync (sync_shards_to_orc.py)
# are both nohup'd background jobs with nothing attached to them. If either dies
# it leaves no trace but a log that stops growing, and the job sits dead looking
# exactly like a job that is working. That failure mode already cost 55 hours on
# the reseed runs.
#
# NOTIFIES ON FAILURE, NOT JUST SUCCESS. Three distinct bad states are reported
# separately, because the remedy differs:
#   died     -- process gone before the sentinel was written (crash / OOM / kill)
#   stalled  -- process alive but producing nothing for STALL_MIN
#   disk     -- pccfs2 nearly full, which is what will kill tokenization first
#
# Progress is read from the filesystem and the sync's own log rather than by
# polling ORC, so the watcher costs nothing and does not depend on the ssh
# multiplexer staying up. ORC is checked once, at completion, to confirm the
# copy actually landed.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-data/fineweb350B-bpe16384-nodedup}
SYNC_LOG=${SYNC_LOG:?set SYNC_LOG to the sync log path}
TOK_LOG=${TOK_LOG:?set TOK_LOG to the tokenization log path}
POLL=${POLL:-600}                # 10 min
STALL_MIN=${STALL_MIN:-25}       # shards land every ~7s, so 25 min of nothing is dead
SYNC_STALL_MIN=${SYNC_STALL_MIN:-60}
MIN_FREE_TB=${MIN_FREE_TB:-2}
# Overridable so the died/stalled paths can be tested while the real jobs run.
TOK_PATTERN=${TOK_PATTERN:-"fineweb2.py --parquet_dir"}
SYNC_PATTERN=${SYNC_PATTERN:-"sync_shards_to_orc.py"}
SENTINEL="$OUT/_TOKENIZATION_COMPLETE"
STATE=${STATE:-/tmp/watch_corpus_build.state}
NOTIFY=${NOTIFY:-$HOME/.claude/notify.sh}   # overridable so the alert paths are testable

fired() { grep -qxF "$1" "$STATE" 2>/dev/null; }
mark()  { echo "$1" >> "$STATE"; }
say()   { echo "[$(date '+%F %T')] $*"; bash "$NOTIFY" "$*"; }
touch "$STATE"

# Matching on the command line is self-referential: `pgrep -f <pat>` also matches
# any shell whose own argv contains the pattern, including the one that launched
# this watcher, which reports a dead job as alive. So confirm the candidate is
# really a python process by reading its executable name from /proc, and never
# count ourselves.
alive() {
    local pid comm
    for pid in $(pgrep -f "$1" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
        case "$comm" in python*) return 0 ;; esac
    done
    return 1
}
tok_alive()  { alive "$TOK_PATTERN"; }
# fineweb2.py's last line on a clean finish. Distinguishing "finished" from
# "killed" needs this: both leave no process, and only the log tells them apart.
tok_finished_cleanly() { grep -q "^Done\. Wrote" "$TOK_LOG" 2>/dev/null; }
sync_alive() { alive "$SYNC_PATTERN"; }
shards()     { ls "$OUT"/*.bin 2>/dev/null | wc -l; }
synced()     { grep -oE '[0-9]+ shards, [0-9.]+ GB total' "$SYNC_LOG" 2>/dev/null | tail -1 | grep -oE '^[0-9]+'; }

last_n=$(shards); last_change=$(date +%s)
last_sync=$(synced); last_sync_change=$(date +%s)

while true; do
    now=$(date +%s)
    n=$(shards); s=$(synced); s=${s:-0}
    [ "$n" != "$last_n" ] && { last_n=$n; last_change=$now; }
    [ "$s" != "$last_sync" ] && { last_sync=$s; last_sync_change=$now; }

    # --- disk: what actually kills tokenization first ---
    free_tb=$(df -BG --output=avail /mnt/pccfs2/backed_up 2>/dev/null | tail -1 | tr -dc '0-9')
    free_tb=$(( ${free_tb:-9999} / 1024 ))
    if [ "$free_tb" -lt "$MIN_FREE_TB" ] && ! fired "disk"; then
        say "DISK LOW — pccfs2 has ${free_tb}TB free; corpus build will fail. ${n} shards written."
        mark "disk"
    fi

    # --- tokenization ---
    if tok_alive; then
        age=$(( (now - last_change) / 60 ))
        if [ "$age" -ge "$STALL_MIN" ] && ! fired "tok-stall:$n"; then
            say "STALLED — tokenization alive but no new shard in ${age}min (stuck at ${n} shards). Log: $TOK_LOG"
            mark "tok-stall:$n"
        fi
    elif tok_finished_cleanly; then
        if [ ! -f "$SENTINEL" ]; then
            # the sync script exits on this file; writing it here is what lets
            # the sync finish instead of polling an already-complete corpus
            touch "$SENTINEL"
            echo "[$(date '+%F %T')] tokenization complete; wrote $SENTINEL"
        fi
    elif [ ! -f "$SENTINEL" ]; then
        if ! fired "tok-died:$n"; then
            say "DIED — tokenization exited before completing, at ${n} shards. Check $TOK_LOG (tail: $(tail -1 "$TOK_LOG" | cut -c1-120))"
            mark "tok-died:$n"
        fi
    fi

    # --- sync ---
    if sync_alive; then
        if [ "$s" -lt "$n" ]; then
            sage=$(( (now - last_sync_change) / 60 ))
            if [ "$sage" -ge "$SYNC_STALL_MIN" ] && ! fired "sync-stall:$s"; then
                say "STALLED — sync alive but stuck at ${s}/${n} shards for ${sage}min. Log: $SYNC_LOG"
                mark "sync-stall:$s"
            fi
        fi
    elif ! fired "sync-done" && ! fired "sync-died:$s"; then
        if [ -f "$SENTINEL" ] && [ "$s" -ge "$n" ]; then
            mark "sync-done"
        else
            say "DIED — shard sync exited at ${s}/${n} shards. Restart: scripts/sync_shards_to_orc.py --src $OUT ..."
            mark "sync-died:$s"
        fi
    fi

    # --- completion: tokenizer done, sentinel written, sync caught up ---
    if ! tok_alive && [ -f "$SENTINEL" ] && [ "$s" -ge "$n" ] && ! fired "done"; then
        gb=$(du -sh "$OUT" 2>/dev/null | cut -f1)
        orc=$(timeout 120 ssh -F /root/.ssh/orc_config -o ConnectTimeout=20 orc-th443 \
              "ls /nobackup/autodelete/grp/grp_pccl/vin/data/$(basename "$OUT")/*.bin 2>/dev/null | wc -l" \
              2>/dev/null | tr -dc '0-9')
        say "DONE — corpus built: ${n} shards (${gb}) local, ${orc:-?} on ORC. Next: append final token count to manifest, verify ORC copy, then the 1B config + LR sweep."
        mark "done"
        exit 0
    fi
    sleep "$POLL"
done
