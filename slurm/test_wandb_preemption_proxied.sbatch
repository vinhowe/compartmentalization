#!/bin/bash
#SBATCH --job-name=tc-preempt-test
#SBATCH --output=logs/tc-preempt-proxied-%A_%a.out
#SBATCH --error=logs/tc-preempt-proxied-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail
mkdir -p logs
export OPENSSL_CONF=/dev/null

# : "${WANDB_ENTITY:?WANDB_ENTITY is required (your W&B username or team)}"
WANDB_ENTITY="${WANDB_ENTITY:-tvin}"
PROJECT="${WANDB_PROJECT:-preempt-test}"
FORWARD_SIGNAL="${FORWARD_SIGNAL:-TERM}"  # default TERM (matches SLURM); set USR1 to test
SWEEP_TIMEOUT="${SWEEP_TIMEOUT:-120}"
CLEANUP_WAIT="${CLEANUP_WAIT:-15}"

echo "Entity=${WANDB_ENTITY} Project=${PROJECT} FORWARD_SIGNAL=${FORWARD_SIGNAL} CLEANUP_WAIT=${CLEANUP_WAIT}s"

# Where the proxy kit lives (same style as run_wandb_agents_proxied.sbatch)
: "${PROXYKIT_DIR:=$HOME/sbatch}"
export PROXYKIT_DIR
# shellcheck source=/dev/null
source "$PROXYKIT_DIR/config.sh"

echo "== tc-preempt-proxied on node $(hostname) =="

# Start the sidecar once (background) on the batch step shell
TRACE=1 "$PROXYKIT_DIR/node/proxy-node.sh" sidecar &
SIDECAR_STEP_PID=$!
echo "SIDECAR_STEP_PID=$SIDECAR_STEP_PID"

# Cleanup on SIGTERM at the batch level (preemption)
on_batch_term() {
  echo "[batch] Caught SIGTERM; keeping proxies up for cleanup window"
}
trap on_batch_term TERM INT

# Working dir for test artifacts
WORKDIR="${PWD}/.wandb_preempt_proxied.$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WORKDIR"
echo "WORKDIR=$WORKDIR"

PROGRAM_PATH="${WORKDIR}/train_test_preemption.py"
SWEEP_YAML="${WORKDIR}/sweep.yaml"
AGENT_LOG="${WORKDIR}/agent.log"
TRAIN_PIDFILE="${WORKDIR}/train.pid"
TRAIN_SIGNAL_FILE="${WORKDIR}/train.signal.log"
export TRAIN_PIDFILE
export TRAIN_SIGNAL_FILE
export WORKDIR

# Minimal Python training script:
# - Writes its PID to $TC_TRAIN_PIDFILE/$TRAIN_PIDFILE
# - Initializes W&B
# - Waits for a signal (TERM or USR1) via threading.Event()
# - On signal: mark_preempting (for TERM/USR1) and wandb.finish(exit_code=1)
cat > "$PROGRAM_PATH" <<'PY'
import wandb
import os, signal, sys, threading, time
stop_event = threading.Event()
sig_seen = {"sig": None}
pending_cleanup = False
cleanup_done = False
run = None
finish_done = threading.Event()
mark_done = threading.Event()
cleanup_wait_s = int(os.environ.get("TC_CLEANUP_WAIT", os.environ.get("CLEANUP_WAIT", "5")))
finish_done = threading.Event()
cleanup_wait_s = int(os.environ.get("TC_CLEANUP_WAIT", os.environ.get("CLEANUP_WAIT", "5")))

sigfile = os.environ.get("TC_TRAIN_SIGNAL_FILE") or os.environ.get("TRAIN_SIGNAL_FILE")

def _siglog(msg: str) -> None:
    if not sigfile:
        return
    try:
        with open(sigfile, "a") as f:
            f.write(msg + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        pass

def _h(signum, frame):
    global pending_cleanup, cleanup_done, run
    sig_seen["sig"] = signum
    print(f"[train] caught signal {signum}", flush=True)
    _siglog(f"CAUGHT {signum}")
    pending_cleanup = True
    if (not cleanup_done) and run is not None and signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None), getattr(signal, "SIGINT", None)):
        def _mark_and_finish_worker():
            try:
                print("[train] marking preempting (handler thread)", flush=True)
                _siglog("HANDLER_MARK_START")
                run.mark_preempting()
                _siglog("HANDLER_MARK_END")
                mark_done.set()
            except Exception as e:
                print(f"[train] mark_preempting failed in handler thread: {e}", flush=True)
                _siglog(f"HANDLER_MARK_ERROR {e}")
                mark_done.set()
            try:
                print("[train] finishing wandb (handler thread)", flush=True)
                _siglog("HANDLER_FINISH_START")
                wandb.finish(exit_code=1)
                _siglog("HANDLER_FINISH_END")
                global cleanup_done
                cleanup_done = True
            except Exception as e:
                print(f"[train] wandb.finish failed in handler thread: {e}", flush=True)
                _siglog(f"HANDLER_FINISH_ERROR {e}")
            finally:
                _siglog("HANDLER_FINISH_POST_WANDB")
                finish_done.set()
        threading.Thread(target=_mark_and_finish_worker, daemon=True).start()
    stop_event.set()

signal.signal(signal.SIGTERM, _h)
signal.signal(signal.SIGINT, _h)
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _h)

pidfile = os.environ.get("TC_TRAIN_PIDFILE") or os.environ.get("TRAIN_PIDFILE")
if pidfile:
    try:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
        print(f"[train] wrote pid {os.getpid()} to {pidfile}", flush=True)
        _siglog(f"PID {os.getpid()} -> {pidfile}")
    except Exception as e:
        print(f"[train] failed to write pidfile: {e}", flush=True)
        _siglog(f"PIDFILE_ERROR {e}")

try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config={"role": "preempt-proxied"})
    print("TRAIN_READY", flush=True)
    _siglog("READY")
    stop_event.wait()
    if (pending_cleanup and not cleanup_done) and run is not None and sig_seen["sig"] in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None), getattr(signal, "SIGINT", None)):
        if not finish_done.is_set():
            def _mark_and_finish_worker2():
                try:
                    print("[train] marking preempting (post-wait thread)", flush=True)
                    _siglog("MARK_START")
                    run.mark_preempting()
                    _siglog("MARK_END")
                    mark_done.set()
                except Exception as e:
                    print(f"[train] mark_preempting failed (post-wait thread): {e}", flush=True)
                    _siglog(f"MARK_ERROR {e}")
                    mark_done.set()
                try:
                    print("[train] finishing wandb (post-wait thread)", flush=True)
                    _siglog("FINISH_START")
                    wandb.finish(exit_code=1)
                    _siglog("FINISH_END")
                    global cleanup_done
                    cleanup_done = True
                except Exception as e:
                    print(f"[train] wandb.finish failed (post-wait thread): {e}", flush=True)
                    _siglog(f"FINISH_ERROR {e}")
                finally:
                    finish_done.set()
            threading.Thread(target=_mark_and_finish_worker2, daemon=True).start()
        finished = finish_done.wait(timeout=cleanup_wait_s)
        if not finished:
            _siglog("FINISH_TIMEOUT")
except Exception as e:
    print(f"[train] exception: {e}", flush=True)
    _siglog(f"EXCEPTION {e}")
    sys.exit(2)
finally:
    print("[train] exiting", flush=True)
    _siglog("EXIT")
PY

# Minimal sweep description
cat > "$SWEEP_YAML" <<YAML
program: $PROGRAM_PATH
method: grid
metric:
  name: step
  goal: maximize
parameters:
  dummy:
    values: [1, 2]
YAML

echo "Creating sweep (proxied via nsenter)..."
nsenter -t "$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)" -U --preserve-credentials -n -- bash -lc '
  set -euo pipefail
  TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
  cd "$TRAIN_DIR"
  source .supercomputer-venv/bin/activate
  which wandb
  wandb --version || true
  timeout -k 5 '"$SWEEP_TIMEOUT"' wandb sweep -p "'"${PROJECT}"'" -e "'"${WANDB_ENTITY}"'" "'"${SWEEP_YAML}"'"
' 2>&1 | tee "${WORKDIR}/sweep.create.log" || true
# Parse both SWEEP_PATH (entity/project/sweepid) and the trailing SWEEP_ID
SWEEP_PATH="$(sed -n 's/^.*sweep with ID: \([^ ]*\).*$/\1/p' "${WORKDIR}/sweep.create.log" | tail -1)"
SWEEP_ID="$(echo "${SWEEP_PATH}" | awk -F/ '{print $NF}')"
if [[ -z "${SWEEP_ID}" ]]; then
  echo "Failed to parse sweep ID from wandb sweep output" >&2
  echo "===== sweep.create.log (tail) =====" >&2
  tail -n 200 "${WORKDIR}/sweep.create.log" >&2 || true
  "$PROXYKIT_DIR/node/proxy-node.sh" signal-stop || true
  wait "$SIDECAR_STEP_PID" || true
  exit 1
fi
echo "SWEEP_ID=${SWEEP_ID}"
# If the CLI did not return a full path, construct it from env
if [[ -z "${SWEEP_PATH:-}" ]]; then
  SWEEP_PATH="${WANDB_ENTITY}/${PROJECT}/${SWEEP_ID}"
fi
echo "SWEEP_PATH=${SWEEP_PATH}"

# Export env for the agent and training process
export WANDB_PROJECT="${PROJECT}"
export TC_TRAIN_PIDFILE="${TRAIN_PIDFILE}"
export TC_TRAIN_SIGNAL_FILE="${TRAIN_SIGNAL_FILE}"
export TC_CLEANUP_WAIT="${CLEANUP_WAIT}"
export SWEEP_ID
export SWEEP_PATH
export WANDB_ENTITY
export SWEEP_PATH

# Run one agent (proxied) on this node.
# The wrapper catches SIGTERM and forwards USR1 (or TERM) ONLY to the train process (not the agent).
srun --export=ALL --gpu-bind=single:1 --unbuffered bash -lc '
  set -euo pipefail
  NS_PID=$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)
  echo "WRAPPER: host=$(hostname) ns=$NS_PID"
  # Forward from the OUTER (srun-level) wrapper by entering the user/net namespace
  send_train_signal_outer() {
    local sig="${FORWARD_SIGNAL:-TERM}"
    local pid=""
    if [[ -f "$TC_TRAIN_PIDFILE" ]]; then
      pid="$(cat "$TC_TRAIN_PIDFILE" || true)"
      echo "[outer] pidfile=$TC_TRAIN_PIDFILE pid=${pid:-<empty>}"
    fi
    if [[ -n "${pid:-}" ]]; then
      nsenter -t "$NS_PID" -U --preserve-credentials -n -- kill -s "$sig" "$pid" || true
      echo "[outer] Sent $sig to train pid=$pid (pidfile via nsenter)"
    else
      local p
      p="$(nsenter -t "$NS_PID" -U --preserve-credentials -n -- pgrep -f -n "train_test_preemption.py" || true)"
      if [[ -n "${p:-}" ]]; then
        nsenter -t "$NS_PID" -U --preserve-credentials -n -- kill -s "$sig" "$p" || true
        echo "[outer] Sent $sig to train pid=$p (pgrep via nsenter)"
      else
        echo "[outer] Train pid not found inside namespace"
      fi
    fi
    # Give the train process a brief window to finish W&B
    local waited=0
    local siglog="${TRAIN_SIGNAL_FILE:-}"
    if [[ -n "${siglog:-}" ]]; then
      for i in $(seq 1 '"$CLEANUP_WAIT"'); do
        if grep -qE "(HANDLER_FINISH_END|FINISH_END|EXIT|HANDLER_FINISH_POST_WANDB|FINISH)" "$siglog" 2>/dev/null; then
          echo "[outer] Observed cleanup markers in $siglog"
          break
        fi
        sleep 1
        waited=$((waited+1))
      done
      echo "[outer] Waited ${waited}s for cleanup markers"
    fi
  }
  trap send_train_signal_outer TERM INT

  nsenter -t "$NS_PID" -U --preserve-credentials -n -- bash -lc '\''
set -euo pipefail
TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR"
source .supercomputer-venv/bin/activate
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}" || true
nvidia-smi -L || true
which python3; python3 -V
which wandb || { echo "wandb not in venv" >&2; exit 1; }

FORWARD="${FORWARD_SIGNAL:-USR1}"

send_train_signal_wrapper() {
  local sig="$1"
  local sent=0
  if [[ -f "$TC_TRAIN_PIDFILE" ]]; then
    local pid
    pid="$(cat "$TC_TRAIN_PIDFILE" || true)"
    echo "[wrapper] pidfile=$TC_TRAIN_PIDFILE pid=${pid:-<empty>}"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -s "$sig" "$pid" || true
      echo "[wrapper] Sent $sig to train pid=$pid (pidfile)"
      sent=1
    fi
  fi
  if [[ "$sent" -eq 0 ]]; then
    local p
    p="$(pgrep -f -n "train_test_preemption.py" || true)"
    if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
      kill -s "$sig" "$p" || true
      echo "[wrapper] Sent $sig to train pid=$p (pgrep fallback)"
      sent=1
    fi
  fi
  if [[ "$sent" -eq 0 ]]; then
    echo "[wrapper] Train pid not running or not found"
  fi
}

on_term() {
  echo "[wrapper] Caught SIGTERM; forwarding $FORWARD to train"
  send_train_signal_wrapper "$FORWARD"
}
trap on_term TERM INT

TARGET="${SWEEP_PATH:-${SWEEP_ID:-}}"
if [[ -z "${TARGET}" ]]; then
  echo "[wrapper] No SWEEP_PATH or SWEEP_ID provided" >&2
  exit 1
fi
echo "[wrapper] Starting wandb agent for TARGET=${TARGET}"
wandb agent --count 1 "$TARGET" 2>&1 | tee -a "'"$AGENT_LOG"'" &
AGENT_PID=$!

READY=0
for i in $(seq 1 120); do
  if grep -q "TRAIN_READY" "'"$AGENT_LOG"'" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$AGENT_PID" 2>/dev/null; then break; fi
  sleep 1
done
tail -n 50 "'"$AGENT_LOG"'" || true
if [[ "$READY" -ne 1 ]]; then
  echo "[wrapper] TRAIN_READY not seen; agent may have exited early" >&2
fi

wait "$AGENT_PID" || true
echo "[wrapper] Agent exited (pid=$AGENT_PID)"
'\'' &
  INNER_SHELL_PID=$!
  wait "$INNER_SHELL_PID" || true

  echo "WRAPPER done on host $(hostname)"
'

# Cleanup proxies
"$PROXYKIT_DIR/node/proxy-node.sh" signal-stop || true
wait "$SIDECAR_STEP_PID" || true
echo "Done. WORKDIR=$WORKDIR"



