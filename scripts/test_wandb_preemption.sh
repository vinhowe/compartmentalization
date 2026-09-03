#!/usr/bin/env bash
set -euo pipefail

# Self-contained W&B preemption test:
# - Creates a minimal sweep
# - Launches a wandb agent in its own session
# - Runs a tiny Python program that catches SIGTERM and finishes W&B
# - Sends TERM to simulate SLURM preemption
#
# Requirements:
# - wandb CLI installed and logged in (WANDB_API_KEY or prior `wandb login`)
# - python3 with wandb installed in PATH
#
# Usage:
#   WANDB_ENTITY=<entity> [WANDB_PROJECT=<project>] [GRACE=15] ./scripts/test_wandb_preemption.sh
#
# Env:
# - WANDB_ENTITY  (required)
# - WANDB_PROJECT (default: preempt-test)
# - GRACE         seconds to wait after TERM before killing group (default 15)

ENTITY="${WANDB_ENTITY:-}"
if [[ -z "${ENTITY}" ]]; then
    echo "WANDB_ENTITY is required (your W&B username or team)" >&2
  exit 1
fi
PROJECT="${WANDB_PROJECT:-preempt-test}"
GRACE="${GRACE:-15}"

echo "Entity=${ENTITY} Project=${PROJECT} GRACE=${GRACE}s"
command -v wandb >/dev/null || { echo "wandb CLI not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

WORKDIR="${PWD}/.wandb_preempt_test.$(date +%Y%m%d-%H%M%S)"
mkdir -p "${WORKDIR}"
echo "WORKDIR=${WORKDIR}"
cd "${WORKDIR}"

# Stop-file based preemption (like SLURM traps in sbatch)
export TC_SHARED_DIR="${WORKDIR}/tc-shared"
export TC_STOP_FILE="${TC_SHARED_DIR}/stop"
mkdir -p "${TC_SHARED_DIR}"
_usr1() {
  echo "[supervisor] USR1 received; creating stop file at ${TC_STOP_FILE}"
  echo stop > "${TC_STOP_FILE}" || true
}
_term() {
  echo "[supervisor] TERM received; creating stop file and sleeping ${GRACE}s"
  echo stop > "${TC_STOP_FILE}" || true
  sleep "${GRACE}" || true
}
trap _usr1 USR1
trap _term TERM INT

cat > train_test_preemption.py <<'PY'
import os, signal, sys, time
import os.path

stop = {"flag": False, "sig": None}

def _h(signum, frame):
    print(f"[train] caught signal {signum}", flush=True)
    stop["flag"] = True
    stop["sig"] = signum

signal.signal(signal.SIGINT, _h)
signal.signal(signal.SIGTERM, _h)
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _h)

run = None
try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config={"role": "preempt-test"})
    print("TRAIN_READY", flush=True)
    step = 0
    stop_file = os.environ.get("TC_STOP_FILE", "")
    while True:
        if stop_file and os.path.exists(stop_file) and not stop["flag"]:
            print("[train] detected stop-file", flush=True)
            stop["flag"] = True
            # Treat stop-file as SIGTERM semantics in this test
            stop["sig"] = getattr(signal, "SIGTERM", None)
        if stop["flag"]:
            print(f"[train] stop flag set, sig={stop['sig']}", flush=True)
            if run is not None and stop["sig"] in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)):
                try:
                    print(f"[train] marking preempting", flush=True)
                    run.mark_preempting()
                except Exception as e:
                    print(f"[train] mark_preempting failed: {e}", flush=True)
            if run is not None:
                try:
                    print(f"[train] finishing wandb", flush=True)
                    exit_code = 1 if stop["sig"] in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)) else 0
                    wandb.finish(exit_code=exit_code)
                except Exception as e:
                    print(f"[train] wandb.finish failed: {e}", flush=True)
            break
        if run is not None:
            try:
                wandb.log({"heartbeat": 1, "step": step})
            except Exception as e:
                print(f"[train] wandb.log failed: {e}", flush=True)
        time.sleep(1.0)
        step += 1
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"[train] exception: {e}", flush=True)
    sys.exit(2)
finally:
    print("[train] exiting", flush=True)
PY

cat > sweep.yaml <<'YAML'
program: train_test_preemption.py
method: grid
metric:
  name: step
  goal: maximize
parameters:
  dummy:
    values: [1, 2]
YAML

echo "Creating sweep..."
CREATE_OUT="$(wandb sweep -p "${PROJECT}" -e "${ENTITY}" sweep.yaml 2>&1 | tee sweep.create.log || true)"
echo "${CREATE_OUT}" | sed -n '1,200p' > /dev/null

SWEEP_ID="$(echo "${CREATE_OUT}" | sed -n 's/^.*Creating sweep with ID: \([^ ]*\).*$/\1/p' | tail -1)"
if [[ -z "${SWEEP_ID}" ]]; then
    echo "Failed to parse sweep ID from wandb sweep output" >&2
    echo "${CREATE_OUT}" >&2
  exit 1
fi
echo "SWEEP_ID=${SWEEP_ID}"

echo "Launching agent in its own session..."
AGENT_LOG="${WORKDIR}/agent.log"
setsid bash -lc "set -euo pipefail; export WANDB_PROJECT='${PROJECT}'; export TC_SHARED_DIR='${TC_SHARED_DIR}'; export TC_STOP_FILE='${TC_STOP_FILE}'; exec wandb agent --count 1 '${SWEEP_ID}'" > "${AGENT_LOG}" 2>&1 &
AGENT_SHELL_PID=$!
sleep 0.5
AGENT_PGID="$(ps -o pgid= -p "${AGENT_SHELL_PID}" | tr -d ' ')"
echo "AGENT_SHELL_PID=${AGENT_SHELL_PID} PGID=${AGENT_PGID}"

echo "Waiting for TRAIN_READY from training script..."
READY=0
for i in $(seq 1 120); do
    if grep -q "TRAIN_READY" "${AGENT_LOG}"; then READY=1; break; fi
    if ! kill -0 "${AGENT_SHELL_PID}" 2>/dev/null; then break; fi
  sleep 1
done
tail -n 50 "${AGENT_LOG}" || true
if [[ "${READY}" -ne 1 ]]; then
    echo "TRAIN_READY not seen; agent may have exited early" >&2
  exit 2
fi

# Send TERM to this supervisor (trap writes stop-file and sleeps GRACE)
kill -TERM "$$" || true

echo "Waiting ${GRACE}s for graceful shutdown..."
for i in $(seq 1 "${GRACE}"); do
    if ! kill -0 "${AGENT_SHELL_PID}" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "${AGENT_SHELL_PID}" 2>/dev/null; then
    echo "Agent still alive; sending SIGKILL to group"
    kill -KILL "-${AGENT_PGID}" || true
fi

echo "===== agent.log (tail) ====="
tail -n 200 "${AGENT_LOG}" || true
echo "Done. WORKDIR=${WORKDIR}"


