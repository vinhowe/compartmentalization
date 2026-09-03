#!/bin/bash
#SBATCH --job-name=wandb-checking
#SBATCH --output=logs/proxy-netcheck-%j.out
#SBATCH --error=logs/proxy-netcheck-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00

set -euo pipefail
mkdir -p logs

echo "SLURM_SUBMIT_HOST=$SLURM_SUBMIT_HOST"

# Where the kit lives
: "${PROXYKIT_DIR:=$HOME/sbatch}"

# Load shared config (exports env)
# shellcheck source=/dev/null
source "$PROXYKIT_DIR/config.sh"

echo "== proxy-netcheck on node $(hostname) =="

# 1) Start the sidecar once (background) on the batch step shell
TRACE=1 "$PROXYKIT_DIR/node/proxy-node.sh" sidecar &
SIDECAR_STEP_PID=$!

echo "SIDECAR_STEP_PID=$SIDECAR_STEP_PID"

# Wait + run checks (child expands $PROXYKIT_DIR)
bash -lc '
    set -euo pipefail
    NS_PID=$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)
    echo "netns ready: pid=$NS_PID"
    [ -e "/proc/$NS_PID/ns/user" ] && [ -e "/proc/$NS_PID/ns/net" ] && kill -0 "$NS_PID"
    nsenter -t "$NS_PID" -U --preserve-credentials -n -- "$PROXYKIT_DIR/tests/check-in-sh.sh"
'

# Cleanup
"$PROXYKIT_DIR/node/proxy-node.sh" signal-stop

wait "$SIDECAR_STEP_PID" || true
echo "Done."
