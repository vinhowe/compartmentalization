#!/bin/bash
#SBATCH --job-name=tc-train-proxied
#SBATCH --output=logs/tc-train-%j.out
#SBATCH --error=logs/tc-train-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00

set -euo pipefail
mkdir -p logs
export OPENSSL_CONF=/dev/null

echo "SLURM_SUBMIT_HOST=$SLURM_SUBMIT_HOST"

# Where the kit lives
: "${PROXYKIT_DIR:=$HOME/sbatch}"

# Load shared config (exports env)
# shellcheck source=/dev/null
source "$PROXYKIT_DIR/config.sh"

echo "== tc-train-proxied on node $(hostname) =="

# 1) Start the sidecar once (background) on the batch step shell
TRACE=1 "$PROXYKIT_DIR/node/proxy-node.sh" sidecar &
SIDECAR_STEP_PID=$!

echo "SIDECAR_STEP_PID=$SIDECAR_STEP_PID"

# 2) Wait for netns and run training inside it
bash -lc '
    set -euo pipefail
    NS_PID=$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)
    echo "netns ready: pid=$NS_PID"
    [ -e "/proc/$NS_PID/ns/user" ] && [ -e "/proc/$NS_PID/ns/net" ] && kill -0 "$NS_PID"
    nsenter -t "$NS_PID" -U --preserve-credentials -n -- bash -lc '\''
        set -euo pipefail
        TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
        cd "$TRAIN_DIR"
        source .supercomputer-venv/bin/activate
        which python3; python3 -V
        python3 train.py --job.config-file ./config/1-domain-baseline-fineweb-8-32-prototype.toml
    '\''
'

# 3) Cleanup
"$PROXYKIT_DIR/node/proxy-node.sh" signal-stop

wait "$SIDECAR_STEP_PID" || true
echo "Done."