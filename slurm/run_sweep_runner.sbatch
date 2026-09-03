#!/bin/bash
#SBATCH --job-name=translation-compression
#SBATCH --output=logs/translation-compression-%A_%a.out
#SBATCH --error=logs/translation-compression-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6-00:00:00


# # # # SBATCH --array=1-192

set -euo pipefail
mkdir -p logs
export OPENSSL_CONF=/dev/null

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch run_sweep_runner.sbatch <SWEEP_YAML>" >&2
    exit 1
fi
SWEEP_YAML="$1"
export SWEEP_YAML

echo "SLURM_SUBMIT_HOST=$SLURM_SUBMIT_HOST"
echo "SWEEP_YAML=$SWEEP_YAML | ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}"

# Where the proxy kit lives
: "${PROXYKIT_DIR:=/grphome/grp_pccl/vin/dev/translation-compression/slurm}"
export PROXYKIT_DIR

# Load shared config (exports env)
# shellcheck source=/dev/null
source "$PROXYKIT_DIR/config.sh"

:

echo "== tc-sweep-runner on node $(hostname) =="

# 1) Start the sidecar once (background) on the batch step shell
TRACE=1 "$PROXYKIT_DIR/node/proxy-node.sh" sidecar &
SIDECAR_STEP_PID=$!

echo "SIDECAR_STEP_PID=$SIDECAR_STEP_PID"

# 2) Run 8 sweep runners (one per GPU) on this node
# Training runs OUTSIDE namespace with HTTPS_PROXY for network access
srun --export=ALL --gpu-bind=single:1 --unbuffered bash -lc '
    set -euo pipefail
    NODE=$(hostname)

    # Wait for proxy sidecar to be ready
    NS_PID=$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)
    echo "AGENT start: host=$NODE localid=$SLURM_LOCALID ns=$NS_PID"

    # Read the Unix socket path
    PROXY_SOCK="$PROXY_ROOT_ON_NODE/proxy-sock.$NODE"
    if [ ! -S "$PROXY_SOCK" ]; then
        echo "[agent:$NODE] ERROR: proxy socket not found at $PROXY_SOCK" >&2
        exit 1
    fi

    # Each task gets unique TCP port for its proxy relay
    PROXY_TCP_PORT=$((19000 + SLURM_LOCALID))

    # Start socat relay: TCP port -> Unix socket (runs outside namespace)
    /usr/bin/socat TCP-LISTEN:$PROXY_TCP_PORT,fork,reuseaddr UNIX-CONNECT:"$PROXY_SOCK" &
    SOCAT_PID=$!
    sleep 0.5
    echo "[agent:$NODE:$SLURM_LOCALID] proxy relay on port $PROXY_TCP_PORT (PID $SOCAT_PID)"

    # Set proxy environment and run training OUTSIDE namespace (with GPU access)
    export HTTPS_PROXY="socks5://127.0.0.1:$PROXY_TCP_PORT"
    export HTTP_PROXY="socks5://127.0.0.1:$PROXY_TCP_PORT"
    # Let slurm handle CUDA_VISIBLE_DEVICES via --gpu-bind
    export OPENSSL_CONF=/dev/null

    TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
    cd "$TRAIN_DIR"
    source .supercomputer-venv/bin/activate
    # Tell uv to use the supercomputer venv
    export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"

    # If CUDA_VISIBLE_DEVICES not set by slurm (e.g., multi-task-per-GPU), compute it
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID / (SLURM_NTASKS_PER_NODE / SLURM_GPUS_ON_NODE)))
    fi
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    nvidia-smi -L || true
    which python3; python3 -V
    which wandb || { echo "wandb not in venv" >&2; exit 1; }

    # Run sweep_runner (outside namespace, with proxy env vars)
    # The runner loops internally: claims work, trains, repeats until sweep is done
    export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
    python sweep_runner.py --sweep "$SWEEP_YAML" --cluster slurm --wandb-buffer --mode poll
    SWEEP_RC=$?

    # Cleanup
    kill $SOCAT_PID 2>/dev/null || true
    echo "AGENT done: localid=$SLURM_LOCALID rc=$SWEEP_RC"
    exit $SWEEP_RC
'

# 4) Cleanup
"$PROXYKIT_DIR/node/proxy-node.sh" signal-stop

wait "$SIDECAR_STEP_PID" || true
echo "All sweep runners finished; proxies torn down."
