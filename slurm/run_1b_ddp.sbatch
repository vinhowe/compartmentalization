#!/bin/bash
#SBATCH --job-name=1b-8comp-ddp
#SBATCH --output=logs/1b-ddp-%j.out
#SBATCH --error=logs/1b-ddp-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=1-00:00:00

set -euo pipefail
mkdir -p logs
export OPENSSL_CONF=/dev/null

CONFIG_FILE="${1:-config/1b-8comp-bpe16384.toml}"

: "${PROXYKIT_DIR:=/grphome/grp_pccl/vin/dev/translation-compression/slurm}"
export PROXYKIT_DIR

# Load shared config (exports env)
source "$PROXYKIT_DIR/config.sh"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
cd "$TRAIN_DIR"

echo "== 1B DDP Training on $(hostname) =="
echo "CONFIG_FILE=$CONFIG_FILE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L || true

# 1) Start proxy sidecar for wandb network access
TRACE=1 "$PROXYKIT_DIR/node/proxy-node.sh" sidecar &
SIDECAR_PID=$!

# 2) Wait for proxy to be ready
NS_PID=$("$PROXYKIT_DIR/node/proxy-node.sh" wait-ns)
echo "Proxy namespace PID=$NS_PID"

NODE=$(hostname)
PROXY_SOCK="$PROXY_ROOT_ON_NODE/proxy-sock.$NODE"
if [ ! -S "$PROXY_SOCK" ]; then
    echo "ERROR: proxy socket not found at $PROXY_SOCK" >&2
    exit 1
fi

# Single socat relay shared by all torchrun workers
PROXY_TCP_PORT=19000
/usr/bin/socat TCP-LISTEN:$PROXY_TCP_PORT,fork,reuseaddr UNIX-CONNECT:"$PROXY_SOCK" &
SOCAT_PID=$!
sleep 1
echo "Proxy relay on port $PROXY_TCP_PORT (PID $SOCAT_PID)"

export HTTPS_PROXY="socks5://127.0.0.1:$PROXY_TCP_PORT"
export HTTP_PROXY="socks5://127.0.0.1:$PROXY_TCP_PORT"

# 3) Activate venv
source .supercomputer-venv/bin/activate
export UV_PROJECT_ENVIRONMENT="$TRAIN_DIR/.supercomputer-venv"
export TC_STORAGE_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"

which python3; python3 -V

# 4) Launch DDP training across all 8 GPUs
echo "Launching torchrun --nproc_per_node=8 train.py --job.config-file $CONFIG_FILE"
torchrun --standalone --nproc_per_node=8 \
    train.py --job.config-file "$CONFIG_FILE"

TRAIN_RC=$?

# 5) Cleanup
kill $SOCAT_PID 2>/dev/null || true
"$PROXYKIT_DIR/node/proxy-node.sh" signal-stop
wait "$SIDECAR_PID" || true

echo "Training finished with rc=$TRAIN_RC"
exit $TRAIN_RC
