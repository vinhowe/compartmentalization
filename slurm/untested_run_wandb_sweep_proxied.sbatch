#!/bin/bash
#SBATCH --job-name=wandb-sweep-test
#SBATCH --output=logs/wandb-sweep-%A_%a.out
#SBATCH --error=logs/wandb-sweep-%A_%a.err
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

set -euo pipefail
mkdir -p logs
export OPENSSL_CONF=/dev/null

# --- Required: sweep id ---
if [ -z "${1:-}" ]; then
  echo "Usage: sbatch run_wandb_sweep_proxied.sbatch <WANDB_SWEEP_ID>" >&2
  exit 1
fi
SWEEP_ID="$1"

# --- Where the kit lives (shared FS) ---
: "${PROXYKIT_DIR:=$HOME/proxy-kit}"

# --- Your WANDB API key ---
export WANDB_API_KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"

# Optional: CUDA wheels, etc.
# export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Load config for env exports
# shellcheck source=/dev/null
source "$PROXYKIT_DIR/config.sh"

echo "Sweep: $SWEEP_ID | Array task: $SLURM_ARRAY_TASK_ID"

# 1) Start sidecar once per node (background)
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc "'$PROXYKIT_DIR/node/proxy-node.sh' sidecar" &
SIDECAR_STEP_PID=$!

# 2) Bootstrap uv venv on each node (through the proxy)
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc "'$PROXYKIT_DIR/node/bootstrap-venv-on-node.sh'"

# 3) Agents: enter shared netns + venv per node
srun --export=ALL --unbuffered bash -lc "
  set -euo pipefail
  . '$PROXYKIT_DIR/config.sh'
  NS_PID=\$('\"$PROXYKIT_DIR\"/node/proxy-node.sh' wait-ns)
  export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
  echo \"AGENT start: node=\$(hostname) task=\$SLURM_ARRAY_TASK_ID localid=\$SLURM_LOCALID ns=\$NS_PID\"
  nsenter -t \"\$NS_PID\" -n -- bash -lc '
    set -euo pipefail
    source \"'$VENV_DIR'/bin/activate\"
    which python; python -V
    which wandb || { echo \"wandb not in venv\" >&2; exit 1; }
    wandb agent --count 1 '\"$SWEEP_ID\"'
  '
  echo \"AGENT done: localid=\$SLURM_LOCALID\"
"

# 4) Cleanup: tell sidecars to stop, then wait
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" --ntasks-per-node=1 \
  bash -lc "'$PROXYKIT_DIR/node/proxy-node.sh' signal-stop"

wait "$SIDECAR_STEP_PID"
echo "All agents finished; proxies torn down."
