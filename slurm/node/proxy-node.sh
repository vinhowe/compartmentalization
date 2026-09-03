#!/usr/bin/env bash
# Per-node sidecar: start/keep a proxied netns; publish ns pid; cleanup on signal.
set -euo pipefail
export OPENSSL_CONF=/dev/null

# Optional verbose tracing: enable with TRACE=1 in the environment
if [[ "${TRACE:-0}" == "1" ]]; then
  # Pretty PS4 with timestamps and PID
  export PS4='+ [$(date "+%F %T")] [$BASHPID] ${FUNCNAME[0]:-main}: '
  # Send xtrace to its own file, not stderr
  TRACE_LOG="/tmp/proxy-trace-${SLURM_JOB_ID:-nojid}-$(hostname).log"
  exec {BASH_XTRACEFD}>>"$TRACE_LOG"
  export BASH_XTRACEFD
  set -x
  echo "[trace] sidecar xtrace -> $TRACE_LOG"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load shared config (expects SLURM_* env present)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../config.sh"

pick_login() {
  # Use the submit node that launched this job (resolve to IPv4)
  local submit_host="${SLURM_SUBMIT_HOST:-}"
  if [[ -z "$submit_host" ]]; then
    echo "[proxy] ERROR: SLURM_SUBMIT_HOST is not set" >&2; return 1
  fi
  local ip
  ip="$(getent ahostsv4 "$submit_host" | awk '$1 != "127.0.0.1" {print $1; exit}')"
  if [[ -z "$ip" ]]; then
    echo "[proxy] ERROR: cannot resolve submit host '$submit_host'" >&2; return 1
  fi
  echo "$ip"
}

ensure_remote_guard() {
  local REMOTE_HOST="$1"
  # Prefer scp; fallback to heredoc
  if ! ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" 'test -x ~/.proxyd/proxy-guard-login.sh'; then
    ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" 'mkdir -p ~/.proxyd'
    scp -q "$SCRIPT_DIR/../login/proxy-guard-login.sh" \
        "$REMOTE_HOST:~/.proxyd/proxy-guard-login.sh"
    ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" 'chmod +x ~/.proxyd/proxy-guard-login.sh'
  fi
}

node_sidecar() {
  umask 077
  mkdir -p "$PROXY_ROOT_ON_NODE"
  mkdir -p "$HOME/.proxyd"
  local NODE="$(hostname)"
  local NS_FILE="$PROXY_ROOT_ON_NODE/ns-pid.$NODE"
  local SENTINEL="$PROXY_ROOT_ON_NODE/run.flag.$NODE"
  : > "$SENTINEL"

  local job_off=$(( ${SLURM_JOB_ID:-$$} % 1500 ))
  local node_off=${SLURM_NODEID:-0}
  local REMOTE_PORT=$(( PROXY_BASE_REMOTE_PORT + job_off*16 + node_off ))
  local REMOTE_HOST
  REMOTE_HOST="$(pick_login)"
  local REMOTE_USER="${SLURM_JOB_USER:-${USER:-$(id -un)}}"
  local REMOTE_LOG="$HOME/.proxyd/microsocks-${SLURM_JOB_ID}-${NODE}.log"
  local REMOTE_SENTINEL="$HOME/.proxyd/sentinel-${SLURM_JOB_ID}-${NODE}.on"
  # Build SSH identity options from common key files and agent
  local SSH_IDENTITY_ARGS=""
  # [ -r "$HOME/.ssh/id_ed25519" ] && SSH_IDENTITY_ARGS+=" -o IdentityFile=$HOME/.ssh/id_ed25519"
  [ -r "$HOME/.ssh/id_rsa" ] && SSH_IDENTITY_ARGS+=" -o IdentityFile=$HOME/.ssh/id_rsa"
  [ -r "$HOME/.ssh/id_ecdsa" ] && SSH_IDENTITY_ARGS+=" -o IdentityFile=$HOME/.ssh/id_ecdsa"
  [ -r "$HOME/.ssh/id_dsa" ] && SSH_IDENTITY_ARGS+=" -o IdentityFile=$HOME/.ssh/id_dsa"
  local SSH_IDENTITY_AGENT_OPT=""
  if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
    SSH_IDENTITY_AGENT_OPT="-o IdentityAgent=$SSH_AUTH_SOCK"
  fi

  ensure_remote_guard "$REMOTE_HOST"

  # Start/refresh guard
  ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -lc '
    mkdir -p ~/.proxyd
    echo $$ > \"$REMOTE_SENTINEL\"
    nohup ~/.proxyd/proxy-guard-login.sh $REMOTE_PORT \"$REMOTE_SENTINEL\" \"$REMOTE_LOG\" \
      >/dev/null 2>&1 & disown
    sleep 0.5
    pgrep -af \"microsocks .* -p $REMOTE_PORT\" >/dev/null
  '"

  # Netns init
  unshare --user --map-root-user \
          --net --pid --fork --mount-proc bash -c '
    set -e
    ip link set lo up
    sleep infinity
  ' &
  NS_INIT=$!
  nsenter -t "$NS_INIT" -U --preserve-credentials -- \
    slirp4netns -c "$NS_INIT" tap0 >"$HOME/.proxyd/slirp-${SLURM_JOB_ID}-${NODE}.log" 2>&1 &
  SLIRP_PID=$!
  sleep 0.5

  # Do all privileged setup from inside the ns init (we are uid 0 in user-ns):
  # If TUN is unavailable, leave the ns up so the checker can report it.
  if nsenter -t "$NS_INIT" -U --preserve-credentials -n \
       env HOME="$HOME" SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" OPENSSL_CONF="/dev/null" \
           REMOTE_HOST="$REMOTE_HOST" REMOTE_USER="$REMOTE_USER" \
           PROXY_BASE_LOCAL_PORT="$PROXY_BASE_LOCAL_PORT" REMOTE_PORT="$REMOTE_PORT" \
           SLURM_JOB_ID="${SLURM_JOB_ID}" NODE="$NODE" \
           SSH_IDENTITY_ARGS="$SSH_IDENTITY_ARGS" SSH_IDENTITY_AGENT_OPT="$SSH_IDENTITY_AGENT_OPT" \
       bash -s <<'NSENTER_CMDS'
set -euo pipefail
exec 2>"$HOME/.proxyd/ns-setup-${SLURM_JOB_ID}-${NODE}.err"
# Wait for slirp to create tap0
for i in {1..50}; do ip link show tap0 >/dev/null 2>&1 && break; sleep 0.1; done
# Preflight: TUN available?
if [ ! -c /dev/net/tun ]; then
  echo "[proxy] /dev/net/tun MISSING in this namespace; tun2socks cannot run" >&2
  exit 42
fi
if [[ "$REMOTE_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  LOGIN_IP="$REMOTE_HOST"
else
  LOGIN_IP=$(getent ahostsv4 "$REMOTE_HOST" | awk '$1 != "127.0.0.1" {print $1; exit}')
fi
# SSH tunnel to remote loopback microsocks
ssh -o ExitOnForwardFailure=yes \
    -l "$REMOTE_USER" \
    -o PreferredAuthentications=publickey \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes $SSH_IDENTITY_AGENT_OPT $SSH_IDENTITY_ARGS \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    -o GlobalKnownHostsFile=/dev/null \
    -o StrictHostKeyChecking=no -fN \
    -L 127.0.0.1:"$PROXY_BASE_LOCAL_PORT":127.0.0.1:"$REMOTE_PORT" \
    "$REMOTE_HOST"
# Keep slirp subnet + login host through tap0
ip route replace 10.0.2.0/24 dev tap0
ip route replace "$LOGIN_IP" via 10.0.2.2 dev tap0
# Route resolver via tap0 to avoid UDP DNS over tun
NS_DNS=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf || true)
if [ -n "$NS_DNS" ] && [ "$NS_DNS" != "10.0.2.3" ]; then
  ip route replace "$NS_DNS" via 10.0.2.2 dev tap0 || true
fi
# Create tun0 and start tun2socks; flip default route
ip link show tun0 >/dev/null 2>&1 || ip tuntap add dev tun0 mode tun
ip addr replace 10.0.0.2/24 dev tun0
ip link set tun0 up
tun2socks -device tun://tun0 -proxy socks5://127.0.0.1:"$PROXY_BASE_LOCAL_PORT" \
          -loglevel info >"$HOME/.proxyd/tun2socks-${SLURM_JOB_ID}-${NODE}.log" 2>&1 &
echo $! > "$HOME/.proxyd/tun2socks-${SLURM_JOB_ID}-${NODE}.pid"
ip route replace default dev tun0
NSENTER_CMDS
    then
    :
  else
    rc=$?
    echo "RC=$rc"
    if [ "$rc" -eq 42 ]; then
      echo "[proxy:$NODE] TUN unavailable; leaving namespace up (slirp only)."
    else
      echo "[proxy:$NODE] setup failed (rc=$rc). Check $HOME/.proxyd/slirp-${SLURM_JOB_ID}-${NODE}.log" >&2
      # Keep the ns alive so you can inspect it; don’t publish PID yet.
      # (We’ll still publish so the test can print a clear error.)
    fi
  fi
  
  # Healthcheck (only if tun2socks likely started)
  if nsenter -t "$NS_INIT" -U --preserve-credentials -n pgrep -af tun2socks >/dev/null 2>&1; then
    nsenter -t "$NS_INIT" -U --preserve-credentials -n bash -c "curl -fksS --max-time 3 https://api.ipify.org"
  else
    echo "[proxy:$NODE] skipping healthcheck (no tun2socks)"
  fi

  # Start Unix socket bridge inside namespace for external access
  local PROXY_SOCK="$PROXY_ROOT_ON_NODE/proxy-sock.$NODE"
  rm -f "$PROXY_SOCK"
  nsenter -t "$NS_INIT" -U --preserve-credentials -n \
    /usr/bin/socat UNIX-LISTEN:"$PROXY_SOCK",fork TCP:127.0.0.1:"$PROXY_BASE_LOCAL_PORT" &
  SOCAT_BRIDGE_PID=$!
  echo "$SOCAT_BRIDGE_PID" > "$PROXY_ROOT_ON_NODE/socat-bridge-pid.$NODE"
  sleep 0.5
  echo "[proxy:$NODE] Unix socket bridge started at $PROXY_SOCK (PID $SOCAT_BRIDGE_PID)"

  echo "$NS_INIT" > "$NS_FILE"
  echo "$PROXY_SOCK" > "$PROXY_ROOT_ON_NODE/proxy-sock-path.$NODE"
  echo "[proxy:$NODE] READY ns_pid=$(cat "$NS_FILE") sock=$PROXY_SOCK remote=$REMOTE_HOST:$REMOTE_PORT"

  # Keep alive
  while [ -f "$SENTINEL" ]; do sleep 2; done

  # Teardown
  echo "[proxy:$NODE] stopping…"
  ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "rm -f '$REMOTE_SENTINEL'" || true
  nsenter -t "$NS_INIT" -U --preserve-credentials -n bash -lc "kill \$(cat $HOME/.proxyd/tun2socks-${SLURM_JOB_ID}-${NODE}.pid) 2>/dev/null || true" || true
  kill "$SOCAT_BRIDGE_PID" 2>/dev/null || true
  kill "$SLIRP_PID" 2>/dev/null || true
  kill "$NS_INIT" 2>/dev/null || true
  rm -f "$NS_FILE" "$PROXY_SOCK" "$PROXY_ROOT_ON_NODE/proxy-sock-path.$NODE" "$PROXY_ROOT_ON_NODE/socat-bridge-pid.$NODE"
}

wait_ns() {
  local NODE="$(hostname)"
  local NS_FILE="$PROXY_ROOT_ON_NODE/ns-pid.$NODE"
  local t=0
  while [ ! -s "$NS_FILE" ] && [ $t -lt "$READY_TIMEOUT_SEC" ]; do
    sleep 1; t=$((t+1))
  done
  [ -s "$NS_FILE" ] && cat "$NS_FILE" || { echo "[agent:$NODE] proxy not ready" >&2; exit 1; }
}

signal_stop() {
  local NODE="$(hostname)"
  local SENTINEL="$PROXY_ROOT_ON_NODE/run.flag.$NODE"
  [ -f "$SENTINEL" ] && rm -f "$SENTINEL" || true
  echo "[cleanup:$NODE] signaled"
}

# CLI
case "${1:-}" in
  sidecar)      node_sidecar ;;
  wait-ns)      wait_ns ;;
  signal-stop)  signal_stop ;;
  *) echo "usage: $0 {sidecar|wait-ns|signal-stop}" >&2; exit 2 ;;
esac