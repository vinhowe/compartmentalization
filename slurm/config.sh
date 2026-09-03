#!/usr/bin/env bash
# Shared configuration for the proxy + venv kit.

# --- Ports (unique per job+node) ---
: "${PROXY_BASE_REMOTE_PORT:=38000}"   # remote microsocks base
: "${PROXY_BASE_LOCAL_PORT:=18000}"    # local SOCKS inside namespace

# --- Implementation knobs ---
: "${READY_TIMEOUT_SEC:=180}"

# --- Per-job state dir on each node ---
: "${PROXY_ROOT_ON_NODE:=/tmp/proxy/${SLURM_JOB_ID:-nojid}}"

# --- uv / Python venv (shared by all tasks on the node) ---
: "${PY_VERSION:=3.12}"                                # 3.10/3.11/3.12
: "${UV_BIN:=$HOME/.local/bin/uv}"
# : "${VENV_DIR:=${PROXY_ROOT_ON_NODE}/venv}"
# : "${UV_CACHE_DIR:=${PROXY_ROOT_ON_NODE}/uv-cache}"

# Optional: private indices (e.g., PyTorch wheels)
: "${PIP_EXTRA_INDEX_URL:=}"

export LOGIN_POOL_STR PROXY_BASE_REMOTE_PORT PROXY_BASE_LOCAL_PORT \
       READY_TIMEOUT_SEC PROXY_ROOT_ON_NODE PY_VERSION UV_BIN \
       PIP_EXTRA_INDEX_URL
       # VENV_DIR UV_CACHE_DIR PIP_EXTRA_INDEX_URL