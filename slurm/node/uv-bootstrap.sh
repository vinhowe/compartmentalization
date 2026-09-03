#!/usr/bin/env bash
set -euo pipefail
# Expects config env exported: PY_VERSION, VENV_DIR, PIP_EXTRA_INDEX_URL

export PATH="$HOME/.local/bin:$PATH"
# Force OpenSSL to skip system config (broken on some clusters)
export OPENSSL_CONF=/dev/null
# export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
[ -n "${PIP_EXTRA_INDEX_URL:-}" ] && export PIP_EXTRA_INDEX_URL

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y
fi

# Interpreter + venv
uv python install "${PY_VERSION:?PY_VERSION required}"
uv venv --python "${PY_VERSION}" "${VENV_DIR:?VENV_DIR required}"

# Dependencies
if [ -f uv.lock ]; then
  uv sync --frozen --no-dev --python "$PY_VERSION"
elif [ -f pyproject.toml ]; then
  uv sync --no-dev --python "$PY_VERSION"
elif [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
else
  uv pip install wandb
fi

# Smoke test
"${VENV_DIR}/bin/python" - <<'PY'
import sys, importlib
print("PY:", sys.executable)
print("VER:", sys.version.split()[0])
try:
  wandb = importlib.import_module("wandb")
  print("wandb:", getattr(wandb, "__version__", "?"))
except Exception as e:
  print("wandb import failed:", e)
  raise SystemExit(1)
PY