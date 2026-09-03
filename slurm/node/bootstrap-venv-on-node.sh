#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../config.sh"

# Reuse helper to wait for ns pid
NS_PID="$("$SCRIPT_DIR/proxy-node.sh" wait-ns)"

# Run the uv bootstrapper inside the proxied netns
nsenter -t "$NS_PID" -U --preserve-credentials -n -- "$SCRIPT_DIR/uv-bootstrap.sh"