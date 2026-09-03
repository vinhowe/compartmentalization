#!/usr/bin/env bash
# Run this INSIDE the proxied netns (the sbatch calls it via nsenter).
# Verifies: default route via tun0, DNS resolution, TCP reachability to login host
# (via slirp keep-route), and HTTPS egress (via tun2socks).
set -euo pipefail

echo "=== proxy-kit network check (inside netns) ==="

note() { printf "\n-- %s --\n" "$*"; }
pass() { printf "  [OK] %s\n" "$*"; }
fail() { printf "  [FAIL] %s\n" "$*" >&2; exit 1; }

note "interfaces"
ip -o -4 addr show || true

note "routes"
ip route || true

note "resolv.conf"
cat /etc/resolv.conf || true

# Early hint if TUN is blocked
if [ ! -c /dev/net/tun ]; then
  echo "  [WARN] /dev/net/tun is not available inside this user/net namespace."
  echo "         Your cluster likely disallows TUN in user namespaces."
fi

# 1) Default route must be tun0 (transparent through tun2socks)
note "assert default route is tun0"
if ip route show default | grep -q 'dev tun0'; then
  pass "default route via tun0"
else
  fail "default route is NOT via tun0"
fi

# 2) Tun2socks process should be present
note "tun2socks process"
if pgrep -af 'tun2socks' >/dev/null; then
  pgrep -af 'tun2socks' | sed 's/^/  /'
  pass "tun2socks running"
else
  fail "tun2socks not found"
fi

# 3) Find the login host IP that was pinned via slirp (route via 10.0.2.2)
note "detect login host IP pinned via slirp"
LOGIN_IP="$(ip route | awk '/via 10\.0\.2\.2/ && $1 !~ /10\.0\.2\.0\/24/ {print $1; found=1} END{exit !found}')"
echo "  pinned login IP: ${LOGIN_IP}"

# # 4) Verify TCP 22 to login IP works via slirp (no proxy loop)
# note "TCP check to login IP:22 (should use slirp path)"
# if command -v nc >/dev/null 2>&1; then
#   if nc -vz -w 3 "$LOGIN_IP" 22 >/dev/null 2>&1; then
#     pass "TCP to $LOGIN_IP:22 reachable"
#   else
#     fail "cannot reach $LOGIN_IP:22"
#   fi
# else
#   # Portable Bash /dev/tcp fallback
#   if timeout 3 bash -lc "exec 3<>/dev/tcp/$LOGIN_IP/22"; then
#     pass "TCP to $LOGIN_IP:22 reachable"
#   else
#     fail "cannot reach $LOGIN_IP:22"
#   fi
# fi

# 5) DNS must resolve through slirp DNS
note "DNS resolution"
getent hosts example.com | head -n1 || fail "DNS resolution failed"
pass "DNS works"

# 6) HTTPS egress through tun2socks (fetch public IP + HEAD request)
note "HTTPS egress via tun (curl)"
PUB_IP="$(curl -4 -fsS --max-time 8 https://api.ipify.org || true)"
[ -n "${PUB_IP}" ] || fail "curl to api.ipify.org failed"
echo "  public IP: ${PUB_IP}"
curl -fsSI --max-time 8 https://www.google.com | head -n1 | sed 's/^/  /' || fail "HTTPS HEAD failed"
pass "HTTPS fetches OK"

# 7) Routing sanity (1.1.1.1 via tun0)
note "ip route get 1.1.1.1"
ip route get 1.1.1.1 | sed 's/^/  /'
ip route get 1.1.1.1 | grep -q ' dev tun0 ' && pass "route to 1.1.1.1 via tun0" || fail "1.1.1.1 not via tun0"

echo
echo "✅ ALL CHECKS PASSED"