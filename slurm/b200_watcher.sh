#!/bin/bash
# Credentials live OUTSIDE the repo: ~/.secrets/telegram.env
# (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). They were hardcoded here
# and must never be again -- this file is tracked in git.
[ -f ~/.secrets/telegram.env ] && . ~/.secrets/telegram.env
# Poll cs-3-1 every 5 min; Telegram-push when the node is fully idle.
# Survives session death and even login-node logout (nohup + &).

LOG=/grphome/grp_pccl/vin/dev/translation-compression/logs/tied/b200-watcher.log
mkdir -p $(dirname $LOG)

send_telegram() {
    python3 - "$1" <<'PY'
import os, sys, urllib.request, urllib.parse
TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT = os.environ['TELEGRAM_CHAT_ID']
msg = sys.argv[1]
url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
data = urllib.parse.urlencode({'chat_id': CHAT, 'text': msg}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10).read()
except Exception as e:
    print(f'telegram send failed: {e}', file=sys.stderr)
PY
}

echo "[watcher] pid=$$ start=$(date -u +%FT%TZ)" >> $LOG
while true; do
    used=$(sinfo -N -h --Format=GresUsed:40 -n cs-3-1 2>/dev/null | grep -oE 'gpu:b200:[0-9]+' | grep -oE '[0-9]+$' | head -1)
    used=${used:-99}
    ts=$(date -u +%FT%TZ)
    echo "[watcher] $ts used=$used" >> $LOG
    if [ "$used" = "0" ]; then
        send_telegram "cs-3-1 fully FREE (b200 x8) at $ts. Ready to migrate tied. Watcher exiting."
        echo "[watcher] $ts SIGNALED, exiting" >> $LOG
        exit 0
    fi
    sleep 300
done
