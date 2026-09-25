#!/bin/bash
# Run inside the dw-2-2 container: wait for the three avgemb arms to reach step 918000,
# then formal eval (all checkpoints) -> merge into val_metrics.json -> plot -> Telegram.
set -u
REPO=/mnt/pccfs2/backed_up/vin/dev/translation-compression
PY=$REPO/.venv/bin/python
OUT=$REPO/out/translation-compression/8-256-avgemb
WT=$REPO/.claude/worktrees/capacity-use
WORK=$REPO/experiment/.eval_avgemb
FIG=$WT/experiment/avgemb/figures/avgemb
ARMS="avgboth avgwte control"
log() { echo "[$(date -u '+%F %T')] $*"; }

for i in $(seq 1 120); do
  missing=0
  for a in $ARMS; do [ -f "$OUT/8-256-n8-tr01comp-s66-$a/checkpoints/step-918000/model.pt" ] || missing=$((missing+1)); done
  [ "$missing" -eq 0 ] && break
  log "waiting: $missing arms short of step 918000"; sleep 300
done
if [ "$missing" -ne 0 ]; then bash /root/.claude/notify.sh "avgemb: $missing arms never reached step 918000; pipeline stopped."; exit 1; fi
sleep 120

mkdir -p "$WORK"; ln -sfn "$REPO/experiment/val_metrics.json" "$WORK/val_metrics.json"; cd "$WORK"
pids=(); i=0
for g in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=offline PYTHONPATH=$REPO TC_STORAGE_ROOT=$REPO \
    $PY -u ../evaluate_checkpoints_fineweb_dedup.py --scan-dir --groups 8-256-avgemb \
    --rank $i --world-size 4 > "$WORK/eval_r$i.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || { bash /root/.claude/notify.sh "avgemb: eval failed, see $WORK/eval_r*.log"; exit 1; }
log "eval done"

$PY -c "
import sys; sys.path.insert(0, '$REPO/experiment')
import merge_eval_results as m
from pathlib import Path
m.HERE = Path('$WORK'); m.TARGET = Path('$REPO/experiment/val_metrics.json')
sys.exit(0 if m.merge_results() else 1)
" || { bash /root/.claude/notify.sh "avgemb: merge aborted; nothing written."; exit 1; }

mkdir -p "$(dirname "$FIG")"
SUMMARY=$($PY "$WT/experiment/avgemb/plot_avgemb.py" "$FIG" 2>&1 | tail -4)
log "$SUMMARY"
bash /root/.claude/notify.sh "avgemb, inverse of the post-hoc experiment: a trained c=8 model (seed 66, step 888k) with each token's rows averaged across the 8 vocabularies, then 30k more steps with optimizer state. Mean per-compartment val loss by steps since the intervention:
$SUMMARY" "$FIG.png"
log done
