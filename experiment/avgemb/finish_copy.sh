#!/bin/bash
# Run inside the dw-2-2 container. Launch copywte once the avgemb pipeline frees GPU 4,
# wait for both copy arms to reach step 918000, then eval -> merge -> combined plot -> Telegram.
set -u
REPO=/mnt/pccfs2/backed_up/vin/dev/translation-compression
PY=$REPO/.venv/bin/python
OUT=$REPO/out/translation-compression/8-256-avgemb-copy
WT=$REPO/.claude/worktrees/capacity-use
WORK=$REPO/experiment/.eval_avgemb_copy
AVGLOG=/mnt/pccfs2/backed_up/vin/dev/tc-avgemb-f718f07/logs/finish_avgemb.log
FIG=$WT/experiment/avgemb/figures/avgemb_all
log() { echo "[$(date -u '+%F %T')] $*"; }

# 1. copywte starts on GPU 4 after the avgemb pipeline is finished (success or failure), max 6h.
for i in $(seq 1 72); do
  grep -qE "\] done|failed|aborted|stopped" "$AVGLOG" 2>/dev/null && break
  sleep 300
done
OUTROOT=$OUT bash /mnt/pccfs2/backed_up/vin/dev/tc-avgemb-f718f07/launch_avgemb.sh 4:8-256-n8-tr01comp-s66-copywte
log "copywte launched"

# 2. Wait for both copy arms.
for i in $(seq 1 120); do
  missing=0
  for a in copyboth copywte; do [ -f "$OUT/8-256-n8-tr01comp-s66-$a/checkpoints/step-918000/model.pt" ] || missing=$((missing+1)); done
  [ "$missing" -eq 0 ] && break
  sleep 300
done
if [ "$missing" -ne 0 ]; then bash /root/.claude/notify.sh "avgemb copy arms: $missing never reached step 918000; pipeline stopped."; exit 1; fi
sleep 120

# 3. Eval + merge.
mkdir -p "$WORK"; ln -sfn "$REPO/experiment/val_metrics.json" "$WORK/val_metrics.json"; cd "$WORK"
pids=(); i=0
for g in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g WANDB_MODE=offline PYTHONPATH=$REPO TC_STORAGE_ROOT=$REPO \
    $PY -u ../evaluate_checkpoints_fineweb_dedup.py --scan-dir --groups 8-256-avgemb-copy \
    --rank $i --world-size 4 > "$WORK/eval_r$i.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || { bash /root/.claude/notify.sh "avgemb copy arms: eval failed, see $WORK/eval_r*.log"; exit 1; }
$PY -c "
import sys; sys.path.insert(0, '$REPO/experiment')
import merge_eval_results as m
from pathlib import Path
m.HERE = Path('$WORK'); m.TARGET = Path('$REPO/experiment/val_metrics.json')
sys.exit(0 if m.merge_results() else 1)
" || { bash /root/.claude/notify.sh "avgemb copy arms: merge aborted; nothing written."; exit 1; }

# 4. Combined plot of all five arms.
SUMMARY=$($PY "$WT/experiment/avgemb/plot_avgemb.py" "$FIG" 2>&1 | tail -6)
log "$SUMMARY"
bash /root/.claude/notify.sh "avgemb, all five arms: averaging vs copying compartment 1's rows into every compartment of a trained c=8 model, then 30k steps with optimizer state. Mean per-compartment val loss by steps since the intervention:
$SUMMARY" "$FIG.png"
log done
