#!/bin/bash
# Auto-chain PD c=2 fresh jobs once they start.
JOBS="11723369 11723370"
S=/apps/slurm/23.11.1/bin
TC=/grphome/grp_pccl/vin/dev/translation-compression
LOG_DIR=$TC/logs
cd $TC

while [ -n "$JOBS" ]; do
    NEW_JOBS=""
    for j in $JOBS; do
        log=$LOG_DIR/1b-resume-7d-$j.out
        if [ -f $log ]; then
            OUT=$(grep -m1 "saving checkpoint to" $log 2>/dev/null | grep -oP '[^ ]+/checkpoints' | sed 's|/checkpoints||')
            CFG=$(grep -m1 "CONFIG=" $log 2>/dev/null | sed 's|.*CONFIG=||;s|  .*||;s| .*||')
            if [ -n "$OUT" ] && [ -n "$CFG" ]; then
                echo "$(date) chaining $j  CONFIG=$CFG  OUT=$OUT"
                J2=$($S/sbatch --parsable --dependency=afterany:$j slurm/run_1b_ddp_chain_dw.sbatch $CFG $OUT)
                J3=$($S/sbatch --parsable --dependency=afterany:$J2 slurm/run_1b_ddp_chain_dw.sbatch $CFG $OUT)
                echo "  chunk2=$J2  chunk3=$J3"
                continue  # don't re-add to NEW_JOBS
            fi
        fi
        NEW_JOBS="$NEW_JOBS $j"
    done
    JOBS="$NEW_JOBS"
    [ -n "$JOBS" ] && sleep 300
done
echo "$(date) All PD jobs chained"
