#!/bin/bash
#SBATCH --job-name=tc-eval-reseed
#SBATCH --output=/grphome/grp_pccl/vin/dev/translation-compression/logs/pool/evalreseed-%j.out
#SBATCH --error=/grphome/grp_pccl/vin/dev/translation-compression/logs/pool/evalreseed-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-1,cs-1-2,m13l-2-1,dw-2-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
set -uo pipefail
R=/grphome/grp_pccl/vin/dev/translation-compression
Q=/nobackup/autodelete/grp/grp_pccl/vin/tc-8-256-reseed/queue
cd "$R"
source .supercomputer-venv/bin/activate
export TC_STORAGE_ROOT="$R"

# Wait for training to drain, so we evaluate the step-1000000 checkpoints rather
# than racing them. Bounded so a stuck run cannot hold the eval forever.
for i in $(seq 1 180); do
    p=$(ls "$Q/pending" 2>/dev/null | wc -l); r=$(ls "$Q/running" 2>/dev/null | wc -l)
    echo "[eval] waiting: pending=$p running=$r"
    [ "$p" -eq 0 ] && [ "$r" -eq 0 ] && break
    sleep 60
done

# The reseed runs live in the tc-8-256-reseed out-dir, which is NOT under
# $R/out, so point the scanner at it explicitly.
export TC_OUT_OVERRIDE=/nobackup/autodelete/grp/grp_pccl/vin/tc-8-256-reseed/out
ln -sfn "$TC_OUT_OVERRIDE" "$R/out/8-256-reseed-orc" 2>/dev/null || true

cd experiment
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank WANDB_MODE=offline PYTHONPATH=$R TC_STORAGE_ROOT=$R \
        python -u evaluate_checkpoints_fineweb_dedup.py --scan-dir \
        --groups 8-256-reseed-orc --rank $rank --world-size 4 \
        > "$R/logs/pool/evalreseed_r${rank}.log" 2>&1 &
done
wait
echo "[eval] done"
