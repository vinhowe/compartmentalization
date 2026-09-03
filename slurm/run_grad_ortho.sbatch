#!/bin/bash
#SBATCH --job-name=grad-ortho
#SBATCH --output=logs/grad-ortho-%j.out
#SBATCH --error=logs/grad-ortho-%j.err
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00

set -euo pipefail
mkdir -p logs

# Args:
#   $1 = run dir name (under 1b-scale/), e.g. "2026-04-09T18-51-52Z__1b-8comp-bpe16384__..."
#   $2 = step number, e.g. "60000"
#   $3 = optional pairs string, e.g. "0,1,2,3,4,5,6,7"
RUN_DIR="${1:?run dir required}"
STEP="${2:?step required}"
PAIRS="${3:-0,1}"

TRAIN_DIR="/grphome/grp_pccl/vin/dev/translation-compression"
STORAGE="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
EXP_DIR="$STORAGE/out/translation-compression/1b-scale/$RUN_DIR"
CKPT_DIR="$EXP_DIR/checkpoints/step-$(printf '%06d' "$STEP")"
OUT_FILE="$STORAGE/out/translation-compression/1b-scale/$RUN_DIR/grad_ortho_step$STEP.json"

cd "$TRAIN_DIR/experiment"
echo "== grad_ortho on $(hostname) =="
echo "EXP=$EXP_DIR"
echo "CKPT=$CKPT_DIR"
echo "PAIRS=$PAIRS"
nvidia-smi -L || true

export TC_STORAGE_ROOT="$STORAGE"
export PYTHONPATH="$TRAIN_DIR:${PYTHONPATH:-}"

source "$TRAIN_DIR/.supercomputer-venv/bin/activate"
which python3; python3 -V

python3 compartment_gradient_orthogonality.py \
    --checkpoint "$CKPT_DIR" \
    --experiment "$EXP_DIR" \
    --pairs "$PAIRS" \
    --batches 8 --batch-size 4 \
    --out "$OUT_FILE"

echo "Done. Output: $OUT_FILE"
