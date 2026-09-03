#!/bin/bash
# Four short c=8 runs to separate "real seed sensitivity" from "artifact of the
# pinned assignment horizon". Run sequentially in ONE allocation so they share
# identical hardware and no queue latency between them.
#
# Built-in validity check: legacy+s1024 must reproduce the existing 30B c=8 run
# (val 5.0154 @ step 500) and pinned+s1024 the fresh one (5.3567). If those two
# do not reproduce, the diagnostic itself is untrustworthy and nothing else here
# means anything.
#SBATCH --job-name=diag-c8-spread
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-1,cs-1-2,m13l-2-1,dw-2-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/nobackup/autodelete/grp/grp_pccl/vin/logs/diag-c8-%j.out
set -uo pipefail
ulimit -c 0
TRAIN_DIR=/grphome/grp_pccl/vin/dev/translation-compression
cd "$TRAIN_DIR"
source .supercomputer-venv/bin/activate
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0
for name in diag-c8-legacy-s1024 diag-c8-pinned-s1024 diag-c8-pinned-s1025 diag-c8-pinned-s1026; do
  echo "===== $name ====="
  export OUT_DIR=/nobackup/autodelete/grp/grp_pccl/vin/out/diag-c8-spread/$name
  export RUN_ID=$name
  mkdir -p "$OUT_DIR"
  srun --ntasks=1 torchrun --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
      train.py --job.config-file "config/$name.toml"
  echo "===== $name rc=$? ====="
done
