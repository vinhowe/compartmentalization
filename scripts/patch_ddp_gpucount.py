"""Fix ddp_train.sbatch to use the ALLOCATED gpu count, not the node's.

`torchrun --nproc_per_node="$SLURM_GPUS_ON_NODE"` is wrong on a shared node:
SLURM_GPUS_ON_NODE reports how many GPUs the *node* has, not how many this job
was granted. A 4-GPU job on a 7-GPU node therefore spawns 7 ranks, four of which
land on devices another job already holds, and the run dies with

    CUDA error: CUDA-capable device(s) is/are busy or unavailable

This never showed up before because every DDP job so far asked for --gpus=8 and
got a whole node. It surfaced the moment we started packing 4-GPU jobs onto
shared nodes, where exactly one job at a time survived.

CUDA_VISIBLE_DEVICES is set by Slurm to precisely our allocation, so counting
its entries is the correct source of truth.
"""
from pathlib import Path
import sys

p = Path(sys.argv[1] if len(sys.argv) > 1 else "slurm/ddp_train.sbatch")
s = p.read_text()

OLD = 'torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" \\'
NEW = '''# SLURM_GPUS_ON_NODE is the NODE's gpu count, not this job's allocation. On a
# shared node that spawns ranks for GPUs we do not own -> "CUDA-capable
# device(s) is/are busy or unavailable". CUDA_VISIBLE_DEVICES is exactly our
# allocation, so count that instead.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\\n' | grep -c .)
else
    NGPU=${SLURM_GPUS_ON_NODE:-1}
fi
echo "[ddp-train] NGPU=$NGPU (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
torchrun --standalone --nproc_per_node="$NGPU" \\'''

if 'NGPU=' in s:
    print("already patched")
elif OLD not in s:
    print("ANCHOR NOT FOUND — not patching")
    sys.exit(1)
else:
    p.write_text(s.replace(OLD, NEW, 1))
    print("patched", p)
