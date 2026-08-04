#!/bin/bash
# Fan the FineWeb val eval out across sivri's idle V100s.
#
# WHY V100 IS OK HERE, despite the standing "never compare V100 to A100" rule:
# that rule is about *training* (bf16 tensor cores, and the project .venv's
# nightly torch needing CC>=7.5). evaluate_checkpoints_fineweb_dedup.py has
# ZERO autocast calls and recomputes loss in float32, so the forward is fp32
# end-to-end and the numbers are architecture-independent. The bio eval is the
# opposite -- it runs autocast(bfloat16) -- and must NOT come here.
#
# Must use the container's base torch (2.5.1+cu121, arch_list includes sm_70).
# The project .venv symlinks python to /root/.local/share/uv/... which does not
# exist outside the dw-2-1 container, and its nightly torch has no sm_70 kernels.
#
# SHARD COLLISION: workers write val_metrics_gpu<rank>.json into the SHARED
# pccfs2 experiment/ dir. Never run this while another eval holds an overlapping
# rank. Merging while shards are in flight is now safe-ish -- merge_eval_results
# unions rather than replaces, and refuses to write if any run would lose
# checkpoints -- but a half-written shard is still a half-written shard. Shards
# are archived (not deleted) only on --cleanup, and only if untouched for 30min.
set -uo pipefail

REMOTE=${REMOTE:-remote@sivri}
CONTAINER=${CONTAINER:-vin.pytorch-v100}
REPO=/mnt/pccfs2/backed_up/vin/dev/translation-compression
WORKDIR=$REPO/experiment
# NOT named GROUPS: that is a bash built-in readonly array of the caller's group
# ids, so assigning to it silently does nothing and $GROUPS expands to "0".
# This shipped once and launched six workers with `--groups 0`.
EVAL_GROUPS=${EVAL_GROUPS:-8-256-reseed}
# Verify with `gom show` on sivri immediately before launching -- availability
# there moves fast and these are shared with other users.
GPUS=${GPUS:-2 3 4 5 9 11}

SSHOPTS="-F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i /root/.ssh/id_ed25519"

read -ra GPU_ARR <<< "$GPUS"
WORLD=${#GPU_ARR[@]}

echo "[launch] $WORLD workers on $REMOTE ($CONTAINER), groups=$EVAL_GROUPS, gpus=${GPU_ARR[*]}"
rank=0
for g in "${GPU_ARR[@]}"; do
    # PYTHONPATH=$REPO is required: this runs the CONTAINER's python, not the
    # project .venv, so the repo root is not otherwise importable and
    # eval_utils' `from src.config.job_config import ...` fails.
    ssh $SSHOPTS "$REMOTE" "docker exec -d -e CUDA_VISIBLE_DEVICES=$g -e PYTHONPATH=$REPO -w $WORKDIR $CONTAINER \
        bash -c 'WANDB_MODE=offline python3 -u evaluate_checkpoints_fineweb_dedup.py \
            --scan-dir --groups $EVAL_GROUPS --rank $rank --world-size $WORLD \
            > eval_sivri_r${rank}.log 2>&1'"
    echo "  rank $rank -> gpu $g  (rc=$?)"
    rank=$((rank + 1))
done
echo "[launch] done; tail experiment/eval_sivri_r*.log to follow"
