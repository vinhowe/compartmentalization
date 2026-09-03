#!/bin/bash
# submit_chain.sh CONFIG TARGET[dw|cs] [N_CHUNKS=3] [EXISTING_OUT_DIR]
#
# Submits N chained sbatch chunks (each 7d), each depending on the prior
# completing for any reason. If EXISTING_OUT_DIR is given, resumes there.
# Otherwise creates a fresh deterministic path under TC_STORAGE_ROOT.
set -euo pipefail

CONFIG="${1:?config required}"
TARGET="${2:?target dw|cs required}"
N=${3:-3}
EXISTING="${4:-}"

if [ "$TARGET" = "dw" ]; then
    SBATCH=slurm/run_1b_ddp_chain_dw.sbatch
elif [ "$TARGET" = "cs" ]; then
    SBATCH=slurm/run_1b_ddp_chain_cs.sbatch
else
    echo "TARGET must be dw or cs" >&2; exit 1
fi

TC_ROOT="/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
S=/apps/slurm/23.11.1/bin

if [ -n "$EXISTING" ]; then
    OUT_DIR="$EXISTING"
    echo "Resuming existing OUT_DIR=$OUT_DIR"
else
    SLUG=$(basename "$CONFIG" .toml)
    TS=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
    HASH=$(cat "$CONFIG" | sha1sum | awk '{print substr($1,1,8)}')
    OUT_DIR="$TC_ROOT/out/translation-compression/1b-scale/${TS}__${SLUG}__${HASH}__s64__chain__phase1"
    mkdir -p "$OUT_DIR"
    echo "Fresh OUT_DIR=$OUT_DIR"
fi

PREV=""
for i in $(seq 1 "$N"); do
    if [ -z "$PREV" ]; then
        JID=$($S/sbatch --parsable "$SBATCH" "$CONFIG" "$OUT_DIR")
    else
        JID=$($S/sbatch --parsable --dependency=afterany:$PREV "$SBATCH" "$CONFIG" "$OUT_DIR")
    fi
    echo "  chunk $i: jobid=$JID dep=${PREV:-none}"
    PREV=$JID
done
echo "Last job: $PREV"
echo "OUT_DIR: $OUT_DIR"
