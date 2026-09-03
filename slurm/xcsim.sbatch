#!/bin/bash
#SBATCH --job-name=xcsim
#SBATCH --partition=cs3
#SBATCH --account=dw87
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/nobackup/autodelete/grp/grp_pccl/vin/logs/xcsim-%j.out
set -uo pipefail
ulimit -c 0
cd /grphome/grp_pccl/vin/dev/translation-compression
source .supercomputer-venv/bin/activate
B=/nobackup/autodelete/grp/grp_pccl/vin/out/redesign-1b-lrsweep
O=/nobackup/autodelete/grp/grp_pccl/vin/xcsim; mkdir -p "$O"
: "${TARGETS:?set TARGETS to 'rundir:checkpoint' entries, space separated}"
for spec in $TARGETS; do
  R="${spec%%:*}"; CK="${spec#*:}"
  echo "======== $(basename "$R") @ $CK ========"
  # c=1 has nothing to compare across compartments; the script refuses, which is
  # correct -- skip rather than record a failure.
  python3 experiment/crosscomp_similarity.py "$B/$R" --checkpoint "$CK" \
      --out_json "$O/$(basename "$R")__${CK//\//_}.json" 2>&1 \
    | grep -E "^canonical|^layer|^  compartment|wrote|Traceback|Error|nothing to compare"
done
