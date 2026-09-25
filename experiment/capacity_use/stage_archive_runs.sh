#!/bin/bash
# The dw-2-2 container cannot see /mnt/pccfs2/backed_up/vin/vault, where
# out/translation-compression/synthetic-compartment-baselines is symlinked. Copy the
# config and step-1M weights of the three runs this analysis needs next to the code,
# where compute_capacity_use.py's STAGED fallback root finds them.
set -eu
A=/mnt/pccfs2/backed_up/vin/vault/tc-archive/synthetic-compartment-baselines
S="$(dirname "$0")/staged_runs/synthetic-compartment-baselines"
for d in \
  2026-03-06T18-11-45Z__english-baseline-rope-bpe16384-8-256__2df56182__s64__4b68526__51c738c2 \
  2026-03-05T22-39-16Z__english-frequency-2comp-rope-bpe16384-8-256__605a1512__s64__4b68526__2acd312f \
  2026-03-05T22-39-06Z__english-uniform-2comp-rope-bpe16384-8-256__11b3d274__s64__4b68526__a6d73c34; do
  mkdir -p "$S/$d/meta" "$S/$d/checkpoints/step-1000000"
  cp "$A/$d/meta/config.json" "$S/$d/meta/"
  cp "$A/$d/checkpoints/step-1000000/model.pt" "$S/$d/checkpoints/step-1000000/"
done
du -sh "$S"
