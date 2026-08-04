#!/usr/bin/env python3
"""Merge per-GPU eval shards into val_metrics.json by UNION.

Workers write val_metrics_gpu<rank>.json; this folds them into val_metrics.json.

THIS REPLACES AN EARLIER VERSION THAT MERGED BY REPLACEMENT, and the difference
matters enough to record. The old one did `merged[key] = value`: a shard's whole
record for a run overwrote the whole record in val_metrics.json. That is only
safe if every shard is strictly newer and strictly more complete than the file it
merges into, and neither holds:

  * Shards are long-lived and are only archived on an explicit --cleanup, so
    ranks from much older passes sit in the directory indefinitely. One holding a
    run at 400k steps silently overwrote the same run at 1e6.
  * A worker evaluates only the checkpoints that were missing, so its record for
    a run legitimately contains *fewer* checkpoints than the merged file has.

Run against a completed reseed evaluation, that took runs-at-1e6 from 38 down to
5. The damage is invisible: the file stays well-formed, every run is still
present, and only the depth of each trajectory quietly shrinks.

WHAT THIS DOES. Per run, build a step -> value map from the original, update it
with each shard, re-emit sorted arrays. Steps in both keep the shard's value (a
newer evaluation of the same checkpoint, which should agree anyway); steps in
either are kept. Metric arrays are index-aligned to `checkpoints`, so they are
zipped apart and back rather than concatenated; a record whose arrays disagree
with its checkpoint list is left alone and reported rather than guessed at.

Shards are found by glob, not by range(world_size) -- the old bound at 8 silently
ignored val_metrics_gpu10..21, so results from wider fan-outs were never merged.

SAFETY. No run may end with fewer checkpoints than it started with. That is
asserted before anything is written; if it trips, nothing is written at all.

Usage:
    python3 merge_eval_results.py             # merge
    python3 merge_eval_results.py --cleanup   # merge, then archive idle shards
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "val_metrics.json"

# A shard touched more recently than this probably belongs to a live worker.
LIVE_WINDOW = 30 * 60


def unpack(rec):
    """{checkpoints, metrics{name: [...]}} -> {step: {name: value}}, or None."""
    cps = rec.get("checkpoints")
    if not isinstance(cps, list):
        return None
    out = {}
    for i, s in enumerate(cps):
        row = {}
        for name, arr in (rec.get("metrics") or {}).items():
            if not isinstance(arr, list) or len(arr) != len(cps):
                return None          # ragged record: refuse to touch it
            row[name] = arr[i]
        out[s] = row
    return out


def repack(step_map, template):
    steps = sorted(step_map)
    names = sorted({n for row in step_map.values() for n in row})
    rec = dict(template)
    rec["checkpoints"] = steps
    packed = {n: [step_map[s][n] for s in steps if n in step_map[s]] for n in names}
    # Drop any metric not defined at every step: a short array silently
    # misaligns with `checkpoints` for every downstream consumer.
    rec["metrics"] = {n: v for n, v in packed.items() if len(v) == len(steps)}
    return rec


def merge_results(cleanup: bool = False) -> bool:
    merged = json.loads(TARGET.read_text())
    before = {k: len(v.get("checkpoints") or []) for k, v in merged.items()}
    ragged = []

    shards = sorted(HERE.glob("val_metrics_gpu*.json"))
    for shard in shards:
        data = json.loads(shard.read_text())
        touched = 0
        for key, rec in data.items():
            new = unpack(rec)
            if new is None:
                ragged.append((shard.name, key))
                continue
            if key in merged:
                cur = unpack(merged[key])
                if cur is None:
                    ragged.append((TARGET.name, key))
                    continue
                cur.update(new)
                merged[key] = repack(cur, merged[key])
            else:
                merged[key] = repack(new, rec)
            touched += 1
        print(f"  {shard.name:28s} {len(data):5d} runs, {touched} merged")

    regressed = [(k, before[k], len(merged[k].get("checkpoints") or []))
                 for k in before
                 if len(merged[k].get("checkpoints") or []) < before[k]]
    if regressed:
        print(f"\n  ABORT: {len(regressed)} runs would lose checkpoints; nothing written")
        for k, b, a in regressed[:10]:
            print(f"    {k[-60:]}  {b} -> {a}")
        return False
    if ragged:
        print(f"\n  {len(ragged)} ragged records left untouched")
        for s, k in ragged[:5]:
            print(f"    {s}: {k[-55:]}")

    shutil.copy2(TARGET, TARGET.with_suffix(".json.bak"))
    TARGET.write_text(json.dumps(merged, indent=4))
    grew = sum(1 for k in before if len(merged[k].get("checkpoints") or []) > before[k])
    print(f"\n  wrote {TARGET.name}: {len(merged)} runs, {grew} gained checkpoints, 0 regressed")

    if cleanup:
        archive = HERE / "merged_shards"
        archive.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        skipped = []
        print("\n  Retiring shards (archived, not deleted)...")
        for shard in shards:
            age = time.time() - shard.stat().st_mtime
            if age < LIVE_WINDOW:
                skipped.append((shard.name, int(age)))
                continue
            shard.replace(archive / f"{shard.stem}.{stamp}.json")
            print(f"    archived {shard.name}")
            bak = shard.with_suffix(".json.bak")
            if bak.exists():
                bak.unlink()
        if skipped:
            print("\n    LEFT IN PLACE (touched recently -- worker probably live):")
            for name, age in skipped:
                print(f"      {name} ({age}s ago)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true",
                    help="archive idle per-GPU shards after a successful merge")
    ap.add_argument("--world-size", type=int, default=None,
                    help="ignored; shards are discovered by glob")
    args = ap.parse_args()
    if args.world_size is not None:
        print("  note: --world-size is ignored; all val_metrics_gpu*.json are merged")
    merge_results(cleanup=args.cleanup)


if __name__ == "__main__":
    main()
