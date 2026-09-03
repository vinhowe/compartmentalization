"""Build document-level-interleaved joint bins from existing per-language bins.

Walks each per-language tokenized bin doc-by-doc (docs are demarcated by the
endoftext boundary token) and emits them round-robin into a new set of joint
shards. Each joint shard is a ~1/N slice of each lang in alternation.

When a lang's stream exhausts, the round-robin continues over the remaining
live langs. This matters at large budgets — e.g. a rust-limited 5-way run
past ~2.6B rust tokens will keep interleaving py/js/go/c only.

No re-tokenization is done — just reading uint32 tokens and rewriting them.

Output:
  <OUT_ROOT>/bins/joint_mixed/train_XXXXXX.bin    (uint32 magic=20251013)
  <OUT_ROOT>/bins/joint_mixed/val_<lang>.bin      (symlink per lang)
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

MAGIC = 20251013
SHARD_TOKENS = 100_000_000


def stream_docs(bin_files, boundary_id: int):
    """Yield lists of ints, one document at a time. A document ends at the
    boundary_id; the boundary itself is part of the yielded doc."""
    buf: list[int] = []
    for fname in bin_files:
        with open(fname, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            assert header[0] == MAGIC, f"bad magic in {fname}"
            ntok = int(header[2])
            tokens = np.frombuffer(f.read(ntok * 4), dtype=np.uint32)
        idxs = np.flatnonzero(tokens == boundary_id)
        prev = 0
        for i in idxs:
            doc = tokens[prev : i + 1]
            if buf:
                buf.extend(doc.tolist())
                yield buf
                buf = []
            else:
                yield doc.tolist()
            prev = i + 1
        if prev < len(tokens):
            buf.extend(tokens[prev:].tolist())
    if buf:
        yield buf


def write_shard(path: Path, tokens: list[int]) -> None:
    arr = np.asarray(tokens, dtype=np.uint32)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--boundary-id", type=int, default=0,
                    help="endoftext token id in the joint BPE (default 0)")
    ap.add_argument("--langs", default="python,javascript",
                    help="Comma-separated langs to interleave (default: 2-way py+js)")
    ap.add_argument("--out-subdir", default="joint_mixed",
                    help="Output subdir under <out_root>/bins/ (default: joint_mixed)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = out_root / "bins" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    print(f"langs: {langs}")

    # Open per-lang train streams
    streams = {}
    for lang in langs:
        files = sorted(glob.glob(str(out_root / "bins" / lang / "train_*.bin")))
        if not files:
            sys.exit(f"missing train shards for {lang} under {out_root}/bins/{lang}/")
        print(f"  {lang}: {len(files)} train shards")
        streams[lang] = stream_docs(files, args.boundary_id)

    live = {lang: True for lang in langs}
    doc_counts = {lang: 0 for lang in langs}
    buf: list[int] = []
    shard_idx = 0
    total_written = 0
    pbar = tqdm(unit="tok", unit_scale=True, desc="mixing")

    # Round-robin across live streams
    lang_cycle = langs[:]
    turn = 0
    while any(live.values()):
        lang = lang_cycle[turn % len(lang_cycle)]
        turn += 1
        if not live[lang]:
            continue
        try:
            doc = next(streams[lang])
            buf.extend(doc)
            doc_counts[lang] += 1
            pbar.update(len(doc))
        except StopIteration:
            live[lang] = False
            print(f"  [{lang}] exhausted after {doc_counts[lang]:,} docs")

        while len(buf) >= SHARD_TOKENS:
            chunk = buf[:SHARD_TOKENS]
            path = out_dir / f"train_{shard_idx:06d}.bin"
            write_shard(path, chunk)
            total_written += SHARD_TOKENS
            shard_idx += 1
            buf = buf[SHARD_TOKENS:]

    # Final partial shard
    if buf:
        path = out_dir / f"train_{shard_idx:06d}.bin"
        write_shard(path, buf)
        total_written += len(buf)
        shard_idx += 1

    pbar.close()
    print(f"docs merged: " + "  ".join(f"{lang}={doc_counts[lang]:,}" for lang in langs))
    print(f"total tokens: {total_written:,}  shards: {shard_idx}")

    # Val symlinks
    for lang in langs:
        val_files = sorted(glob.glob(str(out_root / "bins" / lang / "val_*.bin")))
        if val_files:
            link = out_dir / f"val_{lang}.bin"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(val_files[0])
            print(f"  symlinked val_{lang}.bin -> {val_files[0]}")


if __name__ == "__main__":
    main()
