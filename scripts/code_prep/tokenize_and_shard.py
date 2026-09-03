"""Tokenize filtered JSONL shards with the joint BPE tokenizer and write
GPT-2-format .bin shards for training.

Each JSON record's text is encoded, prefixed with a doc-boundary token (the
tokenizer's <|endoftext|> id), and concatenated. Output shards contain
SHARD_TOKENS tokens each. The first SHARD_TOKENS_VAL tokens go to a val split.

Format matches data/data_common.py write_datafile(model_desc="gpt-2"):
  header[0]=20240520, header[1]=1, header[2]=ntok, then uint16 tokens.

Output layout: <OUT_ROOT>/bins/<lang>/{val_000000.bin, train_000000.bin, ...}
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SHARD_TOKENS = 100_000_000  # 100M tokens per train shard (400 MB uint32)

# Matches the format expected by train.py's DistributedDataLoader:
# magic=20251013, version=1, uint32 tokens after a 256*int32 header.
MAGIC = 20251013


def write_datafile(path: str, tokens: list[int], _model_desc: str = "gpt-2") -> None:
    """Write tokens to disk in the 20251013 uint32 format expected by train.py.
    (The `_model_desc` arg is kept for signature compatibility with earlier
    versions of this script.)"""
    arr = np.asarray(tokens, dtype=np.uint32)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


def process_lang(lang: str, out_root: Path, tokenizer_path: Path, val_tokens: int, target_tokens: int | None):
    out_dir = out_root / "bins" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{lang}] loading tokenizer...", flush=True)
    tok = Tokenizer.from_file(str(tokenizer_path))
    boundary_id = tok.token_to_id("<|endoftext|>")
    assert boundary_id is not None, "tokenizer missing <|endoftext|>"

    shards = sorted((out_root / "filtered" / lang).glob("shard-*.jsonl.gz"))
    print(f"[{lang}] {len(shards)} filtered shards; val_tokens={val_tokens:,}", flush=True)

    buf: list[int] = [boundary_id]  # start with a boundary
    total_written = 0
    docs = 0
    val_written = False
    train_shard_idx = 0
    t0 = time.time()

    def flush(force_val=False):
        nonlocal buf, total_written, val_written, train_shard_idx
        if force_val and not val_written:
            # Write exactly val_tokens to val_000000.bin
            if len(buf) < val_tokens:
                return  # not enough yet
            chunk = buf[:val_tokens]
            path = out_dir / "val_000000.bin"
            write_datafile(str(path), chunk, "gpt-2")
            buf = buf[val_tokens:]
            total_written += val_tokens
            val_written = True
            return
        if not val_written:
            return  # can't write train shards until val is out
        while len(buf) >= SHARD_TOKENS:
            chunk = buf[:SHARD_TOKENS]
            path = out_dir / f"train_{train_shard_idx:06d}.bin"
            write_datafile(str(path), chunk, "gpt-2")
            buf = buf[SHARD_TOKENS:]
            train_shard_idx += 1
            total_written += SHARD_TOKENS

    # Use fast batched encode
    BATCH = 256
    texts: list[str] = []
    for shard in shards:
        with gzip.open(shard, "rt") as fp:
            for line in fp:
                r = json.loads(line)
                texts.append(r["text"])
                docs += 1
                if len(texts) >= BATCH:
                    encoded = tok.encode_batch(texts)
                    for e in encoded:
                        buf.append(boundary_id)
                        buf.extend(e.ids)
                    texts = []
                    if not val_written and len(buf) >= val_tokens + SHARD_TOKENS:
                        flush(force_val=True)
                    flush()
                    if docs % (BATCH * 20) == 0:
                        dt = time.time() - t0
                        cur_total = total_written + len(buf)
                        print(
                            f"[{lang}] docs={docs:,} tokens={cur_total:,} "
                            f"train_shards={train_shard_idx} "
                            f"rate={cur_total/max(dt,1):.0f} tok/s "
                            f"({docs/max(dt,1):.0f} doc/s)",
                            flush=True,
                        )
                    if target_tokens is not None and total_written + len(buf) >= target_tokens:
                        print(f"[{lang}] hit target_tokens={target_tokens:,}; stopping", flush=True)
                        break
        if target_tokens is not None and total_written + len(buf) >= target_tokens:
            break

    # Flush remaining
    if texts:
        encoded = tok.encode_batch(texts)
        for e in encoded:
            buf.append(boundary_id)
            buf.extend(e.ids)
    if not val_written:
        flush(force_val=True)
    flush()
    # Emit tail as final train shard even if partial (matches fineweb pattern)
    if val_written and buf:
        path = out_dir / f"train_{train_shard_idx:06d}.bin"
        write_datafile(str(path), buf, "gpt-2")
        total_written += len(buf)
        train_shard_idx += 1

    dt = time.time() - t0
    print(
        f"[{lang}] DONE docs={docs:,} tokens={total_written:,} "
        f"train_shards={train_shard_idx} took {dt/60:.1f} min",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang", choices=["python", "javascript", "chinese", "go", "rust", "c", "scala"], required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val-tokens", type=int, default=100_000_000,
                    help="First N tokens go to val_000000.bin (default 100M)")
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="Optional stop-after target (default: consume all)")
    args = ap.parse_args()

    process_lang(args.lang, Path(args.out_root), Path(args.tokenizer), args.val_tokens, args.target_tokens)


if __name__ == "__main__":
    main()
