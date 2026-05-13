#!/usr/bin/env python3
"""Prepare Wikipedia (EN + ZH) tokenized with Qwen3, streaming-write to shards.

Outputs (interleaved by alternating shard names so glob+sort gives 50/50 mix):
- data/wiki-mixed-shared-qwen3/wiki_train_NNNNNN_{en,zh}.bin   (no offset)
- data/wiki-mixed-compartmented-qwen3/wiki_train_NNNNNN_{en,zh}.bin   (ZH +V)
- data/wiki-mixed-shared-qwen3/wiki_val_NNNNNN_{en,zh}.bin
- data/wiki-mixed-compartmented-qwen3/wiki_val_NNNNNN_{en,zh}.bin
- data/wiki-en-qwen3/wiki_val_NNNNNN.bin   (per-language val for post-hoc eval)
- data/wiki-zh-qwen3/wiki_val_NNNNNN.bin

Token format: uint32, 1024-byte header (magic 20251013, version, ntok in header[0..2]).
EOS: <|endoftext|> id 151643 (Qwen3 doc separator).

Strategy: streaming write — no in-memory accumulation. Cap article length at
MAX_ARTICLE_TOKENS so a runaway article doesn't blow up memory or stall the
tokenizer. Use fast tokenizer + batched encoding.

Usage:
    .venv/bin/python3 -u scripts/prepare_wiki_qwen3.py
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MAGIC = 20251013
SHARD_TOKENS = 50_000_000
EOS_ID = 151643
TOKENIZER_NAME = "Qwen/Qwen3-8B"
TOKENIZER_VOCAB = 151_936  # round-up of 151,669 to nearest 256
MAX_ARTICLE_TOKENS = 100_000  # truncate single articles past this
ENCODE_BATCH = 256  # tokenizer.batch_encode_plus batch size


def write_shard(path: Path, tokens: np.ndarray) -> None:
    arr = tokens.astype(np.uint32)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(arr)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


class ShardWriter:
    """Buffer tokens to a fixed shard size, write+rotate when full."""

    def __init__(self, out_dir: Path, prefix: str, lang_suffix: str):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.lang_suffix = lang_suffix
        self.buf: list[int] = []
        self.shard_idx = 0
        self.total_written = 0

    def add(self, tokens: list[int]) -> None:
        self.buf.extend(tokens)
        while len(self.buf) >= SHARD_TOKENS:
            chunk = np.array(self.buf[:SHARD_TOKENS], dtype=np.uint32)
            fname = self.out_dir / f"{self.prefix}_{self.shard_idx:06d}_{self.lang_suffix}.bin"
            write_shard(fname, chunk)
            self.total_written += len(chunk)
            self.buf = self.buf[SHARD_TOKENS:]
            self.shard_idx += 1

    def flush(self) -> None:
        if self.buf:
            chunk = np.array(self.buf, dtype=np.uint32)
            fname = self.out_dir / f"{self.prefix}_{self.shard_idx:06d}_{self.lang_suffix}.bin"
            write_shard(fname, chunk)
            self.total_written += len(chunk)
            self.buf = []
            self.shard_idx += 1


def stream_tokenize_lang(
    config: str,
    tokenizer,
    lang_suffix: str,
    shared_train_writer: ShardWriter,
    shared_val_writer: ShardWriter,
    comp_train_writer: ShardWriter,
    comp_val_writer: ShardWriter,
    lang_only_val_writer: ShardWriter,
    offset: int,
    val_fraction: float = 0.01,
    max_train_tokens: int | None = None,
    seed: int = 42,
    log_every: int = 10_000,
) -> tuple[int, int, int]:
    """Stream Wikipedia, tokenize in batches, write directly to shards.

    For each article:
      - if val: write to {shared_val, comp_val (with offset), lang_only_val}
      - else (train): write to {shared_train, comp_train (with offset)}
                       and stop pushing train if max_train_tokens reached.

    Returns (n_train, n_val, n_articles_processed).
    """
    print(f"  Streaming {config} ...")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    rng = random.Random(seed)

    n_train = 0
    n_val = 0
    n_articles = 0
    n_truncated = 0
    t0 = time.monotonic()

    batch_texts: list[str] = []
    batch_is_val: list[bool] = []

    def flush_batch():
        nonlocal n_train, n_val, n_truncated
        if not batch_texts:
            return
        encs = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for toks, is_val in zip(encs, batch_is_val):
            if len(toks) > MAX_ARTICLE_TOKENS:
                toks = toks[:MAX_ARTICLE_TOKENS]
                n_truncated += 1
            toks = list(toks) + [EOS_ID]
            offset_toks = [t + offset for t in toks]
            if is_val:
                shared_val_writer.add(toks)
                comp_val_writer.add(offset_toks)
                lang_only_val_writer.add(toks)  # raw ids (no offset) for per-lang val
                n_val += len(toks)
            else:
                if max_train_tokens is None or n_train < max_train_tokens:
                    shared_train_writer.add(toks)
                    comp_train_writer.add(offset_toks)
                    n_train += len(toks)
                # if past cap: drop train tokens silently
        batch_texts.clear()
        batch_is_val.clear()

    for article in ds:
        text = (article.get("text") or "").strip()
        if not text:
            continue
        is_val = rng.random() < val_fraction
        batch_texts.append(text)
        batch_is_val.append(is_val)
        n_articles += 1
        if len(batch_texts) >= ENCODE_BATCH:
            flush_batch()
        if n_articles % log_every == 0:
            elapsed = time.monotonic() - t0
            rate = n_articles / max(elapsed, 1)
            print(f"    {lang_suffix}: {n_articles:>9d} articles  train={n_train:>14,} tok  val={n_val:>10,} tok  rate={rate:>5.0f} art/s  trunc={n_truncated}")

        # Stop streaming entirely when we've capped train and seen ~10x val budget worth of articles
        if max_train_tokens is not None and n_train >= max_train_tokens and n_articles % log_every == 0:
            # Stop early to avoid pulling unnecessary EN articles
            break

    flush_batch()
    elapsed = time.monotonic() - t0
    print(f"  Done {lang_suffix}: {n_articles} articles, {n_train:,} train, {n_val:,} val ({elapsed:.0f}s, {n_truncated} truncated)")
    return n_train, n_val, n_articles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh_config", default="20231101.zh")
    ap.add_argument("--en_config", default="20231101.en")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading Qwen3 tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    print(f"  vocab_size: {tokenizer.vocab_size}, full len: {len(tokenizer)}, fast: {tokenizer.is_fast}")

    data_dir = Path(args.data_dir)
    shared_dir = data_dir / "wiki-mixed-shared-qwen3"
    comp_dir = data_dir / "wiki-mixed-compartmented-qwen3"
    en_only_dir = data_dir / "wiki-en-qwen3"
    zh_only_dir = data_dir / "wiki-zh-qwen3"

    # ZH first
    print("\n[1/2] ZH wikipedia (no train cap)...")
    zh_shared_train = ShardWriter(shared_dir, "wiki_train", "zh")
    zh_shared_val = ShardWriter(shared_dir, "wiki_val", "zh")
    zh_comp_train = ShardWriter(comp_dir, "wiki_train", "zh")
    zh_comp_val = ShardWriter(comp_dir, "wiki_val", "zh")
    zh_only_val = ShardWriter(zh_only_dir, "wiki_val", "")  # raw zh ids, val-only

    zh_n_train, zh_n_val, _ = stream_tokenize_lang(
        args.zh_config, tokenizer, "zh",
        zh_shared_train, zh_shared_val,
        zh_comp_train, zh_comp_val,
        zh_only_val,
        offset=TOKENIZER_VOCAB,
        seed=args.seed,
    )
    for w in (zh_shared_train, zh_shared_val, zh_comp_train, zh_comp_val, zh_only_val):
        w.flush()

    # EN, capped at ZH train tokens
    print(f"\n[2/2] EN wikipedia (cap train at {zh_n_train:,} tokens)...")
    en_shared_train = ShardWriter(shared_dir, "wiki_train", "en")
    en_shared_val = ShardWriter(shared_dir, "wiki_val", "en")
    en_comp_train = ShardWriter(comp_dir, "wiki_train", "en")
    en_comp_val = ShardWriter(comp_dir, "wiki_val", "en")
    en_only_val = ShardWriter(en_only_dir, "wiki_val", "")

    en_n_train, en_n_val, _ = stream_tokenize_lang(
        args.en_config, tokenizer, "en",
        en_shared_train, en_shared_val,
        en_comp_train, en_comp_val,
        en_only_val,
        offset=0,
        seed=args.seed + 1,  # different seed for val sampling
        max_train_tokens=zh_n_train,
    )
    for w in (en_shared_train, en_shared_val, en_comp_train, en_comp_val, en_only_val):
        w.flush()

    # Summary
    total_shared_train = zh_shared_train.total_written + en_shared_train.total_written
    n_steps = total_shared_train // (256 * 512)
    print("\n=== SUMMARY ===")
    print(f"ZH train: {zh_n_train:,}  EN train: {en_n_train:,}")
    print(f"Mixed train (shared+comp identical token count): {total_shared_train:,}")
    print(f"At batch 256 × seq 512 = 131,072 tok/step:  step budget = {n_steps:,}")
    print(f"Vocab: shared = {TOKENIZER_VOCAB}, compartmented = {2 * TOKENIZER_VOCAB}")
    print(f"\nShard layout (sorted glob alternates en/zh):")
    print(f"  {shared_dir}/wiki_train_*.bin")
    print(f"  {comp_dir}/wiki_train_*.bin")


if __name__ == "__main__":
    main()
