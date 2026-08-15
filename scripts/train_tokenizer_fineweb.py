#!/usr/bin/env python3
"""Train a BPE tokenizer directly on raw English FineWeb parquet, with a manifest.

This exists alongside `scripts/train_tokenizer.py`, which is kept as-is for the
existing runs. The difference is provenance, and it is the whole point:

    old:  FineWeb -> (unrecorded tokenization) -> DEDUPLICATED .bin
                  -> decode back to text with Qwen2.5-72B -> train BPE
    new:  FineWeb parquet (text column) -> train BPE

The old chain cannot be shown to a reader. It round-trips through a corpus whose
construction no script in this repo reproduces, and the resulting
`tokenizers/bpe-16384` carries no record of which steps it went through. Training
on text straight from the published parquet removes every intermediate step, and
the manifest written next to the tokenizer pins the rest: exact source files with
their sha256, the character budget, the trainer settings, and the git SHA.

Deliberately NOT deduplicated. Dedup belongs downstream as a measured condition,
if at all — baking it into the tokenizer's training distribution makes it
invisible and unremovable.

    python3 scripts/train_tokenizer_fineweb.py \
        --parquet-dir /mnt/pccfs2/backed_up/datasets/fineweb/sample-350BT/sample/350BT \
        --vocab-size 16384 \
        --output-dir tokenizers/bpe-16384-fineweb1

Reproducing it later means running the same command: the manifest records which
files were consumed, in order, and how much of each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm

# Same architecture as scripts/train_tokenizer.py, so the new tokenizer differs
# from the old one in TRAINING DATA ONLY and the two stay comparable.
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
MIN_FREQUENCY = 2


def sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def text_iterator(files: list[str], max_chars: int, text_column: str, stats: dict):
    """Yield document text in file order until max_chars is reached.

    File order is the sorted order recorded in the manifest, and the cap is on
    characters rather than documents so the budget does not drift with document
    length. Both are what make a re-run land on the same tokenizer.
    """
    seen = 0
    for path in tqdm(files, desc="parquet files"):
        if seen >= max_chars:
            break
        pf = pq.ParquetFile(path)
        consumed_here = 0
        for batch in pf.iter_batches(batch_size=8192, columns=[text_column]):
            for text in batch.column(text_column).to_pylist():
                if not text:
                    continue
                yield text
                seen += len(text)
                consumed_here += 1
                if seen >= max_chars:
                    break
            if seen >= max_chars:
                break
        stats["files_consumed"].append({"file": os.path.basename(path), "docs": consumed_here})
    stats["total_chars"] = seen


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet-dir", required=True, help="directory of FineWeb .parquet files")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--output-dir", required=True,
                   help="new directory; refuses to overwrite an existing tokenizer")
    p.add_argument("--max-chars", type=int, default=5_000_000_000,
                   help="character budget for BPE training (default 5e9, ~5GB of text)")
    p.add_argument("--max-files", type=int, default=None,
                   help="cap on parquet files consumed (default: as many as the budget needs)")
    p.add_argument("--text-column", default="text")
    p.add_argument("--skip-hashes", action="store_true",
                   help="skip sha256 of source files (faster, weaker manifest)")
    args = p.parse_args()

    out = Path(args.output_dir)
    if (out / "tokenizer.json").exists():
        raise SystemExit(
            f"{out}/tokenizer.json already exists. This script never overwrites a "
            f"tokenizer — existing runs depend on theirs. Choose a new --output-dir."
        )

    files = sorted(str(p_) for p_ in Path(args.parquet_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet files in {args.parquet_dir}")
    if args.max_files:
        files = files[: args.max_files]
    print(f"{len(files)} parquet files available in sorted order")

    stats: dict = {"files_consumed": [], "total_chars": 0}

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=MIN_FREQUENCY,
    )

    print(f"training vocab_size={args.vocab_size} on up to {args.max_chars:,} chars")
    tokenizer.train_from_iterator(
        text_iterator(files, args.max_chars, args.text_column, stats), trainer=trainer
    )

    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>",
    )
    hf.save_pretrained(str(out))

    consumed = [f["file"] for f in stats["files_consumed"]]
    manifest = {
        "created_by": "scripts/train_tokenizer_fineweb.py",
        "command": " ".join(sys.argv),
        "git_sha": git_sha(repo),
        "source": {
            "dataset": "FineWeb (English) raw parquet",
            "parquet_dir": os.path.abspath(args.parquet_dir),
            "files_available": len(files),
            "files_consumed": stats["files_consumed"],
            "text_column": args.text_column,
            "deduplicated": False,
        },
        "budget": {"max_chars": args.max_chars, "chars_consumed": stats["total_chars"]},
        "trainer": {
            "model": "BPE",
            "vocab_size": args.vocab_size,
            "min_frequency": MIN_FREQUENCY,
            "special_tokens": SPECIAL_TOKENS,
            "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        },
        "artifact_sha256": {"tokenizer.json": sha256_file(str(out / "tokenizer.json"))},
    }
    if not args.skip_hashes:
        print("hashing source files consumed...")
        manifest["source"]["sha256"] = {
            os.path.basename(f): sha256_file(f)
            for f in tqdm(files) if os.path.basename(f) in set(consumed)
        }

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nsaved tokenizer + manifest to {out}")
    print(f"  files consumed : {len(consumed)}")
    print(f"  chars consumed : {stats['total_chars']:,}")
    t = "Hello, world! This is a test of the new tokenizer."
    print(f"  roundtrip ok   : {hf.decode(hf.encode(t)) == t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
