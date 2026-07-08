"""Download {python,javascript,go,rust,c,scala} from bigcode's Stack v1 or
starcoderdata, apply lightweight filtering, and write to compressed JSONL
shards.

Sources (--source):
  stack (default):  bigcode/the-stack-dedup, data_dir="data/<lang>"
  starcoder:        bigcode/starcoderdata,   data_dir="<lang>"
Both bundle content directly (unlike Stack v2, which is SWHIDs-only and
requires SWH resolution).

Runs streaming (no local landing of the raw dataset), applies:
 - size: 200 <= len(bytes) <= 1_000_000
 - line-length: max_line_length <= 1000  (strip minified)
 - alnum ratio: alnum/total >= 0.25       (strip binary/base64 blobs)
 - path blacklist: /vendor/, /node_modules/, .min.js, /third_party/, /dist/

Writes to <OUT_ROOT>/filtered/<lang>/shard-XXXXX.jsonl.gz. Each line is JSON:
  {"text": "...", "path": "...", "size": ...}

Stops after TARGET_BYTES per language (default 25 GB, enough for ~7B GPT-2-BPE
tokens — we'll pick the exact target after seeing token counts from a
tokenizer sample).
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset

BLACKLIST_SUBSTRINGS = (
    "/vendor/", "/node_modules/", "/third_party/", "/thirdparty/",
    "/dist/", "/build/", "/target/", "/.min.js", "/.min.css",
    ".min.js", ".bundle.js", "-min.js", "-bundle.js",
)


def keep(row: dict) -> bool:
    """Return True if row survives all filters."""
    content = row.get("content")
    if not isinstance(content, str):
        return False
    n = len(content)
    if n < 200 or n > 1_000_000:
        return False
    path = row.get("max_stars_repo_path") or ""
    lower = path.lower()
    for s in BLACKLIST_SUBSTRINGS:
        if s in lower:
            return False
    # max line length: strip minified
    max_line = 0
    for line in content.splitlines():
        if len(line) > max_line:
            max_line = len(line)
            if max_line > 1000:
                return False
    # alnum ratio
    alnum = sum(1 for c in content if c.isalnum())
    if alnum / max(n, 1) < 0.25:
        return False
    return True


def process_lang(lang: str, out_root: Path, target_bytes: int, shard_bytes: int, log_every: int, source: str = "stack"):
    out_dir = out_root / "filtered" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    if source == "stack":
        ds_name = "bigcode/the-stack-dedup"
        data_dir = f"data/{lang}"
    elif source == "starcoder":
        ds_name = "bigcode/starcoderdata"
        data_dir = lang
    else:
        raise ValueError(f"unknown source: {source}")
    print(f"[{lang}] loading {ds_name} data_dir={data_dir} (streaming)...", flush=True)
    ds = load_dataset(
        ds_name,
        data_dir=data_dir,
        split="train",
        streaming=True,
    )
    shard_idx = 0
    bytes_written = 0
    docs_written = 0
    docs_seen = 0
    t0 = time.time()

    def open_shard(idx):
        return gzip.open(out_dir / f"shard-{idx:05d}.jsonl.gz", "wt", compresslevel=3)

    fp = open_shard(shard_idx)
    shard_bw = 0
    for row in ds:
        docs_seen += 1
        if not keep(row):
            continue
        rec = {
            "text": row["content"],
            "path": row.get("max_stars_repo_path", ""),
            "size": len(row["content"]),
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        b = line.encode("utf-8")
        fp.write(line)
        shard_bw += len(b)
        bytes_written += len(row["content"])  # count filtered-text bytes, not gzip-compressed
        docs_written += 1
        if shard_bw >= shard_bytes:
            fp.close()
            shard_idx += 1
            fp = open_shard(shard_idx)
            shard_bw = 0
        if docs_written % log_every == 0:
            dt = time.time() - t0
            keep_rate = docs_written / max(docs_seen, 1)
            gb = bytes_written / 1024 ** 3
            print(
                f"[{lang}] docs seen={docs_seen:,} kept={docs_written:,} "
                f"({keep_rate:.1%}) bytes={gb:.2f} GB "
                f"rate={docs_written/max(dt,1):.0f} doc/s "
                f"shards={shard_idx+1}",
                flush=True,
            )
        if bytes_written >= target_bytes:
            print(f"[{lang}] hit target_bytes={target_bytes:,}; stopping", flush=True)
            break
    fp.close()
    dt = time.time() - t0
    print(
        f"[{lang}] DONE docs_seen={docs_seen:,} kept={docs_written:,} "
        f"bytes={bytes_written/1024**3:.2f} GB shards={shard_idx+1} "
        f"took {dt/60:.1f} min",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lang", choices=["python", "javascript", "go", "rust", "c", "scala"], required=True)
    ap.add_argument("--source", choices=["stack", "starcoder"], default="stack",
                    help="stack = bigcode/the-stack-dedup (v1); "
                         "starcoder = bigcode/starcoderdata (v1-derived, better-filtered)")
    ap.add_argument("--target-gb", type=float, default=100.0,
                    help="Filtered-text bytes to collect before stopping (default 100)")
    ap.add_argument("--shard-mb", type=int, default=256,
                    help="Approx uncompressed jsonl bytes per shard")
    ap.add_argument("--log-every", type=int, default=5000)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    target_bytes = int(args.target_gb * 1024 ** 3)
    shard_bytes = args.shard_mb * 1024 * 1024
    process_lang(args.lang, out_root, target_bytes, shard_bytes, args.log_every, args.source)


if __name__ == "__main__":
    main()
