"""Prepare train/val bins for each cluster + matched random-partition control.

Inputs (produced by scripts/fineweb_cluster/cluster.py):
  <cluster-dir>/k<K>/assignments.jsonl.gz — one line per doc: {i, text_snip, cluster_id}
  (the i field indexes into the ORIGINAL streamed order; text_snip is truncated)

Since text_snip is only the first ~400 chars we can't tokenize from that. Instead
we RE-STREAM the same subset (same seed, same order) and pair with cluster IDs.

Steps:
  1. Retrain a fresh BPE 16384 on a sample of the streamed text
     (cluster.py used tf-idf; here we need a fresh BPE for the LM)
  2. Stream the same N docs from FineWeb (matching cluster.py's stream)
  3. Tokenize each doc
  4. For each K in --ks:
     - split docs into K semantic clusters (from assignments)
     - split docs into K random partitions (seeded, same sizes)
     - within each partition, take a val slice off the top, rest goes to train
     - concatenate token streams with a boundary token between docs
     - write bins in the magic 20251013 uint32 format used elsewhere

Output layout:
  <out-root>/tokenizer/joint_bpe16384.json
  <out-root>/bins/semantic/k<K>/cluster_<c>/{train,val}_000000.bin
  <out-root>/bins/random/k<K>/cluster_<c>/{train,val}_000000.bin
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np

NUM_TOKEN = "<NUM>"
_NUM_RE = re.compile(r"\d+")
MAGIC = 20251013


def normalize_text(t: str) -> str:
    return _NUM_RE.sub(NUM_TOKEN, t)


def stream_fineweb(n_docs: int, subset: str, min_chars: int, max_chars: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name=subset, split="train", streaming=True)
    kept = 0
    for row in ds:
        t = row.get("text", "")
        if not isinstance(t, str):
            continue
        n = len(t)
        if n < min_chars or n > max_chars:
            continue
        yield t
        kept += 1
        if kept >= n_docs:
            return


def train_bpe(sample_texts: list[str], vocab_size: int, out_path: Path):
    """Train a fresh byte-level BPE 16384 on the given texts (GPT-2 style)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    )
    print(f"[bpe] training BPE {vocab_size} on {len(sample_texts):,} sample docs...")
    t0 = time.time()
    tok.train_from_iterator(iter(sample_texts), trainer=trainer)
    print(f"[bpe] done in {time.time()-t0:.1f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"[bpe] wrote {out_path}")
    return tok


def tokenize_docs(tok, texts: list[str], boundary_id: int) -> list[np.ndarray]:
    """Return one uint32 array per doc, each starting with the boundary token."""
    print(f"[tokenize] encoding {len(texts):,} docs...")
    t0 = time.time()
    encoded = tok.encode_batch(texts)
    out = []
    for e in encoded:
        arr = np.empty(len(e.ids) + 1, dtype=np.uint32)
        arr[0] = boundary_id
        arr[1:] = e.ids
        out.append(arr)
    print(f"[tokenize] done in {time.time()-t0:.1f}s")
    return out


def write_bin(path: Path, tokens: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint32).tobytes())


def build_partition_bins(doc_tokens: list[np.ndarray], doc_partition: np.ndarray,
                          K: int, val_docs_per_cluster: int, out_dir: Path):
    """Split docs into K partitions, write per-cluster train/val bins."""
    for c in range(K):
        idx = np.where(doc_partition == c)[0]
        if len(idx) == 0:
            print(f"  cluster {c}: EMPTY")
            continue
        # First `val_docs_per_cluster` go to val, rest to train
        val_idx = idx[:val_docs_per_cluster]
        train_idx = idx[val_docs_per_cluster:]
        val_tok = np.concatenate([doc_tokens[i] for i in val_idx])
        train_tok = np.concatenate([doc_tokens[i] for i in train_idx])
        cdir = out_dir / f"cluster_{c}"
        write_bin(cdir / "val_000000.bin", val_tok)
        write_bin(cdir / "train_000000.bin", train_tok)
        print(f"  cluster {c}: {len(train_idx):,} train docs ({len(train_tok):,} toks) / "
              f"{len(val_idx):,} val docs ({len(val_tok):,} toks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-dir", required=True,
                    help="Dir containing k<K>/assignments.jsonl.gz")
    ap.add_argument("--out-root", required=True,
                    help="Output storage root (STORAGE_ROOT-style)")
    ap.add_argument("--ks", default="2,4,8", help="Comma-separated K values to prep")
    # These must match the cluster.py invocation exactly to reproduce the stream:
    ap.add_argument("--n-docs", type=int, default=100_000)
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1_000_000)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--bpe-sample-mb", type=int, default=200,
                    help="MB of text to feed the BPE trainer (from the streamed docs)")
    ap.add_argument("--val-docs-per-cluster", type=int, default=200,
                    help="How many docs per cluster reserved for val")
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out_root = Path(args.out_root)
    cluster_dir = Path(args.cluster_dir)

    # Load cluster assignments AND text_snips (to verify re-stream determinism).
    all_assignments = {}
    text_snips = None  # from first K read
    for K in ks:
        p = cluster_dir / f"k{K}" / "assignments.jsonl.gz"
        assignments = np.full(args.n_docs, -1, dtype=np.int32)
        snips = [None] * args.n_docs
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                assignments[r["i"]] = r["cluster_id"]
                snips[r["i"]] = r["text_snip"]
        if (assignments < 0).any():
            raise SystemExit(f"assignment file {p} is missing docs")
        all_assignments[K] = assignments
        if text_snips is None:
            text_snips = snips
        print(f"[k={K}] loaded {len(assignments):,} assignments, "
              f"sizes={np.bincount(assignments, minlength=K).tolist()}")

    # Stream the same N docs (deterministic given same subset+filter).
    # Verify determinism: first N docs' text_snip must match the assignments file.
    print(f"[stream] re-streaming {args.n_docs:,} docs from {args.subset}...")
    t0 = time.time()
    texts = []
    n_snip = 400  # matches cluster.py's default text-snip-len
    for i, t in enumerate(stream_fineweb(args.n_docs, args.subset, args.min_chars, args.max_chars)):
        got_snip = t[:n_snip].replace("\n", " ")
        want_snip = text_snips[i]
        if got_snip != want_snip:
            print(f"[stream] MISMATCH at doc {i}:")
            print(f"  got  : {got_snip[:120]!r}")
            print(f"  want : {want_snip[:120]!r}")
            raise SystemExit("re-stream is not deterministic; save raw text at clustering time instead")
        texts.append(normalize_text(t))
    print(f"[stream] got {len(texts):,} in {time.time()-t0:.1f}s (all snippets matched)")

    # Train fresh BPE 16384 on a sample.
    tok_path = out_root / "tokenizer" / "joint_bpe16384.json"
    if tok_path.exists():
        print(f"[bpe] reusing existing {tok_path}")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
    else:
        # Sample texts to hit `bpe_sample_mb` MB
        target = args.bpe_sample_mb * 1024 * 1024
        served = 0
        sample = []
        for t in texts:
            sample.append(t)
            served += len(t.encode("utf-8"))
            if served >= target:
                break
        tok = train_bpe(sample, args.vocab_size, tok_path)

    boundary_id = tok.token_to_id("<|endoftext|>")
    assert boundary_id is not None

    # Tokenize all docs (do it once, reuse for every K).
    doc_tokens = tokenize_docs(tok, texts, boundary_id)
    total_toks = sum(len(a) for a in doc_tokens)
    print(f"[tokenize] total tokens: {total_toks:,} "
          f"(avg {total_toks/len(doc_tokens):.0f}/doc)")

    # Build bins for each K, both semantic and random partitions.
    rng = np.random.default_rng(args.random_seed)
    for K in ks:
        print(f"\n=== K={K} SEMANTIC ===")
        sem_dir = out_root / "bins" / "semantic" / f"k{K}"
        build_partition_bins(
            doc_tokens, all_assignments[K], K, args.val_docs_per_cluster, sem_dir,
        )

        print(f"\n=== K={K} RANDOM ===")
        # Random equal-size partition: shuffle and split
        rand_partition = np.zeros(args.n_docs, dtype=np.int32)
        perm = rng.permutation(args.n_docs)
        per = args.n_docs // K
        for c in range(K):
            rand_partition[perm[c*per:(c+1)*per]] = c
        # Any leftover docs (n_docs % K) get randomly reassigned
        leftover = perm[K*per:]
        for i in leftover:
            rand_partition[i] = rng.integers(K)
        rand_dir = out_root / "bins" / "random" / f"k{K}"
        build_partition_bins(
            doc_tokens, rand_partition, K, args.val_docs_per_cluster, rand_dir,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
