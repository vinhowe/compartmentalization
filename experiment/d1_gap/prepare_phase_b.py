"""Phase B data prep: stream a large FineWeb sample, cluster each doc via
the fitted K=2 c-BTM pipeline, tokenize, write per-cluster train/val bins
and an interleaved joint_mixed bin.

Reuses:
  --clusterer: joblib pipeline from cluster.py (tf-idf, svd, scaler, kmeans)
  --tokenizer: BPE json from d1_gap/prepare_bins.py

Output:
  <out-root>/tokenizer/joint_bpe16384.json                (symlink)
  <out-root>/bins/cluster_0/{train_*,val_000000}.bin
  <out-root>/bins/cluster_1/{train_*,val_000000}.bin
  <out-root>/bins/joint_mixed/{train_*,val_cluster_*.bin symlinks}
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Load BalancedKMeans into __main__ so joblib can find it when unpickling
# the clusterer produced by scripts/fineweb_cluster/cluster.py.
_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE / "scripts" / "fineweb_cluster"))
from cluster import BalancedKMeans  # noqa: E402,F401  (needed for joblib.load)

MAGIC = 20251013
SHARD_TOKENS = 100_000_000
NUM_TOKEN = "<NUM>"
_NUM_RE = re.compile(r"\d+")


def normalize_text(t: str) -> str:
    return _NUM_RE.sub(NUM_TOKEN, t)


def stream_fineweb(subset: str, min_chars: int, max_chars: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name=subset, split="train", streaming=True)
    for row in ds:
        t = row.get("text", "")
        if not isinstance(t, str):
            continue
        n = len(t)
        if n < min_chars or n > max_chars:
            continue
        yield t


def write_shard(path: Path, tokens: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint32).tobytes())


def flush_train_shards(buf: list[int], shard_idx: int, out_dir: Path) -> tuple[list[int], int]:
    """Flush full 100M-token shards to disk, keep the remainder in the buffer."""
    while len(buf) >= SHARD_TOKENS:
        arr = np.array(buf[:SHARD_TOKENS], dtype=np.uint32)
        write_shard(out_dir / f"train_{shard_idx:06d}.bin", arr)
        shard_idx += 1
        del buf[:SHARD_TOKENS]
    return buf, shard_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusterer", required=True,
                    help="Path to K=2 clusterer.joblib from cluster.py")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to joint_bpe16384.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-docs", type=int, default=2_500_000,
                    help="Target docs to stream and label (default 2.5M)")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1_000_000)
    ap.add_argument("--val-docs-per-cluster", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="Batched cluster-assign + tokenize batch size")
    ap.add_argument("--log-every", type=int, default=10_000)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[load] clusterer + tokenizer...")
    import joblib
    clusterer = joblib.load(args.clusterer)
    tfidf, svd, scaler, kmeans = (clusterer["tfidf"], clusterer["svd"],
                                    clusterer["scaler"], clusterer["kmeans"])
    K = kmeans.cluster_centers_.shape[0]
    print(f"[load] K={K} clusterer with {kmeans.cluster_centers_.shape[0]} centroids")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    boundary_id = tok.token_to_id("<|endoftext|>")
    assert boundary_id is not None

    # Also symlink the tokenizer into the out root for reproducibility
    tokd = out_root / "tokenizer"
    tokd.mkdir(exist_ok=True)
    dst = tokd / "joint_bpe16384.json"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path(args.tokenizer).resolve())

    # Prepare per-cluster + joint buffers
    train_bufs = {c: [] for c in range(K)}
    joint_buf = []
    train_shards = {c: 0 for c in range(K)}
    joint_shards = 0
    val_docs_bufs = {c: [] for c in range(K)}
    train_docs_written = {c: 0 for c in range(K)}
    total_toks_train = {c: 0 for c in range(K)}
    joint_docs = 0

    # Streaming loop
    print(f"[stream] target {args.n_docs:,} docs from {args.subset}")
    t0 = time.time()
    batch_texts: list[str] = []
    batch_normalized: list[str] = []
    seen = 0

    def flush_batch():
        nonlocal joint_docs, joint_shards
        if not batch_texts:
            return
        # Cluster-assign in batch (tf-idf.transform is fast on batched input)
        X = tfidf.transform(batch_normalized)
        Xs = svd.transform(X)
        Xn = scaler.transform(Xs).astype(np.float32)
        # Nearest-centroid: precompute ||c||^2, cross product
        d = ((Xn * Xn).sum(1)[:, None]
             + (kmeans.cluster_centers_ * kmeans.cluster_centers_).sum(1)[None, :]
             - 2.0 * Xn @ kmeans.cluster_centers_.T)
        labels = d.argmin(axis=1).astype(np.int32)
        # Tokenize in batch (raw texts, not normalized — better BPE quality)
        encoded = tok.encode_batch(batch_texts)
        for i, e in enumerate(encoded):
            arr = np.empty(len(e.ids) + 1, dtype=np.uint32)
            arr[0] = boundary_id
            arr[1:] = e.ids
            c = int(labels[i])
            # First val_docs_per_cluster go to val, rest to train
            if len(val_docs_bufs[c]) < args.val_docs_per_cluster:
                val_docs_bufs[c].append(arr)
            else:
                train_bufs[c].extend(arr.tolist())
                total_toks_train[c] += len(arr)
                train_docs_written[c] += 1
                # Also append to joint (interleaved by natural order — same as paper's
                # bins/joint_mixed doc-level round-robin, but here we're interleaving
                # by original doc arrival order which is already a mixed stream)
                joint_buf.extend(arr.tolist())
                joint_docs += 1
        # Flush shards
        for c in range(K):
            train_bufs[c], train_shards[c] = flush_train_shards(
                train_bufs[c], train_shards[c], out_root / "bins" / f"cluster_{c}"
            )
        # joint_mixed shards
        # (inline the same logic to update local joint_shards)
        while len(joint_buf) >= SHARD_TOKENS:
            arr = np.array(joint_buf[:SHARD_TOKENS], dtype=np.uint32)
            write_shard(out_root / "bins" / "joint_mixed" / f"train_{joint_shards:06d}.bin", arr)
            joint_shards += 1
            del joint_buf[:SHARD_TOKENS]
        batch_texts.clear()
        batch_normalized.clear()
        return joint_shards

    for t in stream_fineweb(args.subset, args.min_chars, args.max_chars):
        batch_texts.append(t)
        batch_normalized.append(normalize_text(t))
        seen += 1
        if len(batch_texts) >= args.batch_size:
            joint_shards_updated = flush_batch()
            if joint_shards_updated is not None:
                joint_shards = joint_shards_updated
        if seen % args.log_every == 0:
            elapsed = time.time() - t0
            rate = seen / elapsed
            train_counts = {c: train_docs_written[c] for c in range(K)}
            train_toks = {c: total_toks_train[c] for c in range(K)}
            print(f"  seen={seen:,}  rate={rate:.0f} doc/s  train_docs={train_counts}  "
                  f"train_toks={train_toks}", flush=True)
        if seen >= args.n_docs:
            break
    joint_shards_updated = flush_batch()
    if joint_shards_updated is not None:
        joint_shards = joint_shards_updated

    # Flush residual train buffers into final (partial) shards
    for c in range(K):
        if train_bufs[c]:
            arr = np.array(train_bufs[c], dtype=np.uint32)
            write_shard(out_root / "bins" / f"cluster_{c}" / f"train_{train_shards[c]:06d}.bin", arr)
            train_shards[c] += 1
    if joint_buf:
        arr = np.array(joint_buf, dtype=np.uint32)
        write_shard(out_root / "bins" / "joint_mixed" / f"train_{joint_shards:06d}.bin", arr)
        joint_shards += 1

    # Write val bins per cluster
    for c in range(K):
        val_toks = np.concatenate(val_docs_bufs[c]).astype(np.uint32) if val_docs_bufs[c] else np.zeros(0, dtype=np.uint32)
        write_shard(out_root / "bins" / f"cluster_{c}" / "val_000000.bin", val_toks)
        # joint_mixed points at the per-cluster vals via relative symlink
        link = out_root / "bins" / "joint_mixed" / f"val_cluster_{c}.bin"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(f"../cluster_{c}/val_000000.bin"))

    dt = time.time() - t0
    print(f"\n[done] took {dt/60:.1f} min, streamed {seen:,} docs")
    for c in range(K):
        print(f"  cluster {c}: {train_docs_written[c]:,} train docs "
              f"({total_toks_train[c]:,} toks, {train_shards[c]} shards), "
              f"{len(val_docs_bufs[c])} val docs")
    print(f"  joint_mixed: {joint_docs:,} train docs, {joint_shards} shards")

    # Write a manifest for downstream use
    (out_root / "manifest.json").write_text(json.dumps({
        "clusterer": args.clusterer,
        "tokenizer": args.tokenizer,
        "K": K,
        "n_docs_seen": seen,
        "train_docs_per_cluster": train_docs_written,
        "train_tokens_per_cluster": total_toks_train,
        "val_docs_per_cluster": {c: len(val_docs_bufs[c]) for c in range(K)},
        "boundary_id": boundary_id,
    }, indent=2))


if __name__ == "__main__":
    main()
