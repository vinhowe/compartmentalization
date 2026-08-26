"""Is the aligning force concentrated where the translation tokens are?

The order parameter measured over uniformly-sampled token ids is much lower than
the same quantity measured over the tokens that actually appear in a corpus
batch (at tr=0.5: 0.30 uniform vs 0.52 on the canonical batch). That gap is a
prediction, not an artifact: the only thing that pairs compartment i's code for
token t with compartment j's code for token t is a translation row containing
t, so the aligning gradient a token receives is proportional to how often it is
sampled. Frequent tokens should be aligned and rare tokens should not.

This bins the vocabulary by corpus frequency and computes the embedding order
parameter within each bin, at the final checkpoint of each run. If the mechanism
is right, q rises monotonically with frequency, and the slope is steeper the
larger tr is.

Run from experiment/.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from _run_paths import OUT_ROOT
from compute_embedding_order import RUNS, load_blocks, named_steps, BASE_VOCAB

VAL_BIN_PATTERN = "../data/fineweb350B-dedup-bpe16384/fineweb350b-dedup_val_*.bin"


def token_frequencies(max_tokens: int = 40_000_000) -> np.ndarray:
    """Corpus counts per base-vocab id, from the val shard."""
    files = sorted(glob.glob(VAL_BIN_PATTERN))
    if not files:
        raise FileNotFoundError(VAL_BIN_PATTERN)
    with open(files[0], "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        ntok = min(int(header[2]), max_tokens)
        toks = np.frombuffer(f.read(ntok * 4), dtype=np.uint32).astype(np.int64)
    return np.bincount(toks, minlength=BASE_VOCAB)[:BASE_VOCAB]


def final_step(run_dir: Path) -> int:
    return named_steps(run_dir)[-1]


def q_for(E: np.ndarray) -> float:
    c = E.shape[0]
    En = E / (np.linalg.norm(E, axis=2, keepdims=True) + 1e-12)
    return float(np.mean([(En[i] * En[j]).sum(1).mean()
                          for i in range(c) for j in range(i + 1, c)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tr011,tr01,tr025,tr05,tr075,tr10")
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--per-bin", type=int, default=1500)
    ap.add_argument("--out", default="alignment_by_frequency.json")
    args = ap.parse_args()

    freq = token_frequencies()
    order = np.argsort(freq)          # ascending frequency
    # Drop ids never seen: an unseen token's embedding has had no gradient at
    # all, so including it would measure initialisation, not training.
    seen = order[freq[order] > 0]
    print(f"vocabulary: {len(seen)}/{BASE_VOCAB} ids seen in the val shard")

    bins = np.array_split(seen, args.n_bins)
    rng = np.random.Generator(np.random.PCG64(0))
    picks, labels = [], []
    for b in bins:
        take = b if len(b) <= args.per_bin else rng.choice(b, args.per_bin, replace=False)
        picks.append(np.sort(take))
        labels.append((int(freq[b].min()), int(freq[b].max()), float(np.median(freq[b]))))

    res = {"bins": [{"min": a, "max": c, "median": m} for a, c, m in labels],
           "runs": {}}
    for name in args.runs.split(","):
        rel, tr = RUNS[name]
        run_dir = OUT_ROOT / rel
        cfg = json.loads((run_dir / "meta" / "config.json").read_text())
        c = int(cfg["experiment"]["n_compartments"])
        st = final_step(run_dir)
        qs = []
        for ids in picks:
            E, H, ce, _ = load_blocks(run_dir / "checkpoints" / f"step-{st:06d}",
                                      ids, c)
            qs.append(q_for(E))
        res["runs"][name] = {"tr": tr, "step": st, "q_by_bin": qs}
        row = "  ".join(f"{v:+.3f}" for v in qs)
        print(f"  {name:7s} tr={tr:<6} step {st:>9,}  q by freq bin: {row}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
