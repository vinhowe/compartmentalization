"""Per-compartment BPE variants by dropping merges.

Each compartment tokenizes the same text under its own subset of dropped merges,
so the compartments differ in SEGMENTATION rather than only in vocabulary
relabeling. A dropped merge's token is replaced by its two constituents, applied
recursively, so the text expands. `target_expansion=1.5` means a compartment
spends 50% more tokens on the same text and therefore covers two-thirds as much
of the corpus per token budget -- which is the quantity the experiment measures.

THE CORPUS IS NEVER RETOKENIZED. Expansion is a pure function of (token id, drop
set) computed from the base tokenizer's own merge table, so the .bin shards stay
exactly as the default tokenizer produced them and every expanded id is an
existing vocab id. Storing 8 tokenized copies of FineWeb is what this avoids.

CASCADES ARE THE WHOLE DIFFICULTY. Dropping a merge lengthens not just its own
occurrences but every token built on top of it, recursively -- and 98.7% of the
16,384-token vocab is a merge result over only 210 atoms, so the tree is deep.
Budgeting on a merge's own frequency badly underestimates the expansion it
causes. `expansion_ratio` therefore evaluates the whole drop set exactly rather
than accumulating per-merge estimates.

SELECTION IS A SEARCH, NOT A GREEDY WALK. Adding drops one at a time until a
budget is crossed overshoots when a high-frequency merge lands last (measured
1.15 -> 2.00 on a stand-in); skipping every overshooting candidate instead makes
the result "all merges below a size threshold", which is order-insensitive, so
every compartment converges to the SAME drop set and compartmentalization
vanishes. Both failures disappear if the drop set is a seeded permutation
truncated at length k, with k found by binary search on the exact ratio:
expansion is monotone in k, compartments keep distinct permutations, and the
target is hit to whatever tolerance is asked for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "load_merge_table", "token_frequencies", "pieces_per_token",
    "expansion_ratio", "compartment_seed", "select_dropped_merges",
    "build_expansion_index", "expand",
]


def compartment_seed(base_seed: int, compartment_id: int) -> int:
    """Seed for one compartment's merge permutation.

    A function of the compartment's IDENTITY and nothing else -- never of
    n_compartments, nor of a position in a list of length c. That is what makes
    a c=2 run's schemes exactly the first two of a c=8 run's. Derive this from
    anything that scales with c and every cross-c comparison silently compares
    different tokenizers, which looks exactly like a compartmentalization effect.
    """
    M = (1 << 64) - 1
    z = (int(base_seed) * 0x9E3779B97F4A7C15 + int(compartment_id)) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return (z ^ (z >> 31)) & M


def load_merge_table(tokenizer_dir: str | Path) -> dict[int, tuple[int, int]]:
    """`{token_id: (left_id, right_id)}` for every token that is a merge.

    Read straight from the tokenizer's own merge list, so this is the base
    tokenizer's structure rather than a reconstruction. (tiktoken forces the
    harder route -- it stores only ranks, so the split has to be searched for.)
    """
    tok = json.loads((Path(tokenizer_dir) / "tokenizer.json").read_text())
    vocab, merges = tok["model"]["vocab"], tok["model"]["merges"]
    table: dict[int, tuple[int, int]] = {}
    for entry in merges:
        left, right = entry if isinstance(entry, list) else entry.split(" ", 1)
        merged = left + right
        if merged in vocab and left in vocab and right in vocab:
            table[vocab[merged]] = (vocab[left], vocab[right])
    return table


def token_frequencies(bin_glob: str, vocab_size: int, max_tokens: int = 200_000_000
                      ) -> np.ndarray:
    """Occurrence count per token id over a prefix of the corpus.

    A sample, not the whole corpus: the ratio only needs relative frequencies,
    and these are stable long before 200M tokens.
    """
    import glob as _glob
    from src.datafile import load_data_shard

    counts = np.zeros(vocab_size, dtype=np.int64)
    seen = 0
    for path in sorted(_glob.glob(bin_glob)):
        shard = load_data_shard(path)
        counts += np.bincount(shard, minlength=vocab_size)[:vocab_size]
        seen += len(shard)
        if seen >= max_tokens:
            break
    return counts


def pieces_per_token(table: dict[int, tuple[int, int]], dropped: set[int],
                     vocab_size: int) -> np.ndarray:
    """How many tokens each id becomes under `dropped`, cascades included.

    Iterative rather than recursive: the merge tree is ~16k deep in the worst
    case and Python's stack is not.
    """
    pieces = np.ones(vocab_size, dtype=np.int64)
    # Merge ids are assigned in merge order, so a token's constituents always
    # have smaller ids. Ascending order therefore resolves children first and one
    # pass suffices.
    for tid in sorted(dropped):
        if tid in table:
            left, right = table[tid]
            pieces[tid] = pieces[left] + pieces[right]
    return pieces


def expansion_ratio(freq: np.ndarray, table: dict[int, tuple[int, int]],
                    dropped: set[int]) -> float:
    """Exact corpus expansion under `dropped`. 1.0 = no change."""
    total = float(freq.sum())
    if total == 0:
        return 1.0
    return float((freq * pieces_per_token(table, dropped, len(freq))).sum()) / total


def select_dropped_merges(
    freq: np.ndarray,
    table: dict[int, tuple[int, int]],
    target_expansion: float,
    base_seed: int,
    compartment_id: int,
    tol: float = 0.005,
) -> tuple[set[int], float]:
    """Drop set for one compartment, and the expansion it actually achieves.

    The permutation is seeded per compartment; only its LENGTH is searched. So
    two compartments share no structure beyond both hitting the same ratio, and
    the same (seed, compartment_id) always reproduces the same set.
    """
    if target_expansion < 1.0:
        raise ValueError(f"target_expansion must be >= 1.0, got {target_expansion}")
    merge_ids = np.array(sorted(table), dtype=np.int64)
    order = np.random.default_rng(
        compartment_seed(base_seed, compartment_id)
    ).permutation(merge_ids)

    if target_expansion == 1.0:
        return set(), 1.0
    hi_ratio = expansion_ratio(freq, table, set(order.tolist()))
    if hi_ratio < target_expansion - tol:
        raise ValueError(
            f"target_expansion={target_expansion} is unreachable: dropping EVERY "
            f"merge only reaches {hi_ratio:.4f}. Lower the target."
        )

    lo, hi = 0, len(order)
    while lo < hi:                       # smallest k reaching the target
        mid = (lo + hi) // 2
        if expansion_ratio(freq, table, set(order[:mid].tolist())) >= target_expansion:
            hi = mid
        else:
            lo = mid + 1
    chosen = set(order[:lo].tolist())
    return chosen, expansion_ratio(freq, table, chosen)


def build_expansion_index(table: dict[int, tuple[int, int]], dropped: set[int],
                          vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat CSR-style expansion: `(flat, offsets)`.

    Token `t` expands to `flat[offsets[t]:offsets[t+1]]`. Precomputed once per
    compartment so the per-batch path is a gather rather than a recursion.
    """
    pieces = pieces_per_token(table, dropped, vocab_size)
    offsets = np.zeros(vocab_size + 1, dtype=np.int64)
    np.cumsum(pieces, out=offsets[1:])
    flat = np.zeros(int(offsets[-1]), dtype=np.int32)

    for tid in range(vocab_size):
        if pieces[tid] == 1:
            flat[offsets[tid]] = tid
    for tid in sorted(dropped):          # ascending: children already written
        if tid not in table:
            continue
        left, right = table[tid]
        nl = int(pieces[left])
        flat[offsets[tid]:offsets[tid] + nl] = flat[offsets[left]:offsets[left] + nl]
        nr = int(pieces[right])
        flat[offsets[tid] + nl:offsets[tid] + nl + nr] = flat[offsets[right]:offsets[right] + nr]
    return flat, offsets


def expand(tokens: np.ndarray, flat: np.ndarray, offsets: np.ndarray,
           limit: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Expand `tokens`, returning `(expanded, n_base_consumed)`.

    With `limit`, stops as soon as `limit` output tokens exist and reports how
    many BASE tokens that took -- which is how an example is filled: read until
    the budget is met, and the caller advances its cursor by n_base_consumed.
    The overshoot from the final base token is truncated (~0.1%; a token cannot
    be split), and no expanded tokens are carried across examples, because a
    carried remainder belongs to one compartment's segmentation and would splice
    unrelated text into the next example that compartment received.
    """
    sizes = (offsets[tokens + 1] - offsets[tokens]).astype(np.int64)
    if limit is not None:
        cum = np.cumsum(sizes)
        n_base = int(np.searchsorted(cum, limit, side="left") + 1)
        n_base = min(n_base, len(tokens))
        tokens, sizes = tokens[:n_base], sizes[:n_base]
    else:
        n_base = len(tokens)
    idx = np.repeat(offsets[tokens], sizes) + (
        np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    )
    out = flat[idx]
    return (out[:limit] if limit is not None else out), n_base
