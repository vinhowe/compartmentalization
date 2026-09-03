"""Properties of src/bpe_variants, on a synthetic merge table.

Hermetic on purpose -- no tokenizer, no corpus -- so these run in milliseconds
and fail for reasons that are about the algorithm. The real tokenizer and real
FineWeb frequencies were checked once by hand: text round-trips exactly, the
1.5x target lands at 1.5005, and c=2's drop sets are c=8's first two.

The properties that matter are not the obvious ones. Expansion arithmetic that
ignores cascades silently under-counts, so the run reports 1.5x while training on
something else; and a drop set seeded off anything that scales with
n_compartments makes every cross-c comparison compare different tokenizers,
which is indistinguishable from a compartmentalization effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bpe_variants import (
    build_expansion_index, compartment_seed, expand, expansion_ratio,
    pieces_per_token, select_dropped_merges,
)

VOCAB = 512


@pytest.fixture(scope="module")
def table() -> dict[int, tuple[int, int]]:
    """A merge tree over 32 atoms, ids ascending so children precede parents."""
    rng = np.random.default_rng(0)
    t: dict[int, tuple[int, int]] = {}
    for tid in range(32, VOCAB):
        t[tid] = (int(rng.integers(0, tid)), int(rng.integers(0, tid)))
    return t


@pytest.fixture(scope="module")
def freq() -> np.ndarray:
    return (1.0 / np.arange(1, VOCAB + 1)) * 1e6


def test_pieces_accounts_for_cascades(table):
    """Dropping a parent AND its child must expand further than the parent alone.

    This is the arithmetic that decides whether 1.5x means 1.5x.
    """
    parent = VOCAB - 1
    left, _ = table[parent]
    assert left >= 32, "pick a parent whose child is itself a merge"
    shallow = pieces_per_token(table, {parent}, VOCAB)[parent]
    deep = pieces_per_token(table, {parent, left}, VOCAB)[parent]
    assert deep > shallow, f"cascade ignored: {deep} !> {shallow}"


def test_expansion_ratio_is_one_when_nothing_is_dropped(table, freq):
    assert expansion_ratio(freq, table, set()) == pytest.approx(1.0)


def test_expansion_ratio_is_monotone_in_the_drop_set(table, freq):
    """Binary search over k is only valid if adding drops never shortens."""
    order = list(range(32, VOCAB))
    prev = 1.0
    for k in range(0, len(order), 40):
        r = expansion_ratio(freq, table, set(order[:k]))
        assert r >= prev - 1e-12, f"ratio dropped at k={k}"
        prev = r


@pytest.mark.parametrize("target", [1.15, 1.5, 2.0])
def test_search_hits_the_target(table, freq, target):
    _, got = select_dropped_merges(freq, table, target, 1024, 0, tol=0.01)
    assert abs(got - target) < 0.02, f"target {target}, got {got:.4f}"


def test_unreachable_target_raises(table, freq):
    """Silently landing at 3x while the config says 50x would misreport the
    experiment's headline number."""
    with pytest.raises(ValueError, match="unreachable"):
        select_dropped_merges(freq, table, 50.0, 1024, 0)


def test_drop_set_is_reproducible(table, freq):
    a, _ = select_dropped_merges(freq, table, 1.5, 1024, 3)
    b, _ = select_dropped_merges(freq, table, 1.5, 1024, 3)
    assert a == b and len(a) > 0


def test_seed_changes_the_drop_set(table, freq):
    a, _ = select_dropped_merges(freq, table, 1.5, 1024, 0)
    b, _ = select_dropped_merges(freq, table, 1.5, 1025, 0)
    assert a != b


def test_compartments_differ(table, freq):
    sets = [select_dropped_merges(freq, table, 1.5, 1024, c)[0] for c in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert sets[i] != sets[j], f"compartments {i},{j} share a drop set"


@pytest.mark.parametrize("small,large", [(1, 8), (2, 8), (4, 8), (2, 6), (6, 8)])
def test_smaller_c_is_a_prefix_of_larger_c(table, freq, small, large):
    """c=2's schemes must BE c=8's first two, so the two runs share compartment
    0's data stream rather than merely resembling it."""
    big = [select_dropped_merges(freq, table, 1.5, 1024, c)[0] for c in range(large)]
    lil = [select_dropped_merges(freq, table, 1.5, 1024, c)[0] for c in range(small)]
    assert lil == big[:small]


def test_every_compartment_hits_the_same_ratio(table, freq):
    """Compartments must differ in WHICH merges they drop, not in how much they
    expand -- otherwise expansion is confounded with compartment identity."""
    ratios = [select_dropped_merges(freq, table, 1.5, 1024, c)[1] for c in range(8)]
    assert max(ratios) - min(ratios) < 0.02


def test_compartment_seed_ignores_n_compartments(table):
    ref = compartment_seed(1024, 1)
    assert all(compartment_seed(1024, 1) == ref for _ in range(5))
    assert compartment_seed(1024, 1) != compartment_seed(1024, 2)


# ---------------------------------------------------------------------------
# expansion itself
# ---------------------------------------------------------------------------

def _flatten(tid, table, dropped):
    """Reference expansion by recursion; the index must agree with it."""
    if tid not in dropped or tid not in table:
        return [tid]
    left, right = table[tid]
    return _flatten(left, table, dropped) + _flatten(right, table, dropped)


def test_index_matches_recursive_expansion(table, freq):
    dropped, _ = select_dropped_merges(freq, table, 1.5, 1024, 0)
    flat, off = build_expansion_index(table, dropped, VOCAB)
    for tid in range(VOCAB):
        got = flat[off[tid]:off[tid + 1]].tolist()
        assert got == _flatten(tid, table, dropped), f"token {tid}"


def test_expansion_only_emits_atoms_or_undropped_tokens(table, freq):
    """A dropped token must never survive in the output."""
    dropped, _ = select_dropped_merges(freq, table, 1.5, 1024, 0)
    flat, off = build_expansion_index(table, dropped, VOCAB)
    assert not (set(flat.tolist()) & dropped)


def test_fill_to_budget_reports_base_tokens_consumed(table, freq):
    """The caller advances its cursor by this; a wrong count silently skips or
    repeats text."""
    dropped, _ = select_dropped_merges(freq, table, 1.5, 1024, 0)
    flat, off = build_expansion_index(table, dropped, VOCAB)
    rng = np.random.default_rng(3)
    base = rng.integers(0, VOCAB, size=4000, dtype=np.int64)
    out, n_base = expand(base, flat, off, limit=1024)
    assert len(out) == 1024
    assert n_base < len(base)
    # what was consumed must expand to at least the budget, and one base token
    # fewer must not -- i.e. n_base is minimal.
    assert len(expand(base[:n_base], flat, off)[0]) >= 1024
    assert len(expand(base[:n_base - 1], flat, off)[0]) < 1024


def test_expansion_is_a_pure_resegmentation(table, freq):
    """Concatenating the pieces must reproduce the original token sequence when
    every piece is mapped back -- i.e. no token is lost, added, or reordered."""
    dropped, _ = select_dropped_merges(freq, table, 1.5, 1024, 0)
    flat, off = build_expansion_index(table, dropped, VOCAB)
    rng = np.random.default_rng(5)
    base = rng.integers(0, VOCAB, size=500, dtype=np.int64)
    out, _ = expand(base, flat, off)
    rebuilt = [t for tid in base.tolist() for t in _flatten(tid, table, dropped)]
    assert out.tolist() == rebuilt


def test_merge_selection_is_uniform_over_merges(table, freq):
    """INVARIANT: merges are selected with equal probability, independent of how
    frequent they are.

    Pinned by a test because it is invisible in the output -- a frequency-weighted
    selection produces drop sets that look entirely normal, hits the same
    expansion target, and passes every other test here, while quietly changing
    what the compartments are.
    """
    mids = np.array(sorted(table))
    rank = np.argsort(np.argsort(-freq[mids]))          # 0 = most frequent merge
    n = len(mids)
    # Average rank of the dropped merges, over many compartments. Uniform
    # selection centres on the midpoint; any frequency bias shifts it.
    means = []
    for c in range(40):
        dropped, _ = select_dropped_merges(freq, table, 1.5, 1024, c)
        idx = np.array([np.flatnonzero(mids == d)[0] for d in sorted(dropped)])
        means.append(rank[idx].mean())
    observed = float(np.mean(means))
    midpoint = (n - 1) / 2.0
    assert abs(observed - midpoint) < 0.06 * n, (
        f"mean rank of dropped merges {observed:.1f} vs uniform midpoint "
        f"{midpoint:.1f} -- selection is frequency-biased"
    )
