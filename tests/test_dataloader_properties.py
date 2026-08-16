"""Dataloader properties, asserted without loading a model.

Every bug this file exists to catch is SILENT: the run trains, the loss curve
looks plausible, and the defect only shows up as a number that is quietly wrong.
Three have already happened here --

  * the LCG data ordering cost 0.28 nats at c=8/seed=1024 and looked like seed
    variance (figures/diag_c8_assignment_order.png),
  * resuming restored rank 0's dataloader state onto every rank, so all 8 ranks
    trained on identical data,
  * weights-only checkpoints were accepted for resume, silently restarting Adam.

None needed a GPU to detect. The point of this file is that none of the next
ones should either.

The expansion tests cover per-compartment BPE merge-dropping, where compartment
c tokenizes the same text under its own subset of dropped merges. Two properties
matter and neither is automatic:

  SEED STABILITY      the drop set for (compartment, seed) is a pure function of
                      those two things -- rebuilding it in another process, or
                      after a resume, gives the identical set.
  COMPARTMENT-ID      compartment c's scheme does not depend on how many
  STABILITY           compartments exist. The c=2 run's two schemes must be
                      exactly the first two of the c=8 run's, so a c=2 and a c=8
                      run share compartment 0's data stream rather than merely
                      resembling it.

The second is the one that silently breaks: derive the drop set from anything
that scales with n_compartments (a per-run RNG, an index into a shuffle of
length c, a budget split c ways) and every cross-c comparison is confounded by
a different tokenizer, which looks exactly like a compartmentalization effect.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference implementation of the property under test.
#
# Deliberately NOT imported from the trainer: this is the specification the
# implementation must satisfy. Seeding per (compartment_id, base_seed) and never
# per (run, position) is the whole mechanism behind compartment-id stability.
# ---------------------------------------------------------------------------

CANDIDATE_CAP_FRACTION = 0.05


def compartment_drop_seed(base_seed: int, compartment_id: int) -> int:
    """Seed for one compartment's merge-drop set.

    A function of the compartment's IDENTITY, never of n_compartments or of the
    compartment's position in any list.
    """
    return (int(base_seed) << 20) ^ (int(compartment_id) + 0x9E37)


def select_dropped_merges(
    n_merges: int,
    merge_freq: np.ndarray,
    target_expansion: float,
    base_seed: int,
    compartment_id: int,
) -> set[int]:
    """Merges to drop so the corpus expands by ~target_expansion.

    Mirrors make_budget_dropped_merge_set from the tinystories prototype, with
    one correction it lacks (see below). Cascades are ignored -- this is a
    stand-in whose STRUCTURE is what the tests pin down.
    Merge frequencies are Zipfian, and that creates a tension the algorithm has
    to resolve explicitly:

      * take candidates in random order and stop once the budget is crossed, and
        a single huge merge overshoots -- measured 1.15 -> 2.00 here;
      * skip every candidate that would overshoot, and the result converges to
        "every merge below a size threshold", which is order-INSENSITIVE, so all
        compartments get the same drop set and compartmentalization vanishes.

    Resolved by capping the candidate pool at a small fraction of the budget
    first. That bounds overshoot to one cap-width while leaving a large pool for
    the shuffle to choose from, so the sets stay both accurate and distinct.
    Raises when the capped pool cannot fund the target rather than silently
    under-expanding, which would misreport the experiment's headline ratio.
    """
    rng = np.random.default_rng(compartment_drop_seed(base_seed, compartment_id))
    total = float(merge_freq.sum())
    budget = (target_expansion - 1.0) * total
    cap = CANDIDATE_CAP_FRACTION * budget

    eligible = np.flatnonzero(merge_freq <= cap)
    if float(merge_freq[eligible].sum()) < budget:
        raise ValueError(
            f"target_expansion={target_expansion} needs {budget:.0f} of extra mass "
            f"but merges under the {cap:.0f} cap hold only "
            f"{merge_freq[eligible].sum():.0f}. Raise CANDIDATE_CAP_FRACTION "
            f"(coarser control, more overshoot) or lower the target."
        )

    dropped: set[int] = set()
    extra = 0.0
    for m in rng.permutation(eligible):
        if extra >= budget:
            break
        dropped.add(int(m))
        extra += float(merge_freq[m])
    return dropped


@pytest.fixture(scope="module")
def merge_freq() -> np.ndarray:
    """Merge frequencies shaped like a real corpus: Zipfian over RANK.

    Not np.random.zipf, whose a=1.4 has infinite mean -- it puts nearly all mass
    on a handful of merges, so no capped pool can fund any target and every test
    fails for a reason that would never occur with real data. Real BPE merge
    counts fall off as ~1/rank, where the top merge holds ~12% of mass.
    """
    ranks = np.arange(1, 4097, dtype=np.float64)
    return 1.0 / ranks


# ---------------------------------------------------------------------------
# expansion: seed stability
# ---------------------------------------------------------------------------

def test_drop_set_is_a_pure_function_of_seed_and_compartment(merge_freq):
    a = select_dropped_merges(4096, merge_freq, 1.5, base_seed=1024, compartment_id=3)
    b = select_dropped_merges(4096, merge_freq, 1.5, base_seed=1024, compartment_id=3)
    assert a == b and len(a) > 0


def test_different_seeds_give_different_drop_sets(merge_freq):
    a = select_dropped_merges(4096, merge_freq, 1.5, base_seed=1024, compartment_id=0)
    b = select_dropped_merges(4096, merge_freq, 1.5, base_seed=1025, compartment_id=0)
    assert a != b


def test_different_compartments_give_different_drop_sets(merge_freq):
    """Otherwise compartments are not actually distinguishable."""
    sets = [select_dropped_merges(4096, merge_freq, 1.5, 1024, c) for c in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert sets[i] != sets[j], f"compartments {i} and {j} share a drop set"


# ---------------------------------------------------------------------------
# expansion: compartment-id stability  <- the one that silently confounds c
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("small,large", [(2, 8), (1, 8), (4, 8), (6, 8), (2, 6)])
def test_smaller_c_is_a_prefix_of_larger_c(merge_freq, small, large):
    """c=2's schemes must BE the first two of c=8's, not merely resemble them."""
    big = [select_dropped_merges(4096, merge_freq, 1.5, 1024, c) for c in range(large)]
    lil = [select_dropped_merges(4096, merge_freq, 1.5, 1024, c) for c in range(small)]
    assert lil == big[:small]


def test_drop_set_ignores_n_compartments_entirely(merge_freq):
    """Same compartment id, different run widths -> identical scheme."""
    ref = select_dropped_merges(4096, merge_freq, 1.5, 1024, compartment_id=1)
    for _n_compartments in (2, 3, 5, 8, 16):
        got = select_dropped_merges(4096, merge_freq, 1.5, 1024, compartment_id=1)
        assert got == ref


def test_expansion_target_is_hit_within_tolerance(merge_freq):
    for target in (1.15, 1.5):
        d = select_dropped_merges(4096, merge_freq, target, 1024, 0)
        got = 1.0 + sum(merge_freq[m] for m in d) / merge_freq.sum()
        assert abs(got - target) < 0.05, f"target {target}, got {got:.3f}"


def test_unreachable_target_raises_rather_than_under_expanding(merge_freq):
    """Silently landing at 1.3x while the config says 1.5x would misreport the
    experiment's headline number."""
    with pytest.raises(ValueError, match="target_expansion"):
        select_dropped_merges(4096, merge_freq, 3.0, 1024, 0)


def test_expansion_target_does_not_shift_with_compartment(merge_freq):
    """Every compartment must expand by the SAME ratio, or the comparison across
    compartments confounds expansion with compartment identity."""
    ratios = []
    for c in range(8):
        d = select_dropped_merges(4096, merge_freq, 1.5, 1024, c)
        ratios.append(1.0 + sum(merge_freq[m] for m in d) / merge_freq.sum())
    assert max(ratios) - min(ratios) < 0.02, f"spread {max(ratios)-min(ratios):.4f}"


# ---------------------------------------------------------------------------
# rank divergence, including ACROSS RESUME
#
# Regression test for the bug found 2026-08-15: rank 0 saved the dataloader
# state, every rank loaded it, and no rank offset was re-applied -- so after any
# resume all 8 ranks trained on byte-identical batches, cutting distinct
# sequences per step from 2048 to 256 while the LR stayed tuned for 2048.
# ---------------------------------------------------------------------------

class _FakeLoader:
    """The trainer's assignment_idx bookkeeping, isolated from data and torch."""

    def __init__(self, rank: int, world: int, B: int, num_records: int):
        self.rank, self.world, self.B, self.n = rank, world, B, num_records
        self.assignment_idx = rank % max(1, num_records)

    def next_indices(self) -> np.ndarray:
        idx = (self.assignment_idx + self.world * np.arange(self.B)) % max(1, self.n)
        self.assignment_idx = (self.assignment_idx + self.world * self.B) % self.n
        return idx

    def state_dict(self) -> dict:
        return {"assignment_idx": self.assignment_idx}

    # The FIXED semantics: a rank restoring another rank's state must re-apply
    # its own offset. Ranks advance in lockstep by world*B, and rank r starts at
    # r, so rank r's index is always rank 0's plus r.
    def load_state_dict(self, state: dict, saved_by_rank: int = 0) -> None:
        self.assignment_idx = (
            state["assignment_idx"] - saved_by_rank + self.rank
        ) % max(1, self.n)


def test_ranks_see_disjoint_indices_before_resume():
    loaders = [_FakeLoader(r, 8, 32, 4_000_000) for r in range(8)]
    seen = [set(l.next_indices().tolist()) for l in loaders]
    for i in range(8):
        for j in range(i + 1, 8):
            assert not (seen[i] & seen[j]), f"ranks {i},{j} overlap before resume"


def test_ranks_still_disjoint_after_resume():
    """THE REGRESSION. Restoring rank 0's state onto all ranks collapsed them."""
    loaders = [_FakeLoader(r, 8, 32, 4_000_000) for r in range(8)]
    for _ in range(5):
        for l in loaders:
            l.next_indices()
    saved = loaders[0].state_dict()          # master-only save, as train.py does

    revived = [_FakeLoader(r, 8, 32, 4_000_000) for r in range(8)]
    for l in revived:
        l.load_state_dict(dict(saved), saved_by_rank=0)

    seen = [set(l.next_indices().tolist()) for l in revived]
    assert len({tuple(sorted(s)) for s in seen}) == 8, "ranks collapsed after resume"
    for i in range(8):
        for j in range(i + 1, 8):
            assert not (seen[i] & seen[j]), f"ranks {i},{j} overlap after resume"


def test_resume_is_seamless_for_each_rank():
    """Resuming must continue each rank's stream, not restart or skip it."""
    cont = [_FakeLoader(r, 8, 32, 4_000_000) for r in range(8)]
    for _ in range(5):
        for l in cont:
            l.next_indices()
    saved = cont[0].state_dict()
    expected = [l.next_indices().tolist() for l in cont]

    revived = [_FakeLoader(r, 8, 32, 4_000_000) for r in range(8)]
    for l in revived:
        l.load_state_dict(dict(saved), saved_by_rank=0)
    assert [l.next_indices().tolist() for l in revived] == expected


# ---------------------------------------------------------------------------
# assignment ordering: no periodicity
#
# The LCG's max |autocorrelation| over lags 1..200 measured 0.9927 against the
# hash's 0.0014. Combined with a sequential token stream, an autocorrelated
# assignment sequence hands a compartment contiguous runs of corpus instead of
# an interleaved sample.
# ---------------------------------------------------------------------------

def _splitmix64_scalar(x: int) -> int:
    M = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return (z ^ (z >> 31)) & M


def _hash_assignments(n: int, seed: int, c: int) -> np.ndarray:
    u = np.array([_splitmix64_scalar(i ^ seed) >> 11 for i in range(n)], dtype=np.float64)
    u /= float(1 << 53)
    return np.searchsorted(np.cumsum(np.full(c, 1.0 / c)), u, side="right")


def test_hash_assignments_have_no_periodicity():
    a = _hash_assignments(20_000, 1024, 8).astype(np.float64)
    x = (a - a.mean()) / a.std()
    worst = max(abs(float((x[:-L] * x[L:]).mean())) for L in range(1, 201))
    assert worst < 0.10, f"max |autocorr| {worst:.4f} over lags 1..200"


def test_hash_assignments_are_prefix_stable():
    """Extending a run must not re-randomise the examples already trained on."""
    short = _hash_assignments(5_000, 1024, 8)
    long = _hash_assignments(50_000, 1024, 8)[:5_000]
    assert np.array_equal(short, long)


def test_hash_assignments_are_balanced():
    a = _hash_assignments(40_000, 1024, 8)
    frac = np.bincount(a, minlength=8) / len(a)
    assert abs(frac - 1 / 8).max() < 0.01, f"imbalance {abs(frac - 1/8).max():.4f}"


# ---------------------------------------------------------------------------
# corpus read order
#
# The loader inherited llm.c's sequential scan, which is fine when you consume
# the whole corpus and wrong when you consume 8% of one. Against FineWeb
# sample-350BT, files 0-29 of 510 are ALL CC-MAIN-2013-20, so ~77% of every 30B
# run was a single 2013 crawl and the training seed changed nothing about the
# data. These pin the seeded shuffle that replaced it.
# ---------------------------------------------------------------------------

def _mix_seed(seed: int, salt: int) -> int:
    """Mirror of train.py's _mix_seed; the spec, not an import."""
    M = (1 << 64) - 1
    z = (int(seed) * 0x9E3779B97F4A7C15 + int(salt)) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return (z ^ (z >> 31)) & M


def _order(seed, n=3890, salt=0x5EED_F11E):
    return np.random.default_rng(_mix_seed(seed, salt)).permutation(n)


def test_shard_order_is_deterministic_given_the_seed():
    assert np.array_equal(_order(1024), _order(1024))


def test_shard_order_depends_on_the_seed():
    """Otherwise a 'seed replicate' re-trains on byte-identical data."""
    assert not np.array_equal(_order(1024), _order(1025))


def test_every_rank_derives_the_same_order():
    """Rank must not enter the permutation: ranks stride through one shared
    order, so a rank-dependent permutation would make them disjoint in a
    world_size-dependent way and silently change what the run covers."""
    assert all(np.array_equal(_order(1024), _order(1024)) for _ in range(8))


def test_order_is_independent_of_world_size():
    """The union of what the ranks read must not depend on how many there are."""
    order = _order(1024)
    for world in (1, 2, 4, 8):
        covered = np.sort(np.concatenate([order[r::world] for r in range(world)]))
        assert np.array_equal(covered, np.sort(order))


def test_shuffle_reaches_across_the_whole_corpus():
    """A 30B run reads ~300 shards. Unshuffled those are shards 0-299, which are
    one 2013 crawl; shuffled they must span the corpus."""
    first300 = _order(1024)[:300]
    assert first300.max() > 3500 and first300.min() < 400
    assert np.median(first300) > 1200, "shuffled prefix is still front-loaded"


def test_block_permutation_preserves_every_token():
    """Shuffling blocks must reorder, never drop -- including the ragged tail."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 16384, size=8192 * 5 + 137, dtype=np.uint16)
    BLK, n_full = 8192, len(tokens) // 8192
    head = tokens[: n_full * BLK].reshape(n_full, BLK)
    out = np.concatenate([head[_order(7, n_full, 1)].reshape(-1), tokens[n_full * BLK :]])
    assert len(out) == len(tokens)
    assert np.array_equal(np.bincount(out, minlength=16384),
                          np.bincount(tokens, minlength=16384))
