"""Tests for src/checkpoints.py.

The two properties worth protecting:

1. A weights-only checkpoint is never selected for resume. Accepting one
   silently restarts Adam, which is invisible in the logs and cost real nats on
   the n-gram rungs.
2. An annealed checkpoint never appears in a stable listing. Annealed points sit
   below the stable loss curve, so mixing them produces a plausible wrong figure
   rather than an obvious error.
"""

import json
import os

import pytest

from src import checkpoints as ck


def _write(root, name, iter_num, *, optimizer=False, phase="stable"):
    """Create a checkpoint directory the way train.py would."""
    d = os.path.join(root, "checkpoints", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "model.pt"), "w").close()
    if optimizer:
        open(os.path.join(d, "optimizer.pt"), "w").close()
    # trainer_state.json last, as the writer does
    with open(os.path.join(d, "trainer_state.json"), "w") as f:
        json.dump({"iter_num": iter_num, "best_val_loss": 1.0, "phase": phase}, f)
    return d


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def test_tok_dirname_sorts_lexically_in_numeric_order():
    names = [ck.tok_dirname(t) for t in (100_000_000, 2_000_000_000, 27_000_000_000, 131_000_000_000)]
    assert names == sorted(names), "lexical order must match training order"
    assert names[2] == "tok-000027000M"


@pytest.mark.parametrize("tokens", [1_000_000, 27_000_000_000, 1_600_000_000_000])
def test_tok_dirname_round_trips(tokens):
    assert ck.parse_dirname(ck.tok_dirname(tokens)) == ("tok", tokens)


def test_parse_dirname_accepts_legacy_and_rejects_junk():
    assert ck.parse_dirname("step-000700000") == ("step", 700_000)
    assert ck.parse_dirname("_rolling") is None
    assert ck.parse_dirname("latest") is None
    assert ck.parse_dirname("annealed") is None


# --------------------------------------------------------------------------
# the resume rule
# --------------------------------------------------------------------------


def test_resume_ignores_weights_only_and_picks_the_newest_full_state(tmp_path):
    root = str(tmp_path)
    _write(root, "tok-000027000M", 14_000, optimizer=True)
    _write(root, "tok-000029000M", 15_000)                     # weights only, newer
    _write(root, "tok-000030000M", 16_000)                     # weights only, newest

    got = ck.find_resume_checkpoint(root)
    assert got.name == "tok-000027000M"
    assert got.resumable


def test_rolling_wins_when_it_is_newest(tmp_path):
    root = str(tmp_path)
    _write(root, "tok-000027000M", 14_000, optimizer=True)
    _write(root, "_rolling", 14_900, optimizer=True)
    assert ck.find_resume_checkpoint(root).name == "_rolling"


def test_weights_only_run_refuses_rather_than_restarting(tmp_path):
    """The dangerous case: real progress on disk, none of it resumable."""
    root = str(tmp_path)
    _write(root, "step-000700000", 700_000)
    with pytest.raises(RuntimeError, match="none carries optimizer.pt"):
        ck.find_resume_checkpoint(root)


def test_escape_hatch_permits_the_old_adam_resetting_behaviour(tmp_path, monkeypatch):
    root = str(tmp_path)
    _write(root, "step-000700000", 700_000)
    monkeypatch.setenv(ck.ALLOW_WEIGHTS_ONLY_RESUME, "1")
    got = ck.find_resume_checkpoint(root)
    assert got.name == "step-000700000" and not got.resumable


def test_fresh_run_is_not_an_error(tmp_path):
    assert ck.find_resume_checkpoint(str(tmp_path)) is None


def test_incomplete_checkpoint_is_invisible(tmp_path):
    """A directory without trainer_state.json is a half-written checkpoint."""
    root = str(tmp_path)
    d = os.path.join(root, "checkpoints", "tok-000030000M")
    os.makedirs(d)
    open(os.path.join(d, "model.pt"), "w").close()
    open(os.path.join(d, "optimizer.pt"), "w").close()
    assert list(ck.iter_checkpoints(root)) == []
    assert ck.find_resume_checkpoint(root) is None


# --------------------------------------------------------------------------
# annealed vs stable
# --------------------------------------------------------------------------


def test_annealed_wins_over_a_stale_stable_record(tmp_path):
    """Path says annealed, record says stable — the dangerous disagreement.

    Mislabelling an annealed point as stable puts it on a loss curve it does
    not belong on; the reverse merely drops a point. So it resolves to annealed.
    """
    root = str(tmp_path)
    _write(root, os.path.join("annealed", "tok-000030000M"), 16_500, phase="stable")
    assert [c.phase for c in ck.iter_checkpoints(root)] == ["annealed"]


def test_annealed_wins_when_only_the_record_says_so(tmp_path):
    """A decay child's _rolling sits at the top of its own run directory."""
    root = str(tmp_path)
    _write(root, "_rolling", 14_000, optimizer=True, phase="annealed")
    assert [c.phase for c in ck.iter_checkpoints(root)] == ["annealed"]


def test_annealed_checkpoints_are_tagged_and_off_the_stable_path(tmp_path):
    root = str(tmp_path)
    _write(root, "tok-000030000M", 16_000)
    _write(root, os.path.join("annealed", "tok-000030000M"), 16_500, phase="annealed")

    by_phase = {c.phase: c for c in ck.iter_checkpoints(root)}
    assert set(by_phase) == {"stable", "annealed"}

    # the property that matters: a stable glob cannot reach the annealed point
    stable_glob = [
        d for d in os.listdir(os.path.join(root, "checkpoints"))
        if ck.parse_dirname(d)
    ]
    assert stable_glob == ["tok-000030000M"]
    assert by_phase["annealed"].iter_num == 16_500


def test_rolling_is_stable_not_annealed(tmp_path):
    root = str(tmp_path)
    _write(root, "_rolling", 100, optimizer=True)
    assert [c.phase for c in ck.iter_checkpoints(root)] == ["stable"]


def test_include_rolling_false_excludes_it(tmp_path):
    root = str(tmp_path)
    _write(root, "_rolling", 100, optimizer=True)
    _write(root, "tok-000002000M", 90)
    names = {c.name for c in ck.iter_checkpoints(root, include_rolling=False)}
    assert names == {"tok-000002000M"}


# --------------------------------------------------------------------------
# full_state_at
# --------------------------------------------------------------------------


def test_each_budget_gets_its_exact_step_not_a_nearby_checkpoint():
    """The branch point is the step that reaches the budget, full stop.

    Snapping to the nearest scheduled checkpoint was the original behaviour and
    it moved branch points by billions of tokens, because checkpoint_steps is
    log-spaced and the late gaps are huge.
    """
    tpi = 2_097_152
    assert ck.full_state_steps((27e9,), tpi, 50_000) == {12_875, 50_000}
    assert 12_875 * tpi >= 27e9 > 12_874 * tpi        # exactly the ceiling step


def test_branch_point_leaves_the_intended_decay_length():
    """The failure this prevents: a 27B branch on a 30B run left 2.1% of the
    budget for a decay that wants ~10%, and 90B on a 100B run left 5.6%."""
    tpi = 2_097_152
    for budget, total, want_frac in ((27e9, 14_305, 0.10), (90e9, 47_684, 0.10)):
        branch = min(ck.full_state_steps((budget,), tpi, total))
        frac = (total - branch) / total
        assert frac > want_frac * 0.9, (
            f"branch at {branch} leaves only {100*frac:.1f}% of the budget to decay"
        )


def test_end_of_run_is_always_full_state():
    """An unforkable final checkpoint is the one omission you can't repair later."""
    assert ck.full_state_steps((), 2_097_152, 50_000) == {50_000}


def test_budget_beyond_the_run_claims_nothing_extra():
    steps = {1_000, 50_000}
    assert ck.full_state_steps((1e15,), 2_097_152, 50_000) == {50_000}
