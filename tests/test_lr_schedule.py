"""Tests for src/lr_schedule.py.

The property that matters here is not "the curve has the right shape" but
**resume fidelity**: a run interrupted at an arbitrary point and restarted must
produce the same learning rate at every subsequent step as an uninterrupted
run. Slurm leases end mid-decay, so this is exercised constantly in practice,
and the failure mode is silent — a double-annealed run just reports a slightly
better loss.
"""

import math

import pytest

from src.lr_schedule import lr_at, validate


WSD = dict(
    peak=4e-4,
    warmup_iters=1000,
    min_lr=4e-5,
    schedule="wsd",
    decay_start_iter=27_000,
    decay_end_iter=30_000,
)


# --------------------------------------------------------------------------
# resume fidelity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kill_at", [1, 999, 1000, 26_999, 27_000, 28_500, 29_999, 30_000])
def test_resume_reproduces_uninterrupted_trace(kill_at):
    """Killing at any point and resuming must not perturb a single later LR.

    `kill_at` includes the phase boundaries (last warmup step, first stable
    step, the branch point, mid-decay, the final step) because those are where
    an off-by-one or a steps-since-start bug would hide.
    """
    uninterrupted = [lr_at(i, **WSD) for i in range(31_000)]

    # a resumed process restores iter_num and continues; nothing else carries
    # over, so re-deriving from `it` alone must be sufficient
    resumed = [lr_at(i, **WSD) for i in range(kill_at)]
    resumed += [lr_at(i, **WSD) for i in range(kill_at, 31_000)]

    assert resumed == uninterrupted


def test_schedule_is_pure_in_it():
    """No hidden state: repeated and out-of-order calls agree."""
    forward = {i: lr_at(i, **WSD) for i in range(0, 31_000, 7)}
    backward = {i: lr_at(i, **WSD) for i in reversed(range(0, 31_000, 7))}
    assert forward == backward


def test_decay_does_not_restart_when_child_inherits_absolute_window():
    """A decay child forked at 27k picks up mid-schedule, not from zero.

    This is the concrete bug the absolute window prevents: if the child
    expressed its decay as "3000 steps from wherever I started", a preemption
    at 28.5k would restart it and anneal twice as far.
    """
    parent_at_branch = lr_at(27_000, **WSD)
    assert parent_at_branch == pytest.approx(WSD["peak"])          # stable, not yet decayed

    # child resumes at 28_500 after preemption
    assert lr_at(28_500, **WSD) < WSD["peak"]
    assert lr_at(28_500, **WSD) > WSD["min_lr"]
    # halfway through the window is the cosine midpoint, not the endpoint
    assert lr_at(28_500, **WSD) == pytest.approx(
        WSD["min_lr"] + 0.5 * (WSD["peak"] - WSD["min_lr"])
    )


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def test_warmup_then_stable_then_decay():
    assert lr_at(0, **WSD) == pytest.approx(WSD["peak"] / 1001)
    # the last warmup step is still short of peak; peak lands on the first
    # stable step, which is what `it < warmup_iters` means
    assert lr_at(999, **WSD) == pytest.approx(WSD["peak"] * 1000 / 1001)
    # stable phase is flat all the way to the branch point
    assert lr_at(1_000, **WSD) == lr_at(26_999, **WSD) == WSD["peak"]
    assert lr_at(30_000, **WSD) == pytest.approx(WSD["min_lr"])


def test_overshooting_the_window_holds_min_lr():
    """Past decay_end the LR must clamp, not walk back up the cosine.

    A run that overshoots its budget (an extra checkpoint interval, a requeue
    that replays a few steps) would otherwise start *raising* the LR again.
    """
    assert lr_at(30_001, **WSD) == pytest.approx(WSD["min_lr"])
    assert lr_at(60_000, **WSD) == pytest.approx(WSD["min_lr"])


def test_unset_window_is_a_pure_stable_run():
    """The stable trajectory decay children fork off never decays."""
    cfg = dict(WSD, decay_start_iter=0, decay_end_iter=0)
    assert lr_at(1_000, **cfg) == lr_at(500_000, **cfg) == cfg["peak"]


# --------------------------------------------------------------------------
# legacy parity — existing runs must be bit-identical
# --------------------------------------------------------------------------


def _legacy_reference(it, *, peak, warmup_iters, min_lr, decay_lr, lr_decay_iters):
    """The schedule exactly as it was inlined in train.py before the refactor."""
    if it < warmup_iters:
        return peak * (it + 1) / (warmup_iters + 1)
    if not decay_lr:
        return peak
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak - min_lr)


@pytest.mark.parametrize("decay_lr", [False, True])
def test_legacy_mode_is_bit_identical_to_the_old_inline_schedule(decay_lr):
    """Extracting the function must not move any existing run's LR by an ulp."""
    ref_kw = dict(
        peak=2e-5, warmup_iters=1000, min_lr=6e-5, decay_lr=decay_lr,
        lr_decay_iters=600_000,
    )
    for it in list(range(0, 1_100)) + list(range(1_100, 600_001, 997)):
        assert lr_at(it, schedule="legacy", **ref_kw) == _legacy_reference(it, **ref_kw)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_half_specified_window_is_rejected():
    with pytest.raises(ValueError, match="both be set"):
        validate(schedule="wsd", warmup_iters=1000, decay_start_iter=27_000, decay_end_iter=0)
    with pytest.raises(ValueError, match="both be set"):
        validate(schedule="wsd", warmup_iters=1000, decay_start_iter=0, decay_end_iter=30_000)


def test_inverted_or_warmup_overlapping_window_is_rejected():
    with pytest.raises(ValueError, match="must exceed"):
        validate(schedule="wsd", warmup_iters=1000, decay_start_iter=30_000, decay_end_iter=27_000)
    with pytest.raises(ValueError, match="inside warmup"):
        validate(schedule="wsd", warmup_iters=1000, decay_start_iter=500, decay_end_iter=30_000)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="must be 'legacy' or 'wsd'"):
        validate(schedule="cosine", warmup_iters=1000, decay_start_iter=0, decay_end_iter=0)


def test_legacy_never_validates_wsd_fields():
    """Every existing config must pass validation untouched."""
    validate(schedule="legacy", warmup_iters=1000, decay_start_iter=0, decay_end_iter=0)
