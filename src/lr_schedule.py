"""Learning-rate schedules.

This module exists so the schedule is a *pure function of the global iteration
number* — structurally, not by convention. That property is what makes a run
resumable at an arbitrary point: `iter_num` is restored from
`trainer_state.json`, so a job preempted mid-decay comes back on exactly the LR
it would have had. Expressing any phase in terms of steps-since-process-start
would restart the decay on every preemption and silently double-anneal, whose
only symptom is a slightly better loss.

Keeping it out of `train.py`'s closure means it can be tested against that
property directly (see `tests/test_lr_schedule.py`).

Two modes:

``legacy``
    What every pre-existing config gets. Warmup, then either constant
    (``decay_lr=False``) or cosine down to ``min_lr`` over
    ``[warmup_iters, lr_decay_iters]``.

``wsd``
    Warmup-Stable-Decay. Warmup, then a stable phase at ``peak`` that can be
    extended indefinitely, then a cosine decay to ``min_lr`` over the absolute
    window ``[decay_start_iter, decay_end_iter]``. Cosine-to-a-horizon is
    disqualified for the new runs because it is *defined* by its horizon:
    committing to a token budget would preclude extending the run without a
    re-warm discontinuity. Under WSD a decay is forked from any stable point,
    inheriting the window unchanged, so the stable prefix is trained once.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["lr_at", "validate"]


def lr_at(
    it: int,
    *,
    peak: float,
    warmup_iters: int,
    min_lr: float,
    schedule: str = "legacy",
    decay_lr: bool = False,
    lr_decay_iters: int = 0,
    decay_start_iter: int = 0,
    decay_end_iter: int = 0,
) -> float:
    """Learning rate at absolute iteration ``it``.

    ``it`` is the global iteration number, counted from the start of the
    *lineage*, not from the start of the process. A decay child inherits its
    parent's count from the branch checkpoint rather than resetting to zero —
    otherwise warmup re-triggers at the branch point.
    """
    # 1) linear warmup
    if it < warmup_iters:
        return peak * (it + 1) / (warmup_iters + 1)

    if schedule == "wsd":
        # 2) stable phase. An unset window (both bounds 0) is a pure stable
        # run — the trajectory that decay children fork off.
        if decay_end_iter <= decay_start_iter or it < decay_start_iter:
            return peak
        # 3) decay over the absolute window, clamped past the end so a run that
        # overshoots holds min_lr instead of walking back up the cosine.
        f = min(1.0, (it - decay_start_iter) / (decay_end_iter - decay_start_iter))
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * f)) * (peak - min_lr)

    # legacy
    if not decay_lr:
        return peak
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (peak - min_lr)


def validate(
    *,
    schedule: str,
    warmup_iters: int,
    decay_start_iter: int,
    decay_end_iter: int,
    peak: Optional[float] = None,
    min_lr: Optional[float] = None,
) -> None:
    """Reject half-specified schedules at startup rather than at the end.

    A WSD run missing one bound of its decay window trains as a pure stable
    run: no crash, no warning, just a model that was never annealed and whose
    loss is quietly off the annealed curve. That is worth a hard failure.
    """
    if schedule not in ("legacy", "wsd"):
        raise ValueError(f"lr.schedule must be 'legacy' or 'wsd', got {schedule!r}")
    if schedule != "wsd":
        return
    if peak is not None and min_lr is not None and min_lr > peak:
        # Every legacy config in this repo carries nanoGPT's default
        # min_lr=6e-5 against a peak of 2e-5. That was inert while
        # decay_lr=False, but under WSD it silently turns the decay into a 3x
        # ramp *up* — an anti-anneal that still produces a plausible loss curve.
        raise ValueError(
            f"lr.min_lr ({min_lr:g}) exceeds the peak learning rate ({peak:g}), "
            f"so the 'decay' would raise the LR. Set min_lr at or below the "
            f"peak (a tenth of it is the usual choice)."
        )
    if (decay_start_iter > 0) != (decay_end_iter > 0):
        raise ValueError(
            "lr.decay_start_iter and lr.decay_end_iter must both be set (a decay "
            f"child) or both be 0 (a pure stable run); got "
            f"start={decay_start_iter} end={decay_end_iter}"
        )
    if decay_end_iter and decay_end_iter <= decay_start_iter:
        raise ValueError(
            f"lr.decay_end_iter ({decay_end_iter}) must exceed decay_start_iter "
            f"({decay_start_iter})"
        )
    if decay_start_iter and decay_start_iter < warmup_iters:
        raise ValueError(
            f"lr.decay_start_iter ({decay_start_iter}) is inside warmup "
            f"({warmup_iters}); the stable phase would be empty"
        )
