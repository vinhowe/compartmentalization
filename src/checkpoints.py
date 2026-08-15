"""Checkpoint naming, discovery, and the resume rule.

Two facts about a checkpoint, carried differently on purpose.

**Resumable** — it carries `optimizer.pt`, or it does not. That is the whole
rule. It replaces the old behaviour where `checkpoints/step-*` was weights-only
*and* accepted for resume, silently restarting Adam at a cost of 0.005-0.012
nats on the n-gram rungs.

**Annealed vs stable** — this one gets structure. A decay child's checkpoints
live under `checkpoints/annealed/`, because annealed points sit below the stable
loss curve and a bare `checkpoints/tok-*` glob must not be able to reach them.

Names are token counts, not step counts: a step means different things at
different batch sizes, so `step-500000` is not comparable across the redesign
and `tok-000030000M` is.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional

__all__ = [
    "CHECKPOINTS", "ROLLING", "ANNEALED", "Checkpoint",
    "tok_dirname", "parse_dirname", "checkpoint_iter", "iter_checkpoints",
    "find_resume_checkpoint", "full_state_steps",
]

CHECKPOINTS = "checkpoints"
ROLLING = "_rolling"
ANNEALED = "annealed"

_TOK_RE = re.compile(r"^tok-(\d+)M$")
_STEP_RE = re.compile(r"^step-(\d+)$")

# Escape hatch for the 400+ pre-existing runs whose only checkpoints are
# weights-only `step-*`. Off by default: see find_resume_checkpoint.
ALLOW_WEIGHTS_ONLY_RESUME = "TC_ALLOW_WEIGHTS_ONLY_RESUME"


def tok_dirname(tokens: int) -> str:
    """`27_000_000_000` -> `tok-000027000M`.

    Millions, zero-padded to 9 digits so lexical order is numeric order — every
    consumer globs and sorts these, and a sort that disagrees with training
    order produces a plausible-looking wrong figure.
    """
    return f"tok-{tokens // 1_000_000:09d}M"


def parse_dirname(name: str) -> Optional[tuple[str, int]]:
    """`("tok", tokens)` or `("step", iter_num)`, or None if not a checkpoint."""
    for regex, kind, scale in ((_TOK_RE, "tok", 1_000_000), (_STEP_RE, "step", 1)):
        m = regex.match(name)
        if m:
            return kind, int(m.group(1)) * scale
    return None


@dataclass(frozen=True)
class Checkpoint:
    path: str
    iter_num: int
    tokens: Optional[int]      # None for legacy step-* dirs, which predate token naming
    phase: str                 # "stable" | "annealed"
    resumable: bool            # carries optimizer.pt

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _read_state(path: str) -> Optional[dict]:
    """trainer_state.json, or None if absent/corrupt.

    Written last, so its presence is what marks a checkpoint complete — a
    half-written directory is never mistaken for a resume point.
    """
    try:
        with open(os.path.join(path, "trainer_state.json")) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def checkpoint_iter(path: str) -> Optional[int]:
    """iter_num from trainer_state.json, or None if absent/corrupt."""
    state = _read_state(path)
    try:
        return int(state["iter_num"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


def _load(path: str, phase: str) -> Optional[Checkpoint]:
    state = _read_state(path)
    if state is None or "iter_num" not in state:
        return None                      # incomplete or corrupt — not a checkpoint
    if not os.path.exists(os.path.join(path, "model.pt")):
        return None
    parsed = parse_dirname(os.path.basename(path))
    # "annealed" wins if EITHER the path or the record says so. The path alone
    # calls a decay child's _rolling stable (it sits at the top of its own run
    # directory); the record alone can be stale if a directory was copied. The
    # errors are not symmetric — an annealed point mislabelled stable joins a
    # loss curve it does not belong on, the reverse merely drops a point.
    return Checkpoint(
        path=path,
        iter_num=int(state["iter_num"]),
        tokens=parsed[1] if parsed and parsed[0] == "tok" else state.get("tokens"),
        phase="annealed" if ANNEALED in (phase, state.get("phase")) else "stable",
        resumable=os.path.exists(os.path.join(path, "optimizer.pt")),
    )


def iter_checkpoints(out_dir: str, include_rolling: bool = True) -> Iterator[Checkpoint]:
    """Every complete checkpoint in a run, stable and annealed, unordered.

    Tolerates the token layout and the legacy `step-*` layout, so existing runs
    keep working without migration.
    """
    root = os.path.join(out_dir, CHECKPOINTS)
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or os.path.islink(path):
            continue           # skip the `latest` symlink
        if name == ANNEALED:
            candidates = [(os.path.join(path, s), ANNEALED) for s in sorted(os.listdir(path))]
        elif name == ROLLING:
            candidates = [(path, "stable")] if include_rolling else []
        elif parse_dirname(name):
            candidates = [(path, "stable")]
        else:
            continue
        for cand_path, cand_phase in candidates:
            ck = _load(cand_path, cand_phase)
            if ck:
                yield ck


def find_resume_checkpoint(out_dir: str) -> Optional[Checkpoint]:
    """The newest resumable checkpoint, None for a fresh run, or raise.

    Raises when a run has checkpoints on disk but none carries optimizer state:
    starting over would overwrite real training, which is strictly worse than
    refusing to start. `_rolling` is held to the same rule as everything else —
    it normally carries optimizer.pt, and one that does not is a crashed write,
    not a resume point.
    """
    cks = list(iter_checkpoints(out_dir))
    resumable = [c for c in cks if c.resumable]
    if resumable:
        return max(resumable, key=lambda c: c.iter_num)
    if not cks:
        return None                                  # genuinely a fresh run

    newest = max(cks, key=lambda c: c.iter_num)
    if os.environ.get(ALLOW_WEIGHTS_ONLY_RESUME) == "1":
        # Resume the weights anyway. Adam restarts from zero, but keeping 400k
        # steps of training beats discarding them -- this is what the 400+
        # legacy step-* runs need.
        print(
            f"[checkpoints] WARNING: resuming from weights-only {newest.name} "
            f"({ALLOW_WEIGHTS_ONLY_RESUME}=1). Optimizer state is lost and Adam "
            f"restarts from zero — expect a transient loss bump."
        )
        return newest
    raise RuntimeError(
        f"{out_dir} has checkpoints up to iter {newest.iter_num} ({newest.name}) "
        f"but none carries optimizer.pt, so training cannot resume without "
        f"resetting Adam. Refusing to silently start over and overwrite this run.\n"
        f"  - to resume anyway, resetting Adam: {ALLOW_WEIGHTS_ONLY_RESUME}=1\n"
        f"  - to genuinely start over: move "
        f"{os.path.join(out_dir, CHECKPOINTS)} aside first"
    )


def full_state_steps(
    budgets_tokens, checkpoint_steps, tokens_per_iter: int, max_iters: int
) -> set[int]:
    """Which scheduled steps also save optimizer state.

    A budget rarely lands exactly on a checkpoint, so each claims the first
    scheduled checkpoint at or past it. `max_iters` is always included: the end
    of a stable run is what you would extend or fork from, and an unforkable
    final checkpoint is the one omission you cannot repair afterwards.
    """
    steps = {max_iters}
    scheduled = sorted(s for s in checkpoint_steps if 0 < s <= max_iters)
    for b in budgets_tokens or ():
        for s in scheduled:
            if s * tokens_per_iter >= b:
                steps.add(s)
                break
    return steps
