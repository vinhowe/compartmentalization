#!/usr/bin/env python3
"""Fork a WSD decay ("anneal") off a stable trajectory.

    python3 scripts/launch_anneal.py <parent_run_dir> --at 27000 --decay 3000
    # branch at 27B tokens, decay over 3B, producing a 30B annealed model

The design goal is that this adds no new loading path to train.py. A decay child
is an ordinary run that happens to start from a populated `_rolling`:

  1. copy the parent's full-state checkpoint at the branch point into the
     child's `checkpoints/_rolling/` — train.py's existing auto-resume picks it
     up, complete with optimizer and dataloader state, at the right iter_num;
  2. write a child config with `lr.schedule = "wsd"` and the decay window as
     ABSOLUTE iteration numbers, so a preemption mid-decay resumes on the same
     LR instead of restarting the decay;
  3. run it in `<parent>@anneal-<end>M/`, whose name records both the parent and
     the branch point, and whose checkpoints land under `checkpoints/annealed/`.

The parent is never modified, so the stable run can keep training past the
branch point while the decay runs alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tomli_w

from src import checkpoints as ck

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _drop_nulls(obj):
    if isinstance(obj, dict):
        return {k: _drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_nulls(v) for v in obj if v is not None]
    return obj


def _tokens_per_iter(cfg: dict) -> int:
    t = cfg["training"]
    # ddp_world_size is not in the config; the recorded batch already reflects
    # the launch geometry, so read it back from the parent's own checkpoints
    # where possible and fall back to the single-process product.
    return int(t["gradient_accumulation_steps"]) * int(t["batch_size"]) * int(cfg["model"]["block_size"])


def find_branch_checkpoint(parent: str, at_tokens: int) -> ck.Checkpoint:
    """The full-state checkpoint to fork from.

    Requires optimizer state: forking from weights alone would restart Adam at
    the branch point, which is precisely the thing this layout exists to
    prevent. If the point you want is weights-only, it was not in
    `full_state_at_tokens` and cannot be recovered after the fact.
    """
    cks = [c for c in ck.iter_checkpoints(parent) if c.phase == "stable"]
    if not cks:
        raise SystemExit(f"no checkpoints found in {parent}")
    resumable = [c for c in cks if c.resumable and c.name != ck.ROLLING]
    if not resumable:
        raise SystemExit(
            f"{parent} has no full-state checkpoint to fork from.\n"
            f"  Add the branch point to training.full_state_at_tokens and re-run,\n"
            f"  or fork from the frontier using the run's _rolling checkpoint."
        )

    def distance(c):
        return abs((c.tokens or 0) - at_tokens)

    best = min(resumable, key=distance)
    if best.tokens is None:
        raise SystemExit(f"{best.name} predates token naming; cannot place it on a token axis")
    drift = abs(best.tokens - at_tokens) / 1e9
    if drift > 1.0:
        print(
            f"note: nearest full-state checkpoint is {best.name} "
            f"({best.tokens/1e9:.1f}B), {drift:.1f}B from the requested "
            f"{at_tokens/1e9:.1f}B"
        )
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent", help="stable run directory to fork from")
    ap.add_argument("--at", type=float, required=True,
                    help="branch point in millions of tokens (e.g. 27000 = 27B)")
    ap.add_argument("--decay", type=float, required=True,
                    help="decay length in millions of tokens (e.g. 3000 = 3B, ~10%%)")
    ap.add_argument("--min-lr", type=float, default=None,
                    help="floor of the decay (default: parent's lr.min_lr)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the child run and print its command, but do not run it "
                         "(use this to hand the job to slurm instead of holding it here)")
    args = ap.parse_args()

    parent = os.path.abspath(args.parent.rstrip("/"))
    at_tokens = int(args.at * 1e6)
    decay_tokens = int(args.decay * 1e6)

    cfg_path = os.path.join(parent, "meta", "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"missing {cfg_path} — is {parent} a run directory?")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("seed", None)                     # recorded output, not an input field

    branch = find_branch_checkpoint(parent, at_tokens)
    tpi = branch.tokens // branch.iter_num    # exact, straight from the parent's own history
    decay_iters = max(1, round(decay_tokens / tpi))
    end_iter = branch.iter_num + decay_iters
    end_tokens = end_iter * tpi

    child = f"{parent}@anneal-{ck.tok_dirname(end_tokens)[4:]}"   # strip the "tok-" prefix
    if os.path.exists(child):
        raise SystemExit(f"{child} already exists — remove it or pick another decay length")

    cfg.setdefault("lr", {})
    cfg["lr"]["schedule"] = "wsd"
    cfg["lr"]["decay_start_iter"] = branch.iter_num
    cfg["lr"]["decay_end_iter"] = end_iter
    peak = cfg["optimizer"]["learning_rate"]
    if args.min_lr is not None:
        cfg["lr"]["min_lr"] = args.min_lr
    elif cfg["lr"].get("min_lr", 0) >= peak:
        # Legacy configs carry nanoGPT's default min_lr=6e-5 against a peak of
        # 2e-5. It was inert while decay_lr=False, but inheriting it here would
        # make the "decay" a 3x ramp up. A tenth of peak is the usual floor.
        inherited = cfg["lr"].get("min_lr")
        cfg["lr"]["min_lr"] = peak / 10
        print(f"note: parent's min_lr ({inherited:g}) is >= peak ({peak:g}); "
              f"using {peak/10:g} instead so the decay decays")
    cfg["training"]["max_iters"] = end_iter
    # The child's only named checkpoint is its terminal one; full_state_steps
    # always includes max_iters, so it is resumable without listing anything.
    cfg["training"]["checkpoint_naming"] = "tokens"

    print(f"parent   {parent}")
    print(f"branch   {branch.name}  iter {branch.iter_num}  ({branch.tokens/1e9:.2f}B tokens)")
    print(f"decay    {decay_iters} iters -> iter {end_iter} ({end_tokens/1e9:.2f}B tokens)")
    print(f"child    {child}")

    if args.dry_run:
        print("\n[dry run] nothing written")
        return 0

    # 1) seed the child's rolling checkpoint from the branch point
    rolling = os.path.join(child, ck.CHECKPOINTS, ck.ROLLING)
    os.makedirs(rolling, exist_ok=True)
    for fname in ("model.pt", "optimizer.pt", "dataloader.pt"):
        src = os.path.join(branch.path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(rolling, fname))
    # trainer_state.json last, mirroring the writer: until it lands the
    # directory is not a checkpoint, so an interrupted setup cannot half-resume.
    with open(os.path.join(branch.path, "trainer_state.json")) as f:
        state = json.load(f)
    state["phase"] = "annealed"
    state["forked_from"] = branch.path
    with open(os.path.join(rolling, "trainer_state.json"), "w") as f:
        json.dump(state, f)

    # 2) child config
    os.makedirs(os.path.join(child, "meta"), exist_ok=True)
    child_toml = os.path.join(child, "meta", "config.toml")
    with open(child_toml, "wb") as f:
        # TOML has no null. A dropped key falls back to the dataclass default,
        # which is what the parent's None meant in the first place.
        tomli_w.dump(_drop_nulls(cfg), f)

    # 3) launch
    cmd = [sys.executable, os.path.join(REPO, "train.py"),
           f"--job.config-file={child_toml}"]
    env = dict(os.environ, OUT_DIR=child, RUN_ID=os.path.basename(child)[:32])
    print(f"\nOUT_DIR={child} \\\n  " + " ".join(cmd))
    if args.prepare_only:
        print("\n[prepare-only] child is ready; launch the command above when you want it")
        return 0
    return subprocess.call(cmd, cwd=REPO, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
