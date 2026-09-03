#!/usr/bin/env python3
"""
Prune checkpoint step directories.

Given the absolute path to an out/ directory, recursively find all
checkpoints/step-<N> directories and delete those that are NOT:
  - in the special checkpoint steps set, or
  - multiples of --keep-every (default: 10000)

Example:
  python scripts/prune_checkpoints.py /nobackup/archive/grp/grp_pccl/vin/dev/translation-compression/out/
  python scripts/prune_checkpoints.py /nobackup/.../out/ --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from tqdm.auto import tqdm

# Keep in sync with train.py checkpoint_steps (train.py: lines 515-527)
SPECIAL_CHECKPOINT_STEPS = {
    100,
    850,
    3500,
    7000,
    14000,
    29000,
    60000,
    120000,
    240000,
    500000,
    1000000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete non-special, non-Nth (e.g., 10k) checkpoint step directories under out/."
    )
    parser.add_argument(
        "out_root",
        type=Path,
        help="Absolute path to the out/ directory (e.g., /abs/path/to/out/).",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=10000,
        help="Keep every Nth step (default: 10000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete anything; only print what would be deleted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )
    return parser.parse_args()


STEP_DIR_PATTERN = re.compile(r"^step-(\d+)$")


def parse_step_from_dirname(dirname: str) -> Optional[int]:
    match = STEP_DIR_PATTERN.fullmatch(dirname)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def iter_checkpoint_step_dirs(
    out_root: Path, quiet: bool = False
) -> Iterable[Tuple[int, Path]]:
    """
    Yield (step_number, path) for each checkpoints/step-<N> directory.
    """
    # Find all 'checkpoints' directories under out_root
    for checkpoints_dir in tqdm(
        out_root.rglob("checkpoints"), desc="Finding checkpoints", disable=quiet
    ):
        if not checkpoints_dir.is_dir():
            continue
        # Iterate immediate children looking for 'step-<N>' directories
        for child in checkpoints_dir.iterdir():
            if not child.is_dir():
                continue
            step = parse_step_from_dirname(child.name)
            if step is None:
                continue
            yield step, child


def should_keep_step(step: int, keep_every: int) -> bool:
    if step in SPECIAL_CHECKPOINT_STEPS:
        return True
    return keep_every > 0 and (step % keep_every == 0)


def delete_dir(path: Path) -> None:
    # Use shutil.rmtree to delete directories (handles nested content).
    shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    out_root: Path = args.out_root

    if not out_root.is_absolute():
        print(f"ERROR: out_root must be an absolute path: {out_root}", file=sys.stderr)
        return 2
    if not out_root.exists():
        print(f"ERROR: Path does not exist: {out_root}", file=sys.stderr)
        return 2
    if not out_root.is_dir():
        print(f"ERROR: Not a directory: {out_root}", file=sys.stderr)
        return 2

    # Collect all step dirs
    step_dirs: List[Tuple[int, Path]] = list(iter_checkpoint_step_dirs(out_root))
    # Sort for stable output: by parent checkpoints path then by numeric step
    step_dirs.sort(key=lambda t: (str(t[1].parent), t[0]))

    total = len(step_dirs)
    to_delete: List[Tuple[int, Path]] = []
    kept: List[Tuple[int, Path]] = []

    for step, path in tqdm(step_dirs, desc="Checking checkpoints", disable=args.quiet):
        if should_keep_step(step, args.keep_every):
            kept.append((step, path))
        else:
            to_delete.append((step, path))

    if not args.quiet:
        print(f"Scanned checkpoints under: {out_root}")
        print(f"Total step directories found: {total}")
        print(f"Keeping: {len(kept)}")
        print(f"Deleting: {len(to_delete)} {'(dry-run)' if args.dry_run else ''}")

    for step, path in to_delete:
        if args.dry_run:
            if not args.quiet:
                print(f"[DRY RUN] Would delete: {path} (step={step})")
            continue
        try:
            delete_dir(path)
            if not args.quiet:
                print(f"Deleted: {path} (step={step})")
        except Exception as exc:
            print(f"ERROR deleting {path}: {exc}", file=sys.stderr)

    if not args.quiet:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
