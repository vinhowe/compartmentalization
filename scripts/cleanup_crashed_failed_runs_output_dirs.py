#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

from wandb.errors import CommError
from wandb.apis.public.api import Api

DEFAULT_STORAGE_ROOT = "/nobackup/archive/grp/grp_pccl/vin/dev/translation-compression"
DEFAULT_ENTITY = "pccl"
DEFAULT_PROJECT = "translation-compression"

KNOWN_FAILED_STATES = {
    "aborted",
    "cancelled",
    "canceled",
    "crashed",
    "errored",
    "failed",
    "killed",
    "preempted",
    "stopped",
    "timeout",
}
KNOWN_RUNNING_STATES = {"pending", "prequeue", "queued", "running", "starting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect local output directories, discover associated W&B runs, and "
            "print absolute paths for runs not in a healthy state."
        )
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(DEFAULT_STORAGE_ROOT, "out"),
        help="Root directory that contains project/group/run subdirectories.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="W&B entity (team) the runs belong to.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project containing the runs.",
    )
    return parser.parse_args()


def iter_run_dirs(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, _ in os.walk(root):
        if "wandb" in dirnames:
            yield Path(dirpath)
            dirnames[:] = [d for d in dirnames if d != "wandb"]


def _extract_run_id(name: str) -> str | None:
    prefixes = ("run-", "offline-run-", "resume-")
    for prefix in prefixes:
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix) :]
        if "-" in tail:
            tail = tail.rsplit("-", 1)[-1]
        tail = tail.split(".", 1)[0]
        return tail or None
    return None


def discover_run_ids(wandb_dir: Path) -> list[str]:
    run_ids: set[str] = set()
    for entry in wandb_dir.iterdir():
        candidate = _extract_run_id(entry.name)
        if candidate:
            run_ids.add(candidate)
        if entry.is_dir() and entry.name == "latest-run":
            for child in entry.iterdir():
                candidate = _extract_run_id(child.name)
                if candidate:
                    run_ids.add(candidate)
                if child.is_dir():
                    for grandchild in child.iterdir():
                        candidate = _extract_run_id(grandchild.name)
                        if candidate:
                            run_ids.add(candidate)
    return sorted(run_ids)


def normalize_state(state: str | None) -> str:
    if not state:
        return "running"
    normalized = state.lower()
    if normalized in {"running", "finished", "crashed", "failed"}:
        return normalized
    if normalized in KNOWN_RUNNING_STATES:
        return "running"
    if normalized in KNOWN_FAILED_STATES:
        return "failed" if normalized != "crashed" else "crashed"
    return "failed"


def fetch_run_status(
    api: Api,
    entity: str,
    project: str,
    run_id: str,
    cache: dict[str, str | None],
) -> str | None:
    if run_id in cache:
        return cache[run_id]

    run_path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(run_path)
    except CommError as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            cache[run_id] = None
            return None
        cache[run_id] = None
        return None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"# Unexpected error fetching run {run_path}: {exc}", file=sys.stderr)
        cache[run_id] = None
        return None

    status = normalize_state(getattr(run, "state", None))
    cache[run_id] = status
    return status


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).expanduser()
    if not out_root.exists():
        print(f"Output root {out_root} does not exist.", file=sys.stderr)
        return 1

    api = Api()
    run_cache: dict[str, str | None] = {}

    for run_dir in sorted(iter_run_dirs(out_root)):
        wandb_dir = run_dir / "wandb"
        if not wandb_dir.exists():
            continue
        run_ids = discover_run_ids(wandb_dir) if wandb_dir.exists() else []

        status = "deleted"

        for run_id in run_ids:
            remote_status = fetch_run_status(
                api, args.entity, args.project, run_id, run_cache
            )
            if remote_status is None:
                continue
            status = remote_status
            break

        if status not in {"running", "finished"}:
            print(str(run_dir.resolve()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
