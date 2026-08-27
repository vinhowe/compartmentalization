#!/usr/bin/env python3
"""One-screen status for the ladder-v2 LR sweep.

Reads the per-run logs the launcher writes and reports, for each run: how far
it has got, its most recent loss, and whether it has died. Nothing here talks
to wandb, so it works even when the network does not.

Usage:  .venv/bin/python scripts/status_ladder_v2_lrsweep.py
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
LOG_ROOT = ROOT / "logs" / "ladder-v2-lrsweep"
OUT_ROOT = ROOT / "out" / "ladder-v2-lrsweep"
MAX_ITERS = 1431

ITER_RE = re.compile(r"^iter (\d+): loss ([\d.]+|nan)", re.M)
EVAL_RE = re.compile(r"^step (\d+): train loss ([\d.]+|nan), val loss ([\d.]+|nan)", re.M)
DEAD_RE = re.compile(r"Traceback|OutOfMemoryError|CUDA out of memory|Killed|assert")


def main() -> int:
    if not LOG_ROOT.exists():
        print(f"no logs at {LOG_ROOT}")
        return 1

    rows = []
    for p in sorted(LOG_ROOT.glob("*.log")):
        if p.name.startswith("_"):
            continue
        text = p.read_text(errors="replace")
        iters = ITER_RE.findall(text)
        evals = EVAL_RE.findall(text)
        last_iter = int(iters[-1][0]) if iters else 0
        last_loss = iters[-1][1] if iters else "-"
        last_val = evals[-1][2] if evals else "-"
        # A log is only evidence about the CURRENT attempt if that attempt got
        # as far as creating the run's output directory -- the launcher mkdirs
        # it before exec'ing train.py. After a relaunch the not-yet-started runs
        # still hold their previous log, and reporting those as DIED buries the
        # real failures among stale ones.
        started = (OUT_ROOT / p.stem).exists()
        dead = bool(DEAD_RE.search(text)) and started
        diverged = last_loss == "nan" or last_val == "nan"
        pct = 100.0 * last_iter / MAX_ITERS
        state = ("DIED" if dead else
                 "pending" if not started else
                 "diverged" if diverged else
                 "done" if last_iter >= MAX_ITERS - 2 else
                 "running")
        rows.append((p.stem.replace("ladder-v2-lrsweep-", ""),
                     f"{last_iter}/{MAX_ITERS}", f"{pct:5.1f}%",
                     last_loss, last_val, state))

    if not rows:
        print("no run logs yet")
        return 0

    hdr = ("run", "iters", "pct", "train", "val", "state")
    w = [max(len(str(r[i])) for r in ([hdr] + rows)) for i in range(len(hdr))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for r in sorted(rows, key=lambda r: r[0]):
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))

    pend = sum(1 for r in rows if r[5] == "pending")
    done = sum(1 for r in rows if r[5] == "done")
    dead = sum(1 for r in rows if r[5] == "DIED")
    div = sum(1 for r in rows if r[5] == "diverged")
    print(f"\n{len(rows)} configs | {len(rows)-pend} started | {done} done | "
          f"{div} diverged | {dead} died | {pend} pending")
    if div:
        print("note: divergence at the top of the LR grid is an informative "
              "bracket, not a failure -- it is what makes the argmin interior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
