#!/usr/bin/env python3
"""Launch the ladder-v2 LR sweep across the local GPUs.

One run per GPU, longest-first, greedy-packed. Single-GPU rather than DDP on
purpose: the runs are independent, so N independent 1-GPU runs beat N/8
sequential 8-GPU runs by exactly the DDP communication overhead, and the batch
geometry is unaffected (train.py asserts grad_accum % world_size == 0, which
any world size dividing 2048 satisfies).

Outputs land under the SHARED checkout, not this worktree, so they survive the
worktree being removed.

Usage:
    .venv/bin/python scripts/launch_ladder_v2_lrsweep.py --dry-run
    .venv/bin/python scripts/launch_ladder_v2_lrsweep.py
    .venv/bin/python scripts/launch_ladder_v2_lrsweep.py --gpus 0,2,3
"""

from __future__ import annotations

import argparse
import os
import pathlib
import queue
import re
import subprocess
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
SHARED = pathlib.Path("/mnt/pccfs2/backed_up/vin/dev/translation-compression")
PYTHON = SHARED / ".venv" / "bin" / "python"
OUT_ROOT = SHARED / "out" / "ladder-v2-lrsweep"
LOG_ROOT = SHARED / "logs" / "ladder-v2-lrsweep"

# Hours per run on ONE A100, used only for longest-first ordering. Calibrated
# against a measured 18.8 s/iter for r1-c8, scaled by (6*N_total + attention).
# c1-padded costs what c8 costs -- same 131073-wide head, which is where the
# time goes -- not what c1 costs.
COST = {
    ("r1", "c1"): 1.7, ("r1", "c8"): 7.5, ("r1", "c1-padded"): 7.5,
    ("r3", "c1"): 3.8, ("r3", "c8"): 12.5, ("r3", "c1-padded"): 12.5,
    ("r4", "c1"): 7.5, ("r4", "c8"): 19.1, ("r4", "c1-padded"): 19.1,
}


def jobs(config_dir: pathlib.Path,
         include: list[str] | None = None) -> list[pathlib.Path]:
    js = sorted(config_dir.glob("ladder-v2-lrsweep-*.toml"))
    if include:
        # Substring filter, so a second scheduler on another host can pick up
        # only the arms the first one is not already running. train.py's
        # active-run lock would catch an overlap anyway, but a launcher that
        # relies on a lock to avoid double-scheduling is one preemption away
        # from the ActiveRunError storm that produced 43% rc=1 in the old pool.
        js = [p for p in js if any(tok in p.stem for tok in include)]

    def cost(p: pathlib.Path) -> float:
        # ladder-v2-lrsweep-<rung>-<arm>-lr<value>. Parsed by regex, NOT by
        # splitting on "-": the arm may itself contain a hyphen (c1-padded) and
        # so may the lr tag (lr1e-3), so a positional split silently yields
        # "c8-lr1e" and every run falls back to the default cost -- which
        # quietly destroys the longest-first ordering rather than erroring.
        m = re.match(r"^ladder-v2-lrsweep-(r\d+)-(.+)-lr[\de.+-]+$", p.stem)
        if not m:
            return 1.0
        return COST.get((m.group(1), m.group(2)), 1.0)

    return sorted(js, key=cost, reverse=True)   # longest-first


def run_one(cfg: pathlib.Path, gpu: int, dry: bool) -> tuple[str, int]:
    name = cfg.stem
    out_dir = OUT_ROOT / name
    log = LOG_ROOT / f"{name}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OUT_DIR": str(out_dir),
        "RUN_ID": name,
        # a stray core dump from a crashed worker filled /grphome once and mass
        # -killed a queue; never let one be written
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [str(PYTHON), "train.py", "--job.config-file", str(cfg)]
    if dry:
        print(f"  GPU{gpu}  {name}")
        return name, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        fh.write(f"# {' '.join(cmd)}\n# CUDA_VISIBLE_DEVICES={gpu}\n\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    return name, p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--config-dir", default=str(REPO / "config" / "ladder-v2"))
    ap.add_argument("--include", default=None,
                    help="comma-separated substrings; only matching runs are scheduled")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    include = ([s for s in args.include.split(',') if s.strip()]
               if args.include else None)
    js = jobs(pathlib.Path(args.config_dir), include)
    if not js:
        print("no sweep configs found")
        return 1

    # Preflight: resolve the training glob exactly as train.py will, from the
    # same cwd. train.py resolves data paths RELATIVE to the working directory,
    # so a checkout without a data/ link (e.g. a git worktree, where data/ is
    # gitignored and therefore absent) fails the assert only after model init,
    # compile and wandb -- once per run, on every GPU, minutes in. Fifteen runs
    # died that way. Fail here instead, before anything is scheduled.
    import tomllib
    with open(js[0], "rb") as fh:
        pattern = tomllib.load(fh)["data"]["train_bin"]
    matches = list(REPO.glob(pattern)) if not os.path.isabs(pattern) else []
    if not matches:
        print(f"PREFLIGHT FAILED: no files match {pattern!r} relative to {REPO}")
        print("  train.py resolves data paths against its cwd. In a worktree, link it:")
        print(f"  ln -sfn {SHARED}/data/<corpus> {REPO}/data/<corpus>")
        return 1
    print(f"preflight ok: {len(matches)} shards match {pattern}")

    print(f"{len(js)} runs over {len(gpus)} GPUs {gpus}, longest-first")
    if args.dry_run:
        for i, c in enumerate(js):
            print(f"  {i:2d}  {c.stem}")
        return 0

    pending: queue.Queue = queue.Queue()
    for c in js:
        pending.put(c)

    results: list[tuple[str, int]] = []
    lock = threading.Lock()
    t0 = time.time()

    def worker(gpu: int) -> None:
        while True:
            try:
                cfg = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                print(f"[{time.time()-t0:7.0f}s] GPU{gpu} START {cfg.stem}",
                      flush=True)
            name, rc = run_one(cfg, gpu, dry=False)
            with lock:
                results.append((name, rc))
                tag = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[{time.time()-t0:7.0f}s] GPU{gpu} DONE  {name} {tag}",
                      flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bad = [n for n, rc in results if rc != 0]
    print(f"\n{len(results)} finished in {(time.time()-t0)/3600:.1f}h, "
          f"{len(bad)} failed")
    for n in bad:
        print(f"  FAILED {n}  -> {LOG_ROOT / (n + '.log')}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
